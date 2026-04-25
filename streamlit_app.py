import math
import traceback

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from farm_area import (
    calculate_farm_area_from_df,
    geom_to_geojson_coords,
    read_uploaded_file,
)

st.set_page_config(page_title="Farm Area Calculator", layout="wide")


def haversine_m(lat1, lon1, lat2, lon2):
    r = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)

    a = (
        math.sin(dp / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    )
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def build_line_segments(df, max_gap_m=20.0, max_gap_sec=180.0, min_segment_points=2):
    if df is None or df.empty:
        return []

    work = df.copy()

    if "timestamp" in work.columns:
        work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")
    else:
        work["timestamp"] = pd.NaT

    if work["timestamp"].notna().any():
        work = work.sort_values("timestamp").reset_index(drop=True)
    else:
        work = work.reset_index(drop=True)

    work = work.dropna(subset=["latitude", "longitude"]).copy()
    if work.empty:
        return []

    segments = []
    current_segment = []
    prev = None

    for _, row in work.iterrows():
        point = [float(row["latitude"]), float(row["longitude"])]

        if prev is None:
            current_segment = [point]
            prev = row
            continue

        dist_m = haversine_m(
            prev["latitude"], prev["longitude"],
            row["latitude"], row["longitude"]
        )

        too_far = dist_m > max_gap_m

        too_old = False
        if pd.notna(prev["timestamp"]) and pd.notna(row["timestamp"]):
            gap_sec = (row["timestamp"] - prev["timestamp"]).total_seconds()
            too_old = gap_sec > max_gap_sec

        if too_far or too_old:
            if len(current_segment) >= min_segment_points:
                segments.append(current_segment)
            current_segment = [point]
        else:
            current_segment.append(point)

        prev = row

    if len(current_segment) >= min_segment_points:
        segments.append(current_segment)

    return segments


def classify_on_road_points(df, wheel_th=20.0, rotor_th=50.0):
    if df is None or df.empty:
        return df.iloc[0:0].copy()

    needed = {"left_rpm", "right_rpm", "rotor_rpm"}
    if not needed.issubset(df.columns):
        return df.iloc[0:0].copy()

    out = df.copy()

    out["left_rpm"] = pd.to_numeric(out["left_rpm"], errors="coerce").fillna(0)
    out["right_rpm"] = pd.to_numeric(out["right_rpm"], errors="coerce").fillna(0)
    out["rotor_rpm"] = pd.to_numeric(out["rotor_rpm"], errors="coerce").fillna(0)

    mask = (
        (out["rotor_rpm"] <= rotor_th)
        & (
            (out["left_rpm"] > wheel_th)
            | (out["right_rpm"] > wheel_th)
        )
    )

    return out.loc[mask].copy()


def remove_rows_by_row_id(df, row_ids_to_remove):
    if df is None or df.empty:
        return df.iloc[0:0].copy()

    if "row_id" not in df.columns:
        return df.copy()

    return df.loc[~df["row_id"].isin(row_ids_to_remove)].copy()


def render_map(result):
    center_source = result["clean_df"] if not result["clean_df"].empty else result["all_df"]

    center_lat = center_source["latitude"].mean()
    center_lon = center_source["longitude"].mean()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=22,
        max_zoom=24,
        control_scale=True,
        tiles=None,
    )

    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google Satellite",
        name="Satellite",
        overlay=False,
        control=True,
        max_zoom=24,
        max_native_zoom=21,
    ).add_to(m)

    all_df = result["all_df"].copy()
    clean_df = result["clean_df"].copy()
    removed_df = result["removed_df"].copy()

    on_road_df = classify_on_road_points(all_df, wheel_th=20.0, rotor_th=50.0)

    on_road_row_ids = set(on_road_df["row_id"].tolist()) if "row_id" in on_road_df.columns else set()
    removed_display_df = remove_rows_by_row_id(removed_df, on_road_row_ids)

    if not clean_df.empty:
        fg_farm = folium.FeatureGroup(name="On_Farm", show=True)
        farm_segments = build_line_segments(
            clean_df,
            max_gap_m=20.0,
            max_gap_sec=180.0,
            min_segment_points=2,
        )
        for seg in farm_segments:
            folium.PolyLine(
                locations=seg,
                color="lime",
                weight=3,
                opacity=0.95,
            ).add_to(fg_farm)
        fg_farm.add_to(m)

    if not on_road_df.empty:
        fg_road = folium.FeatureGroup(name="On_Road", show=True)
        road_segments = build_line_segments(
            on_road_df,
            max_gap_m=25.0,
            max_gap_sec=180.0,
            min_segment_points=2,
        )
        for seg in road_segments:
            folium.PolyLine(
                locations=seg,
                color="yellow",
                weight=3,
                opacity=0.95,
            ).add_to(fg_road)
        fg_road.add_to(m)

    if not removed_display_df.empty:
        fg_removed = folium.FeatureGroup(name="Removed_Points", show=True)
        for _, r in removed_display_df.iterrows():
            if pd.notna(r.get("latitude")) and pd.notna(r.get("longitude")):
                popup_text = str(r.get("remove_reason", "removed"))
                folium.CircleMarker(
                    location=[float(r["latitude"]), float(r["longitude"])],
                    radius=2,
                    color="red",
                    fill=True,
                    fill_color="red",
                    fill_opacity=0.85,
                    popup=popup_text,
                ).add_to(fg_removed)
        fg_removed.add_to(m)

    concave_gj = geom_to_geojson_coords(result["concave_geom"], result["to_wgs"])
    if concave_gj:
        folium.GeoJson(
            concave_gj,
            name="Farm_Boundary",
            style_function=lambda x: {
                "color": "yellow",
                "weight": 2,
                "fillColor": "yellow",
                "fillOpacity": 0.20,
            },
        ).add_to(m)

    path_gj = geom_to_geojson_coords(result["path_geom"], result["to_wgs"])
    if path_gj:
        folium.GeoJson(
            path_gj,
            name="Worked_Strip",
            style_function=lambda x: {
                "color": "cyan",
                "weight": 2,
                "fillColor": "cyan",
                "fillOpacity": 0.18,
            },
        ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    bounds_df = result["all_df"] if not result["all_df"].empty else center_source
    if not bounds_df.empty:
        min_lat = bounds_df["latitude"].min()
        max_lat = bounds_df["latitude"].max()
        min_lon = bounds_df["longitude"].min()
        max_lon = bounds_df["longitude"].max()

        pad_lat = max((max_lat - min_lat) * 0.08, 0.00005)
        pad_lon = max((max_lon - min_lon) * 0.08, 0.00005)

        m.fit_bounds([
            [min_lat - pad_lat, min_lon - pad_lon],
            [max_lat + pad_lat, max_lon + pad_lon],
        ])

    return m


try:
    st.title("Farm Area Calculator")
    st.caption("Upload telemetry lat/lon + RPM data and calculate farm covered area in guntha.")

    with st.sidebar:
        st.header("Inputs")

        work_width_ft = st.number_input("Working width (ft)", min_value=1.0, value=4.0, step=0.1)
        max_point_jump_m = st.number_input("Max point jump (m)", min_value=1.0, value=25.0, step=1.0)
        dbscan_eps_m = st.number_input("DBSCAN eps (m)", min_value=1.0, value=15.0, step=1.0)
        dbscan_min_samples = st.number_input("DBSCAN min samples", min_value=2, value=15, step=1)
        lof_neighbors = st.number_input("LOF neighbors", min_value=2, value=20, step=1)
        lof_contamination = st.slider("LOF contamination", min_value=0.0, max_value=0.20, value=0.01, step=0.01)

        point_buffer_m = st.number_input("Boundary point buffer (m)", min_value=0.5, value=2.2, step=0.1)
        boundary_smooth_m = st.number_input("Boundary smooth (m)", min_value=0.0, value=3.0, step=0.1)
        boundary_erode_m = st.number_input("Boundary erode (m)", min_value=0.0, value=2.0, step=0.1)

        show_map = st.checkbox("Show map", value=True)

    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file",
        type=["csv", "xlsx", "xls"]
    )

    if uploaded_file is None:
        st.info("Upload a CSV/XLSX file with: lat, long, left rpm, right rpm, rotor rpm. Timestamp is optional.")
        st.stop()

    input_df = read_uploaded_file(uploaded_file)
    st.success(f"Loaded {len(input_df)} points")

    result = calculate_farm_area_from_df(
        input_df=input_df,
        work_width_ft=work_width_ft,
        max_point_jump_m=max_point_jump_m,
        dbscan_eps_m=dbscan_eps_m,
        dbscan_min_samples=int(dbscan_min_samples),
        lof_neighbors=int(lof_neighbors),
        lof_contamination=lof_contamination,
        point_buffer_m=point_buffer_m,
        boundary_smooth_m=boundary_smooth_m,
        boundary_erode_m=boundary_erode_m,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Covered area (guntha)", f"{result['concave_area_guntha']:.4f}")
    c2.metric("Covered area (m²)", f"{result['concave_area_m2']:.2f}")
    c3.metric("4 ft strip check (guntha)", f"{result['path_area_guntha']:.4f}")

    on_road_preview = classify_on_road_points(result["all_df"], wheel_th=20.0, rotor_th=50.0)
    on_road_row_ids = set(on_road_preview["row_id"].tolist()) if "row_id" in on_road_preview.columns else set()
    removed_preview = remove_rows_by_row_id(result["removed_df"], on_road_row_ids)

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Total points", len(result["all_df"]))
    d2.metric("On_Farm points", len(result["clean_df"]))
    d3.metric("On_Road points", len(on_road_preview))
    d4.metric("Removed points", len(removed_preview))

    if show_map:
        st.subheader("Filtered GPS map")
        st.caption("Use the layer control on the map to toggle On_Farm, On_Road, Removed_Points, Farm_Boundary, and Worked_Strip.")
        try:
            m = render_map(result)
            st_folium(m, width=1400, height=700, returned_objects=[])
        except Exception as map_error:
            st.warning(f"Map rendering failed: {map_error}")
            st.code(traceback.format_exc())

    st.subheader("Preview")
    p1, p2, p3 = st.columns(3)

    with p1:
        st.write("On_Farm points")
        if result["clean_df"].empty:
            st.info("No On_Farm points")
        else:
            cols = [c for c in ["latitude", "longitude", "left_rpm", "right_rpm", "rotor_rpm", "timestamp"] if c in result["clean_df"].columns]
            st.dataframe(result["clean_df"][cols], use_container_width=True)

    with p2:
        st.write("On_Road points")
        if on_road_preview.empty:
            st.info("No On_Road points")
        else:
            cols = [c for c in ["latitude", "longitude", "left_rpm", "right_rpm", "rotor_rpm", "timestamp"] if c in on_road_preview.columns]
            st.dataframe(on_road_preview[cols], use_container_width=True)

    with p3:
        st.write("Removed points")
        if removed_preview.empty:
            st.info("No Removed points")
        else:
            cols = [c for c in ["latitude", "longitude", "left_rpm", "right_rpm", "rotor_rpm", "timestamp", "remove_reason"] if c in removed_preview.columns]
            st.dataframe(removed_preview[cols], use_container_width=True)

    clean_csv = result["clean_df"].to_csv(index=False).encode("utf-8")
    on_road_csv = on_road_preview.to_csv(index=False).encode("utf-8")
    removed_csv = removed_preview.to_csv(index=False).encode("utf-8")

    b1, b2, b3 = st.columns(3)
    with b1:
        st.download_button(
            "Download On_Farm CSV",
            data=clean_csv,
            file_name="on_farm_points.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with b2:
        st.download_button(
            "Download On_Road CSV",
            data=on_road_csv,
            file_name="on_road_points.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with b3:
        st.download_button(
            "Download Removed CSV",
            data=removed_csv,
            file_name="removed_points.csv",
            mime="text/csv",
            use_container_width=True,
        )

except Exception as e:
    st.error(f"Error: {e}")
    st.code(traceback.format_exc())
