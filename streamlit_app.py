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


def build_line_segments(df, max_gap_m=20.0, max_gap_sec=180.0):
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

    segments = []
    current_segment = []
    prev = None

    for _, row in work.iterrows():
        point = [row["latitude"], row["longitude"]]

        if prev is None:
            current_segment = [point]
            prev = row
            continue

        too_far = False
        if all(col in work.columns for col in ["x", "y"]):
            if pd.notna(prev["x"]) and pd.notna(prev["y"]) and pd.notna(row["x"]) and pd.notna(row["y"]):
                dist_m = ((row["x"] - prev["x"]) ** 2 + (row["y"] - prev["y"]) ** 2) ** 0.5
                too_far = dist_m > max_gap_m

        too_old = False
        if pd.notna(prev["timestamp"]) and pd.notna(row["timestamp"]):
            gap_sec = (row["timestamp"] - prev["timestamp"]).total_seconds()
            too_old = gap_sec > max_gap_sec

        cluster_changed = False
        if "cluster_id" in work.columns:
            cluster_changed = row["cluster_id"] != prev["cluster_id"]

        reason_changed = False
        if "remove_reason" in work.columns:
            reason_changed = row["remove_reason"] != prev["remove_reason"]

        if too_far or too_old or cluster_changed or reason_changed:
            if len(current_segment) >= 2:
                segments.append(current_segment)
            current_segment = [point]
        else:
            current_segment.append(point)

        prev = row

    if len(current_segment) >= 2:
        segments.append(current_segment)

    return segments


def render_map(result):
    center_lat = (
        result["clean_df"]["latitude"].mean()
        if not result["clean_df"].empty
        else result["all_df"]["latitude"].mean()
    )
    center_lon = (
        result["clean_df"]["longitude"].mean()
        if not result["clean_df"].empty
        else result["all_df"]["longitude"].mean()
    )

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

    if not result["removed_df"].empty:
        fg_removed = folium.FeatureGroup(name="Removed lines", show=True)
        removed_segments = build_line_segments(
            result["removed_df"],
            max_gap_m=20.0,
            max_gap_sec=180.0,
        )
        for seg in removed_segments:
            folium.PolyLine(
                locations=seg,
                color="red",
                weight=3,
                opacity=0.9,
            ).add_to(fg_removed)
        fg_removed.add_to(m)

    if not result["clean_df"].empty:
        fg_clean = folium.FeatureGroup(name="Filtered farm lines", show=True)
        clean_segments = build_line_segments(
            result["clean_df"],
            max_gap_m=20.0,
            max_gap_sec=180.0,
        )
        for seg in clean_segments:
            folium.PolyLine(
                locations=seg,
                color="lime",
                weight=3,
                opacity=0.95,
            ).add_to(fg_clean)
        fg_clean.add_to(m)

    concave_gj = geom_to_geojson_coords(result["concave_geom"], result["to_wgs"])
    if concave_gj:
        folium.GeoJson(
            concave_gj,
            name="Concave farm boundary",
            style_function=lambda x: {
                "color": "yellow",
                "weight": 2,
                "fillColor": "yellow",
                "fillOpacity": 0.25,
            },
        ).add_to(m)

    path_gj = geom_to_geojson_coords(result["path_geom"], result["to_wgs"])
    if path_gj:
        folium.GeoJson(
            path_gj,
            name="4ft worked strip",
            style_function=lambda x: {
                "color": "cyan",
                "weight": 2,
                "fillColor": "cyan",
                "fillOpacity": 0.20,
            },
        ).add_to(m)

    folium.LayerControl().add_to(m)

    if not result["clean_df"].empty:
        min_lat = result["clean_df"]["latitude"].min()
        max_lat = result["clean_df"]["latitude"].max()
        min_lon = result["clean_df"]["longitude"].min()
        max_lon = result["clean_df"]["longitude"].max()

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
        lof_contamination = st.slider("LOF contamination", min_value=0.0, max_value=0.20, value=0.03, step=0.01)

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

    d1, d2, d3 = st.columns(3)
    d1.metric("Total points", len(result["all_df"]))
    d2.metric("Filtered farm points", len(result["clean_df"]))
    d3.metric("Removed points", len(result["removed_df"]))

    if show_map:
        st.subheader("Filtered GPS map")
        try:
            m = render_map(result)
            st_folium(m, width=1400, height=700, returned_objects=[])
        except Exception as map_error:
            st.warning(f"Map rendering failed: {map_error}")
            st.code(traceback.format_exc())

    st.subheader("Preview")
    p1, p2 = st.columns(2)

    with p1:
        st.write("Filtered farm points")
        if result["clean_df"].empty:
            st.info("No filtered farm points")
        else:
            cols = [c for c in ["latitude", "longitude", "left_rpm", "right_rpm", "rotor_rpm", "timestamp"] if c in result["clean_df"].columns]
            st.dataframe(result["clean_df"][cols], use_container_width=True)

    with p2:
        st.write("Removed points")
        if result["removed_df"].empty:
            st.info("No removed points")
        else:
            cols = [c for c in ["latitude", "longitude", "left_rpm", "right_rpm", "rotor_rpm", "timestamp", "remove_reason"] if c in result["removed_df"].columns]
            st.dataframe(result["removed_df"][cols], use_container_width=True)

    clean_csv = result["clean_df"].to_csv(index=False).encode("utf-8")
    removed_csv = result["removed_df"].to_csv(index=False).encode("utf-8")

    b1, b2 = st.columns(2)
    with b1:
        st.download_button(
            "Download filtered farm points CSV",
            data=clean_csv,
            file_name="cleaned_farm_points.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with b2:
        st.download_button(
            "Download removed points CSV",
            data=removed_csv,
            file_name="removed_points.csv",
            mime="text/csv",
            use_container_width=True,
        )

except Exception as e:
    st.error(f"Error: {e}")
    st.code(traceback.format_exc())
