import math
import traceback

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from farm_area import (
    read_uploaded_file,
    calculate_farm_area_from_df,
    geom_to_geojson_coords,
)


st.set_page_config(
    page_title="Farm Area Calculator",
    page_icon="🌾",
    layout="wide",
)


# =========================
# Helper functions
# =========================

def get_lat_lon_cols(df):
    """Return latitude and longitude column names used by the backend."""
    if df is None or df.empty:
        return None, None

    if "latitude" in df.columns and "longitude" in df.columns:
        return "latitude", "longitude"

    if "lat" in df.columns and "lon" in df.columns:
        return "lat", "lon"

    return None, None


def get_time_column(df):
    """Return available timestamp column."""
    if df is None or df.empty:
        return None

    for col in ["timestamp", "time", "created_at", "datetime", "date_time"]:
        if col in df.columns:
            return col

    return None


def haversine_m(lat1, lon1, lat2, lon2):
    """Distance between two GPS points in meters."""
    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
        return None

    r = 6371000.0

    lat1 = math.radians(float(lat1))
    lon1 = math.radians(float(lon1))
    lat2 = math.radians(float(lat2))
    lon2 = math.radians(float(lon2))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))
    return r * c



def bearing_deg(lat1, lon1, lat2, lon2):
    """Bearing from first GPS point to second GPS point in degrees."""
    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
        return None

    lat1 = math.radians(float(lat1))
    lat2 = math.radians(float(lat2))
    dlon = math.radians(float(lon2) - float(lon1))

    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)

    brg = math.degrees(math.atan2(x, y))
    return (brg + 360.0) % 360.0


def angle_diff_deg(a, b):
    """Smallest difference between two headings in degrees."""
    if a is None or b is None:
        return 0.0
    return abs((a - b + 180.0) % 360.0 - 180.0)


def get_strong_farm_points(df, wheel_th=20.0, rotor_th=50.0):
    """
    Map-only filter.
    Keeps only real farm-working points: both wheels moving and rotor running.
    This makes the map cleaner and avoids drawing turns/noise as farm sari lines.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    needed = {"left_rpm", "right_rpm", "rotor_rpm"}
    if not needed.issubset(df.columns):
        return df.copy()

    out = df.copy()
    out["left_rpm"] = pd.to_numeric(out["left_rpm"], errors="coerce").fillna(0)
    out["right_rpm"] = pd.to_numeric(out["right_rpm"], errors="coerce").fillna(0)
    out["rotor_rpm"] = pd.to_numeric(out["rotor_rpm"], errors="coerce").fillna(0)

    mask = (
        (out["left_rpm"].abs() > wheel_th)
        & (out["right_rpm"].abs() > wheel_th)
        & (out["rotor_rpm"].abs() > rotor_th)
    )

    return out.loc[mask].copy()


def build_sari_line_segments(
    df,
    max_gap_m=25.0,
    max_gap_sec=60.0,
    max_heading_change_deg=35.0,
    min_move_m=1.0,
    min_segment_points=8,
):
    """
    Build sari-wise straight farm lines for map presentation.

    The normal GPS trail connects points by time only, which creates cross-lines.
    This function connects points only when distance, time, and heading direction are valid.
    If heading changes sharply, it starts a new sari line.
    """
    if df is None or df.empty:
        return []

    lat_col, lon_col = get_lat_lon_cols(df)
    if lat_col is None or lon_col is None:
        return []

    work = df.copy()
    work[lat_col] = pd.to_numeric(work[lat_col], errors="coerce")
    work[lon_col] = pd.to_numeric(work[lon_col], errors="coerce")
    work = work.dropna(subset=[lat_col, lon_col]).copy()

    if work.empty:
        return []

    time_col = get_time_column(work)

    if time_col:
        work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
        if work[time_col].notna().any():
            work = work.sort_values(time_col).reset_index(drop=True)
        elif "row_id" in work.columns:
            work = work.sort_values("row_id").reset_index(drop=True)
    elif "row_id" in work.columns:
        work = work.sort_values("row_id").reset_index(drop=True)
    else:
        work = work.reset_index(drop=True)

    segments = []
    current_segment = []
    prev_row = None
    last_heading = None

    for _, row in work.iterrows():
        lat = float(row[lat_col])
        lon = float(row[lon_col])
        point = [lat, lon]

        if prev_row is None:
            current_segment = [point]
            prev_row = row
            last_heading = None
            continue

        dist_m = haversine_m(prev_row[lat_col], prev_row[lon_col], lat, lon)

        # Ignore very tiny movement points for line drawing; they create zig-zag noise.
        if dist_m is not None and dist_m < min_move_m:
            continue

        curr_heading = bearing_deg(prev_row[lat_col], prev_row[lon_col], lat, lon)
        start_new_segment = False

        if dist_m is None or dist_m > max_gap_m:
            start_new_segment = True

        if time_col:
            prev_time = prev_row.get(time_col)
            curr_time = row.get(time_col)
            if pd.notna(prev_time) and pd.notna(curr_time):
                gap_sec = abs((curr_time - prev_time).total_seconds())
                if gap_sec > max_gap_sec:
                    start_new_segment = True

        if last_heading is not None:
            heading_change = angle_diff_deg(curr_heading, last_heading)
            if heading_change > max_heading_change_deg:
                start_new_segment = True

        if start_new_segment:
            if len(current_segment) >= int(min_segment_points):
                segments.append(current_segment)
            current_segment = [point]
            last_heading = None
        else:
            current_segment.append(point)
            last_heading = curr_heading

        prev_row = row

    if len(current_segment) >= int(min_segment_points):
        segments.append(current_segment)

    return segments

def build_line_segments(
    df,
    max_gap_m=8.0,
    max_gap_sec=15.0,
    min_segment_points=3,
):
    """
    Build clean GPS line segments.
    This avoids wrong cross-lines by breaking lines on large distance/time gaps.
    """
    if df is None or df.empty:
        return []

    lat_col, lon_col = get_lat_lon_cols(df)
    if lat_col is None or lon_col is None:
        return []

    work = df.copy()
    work[lat_col] = pd.to_numeric(work[lat_col], errors="coerce")
    work[lon_col] = pd.to_numeric(work[lon_col], errors="coerce")
    work = work.dropna(subset=[lat_col, lon_col]).copy()

    if work.empty:
        return []

    time_col = get_time_column(work)

    if time_col:
        work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
        if work[time_col].notna().any():
            work = work.sort_values(time_col).reset_index(drop=True)
        elif "row_id" in work.columns:
            work = work.sort_values("row_id").reset_index(drop=True)
    elif "row_id" in work.columns:
        work = work.sort_values("row_id").reset_index(drop=True)
    else:
        work = work.reset_index(drop=True)

    segments = []
    current_segment = []
    prev_row = None

    for _, row in work.iterrows():
        lat = float(row[lat_col])
        lon = float(row[lon_col])
        point = [lat, lon]

        if prev_row is None:
            current_segment = [point]
            prev_row = row
            continue

        dist_m = haversine_m(prev_row[lat_col], prev_row[lon_col], lat, lon)
        start_new_segment = False

        if dist_m is None or dist_m > max_gap_m:
            start_new_segment = True

        if time_col:
            prev_time = prev_row.get(time_col)
            curr_time = row.get(time_col)
            if pd.notna(prev_time) and pd.notna(curr_time):
                gap_sec = abs((curr_time - prev_time).total_seconds())
                if gap_sec > max_gap_sec:
                    start_new_segment = True

        if start_new_segment:
            if len(current_segment) >= int(min_segment_points):
                segments.append(current_segment)
            current_segment = [point]
        else:
            current_segment.append(point)

        prev_row = row

    if len(current_segment) >= int(min_segment_points):
        segments.append(current_segment)

    return segments


def classify_on_road_points(df, wheel_th=20.0, rotor_th=50.0):
    """On_Road = rotor OFF and either wheel is moving."""
    if df is None or df.empty:
        return pd.DataFrame()

    needed = {"left_rpm", "right_rpm", "rotor_rpm"}
    if not needed.issubset(df.columns):
        return df.iloc[0:0].copy()

    out = df.copy()
    out["left_rpm"] = pd.to_numeric(out["left_rpm"], errors="coerce").fillna(0)
    out["right_rpm"] = pd.to_numeric(out["right_rpm"], errors="coerce").fillna(0)
    out["rotor_rpm"] = pd.to_numeric(out["rotor_rpm"], errors="coerce").fillna(0)

    mask = (
        (out["rotor_rpm"].abs() <= rotor_th)
        & (
            (out["left_rpm"].abs() > wheel_th)
            | (out["right_rpm"].abs() > wheel_th)
        )
    )

    return out.loc[mask].copy()


def remove_rows_by_row_id(source_df, remove_df):
    """Remove rows from source_df if row_id exists in remove_df."""
    if source_df is None or source_df.empty:
        return pd.DataFrame()

    if remove_df is None or remove_df.empty:
        return source_df.copy()

    if "row_id" not in source_df.columns or "row_id" not in remove_df.columns:
        return source_df.copy()

    remove_ids = set(remove_df["row_id"].dropna().tolist())
    return source_df.loc[~source_df["row_id"].isin(remove_ids)].copy()


def add_geojson_layer(m, geom, to_wgs, name, color, fill_color, fill_opacity):
    """Add backend shapely geometry on map."""
    if geom is None or getattr(geom, "is_empty", True):
        return

    try:
        geojson_data = geom_to_geojson_coords(geom, to_wgs)
        if not geojson_data:
            return

        folium.GeoJson(
            geojson_data,
            name=name,
            style_function=lambda x: {
                "color": color,
                "weight": 2,
                "fillColor": fill_color,
                "fillOpacity": fill_opacity,
            },
        ).add_to(m)
    except Exception:
        pass


def render_map(result, line_gap_m=80.0, line_gap_sec=300.0):
    """Render satellite map with timestamp-based farm lines.

    Map display uses filtered farm points in timestamp order only.
    This is only for presentation and does not change area calculation.
    """
    all_df = result.get("all_df", pd.DataFrame())
    clean_df = result.get("clean_df", pd.DataFrame())
    removed_df = result.get("removed_df", pd.DataFrame())

    concave_geom = result.get("concave_geom")
    path_geom = result.get("path_geom")
    to_wgs = result.get("to_wgs")

    lat_col, lon_col = get_lat_lon_cols(all_df)
    if all_df is None or all_df.empty or lat_col is None or lon_col is None:
        st.warning("No GPS data available for map.")
        return

    valid_points = all_df[[lat_col, lon_col]].dropna()
    if valid_points.empty:
        st.warning("No valid GPS points available for map.")
        return

    center_lat = float(valid_points[lat_col].mean())
    center_lon = float(valid_points[lon_col].mean())

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=20,
        max_zoom=22,
        min_zoom=3,
        control_scale=True,
        zoom_control=True,
        tiles=None,
        prefer_canvas=True,
        scrollWheelZoom=True,
        doubleClickZoom=True,
        dragging=True,
        touchZoom=True,
        boxZoom=True,
        keyboard=True,
    )

    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google Satellite",
        name="Satellite",
        overlay=False,
        control=True,
        max_zoom=22,
    ).add_to(m)

    on_farm_layer = folium.FeatureGroup(name="On_Farm", show=True)
    on_road_layer = folium.FeatureGroup(name="On_Road", show=True)
    removed_layer = folium.FeatureGroup(name="Removed_Points", show=True)

    # Green On_Farm lines
    # Presentation logic: join filtered farm points only in timestamp/order sequence.
    # Do not use heading filtering here, because GPS jitter breaks the sari lines.
    # Use all raw On_Farm RPM points first so complete sari movement is visible.
    map_farm_df = get_strong_farm_points(all_df)

    # If RPM columns are missing or the strict RPM filter returns nothing, fall back to clean_df.
    if map_farm_df is None or map_farm_df.empty:
        map_farm_df = clean_df.copy()

    on_farm_segments = build_line_segments(
        map_farm_df,
        max_gap_m=line_gap_m,
        max_gap_sec=line_gap_sec,
        min_segment_points=2,
    )

    for seg in on_farm_segments:
        folium.PolyLine(
            locations=seg,
            color="lime",
            weight=4,
            opacity=0.95,
        ).add_to(on_farm_layer)

    # Yellow On_Road lines
    on_road_df = classify_on_road_points(all_df)
    on_road_segments = build_line_segments(
        on_road_df,
        max_gap_m=line_gap_m,
        max_gap_sec=line_gap_sec,
        min_segment_points=3,
    )

    for seg in on_road_segments:
        folium.PolyLine(
            locations=seg,
            color="yellow",
            weight=5,
            opacity=0.95,
        ).add_to(on_road_layer)

    # Red removed points
    removed_display_df = remove_rows_by_row_id(removed_df, on_road_df)
    rem_lat_col, rem_lon_col = get_lat_lon_cols(removed_display_df)

    if removed_display_df is not None and not removed_display_df.empty and rem_lat_col and rem_lon_col:
        for _, row in removed_display_df.iterrows():
            lat = row.get(rem_lat_col)
            lon = row.get(rem_lon_col)

            if pd.isna(lat) or pd.isna(lon):
                continue

            reason = row.get("remove_reason", row.get("removed_reason", "removed"))

            folium.CircleMarker(
                location=[float(lat), float(lon)],
                radius=2,
                color="red",
                fill=True,
                fill_color="red",
                fill_opacity=0.9,
                popup=str(reason),
            ).add_to(removed_layer)

    on_farm_layer.add_to(m)
    on_road_layer.add_to(m)
    removed_layer.add_to(m)

    if to_wgs is not None:
        add_geojson_layer(
            m=m,
            geom=concave_geom,
            to_wgs=to_wgs,
            name="Farm_Boundary",
            color="yellow",
            fill_color="yellow",
            fill_opacity=0.12,
        )

        add_geojson_layer(
            m=m,
            geom=path_geom,
            to_wgs=to_wgs,
            name="Worked_Strip",
            color="cyan",
            fill_color="cyan",
            fill_opacity=0.18,
        )

    # Do not call m.fit_bounds() here.
    # fit_bounds re-renders the map to full view and makes manual zoom feel like it is not working.

    folium.LayerControl(collapsed=False).add_to(m)

    st_folium(
        m,
        width="100%",
        height=750,
        returned_objects=[],
        key="farm_map",
    )


# =========================
# Sidebar
# =========================

st.sidebar.title("Settings")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV / Excel file",
    type=["csv", "xlsx", "xls"],
)

st.sidebar.markdown("---")

work_width_ft = st.sidebar.number_input(
    "Working width / sari width (ft)",
    min_value=1.0,
    max_value=20.0,
    value=4.0,
    step=0.5,
)

max_jump_m = st.sidebar.number_input(
    "Max GPS jump allowed (m)",
    min_value=1.0,
    max_value=100.0,
    value=25.0,
    step=1.0,
)

dbscan_eps_m = st.sidebar.number_input(
    "DBSCAN distance eps (m)",
    min_value=1.0,
    max_value=50.0,
    value=15.0,
    step=1.0,
)

dbscan_min_samples = st.sidebar.number_input(
    "DBSCAN min samples",
    min_value=2,
    max_value=100,
    value=15,
    step=1,
)

lof_neighbors = st.sidebar.number_input(
    "LOF neighbors",
    min_value=5,
    max_value=100,
    value=20,
    step=1,
)

lof_contamination = st.sidebar.number_input(
    "LOF contamination",
    min_value=0.0,
    max_value=0.20,
    value=0.01,
    step=0.01,
    format="%.2f",
)

boundary_point_buffer_m = st.sidebar.number_input(
    "Boundary point buffer (m)",
    min_value=0.5,
    max_value=20.0,
    value=2.2,
    step=0.1,
)

boundary_smooth_m = st.sidebar.number_input(
    "Boundary smooth (m)",
    min_value=0.0,
    max_value=20.0,
    value=3.0,
    step=0.1,
)

boundary_erode_m = st.sidebar.number_input(
    "Boundary erode (m)",
    min_value=0.0,
    max_value=20.0,
    value=2.0,
    step=0.1,
)

st.sidebar.markdown("---")

line_gap_m = st.sidebar.number_input(
    "Map line max GPS gap (m)",
    min_value=5.0,
    max_value=150.0,
    value=80.0,
    step=5.0,
)

line_gap_sec = st.sidebar.number_input(
    "Map line max time gap (sec)",
    min_value=10.0,
    max_value=600.0,
    value=300.0,
    step=10.0,
)

show_map = st.sidebar.checkbox("Show map", value=True)


# =========================
# Main UI
# =========================

st.title("Farm Area Calculator")
st.write("Upload machine GPS and RPM data to calculate worked farm area.")

if uploaded_file is None:
    st.info("Please upload CSV or Excel file.")
    st.stop()

try:
    df = read_uploaded_file(uploaded_file)

    if df is None or df.empty:
        st.error("No valid data found in uploaded file.")
        st.stop()

    result = calculate_farm_area_from_df(
        input_df=df,
        work_width_ft=work_width_ft,
        max_point_jump_m=max_jump_m,
        dbscan_eps_m=dbscan_eps_m,
        dbscan_min_samples=int(dbscan_min_samples),
        lof_neighbors=int(lof_neighbors),
        lof_contamination=lof_contamination,
        point_buffer_m=boundary_point_buffer_m,
        boundary_smooth_m=boundary_smooth_m,
        boundary_erode_m=boundary_erode_m,
    )

    all_df = result.get("all_df", pd.DataFrame())
    clean_df = result.get("clean_df", pd.DataFrame())
    removed_df = result.get("removed_df", pd.DataFrame())

    on_road_df = classify_on_road_points(all_df)
    removed_display_df = remove_rows_by_row_id(removed_df, on_road_df)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("4 ft strip check (guntha)", round(result.get("path_area_guntha", 0), 3))

    with c2:
        st.metric("Total points", len(all_df) if all_df is not None else 0)

    with c3:
        st.metric("On_Farm points", len(clean_df) if clean_df is not None else 0)

    with c4:
        st.metric("Removed points", len(removed_display_df) if removed_display_df is not None else 0)

    c5, c6, c7 = st.columns(3)

    with c5:
        st.metric("On_Road points", len(on_road_df) if on_road_df is not None else 0)

    with c6:
        st.metric("Raw path area sq.ft", round(result.get("path_area_sqft", 0), 2))

    with c7:
        st.metric("Boundary area guntha", round(result.get("concave_area_guntha", 0), 3))

    if show_map:
        st.subheader("Filtered GPS map")
        st.caption(
            "Use layer control to toggle On_Farm, On_Road, Removed_Points, Farm_Boundary, and Worked_Strip."
        )
        render_map(result, line_gap_m=line_gap_m, line_gap_sec=line_gap_sec)

    st.subheader("Data Preview")

    tab1, tab2, tab3, tab4 = st.tabs(["On_Farm", "On_Road", "Removed", "All Data"])

    with tab1:
        st.dataframe(clean_df, use_container_width=True)
        if clean_df is not None and not clean_df.empty:
            st.download_button(
                label="Download On_Farm CSV",
                data=clean_df.to_csv(index=False).encode("utf-8"),
                file_name="on_farm_points.csv",
                mime="text/csv",
            )

    with tab2:
        st.dataframe(on_road_df, use_container_width=True)
        if on_road_df is not None and not on_road_df.empty:
            st.download_button(
                label="Download On_Road CSV",
                data=on_road_df.to_csv(index=False).encode("utf-8"),
                file_name="on_road_points.csv",
                mime="text/csv",
            )

    with tab3:
        st.dataframe(removed_display_df, use_container_width=True)
        if removed_display_df is not None and not removed_display_df.empty:
            st.download_button(
                label="Download Removed CSV",
                data=removed_display_df.to_csv(index=False).encode("utf-8"),
                file_name="removed_points.csv",
                mime="text/csv",
            )

    with tab4:
        st.dataframe(all_df, use_container_width=True)
        if all_df is not None and not all_df.empty:
            st.download_button(
                label="Download All Data CSV",
                data=all_df.to_csv(index=False).encode("utf-8"),
                file_name="all_points.csv",
                mime="text/csv",
            )

except Exception as e:
    st.error("Something went wrong while processing the file.")
    st.write(str(e))
    st.code(traceback.format_exc())
