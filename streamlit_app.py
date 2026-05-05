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


# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Farm Area Calculator",
    page_icon="🌾",
    layout="wide",
)


# =========================
# HELPER FUNCTIONS
# =========================

def haversine_m(lat1, lon1, lat2, lon2):
    """
    Distance between two GPS points in meters.
    """
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


def get_time_column(df):
    """
    Finds timestamp column from dataframe.
    """
    for col in ["timestamp", "time", "created_at", "datetime", "date_time"]:
        if col in df.columns:
            return col
    return None


def build_line_segments(
    df,
    max_gap_m=8,
    max_gap_sec=15,
    min_segment_points=3,
):
    """
    Builds clean line segments from GPS points.

    This avoids wrong cross lines by breaking the line when:
    - GPS distance gap is large
    - time gap is large
    - point is invalid
    """
    if df is None or df.empty:
        return []

    work = df.copy()

    if "lat" not in work.columns or "lon" not in work.columns:
        return []

    time_col = get_time_column(work)

    if time_col:
        work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
        work = work.sort_values(time_col)
    elif "row_id" in work.columns:
        work = work.sort_values("row_id")

    segments = []
    current_segment = []

    prev_lat = None
    prev_lon = None
    prev_time = None

    for _, row in work.iterrows():
        lat = row.get("lat")
        lon = row.get("lon")

        if pd.isna(lat) or pd.isna(lon):
            continue

        curr_time = row.get(time_col) if time_col else None

        start_new_segment = False

        if prev_lat is not None and prev_lon is not None:
            dist_m = haversine_m(prev_lat, prev_lon, lat, lon)

            if dist_m is None or dist_m > max_gap_m:
                start_new_segment = True

            if time_col and prev_time is not None and curr_time is not None:
                if not pd.isna(prev_time) and not pd.isna(curr_time):
                    gap_sec = abs((curr_time - prev_time).total_seconds())
                    if gap_sec > max_gap_sec:
                        start_new_segment = True

        if start_new_segment:
            if len(current_segment) >= min_segment_points:
                segments.append(current_segment)

            current_segment = []

        current_segment.append([float(lat), float(lon)])

        prev_lat = lat
        prev_lon = lon
        prev_time = curr_time

    if len(current_segment) >= min_segment_points:
        segments.append(current_segment)

    return segments


def classify_on_road_points(df, wheel_th=20, rotor_th=50):
    """
    On_Road:
    Rotor is OFF and wheels are moving.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    work = df.copy()

    required_cols = ["left_rpm", "right_rpm", "rotor_rpm"]
    for col in required_cols:
        if col not in work.columns:
            work[col] = 0

    on_road_mask = (
        (work["rotor_rpm"].fillna(0) <= rotor_th)
        & (
            (work["left_rpm"].fillna(0).abs() > wheel_th)
            | (work["right_rpm"].fillna(0).abs() > wheel_th)
        )
    )

    return work[on_road_mask].copy()


def remove_rows_by_row_id(source_df, remove_df):
    """
    Removes rows from source_df where row_id exists in remove_df.
    Useful to avoid showing On_Road points also as Removed points.
    """
    if source_df is None or source_df.empty:
        return pd.DataFrame()

    if remove_df is None or remove_df.empty:
        return source_df.copy()

    if "row_id" not in source_df.columns or "row_id" not in remove_df.columns:
        return source_df.copy()

    remove_ids = set(remove_df["row_id"].dropna().tolist())
    return source_df[~source_df["row_id"].isin(remove_ids)].copy()


def add_geojson_layer(m, geom, to_wgs, name, color, fill_color, fill_opacity):
    """
    Adds shapely geometry as GeoJSON layer on folium map.
    """
    if geom is None or geom.is_empty:
        return

    try:
        geojson_data = geom_to_geojson_coords(geom, to_wgs)

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


def render_map(result):
    """
    Renders GPS map with clean segmented lines.
    """
    all_df = result.get("all_df", pd.DataFrame())
    clean_df = result.get("clean_df", pd.DataFrame())
    removed_df = result.get("removed_df", pd.DataFrame())

    concave_geom = result.get("concave_geom")
    path_geom = result.get("path_geom")
    to_wgs = result.get("to_wgs")

    if all_df is None or all_df.empty:
        st.warning("No GPS data available for map.")
        return

    center_lat = all_df["lat"].mean()
    center_lon = all_df["lon"].mean()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=19,
        control_scale=True,
        tiles=None,
    )

    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google Satellite",
        name="Satellite",
        overlay=False,
        control=True,
    ).add_to(m)

    # =========================
    # LAYERS
    # =========================

    on_farm_layer = folium.FeatureGroup(name="On_Farm", show=True)
    on_road_layer = folium.FeatureGroup(name="On_Road", show=True)
    removed_layer = folium.FeatureGroup(name="Removed_Points", show=True)

    # =========================
    # ON FARM GREEN LINES
    # =========================

    on_farm_segments = build_line_segments(
        clean_df,
        max_gap_m=8,
        max_gap_sec=15,
        min_segment_points=3,
    )

    for seg in on_farm_segments:
        folium.PolyLine(
            locations=seg,
            color="lime",
            weight=3,
            opacity=0.9,
        ).add_to(on_farm_layer)

    # =========================
    # ON ROAD YELLOW LINES
    # =========================

    on_road_df = classify_on_road_points(all_df)

    on_road_segments = build_line_segments(
        on_road_df,
        max_gap_m=8,
        max_gap_sec=15,
        min_segment_points=3,
    )

    for seg in on_road_segments:
        folium.PolyLine(
            locations=seg,
            color="yellow",
            weight=3,
            opacity=0.9,
        ).add_to(on_road_layer)

    # =========================
    # REMOVED RED POINTS
    # =========================

    removed_display_df = remove_rows_by_row_id(removed_df, on_road_df)

    if removed_display_df is not None and not removed_display_df.empty:
        for _, row in removed_display_df.iterrows():
            lat = row.get("lat")
            lon = row.get("lon")

            if pd.isna(lat) or pd.isna(lon):
                continue

            reason = row.get("removed_reason", "removed")

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

    # =========================
    # FARM BOUNDARY / WORKED STRIP
    # =========================

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

    # =========================
    # FIT BOUNDS
    # =========================

    try:
        valid_points = all_df[["lat", "lon"]].dropna()
        if not valid_points.empty:
            bounds = [
                [valid_points["lat"].min(), valid_points["lon"].min()],
                [valid_points["lat"].max(), valid_points["lon"].max()],
            ]
            m.fit_bounds(bounds)
    except Exception:
        pass

    folium.LayerControl(collapsed=False).add_to(m)

    st_folium(
        m,
        width="100%",
        height=600,
        returned_objects=[],
    )


# =========================
# SIDEBAR
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

show_map = st.sidebar.checkbox("Show map", value=True)


# =========================
# MAIN UI
# =========================

st.title("Farm Area Calculator")

st.write(
    "Upload machine GPS and RPM data to calculate worked farm area."
)

if uploaded_file is None:
    st.info("Please upload CSV or Excel file.")
    st.stop()

try:
    df = read_uploaded_file(uploaded_file)

    if df is None or df.empty:
        st.error("No valid data found in uploaded file.")
        st.stop()

    result = calculate_farm_area_from_df(
        df,
        work_width_ft=work_width_ft,
        max_jump_m=max_jump_m,
        dbscan_eps_m=dbscan_eps_m,
        dbscan_min_samples=dbscan_min_samples,
        lof_neighbors=lof_neighbors,
        lof_contamination=lof_contamination,
        boundary_point_buffer_m=boundary_point_buffer_m,
        boundary_smooth_m=boundary_smooth_m,
        boundary_erode_m=boundary_erode_m,
    )

    all_df = result.get("all_df", pd.DataFrame())
    clean_df = result.get("clean_df", pd.DataFrame())
    removed_df = result.get("removed_df", pd.DataFrame())

    on_road_df = classify_on_road_points(all_df)
    removed_display_df = remove_rows_by_row_id(removed_df, on_road_df)

    # =========================
    # METRICS
    # =========================

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "4 ft strip check (guntha)",
            round(result.get("path_area_guntha", 0), 3),
        )

    with c2:
        st.metric(
            "Total points",
            len(all_df) if all_df is not None else 0,
        )

    with c3:
        st.metric(
            "On_Farm points",
            len(clean_df) if clean_df is not None else 0,
        )

    with c4:
        st.metric(
            "Removed points",
            len(removed_display_df) if removed_display_df is not None else 0,
        )

    c5, c6, c7 = st.columns(3)

    with c5:
        st.metric(
            "On_Road points",
            len(on_road_df) if on_road_df is not None else 0,
        )

    with c6:
        st.metric(
            "Raw path area sq.ft",
            round(result.get("path_area_sqft", 0), 2),
        )

    with c7:
        st.metric(
            "Boundary area guntha",
            round(result.get("concave_area_guntha", 0), 3),
        )

    # =========================
    # MAP
    # =========================

    if show_map:
        st.subheader("Filtered GPS map")
        st.caption(
            "Use the layer control on the map to toggle On_Farm, On_Road, Removed_Points, Farm_Boundary, and Worked_Strip."
        )
        render_map(result)

    # =========================
    # DATA TABLES
    # =========================

    st.subheader("Data Preview")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["On_Farm", "On_Road", "Removed", "All Data"]
    )

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
