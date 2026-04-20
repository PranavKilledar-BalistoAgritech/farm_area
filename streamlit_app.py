import traceback

import streamlit as st
import folium
from streamlit_folium import st_folium

from farm_area import (
    read_uploaded_file,
    calculate_farm_area_from_df,
    geom_to_geojson_coords,
)

st.set_page_config(page_title="Farm Area Calculator", layout="wide")


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

    m = folium.Map(location=[center_lat, center_lon], zoom_start=18, tiles=None)

    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google Satellite",
        name="Satellite",
        overlay=False,
        control=True,
    ).add_to(m)

    if not result["removed_df"].empty:
        fg_removed = folium.FeatureGroup(name="Removed points", show=True)
        for _, r in result["removed_df"].iterrows():
            folium.CircleMarker(
                location=[r["latitude"], r["longitude"]],
                radius=2,
                color="red",
                fill=True,
                fill_opacity=0.8,
                popup=str(r.get("remove_reason", "removed")),
            ).add_to(fg_removed)
        fg_removed.add_to(m)

    if not result["clean_df"].empty:
        fg_clean = folium.FeatureGroup(name="Filtered farm points", show=True)
        for _, r in result["clean_df"].iterrows():
            folium.CircleMarker(
                location=[r["latitude"], r["longitude"]],
                radius=2,
                color="lime",
                fill=True,
                fill_opacity=0.9,
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
    return m


try:
    st.title("Farm Area Calculator")
    st.caption("Upload telemetry lat/lon data and calculate farm covered area in guntha.")

    with st.sidebar:
        st.header("Inputs")

        work_width_ft = st.number_input("Working width (ft)", min_value=1.0, value=4.0, step=0.5)
        max_point_jump_m = st.number_input("Max point jump (m)", min_value=1.0, value=25.0, step=1.0)
        density_radius_m = st.number_input("Density radius (m)", min_value=1.0, value=10.0, step=1.0)
        min_neighbors = st.number_input("Min neighbors", min_value=1, value=8, step=1)
        dbscan_eps_m = st.number_input("DBSCAN eps (m)", min_value=1.0, value=12.0, step=1.0)
        dbscan_min_samples = st.number_input("DBSCAN min samples", min_value=2, value=10, step=1)
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
        st.info("Upload a CSV or Excel file with latitude, longitude, and optional timestamp.")
        st.stop()

    input_df = read_uploaded_file(uploaded_file)
    st.success(f"Loaded {len(input_df)} points")

    result = calculate_farm_area_from_df(
        input_df=input_df,
        work_width_ft=work_width_ft,
        max_point_jump_m=max_point_jump_m,
        density_radius_m=density_radius_m,
        min_neighbors=int(min_neighbors),
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
            st_folium(m, width=1400, height=650, returned_objects=[])
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
            cols = [c for c in ["latitude", "longitude", "timestamp"] if c in result["clean_df"].columns]
            st.dataframe(result["clean_df"][cols], use_container_width=True)

    with p2:
        st.write("Removed points")
        if result["removed_df"].empty:
            st.info("No removed points")
        else:
            cols = [c for c in ["latitude", "longitude", "timestamp", "remove_reason"] if c in result["removed_df"].columns]
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
