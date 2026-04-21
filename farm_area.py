from typing import List

import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import Point, LineString
from shapely.ops import unary_union
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor


# =========================================================
# COLUMN NORMALIZATION
# =========================================================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    lat_col = None
    lon_col = None
    left_col = None
    right_col = None
    rotor_col = None
    time_col = None

    for c in df.columns:
        lc = c.lower().strip()

        if lc in ["latitude", "lat"]:
            lat_col = c
        elif lc in ["longitude", "lon", "lng", "long"]:
            lon_col = c
        elif lc in ["left rpm", "left_rpm", "leftrpm", "feedback_left_rpm"]:
            left_col = c
        elif lc in ["right rpm", "right_rpm", "rightrpm", "feedback_right_rpm"]:
            right_col = c
        elif lc in ["rotor rpm", "rotor_rpm", "rotorrpm", "feedback_rotor_rpm"]:
            rotor_col = c
        elif lc in ["timestamp", "time", "created_at", "ts", "datetime", "date_time"]:
            time_col = c

    if lat_col is None or lon_col is None:
        raise ValueError("Could not detect lat/long columns.")

    if left_col is None or right_col is None or rotor_col is None:
        raise ValueError(
            "Could not detect RPM columns. Needed columns: left rpm, right rpm, rotor rpm."
        )

    out = pd.DataFrame()
    out["latitude"] = pd.to_numeric(df[lat_col], errors="coerce")
    out["longitude"] = pd.to_numeric(df[lon_col], errors="coerce")
    out["left_rpm"] = pd.to_numeric(df[left_col], errors="coerce").fillna(0)
    out["right_rpm"] = pd.to_numeric(df[right_col], errors="coerce").fillna(0)
    out["rotor_rpm"] = pd.to_numeric(df[rotor_col], errors="coerce").fillna(0)

    if time_col is not None:
        out["timestamp"] = pd.to_datetime(df[time_col], errors="coerce")
    else:
        out["timestamp"] = pd.NaT

    out = out.dropna(subset=["latitude", "longitude"]).copy()
    out = out[
        out["latitude"].between(-90, 90) &
        out["longitude"].between(-180, 180)
    ].copy()

    if out["timestamp"].notna().any():
        out = out.sort_values("timestamp").reset_index(drop=True)
    else:
        out = out.reset_index(drop=True)

    out["row_id"] = np.arange(len(out))
    return out


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()

    if name.endswith(".csv"):
        raw = pd.read_csv(uploaded_file)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        raw = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Only CSV/XLSX/XLS files are supported.")

    return normalize_columns(raw)


# =========================================================
# GEO
# =========================================================
def get_utm_epsg(lat: float, lon: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return 32600 + zone if lat >= 0 else 32700 + zone


def project_points(df: pd.DataFrame):
    mean_lat = df["latitude"].mean()
    mean_lon = df["longitude"].mean()
    utm_epsg = get_utm_epsg(mean_lat, mean_lon)

    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True)
    to_wgs = Transformer.from_crs(f"EPSG:{utm_epsg}", "EPSG:4326", always_xy=True)

    x, y = to_utm.transform(df["longitude"].values, df["latitude"].values)
    df = df.copy()
    df["x"] = x
    df["y"] = y
    return df, to_utm, to_wgs


# =========================================================
# TRACK FEATURES
# =========================================================
def add_track_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    dx_prev = df["x"].diff()
    dy_prev = df["y"].diff()
    df["dist_prev_m"] = np.sqrt(dx_prev**2 + dy_prev**2).fillna(0)

    dx_next = df["x"].shift(-1) - df["x"]
    dy_next = df["y"].shift(-1) - df["y"]
    df["dist_next_m"] = np.sqrt(dx_next**2 + dy_next**2).fillna(0)

    if df["timestamp"].notna().any():
        df["dt_prev_s"] = df["timestamp"].diff().dt.total_seconds().fillna(0)
        df["dt_next_s"] = (
            (df["timestamp"].shift(-1) - df["timestamp"])
            .dt.total_seconds()
            .fillna(0)
        )
    else:
        df["dt_prev_s"] = np.nan
        df["dt_next_s"] = np.nan

    return df


def remove_gross_jump_outliers(df: pd.DataFrame, max_jump_m: float):
    df = df.copy()

    is_jump = (
        (df["dist_prev_m"] > max_jump_m) &
        (df["dist_next_m"] > max_jump_m)
    )

    if len(df) > 0:
        is_jump.iloc[0] = False
        is_jump.iloc[-1] = False

    kept = df.loc[~is_jump].copy()
    removed = df.loc[is_jump].copy()
    removed["remove_reason"] = "gross_jump"
    return kept.reset_index(drop=True), removed.reset_index(drop=True)


# =========================================================
# RPM CLASSIFICATION
# =========================================================
def classify_rpm_points(
    df: pd.DataFrame,
    left_th: float = 20.0,
    right_th: float = 20.0,
    rotor_th: float = 50.0,
) -> pd.DataFrame:
    df = df.copy()

    df["is_strong_farm"] = (
        (df["left_rpm"] > left_th) &
        (df["right_rpm"] > right_th) &
        (df["rotor_rpm"] > rotor_th)
    )

    df["is_turn_candidate"] = (
        (df["left_rpm"] > left_th) &
        (df["right_rpm"] > right_th) &
        (df["rotor_rpm"] <= rotor_th)
    )

    return df


# =========================================================
# STRONG FARM ANCHOR CLUSTER
# =========================================================
def cluster_strong_farm_points(
    df: pd.DataFrame,
    eps_m: float = 15.0,
    min_samples: int = 15,
):
    strong_df = df[df["is_strong_farm"]].copy()
    non_strong_removed = df[~df["is_strong_farm"]].copy()
    non_strong_removed["remove_reason"] = "not_strong_farm"

    if strong_df.empty:
        return strong_df, non_strong_removed

    coords = strong_df[["x", "y"]].values
    labels = DBSCAN(eps=eps_m, min_samples=min_samples).fit_predict(coords)
    strong_df["cluster_id"] = labels

    noise_removed = strong_df[strong_df["cluster_id"] == -1].copy()
    noise_removed["remove_reason"] = "strong_dbscan_noise"

    anchored = strong_df[strong_df["cluster_id"] != -1].copy()

    if anchored.empty:
        removed = pd.concat([non_strong_removed, noise_removed], ignore_index=True)
        return anchored, removed

    counts = anchored["cluster_id"].value_counts()
    valid_clusters = counts[counts >= min_samples].index.tolist()
    tiny_removed = anchored[~anchored["cluster_id"].isin(valid_clusters)].copy()
    tiny_removed["remove_reason"] = "strong_tiny_cluster"

    anchored = anchored[anchored["cluster_id"].isin(valid_clusters)].copy()

    removed = pd.concat([non_strong_removed, noise_removed, tiny_removed], ignore_index=True)
    return anchored.reset_index(drop=True), removed.reset_index(drop=True)


# =========================================================
# TURN POINT RECOVERY
# =========================================================
def _nearest_distance_to_points(source_xy: np.ndarray, target_xy: np.ndarray) -> np.ndarray:
    if len(source_xy) == 0:
        return np.array([])
    if len(target_xy) == 0:
        return np.full(len(source_xy), np.inf)

    chunk = 1000
    out = np.empty(len(source_xy), dtype=float)

    for s in range(0, len(source_xy), chunk):
        e = min(len(source_xy), s + chunk)
        sub = source_xy[s:e]
        dx = sub[:, None, 0] - target_xy[None, :, 0]
        dy = sub[:, None, 1] - target_xy[None, :, 1]
        dist2 = dx * dx + dy * dy
        out[s:e] = np.sqrt(dist2.min(axis=1))

    return out


def recover_valid_turn_points(
    df: pd.DataFrame,
    anchor_df: pd.DataFrame,
    anchor_buffer_m: float = 18.0,
    turn_time_gap_sec: float = 30.0,
):
    turn_df = df[df["is_turn_candidate"]].copy()

    if turn_df.empty:
        return turn_df.copy(), turn_df.copy()

    anchor_xy = anchor_df[["x", "y"]].values if not anchor_df.empty else np.empty((0, 2))
    turn_xy = turn_df[["x", "y"]].values

    turn_df["dist_to_anchor_m"] = _nearest_distance_to_points(turn_xy, anchor_xy)
    turn_df["near_anchor"] = turn_df["dist_to_anchor_m"] <= anchor_buffer_m

    is_anchor = df["is_strong_farm"].values
    prev_anchor_idx = np.full(len(df), -1, dtype=int)
    next_anchor_idx = np.full(len(df), -1, dtype=int)

    last_idx = -1
    for i in range(len(df)):
        if is_anchor[i]:
            last_idx = i
        prev_anchor_idx[i] = last_idx

    next_idx = -1
    for i in range(len(df) - 1, -1, -1):
        if is_anchor[i]:
            next_idx = i
        next_anchor_idx[i] = next_idx

    df = df.copy()
    df["prev_anchor_idx"] = prev_anchor_idx
    df["next_anchor_idx"] = next_anchor_idx

    turn_df = turn_df.merge(
        df[["row_id", "prev_anchor_idx", "next_anchor_idx"]],
        on="row_id",
        how="left",
    )

    if df["timestamp"].notna().any():
        timestamps = df["timestamp"].reset_index(drop=True)

        def time_ok(row) -> bool:
            p = int(row["prev_anchor_idx"])
            n = int(row["next_anchor_idx"])
            cur_ts = row["timestamp"]
            if pd.isna(cur_ts):
                return False

            prev_ok = False
            next_ok = False

            if p >= 0 and pd.notna(timestamps.iloc[p]):
                prev_ok = abs((cur_ts - timestamps.iloc[p]).total_seconds()) <= turn_time_gap_sec

            if n >= 0 and pd.notna(timestamps.iloc[n]):
                next_ok = abs((timestamps.iloc[n] - cur_ts).total_seconds()) <= turn_time_gap_sec

            return prev_ok and next_ok

        turn_df = turn_df.reset_index(drop=True)
        turn_df["time_connected"] = turn_df.apply(time_ok, axis=1)
    else:
        turn_df["time_connected"] = False

    keep_mask = turn_df["near_anchor"] | turn_df["time_connected"]

    kept = turn_df.loc[keep_mask].copy()
    removed = turn_df.loc[~keep_mask].copy()
    removed["remove_reason"] = "turn_not_near_anchor"

    return kept.reset_index(drop=True), removed.reset_index(drop=True)


# =========================================================
# FINAL CLUSTER + OUTLIER CLEANING
# =========================================================
def cluster_final_farm_points(
    df: pd.DataFrame,
    eps_m: float = 15.0,
    min_samples: int = 12,
):
    if df.empty:
        return df.copy(), df.copy()

    coords = df[["x", "y"]].values
    labels = DBSCAN(eps=eps_m, min_samples=min_samples).fit_predict(coords)
    df = df.copy()
    df["cluster_id"] = labels

    noise_removed = df[df["cluster_id"] == -1].copy()
    noise_removed["remove_reason"] = "final_dbscan_noise"

    kept = df[df["cluster_id"] != -1].copy()
    if kept.empty:
        return kept, noise_removed

    counts = kept["cluster_id"].value_counts()
    valid_clusters = counts[counts >= min_samples].index.tolist()

    tiny_removed = kept[~kept["cluster_id"].isin(valid_clusters)].copy()
    tiny_removed["remove_reason"] = "final_tiny_cluster"

    kept = kept[kept["cluster_id"].isin(valid_clusters)].copy()
    removed = pd.concat([noise_removed, tiny_removed], ignore_index=True)

    return kept.reset_index(drop=True), removed.reset_index(drop=True)


def remove_cluster_outliers(
    df: pd.DataFrame,
    lof_neighbors: int = 20,
    contamination: float = 0.03,
):
    if df.empty:
        return df.copy(), df.copy()

    keep_parts = []
    rem_parts = []

    for cluster_id, grp in df.groupby("cluster_id"):
        grp = grp.copy()
        coords = grp[["x", "y"]].values

        if len(grp) < max(lof_neighbors + 2, 10):
            keep_parts.append(grp)
            continue

        n_neighbors = min(lof_neighbors, len(grp) - 1)
        lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
        )
        pred = lof.fit_predict(coords)

        kept = grp[pred == 1].copy()
        rem = grp[pred == -1].copy()
        rem["remove_reason"] = "local_outlier"

        keep_parts.append(kept)
        rem_parts.append(rem)

    keep_df = pd.concat(keep_parts, ignore_index=True) if keep_parts else df.iloc[0:0].copy()
    rem_df = pd.concat(rem_parts, ignore_index=True) if rem_parts else df.iloc[0:0].copy()

    return keep_df, rem_df


# =========================================================
# GEOMETRY
# =========================================================
def split_into_segments(
    df: pd.DataFrame,
    max_gap_m: float,
    max_gap_sec: float,
    min_segment_points: int,
) -> List[pd.DataFrame]:
    if df.empty:
        return []

    df = df.copy()
    if df["timestamp"].notna().any():
        df = df.sort_values("timestamp").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    df = add_track_features(df)

    split_flags = np.zeros(len(df), dtype=bool)
    split_flags[0] = True

    for i in range(1, len(df)):
        too_far = df.loc[i, "dist_prev_m"] > max_gap_m

        if pd.notna(df.loc[i, "timestamp"]) and pd.notna(df.loc[i - 1, "timestamp"]):
            gap_sec = (df.loc[i, "timestamp"] - df.loc[i - 1, "timestamp"]).total_seconds()
            too_old = gap_sec > max_gap_sec
        else:
            too_old = False

        cluster_changed = df.loc[i, "cluster_id"] != df.loc[i - 1, "cluster_id"]
        split_flags[i] = too_far or too_old or cluster_changed

    df["segment_id"] = np.cumsum(split_flags)

    segments = []
    for _, seg in df.groupby("segment_id"):
        if len(seg) >= min_segment_points:
            segments.append(seg.copy())

    return segments


def build_path_buffer_area(
    segments: List[pd.DataFrame],
    work_width_ft: float,
    effective_width_factor: float = 0.90,
    simplify_tolerance_m: float = 0.75,
):
    """
    More accurate worked strip area:
    - use effective width instead of nominal width
    - simplify path slightly to reduce GPS jitter overcount
    """
    if not segments:
        return None

    effective_width_m = work_width_ft * 0.3048 * effective_width_factor
    half_width = effective_width_m / 2.0
    polys = []

    for seg in segments:
        coords = seg[["x", "y"]].values
        if len(coords) < 2:
            continue

        dedup = [tuple(coords[0])]
        for p in coords[1:]:
            if tuple(p) != dedup[-1]:
                dedup.append(tuple(p))

        if len(dedup) < 2:
            continue

        line = LineString(dedup)

        if simplify_tolerance_m > 0:
            line = line.simplify(simplify_tolerance_m, preserve_topology=False)

        if line.is_empty or line.length == 0:
            continue

        poly = line.buffer(
            half_width,
            cap_style=2,
            join_style=2,
        )

        if not poly.is_empty:
            polys.append(poly)

    if not polys:
        return None

    return unary_union(polys)


def build_concave_boundary(
    df: pd.DataFrame,
    point_buffer_m: float,
    smooth_m: float,
    erode_m: float,
    min_cluster_area_m2: float,
):
    if df.empty:
        return None

    point_polys = [Point(xy).buffer(point_buffer_m) for xy in df[["x", "y"]].values]
    geom = unary_union(point_polys)
    geom = geom.buffer(smooth_m).buffer(-erode_m)

    if geom.is_empty:
        return None

    if geom.geom_type == "Polygon":
        return geom if geom.area >= min_cluster_area_m2 else None

    if geom.geom_type == "MultiPolygon":
        geoms = [g for g in geom.geoms if g.area >= min_cluster_area_m2]
        if not geoms:
            return None
        return unary_union(geoms)

    return geom


def geom_to_geojson_coords(geom, to_wgs: Transformer):
    if geom is None:
        return None

    def convert_ring(ring):
        coords = []
        for x, y in ring.coords:
            lon, lat = to_wgs.transform(x, y)
            coords.append([lat, lon])
        return coords

    if geom.geom_type == "Polygon":
        return {
            "type": "Polygon",
            "coordinates": [convert_ring(geom.exterior)],
        }

    if geom.geom_type == "MultiPolygon":
        polys = []
        for poly in geom.geoms:
            polys.append([convert_ring(poly.exterior)])
        return {
            "type": "MultiPolygon",
            "coordinates": polys,
        }

    return None


# =========================================================
# MAIN
# =========================================================
def calculate_farm_area_from_df(
    input_df: pd.DataFrame,
    work_width_ft: float = 4.0,
    max_point_jump_m: float = 25.0,
    density_radius_m: float = 10.0,
    min_neighbors: int = 8,
    dbscan_eps_m: float = 15.0,
    dbscan_min_samples: int = 15,
    lof_neighbors: int = 20,
    lof_contamination: float = 0.03,
    max_segment_gap_m: float = 20.0,
    max_segment_time_sec: float = 180.0,
    min_segment_points: int = 3,
    point_buffer_m: float = 2.2,
    boundary_smooth_m: float = 3.0,
    boundary_erode_m: float = 2.0,
    min_cluster_area_m2: float = 20.0,
):
    all_df = input_df.copy()
    df, _, to_wgs = project_points(all_df)
    df = add_track_features(df)

    df1, rem1 = remove_gross_jump_outliers(df, max_point_jump_m)
    df1 = add_track_features(df1)

    df2 = classify_rpm_points(
        df1,
        left_th=20.0,
        right_th=20.0,
        rotor_th=50.0,
    )

    anchor_df, rem_anchor = cluster_strong_farm_points(
        df2,
        eps_m=dbscan_eps_m,
        min_samples=dbscan_min_samples,
    )

    if anchor_df.empty:
        removed = pd.concat([rem1, rem_anchor], ignore_index=True)
        return {
            "all_df": all_df,
            "clean_df": anchor_df,
            "removed_df": removed,
            "concave_geom": None,
            "path_geom": None,
            "concave_area_m2": 0.0,
            "concave_area_sqft": 0.0,
            "concave_area_guntha": 0.0,
            "path_area_m2": 0.0,
            "path_area_sqft": 0.0,
            "path_area_guntha": 0.0,
            "to_wgs": to_wgs,
        }

    turn_kept, turn_removed = recover_valid_turn_points(
        df2,
        anchor_df=anchor_df,
        anchor_buffer_m=18.0,
        turn_time_gap_sec=30.0,
    )

    final_candidates = pd.concat([anchor_df, turn_kept], ignore_index=True)
    final_candidates = final_candidates.drop_duplicates(subset=["row_id"]).reset_index(drop=True)
    final_candidates["neighbor_count"] = 0
    final_candidates["is_dense"] = True

    farm_df, rem2 = cluster_final_farm_points(
        final_candidates,
        eps_m=dbscan_eps_m,
        min_samples=max(10, dbscan_min_samples // 2),
    )

    if farm_df.empty:
        removed = pd.concat([rem1, rem_anchor, turn_removed, rem2], ignore_index=True)
        return {
            "all_df": all_df,
            "clean_df": farm_df,
            "removed_df": removed,
            "concave_geom": None,
            "path_geom": None,
            "concave_area_m2": 0.0,
            "concave_area_sqft": 0.0,
            "concave_area_guntha": 0.0,
            "path_area_m2": 0.0,
            "path_area_sqft": 0.0,
            "path_area_guntha": 0.0,
            "to_wgs": to_wgs,
        }

    farm_df, rem3 = remove_cluster_outliers(
        farm_df,
        lof_neighbors=lof_neighbors,
        contamination=lof_contamination,
    )

    removed = pd.concat([rem1, rem_anchor, turn_removed, rem2, rem3], ignore_index=True)

    if farm_df.empty:
        return {
            "all_df": all_df,
            "clean_df": farm_df,
            "removed_df": removed,
            "concave_geom": None,
            "path_geom": None,
            "concave_area_m2": 0.0,
            "concave_area_sqft": 0.0,
            "concave_area_guntha": 0.0,
            "path_area_m2": 0.0,
            "path_area_sqft": 0.0,
            "path_area_guntha": 0.0,
            "to_wgs": to_wgs,
        }

    clean_df = farm_df.copy()
    if clean_df["timestamp"].notna().any():
        clean_df = clean_df.sort_values("timestamp").reset_index(drop=True)
    else:
        clean_df = clean_df.reset_index(drop=True)

    concave_geom = build_concave_boundary(
        clean_df,
        point_buffer_m=point_buffer_m,
        smooth_m=boundary_smooth_m,
        erode_m=boundary_erode_m,
        min_cluster_area_m2=min_cluster_area_m2,
    )

    segments = split_into_segments(
        clean_df,
        max_gap_m=max_segment_gap_m,
        max_gap_sec=max_segment_time_sec,
        min_segment_points=min_segment_points,
    )

    path_geom = build_path_buffer_area(
        segments,
        work_width_ft=work_width_ft,
        effective_width_factor=0.90,
        simplify_tolerance_m=0.75,
    )

    concave_area_m2 = concave_geom.area if concave_geom is not None else 0.0
    path_area_m2 = path_geom.area if path_geom is not None else 0.0

    concave_sqft = concave_area_m2 * 10.76391041671
    path_sqft = path_area_m2 * 10.76391041671

    return {
        "all_df": all_df,
        "clean_df": clean_df,
        "removed_df": removed,
        "concave_geom": concave_geom,
        "path_geom": path_geom,
        "concave_area_m2": concave_area_m2,
        "concave_area_sqft": concave_sqft,
        "concave_area_guntha": concave_sqft / 1089.0,
        "path_area_m2": path_area_m2,
        "path_area_sqft": path_sqft,
        "path_area_guntha": path_sqft / 1089.0,
        "to_wgs": to_wgs,
    }
