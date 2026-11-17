"""Spatial basin masks and area-weighted aggregation for the Yangtze River Basin."""

from pathlib import Path
import numpy as np
import shapefile
from shapely.geometry import Point, box, shape
from shapely.ops import unary_union


SUBBASIN_LABELS = {
    0: "Basin",
    1: "Upper",
    2: "Middle",
    3: "Lower",
}

TRIBUTARY_OUTLETS = {
    "Yalong": (101.80, 26.68),
    "Min-Tuo": (103.94, 29.10),
    "Jialing": (106.45, 29.79),
    "Wujiang": (107.42, 29.36),
    "Hanjiang": (113.40, 30.40),
    "Dongting": (112.97, 29.29),
    "Poyang": (116.19, 29.60),
}


def load_basin_mask(mask_path):
    """Load gridded basin fractions and subbasin zone IDs."""
    data = np.load(mask_path, allow_pickle=True)
    return {
        "lat": data["lat"].astype(float),
        "lon": data["lon"].astype(float),
        "frac": data["frac"].astype(float),
        "zone": data["zone"].astype(int),
    }


def _grid_weights(frac, zone=None, zone_id=0, lats=None):
    """Combine polygon area fractions with optional subbasin filtering and latitude cosine weighting."""
    w = frac.copy() if (zone_id == 0 or zone is None) else np.where(zone == zone_id, frac, 0.0)
    if lats is not None:
        w = w * np.cos(np.deg2rad(lats))[:, None]
    total = np.nansum(w)
    if total <= 0:
        raise ValueError(f"Zero mask weight for zone_id={zone_id}")
    return w / total


def area_weighted_series(field, frac, zone=None, zone_id=0, lats=None):
    """Compute area-weighted mean time series for the basin (0) or subbasin."""
    weights = _grid_weights(frac, zone=zone, zone_id=zone_id, lats=lats)
    return np.nansum(field * weights[None, :, :], axis=(1, 2))


def build_tributary_masks(hybas_l6_prefix, lats, lons, frac):
    """Build fractional and latitude-cosine-weighted HydroBASINS Level-6 tributary masks and areas."""
    sf = shapefile.Reader(str(hybas_l6_prefix))
    fields = [f[0] for f in sf.fields[1:]]
    i_id = fields.index("HYBAS_ID")
    i_down = fields.index("NEXT_DOWN")
    i_sub = fields.index("SUB_AREA")
    recs = sf.records()
    shapes = sf.shapes()

    ids = np.array([r[i_id] for r in recs])
    downs = np.array([r[i_down] for r in recs])
    sub_areas = np.array([float(r[i_sub]) for r in recs])

    candidates = {
        idx: shape(shp.__geo_interface__)
        for idx, shp in enumerate(shapes)
        if shp.bbox[2] >= 89.0
        and shp.bbox[0] <= 123.0
        and shp.bbox[3] >= 23.5
        and shp.bbox[1] <= 36.5
    }

    trib_sets = {}
    trib_areas_km2 = {}
    trib_geoms = {}
    for name, (lon0, lat0) in TRIBUTARY_OUTLETS.items():
        pt = Point(lon0, lat0)
        outlet = next(i for i, poly in candidates.items() if poly.contains(pt))
        basin_ids = {recs[outlet][i_id]}
        while True:
            upstream = np.isin(downs, list(basin_ids)) & ~np.isin(ids, list(basin_ids))
            if not upstream.any():
                break
            basin_ids.update(ids[upstream].tolist())
        trib_sets[name] = basin_ids
        in_set = np.isin(ids, list(basin_ids))
        trib_areas_km2[name] = round(float(sub_areas[in_set].sum()), 1)
        trib_geoms[name] = unary_union([
            poly for i, poly in candidates.items() if recs[i][i_id] in basin_ids
        ])

    dlat = float(abs(lats[1] - lats[0])) if len(lats) > 1 else 0.5
    dlon = float(abs(lons[1] - lons[0])) if len(lons) > 1 else 0.5
    trib_names = list(TRIBUTARY_OUTLETS.keys())
    raw_weights = {name: np.zeros(frac.shape, dtype=float) for name in trib_names}

    for iy, ix in np.argwhere(frac > 0):
        cell_box = box(
            lons[ix] - 0.5 * dlon,
            lats[iy] - 0.5 * dlat,
            lons[ix] + 0.5 * dlon,
            lats[iy] + 0.5 * dlat,
        )
        c_area = cell_box.area
        cos_w = float(np.cos(np.deg2rad(lats[iy])))
        for name in trib_names:
            geom = trib_geoms[name]
            if not geom.intersects(cell_box):
                continue
            ov = cell_box.intersection(geom).area / c_area
            if ov > 1e-4:
                raw_weights[name][iy, ix] = ov * cos_w

    stack = np.stack([raw_weights[nm] for nm in trib_names], axis=0)
    max_ov = np.max(stack, axis=0)
    zid = np.where(max_ov > 0.0, np.argmax(stack, axis=0), -1)

    masks = {}
    for name in trib_names:
        w = raw_weights[name]
        masks[name] = w / w.sum()
    return masks, zid, trib_names, trib_areas_km2


def load_coastlines(geojson_path):
    """Parse Natural Earth GeoJSON lines for regional map plotting."""
    import json

    with Path(geojson_path).open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    segments = []
    for feat in payload.get("features", []):
        geom = feat.get("geometry", {})
        gtype = geom.get("type")
        coords = geom.get("coordinates", [])
        if gtype == "LineString":
            segments.append(np.asarray(coords, dtype=float))
        elif gtype == "MultiLineString":
            for part in coords:
                segments.append(np.asarray(part, dtype=float))
    return segments


def _pack_segments(segments):
    """Flatten variable-length polyline arrays with NaN row separators."""
    if not segments:
        return np.zeros((0, 2), dtype=np.float32)
    parts = []
    sep = np.array([[np.nan, np.nan]], dtype=np.float32)
    for seg in segments:
        arr = np.asarray(seg, dtype=np.float32)
        if len(arr) >= 2:
            parts.append(arr)
            parts.append(sep)
    return np.vstack(parts) if parts else np.zeros((0, 2), dtype=np.float32)


def _unpack_segments(packed):
    """Restore NaN-separated polyline segments from an array."""
    if len(packed) == 0:
        return []
    breaks = np.where(~np.isfinite(packed[:, 0]))[0]
    segs = []
    start = 0
    for idx in breaks:
        if idx > start + 1:
            segs.append(packed[start:idx].astype(float))
        start = idx + 1
    if start < len(packed) - 1:
        segs.append(packed[start:].astype(float))
    return segs


def _geom_to_rings(geom, simplify_tol=0.04):
    """Extract largest exterior polygon rings from a simplified Shapely geometry."""
    g = geom.simplify(simplify_tol, preserve_topology=True)
    rings = []
    if g.geom_type == "Polygon":
        rings.append(np.asarray(g.exterior.coords, dtype=float))
    elif g.geom_type == "MultiPolygon":
        for poly in sorted(g.geoms, key=lambda p: p.area, reverse=True):
            if poly.area >= 0.05:
                rings.append(np.asarray(poly.exterior.coords, dtype=float))
    return rings


def extract_basin_vectors(hybas_l6_prefix, coast_geojson, precip_grid=None, years=None):
    """Extract smooth HydroBASINS polygons, tributary catchments, and Asia coastlines."""
    sf = shapefile.Reader(str(hybas_l6_prefix))
    fields = [f[0] for f in sf.fields[1:]]
    i_id = fields.index("HYBAS_ID")
    i_down = fields.index("NEXT_DOWN")
    recs = sf.records()
    shapes = sf.shapes()

    ids = np.array([r[i_id] for r in recs])
    downs = np.array([r[i_down] for r in recs])
    bbox_idx = [
        i for i, shp in enumerate(shapes)
        if shp.bbox[2] >= 89.0 and shp.bbox[0] <= 123.0 and shp.bbox[3] >= 23.5 and shp.bbox[1] <= 36.5
    ]
    cand = {i: shape(shapes[i].__geo_interface__) for i in bbox_idx}

    def _trace(pt):
        hit = next((i for i, poly in cand.items() if poly.contains(Point(*pt))), None)
        if hit is None:
            return set()
        inc = {recs[hit][i_id]}
        while True:
            up = np.isin(downs, list(inc)) & ~np.isin(ids, list(inc))
            if not up.any():
                break
            inc.update(ids[up].tolist())
        return inc

    def _union_ids(id_set):
        geoms = [cand[i] for i in bbox_idx if recs[i][i_id] in id_set]
        return unary_union(geoms)

    tr_yi = _trace((111.3, 30.7))
    tr_dt = _trace((117.2, 30.7))
    tr_zj = _trace((119.5, 32.2))

    vectors = {
        "yrb": _geom_to_rings(_union_ids(tr_zj)),
        "upper": _geom_to_rings(_union_ids(tr_yi)),
        "middle": _geom_to_rings(_union_ids(tr_dt - tr_yi)),
        "lower": _geom_to_rings(_union_ids(tr_zj - tr_dt)),
    }
    for name, pt in TRIBUTARY_OUTLETS.items():
        vectors[f"trib_{name}"] = _geom_to_rings(_union_ids(_trace(pt)), simplify_tol=0.05)

    # Trim coastline segments to Greater East Asia domain for compact repository storage
    raw_segs = load_coastlines(coast_geojson)
    kept = []
    for seg in raw_segs:
        if (
            np.nanmax(seg[:, 0]) >= 60.0 and np.nanmin(seg[:, 0]) <= 175.0
            and np.nanmax(seg[:, 1]) >= 12.0 and np.nanmin(seg[:, 1]) <= 64.0
        ):
            kept.append(seg)
    vectors["coast"] = kept
    if precip_grid is not None and years is not None:
        nyr = len(np.unique(years))
        vectors["pr_annual_clim"] = np.nansum(precip_grid, axis=0) / float(nyr)
    return vectors


def save_basin_vectors(out_path, vectors):
    """Save packed vector rings, coastlines, and climatology arrays into NPZ."""
    payload = {}
    for k, val in vectors.items():
        if k == "pr_annual_clim":
            payload[k] = np.asarray(val, dtype=np.float32)
        else:
            payload[k] = _pack_segments(val)
    np.savez_compressed(out_path, **payload)


def load_basin_vectors(npz_path):
    """Load unpacked smooth basin rings, coastlines, and precipitation climatology."""
    data = np.load(npz_path, allow_pickle=False)
    out = {}
    for k in data.files:
        if k == "pr_annual_clim":
            out[k] = data[k].astype(float)
        else:
            out[k] = _unpack_segments(data[k])
    return out
