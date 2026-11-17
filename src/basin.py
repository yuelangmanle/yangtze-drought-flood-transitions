"""Spatial basin masks and area-weighted aggregation for the Yangtze River Basin."""

from pathlib import Path
import numpy as np
import shapefile
from shapely.geometry import Point, shape


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


def area_weighted_series(field, frac, zone=None, zone_id=0):
    """Compute area-weighted mean time series for the basin (0) or subbasin."""
    if zone_id == 0 or zone is None:
        weights = frac.copy()
    else:
        weights = np.where(zone == zone_id, frac, 0.0)
    total = np.nansum(weights)
    if total <= 0:
        raise ValueError(f"Zero mask weight for zone_id={zone_id}")
    weights = weights / total
    return np.nansum(field * weights[None, :, :], axis=(1, 2))


def build_tributary_masks(hybas_l6_prefix, lats, lons, frac):
    """Trace HydroBASINS Level-6 upstream catchments for the seven major tributaries."""
    sf = shapefile.Reader(str(hybas_l6_prefix))
    fields = [f[0] for f in sf.fields[1:]]
    i_id = fields.index("HYBAS_ID")
    i_down = fields.index("NEXT_DOWN")
    recs = sf.records()
    shapes = sf.shapes()

    ids = np.array([r[i_id] for r in recs])
    downs = np.array([r[i_down] for r in recs])

    candidates = {
        idx: shape(shp.__geo_interface__)
        for idx, shp in enumerate(shapes)
        if shp.bbox[2] >= 89.0
        and shp.bbox[0] <= 123.0
        and shp.bbox[3] >= 23.5
        and shp.bbox[1] <= 36.5
    }

    trib_sets = {}
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

    all_ids = set().union(*trib_sets.values())
    polys = [
        (recs[i][i_id], poly)
        for i, poly in candidates.items()
        if recs[i][i_id] in all_ids
    ]

    trib_names = list(TRIBUTARY_OUTLETS.keys())
    zid = np.full(frac.shape, -1, dtype=int)
    for iy, ix in np.argwhere(frac > 0):
        cell = Point(lons[ix], lats[iy]).buffer(0.01)
        hits = {hid for hid, poly in polys if poly.intersects(cell)}
        for k, name in enumerate(trib_names):
            if hits & trib_sets[name]:
                zid[iy, ix] = k
                break

    masks = {}
    for k, name in enumerate(trib_names):
        sel = zid == k
        w = np.where(sel, 1.0, 0.0)
        masks[name] = w / w.sum()
    return masks, zid, trib_names


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
