"""
Field aggregation: turn scattered geotagged NDVI photos into a field grid that
answers "which areas need help?".

Photo locations are projected to a local meters frame (equirectangular about the
field centroid - accurate over a single sub-km field, and avoids a pyproj/GDAL
dependency), binned into square cells, and each cell is classified
healthy / stressed / severe by its average NDVI. Pure numpy/Python, no I/O.
"""

import math

import numpy as np

EARTH_RADIUS_M = 6_371_000.0


def _tagged_points(feature_collection):
    """Pull (lon, lat, mean_ndvi) for every geotagged feature."""
    pts = []
    for f in feature_collection.get("features", []):
        geom = f.get("geometry")
        if not geom:
            continue
        lon, lat = geom["coordinates"]
        pts.append((lon, lat, f["properties"]["mean_ndvi"]))
    return pts


def build_grid(feature_collection, cell_meters=20.0):
    """Bin geotagged photos into a square grid of cell_meters cells.

    Returns a grid dict with the local projection origin, cell size, grid shape,
    geographic bounds, and per-cell {sum, count, mean_ndvi} arrays. Returns a
    grid with empty=True when there are no geotagged photos.
    """
    pts = _tagged_points(feature_collection)
    if not pts:
        return {"empty": True, "cell_meters": cell_meters}

    lons = np.array([p[0] for p in pts])
    lats = np.array([p[1] for p in pts])
    ndvis = np.array([p[2] for p in pts])

    lat0 = float(lats.mean())
    lon0 = float(lons.mean())
    m_per_deg_lat = math.pi / 180.0 * EARTH_RADIUS_M
    m_per_deg_lon = m_per_deg_lat * math.cos(math.radians(lat0))

    # local meters relative to centroid
    xs = (lons - lon0) * m_per_deg_lon
    ys = (lats - lat0) * m_per_deg_lat

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    ncols = max(1, int(math.floor((x_max - x_min) / cell_meters)) + 1)
    nrows = max(1, int(math.floor((y_max - y_min) / cell_meters)) + 1)

    sums = np.zeros((nrows, ncols))
    counts = np.zeros((nrows, ncols), dtype=int)

    col_idx = np.clip(((xs - x_min) / cell_meters).astype(int), 0, ncols - 1)
    # row 0 = north (max y) so the raster reads top-down like a map
    row_idx = np.clip(((y_max - ys) / cell_meters).astype(int), 0, nrows - 1)

    for r, c, v in zip(row_idx, col_idx, ndvis):
        sums[r, c] += v
        counts[r, c] += 1

    with np.errstate(invalid="ignore"):
        mean = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)

    return {
        "empty": False,
        "cell_meters": cell_meters,
        "nrows": nrows,
        "ncols": ncols,
        "lat0": lat0,
        "lon0": lon0,
        "m_per_deg_lat": m_per_deg_lat,
        "m_per_deg_lon": m_per_deg_lon,
        "x_min": float(x_min),
        "y_max": float(y_max),
        "mean_ndvi": mean,
        "counts": counts,
        "bounds": _grid_bounds(lat0, lon0, m_per_deg_lat, m_per_deg_lon,
                               x_min, x_max, y_min, y_max),
    }


def _grid_bounds(lat0, lon0, mlat, mlon, x_min, x_max, y_min, y_max):
    """Geographic bounds [[south, west], [north, east]] for a Leaflet overlay."""
    south = lat0 + y_min / mlat
    north = lat0 + y_max / mlat
    west = lon0 + x_min / mlon
    east = lon0 + x_max / mlon
    return [[south, west], [north, east]]


def _cell_center_latlon(grid, row, col):
    """Geographic center of a grid cell."""
    x = grid["x_min"] + (col + 0.5) * grid["cell_meters"]
    # row 0 is north (y_max), rows increase southward
    y = grid["y_max"] - (row + 0.5) * grid["cell_meters"]
    lat = grid["lat0"] + y / grid["m_per_deg_lat"]
    lon = grid["lon0"] + x / grid["m_per_deg_lon"]
    return lat, lon


def classify_cells(grid, healthy=0.5, stressed=0.3):
    """Tag every non-empty cell healthy / stressed / severe by mean NDVI.

    Bands mirror classify_zones in ndvi.py (>=healthy, >=stressed), with the
    bottom band relabeled 'severe' for report emphasis.
    """
    if grid.get("empty"):
        return []

    cells = []
    mean = grid["mean_ndvi"]
    counts = grid["counts"]
    for r in range(grid["nrows"]):
        for c in range(grid["ncols"]):
            if counts[r, c] == 0:
                continue
            value = float(mean[r, c])
            if value >= healthy:
                status = "healthy"
            elif value >= stressed:
                status = "stressed"
            else:
                status = "severe"
            lat, lon = _cell_center_latlon(grid, r, c)
            cells.append({
                "row": r,
                "col": c,
                "lat": round(lat, 6),
                "lon": round(lon, 6),
                "mean_ndvi": round(value, 4),
                "photo_count": int(counts[r, c]),
                "status": status,
            })
    return cells


def rank_problems(cells, top_n=5):
    """Worst stressed/severe cells first (lowest NDVI), capped at top_n."""
    problems = [c for c in cells if c["status"] in ("stressed", "severe")]
    problems.sort(key=lambda c: c["mean_ndvi"])
    ranked = problems[:top_n]
    for i, cell in enumerate(ranked, start=1):
        cell["rank"] = i
    return ranked


def summarize(cells):
    """Field rollup: % of cells healthy / stressed / severe, overall mean NDVI."""
    total = len(cells)
    if total == 0:
        return {
            "n_cells": 0, "mean_ndvi": None,
            "pct_healthy": 0.0, "pct_stressed": 0.0, "pct_severe": 0.0,
            "n_problem_cells": 0,
        }
    counts = {"healthy": 0, "stressed": 0, "severe": 0}
    for c in cells:
        counts[c["status"]] += 1
    mean = float(np.mean([c["mean_ndvi"] for c in cells]))
    return {
        "n_cells": total,
        "mean_ndvi": round(mean, 4),
        "pct_healthy": round(100.0 * counts["healthy"] / total, 1),
        "pct_stressed": round(100.0 * counts["stressed"] / total, 1),
        "pct_severe": round(100.0 * counts["severe"] / total, 1),
        "n_problem_cells": counts["stressed"] + counts["severe"],
    }
