"""
Twin-flight repeatability: how much does this system disagree with itself?

Fly the same pattern twice, an hour apart, on the same day. The crop cannot
change in an hour, so every difference between the two maps is measurement
error - optics, exposure, vibration, stitching, illumination drift. That number
is the error bar on every claim the project makes. Without it, a patch reading
0.15 below its surroundings might be real stress or might be noise, and there is
no way to tell.

Different from history.compare(), which asks "did this field change since last
week?" and works at the patch level assuming real change happened. This asks the
opposite: these two maps SHOULD be identical, so quantify how far off they are,
per grid cell.

The subtlety this module exists for: field.build_grid() derives its origin from
the data it is given, so two flights gridded separately land on two different
grids and their cells do not line up. Cells must be compared on ONE shared grid
built from both flights together.

Pure numpy, no I/O - the CLI lives in scripts/compare_flights.py.
"""

import numpy as np

from . import field

# Below this, the free-tier radiometric setup (locked preset + grey card + PIFs)
# is doing its job and reflectance panels stay unbought. Chosen to sit well under
# the NDVI differences that matter agronomically: healthy/stressed sits around
# 0.5/0.3, so 0.05 is comfortably finer than the distinctions being drawn.
DEFAULT_THRESHOLD = 0.05

# A cell holding a single photo is one frame's noise, not a measurement of that
# patch. Requiring 2+ in BOTH flights keeps thin edge-of-coverage cells from
# dominating the error statistic.
DEFAULT_MIN_COUNT = 2


def merge_collections(fc_a, fc_b):
    """One feature collection spanning both flights, for shared-grid geometry."""
    return {
        "type": "FeatureCollection",
        "features": list(fc_a.get("features", [])) + list(fc_b.get("features", [])),
    }


def shared_grid(fc_a, fc_b, cell_meters=20.0):
    """Grid geometry covering both flights. Raises ValueError if either is empty."""
    for name, fc in (("A", fc_a), ("B", fc_b)):
        if not field._tagged_points(fc):
            raise ValueError(
                f"flight {name} has no geotagged photos - a repeatability test "
                f"needs GPS on both flights (check --gps-port, or run tag_gps.py)")
    grid = field.build_grid(merge_collections(fc_a, fc_b), cell_meters=cell_meters)
    if grid.get("empty"):
        raise ValueError("no geotagged photos in either flight")
    return grid


def bin_onto(grid, fc):
    """Bin one flight's photos onto an EXISTING grid's geometry.

    Mirrors the index arithmetic in field.build_grid so cells correspond exactly.
    Returns (mean_ndvi, counts) with NaN where this flight has no coverage.
    """
    pts = field._tagged_points(fc)
    nrows, ncols = grid["nrows"], grid["ncols"]
    sums = np.zeros((nrows, ncols))
    counts = np.zeros((nrows, ncols), dtype=int)

    if pts:
        lons = np.array([p[0] for p in pts], dtype=float)
        lats = np.array([p[1] for p in pts], dtype=float)
        ndvis = np.array([p[2] for p in pts], dtype=float)

        xs = (lons - grid["lon0"]) * grid["m_per_deg_lon"]
        ys = (lats - grid["lat0"]) * grid["m_per_deg_lat"]
        cell = grid["cell_meters"]

        col_idx = np.clip(((xs - grid["x_min"]) / cell).astype(int), 0, ncols - 1)
        row_idx = np.clip(((grid["y_max"] - ys) / cell).astype(int), 0, nrows - 1)

        for r, c, v in zip(row_idx, col_idx, ndvis):
            sums[r, c] += v
            counts[r, c] += 1

    with np.errstate(invalid="ignore"):
        mean = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    return mean, counts


def compare(fc_a, fc_b, cell_meters=20.0, min_count=DEFAULT_MIN_COUNT,
            threshold=DEFAULT_THRESHOLD):
    """Per-cell repeatability statistics between two flights of the same field.

    Returns a dict with the headline error (mean_abs_delta), its spread, the
    systematic component (bias), and the coverage the number rests on.
    """
    grid = shared_grid(fc_a, fc_b, cell_meters=cell_meters)
    mean_a, count_a = bin_onto(grid, fc_a)
    mean_b, count_b = bin_onto(grid, fc_b)

    both = (count_a >= min_count) & (count_b >= min_count)
    n_compared = int(both.sum())

    result = {
        "cell_meters": cell_meters,
        "min_count": min_count,
        "threshold": threshold,
        "n_cells_grid": int(grid["nrows"] * grid["ncols"]),
        "n_cells_a": int((count_a > 0).sum()),
        "n_cells_b": int((count_b > 0).sum()),
        "n_cells_compared": n_compared,
        "n_frames_a": int(count_a.sum()),
        "n_frames_b": int(count_b.sum()),
    }

    if n_compared == 0:
        result.update({
            "ok": False,
            "reason": ("no cell has enough photos in BOTH flights - the two "
                       "flights may not overlap, or coverage is too thin "
                       "(try a larger --cell-meters or --min-count 1)"),
            "mean_abs_delta": None,
        })
        return result

    a = mean_a[both]
    b = mean_b[both]
    delta = b - a

    # Overlap fraction: a headline error computed from three cells out of two
    # hundred is not evidence, so surface how much of the field it covers.
    union = int(((count_a > 0) | (count_b > 0)).sum())

    result.update({
        "ok": True,
        "reason": None,
        "mean_ndvi_a": round(float(a.mean()), 4),
        "mean_ndvi_b": round(float(b.mean()), 4),
        "mean_abs_delta": round(float(np.abs(delta).mean()), 4),
        "rms_delta": round(float(np.sqrt((delta ** 2).mean())), 4),
        "bias": round(float(delta.mean()), 4),
        "p95_abs_delta": round(float(np.percentile(np.abs(delta), 95)), 4),
        "max_abs_delta": round(float(np.abs(delta).max()), 4),
        "overlap_fraction": round(n_compared / union, 3) if union else 0.0,
        "passes": bool(np.abs(delta).mean() <= threshold),
    })
    return result


def verdict(result):
    """Plain-language reading of a compare() result, for the field notebook."""
    if not result.get("ok"):
        return f"INCONCLUSIVE - {result.get('reason')}"

    err = result["mean_abs_delta"]
    thr = result["threshold"]
    bias = result["bias"]

    if result["passes"]:
        line = (f"PASS - the system repeats to {err:.3f} NDVI. Differences "
                f"larger than about {max(err * 2, thr):.2f} between areas can be "
                f"treated as real. Free-tier radiometrics are sufficient; "
                f"reflectance panels stay unbought.")
    else:
        line = (f"FAIL - {err:.3f} NDVI disagreement exceeds the {thr:.2f} "
                f"target. Check the cheap causes FIRST, in order: same exposure "
                f"preset on both flights, sky condition steady between them, "
                f"sharpness filter enabled, both flights the same pattern. Only "
                f"after those, build scrap reflectance panels and re-run.")

    # A large bias with a small spread is a different disease from scattered
    # noise: it points at illumination change between flights, which panels or a
    # downwelling sensor fix. Scattered noise points at vibration or focus.
    if abs(bias) > 0.6 * err and abs(bias) > 0.01:
        line += (f" Note: the error is mostly a uniform shift ({bias:+.3f}), not "
                 f"scatter - that is illumination drift between the flights "
                 f"rather than per-frame noise.")

    if result.get("overlap_fraction", 1.0) < 0.5:
        line += (f" Caution: only {result['overlap_fraction']:.0%} of covered "
                 f"cells appear in both flights, so this rests on partial "
                 f"overlap.")

    return line
