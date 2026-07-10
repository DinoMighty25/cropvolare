"""
Field analysis engine - turns raw NDVI into farmer-facing findings.

Deterministic, offline, no ML: distribution statistics + spatial clustering of
low-NDVI regions into named problem patches + rule-based cause suggestions and a
health verdict. Produces one AnalysisResult dict that history.py and report.py
consume. Pure functions (no file I/O) so every rule is unit-testable.

Two fidelities:
  scope="field"    GPS present -> patches on the georeferenced grid (lat/lon, ha)
  scope="gallery"  no GPS      -> per-frame problem regions + aggregate stats

Honest by construction: single-camera NoIR NDVI is relative and approximate, so
findings are decision-support ("go check"), never diagnoses. Calibration state
travels in the result so the report can caveat it.
"""

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from .field import _cell_center_latlon

# NDVI class thresholds (shared with field.classify_cells)
HEALTHY = 0.5
STRESSED = 0.3   # below this = "stressed"; well below = "severe"
SEVERE = 0.15


# --------------------------------------------------------------------------
# distribution + verdict
# --------------------------------------------------------------------------

def ndvi_distribution(features, healthy=HEALTHY, stressed=STRESSED,
                      severe=SEVERE):
    """Aggregate NDVI stats over per-image features (each has 'mean_ndvi')."""
    vals = np.array([f["properties"]["mean_ndvi"] for f in features],
                    dtype=np.float64)
    if vals.size == 0:
        return {"n": 0, "mean": None, "median": None, "std": None,
                "pct_healthy": 0.0, "pct_moderate": 0.0,
                "pct_stressed": 0.0, "pct_severe": 0.0, "histogram": []}
    n = vals.size
    hist, _ = np.histogram(vals, bins=10, range=(-1.0, 1.0))
    return {
        "n": int(n),
        "mean": round(float(vals.mean()), 4),
        "median": round(float(np.median(vals)), 4),
        "std": round(float(vals.std()), 4),
        "pct_healthy": round(100.0 * np.count_nonzero(vals >= healthy) / n, 1),
        "pct_moderate": round(
            100.0 * np.count_nonzero((vals >= stressed) & (vals < healthy)) / n, 1),
        "pct_stressed": round(
            100.0 * np.count_nonzero((vals >= severe) & (vals < stressed)) / n, 1),
        "pct_severe": round(100.0 * np.count_nonzero(vals < severe) / n, 1),
        "histogram": hist.astype(int).tolist(),
    }


_VERDICT_LINES = {
    "strong": "Crop vigour looks strong and even across the field.",
    "fair": "Crop vigour is fair, with some areas worth a look.",
    "poor": "Crop vigour is low over much of the field.",
    "critical": "Very low vigour - inspect the field promptly.",
}


def health_verdict(dist):
    """Overall field health -> {level, score 0-100, line}. Deterministic rules.

    Score maps mean NDVI [0..0.7] to [0..100] and subtracts the % of severe
    area; the level is derived from the score so the two never disagree.
    """
    if not dist or dist["n"] == 0 or dist["mean"] is None:
        return {"level": "unknown", "score": 0,
                "line": "No usable imagery to assess."}
    score = int(max(0, min(100, (dist["mean"] / 0.7) * 100 - dist["pct_severe"])))
    if score >= 75:
        level = "strong"
    elif score >= 50:
        level = "fair"
    elif score >= 25:
        level = "poor"
    else:
        level = "critical"
    return {"level": level, "score": score, "line": _VERDICT_LINES[level]}


# --------------------------------------------------------------------------
# spatial clustering -> problem patches (GPS / field scope)
# --------------------------------------------------------------------------

def _connected_components(mask):
    """(n_labels, labels) for a binary mask; pure-numpy flood fill if no cv2."""
    if cv2 is not None:
        n, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=4)
        return n, labels
    labels = np.zeros(mask.shape, dtype=np.int32)
    nxt = 1
    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if mask[r, c] and labels[r, c] == 0:
                stack = [(r, c)]
                labels[r, c] = nxt
                while stack:
                    y, x = stack.pop()
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if (0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1]
                                and mask[ny, nx] and labels[ny, nx] == 0):
                            labels[ny, nx] = nxt
                            stack.append((ny, nx))
                nxt += 1
    return nxt, labels


def find_patches_field(grid, stressed=STRESSED, min_cells=2):
    """Cluster below-threshold cells into ranked problem patches (worst first).

    Each patch: centroid lat/lon, area_ha, mean_ndvi, n_cells, cell bbox, and a
    severity score (NDVI deficit x area) used for ranking.
    """
    if grid.get("empty"):
        return []
    mean = grid["mean_ndvi"]
    counts = grid["counts"]
    mask = (counts > 0) & (mean < stressed)
    if not mask.any():
        return []

    _, labels = _connected_components(mask)
    cell_ha = (grid["cell_meters"] ** 2) / 10000.0

    patches = []
    for lab in range(1, int(labels.max()) + 1):
        ys, xs = np.where(labels == lab)
        if ys.size < min_cells:
            continue
        cell_means = mean[ys, xs]
        patch_mean = float(cell_means.mean())
        cy, cx = float(ys.mean()), float(xs.mean())
        lat, lon = _cell_center_latlon(grid, cy, cx)
        deficit = max(0.0, stressed - patch_mean)
        patches.append({
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "n_cells": int(ys.size),
            "area_ha": round(ys.size * cell_ha, 4),
            "mean_ndvi": round(patch_mean, 4),
            "severity": round(deficit * ys.size * cell_ha, 5),
            "status": "severe" if patch_mean < SEVERE else "stressed",
            "bbox_rc": [int(ys.min()), int(xs.min()),
                        int(ys.max()), int(xs.max())],
        })
    patches.sort(key=lambda p: p["severity"], reverse=True)
    for i, p in enumerate(patches, start=1):
        p["rank"] = i
    return patches


# --------------------------------------------------------------------------
# per-image regions (no-GPS / gallery scope)
# --------------------------------------------------------------------------

def find_regions_image(ndvi, stressed=STRESSED, max_px=256, min_frac=0.02):
    """Summarize stressed regions within one frame's NDVI array.

    Returns {stressed_frac, largest_frac, n_regions}. Downscaled first so this
    is cheap even on full-res frames.
    """
    arr = ndvi
    if cv2 is not None and max(arr.shape[:2]) > max_px:
        scale = max_px / float(max(arr.shape[:2]))
        arr = cv2.resize(arr.astype(np.float32),
                         (int(arr.shape[1] * scale), int(arr.shape[0] * scale)),
                         interpolation=cv2.INTER_AREA)
    mask = arr < stressed
    total = mask.size
    stressed_frac = float(mask.sum()) / total
    _, labels = _connected_components(mask)
    largest, n_regions = 0, 0
    for lab in range(1, int(labels.max()) + 1):
        size = int(np.count_nonzero(labels == lab))
        if size / total >= min_frac:
            n_regions += 1
            largest = max(largest, size)
    return {
        "stressed_frac": round(stressed_frac, 3),
        "largest_frac": round(largest / total, 3),
        "n_regions": n_regions,
    }


# --------------------------------------------------------------------------
# cause heuristics
# --------------------------------------------------------------------------

def suggest_causes(patch, dist, grid=None):
    """Ordered 'possible causes to investigate' from spatial pattern heuristics.

    Framed as prompts, never diagnoses - single-camera NDVI can't identify a
    cause, only point you at where and what to check.
    """
    causes = []
    # whole-field low vigour dominates -> systemic, not a local patch
    if dist and dist.get("mean") is not None and dist["mean"] < STRESSED:
        causes.append("field-wide low vigour: water/nutrient deficiency, early "
                      "growth stage, or an off-calibration/low-light capture")

    bbox = patch.get("bbox_rc")
    if bbox:
        h = bbox[2] - bbox[0] + 1
        w = bbox[3] - bbox[1] + 1
        elong = max(h, w) / max(1, min(h, w))
        if elong >= 3:
            causes.append("linear/edge pattern: headland compaction, shading, or "
                          "a field-boundary/road effect")
        elif patch["n_cells"] <= 4:
            causes.append("small compact spot: possible pest/disease focus, "
                          "standing water, or a sprayer/seeder skip")
        else:
            causes.append("broad patch: drainage, irrigation coverage, or a soil "
                          "difference across this area")
    if patch.get("status") == "severe":
        causes.append("severe reading - prioritise a ground check here")
    return causes or ["inspect this area on the ground"]


# --------------------------------------------------------------------------
# top-level
# --------------------------------------------------------------------------

def analyze(feature_collection, grid=None, cells=None, field=None,
            area_ha=None, date=None, flight_id=None):
    """Build the AnalysisResult that history + report consume."""
    meta = feature_collection.get("metadata", {})
    features = feature_collection.get("features", [])
    params = meta.get("params", {})

    dist = ndvi_distribution(features)
    verdict = health_verdict(dist)

    scope = "field" if (grid and not grid.get("empty")) else "gallery"
    patches = []
    if scope == "field":
        patches = find_patches_field(grid)
        for p in patches:
            p["causes"] = suggest_causes(p, dist, grid)

    # worst frames (works in both scopes; the gallery report's "go look" list)
    worst = sorted(features,
                   key=lambda f: f["properties"]["mean_ndvi"])[:10]
    worst_frames = [{
        "filename": f["properties"]["filename"],
        "mean_ndvi": f["properties"]["mean_ndvi"],
        "sharpness": f["properties"].get("sharpness"),
        "brightness": f["properties"].get("brightness"),
        "source_path": f["properties"].get("source_path"),
        "overlay_png": f["properties"].get("overlay_png"),
    } for f in worst]

    return {
        "field": field,
        "flight_id": flight_id,
        "date": date or meta.get("flight_date"),
        "scope": scope,
        "area_ha": area_ha,
        "n_frames": meta.get("n_images", len(features)),
        "n_unreadable": meta.get("n_unreadable", 0),
        "n_filtered": meta.get("n_filtered", 0),
        "distribution": dist,
        "verdict": verdict,
        "patches": patches,
        "worst_frames": worst_frames,
        "calibration": {
            "flatfield": params.get("flatfield", False),
            "leakage_k": params.get("leakage_k"),
            "gamma": params.get("gamma"),
            "min_sharpness": params.get("min_sharpness", 0),
        },
    }
