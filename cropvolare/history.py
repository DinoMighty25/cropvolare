"""
Per-field flight history and change detection.

Each analyzed flight appends one compact record to history/<field>.jsonl, so a
report can compare against the previous flight of the same field: overall NDVI
trend and - when GPS is present - which problem patches are new, worsening,
improving, or resolved. Plain JSON lines, no database.
"""

import json
import math
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_HISTORY_DIR = os.path.join(REPO, "history")

# a patch this flight is "the same" as one last flight if within this many m
PATCH_MATCH_RADIUS_M = 30.0
# |Δ mean NDVI| below this reads as "stable"
STABLE_DELTA = 0.03


def _path(field, history_dir):
    return os.path.join(history_dir, f"{field}.jsonl")


def record_from_result(result):
    """The compact history row distilled from an AnalysisResult."""
    dist = result.get("distribution", {})
    return {
        "flight_id": result.get("flight_id"),
        "date": result.get("date"),
        "scope": result.get("scope"),
        "mean_ndvi": dist.get("mean"),
        "pct_healthy": dist.get("pct_healthy"),
        "pct_stressed": dist.get("pct_stressed"),
        "pct_severe": dist.get("pct_severe"),
        "area_ha": result.get("area_ha"),
        "n_frames": result.get("n_frames"),
        "patches": [{"lat": p["lat"], "lon": p["lon"],
                     "area_ha": p["area_ha"], "mean_ndvi": p["mean_ndvi"]}
                    for p in result.get("patches", [])],
    }


def record_flight(field, result, history_dir=DEFAULT_HISTORY_DIR):
    """Record this flight; return the row. No-op-safe if field is None.

    Keyed by flight_id: re-processing the same flight folder overwrites its row
    (no duplicate accumulation) rather than appending.
    """
    if not field:
        return None
    os.makedirs(history_dir, exist_ok=True)
    row = record_from_result(result)
    fid = row.get("flight_id")
    rows = load_history(field, history_dir)
    if fid is not None:
        rows = [r for r in rows if r.get("flight_id") != fid]
    rows.append(row)
    with open(_path(field, history_dir), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return row


def load_history(field, history_dir=DEFAULT_HISTORY_DIR):
    """All recorded rows for a field, oldest first."""
    path = _path(field, history_dir)
    if not field or not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    return rows


def previous(field, exclude_flight=None, history_dir=DEFAULT_HISTORY_DIR):
    """The most recent prior record from a DIFFERENT flight than exclude_flight."""
    rows = load_history(field, history_dir)
    if exclude_flight is not None:
        rows = [r for r in rows if r.get("flight_id") != exclude_flight]
    return rows[-1] if rows else None


def _meters_between(a, b):
    """Rough planar distance (m) between two lat/lon points at this latitude."""
    m_lat = math.pi / 180.0 * 6_371_000.0
    m_lon = m_lat * math.cos(math.radians(a["lat"]))
    return math.hypot((a["lat"] - b["lat"]) * m_lat,
                      (a["lon"] - b["lon"]) * m_lon)


def compare(current, prior, radius_m=PATCH_MATCH_RADIUS_M):
    """Trend of `current` (AnalysisResult) vs a `prior` history row.

    Always reports the mean-NDVI trend. When both flights are GPS/field scope,
    also matches patches by nearest centroid within radius_m and classifies each
    as new / resolved / worsened / improved / persistent.
    """
    if not prior:
        return None
    cur_mean = current.get("distribution", {}).get("mean")
    prev_mean = prior.get("mean_ndvi")
    delta = (round(cur_mean - prev_mean, 4)
             if cur_mean is not None and prev_mean is not None else None)
    if delta is None:
        overall = "unknown"
    elif delta > STABLE_DELTA:
        overall = "improving"
    elif delta < -STABLE_DELTA:
        overall = "declining"
    else:
        overall = "stable"

    trend = {"prev_date": prior.get("date"), "mean_ndvi_delta": delta,
             "overall": overall, "new": [], "worsened": [], "improved": [],
             "resolved": [], "spatial": False}

    cur_patches = current.get("patches", [])
    prev_patches = prior.get("patches", [])
    if current.get("scope") != "field" or not prev_patches and not cur_patches:
        return trend
    trend["spatial"] = True

    matched_prev = set()
    for cp in cur_patches:
        best, best_d = None, radius_m
        for i, pp in enumerate(prev_patches):
            if i in matched_prev:
                continue
            d = _meters_between(cp, pp)
            if d <= best_d:
                best, best_d = i, d
        if best is None:
            trend["new"].append(cp)
        else:
            matched_prev.add(best)
            pp = prev_patches[best]
            entry = {"lat": cp["lat"], "lon": cp["lon"],
                     "mean_ndvi": cp["mean_ndvi"],
                     "prev_ndvi": pp["mean_ndvi"]}
            if cp["mean_ndvi"] < pp["mean_ndvi"] - STABLE_DELTA:
                trend["worsened"].append(entry)
            elif cp["mean_ndvi"] > pp["mean_ndvi"] + STABLE_DELTA:
                trend["improved"].append(entry)
    # prior patches with no current match cleared up
    for i, pp in enumerate(prev_patches):
        if i not in matched_prev:
            trend["resolved"].append(pp)
    return trend
