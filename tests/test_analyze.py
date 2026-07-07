"""Analysis-engine tests - pure math on synthetic grids/features."""

import numpy as np

from cropvolare import analyze, field


def _fc(mean_values):
    """FeatureCollection with given per-image mean NDVI values (no GPS)."""
    feats = [{
        "type": "Feature", "geometry": None,
        "properties": {"mean_ndvi": v, "filename": f"f{i:03d}.jpg",
                       "sharpness": 100.0},
    } for i, v in enumerate(mean_values)]
    return {"type": "FeatureCollection", "features": feats,
            "metadata": {"n_images": len(feats), "params": {"leakage_k": 2.0}}}


def _grid_from_points(points):
    """points = (lon, lat, mean_ndvi) -> a real field grid via field.build_grid."""
    feats = [{"type": "Feature",
              "geometry": {"type": "Point", "coordinates": [lon, lat]},
              "properties": {"mean_ndvi": v}} for lon, lat, v in points]
    fc = {"type": "FeatureCollection", "features": feats, "metadata": {}}
    return field.build_grid(fc, cell_meters=20.0)


# --- distribution + verdict -------------------------------------------------

def test_distribution_percentages():
    dist = analyze.ndvi_distribution(_fc([0.7, 0.6, 0.4, 0.1])["features"])
    assert dist["n"] == 4
    assert dist["pct_healthy"] == 50.0        # 0.7, 0.6
    assert dist["pct_severe"] == 25.0         # 0.1 < 0.15
    assert len(dist["histogram"]) == 10


def test_verdict_levels():
    strong = analyze.health_verdict(analyze.ndvi_distribution(
        _fc([0.6, 0.65, 0.7])["features"]))
    assert strong["level"] == "strong" and strong["score"] > 70
    poor = analyze.health_verdict(analyze.ndvi_distribution(
        _fc([0.25, 0.2, 0.28])["features"]))
    assert poor["level"] in ("poor", "critical")


def test_verdict_empty():
    assert analyze.health_verdict(analyze.ndvi_distribution([]))["level"] == "unknown"


# --- field patch clustering -------------------------------------------------

def test_single_patch_from_adjacent_low_cells():
    # a healthy field with a 2x2 block of low-NDVI cells in one corner
    pts = []
    for i in range(6):
        for j in range(6):
            lon = -88.0 + j * 0.00025      # ~20 m spacing
            lat = 40.0 + i * 0.00018
            v = 0.1 if (i < 2 and j < 2) else 0.7
            pts.append((lon, lat, v))
    grid = _grid_from_points(pts)
    patches = analyze.find_patches_field(grid, stressed=0.3, min_cells=2)
    assert len(patches) == 1
    p = patches[0]
    assert p["n_cells"] == 4
    assert p["mean_ndvi"] < 0.3
    assert p["area_ha"] > 0
    assert p["rank"] == 1
    assert p["status"] == "severe"


def test_two_patches_ranked_by_severity():
    pts = []
    for i in range(7):
        for j in range(7):
            lon, lat = -88.0 + j * 0.00025, 40.0 + i * 0.00018
            v = 0.7
            if i < 2 and j < 2:        # small severe cluster
                v = 0.05
            elif i >= 4 and j >= 3:    # bigger moderate-stressed cluster
                v = 0.25
            pts.append((lon, lat, v))
    grid = _grid_from_points(pts)
    patches = analyze.find_patches_field(grid, stressed=0.3, min_cells=2)
    assert len(patches) == 2
    # ranked by severity (integrated NDVI deficit = deficit x area), descending
    assert patches[0]["severity"] >= patches[1]["severity"]
    assert patches[0]["rank"] == 1 and patches[1]["rank"] == 2


def test_no_patches_when_healthy():
    pts = [(-88.0 + j * 0.00025, 40.0 + i * 0.00018, 0.7)
           for i in range(4) for j in range(4)]
    assert analyze.find_patches_field(_grid_from_points(pts)) == []


def test_min_cells_filters_singletons():
    pts = []
    for i in range(5):
        for j in range(5):
            lon, lat = -88.0 + j * 0.00025, 40.0 + i * 0.00018
            v = 0.1 if (i == 2 and j == 2) else 0.7   # lone low cell
            pts.append((lon, lat, v))
    grid = _grid_from_points(pts)
    assert analyze.find_patches_field(grid, min_cells=2) == []


# --- per-image regions (no-GPS) --------------------------------------------

def test_image_region_fraction():
    ndvi = np.full((100, 100), 0.6)
    ndvi[:30, :] = 0.1                     # 30% stressed band
    r = analyze.find_regions_image(ndvi, stressed=0.3)
    assert abs(r["stressed_frac"] - 0.30) < 0.05
    assert r["n_regions"] == 1
    assert r["largest_frac"] > 0.2


# --- causes -----------------------------------------------------------------

def test_causes_linear_vs_compact():
    dist = {"mean": 0.6}
    linear = {"bbox_rc": [0, 0, 0, 8], "n_cells": 9, "status": "stressed"}
    compact = {"bbox_rc": [0, 0, 1, 1], "n_cells": 3, "status": "stressed"}
    assert any("linear" in c for c in analyze.suggest_causes(linear, dist))
    assert any("compact" in c for c in analyze.suggest_causes(compact, dist))


def test_causes_flag_fieldwide_low():
    causes = analyze.suggest_causes(
        {"bbox_rc": [0, 0, 2, 2], "n_cells": 9, "status": "stressed"},
        {"mean": 0.2})
    assert any("field-wide" in c for c in causes)


# --- top-level analyze() ----------------------------------------------------

def test_analyze_gallery_scope():
    res = analyze.analyze(_fc([0.7, 0.5, 0.2, 0.1]), grid=None, field="yard")
    assert res["scope"] == "gallery"
    assert res["patches"] == []
    assert res["field"] == "yard"
    assert len(res["worst_frames"]) == 4
    assert res["worst_frames"][0]["mean_ndvi"] == 0.1   # worst first
    assert res["calibration"]["leakage_k"] == 2.0


def test_analyze_field_scope_has_patches():
    pts = [(-88.0 + j * 0.00025, 40.0 + i * 0.00018,
            0.1 if (i < 2 and j < 2) else 0.7)
           for i in range(6) for j in range(6)]
    fc = {"type": "FeatureCollection",
          "features": [{"geometry": {"coordinates": [lon, lat]},
                        "properties": {"mean_ndvi": v, "filename": "x.jpg",
                                       "sharpness": 50}}
                       for lon, lat, v in pts],
          "metadata": {"n_images": len(pts), "params": {}}}
    grid = _grid_from_points(pts)
    res = analyze.analyze(fc, grid=grid, field="north40")
    assert res["scope"] == "field"
    assert len(res["patches"]) == 1
    assert res["patches"][0]["causes"]
