from cropvolare import batch, field


def _fc(points):
    """Build a minimal FeatureCollection from (lon, lat, mean_ndvi) tuples."""
    features = []
    for lon, lat, ndvi in points:
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {"mean_ndvi": ndvi},
        })
    return {"type": "FeatureCollection", "features": features, "metadata": {}}


def test_build_grid_separates_far_points():
    # two points ~100 m apart at 20 m cells -> different cells
    fc = _fc([(-88.0, 40.0, 0.7), (-88.0, 40.001, 0.1)])
    grid = field.build_grid(fc, cell_meters=20.0)
    assert not grid["empty"]
    cells = field.classify_cells(grid)
    assert len(cells) == 2


def test_build_grid_merges_colocated_points():
    # same location, two readings -> one cell, averaged NDVI
    fc = _fc([(-88.0, 40.0, 0.8), (-88.0, 40.0, 0.4)])
    grid = field.build_grid(fc, cell_meters=20.0)
    cells = field.classify_cells(grid)
    assert len(cells) == 1
    assert abs(cells[0]["mean_ndvi"] - 0.6) < 1e-6
    assert cells[0]["photo_count"] == 2


def test_classify_cell_thresholds():
    fc = _fc([
        (-88.0000, 40.0000, 0.7),   # healthy
        (-88.0010, 40.0000, 0.4),   # stressed
        (-88.0020, 40.0000, 0.1),   # severe
    ])
    grid = field.build_grid(fc, cell_meters=20.0)
    cells = field.classify_cells(grid)
    statuses = sorted(c["status"] for c in cells)
    assert statuses == ["healthy", "severe", "stressed"]


def test_rank_problems_worst_first():
    fc = _fc([
        (-88.0000, 40.0000, 0.7),
        (-88.0010, 40.0000, 0.25),
        (-88.0020, 40.0000, 0.05),
    ])
    grid = field.build_grid(fc, cell_meters=20.0)
    cells = field.classify_cells(grid)
    problems = field.rank_problems(cells, top_n=5)
    assert len(problems) == 2                 # healthy one excluded
    assert problems[0]["mean_ndvi"] < problems[1]["mean_ndvi"]
    assert problems[0]["rank"] == 1


def test_rank_problems_respects_top_n():
    fc = _fc([(-88.0 - i * 0.001, 40.0, 0.1) for i in range(6)])
    grid = field.build_grid(fc, cell_meters=20.0)
    cells = field.classify_cells(grid)
    assert len(field.rank_problems(cells, top_n=3)) == 3


def test_summarize_percentages_sum_to_100():
    fc = _fc([
        (-88.0000, 40.0000, 0.7),
        (-88.0010, 40.0000, 0.4),
        (-88.0020, 40.0000, 0.1),
    ])
    grid = field.build_grid(fc, cell_meters=20.0)
    cells = field.classify_cells(grid)
    s = field.summarize(cells)
    assert s["n_cells"] == 3
    # allow for per-value 1-decimal rounding (3 cells -> up to ~0.15 drift)
    assert abs(s["pct_healthy"] + s["pct_stressed"] + s["pct_severe"] - 100.0) < 0.2


def test_empty_grid():
    fc = {"type": "FeatureCollection", "features": [], "metadata": {}}
    grid = field.build_grid(fc)
    assert grid["empty"] is True
    assert field.classify_cells(grid) == []
    assert field.summarize([])["n_cells"] == 0


def test_end_to_end_from_geotagged_dir(geotagged_dir):
    fc = batch.process_directory(str(geotagged_dir))
    grid = field.build_grid(fc, cell_meters=20.0)
    cells = field.classify_cells(grid)
    problems = field.rank_problems(cells)
    # the stressed corner should surface as a problem cell
    assert any(c["status"] in ("stressed", "severe") for c in cells)
    assert len(problems) >= 1
