"""Tests for twin-flight repeatability.

The property that matters most: two flights must be binned onto ONE shared grid.
field.build_grid() derives its origin from whatever data it is handed, so
gridding each flight separately produces misaligned cells and a meaningless
comparison. Several tests below exist specifically to catch that regression.
"""

import math

import pytest

from cropvolare import field, repeat

LAT0, LON0 = 44.5000, -93.2000

# Point spacing, in degrees, for placing photos in known cells. Deliberately
# 30 m against the default 20 m cell: spacing points at exactly the cell size put
# them on cell boundaries, where sub-metre differences between this arithmetic
# and field.py's decided which cell a point fell into (rows collapsed into one).
_SPACING_M = 30.0
_M_PER_DEG_LAT = math.pi / 180.0 * field.EARTH_RADIUS_M
_M_PER_DEG_LON = _M_PER_DEG_LAT * math.cos(math.radians(LAT0))
DLAT = _SPACING_M / _M_PER_DEG_LAT
DLON = _SPACING_M / _M_PER_DEG_LON


def fc(points):
    """Build a feature collection from (lat, lon, ndvi) triples."""
    return {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature",
             "geometry": {"type": "Point", "coordinates": [lon, lat]},
             "properties": {"mean_ndvi": ndvi}}
            for lat, lon, ndvi in points
        ],
    }


def grid_of(n=3, ndvi=0.6, offset=0.0, repeats=2):
    """An n x n lattice of photos, each cell holding `repeats` frames."""
    pts = []
    for r in range(n):
        for c in range(n):
            for _ in range(repeats):
                pts.append((LAT0 + r * DLAT, LON0 + c * DLON, ndvi + offset))
    return fc(pts)


# --------------------------------------------------------------------------
# identical flights
# --------------------------------------------------------------------------

def test_identical_flights_have_zero_error():
    a = grid_of()
    result = repeat.compare(a, grid_of())
    assert result["ok"]
    assert result["mean_abs_delta"] == pytest.approx(0.0)
    assert result["bias"] == pytest.approx(0.0)
    assert result["passes"]


def test_identical_flights_verdict_passes():
    result = repeat.compare(grid_of(), grid_of())
    assert repeat.verdict(result).startswith("PASS")


# --------------------------------------------------------------------------
# known offsets
# --------------------------------------------------------------------------

def test_uniform_offset_shows_up_as_bias():
    result = repeat.compare(grid_of(ndvi=0.6), grid_of(ndvi=0.6, offset=0.10))
    assert result["mean_abs_delta"] == pytest.approx(0.10, abs=1e-6)
    assert result["bias"] == pytest.approx(0.10, abs=1e-6)
    assert not result["passes"]


def test_uniform_offset_is_diagnosed_as_illumination_drift():
    """A pure shift means the light changed; scatter would mean vibration/focus."""
    result = repeat.compare(grid_of(ndvi=0.6), grid_of(ndvi=0.6, offset=0.10))
    assert "illumination drift" in repeat.verdict(result)


def test_small_offset_passes_the_threshold():
    result = repeat.compare(grid_of(ndvi=0.6), grid_of(ndvi=0.6, offset=0.02))
    assert result["passes"]


def test_threshold_is_configurable():
    a, b = grid_of(ndvi=0.6), grid_of(ndvi=0.6, offset=0.04)
    assert repeat.compare(a, b, threshold=0.05)["passes"]
    assert not repeat.compare(a, b, threshold=0.01)["passes"]


def test_sign_of_bias_follows_argument_order():
    a = grid_of(ndvi=0.6)
    b = grid_of(ndvi=0.6, offset=-0.08)
    assert repeat.compare(a, b)["bias"] < 0
    assert repeat.compare(b, a)["bias"] > 0


# --------------------------------------------------------------------------
# shared-grid alignment (the regression these tests exist for)
# --------------------------------------------------------------------------

def test_cells_align_when_flight_b_covers_less_ground():
    """B covers a sub-area of A. Overlapping cells must still match up.

    Gridded separately, B's smaller extent would produce a different origin and
    every cell would be compared against the wrong neighbour, inventing error out
    of nothing.
    """
    a = grid_of(n=4, ndvi=0.6)
    b = grid_of(n=2, ndvi=0.6)          # same corner, smaller footprint
    result = repeat.compare(a, b)
    assert result["ok"]
    assert result["mean_abs_delta"] == pytest.approx(0.0, abs=1e-9)
    # only the cells B actually covers get compared
    assert result["n_cells_compared"] < result["n_cells_a"]


def test_shared_grid_dimensions_cover_both_flights():
    a = grid_of(n=2)
    b = fc([(LAT0 + 5 * DLAT, LON0 + 5 * DLON, 0.6)] * 2)
    grid = repeat.shared_grid(a, b)
    span_m = 5 * _SPACING_M
    assert grid["nrows"] * grid["cell_meters"] >= span_m
    assert grid["ncols"] * grid["cell_meters"] >= span_m


def test_bin_onto_matches_build_grid_on_the_same_data():
    """bin_onto must reproduce build_grid's index arithmetic exactly."""
    a = grid_of(n=3, ndvi=0.55)
    grid = field.build_grid(a, cell_meters=20.0)
    mean, counts = repeat.bin_onto(grid, a)
    assert counts.sum() == grid["counts"].sum()
    assert (counts == grid["counts"]).all()


def test_spatially_varying_error_is_not_hidden_by_averaging():
    """One badly-off cell should raise the max without dominating the mean."""
    a = grid_of(n=3, ndvi=0.6)
    b_pts = []
    for r in range(3):
        for c in range(3):
            v = 0.6 if not (r == 0 and c == 0) else 0.2
            b_pts += [(LAT0 + r * DLAT, LON0 + c * DLON, v)] * 2
    result = repeat.compare(a, fc(b_pts))
    assert result["max_abs_delta"] == pytest.approx(0.4, abs=1e-6)
    assert result["mean_abs_delta"] < 0.1


# --------------------------------------------------------------------------
# coverage guards
# --------------------------------------------------------------------------

def test_min_count_excludes_thin_cells():
    """A cell holding one photo is one frame's noise, not a measurement."""
    a = grid_of(n=2, repeats=1)
    b = grid_of(n=2, repeats=1)
    assert repeat.compare(a, b, min_count=2)["n_cells_compared"] == 0
    assert repeat.compare(a, b, min_count=1)["n_cells_compared"] > 0


def test_non_overlapping_flights_are_inconclusive_not_wrong():
    a = grid_of(n=2)
    far = fc([(LAT0 + 50 * DLAT, LON0 + 50 * DLON, 0.6)] * 2)
    result = repeat.compare(a, far)
    assert not result["ok"]
    assert result["mean_abs_delta"] is None
    assert "overlap" in result["reason"] or "enough photos" in result["reason"]


def test_inconclusive_verdict_is_labelled():
    a = grid_of(n=2)
    far = fc([(LAT0 + 50 * DLAT, LON0 + 50 * DLON, 0.6)] * 2)
    assert repeat.verdict(repeat.compare(a, far)).startswith("INCONCLUSIVE")


def test_partial_overlap_is_flagged_as_a_caution():
    a = grid_of(n=6, ndvi=0.6)
    b = grid_of(n=2, ndvi=0.6)
    result = repeat.compare(a, b)
    assert result["overlap_fraction"] < 0.5
    assert "partial overlap" in repeat.verdict(result)


def test_untagged_flight_raises_a_useful_error():
    a = grid_of(n=2)
    empty = {"type": "FeatureCollection", "features": []}
    with pytest.raises(ValueError) as exc:
        repeat.compare(a, empty)
    assert "geotagged" in str(exc.value)


def test_frame_counts_are_reported():
    result = repeat.compare(grid_of(n=2, repeats=3), grid_of(n=2, repeats=2))
    assert result["n_frames_a"] == 12
    assert result["n_frames_b"] == 8


def test_cell_size_changes_the_grouping():
    a = grid_of(n=4, ndvi=0.6)
    fine = repeat.compare(a, grid_of(n=4, ndvi=0.6), cell_meters=20.0)
    coarse = repeat.compare(a, grid_of(n=4, ndvi=0.6), cell_meters=80.0)
    assert coarse["n_cells_compared"] < fine["n_cells_compared"]
