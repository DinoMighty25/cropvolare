"""Survey-planner geometry tests - pure math, no hardware or I/O."""

import math
import xml.etree.ElementTree as ET

import pytest

from cropvolare import planner

# a ~100 x 100 m square field at (40, -88)
LAT0, LON0 = 40.0, -88.0
DLAT = 100 / 111_195          # ~meters per degree latitude
DLON = 100 / (111_195 * math.cos(math.radians(LAT0)))
SQUARE = [
    (LAT0, LON0),
    (LAT0, LON0 + DLON),
    (LAT0 + DLAT, LON0 + DLON),
    (LAT0 + DLAT, LON0),
]


def test_footprint_math():
    # 2 * 30 * tan(33 deg) ~ 38.96 m
    assert abs(planner.ground_footprint(30, hfov_deg=66.0) - 38.96) < 0.1


def test_square_line_count_matches_spacing():
    lines = planner.survey_lines(SQUARE, altitude_m=30, side_overlap=0.75)
    # spacing ~9.74 m over 100 m -> ~10 lines
    assert 9 <= len(lines) <= 12
    # every endpoint stays within the field bbox (+2 m slack)
    for line in lines:
        for lat, lon in line:
            assert LAT0 - 2e-5 <= lat <= LAT0 + DLAT + 2e-5
            assert LON0 - 3e-5 <= lon <= LON0 + DLON + 3e-5


def test_more_overlap_means_more_lines():
    few = planner.survey_lines(SQUARE, 30, side_overlap=0.5)
    many = planner.survey_lines(SQUARE, 30, side_overlap=0.75)
    assert len(many) > len(few) * 1.5


def test_lines_follow_long_axis():
    # 200 x 50 m rectangle: default heading = longest edge -> long lines
    rect = [
        (LAT0, LON0),
        (LAT0, LON0 + 2 * DLON),                 # 200 m east
        (LAT0 + DLAT / 2, LON0 + 2 * DLON),      # 50 m north
        (LAT0 + DLAT / 2, LON0),
    ]
    lines = planner.survey_lines(rect, 30, side_overlap=0.75)
    to_xy, _ = planner._projection(rect)
    for line in lines:
        (x1, y1), (x2, y2) = [to_xy(*p) for p in line]
        assert math.hypot(x2 - x1, y2 - y1) > 150  # runs the long way


def test_boustrophedon_alternates():
    lines = planner.survey_lines(SQUARE, 30, side_overlap=0.75)
    to_xy, _ = planner._projection(SQUARE)
    dirs = []
    for line in lines:
        (x1, y1), (x2, y2) = [to_xy(*p) for p in line]
        dirs.append((x2 - x1, y2 - y1))
    # consecutive parallel lines run in opposite directions (dot product < 0),
    # regardless of which axis the lines follow
    for a, b in zip(dirs, dirs[1:]):
        assert a[0] * b[0] + a[1] * b[1] < 0


def test_polygon_too_small():
    with pytest.raises(ValueError):
        planner.survey_lines([(40.0, -88.0), (40.001, -88.0)], 30)


def test_plan_stats():
    lines = planner.survey_lines(SQUARE, 30, side_overlap=0.75)
    stats = planner.plan_stats(lines, altitude_m=30, speed_mps=4.0,
                               interval_s=2.0)
    # ~10 lines x 100 m + transits: distance in the 900-1500 m range
    assert 900 <= stats["distance_m"] <= 1500
    assert stats["time_min"] > 0
    assert stats["est_frames"] > 50
    # forward footprint at 30 m ~ 22.4 m; 8 m between frames -> ~0.64 overlap
    assert 0.5 <= stats["forward_overlap"] <= 0.8
    assert stats["overlap_ok"] is True


def test_plan_stats_flags_too_fast():
    lines = planner.survey_lines(SQUARE, 30, side_overlap=0.75)
    stats = planner.plan_stats(lines, altitude_m=30, speed_mps=12.0,
                               interval_s=2.0)
    assert stats["overlap_ok"] is False


def test_kml_is_valid_xml():
    lines = planner.survey_lines(SQUARE, 30)
    kml = planner.to_kml(lines)
    root = ET.fromstring(kml)
    assert root.tag.endswith("kml")
    assert "coordinates" in kml


def test_litchi_csv_shape():
    lines = planner.survey_lines(SQUARE, 30)
    csv = planner.to_litchi_csv(lines, altitude_m=30)
    rows = csv.strip().splitlines()
    assert rows[0].startswith("latitude,longitude,altitude(m)")
    assert len(rows) == 1 + len(planner.waypoints(lines))
    assert rows[1].endswith(",2,-90")  # nadir gimbal
