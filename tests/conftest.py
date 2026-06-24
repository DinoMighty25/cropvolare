"""Shared test fixtures: synthetic geotagged NDVI photos, no hardware needed."""

import cv2
import numpy as np
import pytest

from cropvolare import batch, field, fieldmap, geo


def make_ndvi_jpeg(path, lat, lon, nir=200, red=80, alt=50.0, size=64):
    """Write a uniform JPEG (blue=NIR, red=Red) and stamp GPS EXIF on it.

    Channel layout matches the rest of the suite: BGR with blue=NIR, red=Red,
    so compute_ndvi_from_image sees the values we set. Returns the path (str).
    """
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, :, 0] = nir  # blue channel = NIR
    img[:, :, 2] = red  # red channel = visible red
    path = str(path)
    cv2.imwrite(path, img)
    geo.write_gps(path, lat, lon, alt=alt)
    return path


@pytest.fixture
def ndvi_jpeg_factory(tmp_path):
    """Factory fixture: call to drop a geotagged JPEG into the temp dir."""
    counter = {"n": 0}

    def _make(lat, lon, nir=200, red=80, alt=50.0, name=None):
        counter["n"] += 1
        fname = name or f"img_{counter['n']:03d}.jpg"
        return make_ndvi_jpeg(tmp_path / fname, lat, lon, nir, red, alt)

    return _make


@pytest.fixture
def geotagged_dir(tmp_path):
    """A small field: a healthy cluster and one clearly stressed corner.

    Returns the directory path. Coordinates are spread far enough apart that
    field gridding puts the stressed corner in its own cell.
    """
    d = tmp_path / "flight"
    d.mkdir()
    # healthy cluster near (40.0000, -88.0000): high NIR, low red -> NDVI ~ high
    make_ndvi_jpeg(d / "h1.jpg", 40.00000, -88.00000, nir=220, red=40)
    make_ndvi_jpeg(d / "h2.jpg", 40.00002, -88.00001, nir=210, red=50)
    make_ndvi_jpeg(d / "h3.jpg", 40.00001, -88.00002, nir=215, red=45)
    # stressed corner ~30 m away: red >= NIR -> NDVI low/negative
    make_ndvi_jpeg(d / "s1.jpg", 40.00030, -88.00030, nir=70, red=180)
    # one untagged photo (no GPS) to exercise the n_untagged path
    plain = np.zeros((64, 64, 3), dtype=np.uint8)
    plain[:, :, 0] = 100
    cv2.imwrite(str(d / "untagged.jpg"), plain)
    return d


@pytest.fixture
def three_tier_dir(tmp_path):
    """A field with deterministic healthy, moderate, and severe clusters.

    Clusters are spread >20 m apart so each lands in its own grid cell, making
    the resulting cell statuses assertable.
    """
    d = tmp_path / "field3"
    d.mkdir()
    # healthy: high NIR, very low red -> NDVI well above 0.5
    make_ndvi_jpeg(d / "h.jpg", 40.00000, -88.00000, nir=230, red=30)
    # moderate: NIR a bit above red -> mid NDVI (between thresholds)
    make_ndvi_jpeg(d / "m.jpg", 40.00040, -88.00000, nir=150, red=120)
    # severe: red exceeds NIR -> NDVI near/below zero
    make_ndvi_jpeg(d / "s.jpg", 40.00080, -88.00080, nir=60, red=200)
    return d


@pytest.fixture
def all_untagged_dir(tmp_path):
    """A directory where no photo carries GPS (exercises the empty pipeline)."""
    d = tmp_path / "notags"
    d.mkdir()
    for i in range(3):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        img[:, :, 0] = 180
        img[:, :, 2] = 60
        cv2.imwrite(str(d / f"u{i}.jpg"), img)
    return d


@pytest.fixture
def flight_outputs(geotagged_dir, tmp_path):
    """Run batch -> field -> fieldmap once and yield everything downstream needs.

    Shared by output and CLI content tests so the pipeline isn't recomputed.
    Returns a dict with fc, grid, cells, problems, summary, heatmap_png, bounds.
    """
    fc = batch.process_directory(str(geotagged_dir), flight_date="2026-06-21",
                                 generated="2026-06-21T12:00:00Z")
    grid = field.build_grid(fc, cell_meters=20.0)
    cells = field.classify_cells(grid)
    problems = field.rank_problems(cells, top_n=5)
    summary = field.summarize(cells)
    heatmap_png = str(tmp_path / "heatmap.png")
    bounds = fieldmap.render_grid_png(grid, heatmap_png)
    return {
        "fc": fc, "grid": grid, "cells": cells, "problems": problems,
        "summary": summary, "heatmap_png": heatmap_png, "bounds": bounds,
    }
