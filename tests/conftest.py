"""Shared test fixtures: synthetic geotagged NDVI photos, no hardware needed."""

import cv2
import numpy as np
import pytest

from cropvolare import geo


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
