"""Tests for NDVI computation (no camera hardware required)."""

import numpy as np
import pytest

from cropvolare.ndvi import (
    classify_zones,
    compute_ndvi,
    compute_ndvi_from_image,
    extract_channels,
)


def test_extract_channels_shape():
    img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    nir, red = extract_channels(img)
    assert nir.shape == (100, 100)
    assert red.shape == (100, 100)


def test_extract_channels_range():
    img = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
    nir, red = extract_channels(img)
    assert nir.min() >= 0.0 and nir.max() <= 1.0
    assert red.min() >= 0.0 and red.max() <= 1.0


def test_ndvi_pure_nir():
    """When red = 0, NDVI should be +1."""
    nir = np.ones((10, 10))
    red = np.zeros((10, 10))
    ndvi = compute_ndvi(nir, red)
    np.testing.assert_allclose(ndvi, 1.0)


def test_ndvi_pure_red():
    """When NIR = 0, NDVI should be -1."""
    nir = np.zeros((10, 10))
    red = np.ones((10, 10))
    ndvi = compute_ndvi(nir, red)
    np.testing.assert_allclose(ndvi, -1.0)


def test_ndvi_equal_channels():
    """When NIR == Red, NDVI should be 0."""
    nir = np.full((10, 10), 0.5)
    red = np.full((10, 10), 0.5)
    ndvi = compute_ndvi(nir, red)
    np.testing.assert_allclose(ndvi, 0.0)


def test_ndvi_zero_both():
    """When both channels are 0, NDVI should be 0 (no division by zero)."""
    nir = np.zeros((10, 10))
    red = np.zeros((10, 10))
    ndvi = compute_ndvi(nir, red)
    np.testing.assert_allclose(ndvi, 0.0)


def test_compute_ndvi_from_image():
    # Create a synthetic BGR image: blue=200 (NIR), red=100 (visible)
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, :, 0] = 200  # blue channel -> NIR
    img[:, :, 2] = 100  # red channel -> visible red
    ndvi = compute_ndvi_from_image(img)
    # Expected: (200/255 - 100/255) / (200/255 + 100/255) = 100/300 = 0.333...
    expected = (200 - 100) / (200 + 100)
    np.testing.assert_allclose(ndvi, expected, atol=1e-6)


def test_classify_zones_healthy():
    ndvi = np.full((64, 64), 0.7)
    zones = classify_zones(ndvi, block_size=64)
    assert len(zones) == 1
    assert zones[0]["status"] == "healthy"


def test_classify_zones_stressed():
    ndvi = np.full((64, 64), 0.1)
    zones = classify_zones(ndvi, block_size=64)
    assert len(zones) == 1
    assert zones[0]["status"] == "stressed"


def test_classify_zones_grid():
    ndvi = np.full((128, 128), 0.5)
    zones = classify_zones(ndvi, block_size=64)
    assert len(zones) == 4  # 2x2 grid
