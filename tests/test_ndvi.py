import numpy as np
from cropvolare.ndvi import compute_ndvi, compute_ndvi_from_image, extract_channels, classify_zones


def test_extract_channels_shape():
    img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    nir, red = extract_channels(img)
    assert nir.shape == (100, 100)
    assert red.shape == (100, 100)


def test_extract_channels_normalized():
    img = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
    nir, red = extract_channels(img)
    assert nir.min() >= 0.0 and nir.max() <= 1.0
    assert red.min() >= 0.0 and red.max() <= 1.0


def test_all_nir_gives_plus_one():
    nir = np.ones((10, 10))
    red = np.zeros((10, 10))
    np.testing.assert_allclose(compute_ndvi(nir, red), 1.0)


def test_all_red_gives_minus_one():
    nir = np.zeros((10, 10))
    red = np.ones((10, 10))
    np.testing.assert_allclose(compute_ndvi(nir, red), -1.0)


def test_equal_channels_gives_zero():
    arr = np.full((10, 10), 0.5)
    np.testing.assert_allclose(compute_ndvi(arr, arr), 0.0)


def test_both_zero_no_crash():
    z = np.zeros((10, 10))
    np.testing.assert_allclose(compute_ndvi(z, z), 0.0)


def test_from_image():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, :, 0] = 200  # NIR (blue channel)
    img[:, :, 2] = 100  # red channel
    ndvi = compute_ndvi_from_image(img)
    expected = (200 - 100) / (200 + 100)
    np.testing.assert_allclose(ndvi, expected, atol=1e-6)


def test_zones_healthy():
    ndvi = np.full((64, 64), 0.7)
    zones = classify_zones(ndvi, block_size=64)
    assert len(zones) == 1
    assert zones[0]["status"] == "healthy"


def test_zones_stressed():
    ndvi = np.full((64, 64), 0.1)
    zones = classify_zones(ndvi, block_size=64)
    assert zones[0]["status"] == "stressed"


def test_zones_grid_count():
    ndvi = np.full((128, 128), 0.5)
    zones = classify_zones(ndvi, block_size=64)
    assert len(zones) == 4
