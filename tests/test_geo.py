import cv2
import numpy as np

from cropvolare import geo
from cropvolare.geo import _decimal_to_dms, _dms_to_decimal


def test_write_read_roundtrip(ndvi_jpeg_factory):
    path = ndvi_jpeg_factory(40.123456, -88.654321, alt=72.5)
    gps = geo.read_gps(path)
    assert gps is not None
    assert abs(gps["lat"] - 40.123456) < 1e-4
    assert abs(gps["lon"] - (-88.654321)) < 1e-4
    assert abs(gps["alt"] - 72.5) < 0.5


def test_southern_western_hemisphere(ndvi_jpeg_factory):
    # negative lat (S) and negative lon (W) must survive the round trip
    path = ndvi_jpeg_factory(-33.8688, -151.2093, alt=10.0)
    gps = geo.read_gps(path)
    assert gps["lat"] < 0
    assert gps["lon"] < 0
    assert abs(gps["lat"] - (-33.8688)) < 1e-4
    assert abs(gps["lon"] - (-151.2093)) < 1e-4


def test_negative_altitude(ndvi_jpeg_factory):
    path = ndvi_jpeg_factory(40.0, -88.0, alt=-5.0)
    gps = geo.read_gps(path)
    assert gps["alt"] < 0


def test_untagged_returns_none(tmp_path):
    plain = np.zeros((32, 32, 3), dtype=np.uint8)
    p = str(tmp_path / "plain.jpg")
    cv2.imwrite(p, plain)
    assert geo.read_gps(p) is None
    assert geo.has_gps(p) is False


def test_has_gps_true(ndvi_jpeg_factory):
    path = ndvi_jpeg_factory(1.0, 2.0)
    assert geo.has_gps(path) is True


def test_dms_decimal_inverse():
    for value in (0.0, 12.3456, 88.999, 179.5):
        dms = _decimal_to_dms(value)
        back = _dms_to_decimal(dms, "N")
        assert abs(back - value) < 1e-4
