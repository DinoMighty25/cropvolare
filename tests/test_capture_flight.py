"""Tests for the flight-capture loop core (no camera or GPS hardware).

The loop takes injected capture/gps/save callables, so the whole control flow
is exercised on the laptop with fakes.
"""

import importlib.util
import os

import cv2
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_capture_flight():
    path = os.path.join(REPO_ROOT, "scripts", "capture_flight.py")
    spec = importlib.util.spec_from_file_location("capture_flight", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cf = _load_capture_flight()


def _frame(v):
    return np.full((8, 8, 3), v, dtype=np.uint8)


def test_saves_sequential_jpegs(tmp_path):
    frames = iter([_frame(10), _frame(20), _frame(30)])
    saved = cf.run_capture(
        str(tmp_path),
        capture_fn=lambda: next(frames),
        n_frames=3, interval=0,
        sleep_fn=lambda s: None, log_fn=lambda m: None,
    )
    assert [os.path.basename(p) for p in saved] == [
        "frame_0000.jpg", "frame_0001.jpg", "frame_0002.jpg"]
    assert all(os.path.exists(p) for p in saved)


def test_tags_gps_when_fix_available(tmp_path):
    tag_calls = []
    saved = cf.run_capture(
        str(tmp_path),
        capture_fn=lambda: _frame(50),
        n_frames=2, interval=0,
        gps_fn=lambda: {"lat": 40.0, "lon": -88.0, "alt": 60.0},
        tag_fn=lambda path, lat, lon, alt=None: tag_calls.append((path, lat, lon, alt)),
        sleep_fn=lambda s: None, log_fn=lambda m: None,
    )
    assert len(tag_calls) == 2
    assert tag_calls[0][1] == 40.0 and tag_calls[0][2] == -88.0


def test_no_gps_leaves_frames_untagged(tmp_path):
    tag_calls = []
    cf.run_capture(
        str(tmp_path),
        capture_fn=lambda: _frame(50),
        n_frames=2, interval=0, gps_fn=None,
        tag_fn=lambda *a, **k: tag_calls.append(a),
        sleep_fn=lambda s: None, log_fn=lambda m: None,
    )
    assert tag_calls == []


def test_skips_tag_when_no_fix_yet(tmp_path):
    # gps_fn present but returns None (no lock yet) -> no tagging, no crash
    tag_calls = []
    saved = cf.run_capture(
        str(tmp_path),
        capture_fn=lambda: _frame(50),
        n_frames=2, interval=0,
        gps_fn=lambda: None,
        tag_fn=lambda *a, **k: tag_calls.append(a),
        sleep_fn=lambda s: None, log_fn=lambda m: None,
    )
    assert tag_calls == []
    assert len(saved) == 2


def test_saved_jpegs_are_readable_and_geotaggable(tmp_path):
    # end-to-end with the REAL save + tag path: files load back and carry GPS
    from cropvolare import geo
    saved = cf.run_capture(
        str(tmp_path),
        capture_fn=lambda: _frame(123),
        n_frames=1, interval=0,
        gps_fn=lambda: {"lat": 12.34, "lon": -56.78, "alt": 100.0},
        sleep_fn=lambda s: None, log_fn=lambda m: None,
    )
    img = cv2.imread(saved[0])
    assert img is not None and img.shape == (8, 8, 3)
    gps = geo.read_gps(saved[0])
    assert abs(gps["lat"] - 12.34) < 1e-3 and abs(gps["lon"] - (-56.78)) < 1e-3
