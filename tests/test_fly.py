"""Tests for the fly.py field wrapper - bookkeeping logic only, no hardware.

Camera/subprocess paths are Pi-side; here we cover the parts that decide
whether a flight is safe to start, where it goes, and how it stops.
"""

import importlib.util
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_fly():
    path = os.path.join(REPO_ROOT, "scripts", "fly.py")
    spec = importlib.util.spec_from_file_location("fly", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fly = _load_fly()


def test_make_flight_dir_timestamped_and_unique(tmp_path):
    from datetime import datetime
    now = datetime(2026, 7, 4, 12, 0, 0)
    d1 = fly.make_flight_dir(str(tmp_path), now=now)
    d2 = fly.make_flight_dir(str(tmp_path), now=now)  # same second -> suffixed
    assert os.path.basename(d1) == "2026-07-04_120000"
    assert d1 != d2
    assert os.path.isdir(d1) and os.path.isdir(d2)


def test_preflight_reports_failure(tmp_path):
    msgs = []
    ok = fly.preflight(str(tmp_path),
                       checks=(("doomed", lambda b: (False, "nope")),),
                       log_fn=msgs.append)
    assert ok is False
    assert any("FAIL" in m and "doomed" in m for m in msgs)


def test_preflight_disk_and_write_pass_on_laptop(tmp_path):
    ok = fly.preflight(str(tmp_path),
                       checks=(("disk space", fly.check_disk),
                               ("storage", fly.check_writable)),
                       log_fn=lambda m: None)
    assert ok is True


def test_status_without_active_flight(tmp_path):
    assert fly.status(str(tmp_path), log_fn=lambda m: None) == 1


def test_status_counts_frames(tmp_path):
    base = str(tmp_path)
    d = os.path.join(base, "f1")
    os.makedirs(d)
    with open(os.path.join(d, "frame_0000.jpg"), "wb") as f:
        f.write(b"x")
    fly.write_meta(base, {"dir": d, "pid": 123})
    msgs = []
    rc = fly.status(base, log_fn=msgs.append, alive_fn=lambda pid: True)
    assert rc == 0
    assert any("frames:  1" in m for m in msgs)


def test_stop_writes_stop_file_and_clears_meta(tmp_path):
    base = str(tmp_path)
    d = os.path.join(base, "f2")
    os.makedirs(d)
    for i in range(3):
        with open(os.path.join(d, f"frame_{i:04d}.jpg"), "wb") as f:
            f.write(b"x")
    fly.write_meta(base, {"dir": d, "pid": 999999, "interval": 0})
    rc = fly.stop(base, log_fn=lambda m: None,
                  alive_fn=lambda pid: False, timeout=0)
    assert rc == 0
    assert os.path.exists(os.path.join(d, "STOP"))   # capture loop's cue
    assert fly.read_meta(base) is None               # bookkeeping cleared


def test_stop_without_active_flight(tmp_path):
    assert fly.stop(str(tmp_path), log_fn=lambda m: None) == 1
