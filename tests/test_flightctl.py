"""Tests for the flight-control machinery (shared by fly.py and the GCS)."""

import os

from cropvolare import flightctl


def test_make_flight_dir_timestamped_and_unique(tmp_path):
    from datetime import datetime
    now = datetime(2026, 7, 4, 12, 0, 0)
    d1 = flightctl.make_flight_dir(str(tmp_path), now=now)
    d2 = flightctl.make_flight_dir(str(tmp_path), now=now)  # same second
    assert os.path.basename(d1) == "2026-07-04_120000"
    assert d1 != d2
    assert os.path.isdir(d1) and os.path.isdir(d2)


def test_preflight_reports_failure(tmp_path):
    msgs = []
    ok = flightctl.preflight(str(tmp_path),
                             checks=(("doomed", lambda b: (False, "nope")),),
                             log_fn=msgs.append)
    assert ok is False
    assert any("FAIL" in m and "doomed" in m for m in msgs)


def test_preflight_disk_and_write_pass_on_laptop(tmp_path):
    ok = flightctl.preflight(
        str(tmp_path),
        checks=(("disk space", flightctl.check_disk),
                ("storage", flightctl.check_writable)),
        log_fn=lambda m: None)
    assert ok is True


def test_status_info_without_active_flight(tmp_path):
    info = flightctl.status_info(str(tmp_path))
    assert info["capturing"] is False
    assert info["flight"] is None
    assert info["frames"] == 0


def test_status_info_counts_frames(tmp_path):
    base = str(tmp_path)
    d = os.path.join(base, "f1")
    os.makedirs(d)
    with open(os.path.join(d, "frame_0000.jpg"), "wb") as f:
        f.write(b"x")
    flightctl.write_meta(base, {"dir": d, "pid": 123})
    info = flightctl.status_info(base, alive_fn=lambda pid: True)
    assert info["capturing"] is True
    assert info["flight"] == "f1"
    assert info["frames"] == 1
    assert info["disk_free_mb"] > 0


def test_stop_writes_stop_file_and_clears_meta(tmp_path):
    base = str(tmp_path)
    d = os.path.join(base, "f2")
    os.makedirs(d)
    for i in range(3):
        with open(os.path.join(d, f"frame_{i:04d}.jpg"), "wb") as f:
            f.write(b"x")
    flightctl.write_meta(base, {"dir": d, "pid": 999999, "interval": 0})
    rc = flightctl.stop(base, log_fn=lambda m: None,
                        alive_fn=lambda pid: False, timeout=0)
    assert rc == 0
    assert os.path.exists(os.path.join(d, "STOP"))   # capture loop's cue
    assert flightctl.read_meta(base) is None          # bookkeeping cleared


def test_stop_without_active_flight(tmp_path):
    assert flightctl.stop(str(tmp_path), log_fn=lambda m: None) == 1


def test_list_flights_newest_first(tmp_path):
    import time
    base = str(tmp_path)
    for name, n in (("older", 2), ("newer", 3)):
        d = os.path.join(base, name)
        os.makedirs(d)
        for i in range(n):
            with open(os.path.join(d, f"frame_{i:04d}.jpg"), "wb") as f:
                f.write(b"xx")
        time.sleep(0.05)
    flights = flightctl.list_flights(base)
    assert [f["name"] for f in flights] == ["newer", "older"]
    assert flights[0]["n_frames"] == 3


def test_latest_frame_skips_empty(tmp_path):
    base = str(tmp_path)
    d = os.path.join(base, "f3")
    os.makedirs(d)
    with open(os.path.join(d, "frame_0000.jpg"), "wb") as f:
        f.write(b"real")
    open(os.path.join(d, "frame_0001.jpg"), "w").close()  # 0-byte in-progress
    latest = flightctl.latest_frame(base)
    assert latest.endswith("frame_0000.jpg")


def test_latest_frame_none_when_empty(tmp_path):
    assert flightctl.latest_frame(str(tmp_path)) is None


# --- storage autopilot -------------------------------------------------------

def _mk_flight(base, name, frames, processed=False, mtime=None):
    import cv2
    import numpy as np
    d = base / name
    d.mkdir(parents=True)
    img = np.zeros((16, 16, 3), np.uint8)
    for i in range(frames):
        cv2.imwrite(str(d / f"frame_{i:04d}.jpg"), img)
    if processed:
        a = d / "analysis"
        a.mkdir()
        (a / "report.pdf").write_bytes(b"%PDF fake")
    if mtime:
        os.utime(str(d), (mtime, mtime))
    return d


def test_autopilot_purges_trivial_captures_only(tmp_path):
    base = tmp_path / "flights"
    _mk_flight(base, "real", 20, mtime=1000)
    _mk_flight(base, "junk", 3, mtime=2000)
    deleted = flightctl.storage_autopilot(str(base),
                                          free_mb_fn=lambda: 999999,
                                          log_fn=lambda *_: None)
    assert deleted == ["junk"]
    assert not (base / "junk").exists()
    assert (base / "real").exists()


def test_autopilot_reclaims_oldest_processed_when_low(tmp_path):
    base = tmp_path / "flights"
    _mk_flight(base, "old_done", 20, processed=True, mtime=1000)
    _mk_flight(base, "unprocessed", 20, mtime=2000)
    _mk_flight(base, "new_done", 20, processed=True, mtime=3000)
    free = iter([100, 999999])       # low until one deletion frees space
    deleted = flightctl.storage_autopilot(str(base),
                                          free_mb_fn=lambda: next(free),
                                          log_fn=lambda *_: None)
    assert deleted == ["old_done"]   # oldest processed dies first
    assert (base / "unprocessed").exists()   # raw data is never deleted
    assert (base / "new_done").exists()


def test_autopilot_never_touches_active_flight(tmp_path):
    base = tmp_path / "flights"
    act = _mk_flight(base, "active", 2, mtime=1000)   # tiny AND active
    flightctl.write_meta(str(base), {"dir": str(act), "pid": None})
    deleted = flightctl.storage_autopilot(str(base),
                                          free_mb_fn=lambda: 999999,
                                          log_fn=lambda *_: None)
    assert deleted == []
    assert act.exists()
