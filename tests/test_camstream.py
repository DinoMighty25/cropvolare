"""CameraSession tests - fake camera, injected clock, no hardware."""

import numpy as np
import pytest

from cropvolare.camstream import CameraSession


class _FakeCam:
    def __init__(self, log):
        self.log = log

    def start(self):
        self.log.append("start")

    def stop(self):
        self.log.append("stop")

    def close(self):
        self.log.append("close")

    def capture_array(self):
        arr = np.zeros((4, 4, 3), np.uint8)
        arr[:, :, 0] = 120
        return arr


def _session(log, now):
    return CameraSession(camera_factory=lambda: _FakeCam(log),
                         warmup=0, time_fn=lambda: now[0])


def test_one_camera_shared_by_viewers():
    log, now = [], [100.0]
    s = _session(log, now)
    s.acquire()
    s.acquire()                       # second viewer: no second open
    assert log.count("start") == 1
    assert s.frame() is not None
    s.release()
    assert s.open                     # one viewer still watching
    s.release()
    assert not s.open                 # last viewer left -> camera closed
    assert "stop" in log and "close" in log


def test_pause_and_close_hands_camera_over():
    log, now = [], [100.0]
    s = _session(log, now)
    s.acquire()
    s.pause_and_close(seconds=20)
    assert not s.open
    assert s.frame() is None          # streaming generators see this and end
    with pytest.raises(RuntimeError):
        s.acquire()                   # no reopening during the grace window
    now[0] = 121.0                    # window over (capture guard takes over)
    s.acquire()
    assert s.frame() is not None


def test_frame_none_when_never_opened():
    log, now = [], [100.0]
    s = _session(log, now)
    assert s.frame() is None
