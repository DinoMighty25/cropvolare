"""CameraSession tests - fake camera, real (tiny) threads, no hardware."""

import time

import numpy as np

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


def _wait_for(predicate, timeout=3.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_worker_produces_frames_and_idle_closes():
    log = []
    s = CameraSession(camera_factory=lambda: _FakeCam(log), warmup=0,
                      idle_timeout=0.3)
    assert s.get_frame() is None            # first call: worker starting
    assert _wait_for(lambda: s.get_frame() is not None)
    assert log.count("start") == 1
    # stop asking for frames -> idle watchdog closes the camera
    assert _wait_for(lambda: not s.running)
    assert "stop" in log and "close" in log


def test_pause_and_close_hands_camera_over():
    log = []
    s = CameraSession(camera_factory=lambda: _FakeCam(log), warmup=0,
                      idle_timeout=5.0)
    s.get_frame()
    assert _wait_for(lambda: s.get_frame() is not None)
    s.pause_and_close(seconds=1.0)
    assert not s.running                    # worker joined, camera released
    assert "close" in log
    assert s.paused
    assert s.get_frame() is None            # no reopening during the window
    assert log.count("start") == 1
    time.sleep(1.1)                         # window over
    assert not s.paused
    s.get_frame()
    assert _wait_for(lambda: s.get_frame() is not None)
    assert log.count("start") == 2


def test_failed_open_backs_off():
    calls = []

    def bad_factory():
        calls.append(1)
        raise RuntimeError("no camera here")

    s = CameraSession(camera_factory=bad_factory, warmup=0,
                      fail_cooldown=0.4)
    assert s.get_frame() is None
    assert _wait_for(lambda: not s.running)
    # hammering get_frame during the cooldown must not respawn workers
    for _ in range(10):
        assert s.get_frame() is None
    assert len(calls) == 1
    time.sleep(0.5)                         # cooldown over -> one retry
    s.get_frame()
    assert _wait_for(lambda: len(calls) == 2)
