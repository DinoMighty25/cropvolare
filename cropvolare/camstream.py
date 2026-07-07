"""
Shared live-view camera session for the GCS viewfinder.

Design: a BACKGROUND WORKER THREAD owns the camera and keeps the latest frame
in a buffer; stream requests only read the buffer. Requests never own the
camera, so an aborted/vanished client (phone hops WiFi, curl times out) can't
leak it - the first per-request-refcount version did exactly that, leaving the
camera stuck in "Running state" until a service restart. An idle watchdog
closes the camera when nobody has asked for a frame in idle_timeout seconds.

Safety property (non-negotiable): the session must never hold the camera when
a flight capture wants it. pause_and_close() stops the worker, releases the
camera, and blocks re-opening for a grace window - the GCS start endpoint
calls it before spawning capture.
"""

import threading
import time


class CameraSession:
    """Latest-frame buffer backed by a self-managing camera worker thread.

    camera_factory/time_fn are injectable for hardware-free tests. The real
    factory builds ndvi.create_camera at a preview-friendly resolution.
    """

    def __init__(self, camera_factory=None, resolution=(1152, 648),
                 warmup=0.8, idle_timeout=10.0, fail_cooldown=5.0,
                 time_fn=time.time):
        self._factory = camera_factory
        self._resolution = resolution
        self._warmup = warmup
        self._idle_timeout = idle_timeout
        self._fail_cooldown = fail_cooldown
        self._time = time_fn
        self._lock = threading.Lock()
        self._frame = None
        self._last_use = 0.0
        self._pause_until = 0.0
        self._fail_until = 0.0
        self._worker = None
        self._stop_evt = threading.Event()

    def _default_factory(self):
        from .ndvi import create_camera
        return create_camera(resolution=self._resolution)

    @property
    def paused(self):
        with self._lock:
            return self._time() < self._pause_until

    @property
    def running(self):
        with self._lock:
            return self._worker is not None and self._worker.is_alive()

    def get_frame(self):
        """Latest BGR frame, or None while warming up / paused / failed.

        Calling this marks the session as in-use (feeds the idle watchdog) and
        lazily starts the camera worker. Never blocks on the camera.
        """
        with self._lock:
            now = self._time()
            if now < self._pause_until or now < self._fail_until:
                return None
            self._last_use = now
            if self._worker is None or not self._worker.is_alive():
                self._stop_evt = threading.Event()
                self._worker = threading.Thread(
                    target=self._run, args=(self._stop_evt,), daemon=True)
                self._worker.start()
            return self._frame

    def _run(self, stop_evt):
        try:
            factory = self._factory or self._default_factory
            cam = factory()
            cam.start()
            time.sleep(self._warmup)
        except Exception as exc:  # noqa: BLE001 - camera missing/busy
            print(f"viewfinder: camera unavailable: {exc}")
            with self._lock:
                self._fail_until = self._time() + self._fail_cooldown
                self._worker = None
            return
        try:
            from .ndvi import capture_frame
            while not stop_evt.is_set():
                frame = capture_frame(cam)
                with self._lock:
                    self._frame = frame
                    now = self._time()
                    done = (now < self._pause_until
                            or now - self._last_use > self._idle_timeout)
                if done:
                    break
                stop_evt.wait(0.15)
        except Exception as exc:  # noqa: BLE001 - camera died mid-stream
            print(f"viewfinder: camera error: {exc}")
        finally:
            try:
                cam.stop()
            finally:
                close = getattr(cam, "close", None)
                if close:
                    close()
            with self._lock:
                self._frame = None
                self._worker = None

    def pause_and_close(self, seconds=20.0):
        """Release the camera NOW and refuse to reopen for `seconds`.

        Called right before a capture is spawned: the grace window covers the
        capture process's own camera open; once capture is running, the GCS
        route guard (status.capturing) takes over refusing streams.
        """
        with self._lock:
            self._pause_until = self._time() + seconds
            evt, worker = self._stop_evt, self._worker
        if worker is not None and worker.is_alive():
            evt.set()
            worker.join(timeout=10)
