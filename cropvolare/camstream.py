"""
Shared live-view camera session for the GCS viewfinder.

The old per-snapshot approach opened and closed the camera for every preview
frame (~3-4 s each on a Zero 2 W - the "finicky feed"). This keeps ONE camera
session open while at least one viewfinder stream is watching, so frames cost
a capture_array call instead of a full pipeline setup, and closes it the
moment the last viewer leaves.

Safety property (non-negotiable): the session must never hold the camera when
a flight capture wants it. pause_and_close() releases the camera immediately
and blocks re-opening for a grace window - the GCS start endpoint calls it
before spawning capture, and in-flight stream generators see frame() -> None
and end themselves.
"""

import threading
import time


class CameraSession:
    """Refcounted camera holder for viewfinder streams.

    camera_factory/time_fn are injectable for hardware-free tests. The real
    factory builds ndvi.create_camera at a preview-friendly resolution.
    """

    def __init__(self, camera_factory=None, resolution=(1152, 648),
                 warmup=0.8, time_fn=time.time):
        self._factory = camera_factory
        self._resolution = resolution
        self._warmup = warmup
        self._time = time_fn
        self._lock = threading.RLock()
        self._cam = None
        self._clients = 0
        self._pause_until = 0.0

    def _default_factory(self):
        from .ndvi import create_camera
        return create_camera(resolution=self._resolution)

    def acquire(self):
        """Register a viewer; opens the camera on the first one.

        Raises RuntimeError while paused (capture owns/is taking the camera).
        """
        with self._lock:
            if self._time() < self._pause_until:
                raise RuntimeError("camera reserved for capture")
            if self._cam is None:
                factory = self._factory or self._default_factory
                cam = factory()
                cam.start()
                time.sleep(self._warmup)
                self._cam = cam
            self._clients += 1

    def release(self):
        """Unregister a viewer; closes the camera when the last one leaves."""
        with self._lock:
            self._clients = max(0, self._clients - 1)
            if self._clients == 0:
                self._close_locked()

    def frame(self):
        """One BGR frame, or None if the session is closed/paused (viewer
        generators treat None as 'stop streaming now')."""
        with self._lock:
            if self._cam is None or self._time() < self._pause_until:
                return None
            from .ndvi import capture_frame
            return capture_frame(self._cam)

    def pause_and_close(self, seconds=20.0):
        """Release the camera NOW and refuse to reopen for `seconds`.

        Called right before a capture is spawned: the grace window covers the
        capture process's own camera open; once capture is running, the GCS
        route guard (status.capturing) takes over refusing streams.
        """
        with self._lock:
            self._pause_until = self._time() + seconds
            self._close_locked()

    def _close_locked(self):
        cam, self._cam = self._cam, None
        if cam is not None:
            try:
                cam.stop()
            finally:
                close = getattr(cam, "close", None)
                if close:
                    close()

    @property
    def open(self):
        with self._lock:
            return self._cam is not None
