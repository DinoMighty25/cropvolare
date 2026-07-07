"""
Serial GPS reader shared by flight capture and the GCS web app.

Pi-only at runtime (needs pynmea2 + pyserial), but import-safe everywhere.
Keeps the latest fix in memory and optionally appends every fix to a
track CSV (timestamp,lat,lon,alt) - the flight's ground-truth trail for
post-flight EXIF tagging (tag_gps.py) and the GCS live-coverage view.
"""

import os
import time


class GpsReader:
    """Background thread that keeps the latest GPS fix from a serial NMEA source.

    Returns None from latest() until a fix arrives, so callers save untagged
    frames rather than blocking. track_path, if given, gets one CSV row per
    accepted fix.
    """

    def __init__(self, port="/dev/serial0", baud=9600, track_path=None):
        self.port = port
        self.baud = baud
        self.track_path = track_path
        self._latest = None
        self._track = []          # in-memory breadcrumb for the GCS
        self._stop = False
        self._thread = None

    def start(self):
        import threading
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _accept(self, fix):
        """Record a fix (also used directly by tests - no serial needed)."""
        self._latest = fix
        self._track.append(fix)
        if self.track_path:
            new = not os.path.exists(self.track_path)
            with open(self.track_path, "a") as f:
                if new:
                    f.write("time,lat,lon,alt\n")
                f.write(f"{fix.get('time', '')},{fix['lat']},{fix['lon']},"
                        f"{fix.get('alt') if fix.get('alt') is not None else ''}\n")

    def _run(self):
        try:
            import pynmea2
            import serial
        except ImportError:
            print("GPS: pynmea2/pyserial not installed; frames stay untagged")
            return
        try:
            ser = serial.Serial(self.port, self.baud, timeout=1.0)
        except Exception as exc:  # noqa: BLE001 - report and bail, don't crash flight
            print(f"GPS: could not open {self.port}: {exc}")
            return
        while not self._stop:
            line = ser.readline().decode("ascii", errors="ignore").strip()
            if not line.startswith("$"):
                continue
            try:
                msg = pynmea2.parse(line)
            except Exception:  # noqa: BLE001 - skip malformed sentences
                continue
            lat = getattr(msg, "latitude", None)
            lon = getattr(msg, "longitude", None)
            if lat and lon:
                alt = getattr(msg, "altitude", None)
                self._accept({
                    "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "lat": float(lat), "lon": float(lon),
                    "alt": float(alt) if alt else None,
                })
        ser.close()

    def latest(self):
        return self._latest

    def track(self, since=0):
        """Breadcrumb fixes from index `since` on (GCS backfill after WiFi drop)."""
        return self._track[since:]

    def stop(self):
        self._stop = True
