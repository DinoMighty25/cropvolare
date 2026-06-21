#!/usr/bin/env python3
"""Stamp GPS coordinates onto captured JPEGs (Raspberry Pi side).

Two modes:

  # tag one freshly captured JPEG with the current live fix
  python scripts/tag_gps.py --image capture.jpg

  # tag a whole folder with a single fix (quick bench/test use)
  python scripts/tag_gps.py --dir flights/today/ --lat 40.1 --lon -88.2

Processing happens on the laptop, so this stays deliberately thin. For a real
flight you'd tag each image inline right after capture, or match a recorded GPS
track to image timestamps. Live reads use pynmea2 + pyserial (Pi-only deps).

EXIF GPS needs a JPEG on disk - note that the camera helper in cropvolare.ndvi
returns a numpy array, so the capture step must save JPEGs for this to tag.
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cropvolare import geo


def read_live_fix(port="/dev/serial0", baud=9600, timeout=5.0):
    """Block until one valid GGA/RMC fix arrives; return (lat, lon, alt)."""
    import pynmea2
    import serial

    ser = serial.Serial(port, baud, timeout=timeout)
    try:
        for _ in range(200):  # ~ a few seconds of sentences
            line = ser.readline().decode("ascii", errors="ignore").strip()
            if not line.startswith("$"):
                continue
            try:
                msg = pynmea2.parse(line)
            except pynmea2.ParseError:
                continue
            lat = getattr(msg, "latitude", None)
            lon = getattr(msg, "longitude", None)
            if lat and lon:
                alt = getattr(msg, "altitude", None)
                return float(lat), float(lon), float(alt) if alt else None
    finally:
        ser.close()
    raise RuntimeError("no GPS fix received - check antenna / satellite lock")


def main():
    parser = argparse.ArgumentParser(description="Write GPS EXIF onto JPEGs")
    parser.add_argument("--image", help="single JPEG to tag")
    parser.add_argument("--dir", help="folder of JPEGs to tag")
    parser.add_argument("--lat", type=float, help="latitude (skip live GPS read)")
    parser.add_argument("--lon", type=float, help="longitude (skip live GPS read)")
    parser.add_argument("--alt", type=float)
    parser.add_argument("--port", default="/dev/serial0")
    args = parser.parse_args()

    if not args.image and not args.dir:
        parser.error("give --image or --dir")

    if args.lat is not None and args.lon is not None:
        lat, lon, alt = args.lat, args.lon, args.alt
    else:
        print(f"reading live GPS fix from {args.port} ...")
        lat, lon, alt = read_live_fix(port=args.port)
        print(f"  fix: {lat:.6f}, {lon:.6f}")

    targets = [args.image] if args.image else sorted(
        glob.glob(os.path.join(args.dir, "*.jpg"))
        + glob.glob(os.path.join(args.dir, "*.jpeg")))

    for path in targets:
        geo.write_gps(path, lat, lon, alt=alt)
        print(f"  tagged {path}")


if __name__ == "__main__":
    main()
