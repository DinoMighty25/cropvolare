#!/usr/bin/env python3
"""Burst flight capture: save a sequence of (optionally geotagged) JPEGs.

This is the producer for the processing pipeline. Run it during a flight; it
captures a frame every few seconds, saves it as a JPEG, and - if a GPS is
connected - stamps the current fix into the JPEG's EXIF. The resulting folder
is exactly what scripts/process_flight.py consumes on the laptop.

    # capture 40 frames, one every 2.5 s, into a flight folder
    python scripts/capture_flight.py --outdir flights/today --count 40

    # capture until Ctrl+C, tagging each frame from a serial GPS
    python scripts/capture_flight.py --outdir flights/today --count 0 \
        --gps-port /dev/serial0

Without --gps-port, frames are saved untagged; tag them afterwards with
scripts/tag_gps.py (e.g. against a recorded track), then run process_flight.py.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cropvolare import exposure, geo
from cropvolare.gpsread import GpsReader
from cropvolare.ndvi import capture_frame, create_camera, lock_exposure

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(REPO, "config", "default.json")


def load_config(path):
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def open_camera(cfg, preset_name=None, warmup=2.0, log_fn=print):
    """Create and start the camera, returning (cam, capture_meta dict).

    With preset_name, the exposure/gain/colour-gains come from the config and are
    hard-locked before the first frame - that is what makes this flight
    comparable to other days. Without it, auto-exposure settles and is frozen for
    the session only (fine for a test hop, useless for cross-flight comparison),
    and the metadata records locked="auto" so compare_flights.py can say so.
    """
    cam_cfg = cfg.get("camera", {})
    resolution = tuple(cam_cfg.get("resolution", (2304, 1296)))
    colour_gains = tuple(cam_cfg.get("colour_gains", (1.0, 1.0)))

    preset = None
    if preset_name:
        preset = exposure.get_preset(cfg, preset_name)   # raises with a helpful list
        if preset.get("colour_gains"):
            colour_gains = tuple(preset["colour_gains"])

    cam = None
    for attempt in range(3):
        try:
            cam = create_camera(
                resolution=resolution, colour_gains=colour_gains,
                exposure_us=(preset["exposure_us"] if preset else None))
            break
        except Exception as exc:  # noqa: BLE001 - e.g. viewfinder still releasing it
            if attempt == 2:
                raise
            log_fn(f"camera busy ({exc}); retrying in 3s ...")
            time.sleep(3)

    cam.start()

    if preset:
        # create_camera() sets ExposureTime but not gain; set both explicitly so
        # nothing is left to the ISP's discretion.
        cam.set_controls({
            "AeEnable": False,
            "ExposureTime": int(preset["exposure_us"]),
            "AnalogueGain": float(preset["analogue_gain"]),
        })
        time.sleep(max(warmup, 1.0))   # let the locked values take effect
        exp = int(preset["exposure_us"])
        gain = float(preset["analogue_gain"])
        log_fn(f"exposure LOCKED to preset {preset_name!r}: "
               f"{exp} us, gain {gain:.2f}")
        meta = exposure.capture_meta(preset_name, exp, gain,
                                     colour_gains=colour_gains, locked="preset")
    else:
        time.sleep(warmup)               # auto-exposure settles on the scene
        exp, gain = lock_exposure(cam)   # then freeze for the whole flight
        log_fn(f"exposure auto-locked: {exp} us, gain {gain:.2f}")
        log_fn("WARNING: no --preset, so this flight is NOT comparable to other "
               "flights. Fine for a test hop; not for data.")
        meta = exposure.capture_meta(None, exp, gain,
                                     colour_gains=colour_gains, locked="auto")

    meta["resolution"] = list(resolution)
    return cam, meta


def _default_save(path, frame):
    import cv2
    cv2.imwrite(path, frame)
    # stamp the capture time into EXIF: tag_gps.py's track-matching mode and
    # ODM stitching both need DateTimeOriginal (file mtimes don't survive
    # copies). Never let metadata stamping break a flight.
    try:
        from datetime import datetime

        import piexif
        ts = datetime.now().strftime("%Y:%m:%d %H:%M:%S")
        exif = {"0th": {}, "Exif": {piexif.ExifIFD.DateTimeOriginal: ts},
                "GPS": {}, "1st": {}, "thumbnail": None}
        piexif.insert(piexif.dump(exif), path)
    except Exception:  # noqa: BLE001 - capture must go on without EXIF
        pass


def run_capture(outdir, capture_fn, n_frames, interval, gps_fn=None,
                save_fn=None, tag_fn=None, sleep_fn=None, log_fn=print,
                stop_fn=None, sync_every=5, sync_fn=None, prefix="frame"):
    """Capture loop core. Injected callables make it testable without hardware.

    capture_fn() -> BGR frame; gps_fn() -> {'lat','lon','alt'} or None.
    stop_fn() -> True ends the loop cleanly (used by fly.py via a STOP file).
    Every sync_every frames the OS write cache is flushed (sync_fn, default
    os.sync) so cutting power after landing loses at most a few frames - a
    real flight ended with 16 zero-byte JPEGs because the X306 switch was
    flipped while ~30s of frames sat unflushed in the page cache.
    n_frames=None runs until interrupted by the caller. Returns saved paths.
    """
    save_fn = save_fn or _default_save
    tag_fn = tag_fn or geo.write_gps
    sleep_fn = sleep_fn or time.sleep
    if sync_fn is None:
        sync_fn = getattr(os, "sync", None)  # not available on Windows
    os.makedirs(outdir, exist_ok=True)

    saved = []
    i = 0
    while n_frames is None or i < n_frames:
        if stop_fn is not None and stop_fn():
            log_fn("stop requested - ending capture")
            break
        frame = capture_fn()
        path = os.path.join(outdir, f"{prefix}_{i:04d}.jpg")
        save_fn(path, frame)

        tagged = False
        if gps_fn is not None:
            fix = gps_fn()
            if fix:
                tag_fn(path, fix["lat"], fix["lon"], fix.get("alt"))
                tagged = True

        saved.append(path)
        if sync_fn is not None and sync_every and (i + 1) % sync_every == 0:
            sync_fn()  # bound abrupt-power-off loss to the last few frames
        log_fn(f"[{i + 1}{'/' + str(n_frames) if n_frames else ''}] "
               f"{os.path.basename(path)}{' +gps' if tagged else ''}")
        i += 1
        if n_frames is None or i < n_frames:
            sleep_fn(interval)

    return saved


def main():
    parser = argparse.ArgumentParser(description="Burst flight capture")
    parser.add_argument("-o", "--outdir", required=True,
                        help="folder to write JPEGs into")
    parser.add_argument("--count", type=int, default=40,
                        help="number of frames (0 = until Ctrl+C)")
    parser.add_argument("--interval", type=float, default=2.5,
                        help="seconds between frames")
    parser.add_argument("--warmup", type=float, default=2.0,
                        help="seconds to let the sensor settle before the first frame")
    parser.add_argument("--gps-port", default=None,
                        help="serial GPS port (e.g. /dev/serial0); omit to skip tagging")
    parser.add_argument("--preset", default=None,
                        help="named exposure preset from the config (e.g. full_sun). "
                             "REQUIRED for data flights: without it auto-exposure "
                             "settles per flight and flights are not comparable")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()

    n_frames = None if args.count <= 0 else args.count

    gps = None
    if args.gps_port:
        os.makedirs(args.outdir, exist_ok=True)
        print(f"starting GPS reader on {args.gps_port} ...")
        gps = GpsReader(port=args.gps_port,
                        track_path=os.path.join(args.outdir, "track.csv")).start()

    print("initializing camera ...")
    cfg = load_config(args.config)
    try:
        cam, meta = open_camera(cfg, preset_name=args.preset, warmup=args.warmup)
    except exposure.PresetError as exc:
        print(f"error: {exc}")
        return 1

    os.makedirs(args.outdir, exist_ok=True)
    meta_path = exposure.write_capture_meta(args.outdir, meta)
    print(f"wrote {os.path.basename(meta_path)} (exposure audit trail)")

    print(f"capturing {'until Ctrl+C' if n_frames is None else n_frames} "
          f"frame(s), one every {args.interval}s -> {args.outdir}")

    # a STOP file in the output folder ends capture cleanly - lets fly.py stop
    # a detached run even after the SSH session that started it is gone
    stop_path = os.path.join(args.outdir, "STOP")
    if os.path.exists(stop_path):
        os.remove(stop_path)  # clear a stale request from a previous run
    print(f"to stop: Ctrl+C, or create {stop_path}")

    try:
        saved = run_capture(
            args.outdir,
            capture_fn=lambda: capture_frame(cam),
            n_frames=n_frames,
            interval=args.interval,
            gps_fn=(gps.latest if gps else None),
            stop_fn=lambda: os.path.exists(stop_path),
        )
        print(f"done: {len(saved)} frame(s) saved")
    except KeyboardInterrupt:
        print("\nstopped by user")
    finally:
        cam.stop()
        if gps:
            gps.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
