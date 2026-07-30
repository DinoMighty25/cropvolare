#!/usr/bin/env python3
"""Measure a named exposure preset in real light, once, and save it to the config.

Run this ON THE PI, outdoors, in the light you intend to fly in, with the camera
aimed at sunlit vegetation. It lets auto-exposure settle on the scene, reads the
values it chose, checks nothing is clipping, and writes them to the config under
a name. Every later flight passes --preset <name> and hard-locks those numbers,
which is what makes flights comparable across days.

    # measure in bright sun, aimed at the crop
    python scripts/set_exposure_preset.py full_sun --note "clear, near noon"

    # and one for grey days
    python scripts/set_exposure_preset.py overcast

    # force a specific exposure instead of measuring (e.g. to back off clipping)
    python scripts/set_exposure_preset.py full_sun --exposure 900

    # see what is already saved
    python scripts/set_exposure_preset.py --list

Do this ONCE per lighting condition per season. Redo it if the lens, filter, or
resolution changes - and remember that changing a preset invalidates the
grey-card leakage constant and the flat-field map, so re-run calibrate.py after.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cropvolare import exposure

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(REPO, "config", "default.json")


def load_config(path):
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def measure(cfg, settle=4.0, forced_exposure=None, forced_gain=None):
    """Settle the camera on the current scene and read back what it chose.

    Returns (exposure_us, analogue_gain, colour_gains, saturation_report).
    Imports picamera2 lazily so --list works on a laptop.
    """
    from cropvolare.ndvi import capture_frame, create_camera, lock_exposure

    cam_cfg = cfg.get("camera", {})
    resolution = tuple(cam_cfg.get("resolution", (2304, 1296)))
    # Use the config's colour gains, not create_camera's neutral default: the
    # preset has to describe the camera as it will actually fly.
    colour_gains = tuple(cam_cfg.get("colour_gains", (1.0, 1.0)))

    cam = create_camera(resolution=resolution, colour_gains=colour_gains,
                        exposure_us=forced_exposure)
    cam.start()
    try:
        time.sleep(settle)
        if forced_exposure is None:
            exp, gain = lock_exposure(cam)
        else:
            exp = int(forced_exposure)
            gain = float(forced_gain if forced_gain is not None else 1.0)
            cam.set_controls({"AeEnable": False, "ExposureTime": exp,
                              "AnalogueGain": gain})
            time.sleep(0.5)
        # discard one frame: the first after a control change can still carry the
        # previous settings through the ISP pipeline
        capture_frame(cam)
        time.sleep(0.3)
        frame = capture_frame(cam)
    finally:
        cam.stop()

    return exp, gain, colour_gains, exposure.channel_saturation(frame)


def main():
    p = argparse.ArgumentParser(
        description="Measure and save a named exposure preset")
    p.add_argument("name", nargs="?",
                   help="preset name, e.g. full_sun / overcast / hazy")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--note", default=None,
                   help="sky conditions when measured (recommended)")
    p.add_argument("--settle", type=float, default=4.0,
                   help="seconds to let auto-exposure settle (default 4)")
    p.add_argument("--exposure", type=int, default=None,
                   help="force this exposure in us instead of measuring")
    p.add_argument("--gain", type=float, default=None,
                   help="force this analogue gain (with --exposure)")
    p.add_argument("--list", action="store_true",
                   help="list saved presets and exit")
    p.add_argument("--dry-run", action="store_true",
                   help="measure and report, but do not write the config")
    args = p.parse_args()

    cfg = load_config(args.config)

    if args.list:
        presets = exposure.load_presets(cfg)
        if not presets:
            print("no exposure presets saved yet")
            print(f"measure one: python {os.path.relpath(__file__)} full_sun")
            return 0
        print(f"presets in {args.config}:")
        for name in sorted(presets):
            try:
                exposure.validate(presets[name], name)
                print(f"  {name}: {exposure.describe(presets[name])}")
            except exposure.PresetError as exc:
                print(f"  {name}: INVALID - {exc}")
        return 0

    if not args.name:
        p.error("a preset name is required (or use --list)")

    print(f"measuring preset {args.name!r} ...")
    print("  point the camera at SUNLIT VEGETATION, in the light you will fly in")
    try:
        exp, gain, colour_gains, sat = measure(
            cfg, settle=args.settle,
            forced_exposure=args.exposure, forced_gain=args.gain)
    except RuntimeError as exc:
        print(f"error: {exc}")
        print("this script has to run on the Pi with the camera attached")
        return 1

    level, message = exposure.clipping_verdict(sat)
    print(f"  measured: {exp} us, gain {gain:.2f}, "
          f"colour_gains ({colour_gains[0]:.2f}, {colour_gains[1]:.2f})")
    print(f"  {message}")

    if level == "bad":
        suggested = max(exposure.MIN_EXPOSURE_US, int(exp * 0.6))
        print()
        print(f"NOT SAVED. Re-run with: --exposure {suggested}")
        return 1

    preset = exposure.make_preset(exp, gain, colour_gains=colour_gains,
                                  note=args.note)

    if args.dry_run:
        print(f"\ndry run - not saved. Would store: {exposure.describe(preset)}")
        return 0

    exposure.save_preset(args.config, args.name, preset)
    print(f"\nsaved preset {args.name!r} to {args.config}")
    print(f"  {exposure.describe(preset)}")
    print()
    print("NEXT - this preset invalidates the old calibration constants:")
    print("  1. python scripts/calibrate.py --write        (grey card, leakage_k)")
    print("  2. python scripts/calibrate.py --flatfield-dir <white frames> --write")
    print(f"  3. fly with: python scripts/fly.py --preset {args.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
