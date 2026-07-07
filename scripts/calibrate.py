#!/usr/bin/env python3
"""Calibrate the NDVI rig: red-leakage constant and/or flat-field gain map.

LEAKAGE (grey card). With the red filter mounted, photograph a neutral grey
card filling the frame (same lighting you'll fly in), then:

    python scripts/capture_flight.py -o calib --count 1
    python scripts/calibrate.py --input calib/frame_0000.jpg --write

    # optional sanity check against a healthy plant photo (expect ~0.4-0.6):
    python scripts/calibrate.py --input calib/frame_0000.jpg --plant plant.jpg

A grey card reflects red and NIR equally, so the correct k is solvable
directly - no trial-and-error sweep needed.

FLAT-FIELD (white target). Photograph a plain white sheet filling the frame in
even shade (~20 frames), then build the per-channel lens-shading gain map:

    python scripts/capture_flight.py -o calib_flat --count 20 --interval 0.5
    python scripts/calibrate.py --flatfield-dir calib_flat --write

The gain map removes the radial NDVI "bullseye" caused by the lens shading the
NIR and red channels unequally. After --write, every capture and
process_flight run picks both calibrations up from the config automatically.
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cropvolare.ndvi import (
    build_flatfield,
    compute_ndvi_from_image,
    load_image,
    save_flatfield,
    solve_leakage_k,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(REPO, "config", "default.json")
DEFAULT_GAIN_OUT = os.path.join("calibration", "gain.npy")


def _update_config(path, section, key, value):
    with open(path) as f:
        cfg = json.load(f)
    cfg.setdefault(section, {})[key] = value
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")


def run_leakage(args):
    grey = load_image(args.input)
    k = round(solve_leakage_k(grey, gamma=args.gamma), 3)
    print(f"solved leakage_k = {k}")

    check = compute_ndvi_from_image(grey, gamma=args.gamma, leakage_k=k)
    print(f"grey card mean NDVI with k={k}: {float(check.mean()):.4f} (target ~0)")

    if args.plant:
        plant = load_image(args.plant)
        ndvi = compute_ndvi_from_image(plant, gamma=args.gamma, leakage_k=k)
        mean = float(ndvi.mean())
        verdict = ("looks right" if 0.2 <= mean <= 0.8
                   else "outside 0.2-0.8, re-check filter/lighting")
        print(f"plant mean NDVI: {mean:.4f} ({verdict})")

    if args.write:
        _update_config(args.config, "ndvi", "leakage_k", k)
        print(f"wrote leakage_k={k} to {args.config}")
    else:
        print("(re-run with --write to save it into the config)")


def run_flatfield(args):
    paths = sorted({
        p for pat in ("*.jpg", "*.jpeg", "*.JPG", "*.png")
        for p in glob.glob(os.path.join(args.flatfield_dir, pat))
    })  # set: Windows globs are case-insensitive, *.jpg and *.JPG both match
    if not paths:
        raise SystemExit(f"no images found in {args.flatfield_dir}")
    print(f"building flat-field from {len(paths)} frame(s) ...")
    frames = [load_image(p) for p in paths]
    gain = build_flatfield(frames)

    h, w = gain.shape[:2]
    center = gain[h // 2 - 5:h // 2 + 5, w // 2 - 5:w // 2 + 5].mean()
    corner = gain[:10, :10].mean()
    print(f"corner/center gain ratio: {corner / center:.2f} "
          f"(1.00 = no vignetting; the further from 1, the stronger)")

    gain_out = args.gain_out
    if not os.path.isabs(gain_out):
        gain_out = os.path.join(REPO, gain_out)
    save_flatfield(gain, gain_out)
    print(f"saved gain map to {gain_out}")

    if args.write:
        # forward slashes: the config is shared across Windows and the Pi
        _update_config(args.config, "calibration", "flatfield_path",
                       args.gain_out.replace("\\", "/"))
        print(f"wrote calibration.flatfield_path to {args.config}")
    else:
        print("(re-run with --write to record it in the config)")


def main():
    p = argparse.ArgumentParser(description="Solve leakage_k and/or flat-field")
    p.add_argument("-i", "--input",
                   help="photo of a grey card (leakage calibration)")
    p.add_argument("--flatfield-dir",
                   help="folder of white-target frames (flat-field calibration)")
    p.add_argument("--gain-out", default=DEFAULT_GAIN_OUT,
                   help="where to save the gain map (default calibration/gain.npy)")
    p.add_argument("--gamma", type=float, default=0.8)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--write", action="store_true",
                   help="save results into the config file")
    p.add_argument("--plant", help="optional healthy-plant photo to sanity-check")
    args = p.parse_args()

    if not args.input and not args.flatfield_dir:
        p.error("give --input (grey card) and/or --flatfield-dir (white target)")

    if args.flatfield_dir:
        run_flatfield(args)
    if args.input:
        run_leakage(args)


if __name__ == "__main__":
    main()
