#!/usr/bin/env python3
"""Calibrate the red-leakage correction (k) from a grey-card photo.

With the red filter mounted, photograph a neutral grey card filling the frame
(same lighting you'll fly in), then:

    # capture a card photo on the Pi first:
    python scripts/capture_flight.py -o calib --count 1

    # solve k and save it into config/default.json:
    python scripts/calibrate.py --input calib/frame_0000.jpg --write

    # optional sanity check against a healthy plant photo (expect ~0.4-0.6):
    python scripts/calibrate.py --input calib/frame_0000.jpg --plant plant.jpg

A grey card reflects red and NIR equally, so the correct k is solvable
directly - no trial-and-error sweep needed. After --write, every capture and
process_flight run picks the value up from the config automatically.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cropvolare.ndvi import compute_ndvi_from_image, load_image, solve_leakage_k

DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config", "default.json",
)


def main():
    p = argparse.ArgumentParser(description="Solve leakage_k from a grey card")
    p.add_argument("-i", "--input", required=True,
                   help="photo of a grey card (red filter mounted)")
    p.add_argument("--gamma", type=float, default=0.8)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--write", action="store_true",
                   help="save the solved k into the config file")
    p.add_argument("--plant", help="optional healthy-plant photo to sanity-check")
    args = p.parse_args()

    grey = load_image(args.input)
    k = solve_leakage_k(grey, gamma=args.gamma)
    k = round(k, 3)
    print(f"solved leakage_k = {k}")

    check = compute_ndvi_from_image(grey, gamma=args.gamma, leakage_k=k)
    print(f"grey card mean NDVI with k={k}: {float(check.mean()):.4f} (target ~0)")

    if args.plant:
        plant = load_image(args.plant)
        ndvi = compute_ndvi_from_image(plant, gamma=args.gamma, leakage_k=k)
        mean = float(ndvi.mean())
        verdict = "looks right" if 0.2 <= mean <= 0.8 else "outside 0.2-0.8, re-check filter/lighting"
        print(f"plant mean NDVI: {mean:.4f} ({verdict})")

    if args.write:
        with open(args.config) as f:
            cfg = json.load(f)
        cfg.setdefault("ndvi", {})["leakage_k"] = k
        with open(args.config, "w") as f:
            json.dump(cfg, f, indent=2)
            f.write("\n")
        print(f"wrote leakage_k={k} to {args.config}")
    else:
        print("(re-run with --write to save it into the config)")


if __name__ == "__main__":
    main()
