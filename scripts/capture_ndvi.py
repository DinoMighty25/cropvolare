#!/usr/bin/env python3
"""Capture (or load) an image and compute NDVI.

On the Pi:        python scripts/capture_ndvi.py
On any laptop:    python scripts/capture_ndvi.py --input photo.jpg
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cropvolare.ndvi import (
    apply_flatfield,
    capture_image,
    classify_zones,
    compute_ndvi_from_image,
    compute_vari,
    create_camera,
    load_flatfield,
    load_image,
    save_ndvi_image,
    save_ndvi_tiff,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(REPO, "config", "default.json")


def load_config(path):
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(description="Capture and compute NDVI")
    parser.add_argument("-i", "--input",
                        help="process an existing image instead of capturing")
    parser.add_argument("-o", "--output", default="output/ndvi_map.png")
    parser.add_argument("--tiff", help="also save a 16-bit TIFF to this path")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--leakage-k", type=float)
    parser.add_argument("--block-size", type=int)
    parser.add_argument("--vari", action="store_true",
                        help="also report VARI (use on an unfiltered photo)")
    parser.add_argument("--flatfield",
                        help="gain map .npy (default: calibration.flatfield_path "
                             "from the config)")
    parser.add_argument("--no-flatfield", action="store_true",
                        help="disable flat-field correction")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--print-zones", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ndvi_cfg = cfg.get("ndvi", {})
    gamma = args.gamma if args.gamma is not None else ndvi_cfg.get("gamma", 0.8)
    leakage_k = (args.leakage_k if args.leakage_k is not None
                 else ndvi_cfg.get("leakage_k", 0.6))
    block_size = (args.block_size if args.block_size is not None
                  else ndvi_cfg.get("block_size", 64))

    if args.input:
        print(f"loading {args.input}...")
        image = load_image(args.input)
    else:
        print("initializing camera (controls locked)...")
        cam_cfg = cfg.get("camera", {})
        resolution = tuple(cam_cfg.get("resolution", (2304, 1296)))
        cam = create_camera(resolution=resolution)
        print("capturing...")
        image = capture_image(cam)
    print(f"got image: {image.shape}, {image.dtype}")

    if not args.no_flatfield:
        ff_path = args.flatfield or cfg.get("calibration", {}).get("flatfield_path")
        if ff_path:
            full = ff_path if os.path.isabs(ff_path) else os.path.join(REPO, ff_path)
            if os.path.exists(full):
                image = apply_flatfield(image, load_flatfield(full))
                print(f"flat-field applied ({ff_path})")
            else:
                print(f"warning: gain map missing ({full}) - no flat-field")

    print(f"computing ndvi (gamma={gamma}, leakage_k={leakage_k})...")
    ndvi = compute_ndvi_from_image(image, gamma=gamma, leakage_k=leakage_k)
    print(f"ndvi range: {ndvi.min():.4f} to {ndvi.max():.4f}")
    print(f"mean NDVI: {float(ndvi.mean()):.4f}")

    if args.vari:
        vari = compute_vari(image)
        print(f"mean VARI (cross-check): {float(vari.mean()):.4f}")

    print("classifying zones...")
    zones = classify_zones(ndvi, block_size=block_size)
    stressed = [z for z in zones if z["status"] == "stressed"]
    print(f"{len(zones)} zones, {len(stressed)} stressed")

    if args.print_zones:
        print(json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mean_ndvi": round(float(ndvi.mean()), 4),
            "total_zones": len(zones),
            "stressed_zones": len(stressed),
            "zones": zones,
        }, indent=2))

    if not args.no_save:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        save_ndvi_image(ndvi, args.output)
        print(f"saved PNG to {args.output}")
        if args.tiff:
            os.makedirs(os.path.dirname(args.tiff) or ".", exist_ok=True)
            save_ndvi_tiff(ndvi, args.tiff)
            print(f"saved 16-bit TIFF to {args.tiff}")


if __name__ == "__main__":
    main()
