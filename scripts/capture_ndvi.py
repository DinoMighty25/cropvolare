#!/usr/bin/env python3
"""Quick script to capture an image and compute NDVI on the Pi."""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cropvolare.ndvi import (
    capture_image,
    classify_zones,
    compute_ndvi_from_image,
    create_camera,
    save_ndvi_image,
)


def main():
    parser = argparse.ArgumentParser(description="Capture and compute NDVI")
    parser.add_argument("-o", "--output", default="output/ndvi_map.png")
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--print-zones", action="store_true")
    args = parser.parse_args()

    print("initializing camera...")
    cam = create_camera()

    print("capturing (waiting for auto-exposure)...")
    image = capture_image(cam)
    print(f"got image: {image.shape}, {image.dtype}")

    print("computing ndvi...")
    ndvi = compute_ndvi_from_image(image)
    print(f"ndvi range: {ndvi.min():.4f} to {ndvi.max():.4f}")

    mean_ndvi = float(ndvi.mean())
    print(f"mean NDVI: {mean_ndvi:.4f}")

    print("classifying zones...")
    zones = classify_zones(ndvi, block_size=args.block_size)
    stressed = [z for z in zones if z["status"] == "stressed"]
    print(f"{len(zones)} zones, {len(stressed)} stressed")

    if args.print_zones:
        print(json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mean_ndvi": round(mean_ndvi, 4),
            "total_zones": len(zones),
            "stressed_zones": len(stressed),
            "zones": zones,
        }, indent=2))

    if not args.no_save:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        save_ndvi_image(ndvi, args.output)
        print(f"saved to {args.output}")


if __name__ == "__main__":
    main()
