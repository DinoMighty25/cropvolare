#!/usr/bin/env python3
"""Capture an image and compute NDVI from the Raspberry Pi NoIR camera.

Usage:
    python scripts/capture_ndvi.py
    python scripts/capture_ndvi.py --output output/ndvi_map.png
    python scripts/capture_ndvi.py --no-save --print-zones
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

# Allow running from repo root
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
    parser.add_argument(
        "--output", "-o",
        default="output/ndvi_map.png",
        help="Path to save the NDVI heatmap image (default: output/ndvi_map.png)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=64,
        help="Zone grid block size in pixels (default: 64)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving the heatmap image",
    )
    parser.add_argument(
        "--print-zones",
        action="store_true",
        help="Print zone classification results as JSON",
    )
    args = parser.parse_args()

    print("[Cropvolare] Initializing camera...")
    cam = create_camera()

    print("[Cropvolare] Capturing image...")
    image = capture_image(cam)

    print("[Cropvolare] Computing NDVI...")
    ndvi = compute_ndvi_from_image(image)

    mean_ndvi = float(ndvi.mean())
    print(f"[Cropvolare] Mean NDVI: {mean_ndvi:.4f}")

    zones = classify_zones(ndvi, block_size=args.block_size)
    stressed = [z for z in zones if z["status"] == "stressed"]
    print(f"[Cropvolare] Zones: {len(zones)} total, {len(stressed)} stressed")

    if args.print_zones:
        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mean_ndvi": round(mean_ndvi, 4),
            "total_zones": len(zones),
            "stressed_zones": len(stressed),
            "zones": zones,
        }
        print(json.dumps(result, indent=2))

    if not args.no_save:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        save_ndvi_image(ndvi, args.output)
        print(f"[Cropvolare] Heatmap saved to {args.output}")


if __name__ == "__main__":
    main()
