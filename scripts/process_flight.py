#!/usr/bin/env python3
"""Process a folder of geotagged drone photos into a farmer-ready field report.

    python scripts/process_flight.py --input flights/2026-06-21/ \
           --outdir output/2026-06-21/

Produces, in --outdir:
    field.geojson   the durable per-photo NDVI data (also the ODM hand-off)
    heatmap.png     the colorized field NDVI overlay
    report.pdf      one-page farmer report
    map.html        standalone interactive web map
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cropvolare import batch, field, fieldmap, report, webmap

DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config", "default.json",
)


def load_config(path):
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(description="Process a flight into a field report")
    parser.add_argument("-i", "--input", required=True,
                        help="directory of geotagged photos")
    parser.add_argument("-o", "--outdir", default="output/flight")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--cell-meters", type=float)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--leakage-k", type=float)
    parser.add_argument("--block-size", type=int)
    parser.add_argument("--top-n", type=int)
    parser.add_argument("--no-overlays", action="store_true",
                        help="skip writing a per-photo NDVI overlay for each image")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ndvi_cfg = cfg.get("ndvi", {})
    field_cfg = cfg.get("field", {})

    gamma = args.gamma if args.gamma is not None else ndvi_cfg.get("gamma", 0.8)
    leakage_k = (args.leakage_k if args.leakage_k is not None
                 else ndvi_cfg.get("leakage_k", 0.6))
    block_size = (args.block_size if args.block_size is not None
                  else ndvi_cfg.get("block_size", 64))
    cell_meters = (args.cell_meters if args.cell_meters is not None
                   else field_cfg.get("cell_meters", 20.0))
    top_n = args.top_n if args.top_n is not None else field_cfg.get("top_n_problems", 5)
    healthy = field_cfg.get("healthy_threshold", 0.5)
    stressed = field_cfg.get("stressed_threshold", 0.3)
    opacity = field_cfg.get("overlay_opacity", 0.6)
    # Per-photo NDVI overlays default on: they're the gallery content for
    # no-GPS flights and useful output otherwise. Disable with --no-overlays.
    write_overlays = not args.no_overlays

    os.makedirs(args.outdir, exist_ok=True)
    now = datetime.now(timezone.utc)
    overlay_dir = os.path.join(args.outdir, "overlays") if write_overlays else None

    print(f"ingesting photos from {args.input} ...")
    fc = batch.process_directory(
        args.input, gamma=gamma, leakage_k=leakage_k, block_size=block_size,
        overlay_dir=overlay_dir,
        flight_date=now.date().isoformat(),
        generated=now.isoformat(),
    )
    meta = fc["metadata"]
    print(f"  {meta['n_images']} images ({meta['n_untagged']} untagged)")

    geojson_path = os.path.join(args.outdir, "field.geojson")
    with open(geojson_path, "w") as f:
        json.dump(fc, f, indent=2)
    print(f"  wrote {geojson_path}")

    print(f"aggregating into a {cell_meters} m field grid ...")
    grid = field.build_grid(fc, cell_meters=cell_meters)
    cells = field.classify_cells(grid, healthy=healthy, stressed=stressed)
    problems = field.rank_problems(cells, top_n=top_n)
    summary = field.summarize(cells)
    print(f"  {summary['n_cells']} cells, {summary['n_problem_cells']} need attention "
          f"(healthy {summary['pct_healthy']}% / stressed {summary['pct_stressed']}% "
          f"/ severe {summary['pct_severe']}%)")

    report_path = os.path.join(args.outdir, "report.pdf")
    if grid.get("empty"):
        print("  no GPS in photos - building a per-image NDVI gallery (no field map)")
        report.build_gallery_report(fc, report_path,
                                    title="Field NDVI Report (no GPS)")
        print(f"  wrote {report_path}")
        print("  skipped heatmap.png + map.html (a field map needs geotagged photos)")
    else:
        heatmap_path = os.path.join(args.outdir, "heatmap.png")
        fieldmap.render_grid_png(grid, heatmap_path)
        print(f"  wrote {heatmap_path}")

        report.build_report(fc, grid, cells, problems, summary,
                            heatmap_path, report_path)
        print(f"  wrote {report_path}")

        map_path = os.path.join(args.outdir, "map.html")
        webmap.build_webmap(fc, grid, cells, problems, heatmap_path,
                            map_path, overlay_opacity=opacity)
        print(f"  wrote {map_path}")

    print("done.")


if __name__ == "__main__":
    main()
