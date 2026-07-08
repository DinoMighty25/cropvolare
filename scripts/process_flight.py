#!/usr/bin/env python3
"""Process a folder of geotagged drone photos into a farmer-ready field report.

    python scripts/process_flight.py --input flights/2026-06-21/ \
           --outdir output/2026-06-21/

Produces, in --outdir:
    field.geojson   the durable per-photo NDVI data (also the ODM hand-off)
    heatmap.png     the colorized field NDVI overlay
    report.pdf      one-page farmer report
    map.html        standalone interactive web map

The whole pipeline is also importable as run() - cropvolare.jobs uses it to
process flights ON the Pi, so the CLI and the on-device path can never drift.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cropvolare import analyze, batch, field, fieldmap, history, report, webmap
from cropvolare.ndvi import load_flatfield

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(REPO, "config", "default.json")


def resolve_flatfield(cli_path, no_flatfield, cal_cfg):
    """Pick the gain map: CLI flag > config; returns (gain, label)."""
    if no_flatfield:
        return None, "disabled (--no-flatfield)"
    path = cli_path or cal_cfg.get("flatfield_path")
    if not path:
        return None, "not configured (run calibrate.py --flatfield-dir)"
    full = path if os.path.isabs(path) else os.path.join(REPO, path)
    if not os.path.exists(full):
        return None, (f"WARNING: gain map not found ({path}) - processing "
                      f"WITHOUT flat-field. Build it: calibrate.py "
                      f"--flatfield-dir <white-target frames> --write")
    return load_flatfield(full), f"active ({path})"


def load_config(path):
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def run(input_dir, outdir, config_path=DEFAULT_CONFIG, *, gamma=None,
        leakage_k=None, block_size=None, cell_meters=None, top_n=None,
        min_sharpness=None, flatfield=None, no_flatfield=False,
        write_overlays=True, field_name=None, basic=False, process_scale=1.0,
        progress_fn=None, log_fn=print):
    """The full flight -> report pipeline as one callable.

    Every parameter defaulting to None falls back to the config file, exactly
    like the CLI flags do. progress_fn(done, total) fires per frame during the
    NDVI pass (the bulk of the wall time). Returns a summary dict:
    {result, trend, verdict, report_path, n_images, basic} - result is the
    AnalysisResult (None when basic=True).
    """
    cfg = load_config(config_path)
    ndvi_cfg = cfg.get("ndvi", {})
    field_cfg = cfg.get("field", {})
    cal_cfg = cfg.get("calibration", {})

    gamma = gamma if gamma is not None else ndvi_cfg.get("gamma", 0.8)
    leakage_k = (leakage_k if leakage_k is not None
                 else ndvi_cfg.get("leakage_k", 0.6))
    block_size = (block_size if block_size is not None
                  else ndvi_cfg.get("block_size", 64))
    cell_meters = (cell_meters if cell_meters is not None
                   else field_cfg.get("cell_meters", 20.0))
    top_n = top_n if top_n is not None else field_cfg.get("top_n_problems", 5)
    healthy = field_cfg.get("healthy_threshold", 0.5)
    stressed = field_cfg.get("stressed_threshold", 0.3)
    opacity = field_cfg.get("overlay_opacity", 0.6)
    min_sharpness = (min_sharpness if min_sharpness is not None
                     else ndvi_cfg.get("min_sharpness", 0.0))

    gain, ff_label = resolve_flatfield(flatfield, no_flatfield, cal_cfg)
    log_fn(f"flat-field: {ff_label}")

    os.makedirs(outdir, exist_ok=True)
    now = datetime.now(timezone.utc)
    overlay_dir = os.path.join(outdir, "overlays") if write_overlays else None

    log_fn(f"ingesting photos from {input_dir} ...")
    fc = batch.process_directory(
        input_dir, gamma=gamma, leakage_k=leakage_k, block_size=block_size,
        overlay_dir=overlay_dir, gain=gain, min_sharpness=min_sharpness,
        process_scale=process_scale, progress_fn=progress_fn,
        flight_date=now.date().isoformat(),
        generated=now.isoformat(),
    )
    meta = fc["metadata"]
    log_fn(f"  {meta['n_images']} images kept "
           f"({meta['n_untagged']} untagged, {meta['n_unreadable']} unreadable, "
           f"{meta['n_filtered']} filtered below sharpness {min_sharpness})")

    geojson_path = os.path.join(outdir, "field.geojson")
    with open(geojson_path, "w") as f:
        json.dump(fc, f, indent=2)
    log_fn(f"  wrote {geojson_path}")

    log_fn(f"aggregating into a {cell_meters} m field grid ...")
    grid = field.build_grid(fc, cell_meters=cell_meters)
    cells = field.classify_cells(grid, healthy=healthy, stressed=stressed)
    problems = field.rank_problems(cells, top_n=top_n)
    summary = field.summarize(cells)
    log_fn(f"  {summary['n_cells']} cells, {summary['n_problem_cells']} need attention "
           f"(healthy {summary['pct_healthy']}% / stressed {summary['pct_stressed']}% "
           f"/ severe {summary['pct_severe']}%)")

    # render heatmap + web map whenever there's GPS (used by both report styles)
    heatmap_path = None
    if not grid.get("empty"):
        heatmap_path = os.path.join(outdir, "heatmap.png")
        fieldmap.render_grid_png(grid, heatmap_path)
        log_fn(f"  wrote {heatmap_path}")
        map_path = os.path.join(outdir, "map.html")
        webmap.build_webmap(fc, grid, cells, problems, heatmap_path,
                            map_path, overlay_opacity=opacity)
        log_fn(f"  wrote {map_path}")

    report_path = os.path.join(outdir, "report.pdf")

    if basic:
        if grid.get("empty"):
            report.build_gallery_report(fc, report_path)
        else:
            report.build_report(fc, grid, cells, problems, summary,
                                heatmap_path, report_path)
        log_fn(f"  wrote {report_path} (basic)")
        log_fn("done.")
        return {"result": None, "trend": None, "verdict": None,
                "report_path": report_path, "n_images": meta["n_images"],
                "basic": True}

    # --- analysis engine -> history -> agronomy report --------------------
    area_ha = None
    if not grid.get("empty"):
        (s, w), (n, e) = grid["bounds"]
        area_ha = round(abs((n - s) * grid["m_per_deg_lat"]) *
                        abs((e - w) * grid["m_per_deg_lon"]) / 10000.0, 3)

    flight_id = os.path.basename(os.path.normpath(input_dir))
    result = analyze.analyze(fc, grid=grid if not grid.get("empty") else None,
                             cells=cells, field=field_name, area_ha=area_ha,
                             date=now.date().isoformat(), flight_id=flight_id)
    v = result["verdict"]
    log_fn(f"analysis: {v['level'].upper()} (score {v['score']}) - {v['line']}")
    for p in result["patches"][:3]:
        log_fn(f"  problem #{p['rank']}: {p['area_ha']} ha @ "
               f"{p['lat']:.5f},{p['lon']:.5f} NDVI {p['mean_ndvi']} - "
               f"{(p['causes'] or [''])[0]}")

    trend = None
    if field_name:
        hist_dir = os.environ.get("CROPVOLARE_HISTORY_DIR",
                                  history.DEFAULT_HISTORY_DIR)
        prior = history.previous(field_name, exclude_flight=flight_id,
                                 history_dir=hist_dir)
        trend = history.compare(result, prior)
        history.record_flight(field_name, result, history_dir=hist_dir)
        if trend:
            log_fn(f"  trend vs {trend['prev_date']}: {trend['overall']} "
                   f"(mean NDVI {trend['mean_ndvi_delta']:+})")
        with open(os.path.join(outdir, "analysis.json"), "w") as f:
            json.dump({"result": result, "trend": trend}, f, indent=2)

    report.build_agronomy_report(result, report_path,
                                 fieldmap_png=heatmap_path, trend=trend)
    log_fn(f"  wrote {report_path}")
    log_fn("done.")
    return {"result": result, "trend": trend, "verdict": v,
            "report_path": report_path, "n_images": meta["n_images"],
            "basic": False}


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
    parser.add_argument("--min-sharpness", type=float,
                        help="drop frames below this sharpness (0 = keep all; "
                             "~15 filters grounded/defocused frames)")
    parser.add_argument("--flatfield",
                        help="gain map .npy (default: calibration.flatfield_path "
                             "from the config)")
    parser.add_argument("--no-flatfield", action="store_true",
                        help="disable flat-field correction")
    parser.add_argument("--no-overlays", action="store_true",
                        help="skip writing a per-photo NDVI overlay for each image")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="downscale frames before NDVI (0.5 = the Pi's "
                             "on-device fast path; laptop default 1.0)")
    parser.add_argument("--field", default=None,
                        help="field name: keys flight history for trend analysis")
    parser.add_argument("--basic", action="store_true",
                        help="emit the plain gallery/field report instead of the "
                             "agronomy report (debugging)")
    args = parser.parse_args()

    run(args.input, args.outdir, config_path=args.config, gamma=args.gamma,
        leakage_k=args.leakage_k, block_size=args.block_size,
        cell_meters=args.cell_meters, top_n=args.top_n,
        min_sharpness=args.min_sharpness, flatfield=args.flatfield,
        no_flatfield=args.no_flatfield, write_overlays=not args.no_overlays,
        field_name=args.field, basic=args.basic, process_scale=args.scale)


if __name__ == "__main__":
    main()
