#!/usr/bin/env python3
"""Twin-flight repeatability: measure how much this system disagrees with itself.

Fly the same pattern twice, an hour apart, on the same day, with the same
exposure preset. Process both. Then run this. The crop cannot change in an hour,
so every difference between the two maps is measurement error - and that number
is the error bar on every claim the project makes.

    # after processing both flights
    python scripts/compare_flights.py output/flight_a output/flight_b

    # tighter: drop soft frames first, as a data flight should
    python scripts/compare_flights.py output/a output/b --min-sharpness 15

Takes the PROCESSED output directories (the ones containing field.geojson), not
the raw photo folders. Writes comparison.json next to the first one.

Reads capture_meta.json from both flights and refuses to report a clean pass if
the exposure did not match - otherwise the number measures the exposure
difference rather than the system's noise, and reads as a hardware problem.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cropvolare import exposure, repeat

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(REPO, "config", "default.json")


def load_geojson(path):
    """Accept either a processed output dir or a field.geojson path directly."""
    if os.path.isdir(path):
        candidate = os.path.join(path, "field.geojson")
        if not os.path.exists(candidate):
            raise SystemExit(
                f"no field.geojson in {path}\n"
                f"  run process_flight.py on the flight first:\n"
                f"    python scripts/process_flight.py -i <flight folder> -o {path}")
        path = candidate
    with open(path) as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser(
        description="Measure repeatability between two flights of the same field")
    p.add_argument("flight_a", help="processed output dir of the first flight")
    p.add_argument("flight_b", help="processed output dir of the second flight")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--cell-meters", type=float, default=None,
                   help="grid cell size (default: field.cell_meters from config)")
    p.add_argument("--min-count", type=int, default=repeat.DEFAULT_MIN_COUNT,
                   help="photos required in a cell in BOTH flights to compare it")
    p.add_argument("--threshold", type=float, default=repeat.DEFAULT_THRESHOLD,
                   help="pass/fail line on mean absolute NDVI difference")
    p.add_argument("--json", dest="json_out", default=None,
                   help="where to write comparison.json (default: beside flight_a)")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    cfg = {}
    if os.path.exists(args.config):
        with open(args.config) as f:
            cfg = json.load(f)
    cell_meters = (args.cell_meters if args.cell_meters is not None
                   else cfg.get("field", {}).get("cell_meters", 20.0))

    fc_a = load_geojson(args.flight_a)
    fc_b = load_geojson(args.flight_b)

    # --- exposure audit: is this comparison even meaningful? ---------------
    meta_a = exposure.read_capture_meta(args.flight_a)
    meta_b = exposure.read_capture_meta(args.flight_b)
    exp_ok, exp_reason = exposure.comparable(meta_a, meta_b)

    print("=" * 68)
    print("TWIN-FLIGHT REPEATABILITY")
    print("=" * 68)
    print(f"A: {args.flight_a}")
    print(f"B: {args.flight_b}")
    print()
    print(f"exposure check: {'OK' if exp_ok else 'PROBLEM'} - {exp_reason}")

    try:
        result = repeat.compare(fc_a, fc_b, cell_meters=cell_meters,
                                min_count=args.min_count,
                                threshold=args.threshold)
    except ValueError as exc:
        print(f"\ncannot compare: {exc}")
        return 2

    result["exposure_ok"] = exp_ok
    result["exposure_reason"] = exp_reason
    result["flight_a"] = args.flight_a
    result["flight_b"] = args.flight_b

    print()
    print(f"grid:      {cell_meters:.0f} m cells, "
          f"{result['n_cells_compared']} cell(s) compared "
          f"(A covers {result['n_cells_a']}, B covers {result['n_cells_b']})")
    print(f"frames:    {result['n_frames_a']} in A, {result['n_frames_b']} in B")

    if not result["ok"]:
        print(f"\nINCONCLUSIVE: {result['reason']}")
        _write(args, result)
        return 2

    print()
    print(f"mean NDVI:            A {result['mean_ndvi_a']:+.4f}   "
          f"B {result['mean_ndvi_b']:+.4f}")
    print(f"mean |difference|:    {result['mean_abs_delta']:.4f}   "
          f"<-- THE NUMBER (threshold {args.threshold:.2f})")
    print(f"RMS difference:       {result['rms_delta']:.4f}")
    print(f"bias (B - A):         {result['bias']:+.4f}")
    print(f"95th percentile:      {result['p95_abs_delta']:.4f}")
    print(f"worst cell:           {result['max_abs_delta']:.4f}")
    print()
    print(repeat.verdict(result))

    if not exp_ok:
        print()
        print("NOTE: because the exposure check failed, treat this number as an "
              "upper bound. It includes the exposure mismatch, not just the "
              "system's own noise.")

    print()
    print("Write the mean |difference| on the protocol sheet and into "
          "docs/results.md - it is the error bar for the season.")

    _write(args, result)
    # exit 0 only when the number is trustworthy AND passing
    return 0 if (result["passes"] and exp_ok) else 1


def _write(args, result):
    out = args.json_out or os.path.join(
        args.flight_a if os.path.isdir(args.flight_a) else ".", "comparison.json")
    try:
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
        if not args.quiet:
            print(f"wrote {out}")
    except OSError as exc:
        print(f"could not write {out}: {exc}")


if __name__ == "__main__":
    sys.exit(main())
