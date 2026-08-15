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
    ANCHOR_TARGETS,
    channel_nir_ratio,
    solve_two_point,
    build_flatfield,
    check_clipping,
    compute_ndvi_from_image,
    load_image,
    save_flatfield,
    solve_red_gain_from_anchor,
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
    img = load_image(args.input)

    clip = check_clipping(img)
    print(f"clipping check: NIR {clip['nir_clipped']*100:.2f}% / "
          f"red {clip['red_clipped']*100:.2f}% at ceiling")
    if not clip["ok"]:
        print("  !! target is clipped or near-black - the channel RATIO is")
        print("     corrupted and the solved gain will be wrong. Reshoot in")
        print("     shade or with a shorter exposure, then try again.")
        if not args.force:
            raise SystemExit("aborting (pass --force to override)")

    k = args.k
    gain = solve_red_gain_from_anchor(img, anchor=args.anchor,
                                      assumed_ndvi=args.assumed_ndvi,
                                      gamma=args.gamma, k=k)
    assumed = (args.assumed_ndvi if args.assumed_ndvi is not None
               else ANCHOR_TARGETS[args.anchor][0])
    print(f"anchor '{args.anchor}' assumed NDVI = {assumed}")
    print(f"solved: leakage_k = {k}, red_gain = {gain}")

    check = compute_ndvi_from_image(img, gamma=args.gamma, leakage_k=k,
                                    red_gain=gain)
    print(f"anchor reads NDVI {float(check.mean()):.4f} (target {assumed})")

    if args.plant:
        ndvi = compute_ndvi_from_image(load_image(args.plant), gamma=args.gamma,
                                       leakage_k=k, red_gain=gain)
        mean = float(ndvi.mean())
        verdict = ("looks right" if 0.4 <= mean <= 0.9
                   else "outside 0.4-0.9 for healthy canopy - recheck filter")
        print(f"plant mean NDVI: {mean:.4f} ({verdict})")
    if args.dead:
        ndvi = compute_ndvi_from_image(load_image(args.dead), gamma=args.gamma,
                                       leakage_k=k, red_gain=gain)
        mean = float(ndvi.mean())
        verdict = "looks right" if mean < 0.3 else "too high for dead material"
        print(f"dead/bare mean NDVI: {mean:.4f} ({verdict})")

    if args.write:
        _update_config(args.config, "ndvi", "leakage_k", k)
        _update_config(args.config, "ndvi", "red_gain", gain)
        print(f"wrote leakage_k={k}, red_gain={gain} to {args.config}")
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


def run_two_point(args):
    """Solve k and red_gain from ordinary flight frames - no target required."""
    paths = sorted({p for pat in ("*.jpg", "*.jpeg", "*.JPG", "*.png")
                    for p in glob.glob(os.path.join(args.two_point, pat))})
    if not paths:
        raise SystemExit(f"no images found in {args.two_point}")
    paths = paths[:args.max_frames]

    ks, gains, ratios = [], [], []
    for path in paths:
        img = load_image(path)
        try:
            k, gain = solve_two_point(img, veg_ndvi=args.veg_ndvi,
                                      soil_ndvi=args.soil_ndvi, gamma=args.gamma)
        except ValueError as exc:
            print(f"  {os.path.basename(path)}: skipped ({exc})")
            continue
        ks.append(k)
        gains.append(gain)
        ratios.append(channel_nir_ratio(img, gamma=args.gamma))
        print(f"  {os.path.basename(path)}: k={k:.3f} red_gain={gain:.3f}")

    if not ks:
        raise SystemExit("no frame had both vegetation and bare soil in view")

    k = round(float(sum(ks) / len(ks)), 3)
    gain = round(float(sum(gains) / len(gains)), 3)
    gb = sum(ratios) / len(ratios)
    spread_k = max(ks) - min(ks)
    spread_g = max(gains) - min(gains)

    print(f"\nsolved across {len(ks)} frame(s):")
    print(f"  leakage_k = {k}   (frame-to-frame spread {spread_k:.3f})")
    print(f"  red_gain  = {gain}   (frame-to-frame spread {spread_g:.3f})")
    if spread_k > 0.3 or spread_g > 0.6:
        print("  !! frames disagree a lot - scene content varies too much. Prefer")
        print("     frames with a similar mix of canopy and bare ground.")

    print(f"\nindependent check: green/blue = {gb:.3f}, which predicts k ~ {gb:.2f}")
    if abs(gb - k) > 0.6:
        print("  !! that disagrees with the solved k. Treat both as unreliable and")
        print("     shoot a neutral target (--anchor) instead.")
    else:
        print("  consistent with the solved k.")

    if args.write:
        _update_config(args.config, "ndvi", "leakage_k", k)
        _update_config(args.config, "ndvi", "red_gain", gain)
        print(f"\nwrote leakage_k={k}, red_gain={gain} to {args.config}")
    else:
        print("\n(re-run with --write to save these into the config)")


def main():
    p = argparse.ArgumentParser(description="Solve leakage_k and/or flat-field")
    p.add_argument("-i", "--input",
                   help="photo of the calibration anchor (grey card, PTFE, "
                        "white paper, concrete or asphalt)")
    p.add_argument("--anchor", default="concrete", choices=sorted(ANCHOR_TARGETS),
                   help="what the anchor photo shows (default: concrete)")
    p.add_argument("--assumed-ndvi", type=float, default=None,
                   help="override the anchor's assumed true NDVI")
    p.add_argument("--k", type=float, default=1.0,
                   help="NIR bleed constant; 1.0 = Bayer-NIR-transparency prior")
    p.add_argument("--dead", help="optional dead/bare photo to sanity-check")
    p.add_argument("--force", action="store_true",
                   help="calibrate even if the anchor photo is clipped")
    p.add_argument("--two-point",
                   help="folder of ordinary crop frames: solve k AND red_gain "
                        "from canopy + bare soil already in the scene")
    p.add_argument("--veg-ndvi", type=float, default=0.75,
                   help="assumed true NDVI of dense canopy (default 0.75)")
    p.add_argument("--soil-ndvi", type=float, default=0.15,
                   help="assumed true NDVI of bare soil (default 0.15)")
    p.add_argument("--max-frames", type=int, default=12,
                   help="how many frames to average over (default 12)")
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

    if not args.input and not args.flatfield_dir and not args.two_point:
        p.error("give --two-point (crop frames), --input (neutral target) "
                "and/or --flatfield-dir (white target)")

    if args.flatfield_dir:
        run_flatfield(args)
    if args.two_point:
        run_two_point(args)
    if args.input:
        run_leakage(args)


if __name__ == "__main__":
    main()
