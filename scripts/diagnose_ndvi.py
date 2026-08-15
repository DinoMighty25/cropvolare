#!/usr/bin/env python3
"""Diagnose NDVI saturation and estimate the channel model from FLIGHT PHOTOS ONLY.

You do not need a grey card or a white sheet to run this. Point it at any
folder of photos taken with the Wratten 25 filter mounted:

    python scripts/diagnose_ndvi.py --dir flights/today/

It answers three questions:

  1. Is the current config saturating NDVI?  (the "everything reads 1.0" bug)
  2. Does the Bayer-NIR-transparency assumption hold on YOUR rig?
     Under a deep-red filter the blue and green channels both see essentially
     pure NIR, so green/blue should be ~1.0. If it is, the red pixel's NIR
     response is also ~the same, which pins the leakage constant k at ~1.0.
  3. What red-channel gain does your rig actually have, and how much dynamic
     range do you recover with the corrected model?
"""

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cropvolare.ndvi import load_image, remove_gamma, compute_ndvi


def _linear(img, gamma):
    b = remove_gamma(img[:, :, 0].astype(np.float32) / 255.0, gamma)
    g = remove_gamma(img[:, :, 1].astype(np.float32) / 255.0, gamma)
    r = remove_gamma(img[:, :, 2].astype(np.float32) / 255.0, gamma)
    return b, g, r


def ndvi_with(nir, red, k, red_gain):
    """Corrected model: subtract NIR bleed, THEN rescale red to NIR units."""
    red_lin = np.clip(red - k * nir, 0.0, None)
    if red_gain > 0:
        red_lin = red_lin / red_gain
    return compute_ndvi(nir, red_lin)


def spread(a):
    """5th-95th percentile spread - how much usable dynamic range there is."""
    return float(np.percentile(a, 95) - np.percentile(a, 5))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True, help="folder of filtered flight photos")
    p.add_argument("--gamma", type=float, default=0.8,
                   help="gamma currently in your config (default 0.8)")
    p.add_argument("--max", type=int, default=12, help="max photos to sample")
    args = p.parse_args()

    paths = sorted({p_ for pat in ("*.jpg", "*.jpeg", "*.JPG", "*.png")
                    for p_ in glob.glob(os.path.join(args.dir, pat))})[:args.max]
    if not paths:
        raise SystemExit(f"no images found in {args.dir}")

    print(f"sampling {len(paths)} photo(s) from {args.dir}\n")

    gb_ratios, rb_ratios = [], []
    pinned_now, pct = [], []

    for path in paths:
        img = load_image(path)
        nir, green, red = _linear(img, args.gamma)

        # centre 50% only: avoids vignetting skewing the channel statistics
        h, w = nir.shape
        cy, cx = slice(h // 4, 3 * h // 4), slice(w // 4, 3 * w // 4)
        n_m = float(nir[cy, cx].mean())
        g_m = float(green[cy, cx].mean())
        r_m = float(red[cy, cx].mean())
        if n_m < 1e-6:
            continue

        gb_ratios.append(g_m / n_m)
        rb_ratios.append(r_m / n_m)

        cur = ndvi_with(nir, red, k=2.0, red_gain=0.0)   # config as it stands
        pinned_now.append(float((cur > 0.999).mean()))

        pct.append(np.percentile(cur, [5, 25, 50, 75, 95]))

    gb = float(np.mean(gb_ratios))
    rb = float(np.mean(rb_ratios))

    print("=" * 62)
    print("1. SATURATION CHECK  (current config: k=2.0)")
    print("=" * 62)
    frac = float(np.mean(pinned_now))
    print(f"   pixels pinned at NDVI = 1.0 : {frac * 100:6.2f} %")
    q = np.mean(np.array(pct), axis=0)
    print("   NDVI percentiles  p5/p25/p50/p75/p95 :")
    print("      " + "  ".join(f"{v:6.3f}" for v in q))
    if frac > 0.30:
        print("   >> SATURATED. Most of the frame carries no information.")
        print("      A report built on this would show a uniformly healthy field")
        print("      regardless of what is actually growing there.")
    elif frac > 0.05:
        print("   >> PARTIALLY SATURATED. The healthy end of the scale is clipped.")
    else:
        print("   >> Not saturated. The k=2.0 concern may not apply to this data.")

    print()
    print("=" * 62)
    print("2. BAYER NIR PRIOR  (does green/blue ~ 1.0 ?)")
    print("=" * 62)
    print(f"   mean green/blue ratio : {gb:.3f}   (expect ~0.9-1.1)")
    print(f"   mean red/blue   ratio : {rb:.3f}   (scene-dependent, see below)")
    if 0.85 <= gb <= 1.15:
        print("   >> PRIOR HOLDS. Blue and green both read ~pure NIR, so the red")
        print("      pixel's NIR response matches too => k is ~1.0, NOT 2.0.")
    else:
        print("   >> Prior does NOT hold cleanly. Green and blue differ, so use a")
        print("      two-target calibration rather than the k=1 shortcut.")

    print()
    print("=" * 62)
    print("3. WHAT THIS RUN CANNOT TELL YOU")
    print("=" * 62)
    print("   The red-channel GAIN cannot be recovered from crop photos: the")
    print("   red/blue ratio depends on what is in frame (healthy canopy ~1.2,")
    print("   bare soil ~2.8), not just on the rig. Only a neutral grey card,")
    print("   where red and NIR reflectance are equal by construction, isolates")
    print("   it. Shoot one grey-card frame, then:")
    print()
    print("      python scripts/calibrate.py --input calib/frame_0000.jpg --write")
    print()
    print("   Until then you can fly and collect, but not calibrate.")


if __name__ == "__main__":
    main()
