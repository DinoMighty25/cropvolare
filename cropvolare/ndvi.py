"""
NDVI pipeline for Raspberry Pi + Arducam NoIR V3 w/ red NDVI filter.

Single-camera NDVI: the filter blocks blue visible light, so the blue Bayer
channel picks up mostly NIR (700-1000nm) and the red channel gets visible red
(580-700nm). Then it's just (NIR - Red) / (NIR + Red), after two corrections:

  1. gamma linearization  - the sensor stores values gamma-encoded; NDVI needs
                            linear reflectance, so we undo it before the math.
  2. red-leakage subtract - some NIR bleeds into the red channel through the
                            filter. We subtract a fraction (k) of NIR from red.

Everything below the camera helpers is pure numpy and runs anywhere, so you can
develop and test the whole pipeline on saved images without a Pi.
"""

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None


# --------------------------------------------------------------------------
# Camera (Pi only)
# --------------------------------------------------------------------------

def create_camera(resolution=(2304, 1296), colour_gains=(1.0, 1.0),
                  exposure_us=None, tuning_file="imx708_noir.json"):
    """Set up the NoIR camera for NDVI capture.

    Uses the NoIR tuning file: the standard tuning's colour-correction matrix
    remixes the channels for pleasing photos, which scrambles the NIR/red
    separation NDVI depends on (measured symptom: sky scoring higher than
    vegetation). The noir tuning keeps channels honest; colour gains default
    to neutral (1.0, 1.0) for the same reason.

    White balance is always locked and focus fixed at infinity, so channel
    ratios stay comparable between frames. Exposure: pass exposure_us to
    hard-lock a value, or leave None to let auto-exposure settle during
    warmup - the capture helpers then freeze it with lock_exposure() so every
    frame in the session matches. (Never fly with AE still enabled: per-frame
    exposure changes make NDVI incomparable.)
    Returns a configured (but not started) Picamera2.
    """
    if Picamera2 is None:
        raise RuntimeError(
            "picamera2 not installed - run: sudo apt install python3-picamera2"
        )

    tuning = None
    if tuning_file:
        try:
            tuning = Picamera2.load_tuning_file(tuning_file)
        except Exception as exc:  # noqa: BLE001 - fall back to default tuning
            print(f"warning: could not load tuning file {tuning_file}: {exc}")

    cam = Picamera2(tuning=tuning) if tuning is not None else Picamera2()
    config = cam.create_still_configuration(
        main={"size": resolution, "format": "RGB888"},
    )
    cam.configure(config)

    controls = {
        "AfMode": 0,                 # manual focus
        "LensPosition": 0.0,         # focus at infinity
        "AwbEnable": False,          # locked white balance
        "ColourGains": colour_gains,
    }
    if exposure_us is not None:
        controls["AeEnable"] = False
        controls["ExposureTime"] = int(exposure_us)
    else:
        controls["AeEnable"] = True  # settle now, freeze via lock_exposure()
    cam.set_controls(controls)

    return cam


def lock_exposure(cam):
    """Freeze auto-exposure at its current settled values.

    Call after the sensor has had a couple of seconds running with AE enabled;
    from then on every frame uses identical exposure and analogue gain, which
    NDVI comparability requires. Returns (exposure_us, analogue_gain).
    """
    meta = cam.capture_metadata()
    exposure = int(meta["ExposureTime"])
    gain = float(meta["AnalogueGain"])
    cam.set_controls({
        "AeEnable": False,
        "ExposureTime": exposure,
        "AnalogueGain": gain,
    })
    return exposure, gain


def capture_frame(cam):
    """Grab one frame from an ALREADY-STARTED camera. Returns BGR.

    No channel conversion: picamera2's "RGB888" format is already laid out
    [B, G, R] in memory (see the picamera2 manual's format table - the naming
    is counter-intuitive). Converting again would swap red and blue, putting
    the sensor's red channel in the NIR slot and sign-flipping NDVI.

    Use this in a capture loop (e.g. a flight) where the camera is started once
    and kept running - avoids the start/warmup/stop cost on every shot.
    """
    return cam.capture_array()


def capture_image(cam, warmup=2):
    """Start the camera, settle+lock exposure, grab one frame, stop. BGR array."""
    import time
    cam.start()
    time.sleep(warmup)   # let auto-exposure settle on the scene
    lock_exposure(cam)   # then freeze it (no-op if already hard-locked)
    frame = capture_frame(cam)
    cam.stop()
    return frame


def load_image(path):
    """Read an image file from disk as a BGR numpy array.

    Lets you run the whole pipeline on saved photos without a camera.
    """
    if cv2 is None:
        raise RuntimeError("opencv required for load_image")
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return img


# --------------------------------------------------------------------------
# NDVI math (pure numpy, hardware-free)
# --------------------------------------------------------------------------

def extract_channels(image):
    """Pull out NIR (blue ch) and red (red ch) as float arrays in [0,1].

    float32 throughout: NDVI needs ~3 significant digits and the whole chain
    preserves dtype, so this halves memory and roughly doubles throughput -
    what lets the 512 MB Pi Zero process a flight on-device.
    """
    nir = image[:, :, 0].astype(np.float32) / np.float32(255.0)
    red = image[:, :, 2].astype(np.float32) / np.float32(255.0)
    return nir, red


def remove_gamma(channel, gamma=0.8):
    """Linearize a [0,1] channel by undoing gamma encoding: c ** (1/gamma)."""
    if not gamma or gamma == 1.0:
        return channel
    return np.power(np.clip(channel, 0.0, 1.0), 1.0 / gamma)


def correct_leakage(nir, red, k=1.0, red_gain=1.0):
    """Recover visible-red in NIR-comparable units.

    Two separate physical effects, which the old single-constant version
    conflated (and which made vegetation NDVI saturate at 1.0):

      k         NIR bleeding THROUGH the filter into the red pixel. Bayer dyes
                are near-transparent above ~800nm, so the red pixel's NIR
                response closely matches the blue pixel's => k ~ 1.0.
      red_gain  the red pixel's SENSITIVITY to visible red relative to the blue
                pixel's sensitivity to NIR. A Wratten 25 passes a lot of red,
                so this is typically ~2-3. It is a DIVISOR, not a subtraction.

    Solve red_gain from a grey card (see solve_calibration). Note the upper
    clip is gone: after dividing by red_gain the value is back in NIR units
    and clipping at 1.0 would truncate bright soil.
    """
    red_lin = np.clip(red - k * nir, 0.0, None) if k else red
    if red_gain and red_gain > 0:
        red_lin = red_lin / red_gain
    return red_lin


def compute_ndvi(nir, red):
    """Per-pixel NDVI. Returns array in [-1, 1], zero where both channels are 0."""
    denom = nir + red
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, (nir - red) / denom, 0.0)


def compute_ndvi_from_image(image, gamma=0.8, leakage_k=1.0, red_gain=1.0):
    """Full single-image NDVI: extract -> linearize -> de-leak -> rescale -> NDVI.

    Pass gamma=1.0, leakage_k=0.0, red_gain=1.0 for the raw uncorrected index.
    """
    nir, red = extract_channels(image)
    nir = remove_gamma(nir, gamma)
    red = remove_gamma(red, gamma)
    red = correct_leakage(nir, red, leakage_k, red_gain)
    return compute_ndvi(nir, red)


def solve_calibration(image, gamma=0.8, k=1.0, center_frac=0.5):
    """Solve the red-channel gain from a photo of a neutral grey card.

    A grey card reflects red and NIR equally, so with
        red_meas = red_gain * REF + k * nir_meas
        nir_meas = REF
    the ratio gives directly:
        red_gain = red_meas/nir_meas - k

    k is fixed at the Bayer-NIR-transparency prior (~1.0) rather than solved,
    because ONE target cannot separate two unknowns - that is exactly the
    conflation that produced the old k=2.0 and saturated every healthy pixel.
    Verify the prior on your own rig with scripts/diagnose_ndvi.py: under a
    deep-red filter the green/blue channel ratio should sit near 1.0.

    Returns (k, red_gain). With these the card reads NDVI = 0.
    """
    nir, red = extract_channels(image)
    nir = remove_gamma(nir, gamma)
    red = remove_gamma(red, gamma)

    h, w = nir.shape
    dy = int(h * (1 - center_frac) / 2)
    dx = int(w * (1 - center_frac) / 2)
    nir_mean = nir[dy:h - dy, dx:w - dx].mean()
    red_mean = red[dy:h - dy, dx:w - dx].mean()

    if nir_mean < 1e-6:
        raise ValueError("grey-card image is too dark to calibrate from")

    red_gain = float(red_mean / nir_mean) - k
    if red_gain <= 0:
        raise ValueError(
            f"solved red_gain={red_gain:.3f} <= 0. The red channel reads less "
            f"than {k}x the NIR channel, which means the filter is not a "
            f"deep-red type, the channels are swapped, or the card was shaded.")
    return float(k), round(red_gain, 3)


def solve_leakage_k(image, gamma=0.8, center_frac=0.5):
    """Deprecated: conflates NIR bleed with red-channel gain. See solve_calibration.

    Kept so old calls do not break. The value it returns saturates healthy
    vegetation at NDVI = 1.0 when used as a pure subtraction.
    """
    import warnings
    warnings.warn("solve_leakage_k conflates leakage with channel gain and "
                  "saturates vegetation; use solve_calibration",
                  DeprecationWarning, stacklevel=2)
    nir, red = extract_channels(image)
    nir = remove_gamma(nir, gamma); red = remove_gamma(red, gamma)
    h, w = nir.shape
    dy = int(h * (1 - center_frac) / 2); dx = int(w * (1 - center_frac) / 2)
    n = nir[dy:h - dy, dx:w - dx].mean()
    if n < 1e-6:
        raise ValueError("grey-card image is too dark to calibrate from")
    return max(0.0, float((red[dy:h - dy, dx:w - dx].mean() - n) / n))


# Reference targets you can calibrate against WITHOUT buying a grey card.
# What matters is SPECTRAL NEUTRALITY between visible-red and NIR, not the 18%
# reflectance of a photographic card - red_gain is solved from a RATIO, so any
# brightness works provided neither channel clips.
ANCHOR_TARGETS = {
    "ptfe":     (0.02, "PTFE/plumber's tape over card - same material as "
                       "Spectralon standards, flat 400-1500nm. Shoot in shade."),
    "paper":    (0.03, "White printer paper in open shade. Optical brighteners "
                       "emit ~440nm, which the Wratten 25 blocks anyway."),
    "concrete": (0.08, "Dry concrete / pavement in full sun."),
    "asphalt":  (0.06, "Dry asphalt road, not freshly laid."),
    "drysoil":  (0.15, "Bare dry soil - use only if nothing better is in frame."),
}


def solve_red_gain_from_anchor(image, anchor="concrete", assumed_ndvi=None,
                               gamma=0.8, k=1.0, roi=None):
    """Solve red_gain from a patch of known-ish NDVI - no grey card required.

    The empirical-line idea: any surface whose true NDVI you can bound tightly
    pins the red-channel gain, because
        NDVI = (n - x)/(n + x)  with  x = (red - k*nir)/red_gain
    inverts to
        red_gain = (red - k*nir) * (1 + NDVI) / (nir * (1 - NDVI))

    A PTFE or white-paper target is neutral by construction (NDVI ~ 0). Dry
    concrete and asphalt sit near 0.06-0.08 and are already in most farm
    scenes, so they cost nothing.

    Accuracy: getting the assumed NDVI wrong by +/-0.10 moves recovered NDVI by
    at most ~0.12, and the error shrinks toward the healthy end - a canopy at
    true 0.85 still reads 0.83-0.88 across that whole band. Rank ordering is
    preserved exactly, so relative stress maps stay valid regardless.

    roi: (y0, y1, x0, x1) slice of the anchor surface. Defaults to the centre
    quarter, so fill the frame with the target if you leave it out.
    """
    if assumed_ndvi is None:
        if anchor not in ANCHOR_TARGETS:
            raise ValueError(f"unknown anchor {anchor!r}; "
                             f"choose from {sorted(ANCHOR_TARGETS)} "
                             f"or pass assumed_ndvi")
        assumed_ndvi = ANCHOR_TARGETS[anchor][0]
    if not -0.99 < assumed_ndvi < 0.99:
        raise ValueError("assumed_ndvi must be within (-0.99, 0.99)")

    nir, red = extract_channels(image)
    nir = remove_gamma(nir, gamma)
    red = remove_gamma(red, gamma)

    if roi is None:
        h, w = nir.shape
        roi = (h // 4, 3 * h // 4, w // 4, 3 * w // 4)
    y0, y1, x0, x1 = roi
    n = float(nir[y0:y1, x0:x1].mean())
    r = float(red[y0:y1, x0:x1].mean())

    if n < 1e-6:
        raise ValueError("anchor patch is too dark to calibrate from")

    leaked = r - k * n
    if leaked <= 0:
        raise ValueError(
            f"anchor red channel ({r:.3f}) is below k*NIR ({k * n:.3f}). Either "
            "the filter is not deep-red, the channels are swapped, or this "
            "patch is vegetation rather than a neutral surface.")

    red_gain = leaked * (1.0 + assumed_ndvi) / (n * (1.0 - assumed_ndvi))
    return round(float(red_gain), 3)


def check_clipping(image, warn_frac=0.01):
    """Guard for anchor shots: a clipped channel silently corrupts the ratio."""
    nir = image[:, :, 0]
    red = image[:, :, 2]
    out = {
        "nir_clipped": float((nir >= 254).mean()),
        "red_clipped": float((red >= 254).mean()),
        "nir_black": float((nir <= 1).mean()),
    }
    out["ok"] = (out["nir_clipped"] < warn_frac
                 and out["red_clipped"] < warn_frac
                 and out["nir_black"] < 0.5)
    return out


def solve_two_point(image, veg_ndvi=0.75, soil_ndvi=0.15, gamma=0.8,
                    veg_pct=5.0, soil_pct=75.0, mask=None):
    """Solve BOTH k and red_gain from one ordinary crop photo. No target needed.

    A single neutral target gives one equation for two unknowns, which is what
    made the old single-constant calibration conflate NIR bleed with channel
    gain. Any frame containing both canopy and bare ground gives two equations:

        red/nir = k + red_gain * (1 - NDVI) / (1 + NDVI)

    evaluated over a vegetated population and a bare-soil population, which
    solves as a 2x2 linear system.

    Pixels are split by their red/NIR ratio - lowest veg_pct percent are taken
    as vegetation, highest (100 - soil_pct) percent as bare soil. Pass an
    explicit boolean `mask` of vegetation instead if you have one.

    veg_ndvi / soil_ndvi are the assumed true values of those two populations.
    Moving them within sensible bounds (0.70-0.80 and 0.10-0.20) shifts the
    solved constants but leaves the RANK ORDERING of the output identical, so
    stress maps stay valid even when the absolute scale is uncertain.

    Returns (k, red_gain).
    """
    nir, red = extract_channels(image)
    nir = remove_gamma(nir, gamma)
    red = remove_gamma(red, gamma)

    ratio = red / np.maximum(nir, 1e-6)
    if mask is not None:
        veg, soil = mask, ~mask
    else:
        veg = ratio < np.percentile(ratio, veg_pct)
        soil = ratio > np.percentile(ratio, soil_pct)
    if veg.sum() < 100 or soil.sum() < 100:
        raise ValueError("not enough vegetation or bare-soil pixels to solve "
                         "from this frame; use one with both in view")

    r_veg = float(red[veg].mean() / max(nir[veg].mean(), 1e-6))
    r_soil = float(red[soil].mean() / max(nir[soil].mean(), 1e-6))

    c_veg = (1.0 - veg_ndvi) / (1.0 + veg_ndvi)
    c_soil = (1.0 - soil_ndvi) / (1.0 + soil_ndvi)
    if abs(c_veg - c_soil) < 1e-6:
        raise ValueError("veg_ndvi and soil_ndvi must differ")

    red_gain = (r_veg - r_soil) / (c_veg - c_soil)
    k = r_veg - red_gain * c_veg

    if red_gain <= 0:
        raise ValueError(
            f"solved red_gain={red_gain:.3f} <= 0. The vegetation and soil "
            "populations are not separating - check the filter is deep-red "
            "and that the frame really contains both.")
    return round(float(max(k, 0.0)), 3), round(float(red_gain), 3)


def channel_nir_ratio(image, gamma=0.8, center_frac=0.5):
    """green/blue mean ratio - an independent sanity check on the solved k.

    Under a deep-red filter both the green and blue pixels see essentially
    only NIR, so this ratio estimates how much more NIR the green pixel
    collects than the blue one. The red pixel sits close to green, so this
    predicts k. If it disagrees badly with solve_two_point's k, distrust both.
    """
    b = remove_gamma(image[:, :, 0].astype(np.float32) / 255.0, gamma)
    g = remove_gamma(image[:, :, 1].astype(np.float32) / 255.0, gamma)
    h, w = b.shape
    dy = int(h * (1 - center_frac) / 2)
    dx = int(w * (1 - center_frac) / 2)
    b_m = float(b[dy:h - dy, dx:w - dx].mean())
    if b_m < 1e-6:
        raise ValueError("frame too dark for a channel ratio")
    return float(g[dy:h - dy, dx:w - dx].mean()) / b_m


def compute_vari(image):
    """VARI = (G - R) / (G + R - B). Filter-free vegetation index cross-check.

    Use this on an UNFILTERED RGB photo - with the NDVI filter on, the blue
    channel is NIR, so VARI would be meaningless.
    """
    blue = image[:, :, 0].astype(np.float64)
    green = image[:, :, 1].astype(np.float64)
    red = image[:, :, 2].astype(np.float64)
    denom = green + red - blue
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(np.abs(denom) > 1e-6, (green - red) / denom, 0.0)


# --------------------------------------------------------------------------
# Flat-field calibration
# --------------------------------------------------------------------------

def build_flatfield(frames, smooth_sigma=25):
    """Build a per-pixel gain map from frames of a uniform white target.

    Averages the frames, Gaussian-smooths the average (vignetting is smooth by
    nature - smoothing kills paper texture and sensor noise in the flats), then
    computes the gain that flattens each pixel to ITS CHANNEL's mean. Per-channel
    normalization matters for NDVI: the gain corrects spatial shading only and
    never rebalances red against NIR, so it stays orthogonal to the leakage
    calibration. Pass smooth_sigma=0 to skip smoothing (or when cv2 is absent).
    """
    if len(frames) == 0:
        raise ValueError("need at least one frame to build a flat-field map")
    stack = np.stack([f.astype(np.float64) for f in frames], axis=0)
    avg = stack.mean(axis=0)
    if smooth_sigma and cv2 is not None:
        avg = cv2.GaussianBlur(avg, (0, 0), smooth_sigma)
    target = avg.mean(axis=(0, 1), keepdims=True)  # per-channel
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(avg > 1e-6, target / avg, 1.0)


def save_flatfield(gain, path, max_px=288):
    """Persist a gain map compactly (downscaled float32 .npy).

    Vignetting is smooth, so a <=max_px-wide map loses nothing meaningful and
    keeps the file ~0.5 MB instead of tens of MB. apply_flatfield resizes it
    back up to the image automatically.
    """
    import os
    gain = np.asarray(gain, dtype=np.float32)
    h, w = gain.shape[:2]
    if cv2 is not None and max(h, w) > max_px:
        scale = max_px / float(max(h, w))
        gain = cv2.resize(gain, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_AREA)
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    np.save(path, gain)


def load_flatfield(path):
    """Load a gain map saved by save_flatfield."""
    return np.load(path)


def apply_flatfield(image, gain):
    """Apply a gain map (from build_flatfield) to an image, returns uint8.

    The gain is resized to the image automatically when shapes differ (gains
    are stored downscaled by save_flatfield).
    """
    gain = np.asarray(gain, dtype=np.float32)
    if gain.shape[:2] != image.shape[:2]:
        if cv2 is None:
            raise RuntimeError("opencv required to resize a stored gain map")
        gain = cv2.resize(gain, (image.shape[1], image.shape[0]),
                          interpolation=cv2.INTER_LINEAR)
    corrected = image.astype(np.float32) * gain
    return np.clip(corrected, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# Zones + visualization
# --------------------------------------------------------------------------

def classify_zones(ndvi, block_size=64):
    """Break NDVI map into a grid and tag each block as healthy/moderate/stressed."""
    h, w = ndvi.shape
    zones = []

    for ri, y in enumerate(range(0, h, block_size)):
        for ci, x in enumerate(range(0, w, block_size)):
            block = ndvi[y:y+block_size, x:x+block_size]
            if block.size == 0:
                continue

            mean = float(np.mean(block))
            if mean >= 0.5:
                status = "healthy"
            elif mean >= 0.3:
                status = "moderate"
            else:
                status = "stressed"

            zones.append({
                "zone_row": ri,
                "zone_col": ci,
                "mean_ndvi": round(mean, 4),
                "status": status,
            })

    return zones


# NDVI colormap: red (stressed) -> yellow -> green (healthy). Built as a
# (pos, R, G, B) gradient over NDVI in [-1, 1]; this RdYlGn ramp is far more
# readable for a farmer report than OpenCV's JET.
_NDVI_COLOR_POINTS = [
    (-1.0, 165,   0,  38),
    (-0.5, 215,  48,  39),
    ( 0.0, 255, 255, 191),
    ( 0.5, 145, 207,  96),
    ( 1.0,  26, 152,  80),
]


def _build_ndvi_lut():
    positions = np.array([p[0] for p in _NDVI_COLOR_POINTS])
    rgb = np.array([p[1:] for p in _NDVI_COLOR_POINTS], dtype=np.float64)
    # map LUT index 0..255 back to NDVI -1..1
    ndvi_axis = np.linspace(-1.0, 1.0, 256)
    lut = np.empty((256, 3), dtype=np.float64)
    for c in range(3):
        lut[:, c] = np.interp(ndvi_axis, positions, rgb[:, c])
    return lut[:, ::-1].astype(np.uint8)  # RGB -> BGR for opencv


_NDVI_LUT = _build_ndvi_lut()


def colorize_ndvi(ndvi):
    """Turn NDVI array into a BGR heatmap using the NDVI colormap (numpy only)."""
    idx = np.clip((ndvi + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)
    return _NDVI_LUT[idx]


def save_ndvi_image(ndvi, path):
    """Save colorized NDVI heatmap as a PNG."""
    if cv2 is None:
        raise RuntimeError("opencv required for save_ndvi_image")
    cv2.imwrite(path, colorize_ndvi(ndvi))


def save_ndvi_tiff(ndvi, path):
    """Save NDVI as a 16-bit TIFF, preserving the full value range.

    NDVI in [-1, 1] is mapped to uint16 [0, 65535]. To recover the float:
        ndvi = arr.astype(float) / 65535.0 * 2.0 - 1.0
    """
    if cv2 is None:
        raise RuntimeError("opencv required for save_ndvi_tiff")
    arr = np.clip((ndvi + 1.0) / 2.0 * 65535.0, 0, 65535).astype(np.uint16)
    cv2.imwrite(path, arr)
