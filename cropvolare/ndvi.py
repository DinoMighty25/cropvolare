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


def _to_bgr(frame):
    """picamera2 gives RGB; flip to BGR for opencv."""
    if cv2 is not None:
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame[:, :, ::-1].copy()


def capture_frame(cam):
    """Grab one frame from an ALREADY-STARTED camera. Returns BGR.

    Use this in a capture loop (e.g. a flight) where the camera is started once
    and kept running - avoids the start/warmup/stop cost on every shot.
    """
    return _to_bgr(cam.capture_array())


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
    """Pull out NIR (blue ch) and red (red ch) as float arrays in [0,1]."""
    nir = image[:, :, 0].astype(np.float64) / 255.0
    red = image[:, :, 2].astype(np.float64) / 255.0
    return nir, red


def remove_gamma(channel, gamma=0.8):
    """Linearize a [0,1] channel by undoing gamma encoding: c ** (1/gamma)."""
    if not gamma or gamma == 1.0:
        return channel
    return np.power(np.clip(channel, 0.0, 1.0), 1.0 / gamma)


def correct_leakage(nir, red, k=0.6):
    """Subtract NIR bleed from the red channel: red = clip(red - k*nir, 0, 1)."""
    if not k:
        return red
    return np.clip(red - k * nir, 0.0, 1.0)


def compute_ndvi(nir, red):
    """Per-pixel NDVI. Returns array in [-1, 1], zero where both channels are 0."""
    denom = nir + red
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, (nir - red) / denom, 0.0)


def compute_ndvi_from_image(image, gamma=0.8, leakage_k=0.6):
    """Full single-image NDVI: extract -> linearize -> de-leak -> NDVI.

    Pass gamma=1.0 and leakage_k=0.0 to get the raw uncorrected index.
    """
    nir, red = extract_channels(image)
    nir = remove_gamma(nir, gamma)
    red = remove_gamma(red, gamma)
    red = correct_leakage(nir, red, leakage_k)
    return compute_ndvi(nir, red)


def solve_leakage_k(image, gamma=0.8, center_frac=0.5):
    """Solve the red-leakage constant k from a photo of a neutral grey card.

    A grey card reflects red and NIR equally, so any excess in the red channel
    is NIR bleeding through the filter: red_lin - k*nir_lin = nir_lin, giving
    k = (red_lin - nir_lin) / nir_lin. Only the central region of the frame is
    used (avoids vignetting). Returns k >= 0; with this k the card reads
    NDVI = 0, which is the calibration target.
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
    return max(0.0, (red_mean - nir_mean) / nir_mean)


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

def build_flatfield(frames):
    """Build a per-pixel gain map from frames of a uniform white panel.

    Averages the frames, then computes the gain that flattens each pixel to the
    global mean - this corrects lens vignetting and sensor non-uniformity.
    Save the result with np.save(...) and reuse it across a capture session.
    """
    if len(frames) == 0:
        raise ValueError("need at least one frame to build a flat-field map")
    stack = np.stack([f.astype(np.float64) for f in frames], axis=0)
    avg = stack.mean(axis=0)
    target = avg.mean()
    return np.where(avg > 1e-6, target / avg, 1.0)


def apply_flatfield(image, gain):
    """Apply a gain map (from build_flatfield) to an image, returns uint8."""
    corrected = image.astype(np.float64) * gain
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
