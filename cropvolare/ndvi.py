"""
NDVI pipeline for Raspberry Pi + Arducam NoIR V3 w/ Wratten 25A filter.

Single-camera NDVI: the Wratten 25A blocks blue visible light, so the
blue Bayer channel picks up mostly NIR (700-1000nm) and the red channel
gets visible red (580-700nm). Then it's just (NIR - Red) / (NIR + Red).
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


def create_camera(resolution=(2304, 1296)):
    """Set up the NoIR camera. Returns a configured (but not started) Picamera2."""
    if Picamera2 is None:
        raise RuntimeError(
            "picamera2 not installed — run: sudo apt install python3-picamera2"
        )

    cam = Picamera2()
    config = cam.create_still_configuration(
        main={"size": resolution, "format": "RGB888"},
    )
    cam.configure(config)
    return cam


def capture_image(cam, warmup=2):
    """Grab a single frame, returns BGR numpy array."""
    import time
    cam.start()
    time.sleep(warmup)  # let auto-exposure settle
    frame = cam.capture_array()
    cam.stop()
    # picamera2 gives us RGB, flip to BGR for opencv
    if cv2 is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    else:
        frame = frame[:, :, ::-1].copy()
    return frame


def extract_channels(image):
    """Pull out NIR (blue ch) and red (red ch) as float arrays in [0,1]."""
    nir = image[:, :, 0].astype(np.float64) / 255.0
    red = image[:, :, 2].astype(np.float64) / 255.0
    return nir, red


def compute_ndvi(nir, red):
    """Per-pixel NDVI. Returns array in [-1, 1], zero where both channels are 0."""
    denom = nir + red
    return np.where(denom > 0, (nir - red) / denom, 0.0)


def compute_ndvi_from_image(image):
    """Extract channels + compute NDVI in one shot."""
    nir, red = extract_channels(image)
    return compute_ndvi(nir, red)


def calibrate_with_reference(image, ref_roi, known_nir=0.5, known_red=0.5):
    """Normalize channels using a grey reference target in the frame.

    ref_roi is (x, y, w, h) for the target location. We measure the
    average channel values there and scale the whole image so the target
    matches the known reflectance values. This handles varying sunlight.
    """
    x, y, w, h = ref_roi
    roi = image[y:y+h, x:x+w]

    measured_nir = roi[:, :, 0].mean() / 255.0
    measured_red = roi[:, :, 2].mean() / 255.0

    nir_scale = known_nir / measured_nir if measured_nir > 0.01 else 1.0
    red_scale = known_red / measured_red if measured_red > 0.01 else 1.0

    cal = image.astype(np.float64)
    cal[:, :, 0] *= nir_scale
    cal[:, :, 2] *= red_scale

    return np.clip(cal, 0, 255).astype(np.uint8)


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


def colorize_ndvi(ndvi):
    """Turn NDVI array into a JET colormap image (BGR). Needs OpenCV."""
    if cv2 is None:
        raise RuntimeError("opencv required for colorize_ndvi")

    normalized = ((ndvi + 1) / 2 * 255).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_JET)


def save_ndvi_image(ndvi, path):
    """Save colorized NDVI heatmap to a file."""
    if cv2 is None:
        raise RuntimeError("opencv required for save_ndvi_image")
    cv2.imwrite(path, colorize_ndvi(ndvi))
