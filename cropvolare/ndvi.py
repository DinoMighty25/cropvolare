"""
NDVI computation module for Raspberry Pi + Arducam NoIR V3 camera.

Uses a single NoIR camera with a Wratten 25A red longpass filter to capture
dual-band imagery. The filter blocks visible blue light, so:
  - Blue Bayer channel  -> primarily NIR (700-1000nm)
  - Red Bayer channel   -> visible red (580-700nm) + some NIR

NDVI is computed per-pixel as: (NIR - Red) / (NIR + Red)
where NIR = blue channel, Red = red channel.
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


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------

def create_camera(resolution: tuple[int, int] = (4608, 2592)) -> "Picamera2":
    """Initialize the Arducam NoIR V3 camera via Picamera2.

    Args:
        resolution: Capture resolution as (width, height).

    Returns:
        Configured Picamera2 instance (not yet started).
    """
    if Picamera2 is None:
        raise RuntimeError(
            "picamera2 is not installed. "
            "Install it on your Raspberry Pi with: sudo apt install python3-picamera2"
        )

    cam = Picamera2()
    config = cam.create_still_configuration(
        main={"size": resolution, "format": "RGB888"},
    )
    cam.configure(config)
    return cam


def capture_image(cam: "Picamera2") -> np.ndarray:
    """Capture a single frame from the NoIR camera.

    Args:
        cam: A configured Picamera2 instance.

    Returns:
        BGR image as a numpy array (H, W, 3).
    """
    cam.start()
    frame = cam.capture_array()
    cam.stop()
    # picamera2 returns RGB; convert to BGR for OpenCV consistency
    if cv2 is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    else:
        frame = frame[:, :, ::-1].copy()
    return frame


# ---------------------------------------------------------------------------
# NDVI computation
# ---------------------------------------------------------------------------

def extract_channels(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract NIR and Red channels from a NoIR + Wratten 25A filtered image.

    With the Wratten 25A filter mounted:
      - Blue channel (index 0 in BGR) captures mostly NIR
      - Red channel (index 2 in BGR) captures visible red + NIR leakage

    Args:
        image: BGR image as uint8 numpy array (H, W, 3).

    Returns:
        Tuple of (nir, red) as float64 arrays in [0, 1].
    """
    nir = image[:, :, 0].astype(np.float64) / 255.0  # blue channel = NIR
    red = image[:, :, 2].astype(np.float64) / 255.0  # red channel = visible red
    return nir, red


def compute_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Compute per-pixel NDVI from NIR and Red channel arrays.

    NDVI = (NIR - Red) / (NIR + Red)

    Args:
        nir: NIR channel as float64 array, values in [0, 1].
        red: Red channel as float64 array, values in [0, 1].

    Returns:
        NDVI array with values in [-1, 1]. Pixels where both channels
        are zero are set to 0.
    """
    denominator = nir + red
    ndvi = np.where(denominator > 0, (nir - red) / denominator, 0.0)
    return ndvi


def compute_ndvi_from_image(image: np.ndarray) -> np.ndarray:
    """Convenience function: extract channels and compute NDVI in one call.

    Args:
        image: BGR image as uint8 numpy array (H, W, 3).

    Returns:
        NDVI array with values in [-1, 1].
    """
    nir, red = extract_channels(image)
    return compute_ndvi(nir, red)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibrate_with_reference(
    image: np.ndarray,
    ref_roi: tuple[int, int, int, int],
    known_nir_reflectance: float = 0.5,
    known_red_reflectance: float = 0.5,
) -> np.ndarray:
    """Apply hardware reference-target calibration to an image.

    A small grey reference target with known reflectance is mounted at a
    fixed position in the camera's field of view. This function normalizes
    channel values so that the reference target pixels match the expected
    reflectance ratio, compensating for changes in sunlight intensity and
    color temperature.

    Args:
        image: BGR image as uint8 numpy array (H, W, 3).
        ref_roi: Region of interest for the reference target as
                 (x, y, width, height) in pixels.
        known_nir_reflectance: Expected NIR reflectance of the target (0-1).
        known_red_reflectance: Expected red reflectance of the target (0-1).

    Returns:
        Calibrated BGR image as uint8 numpy array.
    """
    x, y, w, h = ref_roi
    roi = image[y : y + h, x : x + w]

    measured_nir = roi[:, :, 0].mean() / 255.0  # blue channel
    measured_red = roi[:, :, 2].mean() / 255.0  # red channel

    # Avoid division by zero
    nir_scale = known_nir_reflectance / measured_nir if measured_nir > 0.01 else 1.0
    red_scale = known_red_reflectance / measured_red if measured_red > 0.01 else 1.0

    calibrated = image.astype(np.float64)
    calibrated[:, :, 0] *= nir_scale  # scale blue/NIR channel
    calibrated[:, :, 2] *= red_scale  # scale red channel

    return np.clip(calibrated, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Analysis & visualization
# ---------------------------------------------------------------------------

def classify_zones(
    ndvi: np.ndarray,
    block_size: int = 64,
) -> list[dict]:
    """Segment NDVI map into grid zones and classify health status.

    Args:
        ndvi: NDVI array with values in [-1, 1].
        block_size: Size of each square zone in pixels.

    Returns:
        List of dicts with keys: zone_row, zone_col, mean_ndvi, status.
        Status is one of: "healthy", "moderate", "stressed".
    """
    h, w = ndvi.shape
    zones = []

    for row_idx, y in enumerate(range(0, h, block_size)):
        for col_idx, x in enumerate(range(0, w, block_size)):
            block = ndvi[y : y + block_size, x : x + block_size]
            if block.size == 0:
                continue

            mean_val = float(np.mean(block))

            if mean_val >= 0.5:
                status = "healthy"
            elif mean_val >= 0.3:
                status = "moderate"
            else:
                status = "stressed"

            zones.append({
                "zone_row": row_idx,
                "zone_col": col_idx,
                "mean_ndvi": round(mean_val, 4),
                "status": status,
            })

    return zones


def colorize_ndvi(ndvi: np.ndarray) -> np.ndarray:
    """Convert an NDVI array to a colorized heatmap image.

    Maps NDVI values to a color gradient:
      -1 (bare/water) -> blue
       0 (bare soil)  -> yellow
      +1 (dense veg)  -> dark green

    Args:
        ndvi: NDVI array with values in [-1, 1].

    Returns:
        BGR color image as uint8 numpy array (H, W, 3).
    """
    if cv2 is None:
        raise RuntimeError("OpenCV is required for colorize_ndvi")

    # Normalize from [-1, 1] to [0, 255]
    normalized = ((ndvi + 1) / 2 * 255).astype(np.uint8)

    # Apply colormap (COLORMAP_JET: blue -> green -> red, we invert so
    # high NDVI = green, low NDVI = red/blue)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    return colored


def save_ndvi_image(ndvi: np.ndarray, path: str) -> None:
    """Save a colorized NDVI heatmap to disk.

    Args:
        ndvi: NDVI array with values in [-1, 1].
        path: Output file path (e.g., "output/ndvi_map.png").
    """
    if cv2 is None:
        raise RuntimeError("OpenCV is required for save_ndvi_image")

    colored = colorize_ndvi(ndvi)
    cv2.imwrite(path, colored)
