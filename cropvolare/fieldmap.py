"""
Render the field grid to a colorized heatmap PNG.

This is the single field-overview raster shared by both deliverables: the PDF
embeds it, and the web map drops it on Leaflet as an ImageOverlay. It reuses the
same RdYlGn colormap as the single-image pipeline (ndvi.colorize_ndvi) so the
field view is visually consistent with per-photo overlays.
"""

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from .ndvi import colorize_ndvi

# BGRA grey for cells that had no photos (transparent in the overlay)
_EMPTY_RGBA = (200, 200, 200, 0)


def render_grid_png(grid, path, upscale=16):
    """Rasterize the grid's per-cell mean NDVI to a PNG with transparency.

    Empty cells become fully transparent so the basemap shows through on the web
    map. Returns the geographic bounds [[south, west], [north, east]] for the
    Leaflet overlay (None if the grid is empty).
    """
    if cv2 is None:
        raise RuntimeError("opencv required for render_grid_png")
    if grid.get("empty"):
        return None

    mean = grid["mean_ndvi"]
    counts = grid["counts"]

    # colorize_ndvi needs finite values; fill empty cells with 0 then mask them
    filled = np.where(counts > 0, mean, 0.0).astype(np.float64)
    bgr = colorize_ndvi(filled)  # (nrows, ncols, 3) BGR uint8

    alpha = np.where(counts > 0, 255, 0).astype(np.uint8)
    bgra = np.dstack([bgr, alpha])

    # blocky upscale so individual cells stay crisp (nearest-neighbour)
    big = cv2.resize(
        bgra,
        (grid["ncols"] * upscale, grid["nrows"] * upscale),
        interpolation=cv2.INTER_NEAREST,
    )
    cv2.imwrite(path, big)
    return grid["bounds"]
