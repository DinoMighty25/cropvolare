"""
Batch processing: a directory of geotagged photos -> a GeoJSON FeatureCollection.

Each photo becomes one Point Feature carrying its NDVI summary and GPS location.
The FeatureCollection is the single hand-off the rest of the pipeline (field
aggregation, heatmap, PDF, web map) consumes. All NDVI math is reused from
ndvi.py - nothing is recomputed here.
"""

import glob
import os

import numpy as np

from . import geo
from .ndvi import (
    classify_zones,
    compute_ndvi_from_image,
    load_image,
    save_ndvi_image,
)

DEFAULT_PATTERNS = ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG")


def _image_status(zone_counts):
    """Roll a photo's zone counts up to one healthy/moderate/stressed label."""
    if not zone_counts:
        return "unknown"
    return max(zone_counts, key=zone_counts.get)


def process_image(path, gamma=0.8, leakage_k=0.6, block_size=64,
                  overlay_dir=None):
    """One photo -> one GeoJSON Feature dict.

    Reuses load_image + compute_ndvi_from_image + classify_zones. If overlay_dir
    is given, also writes a colorized per-photo NDVI PNG there.
    """
    image = load_image(path)
    ndvi = compute_ndvi_from_image(image, gamma=gamma, leakage_k=leakage_k)
    zones = classify_zones(ndvi, block_size=block_size)

    zone_counts = {"healthy": 0, "moderate": 0, "stressed": 0}
    for z in zones:
        zone_counts[z["status"]] = zone_counts.get(z["status"], 0) + 1

    gps = geo.read_gps(path)
    gps_ok = gps is not None

    overlay_png = None
    if overlay_dir:
        os.makedirs(overlay_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(path))[0]
        overlay_png = os.path.join(overlay_dir, f"{base}.png")
        save_ndvi_image(ndvi, overlay_png)

    properties = {
        "filename": os.path.basename(path),
        "altitude_m": gps.get("alt") if gps_ok else None,
        "timestamp": gps.get("timestamp") if gps_ok else None,
        "mean_ndvi": round(float(ndvi.mean()), 4),
        "min_ndvi": round(float(ndvi.min()), 4),
        "max_ndvi": round(float(ndvi.max()), 4),
        "status": _image_status(zone_counts),
        "n_zones": len(zones),
        "zone_status_counts": zone_counts,
        "overlay_png": overlay_png,
        "gps_ok": gps_ok,
    }

    geometry = None
    if gps_ok:
        # GeoJSON is lon, lat order
        geometry = {"type": "Point", "coordinates": [gps["lon"], gps["lat"]]}

    return {"type": "Feature", "geometry": geometry, "properties": properties}


def _bbox(features):
    """[min_lon, min_lat, max_lon, max_lat] over geotagged features, or None."""
    coords = [f["geometry"]["coordinates"] for f in features
              if f.get("geometry")]
    if not coords:
        return None
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return [min(lons), min(lats), max(lons), max(lats)]


def process_directory(input_dir, gamma=0.8, leakage_k=0.6, block_size=64,
                      patterns=DEFAULT_PATTERNS, overlay_dir=None,
                      flight_date=None, generated=None):
    """Process every photo in input_dir into a GeoJSON FeatureCollection.

    Untagged photos are still processed (NDVI computed) but kept with a null
    geometry and counted in metadata.n_untagged so nothing is silently dropped.

    flight_date / generated are passed in (no Date.now() here) so the result is
    reproducible; the orchestrator stamps real timestamps.
    """
    paths = []
    for pat in patterns:
        paths.extend(glob.glob(os.path.join(input_dir, pat)))
    paths = sorted(set(paths))

    features = []
    for p in paths:
        features.append(process_image(
            p, gamma=gamma, leakage_k=leakage_k,
            block_size=block_size, overlay_dir=overlay_dir,
        ))

    tagged = [f for f in features if f.get("geometry")]
    n_untagged = len(features) - len(tagged)

    return {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "flight_date": flight_date,
            "n_images": len(features),
            "n_untagged": n_untagged,
            "bbox": _bbox(tagged),
            "params": {
                "gamma": gamma,
                "leakage_k": leakage_k,
                "block_size": block_size,
            },
            "generated": generated,
            "source": "gps_tiles",
        },
    }
