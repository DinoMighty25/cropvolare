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

try:
    import cv2
except ImportError:
    cv2 = None

from . import geo
from .ndvi import (
    apply_flatfield,
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


def _sharpness(image):
    """Variance of the Laplacian - low for grounded/defocused frames."""
    if cv2 is None:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def process_image(path, gamma=0.8, leakage_k=0.6, block_size=64,
                  overlay_dir=None, gain=None):
    """One photo -> one GeoJSON Feature dict.

    Reuses load_image + compute_ndvi_from_image + classify_zones. gain, if
    given, is a flat-field map (see ndvi.build_flatfield) applied before NDVI
    to remove per-channel lens shading. If overlay_dir is given, also writes a
    colorized per-photo NDVI PNG there.
    """
    image = load_image(path)
    if gain is not None:
        image = apply_flatfield(image, gain)
    sharpness = _sharpness(image)
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
        "sharpness": round(sharpness, 1) if sharpness is not None else None,
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
                      flight_date=None, generated=None, gain=None,
                      min_sharpness=0.0):
    """Process every photo in input_dir into a GeoJSON FeatureCollection.

    Untagged photos are still processed (NDVI computed) but kept with a null
    geometry and counted in metadata.n_untagged so nothing is silently dropped.
    gain, if given, flat-fields every frame before NDVI. min_sharpness > 0
    drops frames below that sharpness (grounded/defocused captures - focus is
    locked at infinity, so near-ground frames are featureless blur); dropped
    frames are counted in metadata.n_filtered.

    flight_date / generated are passed in (no Date.now() here) so the result is
    reproducible; the orchestrator stamps real timestamps.
    """
    paths = []
    for pat in patterns:
        paths.extend(glob.glob(os.path.join(input_dir, pat)))
    paths = sorted(set(paths))

    features = []
    unreadable = []
    n_filtered = 0
    for p in paths:
        try:
            feature = process_image(
                p, gamma=gamma, leakage_k=leakage_k,
                block_size=block_size, overlay_dir=overlay_dir, gain=gain,
            )
        except Exception as exc:  # noqa: BLE001 - a bad frame must not kill a whole flight
            # real flights produce the occasional 0-byte / truncated JPEG
            # (e.g. a mid-write frame when capture stops); skip and count it
            unreadable.append(os.path.basename(p))
            print(f"  skipped unreadable frame {os.path.basename(p)}: {exc}")
            continue

        sharpness = feature["properties"]["sharpness"]
        if min_sharpness and sharpness is not None and sharpness < min_sharpness:
            n_filtered += 1
            continue
        features.append(feature)

    tagged = [f for f in features if f.get("geometry")]
    n_untagged = len(features) - len(tagged)

    return {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "flight_date": flight_date,
            "n_images": len(features),
            "n_unreadable": len(unreadable),
            "n_filtered": n_filtered,
            "n_untagged": n_untagged,
            "bbox": _bbox(tagged),
            "params": {
                "gamma": gamma,
                "leakage_k": leakage_k,
                "block_size": block_size,
                "min_sharpness": min_sharpness,
                "flatfield": gain is not None,
            },
            "generated": generated,
            "source": "gps_tiles",
        },
    }
