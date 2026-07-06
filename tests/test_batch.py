from cropvolare import batch


def test_process_image_high_ndvi(ndvi_jpeg_factory):
    path = ndvi_jpeg_factory(40.0, -88.0, nir=255, red=0)
    feat = batch.process_image(path)
    assert feat["type"] == "Feature"
    assert feat["geometry"]["type"] == "Point"
    # GeoJSON lon, lat order
    lon, lat = feat["geometry"]["coordinates"]
    assert abs(lat - 40.0) < 1e-4
    assert abs(lon - (-88.0)) < 1e-4
    assert feat["properties"]["mean_ndvi"] > 0.5
    assert feat["properties"]["status"] == "healthy"
    assert feat["properties"]["gps_ok"] is True


def test_process_image_untagged_has_null_geometry(tmp_path):
    import cv2
    import numpy as np
    p = str(tmp_path / "x.jpg")
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, :, 0] = 200
    cv2.imwrite(p, img)
    feat = batch.process_image(p)
    assert feat["geometry"] is None
    assert feat["properties"]["gps_ok"] is False


def test_process_directory_collection(geotagged_dir):
    fc = batch.process_directory(str(geotagged_dir))
    assert fc["type"] == "FeatureCollection"
    meta = fc["metadata"]
    assert meta["n_images"] == 5          # 4 geotagged + 1 untagged
    assert meta["n_untagged"] == 1
    assert meta["source"] == "gps_tiles"
    # bbox computed only over the 4 tagged photos
    assert meta["bbox"] is not None
    min_lon, min_lat, max_lon, max_lat = meta["bbox"]
    assert min_lon <= max_lon and min_lat <= max_lat


def test_process_directory_skips_unreadable(geotagged_dir):
    # a 0-byte / truncated JPEG (like a mid-write frame at capture stop) must
    # be skipped and counted, not crash the whole flight
    open(str(geotagged_dir / "frame_bad.jpg"), "w").close()  # 0 bytes
    fc = batch.process_directory(str(geotagged_dir))
    assert fc["metadata"]["n_unreadable"] == 1
    assert fc["metadata"]["n_images"] == 5   # the 5 good ones still processed


def test_process_directory_overlays(geotagged_dir, tmp_path):
    overlay_dir = tmp_path / "overlays"
    fc = batch.process_directory(str(geotagged_dir), overlay_dir=str(overlay_dir))
    tagged = [f for f in fc["features"] if f["properties"]["gps_ok"]]
    assert all(f["properties"]["overlay_png"] for f in tagged)
    import os
    assert len(os.listdir(overlay_dir)) == 5
