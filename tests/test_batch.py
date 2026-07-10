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


def test_sharpness_in_properties(ndvi_jpeg_factory):
    path = ndvi_jpeg_factory(40.0, -88.0)
    feat = batch.process_image(path)
    assert feat["properties"]["sharpness"] is not None
    assert feat["properties"]["sharpness"] >= 0.0


def test_min_sharpness_filters_featureless(geotagged_dir, tmp_path):
    import cv2
    import numpy as np
    # add one TEXTURED frame; the fixture's 5 uniform frames have ~0 sharpness
    rng = np.random.RandomState(7)
    textured = rng.randint(0, 256, (64, 64, 3)).astype(np.uint8)
    cv2.imwrite(str(geotagged_dir / "textured.jpg"), textured)

    fc = batch.process_directory(str(geotagged_dir), min_sharpness=50.0)
    names = [f["properties"]["filename"] for f in fc["features"]]
    assert names == ["textured.jpg"]           # only the sharp frame survives
    assert fc["metadata"]["n_filtered"] == 5   # uniform frames filtered+counted
    assert fc["metadata"]["params"]["min_sharpness"] == 50.0


def test_flatfield_removes_ndvi_bullseye(tmp_path):
    import cv2
    import numpy as np
    from cropvolare.ndvi import build_flatfield

    # per-channel lens shading: NIR (blue ch) falls off toward the corners
    # HARDER than red - exactly the mismatch that made the real flight's NDVI
    # a radial bullseye (brightness-only vignetting cancels in the ratio)
    yy, xx = np.mgrid[0:240, 0:320]
    r2 = ((yy / 240 - 0.5) ** 2 + (xx / 320 - 0.5) ** 2)
    r2 = r2 / r2.max()
    fall_nir = 1.0 - 0.5 * r2
    fall_red = 1.0 - 0.2 * r2

    def shade(nir0, red0):
        img = np.zeros((240, 320, 3), np.float64)
        img[:, :, 0] = nir0 * fall_nir
        img[:, :, 2] = red0 * fall_red
        return np.clip(img, 0, 255).astype(np.uint8)

    flat = shade(200, 200)                     # white target through the lens
    scene = shade(150, 90)                     # uniform "crop"
    scene_path = str(tmp_path / "scene.png")   # png: lossless
    cv2.imwrite(scene_path, scene)

    gain = build_flatfield([flat], smooth_sigma=5)

    raw = batch.process_image(scene_path)
    fixed = batch.process_image(scene_path, gain=gain)
    spread_raw = raw["properties"]["max_ndvi"] - raw["properties"]["min_ndvi"]
    spread_fixed = (fixed["properties"]["max_ndvi"]
                    - fixed["properties"]["min_ndvi"])
    # the radial NDVI gradient collapses once the shading is corrected
    assert spread_fixed < spread_raw / 3


def test_brightness_and_source_path_in_properties(ndvi_jpeg_factory):
    path = ndvi_jpeg_factory(40.0, -88.0, nir=200, red=80)
    feat = batch.process_image(path)
    p = feat["properties"]
    assert p["source_path"] == path
    assert 80 < p["brightness"] < 120   # (200+0+80)/3 ~ 93 mean


def test_min_brightness_filters_dark_frames(geotagged_dir, tmp_path):
    import cv2
    import numpy as np
    # a near-black frame: the post-landing / carry-home signature
    dark = np.full((64, 64, 3), 6, dtype=np.uint8)
    cv2.imwrite(str(geotagged_dir / "dark.jpg"), dark)

    fc = batch.process_directory(str(geotagged_dir), min_brightness=25.0)
    names = [f["properties"]["filename"] for f in fc["features"]]
    assert "dark.jpg" not in names
    assert fc["metadata"]["n_filtered"] == 1
    assert fc["metadata"]["params"]["min_brightness"] == 25.0
    assert len(names) == 5              # the normal fixture frames survive


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


# --- process_scale: the on-device fast path ---------------------------------

def test_process_scale_ndvi_within_tolerance(tmp_path):
    # textured (non-uniform) frames so downscaling actually averages pixels
    import cv2
    import numpy as np
    rng = np.random.default_rng(42)
    d = tmp_path / "tex"
    d.mkdir()
    for i in range(3):
        img = np.zeros((128, 128, 3), np.uint8)
        img[:, :, 0] = rng.integers(120, 240, (128, 128))
        img[:, :, 2] = rng.integers(20, 120, (128, 128))
        cv2.imwrite(str(d / f"t{i}.jpg"), img)

    full = batch.process_directory(str(d))
    half = batch.process_directory(str(d), process_scale=0.5)
    assert half["metadata"]["params"]["process_scale"] == 0.5
    assert len(full["features"]) == len(half["features"]) == 3
    for a, b in zip(full["features"], half["features"]):
        assert abs(a["properties"]["mean_ndvi"]
                   - b["properties"]["mean_ndvi"]) < 0.02
        # sharpness is measured BEFORE downscaling, so min_sharpness
        # thresholds mean the same thing at every scale
        assert a["properties"]["sharpness"] == b["properties"]["sharpness"]


def test_progress_fn_fires_per_frame(geotagged_dir):
    calls = []
    batch.process_directory(str(geotagged_dir),
                            progress_fn=lambda done, total: calls.append((done, total)))
    assert len(calls) == 5                       # every frame reported
    assert calls[-1] == (5, 5)
    assert [c[0] for c in calls] == [1, 2, 3, 4, 5]
