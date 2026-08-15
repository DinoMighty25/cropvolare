import cv2
import numpy as np
from cropvolare.ndvi import (
    apply_flatfield,
    build_flatfield,
    classify_zones,
    colorize_ndvi,
    compute_ndvi,
    compute_ndvi_from_image,
    compute_vari,
    correct_leakage,
    capture_frame,
    extract_channels,
    remove_gamma,
    save_ndvi_tiff,
    solve_leakage_k,
)


class _FakeCam:
    """Minimal stand-in for a started Picamera2: returns a fixed RGB array."""

    def __init__(self, rgb):
        self._rgb = rgb

    def capture_array(self):
        return self._rgb


def test_capture_frame_keeps_channel_order():
    # picamera2 "RGB888" is already [B, G, R] in memory - capture_frame must
    # NOT convert, or red and blue swap and NDVI sign-flips
    arr = np.zeros((2, 2, 3), dtype=np.uint8)
    arr[:, :, 0] = 200  # channel 0 (blue slot = NIR under the red filter)
    arr[:, :, 2] = 50   # channel 2 (red slot)
    out = capture_frame(_FakeCam(arr))
    assert out[0, 0, 0] == 200
    assert out[0, 0, 2] == 50


def test_lock_exposure_freezes_settled_values():
    from cropvolare.ndvi import lock_exposure

    class _MeteringCam:
        def __init__(self):
            self.controls = None

        def capture_metadata(self):
            return {"ExposureTime": 8000, "AnalogueGain": 2.5}

        def set_controls(self, controls):
            self.controls = controls

    cam = _MeteringCam()
    exposure, gain = lock_exposure(cam)
    assert (exposure, gain) == (8000, 2.5)
    assert cam.controls == {"AeEnable": False, "ExposureTime": 8000,
                            "AnalogueGain": 2.5}


def test_extract_channels_shape():
    img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    nir, red = extract_channels(img)
    assert nir.shape == (100, 100)
    assert red.shape == (100, 100)


def test_extract_channels_normalized():
    img = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
    nir, red = extract_channels(img)
    assert nir.min() >= 0.0 and nir.max() <= 1.0
    assert red.min() >= 0.0 and red.max() <= 1.0


def test_all_nir_gives_plus_one():
    nir = np.ones((10, 10))
    red = np.zeros((10, 10))
    np.testing.assert_allclose(compute_ndvi(nir, red), 1.0)


def test_all_red_gives_minus_one():
    nir = np.zeros((10, 10))
    red = np.ones((10, 10))
    np.testing.assert_allclose(compute_ndvi(nir, red), -1.0)


def test_equal_channels_gives_zero():
    arr = np.full((10, 10), 0.5)
    np.testing.assert_allclose(compute_ndvi(arr, arr), 0.0)


def test_both_zero_no_crash():
    z = np.zeros((10, 10))
    np.testing.assert_allclose(compute_ndvi(z, z), 0.0)


def test_from_image_raw():
    # gamma off, leakage off -> plain (NIR - Red) / (NIR + Red)
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, :, 0] = 200  # NIR (blue channel)
    img[:, :, 2] = 100  # red channel
    ndvi = compute_ndvi_from_image(img, gamma=1.0, leakage_k=0.0)
    expected = (200 - 100) / (200 + 100)
    np.testing.assert_allclose(ndvi, expected, atol=1e-6)


# --- gamma -----------------------------------------------------------------

def test_remove_gamma_identity_at_endpoints():
    ch = np.array([0.0, 1.0])
    np.testing.assert_allclose(remove_gamma(ch, gamma=0.8), [0.0, 1.0])


def test_remove_gamma_disabled():
    ch = np.array([0.3, 0.7])
    np.testing.assert_allclose(remove_gamma(ch, gamma=1.0), ch)


def test_remove_gamma_changes_midtones():
    ch = np.array([0.5])
    out = remove_gamma(ch, gamma=0.8)
    assert not np.isclose(out[0], 0.5)
    assert 0.0 <= out[0] <= 1.0


# --- leakage ---------------------------------------------------------------

def test_correct_leakage_subtracts_nir():
    nir = np.full((4, 4), 0.5)
    red = np.full((4, 4), 0.5)
    out = correct_leakage(nir, red, k=0.6)
    np.testing.assert_allclose(out, 0.2)


def test_correct_leakage_clips_to_zero():
    nir = np.ones((4, 4))
    red = np.full((4, 4), 0.1)
    out = correct_leakage(nir, red, k=0.6)
    np.testing.assert_allclose(out, 0.0)


def test_correct_leakage_disabled():
    nir = np.full((4, 4), 0.5)
    red = np.full((4, 4), 0.3)
    np.testing.assert_allclose(correct_leakage(nir, red, k=0.0), red)


def test_leakage_raises_ndvi_over_raw():
    # subtracting NIR from red increases (NIR - Red), so NDVI goes up
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    img[:, :, 0] = 180  # NIR
    img[:, :, 2] = 120  # red
    raw = compute_ndvi_from_image(img, gamma=1.0, leakage_k=0.0).mean()
    corrected = compute_ndvi_from_image(img, gamma=1.0, leakage_k=0.6).mean()
    assert corrected > raw


# --- VARI ------------------------------------------------------------------

def test_vari_formula():
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    img[:, :, 0] = 50   # blue
    img[:, :, 1] = 150  # green
    img[:, :, 2] = 100  # red
    expected = (150 - 100) / (150 + 100 - 50)
    np.testing.assert_allclose(compute_vari(img), expected, atol=1e-6)


def test_vari_no_crash_on_zero_denom():
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    np.testing.assert_allclose(compute_vari(img), 0.0)


# --- flat-field ------------------------------------------------------------

def test_flatfield_flattens_vignette():
    # a darker corner should be brightened back toward the mean
    # (smooth_sigma=0: smoothing would wash out a single-pixel corner on a
    # frame this tiny - the smoothing path has its own test below)
    frame = np.full((10, 10, 3), 200, dtype=np.uint8)
    frame[0, 0] = 100
    gain = build_flatfield([frame, frame], smooth_sigma=0)
    corrected = apply_flatfield(frame, gain)
    # corrected corner should be closer to the bulk value than the raw corner
    assert corrected[0, 0, 0] > frame[0, 0, 0]


def test_flatfield_uniform_is_noop():
    frame = np.full((6, 6, 3), 180, dtype=np.uint8)
    gain = build_flatfield([frame])
    np.testing.assert_allclose(gain, 1.0)


def test_apply_flatfield_corrects_to_target():
    # a frame with a dark corner; its own gain map should flatten it back
    frame = np.full((8, 8, 3), 200, dtype=np.uint8)
    frame[0, 0] = 120
    gain = build_flatfield([frame], smooth_sigma=0)
    corrected = apply_flatfield(frame, gain)
    # every pixel should land near the frame's mean brightness
    target = int(round(frame.mean()))
    assert abs(int(corrected[0, 0, 0]) - target) <= 1
    assert abs(int(corrected[4, 4, 0]) - target) <= 1


def test_flatfield_smoothing_ignores_texture():
    # noisy texture in the flat frames shouldn't imprint on the gain map
    rng = np.random.RandomState(42)
    frame = (200 + rng.randint(-30, 31, (200, 200, 3))).astype(np.uint8)
    noisy_gain = build_flatfield([frame], smooth_sigma=0)
    smooth_gain = build_flatfield([frame], smooth_sigma=25)
    assert smooth_gain.std() < noisy_gain.std() / 5


def test_flatfield_save_load_roundtrip_resizes(tmp_path):
    from cropvolare.ndvi import load_flatfield, save_flatfield
    # radial vignette on a 640x480 flat: corners at ~60% of center
    yy, xx = np.mgrid[0:480, 0:640]
    radius = np.sqrt((yy / 480 - 0.5) ** 2 + (xx / 640 - 0.5) ** 2)
    falloff = 1.0 - 0.55 * (radius / radius.max()) ** 2
    flat = np.clip(220 * falloff, 0, 255).astype(np.uint8)
    flat = np.dstack([flat] * 3)

    gain = build_flatfield([flat], smooth_sigma=10)
    path = str(tmp_path / "gain.npy")
    save_flatfield(gain, path, max_px=96)

    loaded = load_flatfield(path)
    assert max(loaded.shape[:2]) <= 96          # stored downscaled
    assert loaded.dtype == np.float32

    # applying the small stored gain to the full-size vignetted frame
    # flattens it: corner and center end up close
    corrected = apply_flatfield(flat, loaded)
    center = corrected[230:250, 310:330, 0].mean()
    corner = corrected[0:20, 0:20, 0].mean()
    assert abs(center - corner) < 12            # raw difference was ~120


def test_apply_flatfield_clips_to_uint8():
    frame = np.full((4, 4, 3), 200, dtype=np.uint8)
    gain = np.full((4, 4, 3), 5.0)  # huge gain would overflow
    corrected = apply_flatfield(frame, gain)
    assert corrected.dtype == np.uint8
    assert corrected.max() <= 255


# --- grey-card leakage calibration ------------------------------------------

def test_solve_leakage_k_recovers_bleed():
    # grey card: equal true reflectance, but red channel reads 60% higher
    # from NIR bleed -> k should come out at exactly 0.6
    img = np.zeros((40, 40, 3), dtype=np.uint8)
    img[:, :, 0] = 100   # NIR (blue channel)
    img[:, :, 2] = 160   # red channel = nir + 0.6*nir
    k = solve_leakage_k(img, gamma=1.0)
    assert abs(k - 0.6) < 1e-6
    # and with that k the card reads NDVI ~ 0 (the calibration target)
    ndvi = compute_ndvi_from_image(img, gamma=1.0, leakage_k=k)
    assert abs(float(ndvi.mean())) < 1e-6


def test_solve_leakage_k_clips_at_zero():
    # red below NIR means no bleed to correct -> k = 0, never negative
    img = np.zeros((40, 40, 3), dtype=np.uint8)
    img[:, :, 0] = 150
    img[:, :, 2] = 100
    assert solve_leakage_k(img, gamma=1.0) == 0.0


def test_solve_leakage_k_rejects_black_frame():
    import pytest
    with pytest.raises(ValueError):
        solve_leakage_k(np.zeros((40, 40, 3), dtype=np.uint8), gamma=1.0)


# --- 16-bit TIFF round-trip -----------------------------------------------

def test_tiff_roundtrip(tmp_path):
    ndvi = np.linspace(-1.0, 1.0, 64).reshape(8, 8)
    path = str(tmp_path / "ndvi.tiff")
    save_ndvi_tiff(ndvi, path)
    arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    assert arr.dtype == np.uint16
    recovered = arr.astype(np.float64) / 65535.0 * 2.0 - 1.0
    # 16-bit quantization keeps NDVI well within 1e-3
    np.testing.assert_allclose(recovered, ndvi, atol=1e-3)


# --- zones + colormap ------------------------------------------------------

def test_zones_healthy():
    ndvi = np.full((64, 64), 0.7)
    zones = classify_zones(ndvi, block_size=64)
    assert len(zones) == 1
    assert zones[0]["status"] == "healthy"


def test_zones_stressed():
    ndvi = np.full((64, 64), 0.1)
    zones = classify_zones(ndvi, block_size=64)
    assert zones[0]["status"] == "stressed"


def test_zones_grid_count():
    ndvi = np.full((128, 128), 0.5)
    zones = classify_zones(ndvi, block_size=64)
    assert len(zones) == 4


def test_colorize_shape_and_type():
    ndvi = np.linspace(-1, 1, 64).reshape(8, 8)
    out = colorize_ndvi(ndvi)
    assert out.shape == (8, 8, 3)
    assert out.dtype == np.uint8


def test_ndvi_chain_is_float32():
    # float32 end to end is what fits flights in the Pi Zero's 512 MB
    img = np.zeros((8, 8, 3), np.uint8)
    img[:, :, 0] = 200
    img[:, :, 2] = 80
    out = compute_ndvi_from_image(img)
    assert out.dtype == np.float32


# --- two-point calibration (grey-card-free) --------------------------------

def _synthetic_rig(nir_ref, red_ref, k=1.43, red_gain=2.88, gain=0.30,
                   gamma=0.8, size=64):
    """Frame from a rig with known k and red_gain, gamma-encoded like a JPEG."""
    import numpy as np
    nir = gain * nir_ref
    red = gain * (red_gain * red_ref + k * nir_ref)
    enc = lambda x: np.clip(np.power(np.clip(x, 0, 1), gamma) * 255, 0, 255)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, :, 0] = enc(nir)
    img[:, :, 1] = enc(nir)
    img[:, :, 2] = enc(red)
    return img


def test_solve_two_point_recovers_both_constants():
    """Canopy + bare soil in one frame pins k AND red_gain."""
    import numpy as np
    from cropvolare.ndvi import solve_two_point
    veg = _synthetic_rig(0.75, 0.107, size=64)    # true NDVI ~0.75
    soil = _synthetic_rig(0.30, 0.222, size=64)   # true NDVI ~0.15
    frame = np.vstack([veg, soil])                # half canopy, half soil
    # explicit mask: a two-valued frame makes percentile splits degenerate
    mask = np.zeros(frame.shape[:2], dtype=bool)
    mask[:veg.shape[0], :] = True
    k, gain = solve_two_point(frame, mask=mask)
    assert abs(k - 1.43) < 0.25, f"k={k}"
    assert abs(gain - 2.88) < 0.5, f"red_gain={gain}"


def test_solve_two_point_percentile_split_on_mixed_frame():
    """Percentile split works on a frame with a real spread of cover."""
    import numpy as np
    from cropvolare.ndvi import solve_two_point
    bands = [_synthetic_rig(n, r, size=32) for n, r in
             [(0.75, 0.107), (0.68, 0.12), (0.50, 0.18), (0.30, 0.222)]]
    frame = np.vstack(bands)
    k, gain = solve_two_point(frame, veg_pct=25, soil_pct=75)
    assert abs(k - 1.43) < 0.35, f"k={k}"
    assert abs(gain - 2.88) < 0.7, f"red_gain={gain}"


def test_solve_two_point_needs_both_populations():
    """A uniform frame cannot supply two independent equations."""
    import pytest
    from cropvolare.ndvi import solve_two_point
    flat = _synthetic_rig(0.75, 0.107, size=128)
    with pytest.raises(ValueError):
        solve_two_point(flat, veg_pct=0.01, soil_pct=99.99)


def test_correct_leakage_does_not_saturate_vegetation():
    """The regression that made every healthy pixel read exactly 1.0."""
    import numpy as np
    from cropvolare.ndvi import compute_ndvi_from_image
    veg = _synthetic_rig(0.80, 0.05)
    ndvi = compute_ndvi_from_image(veg, gamma=0.8, leakage_k=1.43, red_gain=2.88)
    assert float((ndvi > 0.999).mean()) == 0.0, "healthy canopy pinned at 1.0"
    assert 0.5 < float(ndvi.mean()) < 0.95


def test_channel_nir_ratio_predicts_k():
    from cropvolare.ndvi import channel_nir_ratio
    frame = _synthetic_rig(0.5, 0.2)
    # green and blue are both pure NIR in this synthetic rig -> ratio 1.0
    assert abs(channel_nir_ratio(frame) - 1.0) < 0.05
