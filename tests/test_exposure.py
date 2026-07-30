"""Tests for named exposure presets - the cross-flight comparability mechanism."""

import json

import numpy as np
import pytest

from cropvolare import exposure


# --------------------------------------------------------------------------
# preset validation
# --------------------------------------------------------------------------

def test_make_and_describe_roundtrip():
    p = exposure.make_preset(1200, 1.0, colour_gains=(0.88, 0.97), note="clear")
    assert p["exposure_us"] == 1200
    assert p["analogue_gain"] == 1.0
    assert p["colour_gains"] == [0.88, 0.97]
    assert p["note"] == "clear"
    assert "measured_at" in p
    text = exposure.describe(p)
    assert "1200 us" in text and "clear" in text


def test_get_preset_returns_saved_values():
    cfg = {"exposure_presets": {"full_sun": {"exposure_us": 900,
                                             "analogue_gain": 1.0}}}
    assert exposure.get_preset(cfg, "full_sun")["exposure_us"] == 900


def test_missing_preset_names_the_available_ones():
    """The realistic failure is a typo in the field, so the error must be useful."""
    cfg = {"exposure_presets": {"full_sun": {"exposure_us": 900,
                                             "analogue_gain": 1.0}}}
    with pytest.raises(exposure.PresetError) as exc:
        exposure.get_preset(cfg, "fullsun")
    assert "full_sun" in str(exc.value)


def test_missing_preset_on_empty_config_suggests_measuring_one():
    with pytest.raises(exposure.PresetError) as exc:
        exposure.get_preset({}, "full_sun")
    assert "set_exposure_preset" in str(exc.value)


@pytest.mark.parametrize("preset", [
    {"analogue_gain": 1.0},                              # no exposure
    {"exposure_us": 1000},                               # no gain
    {"exposure_us": 5, "analogue_gain": 1.0},            # implausibly short
    {"exposure_us": 999_999, "analogue_gain": 1.0},      # implausibly long
    {"exposure_us": 1000, "analogue_gain": 0.1},         # gain below unity
    {"exposure_us": 1000, "analogue_gain": 99.0},        # absurd gain
    {"exposure_us": "bright", "analogue_gain": 1.0},     # non-numeric
    {"exposure_us": 1000, "analogue_gain": 1.0, "colour_gains": [1.0]},
])
def test_invalid_presets_are_rejected(preset):
    with pytest.raises(exposure.PresetError):
        exposure.validate(preset, "bad")


def test_save_preset_preserves_the_rest_of_the_config(tmp_path):
    """The config also holds calibration constants; saving must not lose them."""
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "ndvi": {"leakage_k": 2.0, "gamma": 0.8},
        "calibration": {"flatfield_path": "calibration/gain.npy"},
    }))

    exposure.save_preset(str(path), "full_sun",
                         exposure.make_preset(1200, 1.0))

    cfg = json.loads(path.read_text())
    assert cfg["ndvi"]["leakage_k"] == 2.0
    assert cfg["calibration"]["flatfield_path"] == "calibration/gain.npy"
    assert cfg["exposure_presets"]["full_sun"]["exposure_us"] == 1200


def test_save_preset_adds_without_clobbering_siblings(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{}")
    exposure.save_preset(str(path), "full_sun", exposure.make_preset(900, 1.0))
    exposure.save_preset(str(path), "overcast", exposure.make_preset(8000, 1.5))
    cfg = json.loads(path.read_text())
    assert set(cfg["exposure_presets"]) == {"full_sun", "overcast"}


def test_save_preset_creates_a_missing_config(tmp_path):
    path = tmp_path / "new.json"
    exposure.save_preset(str(path), "full_sun", exposure.make_preset(900, 1.0))
    assert json.loads(path.read_text())["exposure_presets"]["full_sun"]


def test_preset_names_sorted():
    cfg = {"exposure_presets": {"overcast": {"exposure_us": 8000,
                                             "analogue_gain": 1.0},
                                "full_sun": {"exposure_us": 900,
                                             "analogue_gain": 1.0}}}
    assert exposure.preset_names(cfg) == ["full_sun", "overcast"]


# --------------------------------------------------------------------------
# clipping analysis
# --------------------------------------------------------------------------

def _frame(nir_value, red_value, shape=(40, 40)):
    """BGR frame: channel 0 is the NIR-bearing channel, channel 2 is red."""
    f = np.zeros((*shape, 3), dtype=np.uint8)
    f[:, :, 0] = nir_value
    f[:, :, 2] = red_value
    return f


def test_saturation_counts_the_right_channels():
    frame = _frame(255, 100)
    sat = exposure.channel_saturation(frame)
    assert sat["nir"] == pytest.approx(1.0)
    assert sat["red"] == pytest.approx(0.0)


def test_well_exposed_frame_passes():
    level, msg = exposure.clipping_verdict(
        exposure.channel_saturation(_frame(180, 90)))
    assert level == "ok"
    assert "no meaningful clipping" in msg


def test_blown_nir_is_rejected_with_advice():
    """Saturated NIR makes healthy crop read as stressed - the worst failure."""
    level, msg = exposure.clipping_verdict(
        exposure.channel_saturation(_frame(255, 90)))
    assert level == "bad"
    assert "--exposure" in msg


def test_lens_cap_dark_frame_warns():
    level, _ = exposure.clipping_verdict(
        exposure.channel_saturation(_frame(3, 2)))
    assert level == "warn"


def test_small_amount_of_clipping_warns_but_is_usable():
    frame = _frame(180, 90)
    frame[:2, :, 0] = 255          # ~5% of pixels
    level, _ = exposure.clipping_verdict(exposure.channel_saturation(frame))
    assert level in ("warn", "bad")


# --------------------------------------------------------------------------
# capture metadata + comparability
# --------------------------------------------------------------------------

def test_capture_meta_roundtrip(tmp_path):
    meta = exposure.capture_meta("full_sun", 1200, 1.0,
                                 colour_gains=(0.88, 0.97))
    exposure.write_capture_meta(str(tmp_path), meta)
    back = exposure.read_capture_meta(str(tmp_path))
    assert back["preset"] == "full_sun"
    assert back["locked"] == "preset"
    assert back["exposure_us"] == 1200


def test_read_capture_meta_missing_returns_none(tmp_path):
    assert exposure.read_capture_meta(str(tmp_path)) is None


def test_matching_presets_are_comparable():
    a = exposure.capture_meta("full_sun", 1200, 1.0)
    b = exposure.capture_meta("full_sun", 1200, 1.0)
    ok, reason = exposure.comparable(a, b)
    assert ok
    assert "full_sun" in reason


def test_auto_exposure_flights_are_not_comparable():
    """The twin-flight test is meaningless if AE settled separately per flight."""
    a = exposure.capture_meta(None, 1200, 1.0, locked="auto")
    b = exposure.capture_meta(None, 1400, 1.0, locked="auto")
    ok, reason = exposure.comparable(a, b)
    assert not ok
    assert "auto-exposure" in reason


def test_different_exposure_values_are_not_comparable():
    a = exposure.capture_meta("full_sun", 1200, 1.0)
    b = exposure.capture_meta("full_sun", 1500, 1.0)
    ok, reason = exposure.comparable(a, b)
    assert not ok
    assert "exposure differs" in reason


def test_different_gain_is_not_comparable():
    a = exposure.capture_meta("full_sun", 1200, 1.0)
    b = exposure.capture_meta("full_sun", 1200, 2.0)
    ok, reason = exposure.comparable(a, b)
    assert not ok
    assert "gain differs" in reason


def test_different_preset_names_are_not_comparable():
    a = exposure.capture_meta("full_sun", 1200, 1.0)
    b = exposure.capture_meta("overcast", 1200, 1.0)
    ok, reason = exposure.comparable(a, b)
    assert not ok
    assert "different presets" in reason


def test_missing_metadata_is_not_comparable():
    a = exposure.capture_meta("full_sun", 1200, 1.0)
    ok, reason = exposure.comparable(a, None)
    assert not ok
    assert "missing" in reason
