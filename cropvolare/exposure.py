"""
Named exposure presets: the mechanism that makes flights comparable ACROSS days.

lock_exposure() in ndvi.py freezes auto-exposure at whatever it settled on, so
every frame WITHIN one flight matches. That is not enough for the questions this
project exists to answer. Two flights an hour apart settle on different values
because the light differs, so their NDVI differs for a reason that has nothing
to do with the crop - and "is this patch worse than last week?" becomes
unanswerable.

A preset is a measured (exposure_us, analogue_gain, colour_gains) triple stored
in the config under a name like "full_sun". Measure it once in representative
light with scripts/set_exposure_preset.py; every later flight hard-locks those
exact numbers, so a difference between two flights is a difference in the field.

Pure dict/JSON handling, no camera imports - the Pi-only bits live in the CLI.
"""

import json
import os
from datetime import datetime

CONFIG_SECTION = "exposure_presets"

# Sanity bounds. The IMX708 accepts a much wider exposure range than this, but
# anything outside it means a measurement went wrong (dark lens cap, indoor
# light saved as a "full sun" preset) and flying it would waste a field day.
MIN_EXPOSURE_US = 20
MAX_EXPOSURE_US = 200_000
MIN_GAIN = 1.0
MAX_GAIN = 16.0


class PresetError(ValueError):
    """A preset is missing, malformed, or outside sane bounds."""


def load_presets(cfg):
    """Every preset in a loaded config dict, as {name: preset}."""
    presets = cfg.get(CONFIG_SECTION) or {}
    if not isinstance(presets, dict):
        raise PresetError(f"config {CONFIG_SECTION!r} must be an object")
    return presets


def preset_names(cfg):
    return sorted(load_presets(cfg))


def validate(preset, name="<preset>"):
    """Raise PresetError unless the preset is complete and in range.

    Returns the preset so callers can validate-and-assign in one expression.
    """
    if not isinstance(preset, dict):
        raise PresetError(f"preset {name!r} must be an object")

    for key in ("exposure_us", "analogue_gain"):
        if key not in preset:
            raise PresetError(f"preset {name!r} is missing {key!r}")

    try:
        exposure = int(preset["exposure_us"])
        gain = float(preset["analogue_gain"])
    except (TypeError, ValueError) as exc:
        raise PresetError(f"preset {name!r} has non-numeric values: {exc}") from exc

    if not MIN_EXPOSURE_US <= exposure <= MAX_EXPOSURE_US:
        raise PresetError(
            f"preset {name!r} exposure_us={exposure} outside "
            f"{MIN_EXPOSURE_US}-{MAX_EXPOSURE_US} - re-measure it")
    if not MIN_GAIN <= gain <= MAX_GAIN:
        raise PresetError(
            f"preset {name!r} analogue_gain={gain} outside "
            f"{MIN_GAIN}-{MAX_GAIN} - re-measure it")

    gains = preset.get("colour_gains")
    if gains is not None:
        if len(gains) != 2:
            raise PresetError(f"preset {name!r} colour_gains must be [red, blue]")
        try:
            [float(g) for g in gains]
        except (TypeError, ValueError) as exc:
            raise PresetError(
                f"preset {name!r} has non-numeric colour_gains") from exc

    return preset


def get_preset(cfg, name):
    """Look up and validate a preset by name.

    The error message lists what IS available: the realistic failure is a typo
    at the field, over SSH, with the drone in the air.
    """
    presets = load_presets(cfg)
    if name not in presets:
        available = ", ".join(sorted(presets)) or "none saved yet"
        raise PresetError(
            f"no exposure preset named {name!r} (available: {available}). "
            f"Measure one first: python scripts/set_exposure_preset.py {name}")
    return validate(presets[name], name)


def make_preset(exposure_us, analogue_gain, colour_gains=None, note=None,
                now=None):
    """Build a preset dict, stamped with when it was measured."""
    preset = {
        "exposure_us": int(exposure_us),
        "analogue_gain": round(float(analogue_gain), 3),
        "measured_at": (now or datetime.now()).isoformat(timespec="seconds"),
    }
    if colour_gains is not None:
        preset["colour_gains"] = [round(float(g), 3) for g in colour_gains]
    if note:
        preset["note"] = str(note)
    return validate(preset)


def save_preset(config_path, name, preset):
    """Merge one preset into the config file on disk, preserving everything else.

    Read-modify-write rather than rewriting from a template: the config also
    holds calibration constants that must survive this.
    """
    validate(preset, name)
    cfg = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
    cfg.setdefault(CONFIG_SECTION, {})[name] = preset
    tmp = config_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    os.replace(tmp, config_path)   # atomic: never leave a truncated config
    return cfg


def describe(preset):
    """One-line human summary for logs and the field checklist."""
    bits = [f"{int(preset['exposure_us'])} us",
            f"gain {float(preset['analogue_gain']):.2f}"]
    if preset.get("colour_gains"):
        r, b = preset["colour_gains"]
        bits.append(f"colour_gains ({r:.2f}, {b:.2f})")
    if preset.get("measured_at"):
        bits.append(f"measured {preset['measured_at']}")
    if preset.get("note"):
        bits.append(f"- {preset['note']}")
    return ", ".join(bits)


# --------------------------------------------------------------------------
# clipping: the one way a "correct-looking" preset silently ruins a season
# --------------------------------------------------------------------------

# A saturated pixel has lost its value permanently: NIR pinned at 255 makes NDVI
# read low exactly where the crop is healthiest, so the map inverts in the
# brightest, most vigorous parts of the field. Highlight clipping is far more
# damaging here than slight underexposure, which is why the thresholds are tight.
CLIP_LEVEL = 250
CLIP_WARN = 0.005    # 0.5% of pixels
CLIP_BAD = 0.02      # 2% of pixels


def channel_saturation(frame, clip_level=CLIP_LEVEL):
    """Fraction of pixels at or above clip_level in the NIR and red channels.

    Channel order follows ndvi.extract_channels: picamera2's "RGB888" is laid
    out [B, G, R] in memory, so index 0 is the NIR-bearing channel and index 2
    is red. Returns {"nir": f, "red": f, "mean_nir": v, "mean_red": v}.
    """
    import numpy as np

    nir = frame[:, :, 0]
    red = frame[:, :, 2]
    total = float(nir.size) or 1.0
    return {
        "nir": float((nir >= clip_level).sum()) / total,
        "red": float((red >= clip_level).sum()) / total,
        "mean_nir": float(nir.mean()),
        "mean_red": float(red.mean()),
    }


def clipping_verdict(sat, warn=CLIP_WARN, bad=CLIP_BAD):
    """Grade a saturation report -> (level, message) where level is ok/warn/bad.

    NIR is called out separately because it is the channel that matters most and
    the one most likely to clip: with a red filter on a NoIR sensor, vegetation
    is dramatically brighter in NIR than in red.
    """
    worst = max(sat["nir"], sat["red"])
    detail = (f"NIR {sat['nir'] * 100:.2f}% clipped (mean {sat['mean_nir']:.0f}), "
              f"red {sat['red'] * 100:.2f}% clipped (mean {sat['mean_red']:.0f})")

    if worst >= bad:
        return "bad", (
            f"CLIPPING TOO HIGH - {detail}. Healthy vegetation will read as "
            f"stressed because saturated NIR cannot go higher. Re-measure with "
            f"a shorter exposure (try --exposure at ~60% of the value above).")
    if worst >= warn:
        return "warn", (
            f"some clipping - {detail}. Usable, but re-measuring ~25% shorter "
            f"would be safer, especially if the field has brighter areas than "
            f"what the camera is pointed at now.")
    if sat["mean_nir"] < 40:
        return "warn", (
            f"very dark - {detail}. Nothing is clipped, but an 8-bit pipeline "
            f"this dark wastes most of its levels. Check the lens cap and that "
            f"the camera is aimed at sunlit vegetation.")
    return "ok", f"no meaningful clipping - {detail}"


# --------------------------------------------------------------------------
# capture metadata: the audit trail that proves two flights are comparable
# --------------------------------------------------------------------------

META_NAME = "capture_meta.json"


def capture_meta(preset_name, exposure_us, analogue_gain, colour_gains=None,
                 locked="preset", now=None, extra=None):
    """The record written into every flight folder.

    locked="preset" means the values came from a named preset (comparable across
    flights); "auto" means auto-exposure settled on them (comparable only within
    this flight). compare_flights.py reads this and refuses to report a clean
    error number for two "auto" flights, because that number would be
    meaningless.
    """
    meta = {
        "preset": preset_name,
        "locked": locked,
        "exposure_us": int(exposure_us),
        "analogue_gain": round(float(analogue_gain), 3),
        "captured_at": (now or datetime.now()).isoformat(timespec="seconds"),
    }
    if colour_gains is not None:
        meta["colour_gains"] = [round(float(g), 3) for g in colour_gains]
    if extra:
        meta.update(extra)
    return meta


def write_capture_meta(outdir, meta):
    path = os.path.join(outdir, META_NAME)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    return path


def read_capture_meta(path):
    """Read capture metadata from a file, a flight dir, or an output dir.

    Returns None when absent - older flights predate this file and callers
    degrade to a warning rather than failing.
    """
    if os.path.isdir(path):
        path = os.path.join(path, META_NAME)
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def comparable(meta_a, meta_b):
    """Are two flights radiometrically comparable? -> (ok, reason).

    Called by compare_flights.py before it reports anything. The whole point of
    a twin-flight test is that the ONLY difference between the flights is time;
    if the exposure differs, the test measures the exposure change instead of
    the system's noise, and reports a bad number for the wrong reason.
    """
    if meta_a is None or meta_b is None:
        return False, ("capture_meta.json missing for at least one flight - "
                       "cannot verify the exposure matched (pre-preset flight?)")

    if meta_a.get("locked") != "preset" or meta_b.get("locked") != "preset":
        return False, ("at least one flight used auto-exposure, so the two "
                       "flights settled on different settings - re-fly with "
                       "--preset before trusting any error number")

    if meta_a.get("preset") != meta_b.get("preset"):
        return False, (f"different presets: {meta_a.get('preset')!r} vs "
                       f"{meta_b.get('preset')!r}")

    if int(meta_a["exposure_us"]) != int(meta_b["exposure_us"]):
        return False, (f"exposure differs: {meta_a['exposure_us']} us vs "
                       f"{meta_b['exposure_us']} us")

    if abs(float(meta_a["analogue_gain"]) - float(meta_b["analogue_gain"])) > 1e-6:
        return False, (f"gain differs: {meta_a['analogue_gain']} vs "
                       f"{meta_b['analogue_gain']}")

    return True, f"both flights locked to preset {meta_a.get('preset')!r}"
