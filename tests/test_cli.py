"""End-to-end CLI tests: run the scripts as real subprocesses.

Marked e2e (slower). These cover the script entry points that unit tests miss:
argument handling, config loading, flag-over-config precedence, and the full
process_flight chain producing all artifacts.
"""

import json
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

pytestmark = pytest.mark.e2e


def run_script(script, *args):
    """Run scripts/<script> with the current interpreter; return CompletedProcess."""
    path = os.path.join(REPO_ROOT, "scripts", script)
    return subprocess.run(
        [sys.executable, path, *args],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )


# --- agronomy report + history (analysis engine) ---------------------------

def test_process_flight_agronomy_and_trend(geotagged_dir, tmp_path):
    import shutil
    pypdf = pytest.importorskip("pypdf")
    env = dict(os.environ, CROPVOLARE_HISTORY_DIR=str(tmp_path / "history"))
    # two distinct flight folders of the same field (different flight_id)
    flight2 = tmp_path / "flight2"
    shutil.copytree(str(geotagged_dir), str(flight2))

    def run(indir, outdir):
        return subprocess.run(
            [sys.executable, os.path.join(REPO_ROOT, "scripts", "process_flight.py"),
             "--input", str(indir), "--outdir", str(outdir),
             "--field", "yard", "--no-flatfield"],
            capture_output=True, text=True, cwd=REPO_ROOT, env=env)

    r1 = run(geotagged_dir, tmp_path / "o1")
    assert r1.returncode == 0, r1.stderr
    assert "analysis:" in r1.stdout
    text = "".join(p.extract_text()
                   for p in pypdf.PdfReader(str(tmp_path / "o1" / "report.pdf")).pages)
    assert "Crop Health Report" in text
    assert "Areas needing attention" in text

    r2 = run(flight2, tmp_path / "o2")           # second flight -> trend section
    assert r2.returncode == 0, r2.stderr
    text2 = "".join(p.extract_text()
                    for p in pypdf.PdfReader(str(tmp_path / "o2" / "report.pdf")).pages)
    assert "Change over time" in text2


# --- process_flight --------------------------------------------------------

def test_process_flight_produces_all_artifacts(geotagged_dir, tmp_path):
    outdir = tmp_path / "out"
    result = run_script("process_flight.py",
                        "--input", str(geotagged_dir),
                        "--outdir", str(outdir),
                        "--cell-meters", "20")
    assert result.returncode == 0, result.stderr

    for name in ("field.geojson", "heatmap.png", "report.pdf", "map.html"):
        assert (outdir / name).exists(), f"missing {name}"

    fc = json.loads((outdir / "field.geojson").read_text())
    assert fc["type"] == "FeatureCollection"
    assert fc["metadata"]["n_images"] == 5
    assert fc["metadata"]["n_untagged"] == 1
    # the stressed corner must surface somewhere in the data
    statuses = [f["properties"]["status"] for f in fc["features"]]
    assert any(s in ("stressed", "moderate") for s in statuses)

    assert (outdir / "report.pdf").read_bytes()[:5] == b"%PDF-"
    assert "leaflet" in (outdir / "map.html").read_text(encoding="utf-8").lower()


def test_process_flight_no_gps_builds_gallery(all_untagged_dir, tmp_path):
    # with no GPS the run still succeeds and produces a per-image gallery PDF
    outdir = tmp_path / "out2"
    result = run_script("process_flight.py",
                        "--input", str(all_untagged_dir),
                        "--outdir", str(outdir))
    assert result.returncode == 0, result.stderr
    # gallery report is produced; the field map / web map are skipped (need GPS)
    assert (outdir / "report.pdf").exists()
    assert (outdir / "report.pdf").read_bytes()[:5] == b"%PDF-"
    assert not (outdir / "map.html").exists()
    assert not (outdir / "heatmap.png").exists()
    fc = json.loads((outdir / "field.geojson").read_text())
    assert fc["metadata"]["n_untagged"] == fc["metadata"]["n_images"]
    assert fc["metadata"]["bbox"] is None


# --- capture_ndvi ----------------------------------------------------------

def test_capture_ndvi_input_tiff_vari(ndvi_jpeg_factory, tmp_path):
    img = ndvi_jpeg_factory(40.0, -88.0, nir=210, red=60)
    png = tmp_path / "out.png"
    tiff = tmp_path / "out.tiff"
    result = run_script("capture_ndvi.py",
                        "--input", img,
                        "-o", str(png),
                        "--tiff", str(tiff),
                        "--vari")
    assert result.returncode == 0, result.stderr
    assert png.exists() and tiff.exists()
    assert "mean NDVI" in result.stdout
    assert "mean VARI" in result.stdout  # proves the --vari path ran


def test_capture_ndvi_print_zones_json(ndvi_jpeg_factory, tmp_path):
    img = ndvi_jpeg_factory(40.0, -88.0)
    result = run_script("capture_ndvi.py", "--input", img,
                        "--no-save", "--print-zones")
    assert result.returncode == 0, result.stderr
    # the --print-zones block is valid JSON with the expected keys
    start = result.stdout.index("{")
    payload = json.loads(result.stdout[start:])
    assert "mean_ndvi" in payload and "zones" in payload


# --- config loading + flag-over-config precedence --------------------------

def _write_config(tmp_path, gamma):
    cfg = {"ndvi": {"gamma": gamma, "leakage_k": 0.6, "block_size": 64}}
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))
    return str(p)


def test_config_gamma_is_honored(ndvi_jpeg_factory, tmp_path):
    img = ndvi_jpeg_factory(40.0, -88.0)
    cfg = _write_config(tmp_path, gamma=0.5)
    result = run_script("capture_ndvi.py", "--input", img,
                        "--config", cfg, "--no-save")
    assert result.returncode == 0, result.stderr
    assert "gamma=0.5" in result.stdout


def test_cli_flag_overrides_config(ndvi_jpeg_factory, tmp_path):
    img = ndvi_jpeg_factory(40.0, -88.0)
    cfg = _write_config(tmp_path, gamma=0.5)
    result = run_script("capture_ndvi.py", "--input", img,
                        "--config", cfg, "--gamma", "0.9", "--no-save")
    assert result.returncode == 0, result.stderr
    assert "gamma=0.9" in result.stdout  # flag wins over config


# --- flat-field: calibrate build + process_flight apply ---------------------

def _write_flat_frames(d, n=3):
    """Synthetic vignetted white-target frames (NIR falls off harder)."""
    import cv2
    import numpy as np
    os.makedirs(d, exist_ok=True)
    yy, xx = np.mgrid[0:120, 0:160]
    r2 = ((yy / 120 - 0.5) ** 2 + (xx / 160 - 0.5) ** 2)
    r2 = r2 / r2.max()
    img = np.zeros((120, 160, 3), np.float64)
    img[:, :, 0] = 200 * (1.0 - 0.5 * r2)
    img[:, :, 1] = 200 * (1.0 - 0.3 * r2)
    img[:, :, 2] = 200 * (1.0 - 0.2 * r2)
    img = np.clip(img, 0, 255).astype(np.uint8)
    for i in range(n):
        cv2.imwrite(os.path.join(d, f"flat_{i}.jpg"), img)


def test_calibrate_flatfield_builds_and_writes(tmp_path):
    flat_dir = str(tmp_path / "flats")
    _write_flat_frames(flat_dir)
    gain_out = str(tmp_path / "gain.npy")
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({"calibration": {}}))

    result = run_script("calibrate.py", "--flatfield-dir", flat_dir,
                        "--gain-out", gain_out, "--config", str(cfg_path),
                        "--write")
    assert result.returncode == 0, result.stderr
    assert "corner/center gain ratio" in result.stdout
    assert os.path.exists(gain_out)
    cfg = json.loads(cfg_path.read_text())
    # written with forward slashes - the config is shared with the Pi
    assert cfg["calibration"]["flatfield_path"] == gain_out.replace("\\", "/")


def test_process_flight_applies_flatfield(geotagged_dir, tmp_path):
    flat_dir = str(tmp_path / "flats")
    _write_flat_frames(flat_dir)
    gain_out = str(tmp_path / "gain.npy")
    run_script("calibrate.py", "--flatfield-dir", flat_dir,
               "--gain-out", gain_out)

    outdir = tmp_path / "out_ff"
    result = run_script("process_flight.py",
                        "--input", str(geotagged_dir),
                        "--outdir", str(outdir),
                        "--flatfield", gain_out)
    assert result.returncode == 0, result.stderr
    assert "flat-field: active" in result.stdout
    fc = json.loads((outdir / "field.geojson").read_text())
    assert fc["metadata"]["params"]["flatfield"] is True


def test_process_flight_min_sharpness_filters(geotagged_dir, tmp_path):
    # the fixture's uniform frames are all near-zero sharpness
    outdir = tmp_path / "out_sh"
    result = run_script("process_flight.py",
                        "--input", str(geotagged_dir),
                        "--outdir", str(outdir),
                        "--no-flatfield",
                        "--min-sharpness", "50")
    assert result.returncode == 0, result.stderr
    fc = json.loads((outdir / "field.geojson").read_text())
    assert fc["metadata"]["n_filtered"] == 5
    assert fc["metadata"]["n_images"] == 0
    assert (outdir / "report.pdf").exists()  # empty flight still reports


# --- calibrate (grey-card leakage solve) ------------------------------------

def test_calibrate_solves_and_writes_config(tmp_path):
    import cv2
    import numpy as np
    grey = np.zeros((64, 64, 3), dtype=np.uint8)
    grey[:, :, 0] = 100   # NIR
    grey[:, :, 2] = 160   # red with 0.6 bleed
    card = str(tmp_path / "grey.png")   # png: lossless, exact k
    cv2.imwrite(card, grey)
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({"ndvi": {"leakage_k": 0.6}}))

    result = run_script("calibrate.py", "--input", card, "--gamma", "1.0",
                        "--config", str(cfg_path), "--write")
    assert result.returncode == 0, result.stderr
    assert "solved leakage_k = 0.6" in result.stdout
    cfg = json.loads(cfg_path.read_text())
    assert cfg["ndvi"]["leakage_k"] == 0.6


# --- ground_station (laptop helper) ----------------------------------------

def test_ground_station_local_analyze(geotagged_dir, tmp_path):
    # no --host: analyze a local folder, producing the report
    outdir = tmp_path / "gs"
    result = run_script("ground_station.py",
                        "--input", str(geotagged_dir),
                        "--outdir", str(outdir))
    assert result.returncode == 0, result.stderr
    assert (outdir / "report.pdf").exists()
    assert (outdir / "field.geojson").exists()
