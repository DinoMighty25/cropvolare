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


def test_process_flight_cell_meters_from_config(all_untagged_dir, tmp_path):
    # even with no GPS the run should succeed and report an empty field
    outdir = tmp_path / "out2"
    result = run_script("process_flight.py",
                        "--input", str(all_untagged_dir),
                        "--outdir", str(outdir))
    assert result.returncode == 0, result.stderr
    # report + map still generated for an all-untagged flight
    assert (outdir / "report.pdf").exists()
    assert (outdir / "map.html").exists()
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
