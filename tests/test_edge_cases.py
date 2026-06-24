"""Edge-case and robustness coverage: grid corners, the no-GPS pipeline,
dependency guards, config loading, and deliverable content (not just magic bytes).
"""

import importlib.util
import json
import os

import pytest

from cropvolare import batch, field, fieldmap, geo, ndvi, report, webmap

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fc(points):
    feats = [{
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
        "properties": {"mean_ndvi": v},
    } for lon, lat, v in points]
    return {"type": "FeatureCollection", "features": feats, "metadata": {}}


def _load_script(name):
    """Import a script under scripts/ as a module (without running main())."""
    path = os.path.join(REPO_ROOT, "scripts", name)
    spec = importlib.util.spec_from_file_location(name[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- field grid edge cases -------------------------------------------------

def test_single_point_grid():
    grid = field.build_grid(_fc([(-88.0, 40.0, 0.6)]), cell_meters=20.0)
    cells = field.classify_cells(grid)
    assert len(cells) == 1
    assert grid["nrows"] >= 1 and grid["ncols"] >= 1


def test_rectangular_field():
    # points spread along longitude only -> wider than tall
    pts = [(-88.0 - i * 0.0005, 40.0, 0.6) for i in range(4)]
    grid = field.build_grid(_fc(pts), cell_meters=20.0)
    assert grid["ncols"] > grid["nrows"]


def test_large_cell_collapses_to_one():
    # two points ~100 m apart, but a 500 m cell -> they share one cell
    grid = field.build_grid(_fc([(-88.0, 40.0, 0.8), (-88.0, 40.001, 0.2)]),
                            cell_meters=500.0)
    cells = field.classify_cells(grid)
    assert len(cells) == 1
    assert cells[0]["photo_count"] == 2
    assert abs(cells[0]["mean_ndvi"] - 0.5) < 1e-6


def test_southern_hemisphere_sign_preserved():
    grid = field.build_grid(_fc([(-70.6, -33.4, 0.4)]), cell_meters=20.0)
    cells = field.classify_cells(grid)
    assert cells[0]["lat"] < 0 and cells[0]["lon"] < 0


# --- the no-GPS / empty pipeline -------------------------------------------

def test_all_untagged_directory(all_untagged_dir):
    fc = batch.process_directory(str(all_untagged_dir))
    meta = fc["metadata"]
    assert meta["n_untagged"] == meta["n_images"]
    assert meta["bbox"] is None
    grid = field.build_grid(fc)
    assert grid["empty"] is True
    assert field.classify_cells(grid) == []


def test_empty_grid_render_returns_none(all_untagged_dir, tmp_path):
    fc = batch.process_directory(str(all_untagged_dir))
    grid = field.build_grid(fc)
    out = tmp_path / "none.png"
    assert fieldmap.render_grid_png(grid, str(out)) is None
    assert not out.exists()  # nothing written for an empty grid


def test_report_and_webmap_without_gps(all_untagged_dir, tmp_path):
    fc = batch.process_directory(str(all_untagged_dir))
    grid = field.build_grid(fc)
    cells = field.classify_cells(grid)
    problems = field.rank_problems(cells)
    summary = field.summarize(cells)

    pdf = tmp_path / "r.pdf"
    report.build_report(fc, grid, cells, problems, summary, None, str(pdf))
    assert pdf.read_bytes()[:5] == b"%PDF-"

    html = tmp_path / "m.html"
    webmap.build_webmap(fc, grid, cells, problems, None, str(html))
    assert "leaflet" in html.read_text(encoding="utf-8").lower()


def test_gallery_report_for_untagged(all_untagged_dir, tmp_path):
    overlay_dir = tmp_path / "ov"
    fc = batch.process_directory(str(all_untagged_dir), overlay_dir=str(overlay_dir))
    out = tmp_path / "gallery.pdf"
    report.build_gallery_report(fc, str(out))
    assert out.read_bytes()[:5] == b"%PDF-"
    # every untagged photo still got a per-image NDVI overlay for the gallery
    assert all(f["properties"]["overlay_png"] for f in fc["features"])


# --- dependency import guards ----------------------------------------------

def test_save_image_requires_cv2(monkeypatch, tmp_path):
    import numpy as np
    monkeypatch.setattr(ndvi, "cv2", None)
    with pytest.raises(RuntimeError, match="opencv"):
        ndvi.save_ndvi_image(np.zeros((4, 4)), str(tmp_path / "x.png"))


def test_geo_requires_piexif(monkeypatch, tmp_path):
    monkeypatch.setattr(geo, "piexif", None)
    with pytest.raises(RuntimeError, match="piexif"):
        geo.read_gps(str(tmp_path / "nope.jpg"))


# --- config loading --------------------------------------------------------

def test_load_config_missing_returns_empty(tmp_path):
    mod = _load_script("process_flight.py")
    assert mod.load_config(str(tmp_path / "does_not_exist.json")) == {}


def test_load_config_reads_json(tmp_path):
    mod = _load_script("capture_ndvi.py")
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"ndvi": {"gamma": 0.42}}))
    cfg = mod.load_config(str(p))
    assert cfg["ndvi"]["gamma"] == 0.42


# --- deliverable content (beyond magic bytes) ------------------------------

def test_pdf_contains_summary_text(flight_outputs, tmp_path):
    pypdf = pytest.importorskip("pypdf")  # optional dev tool; skip if absent
    pdf_path = tmp_path / "content.pdf"
    report.build_report(
        flight_outputs["fc"], flight_outputs["grid"], flight_outputs["cells"],
        flight_outputs["problems"], flight_outputs["summary"],
        flight_outputs["heatmap_png"], str(pdf_path),
    )
    reader = pypdf.PdfReader(str(pdf_path))
    text = "".join(page.extract_text() for page in reader.pages)
    assert "Healthy" in text
    assert "NDVI" in text


def test_webmap_has_overlay_and_problem_marker(flight_outputs, tmp_path):
    html_path = tmp_path / "content.html"
    webmap.build_webmap(
        flight_outputs["fc"], flight_outputs["grid"], flight_outputs["cells"],
        flight_outputs["problems"], flight_outputs["heatmap_png"], str(html_path),
    )
    html = html_path.read_text(encoding="utf-8")
    assert "imageoverlay" in html.lower()       # NDVI overlay present
    assert "Problem #" in html                   # ranked problem popup present
