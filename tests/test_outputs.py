"""Smoke tests for the deliverable generators - assert artifacts are produced."""

from cropvolare import batch, field, fieldmap, report, webmap


def _pipeline(geotagged_dir, tmp_path):
    fc = batch.process_directory(str(geotagged_dir), flight_date="2026-06-21",
                                 generated="2026-06-21T12:00:00Z")
    grid = field.build_grid(fc, cell_meters=20.0)
    cells = field.classify_cells(grid)
    problems = field.rank_problems(cells, top_n=5)
    summary = field.summarize(cells)
    png = str(tmp_path / "heatmap.png")
    bounds = fieldmap.render_grid_png(grid, png)
    return fc, grid, cells, problems, summary, png, bounds


def test_heatmap_png_written(geotagged_dir, tmp_path):
    *_, png, bounds = _pipeline(geotagged_dir, tmp_path)
    import os
    assert os.path.getsize(png) > 0
    assert bounds is not None and len(bounds) == 2


def test_pdf_is_valid(geotagged_dir, tmp_path):
    fc, grid, cells, problems, summary, png, _ = _pipeline(geotagged_dir, tmp_path)
    out = str(tmp_path / "report.pdf")
    report.build_report(fc, grid, cells, problems, summary, png, out)
    with open(out, "rb") as f:
        head = f.read(5)
    assert head == b"%PDF-"


def test_webmap_is_leaflet(geotagged_dir, tmp_path):
    fc, grid, cells, problems, summary, png, _ = _pipeline(geotagged_dir, tmp_path)
    out = str(tmp_path / "map.html")
    webmap.build_webmap(fc, grid, cells, problems, png, out)
    with open(out, encoding="utf-8") as f:
        html = f.read()
    assert "leaflet" in html.lower()
