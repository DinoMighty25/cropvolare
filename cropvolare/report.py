"""
One-page farmer PDF report from the aggregated field data.

Uses fpdf2 (pure-Python, tiny, trivial image embedding). Layout is fixed and
single-page: header, the field heatmap as the hero element, a stats band, and a
ranked table of the worst areas with their coordinates.
"""

import os

from fpdf import FPDF

_STATUS_LABEL = {
    "healthy": "Healthy",
    "stressed": "Stressed",
    "severe": "Severe",
}


def build_report(feature_collection, grid, cells, problems, summary,
                 fieldmap_png, out_path, title="Field NDVI Report"):
    """Write a one-page PDF to out_path. fieldmap_png may be None (no GPS data)."""
    meta = feature_collection.get("metadata", {})
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=False)
    pdf.add_page()

    # --- header ---------------------------------------------------------
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 10)
    centroid = _centroid(meta.get("bbox"))
    flight_date = meta.get("flight_date") or "unknown"
    line = (f"Date: {flight_date}    Images: {meta.get('n_images', 0)}"
            f"    Untagged: {meta.get('n_untagged', 0)}")
    if centroid:
        line += f"    Field center: {centroid[0]:.5f}, {centroid[1]:.5f}"
    pdf.cell(0, 6, line, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    # --- field map hero -------------------------------------------------
    if fieldmap_png:
        # keep within page width (A4 = 210mm, margins ~10mm)
        pdf.image(fieldmap_png, x=15, w=180)
        pdf.ln(3)
    else:
        pdf.set_font("Helvetica", "I", 11)
        pdf.cell(0, 8, "No geotagged photos - field map unavailable.",
                 new_x="LMARGIN", new_y="NEXT")

    # --- stats band -----------------------------------------------------
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Field health summary", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    mean_ndvi = summary.get("mean_ndvi")
    mean_str = f"{mean_ndvi:.3f}" if mean_ndvi is not None else "n/a"
    pdf.cell(0, 6,
             f"Healthy: {summary['pct_healthy']}%    "
             f"Stressed: {summary['pct_stressed']}%    "
             f"Severe: {summary['pct_severe']}%    "
             f"Mean NDVI: {mean_str}",
             new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6,
             f"Grid cells: {summary['n_cells']}    "
             f"Problem areas: {summary['n_problem_cells']}",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # --- ranked problem table ------------------------------------------
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Areas needing attention (worst first)",
             new_x="LMARGIN", new_y="NEXT")

    if problems:
        pdf.set_font("Helvetica", "B", 10)
        widths = (15, 45, 45, 30, 30)
        headers = ("Rank", "Latitude", "Longitude", "Mean NDVI", "Photos")
        for w, h in zip(widths, headers):
            pdf.cell(w, 7, h, border=1)
        pdf.ln()
        pdf.set_font("Helvetica", "", 10)
        for p in problems:
            row = (
                str(p["rank"]),
                f"{p['lat']:.6f}",
                f"{p['lon']:.6f}",
                f"{p['mean_ndvi']:.3f} ({_STATUS_LABEL.get(p['status'], p['status'])})",
                str(p["photo_count"]),
            )
            for w, val in zip(widths, row):
                pdf.cell(w, 7, val, border=1)
            pdf.ln()
    else:
        pdf.set_font("Helvetica", "I", 11)
        pdf.cell(0, 7, "No stressed areas detected.", new_x="LMARGIN", new_y="NEXT")

    # --- footer ---------------------------------------------------------
    params = meta.get("params", {})
    pdf.set_y(-18)
    pdf.set_font("Helvetica", "I", 8)
    pdf.cell(0, 5,
             f"gamma={params.get('gamma')}  leakage_k={params.get('leakage_k')}  "
             f"cell={grid.get('cell_meters')}m  source={meta.get('source')}  "
             f"generated={meta.get('generated')}",
             new_x="LMARGIN", new_y="NEXT")

    pdf.output(out_path)


def _centroid(bbox):
    if not bbox:
        return None
    min_lon, min_lat, max_lon, max_lat = bbox
    return ((min_lat + max_lat) / 2.0, (min_lon + max_lon) / 2.0)


def build_gallery_report(feature_collection, out_path,
                         title="Field NDVI Report (no GPS)"):
    """Contact-sheet PDF of every photo's NDVI overlay, for flights without GPS.

    Each photo gets a thumbnail (its colorized NDVI map) plus a caption with the
    filename, mean NDVI, and health status. Used when no photo carries GPS, so a
    georeferenced field map isn't possible but per-image NDVI still is.
    Requires each feature's properties to include an existing 'overlay_png'.
    """
    meta = feature_collection.get("metadata", {})
    feats = feature_collection.get("features", [])

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=False)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")

    ndvis = [f["properties"]["mean_ndvi"] for f in feats]
    mean = round(sum(ndvis) / len(ndvis), 3) if ndvis else None
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6,
             f"Date: {meta.get('flight_date') or 'unknown'}    "
             f"Images: {len(feats)}    "
             f"Mean NDVI: {mean if mean is not None else 'n/a'}",
             new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6,
             "No GPS data - per-image gallery (a field map needs geotagged photos).",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    # thumbnail grid
    cols = 3
    margin = 10
    cell_w = (210 - 2 * margin) / cols
    thumb_w = cell_w - 4
    cap_h = 9
    row_h = thumb_w + cap_h           # square-worst-case slot keeps rows clear
    page_bottom = 297 - margin
    x0 = margin
    y = pdf.get_y()

    for r in range(0, len(feats), cols):
        if y + row_h > page_bottom:
            pdf.add_page()
            y = margin
        for ci, f in enumerate(feats[r:r + cols]):
            p = f["properties"]
            x = x0 + ci * cell_w
            overlay = p.get("overlay_png")
            if overlay and os.path.exists(overlay):
                try:
                    pdf.image(overlay, x=x + 2, y=y, w=thumb_w)
                except Exception:  # noqa: BLE001 - a bad thumbnail shouldn't kill the report
                    pass
            name = p.get("filename", "")
            if len(name) > 20:
                name = name[:17] + "..."
            pdf.set_xy(x + 2, y + thumb_w + 1)
            pdf.set_font("Helvetica", "", 7)
            pdf.multi_cell(thumb_w, 3,
                           f"{name}\nNDVI {p['mean_ndvi']} ({p['status']})",
                           align="C")
        y += row_h

    pdf.output(out_path)
