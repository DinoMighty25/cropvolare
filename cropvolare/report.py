"""
One-page farmer PDF report from the aggregated field data.

Uses fpdf2 (pure-Python, tiny, trivial image embedding). Layout is fixed and
single-page: header, the field heatmap as the hero element, a stats band, and a
ranked table of the worst areas with their coordinates.
"""

import os
import tempfile

from fpdf import FPDF

try:
    import cv2
except ImportError:
    cv2 = None


def _thumbnail(src, out_dir, max_px=360):
    """Downscale an overlay to a small JPEG for the gallery, return its path.

    Without this the PDF embeds every full-resolution overlay (hundreds of MB
    for a real flight). Falls back to the original if cv2 is unavailable.
    """
    if cv2 is None:
        return src
    img = cv2.imread(src)
    if img is None:
        return src
    h, w = img.shape[:2]
    scale = min(1.0, max_px / float(max(h, w)))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    path = os.path.join(out_dir, os.path.basename(src) + ".thumb.jpg")
    cv2.imwrite(path, img)
    return path

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
                         title="Field NDVI Report (no GPS)", max_gallery=60):
    """Summary + contact-sheet PDF of NDVI overlays, for flights without GPS.

    Page 1 leads with what matters: flight stats and a "frames needing
    attention" table of the lowest-NDVI frames (the go-look-here list even
    without coordinates). The thumbnail gallery follows, capped at max_gallery
    frames (evenly sampled across the flight when over the cap) so a
    several-hundred-frame flight stays a few readable pages.
    Requires each feature's properties to include an existing 'overlay_png'.
    """
    meta = feature_collection.get("metadata", {})
    feats = feature_collection.get("features", [])

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=False)
    pdf.add_page()

    # --- summary page ----------------------------------------------------
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")

    ndvis = [f["properties"]["mean_ndvi"] for f in feats]
    mean = round(sum(ndvis) / len(ndvis), 3) if ndvis else None
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6,
             f"Date: {meta.get('flight_date') or 'unknown'}    "
             f"Images: {len(feats)}    "
             f"Unreadable: {meta.get('n_unreadable', 0)}    "
             f"Filtered (blur): {meta.get('n_filtered', 0)}",
             new_x="LMARGIN", new_y="NEXT")
    if ndvis:
        pdf.cell(0, 6,
                 f"Mean NDVI: {mean}    "
                 f"Range: {min(ndvis):.3f} to {max(ndvis):.3f}",
                 new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6,
             "No GPS data - per-image gallery (a field map needs geotagged photos).",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # frames needing attention: lowest mean NDVI first
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Frames needing attention (lowest NDVI first)",
             new_x="LMARGIN", new_y="NEXT")
    worst = sorted(feats, key=lambda f: f["properties"]["mean_ndvi"])[:10]
    if worst:
        pdf.set_font("Helvetica", "B", 10)
        widths = (15, 70, 35, 35)
        for wd, hd in zip(widths, ("Rank", "Frame", "Mean NDVI", "Sharpness")):
            pdf.cell(wd, 7, hd, border=1)
        pdf.ln()
        pdf.set_font("Helvetica", "", 10)
        for rank, f in enumerate(worst, start=1):
            p = f["properties"]
            sharp = p.get("sharpness")
            row = (str(rank), p.get("filename", ""),
                   f"{p['mean_ndvi']:.3f}",
                   f"{sharp:.0f}" if sharp is not None else "n/a")
            for wd, val in zip(widths, row):
                pdf.cell(wd, 7, val, border=1)
            pdf.ln()
        pdf.set_font("Helvetica", "I", 8)
        pdf.cell(0, 5, "Low sharpness can mean the frame was captured while "
                       "grounded/too low - judge those with care.",
                 new_x="LMARGIN", new_y="NEXT")
    else:
        pdf.set_font("Helvetica", "I", 11)
        pdf.cell(0, 7, "No frames to report.", new_x="LMARGIN", new_y="NEXT")

    # --- gallery (capped, evenly sampled) --------------------------------
    gallery = feats
    if max_gallery and len(feats) > max_gallery:
        stride = -(-len(feats) // max_gallery)  # ceil division
        gallery = feats[::stride][:max_gallery]

    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    note = (f"Gallery - showing {len(gallery)} of {len(feats)} frames "
            f"(evenly sampled)" if len(gallery) < len(feats)
            else f"Gallery - all {len(gallery)} frames")
    pdf.cell(0, 8, note, new_x="LMARGIN", new_y="NEXT")

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

    thumb_dir = tempfile.mkdtemp(prefix="cropvolare_thumbs_")
    try:
        for r in range(0, len(gallery), cols):
            if y + row_h > page_bottom:
                pdf.add_page()
                y = margin
            for ci, f in enumerate(gallery[r:r + cols]):
                p = f["properties"]
                x = x0 + ci * cell_w
                overlay = p.get("overlay_png")
                if overlay and os.path.exists(overlay):
                    try:
                        pdf.image(_thumbnail(overlay, thumb_dir),
                                  x=x + 2, y=y, w=thumb_w)
                    except Exception:  # noqa: BLE001 - a bad thumbnail shouldn't kill the report
                        pass
                name = p.get("filename", "")
                if len(name) > 20:
                    name = name[:17] + "..."
                sharp = p.get("sharpness")
                sharp_txt = f"  sharp {sharp:.0f}" if sharp is not None else ""
                pdf.set_xy(x + 2, y + thumb_w + 1)
                pdf.set_font("Helvetica", "", 7)
                pdf.multi_cell(thumb_w, 3,
                               f"{name}\nNDVI {p['mean_ndvi']} "
                               f"({p['status']}){sharp_txt}",
                               align="C")
            y += row_h

        pdf.output(out_path)
    finally:
        import shutil
        shutil.rmtree(thumb_dir, ignore_errors=True)


# --------------------------------------------------------------------------
# agronomy report - farmer-first, driven by the analysis engine
# --------------------------------------------------------------------------

_LEVEL_LABEL = {"strong": "STRONG", "fair": "FAIR", "poor": "POOR",
                "critical": "CRITICAL", "unknown": "UNKNOWN"}


def _trend_sentence(trend):
    d = trend.get("mean_ndvi_delta")
    if d is None:
        return "Change vs last flight: not comparable."
    arrow = {"improving": "up", "declining": "down", "stable": "unchanged"}.get(
        trend["overall"], "")
    return (f"Vs last flight ({trend.get('prev_date')}): field NDVI {arrow} "
            f"{d:+.3f} ({trend['overall']}).")


def build_agronomy_report(result, out_path, fieldmap_png=None, trend=None):
    """Farmer-facing PDF from an AnalysisResult (+ optional trend/heatmap).

    Leads with a plain verdict, then the problem-area table (the actionable
    part), the trend vs last flight, the map/frames, and honest methodology.
    """
    dist = result.get("distribution", {})
    verdict = result.get("verdict", {})
    cal = result.get("calibration", {})

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()

    # header + verdict
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "Crop Health Report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    area = (f"{result['area_ha']:.2f} ha" if result.get("area_ha")
            else "area n/a (no GPS)")
    pdf.cell(0, 6, f"Field: {result.get('field') or 'unnamed'}    "
                   f"Date: {result.get('date') or 'unknown'}    "
                   f"Scanned: {area}    Images: {result.get('n_frames', 0)}",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 9, f"Overall: {_LEVEL_LABEL.get(verdict.get('level'), '?')} "
                   f"(score {verdict.get('score', 0)}/100)",
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6, verdict.get("line", ""))
    pdf.ln(2)

    # health summary
    if dist.get("mean") is not None:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Field health", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 6, f"Mean NDVI {dist['mean']}    "
                       f"Healthy {dist['pct_healthy']}%   "
                       f"Moderate {dist['pct_moderate']}%   "
                       f"Stressed {dist['pct_stressed']}%   "
                       f"Severe {dist['pct_severe']}%",
                 new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

    # trend
    if trend:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Change over time", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, _trend_sentence(trend))
        if trend.get("spatial"):
            pdf.cell(0, 6, f"Problem areas - new: {len(trend['new'])}, "
                           f"worsened: {len(trend['worsened'])}, "
                           f"improved: {len(trend['improved'])}, "
                           f"resolved: {len(trend['resolved'])}",
                     new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

    # problem areas (the actionable table)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Areas needing attention", new_x="LMARGIN", new_y="NEXT")
    patches = result.get("patches", [])
    if patches:
        pdf.set_font("Helvetica", "B", 9)
        widths = (10, 44, 22, 20, 84)
        for wd, hd in zip(widths, ("#", "Location", "Size (ha)", "NDVI",
                                   "First thing to check")):
            pdf.cell(wd, 7, hd, border=1)
        pdf.ln()
        pdf.set_font("Helvetica", "", 8)
        for p in patches[:8]:
            cause = (p.get("causes") or ["inspect on the ground"])[0]
            if len(cause) > 62:
                cause = cause[:59] + "..."
            loc = f"{p['lat']:.5f},{p['lon']:.5f}"
            cells = [(widths[0], str(p["rank"])), (widths[1], loc),
                     (widths[2], f"{p['area_ha']:.3f}"),
                     (widths[3], f"{p['mean_ndvi']:.2f}"), (widths[4], cause)]
            row_h = 5 * max(1, (len(cause) // 42) + 1)
            y0 = pdf.get_y()
            x0 = pdf.get_x()
            for wd, val in cells:
                x = pdf.get_x()
                pdf.multi_cell(wd, 5, val, border=1,
                               new_x="RIGHT", new_y="TOP", max_line_height=5)
                pdf.set_xy(x + wd, y0)
            pdf.set_xy(x0, y0 + row_h)
    elif result.get("scope") == "gallery":
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, "No GPS on these photos, so problem areas can't be "
                             "mapped. Lowest-NDVI frames (inspect what they show):")
        worst = result.get("worst_frames", [])[:6]
        thumb_dir = tempfile.mkdtemp(prefix="cropvolare_agro_")
        try:
            cols, cw = 3, (210 - 20) / 3
            tw = cw - 4
            x0, y = 10, pdf.get_y() + 2
            for r in range(0, len(worst), cols):
                if y + tw + 10 > 285:
                    pdf.add_page(); y = 15
                for ci, wf in enumerate(worst[r:r + cols]):
                    x = x0 + ci * cw
                    # show the actual PHOTO - "inspect what they show" needs
                    # the scene, not an NDVI heat-blob (a worst frame's NDVI
                    # render is a featureless solid-red square)
                    img_path = wf.get("source_path")
                    if not (img_path and os.path.exists(img_path)):
                        img_path = wf.get("overlay_png")
                    if img_path and os.path.exists(img_path):
                        try:
                            pdf.image(_thumbnail(img_path, thumb_dir),
                                      x=x + 2, y=y, w=tw)
                        except Exception:  # noqa: BLE001
                            pass
                    pdf.set_xy(x + 2, y + tw + 1)
                    pdf.set_font("Helvetica", "", 7)
                    pdf.multi_cell(tw, 3,
                                   f"{wf['filename']}\nNDVI {wf['mean_ndvi']}",
                                   align="C")
                y += tw + 10
            pdf.set_y(y)
        finally:
            import shutil
            shutil.rmtree(thumb_dir, ignore_errors=True)
    else:
        pdf.set_font("Helvetica", "I", 11)
        pdf.cell(0, 7, "No distinct problem areas detected.",
                 new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    # map
    if fieldmap_png and os.path.exists(fieldmap_png):
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Field map (red = low NDVI)", new_x="LMARGIN", new_y="NEXT")
        try:
            pdf.image(fieldmap_png, w=180)
        except Exception:  # noqa: BLE001
            pass

    # methodology / caveats
    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 8)
    ff = "on" if cal.get("flatfield") else "OFF"
    pdf.multi_cell(0, 4,
        "Method: single-camera NoIR NDVI (relative, approximate). "
        f"Calibration - flat-field: {ff}, leakage_k: {cal.get('leakage_k')}, "
        f"blur filter: {cal.get('min_sharpness')}. "
        "Findings are decision-support to guide ground inspection, not a "
        "diagnosis. Verify problem areas in person before acting.")

    pdf.output(out_path)
