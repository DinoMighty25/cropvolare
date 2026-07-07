"""Agronomy-report PDF tests (content asserted via pypdf where available)."""

from cropvolare import report


def _result(scope="field", patches=None, worst=None):
    return {
        "field": "north40", "date": "2026-07-07", "scope": scope,
        "area_ha": 2.5, "n_frames": 80, "n_unreadable": 0, "n_filtered": 5,
        "distribution": {"mean": 0.42, "median": 0.44, "std": 0.12,
                         "pct_healthy": 40.0, "pct_moderate": 35.0,
                         "pct_stressed": 15.0, "pct_severe": 10.0,
                         "histogram": [0] * 10},
        "verdict": {"level": "fair", "score": 55,
                    "line": "Crop vigour is fair, with some areas worth a look."},
        "patches": patches or [],
        "worst_frames": worst or [],
        "calibration": {"flatfield": False, "leakage_k": 2.0, "gamma": 0.8,
                        "min_sharpness": 15},
    }


def test_agronomy_report_valid_pdf(tmp_path):
    out = tmp_path / "r.pdf"
    report.build_agronomy_report(_result(), str(out))
    assert out.read_bytes()[:5] == b"%PDF-"


def test_report_has_verdict_and_patches(tmp_path):
    import pytest
    pypdf = pytest.importorskip("pypdf")
    patches = [{"rank": 1, "lat": 40.001, "lon": -88.002, "area_ha": 0.34,
                "mean_ndvi": 0.18, "status": "stressed",
                "causes": ["broad patch: drainage, irrigation coverage, or a "
                           "soil difference across this area"]}]
    out = tmp_path / "r.pdf"
    report.build_agronomy_report(_result(patches=patches), str(out))
    text = "".join(p.extract_text() for p in pypdf.PdfReader(str(out)).pages)
    assert "Crop Health Report" in text
    assert "Areas needing attention" in text
    assert "north40" in text
    assert "40.001" in text                       # patch coordinate
    assert "decision-support" in text or "decision" in text  # caveat present


def test_report_with_trend(tmp_path):
    import pytest
    pypdf = pytest.importorskip("pypdf")
    trend = {"prev_date": "2026-07-01", "mean_ndvi_delta": -0.08,
             "overall": "declining", "spatial": True,
             "new": [1], "worsened": [1, 2], "improved": [], "resolved": []}
    out = tmp_path / "r.pdf"
    report.build_agronomy_report(_result(), str(out), trend=trend)
    text = "".join(p.extract_text() for p in pypdf.PdfReader(str(out)).pages)
    assert "Change over time" in text
    assert "declining" in text


def test_gallery_scope_lists_worst_frames(tmp_path):
    import pytest
    pypdf = pytest.importorskip("pypdf")
    worst = [{"filename": "frame_0042.jpg", "mean_ndvi": 0.05, "sharpness": 30}]
    out = tmp_path / "r.pdf"
    report.build_agronomy_report(_result(scope="gallery", worst=worst), str(out))
    text = "".join(p.extract_text() for p in pypdf.PdfReader(str(out)).pages)
    assert "frame_0042.jpg" in text
