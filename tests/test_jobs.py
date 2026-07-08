"""Job runner tests: on-device flight processing, no camera required."""

import json
import shutil
import threading

import pytest

from cropvolare import jobs


@pytest.fixture
def flights_base(tmp_path, geotagged_dir):
    """A flights/ dir holding one small real flight folder."""
    base = tmp_path / "flights"
    base.mkdir()
    shutil.copytree(str(geotagged_dir), str(base / "flightA"))
    return base


def test_job_processes_flight_to_done(flights_base, tmp_path, monkeypatch):
    # the real pipeline end to end: progress -> done -> artifacts on disk
    monkeypatch.setenv("CROPVOLARE_HISTORY_DIR", str(tmp_path / "hist"))
    runner = jobs.JobRunner(base=str(flights_base),
                            capture_active_fn=lambda: False)
    ok, err = runner.start("flightA")
    assert ok, err
    status = runner.wait(timeout=120)
    assert status["state"] == "done"
    assert status["pct"] == 100
    assert status["verdict"]["level"] in {"strong", "fair", "poor",
                                          "critical", "unknown"}
    adir = flights_base / "flightA" / "analysis"
    assert (adir / "report.pdf").exists()
    result = json.loads((adir / "result.json").read_text())
    assert result["verdict"]["level"] == status["verdict"]["level"]
    # the status mirror survives a GCS restart
    assert json.loads((adir / "status.json").read_text())["state"] == "done"


def test_one_job_at_a_time_and_on_done(flights_base):
    release = threading.Event()
    finished = []

    def slow_run(indir, outdir, **kw):
        release.wait(10)
        return {"result": None, "verdict": {"level": "fair"}}

    runner = jobs.JobRunner(base=str(flights_base), run_fn=slow_run,
                            capture_active_fn=lambda: False,
                            on_done=finished.append)
    ok, _ = runner.start("flightA")
    assert ok
    ok2, err2 = runner.start("flightA")
    assert not ok2
    assert "already running" in err2
    release.set()
    status = runner.wait(timeout=20)
    assert status["state"] == "done"
    assert finished and finished[0]["state"] == "done"


def test_refuses_while_capture_running(flights_base):
    runner = jobs.JobRunner(base=str(flights_base),
                            capture_active_fn=lambda: True)
    ok, err = runner.start("flightA")
    assert not ok
    assert "capture" in err


def test_refuses_unknown_or_empty_flight(flights_base):
    runner = jobs.JobRunner(base=str(flights_base),
                            capture_active_fn=lambda: False)
    ok, err = runner.start("nope")
    assert not ok and "unknown flight" in err
    (flights_base / "empty").mkdir()
    ok, err = runner.start("empty")
    assert not ok and "no frames" in err


def test_scale_and_field_reach_the_pipeline(flights_base, tmp_path):
    cfgp = tmp_path / "cfg.json"
    cfgp.write_text(json.dumps({"processing": {"scale": 0.25}}))
    seen = {}

    def spy_run(indir, outdir, **kw):
        seen.update(kw, indir=indir, outdir=outdir)
        return {"result": None, "verdict": None}

    runner = jobs.JobRunner(base=str(flights_base), config_path=str(cfgp),
                            run_fn=spy_run, capture_active_fn=lambda: False)
    ok, _ = runner.start("flightA", field="yard")
    assert ok
    status = runner.wait(timeout=20)
    assert status["state"] == "done"
    assert seen["process_scale"] == 0.25
    assert seen["field_name"] == "yard"
    assert seen["outdir"].endswith("analysis")


def test_error_reaches_status(flights_base):
    def bad_run(indir, outdir, **kw):
        raise RuntimeError("boom")

    runner = jobs.JobRunner(base=str(flights_base), run_fn=bad_run,
                            capture_active_fn=lambda: False)
    ok, _ = runner.start("flightA")
    assert ok
    status = runner.wait(timeout=20)
    assert status["state"] == "error"
    assert "boom" in status["error"]


def test_list_reports(flights_base):
    adir = flights_base / "flightA" / "analysis"
    adir.mkdir()
    (adir / "result.json").write_text(json.dumps({
        "date": "2026-07-08", "field": "yard", "n_frames": 5,
        "verdict": {"level": "fair", "score": 61, "line": "ok-ish"},
        "distribution": {"mean": 0.41},
    }))
    (adir / "report.pdf").write_bytes(b"%PDF-1.4 fake")
    # an unprocessed flight next to it must not appear
    (flights_base / "raw").mkdir()

    rows = jobs.list_reports(str(flights_base))
    assert len(rows) == 1
    row = rows[0]
    assert row["flight"] == "flightA"
    assert row["field"] == "yard"
    assert row["level"] == "fair"
    assert row["score"] == 61
    assert row["mean_ndvi"] == 0.41
    assert row["has_pdf"] is True
