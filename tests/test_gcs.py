"""GCS web app tests via the Flask test client - no browser, no camera."""

import importlib.util
import json
import os

import cv2
import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_gcs():
    path = os.path.join(REPO_ROOT, "scripts", "gcs.py")
    spec = importlib.util.spec_from_file_location("gcs", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gcs = _load_gcs()

SQUARE = [[40.0, -88.0], [40.0, -87.999], [40.0009, -87.999], [40.0009, -88.0]]


@pytest.fixture
def client(tmp_path):
    app = gcs.create_app(base=str(tmp_path / "flights"),
                         fields_dir=str(tmp_path / "fields"),
                         start_fn=lambda base, **kw: 0,
                         stop_fn=lambda base, **kw: 0,
                         config={"ndvi": {"gamma": 0.8, "leakage_k": 2.0}})
    app.testing = True
    return app.test_client()


def _add_flight(tmp_path, name="f1", frames=2):
    d = tmp_path / "flights" / name
    os.makedirs(d, exist_ok=True)
    img = np.zeros((48, 64, 3), np.uint8)
    img[:, :, 0] = 150
    img[:, :, 2] = 90
    for i in range(frames):
        cv2.imwrite(str(d / f"frame_{i:04d}.jpg"), img)
    return d


def test_dashboard_and_planner_pages(client):
    assert client.get("/").status_code == 200
    assert b"CropVolare" in client.get("/").data
    assert client.get("/planner").status_code == 200


def test_status_empty(client):
    s = client.get("/api/status").get_json()
    assert s["capturing"] is False
    assert s["frames"] == 0
    assert s["gps"] is None


def test_start_stop_routes(client):
    r = client.post("/api/start", json={})
    assert r.status_code == 200 and r.get_json()["ok"] is True
    r = client.post("/api/stop")
    assert r.status_code == 200 and r.get_json()["ok"] is True


def test_preview_404_without_frames(client):
    assert client.get("/api/preview.jpg").status_code == 404
    assert client.get("/api/preview_ndvi.jpg").status_code == 404


def test_previews_serve_jpeg(client, tmp_path):
    _add_flight(tmp_path)
    r = client.get("/api/preview.jpg")
    assert r.status_code == 200
    assert r.data[:2] == b"\xff\xd8"          # JPEG magic
    r = client.get("/api/preview_ndvi.jpg")
    assert r.status_code == 200
    assert r.data[:2] == b"\xff\xd8"


def test_flights_listing(client, tmp_path):
    _add_flight(tmp_path, "f1", frames=3)
    flights = client.get("/api/flights").get_json()
    assert flights[0]["name"] == "f1"
    assert flights[0]["n_frames"] == 3


def test_fields_roundtrip(client):
    r = client.post("/api/fields", json={"name": "yard", "polygon": SQUARE})
    assert r.status_code == 200
    fields = client.get("/api/fields").get_json()
    assert fields[0]["name"] == "yard"
    assert fields[0]["polygon"] == SQUARE


def test_field_name_validation(client):
    r = client.post("/api/fields",
                    json={"name": "../evil", "polygon": SQUARE})
    assert r.status_code == 400
    r = client.post("/api/fields", json={"name": "ok", "polygon": SQUARE[:2]})
    assert r.status_code == 400


def test_plan_endpoint(client):
    r = client.post("/api/plan", json={"polygon": SQUARE, "altitude": 30,
                                       "overlap": 0.75})
    j = r.get_json()
    assert r.status_code == 200 and j["ok"] is True
    assert len(j["lines"]) >= 5
    assert j["stats"]["distance_m"] > 0


def test_plan_rejects_bad_polygon(client):
    r = client.post("/api/plan", json={"polygon": SQUARE[:2]})
    assert r.status_code == 400


def test_export_kml_and_litchi(client):
    client.post("/api/fields", json={"name": "yard", "polygon": SQUARE})
    r = client.get("/api/export?field=yard&fmt=kml&altitude=30")
    assert r.status_code == 200
    assert b"<kml" in r.data
    r = client.get("/api/export?field=yard&fmt=litchi&altitude=30")
    assert r.status_code == 200
    assert r.data.startswith(b"latitude,longitude")
    assert client.get("/api/export?field=nope&fmt=kml").status_code == 404


def _snap_app(tmp_path, session, service_active=False):
    app = gcs.create_app(base=str(tmp_path / "flights"),
                         fields_dir=str(tmp_path / "fields"),
                         session=session,
                         flight_service_active_fn=lambda: service_active,
                         config={"ndvi": {"gamma": 0.8, "leakage_k": 2.0}},
                         snapshot_wait=0.2)
    app.testing = True
    return app.test_client()


def test_snapshot_serves_from_session_buffer(tmp_path):
    # snapshot reads the SAME session as the stream - never its own camera
    c = _snap_app(tmp_path, _FakeSession(n_frames=5))
    r = c.get("/api/snapshot.jpg")
    assert r.status_code == 200
    assert r.data[:2] == b"\xff\xd8"
    r = c.get("/api/snapshot.jpg?ndvi=1")   # NDVI-rendered viewfinder
    assert r.status_code == 200
    assert r.data[:2] == b"\xff\xd8"


def test_snapshot_refused_while_flight_service_runs(tmp_path):
    # boot spin-up window: service active but no frames yet - camera is
    # spoken for, the viewfinder must not touch it
    c = _snap_app(tmp_path, _FakeSession(), service_active=True)
    assert c.get("/api/snapshot.jpg").status_code == 409


def test_snapshot_503_when_camera_never_warms(tmp_path):
    session = _FakeSession(n_frames=0)
    session.info = lambda: {"last_error": "picamera2 not installed"}
    c = _snap_app(tmp_path, session)
    r = c.get("/api/snapshot.jpg")
    assert r.status_code == 503
    assert b"camera unavailable" in r.data


def test_status_reports_viewfinder(tmp_path):
    session = _FakeSession()
    session.info = lambda: {"running": True, "warmed": True,
                            "frames_captured": 7, "last_error": None}
    c = _snap_app(tmp_path, session)
    s = c.get("/api/status").get_json()
    assert s["viewfinder"]["frames_captured"] == 7


class _FakeSession:
    """Stands in for CameraSession: serves N frames then reports closed."""

    def __init__(self, n_frames=2, paused=False):
        self.n = n_frames
        self.paused = paused
        self.pause_called = None

    def get_frame(self):
        if self.n <= 0:
            return None            # camera closed -> stream should end
        self.n -= 1
        arr = np.zeros((48, 64, 3), np.uint8)
        arr[:, :, 0] = 150
        return arr

    def pause_and_close(self, seconds=20):
        self.pause_called = seconds
        self.paused = True

    def info(self):
        return {"running": self.n > 0, "warmed": self.n > 0,
                "paused": self.paused, "frames_captured": 0,
                "last_error": None}


def _stream_app(tmp_path, session, service_active=False, start_fn=None):
    app = gcs.create_app(base=str(tmp_path / "flights"),
                         fields_dir=str(tmp_path / "fields"),
                         session=session,
                         start_fn=start_fn or (lambda base, **kw: 0),
                         stop_fn=lambda base, **kw: 0,
                         flight_service_active_fn=lambda: service_active,
                         config={"ndvi": {"gamma": 0.8, "leakage_k": 2.0}})
    app.testing = True
    return app.test_client()


def test_stream_serves_mjpeg_until_camera_closes(tmp_path):
    session = _FakeSession(n_frames=2)
    c = _stream_app(tmp_path, session)
    r = c.get("/api/stream.mjpg")
    assert r.status_code == 200
    assert "multipart/x-mixed-replace" in r.content_type
    body = r.data                      # consumes the generator to its end
    assert body.count(b"--frame") == 2
    assert b"\xff\xd8" in body         # JPEG magic inside a part


def test_stream_refused_while_flight_service_runs(tmp_path):
    c = _stream_app(tmp_path, _FakeSession(), service_active=True)
    assert c.get("/api/stream.mjpg").status_code == 409


def test_stream_409_when_session_paused(tmp_path):
    c = _stream_app(tmp_path, _FakeSession(paused=True))
    assert c.get("/api/stream.mjpg").status_code == 409


def test_start_hands_camera_over(tmp_path):
    session = _FakeSession()
    c = _stream_app(tmp_path, session)
    r = c.post("/api/start", json={})
    assert r.status_code == 200
    assert session.pause_called == 20  # viewfinder released before capture


def test_track_empty_without_gps(client):
    j = client.get("/api/track").get_json()
    assert j["fixes"] == []


def test_track_with_injected_gps(tmp_path):
    from cropvolare.gpsread import GpsReader
    gps = GpsReader()
    for i in range(4):
        gps._accept({"time": "t", "lat": 40.0 + i * 1e-5, "lon": -88.0,
                     "alt": 50.0})
    app = gcs.create_app(base=str(tmp_path / "flights"),
                         fields_dir=str(tmp_path / "fields"), gps=gps,
                         config={})
    app.testing = True
    c = app.test_client()
    j = c.get("/api/track").get_json()
    assert len(j["fixes"]) == 4
    j = c.get("/api/track?since=3").get_json()   # WiFi-drop backfill
    assert len(j["fixes"]) == 1
    s = c.get("/api/status").get_json()
    assert s["gps"]["lat"] == pytest.approx(40.00003)


# --- on-device processing + reports + ops ------------------------------------

class _FakeRunner:
    """Stands in for jobs.JobRunner: records start() calls."""

    def __init__(self, start_ok=True, err=None):
        self.calls = []
        self.start_ok = start_ok
        self.err = err
        self._status = {"state": "idle", "flight": None, "pct": 0,
                        "eta_s": None, "verdict": None, "error": None}

    def start(self, flight, field=None):
        self.calls.append((flight, field))
        if not self.start_ok:
            return False, self.err
        self._status.update(state="running", flight=flight)
        return True, None

    def status(self):
        return dict(self._status)


def _proc_app(tmp_path, runner, config=None, shutdown_fn=None):
    app = gcs.create_app(base=str(tmp_path / "flights"),
                         fields_dir=str(tmp_path / "fields"),
                         start_fn=lambda base, **kw: 0,
                         stop_fn=lambda base, **kw: 0,
                         config=config if config is not None else {"ndvi": {}},
                         runner=runner,
                         shutdown_fn=shutdown_fn or (lambda: 0),
                         version="abc1234 2026-07-08")
    app.testing = True
    return app.test_client()


def test_farmer_pages_and_manifest(client):
    assert b"CropVolare" in client.get("/").data
    assert client.get("/pro").status_code == 200
    assert client.get("/report").status_code == 200
    m = client.get("/manifest.json").get_json()
    assert m["name"] == "CropVolare"
    assert m["display"] == "standalone"
    assert client.get("/static/icon-192.png").status_code == 200


def test_process_endpoints(tmp_path):
    runner = _FakeRunner()
    c = _proc_app(tmp_path, runner)
    r = c.post("/api/process/f1", json={"field": "yard"})
    assert r.status_code == 200
    assert runner.calls == [("f1", "yard")]
    s = c.get("/api/process").get_json()
    assert s["state"] == "running" and s["flight"] == "f1"


def test_process_validates_names(tmp_path):
    runner = _FakeRunner()
    c = _proc_app(tmp_path, runner)
    assert c.post("/api/process/a b").status_code == 400        # bad flight
    r = c.post("/api/process/f1", json={"field": "../evil"})
    assert r.status_code == 400                                  # bad field
    assert runner.calls == []


def test_process_maps_runner_errors_to_http(tmp_path):
    c = _proc_app(tmp_path, _FakeRunner(start_ok=False,
                                        err="unknown flight: f9"))
    assert c.post("/api/process/f9").status_code == 404
    c = _proc_app(tmp_path, _FakeRunner(start_ok=False,
                                        err="a processing job is already running"))
    assert c.post("/api/process/f1").status_code == 409


def test_stop_triggers_auto_process(tmp_path):
    from cropvolare import flightctl
    runner = _FakeRunner()
    d = _add_flight(tmp_path, "big", frames=55)
    flightctl.write_meta(str(tmp_path / "flights"), {"dir": str(d), "pid": None})
    c = _proc_app(tmp_path, runner,
                  config={"gcs": {"auto_process": True,
                                  "auto_process_min_frames": 50,
                                  "default_field": "home"}})
    j = c.post("/api/stop").get_json()
    assert j["ok"] is True
    assert j["processing"] is True
    assert runner.calls == [("big", "home")]


def test_stop_skips_auto_process_for_small_captures(tmp_path):
    from cropvolare import flightctl
    runner = _FakeRunner()
    d = _add_flight(tmp_path, "tiny", frames=3)
    flightctl.write_meta(str(tmp_path / "flights"), {"dir": str(d), "pid": None})
    c = _proc_app(tmp_path, runner, config={"gcs": {"auto_process": True}})
    j = c.post("/api/stop").get_json()
    assert j["processing"] is False
    assert runner.calls == []


def test_stop_respects_auto_process_off(tmp_path):
    from cropvolare import flightctl
    runner = _FakeRunner()
    d = _add_flight(tmp_path, "big", frames=55)
    flightctl.write_meta(str(tmp_path / "flights"), {"dir": str(d), "pid": None})
    c = _proc_app(tmp_path, runner, config={"gcs": {"auto_process": False}})
    j = c.post("/api/stop").get_json()
    assert j["processing"] is False
    assert runner.calls == []


def test_reports_listing_and_artifacts(tmp_path):
    c = _proc_app(tmp_path, _FakeRunner())
    adir = tmp_path / "flights" / "f1" / "analysis"
    os.makedirs(str(adir))
    with open(str(adir / "result.json"), "w") as f:
        json.dump({"date": "2026-07-08", "field": "yard", "n_frames": 5,
                   "verdict": {"level": "fair", "score": 61, "line": "ok"},
                   "distribution": {"mean": 0.41}}, f)
    with open(str(adir / "report.pdf"), "wb") as f:
        f.write(b"%PDF-1.4 fake")

    rows = c.get("/api/reports").get_json()
    assert len(rows) == 1
    assert rows[0]["flight"] == "f1"
    assert rows[0]["level"] == "fair"
    assert rows[0]["has_pdf"] is True
    assert c.get("/api/reports/f1/report.pdf").data.startswith(b"%PDF")
    assert c.get("/api/reports/f1/result.json").get_json()["field"] == "yard"
    assert c.get("/api/reports/nope/report.pdf").status_code == 404
    assert c.get("/api/reports/a b/report.pdf").status_code == 400


def test_shutdown_endpoint(tmp_path):
    calls = []
    c = _proc_app(tmp_path, _FakeRunner(),
                  shutdown_fn=lambda: calls.append(1) or 0)
    assert c.post("/api/shutdown").status_code == 200
    assert calls == [1]


def test_shutdown_refused_while_capturing(tmp_path):
    from cropvolare import flightctl
    d = _add_flight(tmp_path, "f1")
    flightctl.write_meta(str(tmp_path / "flights"),
                         {"dir": str(d), "pid": os.getpid()})  # "alive" pid
    calls = []
    c = _proc_app(tmp_path, _FakeRunner(),
                  shutdown_fn=lambda: calls.append(1) or 0)
    assert c.post("/api/shutdown").status_code == 409
    assert calls == []


def test_status_includes_version_and_processing(tmp_path):
    c = _proc_app(tmp_path, _FakeRunner())
    s = c.get("/api/status").get_json()
    assert s["version"] == "abc1234 2026-07-08"
    assert s["processing"]["state"] == "idle"
