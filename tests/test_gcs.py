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
