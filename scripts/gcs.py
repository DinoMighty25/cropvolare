#!/usr/bin/env python3
"""CropVolare Ground Control Station - phone-friendly web app served by the Pi.

    python scripts/gcs.py                 # serve on 0.0.0.0:8080
    python scripts/gcs.py --gps-port /dev/serial0

At the field, open http://<pi-ip>:8080 on your phone (hotspot):
  /          dashboard - live status, camera + NDVI preview, START/STOP
  /planner   field planner - draw a polygon, get the survey pattern,
             export KML/Litchi, watch the live GPS breadcrumb (when GPS wired)

No auth: hotspot/LAN use only. All flight machinery is cropvolare.flightctl
(same code as fly.py); previews only read saved frames - the GCS never opens
the camera, so it can't fight the capture process for it.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, Response, jsonify, request, send_from_directory

from cropvolare import flightctl, planner

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(REPO, "cropvolare", "gcs_static")
FIELDS_DIR = os.path.join(REPO, "fields")
DEFAULT_CONFIG = os.path.join(REPO, "config", "default.json")

_FIELD_NAME = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


def _load_config(path=DEFAULT_CONFIG):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _take_snapshot():
    """One live frame for the idle viewfinder. Opens the camera, grabs a frame,
    and RELEASES it completely - the GCS must never hold the camera, or the
    next capture start would fail."""
    from cropvolare.ndvi import capture_frame, create_camera
    cam = create_camera(resolution=(1152, 648))
    try:
        cam.start()
        time.sleep(0.8)  # brief AE settle; viewfinder, not science
        return capture_frame(cam)
    finally:
        try:
            cam.stop()
        finally:
            close = getattr(cam, "close", None)
            if close:
                close()


def _flight_service_active():
    """True while the boot-capture systemd unit is running (incl. its ~2 min
    camera spin-up, when no meta/frames exist yet) - the snapshot viewfinder
    must stay away from the camera for that whole window."""
    try:
        r = subprocess.run(["systemctl", "is-active", "--quiet",
                            "cropvolare-flight"], timeout=5)
        return r.returncode == 0
    except Exception:  # noqa: BLE001 - no systemd (laptop/tests) = not active
        return False


def create_app(base=None, fields_dir=None, gps=None,
               start_fn=None, stop_fn=None, config=None,
               snap_fn=None, flight_service_active_fn=None):
    """App factory; injectable pieces keep it fully testable on the laptop."""
    base = base or flightctl.DEFAULT_BASE
    fields_dir = fields_dir or FIELDS_DIR
    start_fn = start_fn or flightctl.start
    stop_fn = stop_fn or flightctl.stop
    snap_fn = snap_fn or _take_snapshot
    service_active_fn = flight_service_active_fn or _flight_service_active
    snap_lock = threading.Lock()
    cfg = config if config is not None else _load_config()
    ndvi_cfg = cfg.get("ndvi", {})

    app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="/static")
    state = {"gps": gps}

    # --- pages ------------------------------------------------------------
    @app.get("/")
    def index():
        return send_from_directory(STATIC_DIR, "index.html")

    @app.get("/planner")
    def planner_page():
        return send_from_directory(STATIC_DIR, "planner.html")

    # --- flight control ----------------------------------------------------
    @app.get("/api/status")
    def api_status():
        info = flightctl.status_info(base)
        info["gps"] = state["gps"].latest() if state["gps"] else None
        return jsonify(info)

    @app.post("/api/start")
    def api_start():
        body = request.get_json(silent=True) or {}
        log = []
        rc = start_fn(base, interval=float(body.get("interval", 2.0)),
                      count=0, gps_port=app.config.get("GPS_PORT"),
                      log_fn=log.append)
        return jsonify({"ok": rc == 0, "log": log}), (200 if rc == 0 else 409)

    @app.post("/api/stop")
    def api_stop():
        log = []
        rc = stop_fn(base, log_fn=log.append)
        return jsonify({"ok": rc == 0, "log": log}), (200 if rc == 0 else 409)

    @app.get("/api/flights")
    def api_flights():
        return jsonify(flightctl.list_flights(base))

    # --- previews (read saved frames only; never touches the camera) -------
    def _load_preview():
        import cv2
        path = flightctl.latest_frame(base)
        if not path:
            return None
        img = cv2.imread(path)
        if img is None:
            return None
        h, w = img.shape[:2]
        scale = 640.0 / max(h, w)
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)
        return img

    def _jpeg(img):
        import cv2
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return Response(buf.tobytes(), mimetype="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    @app.get("/api/preview.jpg")
    def api_preview():
        img = _load_preview()
        if img is None:
            return ("no frames yet", 404)
        return _jpeg(img)

    @app.get("/api/snapshot.jpg")
    def api_snapshot():
        # live viewfinder - only when nothing else can want the camera
        if flightctl.status_info(base)["capturing"] or service_active_fn():
            return ("camera busy: capture is running - the preview shows "
                    "flight frames instead", 409)
        with snap_lock:  # serialize concurrent viewfinder requests
            try:
                frame = snap_fn()
            except Exception as exc:  # noqa: BLE001 - no camera here (laptop) etc.
                return (f"camera unavailable: {exc}", 503)
        if request.args.get("ndvi") == "1":
            from cropvolare.ndvi import colorize_ndvi, compute_ndvi_from_image
            ndvi = compute_ndvi_from_image(
                frame, gamma=ndvi_cfg.get("gamma", 0.8),
                leakage_k=ndvi_cfg.get("leakage_k", 2.0))
            frame = colorize_ndvi(ndvi)
        return _jpeg(frame)

    @app.get("/api/preview_ndvi.jpg")
    def api_preview_ndvi():
        from cropvolare.ndvi import colorize_ndvi, compute_ndvi_from_image
        img = _load_preview()
        if img is None:
            return ("no frames yet", 404)
        ndvi = compute_ndvi_from_image(
            img, gamma=ndvi_cfg.get("gamma", 0.8),
            leakage_k=ndvi_cfg.get("leakage_k", 2.0))
        return _jpeg(colorize_ndvi(ndvi))

    # --- fields (saved polygons) -------------------------------------------
    @app.get("/api/fields")
    def api_fields():
        out = []
        if os.path.isdir(fields_dir):
            for fn in sorted(os.listdir(fields_dir)):
                if fn.endswith(".geojson"):
                    try:
                        with open(os.path.join(fields_dir, fn)) as f:
                            gj = json.load(f)
                        out.append({"name": fn[:-8],
                                    "polygon": gj["polygon"]})
                    except (OSError, ValueError, KeyError):
                        continue
        return jsonify(out)

    @app.post("/api/fields")
    def api_save_field():
        body = request.get_json(silent=True) or {}
        name = body.get("name", "")
        polygon = body.get("polygon")
        if not _FIELD_NAME.match(name):
            return jsonify({"ok": False,
                            "error": "name: letters/digits/_- only"}), 400
        if not polygon or len(polygon) < 3:
            return jsonify({"ok": False, "error": "polygon needs >=3 points"}), 400
        os.makedirs(fields_dir, exist_ok=True)
        with open(os.path.join(fields_dir, f"{name}.geojson"), "w") as f:
            json.dump({"name": name, "polygon": polygon}, f, indent=2)
        return jsonify({"ok": True})

    # --- survey planning -----------------------------------------------------
    @app.post("/api/plan")
    def api_plan():
        body = request.get_json(silent=True) or {}
        polygon = [tuple(p) for p in body.get("polygon", [])]
        alt = float(body.get("altitude", 30.0))
        overlap = float(body.get("overlap", 0.75))
        speed = float(body.get("speed", 4.0))
        interval = float(body.get("interval", 2.0))
        try:
            lines = planner.survey_lines(polygon, alt, side_overlap=overlap)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        stats = planner.plan_stats(lines, alt, speed_mps=speed,
                                   interval_s=interval)
        return jsonify({"ok": True, "lines": lines, "stats": stats})

    @app.get("/api/export")
    def api_export():
        name = request.args.get("field", "")
        fmt = request.args.get("fmt", "kml")
        alt = float(request.args.get("altitude", 30.0))
        overlap = float(request.args.get("overlap", 0.75))
        if not _FIELD_NAME.match(name):
            return ("bad field name", 400)
        path = os.path.join(fields_dir, f"{name}.geojson")
        if not os.path.exists(path):
            return ("unknown field", 404)
        with open(path) as f:
            polygon = [tuple(p) for p in json.load(f)["polygon"]]
        lines = planner.survey_lines(polygon, alt, side_overlap=overlap)
        if fmt == "litchi":
            body = planner.to_litchi_csv(lines, altitude_m=alt)
            return Response(body, mimetype="text/csv", headers={
                "Content-Disposition":
                    f"attachment; filename={name}_litchi.csv"})
        body = planner.to_kml(lines, name=name)
        return Response(body,
                        mimetype="application/vnd.google-earth.kml+xml",
                        headers={"Content-Disposition":
                                 f"attachment; filename={name}.kml"})

    # --- live GPS breadcrumb (Phase C; empty until a GPS is wired) ----------
    @app.get("/api/track")
    def api_track():
        since = int(request.args.get("since", 0))
        fixes = state["gps"].track(since) if state["gps"] else []
        return jsonify({"since": since, "fixes": fixes})

    return app


def main():
    p = argparse.ArgumentParser(description="CropVolare ground control station")
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--base", default=None, help="flights base dir")
    p.add_argument("--gps-port", default=None,
                   help="serial GPS for the live breadcrumb (e.g. /dev/serial0)")
    args = p.parse_args()

    cfg = _load_config()
    gcs_cfg = cfg.get("gcs", {})
    port = args.port or gcs_cfg.get("port", 8080)
    gps_port = args.gps_port or gcs_cfg.get("gps_port")

    gps = None
    if gps_port:
        from cropvolare.gpsread import GpsReader
        print(f"starting GPS reader on {gps_port} ...")
        gps = GpsReader(port=gps_port).start()

    app = create_app(base=args.base, gps=gps, config=cfg)
    app.config["GPS_PORT"] = gps_port
    print(f"ground station: http://0.0.0.0:{port}  (phone: http://<pi-ip>:{port})")
    app.run(host="0.0.0.0", port=port, threaded=True)


if __name__ == "__main__":
    main()
