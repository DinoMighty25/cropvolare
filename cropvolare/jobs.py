"""
On-device processing jobs: run the flight -> report pipeline ON the Pi.

One background job at a time (the Zero 2 W has 512 MB - two concurrent
pipelines would OOM it), running the same process_flight.run() the laptop CLI
uses, at the reduced processing.scale from the config. Progress is mirrored to
<flight>/analysis/status.json so a phone reconnecting after a GCS restart can
still see where things stand, and the AnalysisResult lands in
<flight>/analysis/result.json for the mobile report view.

The runner NEVER starts while capture is running: processing competes with
capture for CPU/RAM and the flight folder is still being written.
"""

import json
import os
import threading
import time

from . import flightctl

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(REPO, "config", "default.json")
PROCESS_FLIGHT = os.path.join(REPO, "scripts", "process_flight.py")

ANALYSIS_DIRNAME = "analysis"

_run_fn_cache = None


def _load_run():
    """Import process_flight.run() (scripts/ isn't a package; load by path)."""
    global _run_fn_cache
    if _run_fn_cache is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location("process_flight",
                                                      PROCESS_FLIGHT)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _run_fn_cache = mod.run
    return _run_fn_cache


def _load_config(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def analysis_dir(base, flight):
    return os.path.join(base, flight, ANALYSIS_DIRNAME)


def list_reports(base):
    """Newest-first list of processed flights for the GCS Reports tab.

    A flight counts as a report once its analysis/result.json exists; the
    entry carries just enough for a traffic-light list row.
    """
    out = []
    for e in flightctl.list_flights(base):
        rj = os.path.join(analysis_dir(base, e["name"]), "result.json")
        if not os.path.exists(rj):
            continue
        try:
            with open(rj) as f:
                result = json.load(f)
        except (OSError, ValueError):
            continue
        verdict = result.get("verdict") or {}
        out.append({
            "flight": e["name"],
            "date": result.get("date"),
            "field": result.get("field"),
            "level": verdict.get("level"),
            "score": verdict.get("score"),
            "line": verdict.get("line"),
            "mean_ndvi": (result.get("distribution") or {}).get("mean"),
            "n_frames": result.get("n_frames"),
            "has_pdf": os.path.exists(
                os.path.join(analysis_dir(base, e["name"]), "report.pdf")),
        })
    return out


class JobRunner:
    """One-at-a-time background processing of a flight folder.

    Injectable run_fn / capture_active_fn keep it fully testable without a
    camera or the heavyweight pipeline. on_done(status) fires after every
    finished job (success or error) - the GCS hangs storage cleanup on it.
    """

    def __init__(self, base=None, config_path=DEFAULT_CONFIG, run_fn=None,
                 capture_active_fn=None, on_done=None):
        self.base = base or flightctl.DEFAULT_BASE
        self.config_path = config_path
        self._run_fn = run_fn
        self._capture_active = capture_active_fn or (
            lambda: flightctl.status_info(self.base)["capturing"])
        self._on_done = on_done
        self._lock = threading.Lock()        # guards _status
        self._start_lock = threading.Lock()  # serializes start() itself
        self._thread = None
        self._status = {"state": "idle", "flight": None, "pct": 0,
                        "eta_s": None, "verdict": None, "error": None}

    # -- status ------------------------------------------------------------

    def status(self):
        with self._lock:
            return dict(self._status)

    def _set(self, **kw):
        with self._lock:
            self._status.update(kw)
            snapshot = dict(self._status)
        flight = snapshot.get("flight")
        if flight:
            adir = analysis_dir(self.base, flight)
            try:
                os.makedirs(adir, exist_ok=True)
                with open(os.path.join(adir, "status.json"), "w") as f:
                    json.dump(snapshot, f)
            except OSError:
                pass  # a failed status mirror must never kill the job
        return snapshot

    # -- control -----------------------------------------------------------

    def start(self, flight, field=None):
        """Kick off processing of flights/<flight>. Returns (ok, error)."""
        with self._start_lock:
            if self._thread is not None and self._thread.is_alive():
                return False, "a processing job is already running"
            if self._capture_active():
                return False, "capture is running - process after landing"
            flight_dir = os.path.join(self.base, flight)
            if not os.path.isdir(flight_dir):
                return False, f"unknown flight: {flight}"
            if flightctl.count_frames(flight_dir) == 0:
                return False, "flight has no frames"

            with self._lock:
                self._status = {"state": "running", "flight": flight, "pct": 0,
                                "eta_s": None, "verdict": None, "error": None}
            self._set()  # mirror the starting state immediately
            self._thread = threading.Thread(target=self._work,
                                            args=(flight, field), daemon=True)
            self._thread.start()
            return True, None

    def wait(self, timeout=None):
        """Block until the current job finishes (tests + CLI use)."""
        t = self._thread
        if t:
            t.join(timeout)
        return self.status()

    # -- the job -----------------------------------------------------------

    def _work(self, flight, field):
        flight_dir = os.path.join(self.base, flight)
        outdir = analysis_dir(self.base, flight)
        cfg = _load_config(self.config_path)
        scale = cfg.get("processing", {}).get("scale", 0.5)
        started = time.time()
        last_pct = {"v": -1}

        def progress(done, total):
            pct = int(100 * done / total) if total else 100
            if pct == last_pct["v"]:
                return
            last_pct["v"] = pct
            elapsed = time.time() - started
            eta = int(elapsed / done * (total - done)) if done else None
            self._set(pct=pct, eta_s=eta)

        try:
            run_fn = self._run_fn or _load_run()
            summary = run_fn(flight_dir, outdir, config_path=self.config_path,
                             field_name=field, process_scale=scale,
                             progress_fn=progress, log_fn=lambda *_: None)
            result = summary.get("result")
            if result is not None:
                with open(os.path.join(outdir, "result.json"), "w") as f:
                    json.dump(result, f, indent=2)
            status = self._set(state="done", pct=100, eta_s=0,
                               verdict=summary.get("verdict"))
        except Exception as exc:  # noqa: BLE001 - job errors surface via status
            status = self._set(state="error", error=str(exc))
        if self._on_done:
            try:
                self._on_done(status)
            except Exception:  # noqa: BLE001 - cleanup hook must not crash us
                pass
