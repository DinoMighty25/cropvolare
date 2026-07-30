"""
Flight capture control: start/stop/status for detached capture runs.

This is the machinery behind both the fly.py CLI and the GCS web app - one
implementation, two frontends. A capture run is a detached capture_flight.py
process writing into a timestamped flights/ folder; bookkeeping lives in
flights/active.json while a run is up, and stopping works through a STOP file
in the flight folder (works from any later session/process).
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

from . import exposure

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAPTURE_SCRIPT = os.path.join(REPO, "scripts", "capture_flight.py")
DEFAULT_CONFIG = os.path.join(REPO, "config", "default.json")
ACTIVE_META = "active.json"  # lives in the flights base dir while capturing
DEFAULT_BASE = os.path.join(REPO, "flights")

MIN_FREE_MB = 200


def _load_config(path=None):
    path = path or DEFAULT_CONFIG
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


# --------------------------------------------------------------------------
# preflight checks
# --------------------------------------------------------------------------

def check_disk(base):
    free_mb = shutil.disk_usage(os.path.abspath(base)).free / (1024 * 1024)
    ok = free_mb > MIN_FREE_MB
    return ok, f"{free_mb:.0f} MB free" + ("" if ok else f" (need > {MIN_FREE_MB} MB)")


def check_writable(base):
    probe = os.path.join(base, ".write_test")
    try:
        with open(probe, "w") as f:
            f.write("ok")
        os.remove(probe)
        return True, "folder writable"
    except OSError as exc:
        return False, f"cannot write: {exc}"


def check_camera(_base):
    try:
        from picamera2 import Picamera2  # noqa: F401
        return True, "picamera2 available"
    except ImportError:
        return False, "picamera2 not installed (are you on the Pi?)"


DEFAULT_CHECKS = (
    ("disk space", check_disk),
    ("storage", check_writable),
    ("camera", check_camera),
)


def preflight(base, checks=DEFAULT_CHECKS, log_fn=print):
    """Run each (name, fn) check; fn(base) -> (ok, detail). True if all pass."""
    os.makedirs(base, exist_ok=True)
    all_ok = True
    for name, fn in checks:
        ok, detail = fn(base)
        log_fn(f"  [{'OK' if ok else 'FAIL'}] {name}: {detail}")
        all_ok = all_ok and ok
    return all_ok


# --------------------------------------------------------------------------
# flight bookkeeping
# --------------------------------------------------------------------------

def make_flight_dir(base, now=None):
    """Create flights/<YYYY-mm-dd_HHMMSS>, suffixed if it somehow exists."""
    now = now or datetime.now()
    stamp = now.strftime("%Y-%m-%d_%H%M%S")
    path = os.path.join(base, stamp)
    n = 2
    while os.path.exists(path):
        path = os.path.join(base, f"{stamp}-{n}")
        n += 1
    os.makedirs(path)
    return path


def _meta_path(base):
    return os.path.join(base, ACTIVE_META)


def read_meta(base):
    try:
        with open(_meta_path(base)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def write_meta(base, meta):
    with open(_meta_path(base), "w") as f:
        json.dump(meta, f, indent=2)


def clear_meta(base):
    try:
        os.remove(_meta_path(base))
    except OSError:
        pass


def pid_alive(pid):
    if not pid:
        return False
    if os.name == "nt":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"],
                             capture_output=True, text=True)
        return str(pid) in out.stdout
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def count_frames(flight_dir):
    return len(glob.glob(os.path.join(flight_dir, "*.jpg")))


def list_flights(base):
    """Newest-first [{name, n_frames, size_mb}] for every flight folder."""
    if not os.path.isdir(base):
        return []
    entries = []
    for name in os.listdir(base):
        d = os.path.join(base, name)
        if not os.path.isdir(d):
            continue
        jpgs = glob.glob(os.path.join(d, "*.jpg"))
        size = sum(os.path.getsize(p) for p in jpgs)
        entries.append({
            "name": name,
            "n_frames": len(jpgs),
            "size_mb": round(size / (1024 * 1024), 1),
            "mtime": os.path.getmtime(d),
        })
    entries.sort(key=lambda e: e["mtime"], reverse=True)
    for e in entries:
        e.pop("mtime")
    return entries


def latest_frame(base):
    """Path of the newest jpg across the active (or newest) flight, or None."""
    meta = read_meta(base)
    dirs = []
    if meta:
        dirs.append(meta["dir"])
    dirs += [os.path.join(base, e["name"]) for e in list_flights(base)]
    for d in dirs:
        frames = sorted(glob.glob(os.path.join(d, "*.jpg")),
                        key=os.path.getmtime)
        # newest first; skip 0-byte in-progress writes
        for f in reversed(frames):
            if os.path.getsize(f) > 0:
                return f
    return None


# --------------------------------------------------------------------------
# storage autopilot
# --------------------------------------------------------------------------

def free_mb(base):
    return shutil.disk_usage(os.path.abspath(base)).free / (1024 * 1024)


def storage_autopilot(base, min_free_mb=2048, min_keep_frames=15,
                      log_fn=print, free_mb_fn=None):
    """Keep the SD card healthy without SSH: runs at GCS boot + after jobs.

    Two rules, in order:
      1. purge trivial captures (< min_keep_frames frames) - bench/desk junk
         from power-on tests that otherwise accumulates forever;
      2. while free space < min_free_mb, delete the OLDEST flight that has
         already been processed (analysis/report.pdf exists). Unprocessed
         real flights are never touched - data loss is worse than a full card.

    The active flight (capture in progress) is always exempt. Returns the list
    of deleted flight names.
    """
    free_fn = free_mb_fn or (lambda: free_mb(base))
    meta = read_meta(base)
    active = os.path.basename(meta["dir"]) if meta else None
    deleted = []

    flights = list_flights(base)  # newest first
    for e in flights:
        if e["name"] == active:
            continue
        if e["n_frames"] < min_keep_frames:
            shutil.rmtree(os.path.join(base, e["name"]), ignore_errors=True)
            deleted.append(e["name"])
            log_fn(f"storage: purged trivial capture {e['name']} "
                   f"({e['n_frames']} frames)")

    # oldest-first pass over processed flights until there's room
    remaining = [e for e in reversed(list_flights(base)) if e["name"] != active]
    for e in remaining:
        if free_fn() >= min_free_mb:
            break
        report = os.path.join(base, e["name"], "analysis", "report.pdf")
        if not os.path.exists(report):
            continue
        shutil.rmtree(os.path.join(base, e["name"]), ignore_errors=True)
        deleted.append(e["name"])
        log_fn(f"storage: deleted oldest processed flight {e['name']} "
               f"(free space below {min_free_mb} MB)")
    return deleted


# --------------------------------------------------------------------------
# commands (shared by fly.py CLI and the GCS web app)
# --------------------------------------------------------------------------

def status_info(base, alive_fn=pid_alive):
    """Machine-readable status dict (the GCS /api/status backbone)."""
    meta = read_meta(base)
    info = {
        "capturing": False,
        "flight": None,
        "frames": 0,
        "last_frame_age_s": None,
        "disk_free_mb": round(
            shutil.disk_usage(os.path.abspath(base)).free / (1024 * 1024))
        if os.path.isdir(base) else None,
    }
    if not meta:
        return info
    d = meta["dir"]
    frames = sorted(glob.glob(os.path.join(d, "*.jpg")), key=os.path.getmtime)
    info.update({
        "capturing": alive_fn(meta.get("pid")),
        "flight": os.path.basename(d),
        "pid": meta.get("pid"),
        "preset": meta.get("preset"),
        "frames": len(frames),
        "last_frame_age_s": (round(time.time() - os.path.getmtime(frames[-1]))
                             if frames else None),
    })
    return info


def start(base, interval=2.0, count=0, gps_port=None, foreground=False,
          skip_checks=False, log_fn=print, first_frame_timeout=90, preset=None):
    """first_frame_timeout: seconds to wait for frame_0000 before reporting
    failure. The Zero 2 W needs ~30-45 s of process/camera spin-up, so the CLI
    default is generous. Pass 0 to return right after spawning (the GCS does -
    its dashboard live-polls the frame counter instead of blocking).

    preset names an exposure preset from the config. It is validated HERE,
    before the capture process is spawned: capture runs detached with its output
    going to a log file, so a mistyped preset would otherwise look like a
    successful start and only surface as a missing first frame - in the field,
    with the drone waiting.
    """
    meta = read_meta(base)
    if meta and pid_alive(meta.get("pid")):
        log_fn(f"already capturing -> {meta['dir']} (pid {meta['pid']})")
        log_fn("stop the current run first")
        return 1

    if preset:
        try:
            cfg = _load_config()
            resolved = exposure.get_preset(cfg, preset)
            log_fn(f"exposure preset {preset!r}: {exposure.describe(resolved)}")
        except exposure.PresetError as exc:
            log_fn(f"preset FAILED: {exc}")
            return 1
    else:
        log_fn("no --preset: auto-exposure will settle per flight, so this "
               "flight will NOT be comparable to other days")

    log_fn("preflight:")
    if not preflight(base, log_fn=log_fn) and not skip_checks:
        log_fn("preflight FAILED - fix the above (or skip checks to override)")
        return 1

    outdir = make_flight_dir(base)
    n_frames_arg = str(count)
    cmd = [sys.executable, CAPTURE_SCRIPT, "-o", outdir,
           "--count", n_frames_arg, "--interval", str(interval)]
    if gps_port:
        cmd += ["--gps-port", gps_port]
    if preset:
        cmd += ["--preset", preset]

    if foreground:
        # boot/systemd mode: stay attached; stop still works via STOP file
        log_fn(f"capturing (foreground) -> {outdir}")
        proc = subprocess.Popen(cmd)
        write_meta(base, {"dir": outdir, "pid": proc.pid,
                          "started": datetime.now().isoformat(timespec="seconds"),
                          "interval": interval, "preset": preset})
        try:
            return proc.wait()
        finally:
            clear_meta(base)

    log_path = os.path.join(outdir, "capture.log")
    log_file = open(log_path, "ab")
    kwargs = {"stdout": log_file, "stderr": subprocess.STDOUT,
              "stdin": subprocess.DEVNULL}
    if os.name == "nt":
        kwargs["creationflags"] = (subprocess.DETACHED_PROCESS
                                   | subprocess.CREATE_NEW_PROCESS_GROUP)
    else:
        kwargs["start_new_session"] = True  # survives SSH disconnect
    proc = subprocess.Popen(cmd, **kwargs)
    write_meta(base, {"dir": outdir, "pid": proc.pid,
                      "started": datetime.now().isoformat(timespec="seconds"),
                      "interval": interval, "preset": preset})

    log_fn(f"capture started (pid {proc.pid}) -> {outdir}")
    if not first_frame_timeout:
        log_fn("capture starting - first frame takes ~30-45 s on the Pi; "
               "watch the frame counter")
        return 0
    log_fn("waiting for the first frame (~30-45 s on the Pi) ...")
    deadline = time.time() + first_frame_timeout
    while time.time() < deadline:
        if count_frames(outdir):
            log_fn("first frame saved - CAPTURE CONFIRMED, safe to fly")
            log_fn("(SSH may disconnect now; capture keeps running)")
            return 0
        if proc.poll() is not None:
            break
        time.sleep(0.5)
    log_fn(f"no frame yet - check {log_path}")
    return 1


def stop(base, log_fn=print, alive_fn=pid_alive, timeout=None):
    meta = read_meta(base)
    if not meta:
        log_fn("no active flight to stop")
        return 1
    d = meta["dir"]
    with open(os.path.join(d, "STOP"), "w") as f:
        f.write("stop requested\n")
    log_fn("stop requested, waiting for capture to finish its cycle ...")

    if timeout is None:
        timeout = meta.get("interval", 2.5) + 10
    deadline = time.time() + timeout
    while alive_fn(meta.get("pid")) and time.time() < deadline:
        time.sleep(0.5)

    if alive_fn(meta.get("pid")):
        log_fn("still running - it will stop on its next cycle")
    else:
        log_fn("capture stopped")
    log_fn(f"{count_frames(d)} frame(s) in {d}")
    clear_meta(base)
    return 0