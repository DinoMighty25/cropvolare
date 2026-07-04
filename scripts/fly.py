#!/usr/bin/env python3
"""One-command field capture - the script you run over SSH at the farm.

    python scripts/fly.py            # preflight checks + start capturing
    python scripts/fly.py status     # still running? how many photos so far?
    python scripts/fly.py stop       # finish cleanly and summarize

Start runs the capture DETACHED from your SSH session: if Wi-Fi drops while
the drone flies away, capture keeps going. Every run gets its own timestamped
folder under flights/, so nothing is ever overwritten. Stop works through a
STOP file in the flight folder, so it works from any later SSH session.
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CAPTURE_SCRIPT = os.path.join(HERE, "capture_flight.py")
ACTIVE_META = "active.json"  # lives in the flights base dir while capturing

MIN_FREE_MB = 200


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


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

def start(base, interval, count, gps_port, foreground=False, skip_checks=False,
          log_fn=print):
    meta = read_meta(base)
    if meta and pid_alive(meta.get("pid")):
        log_fn(f"already capturing -> {meta['dir']} (pid {meta['pid']})")
        log_fn("run 'python scripts/fly.py stop' first")
        return 1

    log_fn("preflight:")
    if not preflight(base, log_fn=log_fn) and not skip_checks:
        log_fn("preflight FAILED - fix the above (or --skip-checks to override)")
        return 1

    outdir = make_flight_dir(base)
    cmd = [sys.executable, CAPTURE_SCRIPT, "-o", outdir,
           "--count", str(count), "--interval", str(interval)]
    if gps_port:
        cmd += ["--gps-port", gps_port]

    if foreground:
        # boot/systemd mode: stay attached; stop still works via fly.py stop
        log_fn(f"capturing (foreground) -> {outdir}")
        proc = subprocess.Popen(cmd)
        write_meta(base, {"dir": outdir, "pid": proc.pid,
                          "started": datetime.now().isoformat(timespec="seconds"),
                          "interval": interval})
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
                      "interval": interval})

    log_fn(f"capture started (pid {proc.pid}) -> {outdir}")
    log_fn("waiting for the first frame ...")
    deadline = time.time() + 20
    while time.time() < deadline:
        if count_frames(outdir):
            log_fn("first frame saved - CAPTURE CONFIRMED, safe to fly")
            log_fn("(SSH may disconnect now; capture keeps running)")
            log_fn("when landed: python scripts/fly.py stop")
            return 0
        if proc.poll() is not None:
            break
        time.sleep(0.5)
    log_fn(f"no frame yet - check {log_path}")
    return 1


def status(base, log_fn=print, alive_fn=pid_alive):
    meta = read_meta(base)
    if not meta:
        log_fn("no active flight (nothing started, or already stopped)")
        return 1
    d = meta["dir"]
    running = alive_fn(meta.get("pid"))
    frames = sorted(glob.glob(os.path.join(d, "*.jpg")), key=os.path.getmtime)
    last = (f"{time.time() - os.path.getmtime(frames[-1]):.0f}s ago"
            if frames else "none yet")
    log_fn(f"flight:  {d}")
    log_fn(f"running: {'yes' if running else 'NO (process gone)'} "
           f"(pid {meta.get('pid')})")
    log_fn(f"frames:  {len(frames)} (last: {last})")
    return 0 if running else 1


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
    log_fn("next, on the laptop:")
    log_fn(f"  python scripts/ground_station.py --host <user>@<pi-ip> "
           f"--remote cropvolare/{os.path.relpath(d, REPO)} "
           f"--input {os.path.relpath(d, REPO)} --open")
    clear_meta(base)
    return 0


def main():
    p = argparse.ArgumentParser(description="One-command field capture")
    p.add_argument("command", nargs="?", default="start",
                   choices=["start", "status", "stop"])
    p.add_argument("--base", default=os.path.join(REPO, "flights"))
    p.add_argument("--interval", type=float, default=2.0,
                   help="seconds between frames")
    p.add_argument("--count", type=int, default=0,
                   help="frames to capture (0 = until stopped)")
    p.add_argument("--gps-port", default=None,
                   help="serial GPS port (e.g. /dev/serial0)")
    p.add_argument("--foreground", action="store_true",
                   help="stay attached (for systemd boot mode)")
    p.add_argument("--skip-checks", action="store_true")
    args = p.parse_args()

    if args.command == "start":
        sys.exit(start(args.base, args.interval, args.count, args.gps_port,
                       foreground=args.foreground,
                       skip_checks=args.skip_checks))
    elif args.command == "status":
        sys.exit(status(args.base))
    else:
        sys.exit(stop(args.base))


if __name__ == "__main__":
    main()
