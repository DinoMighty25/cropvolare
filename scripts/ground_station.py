#!/usr/bin/env python3
"""Ground-station helper - runs on your LAPTOP, not the Pi.

One command to go from "photos sitting on the Pi" to "report.pdf open on screen".
Optionally pulls the flight folder off the Pi over SSH, then runs the full NDVI
analysis (process_flight.py) on your laptop where there's CPU and RAM to spare.

    # pull the folder from the Pi, analyze, and open the report:
    python scripts/ground_station.py --host dinomighty@192.168.1.50 \
        --remote flights/today --input flights/today --outdir output/today --open

    # analyze a folder you already copied (SD card / manual scp):
    python scripts/ground_station.py --input flights/today --open

Processing always happens here on the laptop; the Pi only ever captures.
"""

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROCESS_FLIGHT = os.path.join(HERE, "process_flight.py")


def pull_from_pi(host, remote, dest):
    """scp -r the remote flight folder from the Pi to a local destination."""
    parent = os.path.dirname(os.path.abspath(dest))
    os.makedirs(parent or ".", exist_ok=True)
    cmd = ["scp", "-r", f"{host}:{remote}", dest]
    print("pulling:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def open_file(path):
    """Open a file with the OS default viewer (cross-platform)."""
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.run(["open", path], check=False)
        else:
            subprocess.run(["xdg-open", path], check=False)
    except Exception as exc:  # noqa: BLE001 - opening is best-effort
        print(f"could not open {path}: {exc}")


def main():
    p = argparse.ArgumentParser(
        description="Pull a flight off the Pi (optional) and analyze it")
    p.add_argument("-i", "--input", required=True,
                   help="local flight folder (also the scp destination if --host)")
    p.add_argument("-o", "--outdir", default=None,
                   help="output folder (default: output/<input folder name>)")
    p.add_argument("--host", help="Pi SSH target, e.g. dinomighty@192.168.1.50")
    p.add_argument("--remote", help="flight folder path on the Pi (used with --host)")
    # pass-throughs to process_flight.py
    p.add_argument("--config")
    p.add_argument("--cell-meters")
    p.add_argument("--gamma")
    p.add_argument("--leakage-k")
    p.add_argument("--block-size")
    p.add_argument("--top-n")
    p.add_argument("--no-overlays", action="store_true")
    p.add_argument("--open", action="store_true",
                   help="open report.pdf when analysis finishes")
    args = p.parse_args()

    if args.host:
        if not args.remote:
            p.error("--remote is required when using --host")
        pull_from_pi(args.host, args.remote, args.input)

    outdir = args.outdir or os.path.join(
        "output", os.path.basename(os.path.normpath(args.input)))

    cmd = [sys.executable, PROCESS_FLIGHT, "--input", args.input, "--outdir", outdir]
    for flag in ("config", "cell_meters", "gamma", "leakage_k", "block_size", "top_n"):
        val = getattr(args, flag)
        if val is not None:
            cmd += ["--" + flag.replace("_", "-"), str(val)]
    if args.no_overlays:
        cmd.append("--no-overlays")

    print("analyzing:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    report = os.path.join(outdir, "report.pdf")
    print(f"\nreport: {report}")
    if args.open and os.path.exists(report):
        open_file(report)


if __name__ == "__main__":
    main()
