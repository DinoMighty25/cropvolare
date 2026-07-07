#!/usr/bin/env python3
"""One-command field capture - the script you run over SSH at the farm.

    python scripts/fly.py            # preflight checks + start capturing
    python scripts/fly.py status     # still running? how many photos so far?
    python scripts/fly.py stop       # finish cleanly and summarize

All the machinery lives in cropvolare.flightctl (shared with the GCS web app);
this is just the command-line frontend. Start runs the capture DETACHED from
your SSH session, every run gets its own timestamped flights/ folder, and stop
works through a STOP file so it works from any later session.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cropvolare import flightctl


def main():
    p = argparse.ArgumentParser(description="One-command field capture")
    p.add_argument("command", nargs="?", default="start",
                   choices=["start", "status", "stop"])
    p.add_argument("--base", default=flightctl.DEFAULT_BASE)
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
        sys.exit(flightctl.start(args.base, args.interval, args.count,
                                 args.gps_port, foreground=args.foreground,
                                 skip_checks=args.skip_checks))
    elif args.command == "status":
        info = flightctl.status_info(args.base)
        if info["flight"] is None:
            print("no active flight (nothing started, or already stopped)")
            sys.exit(1)
        print(f"flight:  {info['flight']}")
        print(f"running: {'yes' if info['capturing'] else 'NO (process gone)'} "
              f"(pid {info.get('pid')})")
        age = (f"{info['last_frame_age_s']}s ago"
               if info["last_frame_age_s"] is not None else "none yet")
        print(f"frames:  {info['frames']} (last: {age})")
        sys.exit(0 if info["capturing"] else 1)
    else:
        rc = flightctl.stop(args.base)
        if rc == 0:
            print("next, on the laptop:")
            print("  python scripts/ground_station.py --host <user>@<pi-ip> "
                  "--latest --open")
        sys.exit(rc)


if __name__ == "__main__":
    main()
