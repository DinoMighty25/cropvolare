#!/bin/bash
# Witty Pi "before shutdown" hook: end an in-progress flight capture cleanly.
#
# WHY THIS EXISTS
# The Witty Pi cuts power when the battery hits the low-voltage threshold. If a
# capture is running at that moment, the process dies mid-write: the last frames
# are zero-byte JPEGs, the flight folder has no capture_meta.json, and the flight
# is unusable. Field data cannot be re-collected retroactively, so the shutdown
# path has to be graceful.
#
# capture_flight.py already flushes the write cache every few frames and stops on
# a STOP file. This hook creates that STOP file (via fly.py stop), waits for the
# capture process to finish its cycle, then syncs the filesystem - so the worst
# case is losing the single frame in flight rather than the whole flight.
#
# INSTALL (on the Pi)
#   1. Make it executable:
#        chmod +x ~/cropvolare/scripts/wittypi_before_shutdown.sh
#   2. Hook it into the Witty Pi software:
#        nano ~/wittypi/beforeShutdown.sh
#      and add the ABSOLUTE path to this script. Print it rather than typing it,
#      because it depends on the account the repo lives under:
#        realpath ~/cropvolare/scripts/wittypi_before_shutdown.sh
#      e.g. /home/dinomighty/cropvolare/scripts/wittypi_before_shutdown.sh
#      A wrong path here fails silently - Witty Pi cuts power with the capture
#      still running, which is the exact loss this hook exists to prevent, so
#      verify with step 4 rather than assuming.
#   3. Set the voltage thresholds in the Witty Pi menu (sudo ~/wittypi/wittyPi.sh):
#        low voltage threshold      3.4 V   (shut down below this)
#        recovery voltage threshold 3.7 V   (boot again above this)
#      3.4 V leaves usable margin above the cell's damage point while giving the
#      shutdown time to complete.
#   4. TEST IT before trusting it - see the bottom of this file.
#
# The hook must be quick: the Witty Pi will cut power regardless after its own
# grace period, so everything here is bounded by TIMEOUT.

set -u

REPO="${CROPVOLARE_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LOG="${CROPVOLARE_SHUTDOWN_LOG:-$REPO/flights/shutdown.log}"
TIMEOUT="${CROPVOLARE_SHUTDOWN_TIMEOUT:-25}"
PYTHON="${CROPVOLARE_PYTHON:-python3}"

mkdir -p "$(dirname "$LOG")" 2>/dev/null || true

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

log "low-battery shutdown requested"

# Record the voltage if the Witty Pi utilities are available - turns "why did it
# die?" into a number you can read afterwards.
#
# Witty Pi installs under whichever account was used, so its path is not
# knowable up front. Probe instead of hardcoding /home/pi: the first candidate is
# a sibling of this repo, which is the reliable one because beforeShutdown.sh
# runs as ROOT during shutdown - $HOME is /root there, not the owning user's
# home, so anything $HOME-derived would miss.
if [ -z "${WITTYPI_DIR:-}" ]; then
    for _d in "$(dirname "$REPO")/wittypi" "$HOME/wittypi" \
              /home/pi/wittypi /opt/wittypi; do
        if [ -r "$_d/utilities.sh" ]; then
            WITTYPI_DIR="$_d"
            break
        fi
    done
fi

if [ -n "${WITTYPI_DIR:-}" ] && [ -r "$WITTYPI_DIR/utilities.sh" ]; then
    # shellcheck disable=SC1091
    . "$WITTYPI_DIR/utilities.sh" 2>/dev/null || true
    if command -v get_input_voltage >/dev/null 2>&1; then
        log "input voltage: $(get_input_voltage 2>/dev/null || echo unknown) V"
    fi
else
    log "wittypi utilities not found - no voltage logged (set WITTYPI_DIR)"
fi

ACTIVE="$REPO/flights/active.json"
if [ ! -f "$ACTIVE" ]; then
    log "no active flight - nothing to stop"
    sync
    exit 0
fi

log "active flight found; requesting clean stop"

# fly.py stop writes the STOP file and waits for the capture process to exit.
# Bounded by `timeout` so a wedged process cannot hold up the shutdown past the
# Witty Pi's own grace period.
if timeout "$TIMEOUT" "$PYTHON" "$REPO/scripts/fly.py" stop >>"$LOG" 2>&1; then
    log "capture stopped cleanly"
else
    rc=$?
    log "fly.py stop returned $rc (timeout or error) - falling back to STOP file"
    # Last resort: write the STOP file directly, in case fly.py itself failed.
    FLIGHT_DIR=$("$PYTHON" -c "
import json,sys
try:
    print(json.load(open('$ACTIVE'))['dir'])
except Exception:
    sys.exit(1)
" 2>/dev/null)
    if [ -n "${FLIGHT_DIR:-}" ] && [ -d "$FLIGHT_DIR" ]; then
        echo "stop requested (low battery)" > "$FLIGHT_DIR/STOP"
        log "wrote STOP directly into $FLIGHT_DIR"
        sleep 4
    fi
fi

# Flush everything to the SD card. Without this, the final frames and the flight
# metadata can still be sitting in the page cache when power goes.
sync
log "filesystem synced - safe to power off"
exit 0

# ---------------------------------------------------------------------------
# TESTING (do this on the bench, before it matters)
#
#   1. Start a capture:        python scripts/fly.py --preset full_sun
#   2. Run the hook by hand:   ./scripts/wittypi_before_shutdown.sh
#   3. Confirm:                python scripts/fly.py status   -> not capturing
#                              tail flights/shutdown.log
#                              ls -l flights/<latest>/        -> no 0-byte JPEGs,
#                                                                capture_meta.json
#                                                                present
#   4. Then test for real: run a capture on battery until the Witty Pi cuts
#      power, and check the final flight folder is intact and processable.
# ---------------------------------------------------------------------------
