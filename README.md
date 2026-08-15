# Cropvolare

Affordable NDVI crop monitoring using a Raspberry Pi and a NoIR camera.

## What's NDVI?

NDVI (Normalized Difference Vegetation Index) measures how healthy plants are based on how they reflect light. Healthy plants reflect a lot of near-infrared and absorb red light. Stressed or dead plants don't.

The formula: `(NIR - Red) / (NIR + Red)`

| NDVI | What it means |
|------|---------------|
| 0.6 – 1.0 | Healthy |
| 0.3 – 0.6 | Some stress |
| 0.1 – 0.3 | Not great |
| < 0.1 | Bare soil / water |

## Hardware

- Raspberry Pi Zero 2W
- Arducam NoIR V3 (IMX708)
- MicroSD card

## Getting started

On the Pi:
```bash
sudo apt update && sudo apt install python3-picamera2 python3-opencv python3-numpy
git clone https://github.com/DinoMighty25/cropvolare.git
cd cropvolare
```

Just want to mess with the code on your computer:
```bash
git clone https://github.com/DinoMighty25/cropvolare.git
cd cropvolare
pip install -r requirements.txt
```

## Usage

### Single image (bench / handheld)

```bash
python scripts/capture_ndvi.py                       # capture on the Pi
python scripts/capture_ndvi.py --input photo.jpg     # process a saved photo
python scripts/capture_ndvi.py -o my_field.png --print-zones --vari
```

Or in Python:
```python
from cropvolare.ndvi import create_camera, capture_image, compute_ndvi_from_image, classify_zones

cam = create_camera()
image = capture_image(cam)
ndvi = compute_ndvi_from_image(image)

for z in classify_zones(ndvi):
    if z["status"] == "stressed":
        print(f"zone ({z['zone_row']},{z['zone_col']}): {z['mean_ndvi']:.2f}")
```

### Whole flight → farmer report

Process a folder of geotagged drone photos into a field-level report:

```bash
python scripts/process_flight.py --input flights/2026-06-21/ --outdir output/2026-06-21/
```

This produces, in `--outdir`:

| File | What it is |
|------|------------|
| `field.geojson` | Per-photo NDVI + GPS (durable data; also the stitching hand-off) |
| `heatmap.png` | Colorized field NDVI overlay |
| `report.pdf` | One-page farmer report: map, % healthy/stressed/severe, ranked problem areas |
| `map.html` | Standalone interactive web map (opens in any browser, no server) |

Photos need GPS in their EXIF. On the Pi, tag captured JPEGs with `scripts/tag_gps.py`
(reads a serial GPS via pynmea2). Processing runs on a laptop after the flight.

### Filter & calibration (once, after mounting the red filter)

The pipeline expects a **deep-red filter** (Wratten 25 / Tiffen #25 / Rosco #19
or #26 / ~610-650nm longpass glass) on the NoIR camera: it blocks blue-green so
the blue channel reads NIR and the red channel reads visible red. Don't use a
720nm+ "IR-only" filter (kills the red channel) or a blue "superblue" filter
(opposite channel mapping).

Two constants describe the rig, and they are **different physical effects**:

| constant | what it is | how it applies |
| --- | --- | --- |
| `leakage_k` | NIR bleeding through the filter into the red pixel | subtracted |
| `red_gain` | red pixel's sensitivity to visible red, relative to the blue pixel's to NIR | divided |

Solving only one of them (the old behaviour) conflates the two and drives the
red channel to zero over vegetation, which pins healthy canopy at NDVI = 1.0.
On real flight frames that silently flattened **36% of vegetated pixels** into a
single value — a field that looks uniformly healthy no matter what is growing.

**Calibrate from crop photos you already have (no target needed).** Any frame
containing both canopy and bare ground gives two equations, which solves both
constants at once:

```
python scripts/calibrate.py --two-point flights/2026-07-30/ --write
```

It averages across frames, reports the frame-to-frame spread, and cross-checks
the solved `leakage_k` against the green/blue channel ratio — two independent
routes to the same number. If they disagree, it says so.

**Or calibrate from a neutral target.** What matters is spectral neutrality
between red and NIR, *not* 18% reflectance — the constants come from a ratio, so
any brightness works provided neither channel clips:

```
python scripts/calibrate.py --input calib/frame_0000.jpg --anchor ptfe --write
```

| `--anchor` | target | notes |
| --- | --- | --- |
| `ptfe` | plumber's PTFE tape over card | same material as Spectralon standards |
| `paper` | white printer paper, open shade | brighteners emit ~440nm, filter blocks it |
| `concrete` | dry pavement | already in most farm scenes |
| `asphalt` | dry road | not freshly laid |

Clipped targets are rejected automatically — a blown channel corrupts the ratio.

**Diagnose an existing flight** (saturation + whether the channel assumptions
hold on your rig) without any calibration data at all:

```
python scripts/diagnose_ndvi.py --dir flights/2026-07-30/
```

**Flat-field (removes the NDVI "bullseye").** The lens shades the NIR and red
channels unequally, which paints a false radial health gradient on every frame.
Fix it once: photograph a plain white sheet filling the frame in even shade
(~20 frames), then build the gain map:

```bash
python scripts/capture_flight.py -o calib_flat --count 20 --interval 0.5
python scripts/calibrate.py --flatfield-dir calib_flat --write
```

The map is saved to `calibration/gain.npy` and applied automatically by every
analysis run (disable with `--no-flatfield`). Rebuild it if the filter or
camera changes.

**Blurry-frame filter.** Focus is locked at infinity, so frames captured while
grounded or very low are featureless blur. Every frame gets a `sharpness`
score in the report; pass `--min-sharpness 15` to `process_flight.py` to drop
them (default 0 = keep everything, so uniform crop seen from high altitude is
never silently discarded).

### Ground control station (phone web app)

The Pi serves a phone-friendly GCS on port 8080. Enable it once:

```bash
pip install --break-system-packages flask
sudo cp scripts/cropvolare-gcs.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now cropvolare-gcs
```

Then from your phone (hotspot): **`http://<pi-ip>:8080`**

- **Dashboard** — live capture status, frame count, disk, camera + NDVI
  preview, big START/STOP buttons. No SSH at the field.
- **Field planner** (`/planner`) — tap the map to outline a field, set
  altitude/overlap, and get the lawnmower survey pattern with distance/time/
  frame estimates and a too-fast overlap warning. Fields are saved and reused.
  Export the pattern as **QGC WPL `.waypoints`** for Mission Planner / ArduPilot
(the Iris path — *Flight Plan → Load WP File*, review, then *Write WPs*), or as
**KML** / **Litchi CSV** for DJI waypoint apps. The WPL export pins ground speed
with `DO_CHANGE_SPEED`, which is what keeps forward overlap at the planned
value.
- **Live coverage** — once a serial GPS is wired (`gcs.gps_port` in the config
  or `--gps-port`), the planner shows the drone's live breadcrumb over your
  polygon while you fly. Phone-Pi WiFi drops beyond ~50-100 m; the trail
  backfills on reconnect, and `track.csv` in the flight folder is always
  complete (capture never depends on the phone).

No authentication — hotspot/LAN use only.

**No hotspot needed (field AP mode).** With the fallback installed, the Pi
broadcasts its own WiFi when it can't find a known network ~75s after boot —
i.e. automatically, exactly when you're in a field. Connect the phone to WiFi
**`CropVolare`** and open **`http://10.42.0.1:8080`**. At home nothing changes
(the Pi joins home WiFi first; the AP profile never autoconnects on its own).
One-time setup on the Pi:

```bash
sudo nmcli connection add type wifi ifname wlan0 con-name cropvolare-ap \
    autoconnect no ssid CropVolare
sudo nmcli connection modify cropvolare-ap 802-11-wireless.mode ap \
    802-11-wireless.band bg ipv4.method shared \
    wifi-sec.key-mgmt wpa-psk wifi-sec.psk "cropvolare"
sudo cp scripts/cropvolare-ap.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable cropvolare-ap
```

Phones keep using cellular for internet while connected to the Pi's AP, so
the planner's satellite tiles still load if you have signal (and the pages
themselves work fully offline).

### Field workflow (fly day)

Everything you do at the field, in order:

```bash
# 1. power the Pi, SSH in from your phone or laptop:
ssh dinomighty@<pi-ip>
cd cropvolare

# 2. start capture - one command, no arguments:
python scripts/fly.py
#    runs preflight checks (camera, disk), starts capturing into a new
#    timestamped flights/ folder, and waits until the FIRST FRAME is saved
#    before telling you it's safe to fly. SSH dropping does NOT stop it.

# 3. fly slow, overlapping passes over the field.

# 4. land, SSH back in (any session):
python scripts/fly.py status    # optional: frame count + last-frame age
python scripts/fly.py stop      # finish cleanly, prints the folder + next step

# 5. on the laptop, pull + analyze + open the report:
python scripts/ground_station.py --host dinomighty@<pi-ip> \
    --remote cropvolare/flights/<dir> --input flights/<dir> --open
```

Optional zero-SSH mode: install `scripts/cropvolare-flight.service` (see the
comments inside it) and the Pi starts capturing at power-on — plug in, fly,
then `fly.py stop` after landing.

### Ground station (laptop, one command)

`scripts/ground_station.py` pulls a flight folder off the Pi (optional) and runs
the analysis on the laptop, where there's CPU/RAM to spare — the Pi only captures.

```bash
# pull from the Pi over SSH, analyze, and open the report:
python scripts/ground_station.py --host dinomighty@<pi-ip> \
    --remote flights/today --input flights/today --outdir output/today --open

# analyze a folder you already copied (SD card / manual scp):
python scripts/ground_station.py --input flights/today --open
```

## Running on the Raspberry Pi

Tested on a Pi Zero 2 W + Camera Module 3 (IMX708), Raspberry Pi OS Bookworm.

Install dependencies (use apt for the heavy libs — building them with pip on a
Zero 2 W is slow):

```bash
sudo apt update
sudo apt install -y python3-picamera2 python3-opencv python3-numpy
pip install --break-system-packages piexif pynmea2 pyserial
```

Confirm the camera is detected, then capture:

```bash
rpicam-hello --list-cameras        # should list imx708
python scripts/capture_ndvi.py -o ndvi.png   # single shot + NDVI
```

Capture a flight (burst of geotagged JPEGs for `process_flight.py`):

```bash
# 40 frames, one every 2 s, tagging from a serial GPS
#
# INTERVAL MATTERS: the camera is not triggered by the flight controller, so
# forward overlap is just interval x ground speed against the along-track
# footprint. At 30 m AGL that footprint is only ~22 m, so a 5 s interval gives
# 11% overlap at 4 m/s - nothing stitches. Required interval for 75% overlap:
#
#   altitude |  2 m/s  3 m/s  4 m/s
#      30 m  |  2.8s   1.9s   1.4s
#      60 m  |  5.6s   3.7s   2.8s
#
# planner.overlap_warnings(alt, speed, interval) checks this before you fly.
python scripts/capture_flight.py --outdir flights/today --count 40 --gps-port /dev/serial0

# no GPS connected? capture untagged, then tag afterwards
python scripts/capture_flight.py --outdir flights/today --count 40
python scripts/tag_gps.py --dir flights/today --lat 40.1 --lon -88.2
```

Then copy `flights/today/` to the laptop and run `process_flight.py` on it.

## Crop health analysis (the farmer report)

`process_flight.py` runs an offline analysis engine (`cropvolare/analyze.py`) over
the NDVI and writes a farmer-first **agronomy report**: an overall health verdict,
the ranked problem areas with a first thing to check, a trend vs the last flight, and
honest methodology caveats. No cloud, no API, fully deterministic.

```bash
# analyze a flight and key it to a field for trend tracking:
python scripts/process_flight.py --input flights/today --outdir output/today --field north40
```

- **Problem areas** — with GPS, contiguous low-NDVI cells are clustered into patches
  (location, size in ha, severity, suggested cause). Without GPS, the report lists and
  shows the lowest-NDVI frames instead.
- **Trends** — `--field NAME` appends each flight to `history/<name>.jsonl`; the next
  flight's report compares mean NDVI and (with GPS) which patches are new / worsening /
  improving / resolved. Re-processing the same flight folder updates its record rather
  than duplicating.
- `--basic` emits the old plain gallery/field report instead (debugging).

Findings are decision-support to guide ground inspection, never a diagnosis —
single-camera NoIR NDVI is relative and approximate, and the report states its
calibration state on every page.

## Project structure

```
cropvolare/
├── cropvolare/             # main package
│   ├── ndvi.py             # NDVI math + camera (single image)
│   ├── geo.py              # EXIF GPS read/write
│   ├── batch.py            # folder of photos -> GeoJSON NDVI records
│   ├── field.py            # bin photos into a field grid, rank problem areas
│   ├── fieldmap.py         # field heatmap PNG
│   ├── report.py           # one-page PDF report
│   └── webmap.py           # standalone interactive web map
├── scripts/
│   ├── gcs.py              # Pi-side ground control station (phone web app)
│   ├── fly.py              # Pi-side one-command field capture (start/status/stop)
│   ├── capture_flight.py   # Pi-side burst capture -> geotagged JPEGs
│   ├── capture_ndvi.py     # single-image cli
│   ├── ground_station.py   # laptop: pull flight off the Pi + analyze + open report
│   ├── process_flight.py   # whole-flight pipeline -> report
│   ├── tag_gps.py          # Pi-side GPS EXIF tagging
│   ├── cropvolare-flight.service  # optional: capture at power-on (systemd)
│   └── cropvolare-gcs.service     # optional: GCS web app at power-on (systemd)
├── tests/
├── config/
│   └── default.json
└── docs/
```

## Running tests

Install the dev/test deps once, then run the suite (Pi-only hardware tests are
skipped by default):

```bash
pip install -r requirements-dev.txt
pytest                       # full laptop suite (skips hardware)
pytest tests/test_ndvi.py    # fast inner loop: pure NDVI math
pytest -m e2e                # end-to-end CLI tests only
pytest --cov=cropvolare --cov-report=term-missing   # coverage
```

Test layers: `test_ndvi.py` (pure math) → `test_geo/batch/field.py` (units) →
`test_outputs.py` (PDF/HTML/PNG generation) → `test_cli.py` (end-to-end scripts)
→ `test_edge_cases.py` (grid corners, no-GPS pipeline, dependency guards).

## What's next

- [ ] Disease detection (on-device ML)
- [ ] LoRa for field connectivity
- [ ] SMS/WhatsApp alerts for farmers
- [ ] Solar power setup

## License

MIT
