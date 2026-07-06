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

With the filter mounted, calibrate the NIR-bleed correction from one photo of a
neutral grey card (photography 18% grey card, in the light you'll fly in):

```bash
# on the Pi: photograph the grey card filling the frame
python scripts/capture_flight.py -o calib --count 1

# solve the leakage constant and save it into the config
python scripts/calibrate.py --input calib/frame_0000.jpg --write

# optional sanity check: a healthy plant should read ~0.4-0.6
python scripts/calibrate.py --input calib/frame_0000.jpg --plant plant.jpg
```

Every capture and analysis run reads the saved value from
`config/default.json` automatically. Re-calibrate if you change the filter.

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
# 40 frames, one every 2.5 s, tagging from a serial GPS
python scripts/capture_flight.py --outdir flights/today --count 40 --gps-port /dev/serial0

# no GPS connected? capture untagged, then tag afterwards
python scripts/capture_flight.py --outdir flights/today --count 40
python scripts/tag_gps.py --dir flights/today --lat 40.1 --lon -88.2
```

Then copy `flights/today/` to the laptop and run `process_flight.py` on it.

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
│   ├── fly.py              # Pi-side one-command field capture (start/status/stop)
│   ├── capture_flight.py   # Pi-side burst capture -> geotagged JPEGs
│   ├── capture_ndvi.py     # single-image cli
│   ├── ground_station.py   # laptop: pull flight off the Pi + analyze + open report
│   ├── process_flight.py   # whole-flight pipeline -> report
│   ├── tag_gps.py          # Pi-side GPS EXIF tagging
│   └── cropvolare-flight.service  # optional: capture at power-on (systemd)
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
