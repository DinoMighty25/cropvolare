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
│   ├── capture_ndvi.py     # single-image cli
│   ├── process_flight.py   # whole-flight pipeline -> report
│   └── tag_gps.py          # Pi-side GPS EXIF tagging
├── tests/
├── config/
│   └── default.json
└── docs/
```

## Running tests

```bash
pytest tests/
```

## What's next

- [ ] Disease detection (on-device ML)
- [ ] LoRa for field connectivity
- [ ] SMS/WhatsApp alerts for farmers
- [ ] Solar power setup

## License

MIT
