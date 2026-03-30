# Cropvolare

Cheap NDVI crop monitoring using a Raspberry Pi and a NoIR camera. Point it at your field, get a vegetation health map. That's it for now.

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

```bash
python scripts/capture_ndvi.py
python scripts/capture_ndvi.py -o my_field.png --print-zones
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

## Project structure

```
cropvolare/
├── cropvolare/          # main package
│   └── ndvi.py
├── scripts/
│   └── capture_ndvi.py  # cli tool
├── tests/
│   └── test_ndvi.py
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
