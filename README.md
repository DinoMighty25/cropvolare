# Cropvolare

NDVI crop health monitoring on a Raspberry Pi. Uses an Arducam NoIR V3 camera with a Wratten 25A red filter to get vegetation health maps from a single camera — no expensive multispectral sensors needed.

## How it works

The NoIR camera has no IR-cut filter, so it picks up near-infrared light. Slap a Wratten 25A (red longpass) filter on it and the Bayer channels split nicely:

- **Blue channel** → mostly NIR (700-1000nm), since visible blue is blocked by the filter
- **Red channel** → visible red (580-700nm)

Then NDVI is just `(NIR - Red) / (NIR + Red)`. Healthy plants reflect a ton of NIR and absorb red, so they score high (~0.5-1.0). Stressed or bare ground scores low (<0.3).

## Hardware

- Raspberry Pi Zero 2W
- Arducam NoIR V3 (IMX708, no IR filter)
- Wratten 25A red gel filter
- 32GB MicroSD

## Setup

**Raspberry Pi:**
```bash
sudo apt update && sudo apt install python3-picamera2 python3-opencv python3-numpy
git clone https://github.com/DinoMighty25/cropvolare.git
cd cropvolare
```

**Desktop (dev/testing only, no camera capture):**
```bash
git clone https://github.com/DinoMighty25/cropvolare.git
cd cropvolare
pip install -r requirements.txt
```

## Usage

Capture + compute on the Pi:
```bash
python scripts/capture_ndvi.py
python scripts/capture_ndvi.py -o my_field.png --print-zones
```

Or use it in your own code:
```python
from cropvolare.ndvi import create_camera, capture_image, compute_ndvi_from_image, classify_zones

cam = create_camera()
image = capture_image(cam)
ndvi = compute_ndvi_from_image(image)

zones = classify_zones(ndvi)
for z in zones:
    if z["status"] == "stressed":
        print(f"zone ({z['zone_row']},{z['zone_col']}): {z['mean_ndvi']:.2f}")
```

## NDVI values

| Range | Meaning |
|-------|---------|
| 0.6 – 1.0 | Dense healthy vegetation |
| 0.3 – 0.6 | Moderate / early stress |
| 0.1 – 0.3 | Sparse / significant stress |
| -0.1 – 0.1 | Bare soil |
| < -0.1 | Water |

## Project structure

```
cropvolare/
├── cropvolare/        # main package
│   ├── __init__.py
│   └── ndvi.py        # capture, ndvi math, calibration, visualization
├── scripts/
│   └── capture_ndvi.py
├── tests/
│   └── test_ndvi.py
├── config/
│   └── default.json
├── docs/
├── requirements.txt
└── requirements-pi.txt
```

## Tests

```bash
pytest tests/
```

## Roadmap

- [x] NDVI capture + computation
- [x] Reference target calibration
- [x] Zone health classification
- [ ] Disease detection (TFLite on-device)
- [ ] LoRa connectivity
- [ ] Farmer alerts (SMS/WhatsApp)
- [ ] Solar power management

## License

MIT
