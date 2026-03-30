# Cropvolare

**Affordable NDVI crop health monitoring for smallholder farmers.**

Cropvolare uses a Raspberry Pi Zero 2W and an Arducam NoIR V3 camera with a Wratten 25A red filter to compute NDVI (Normalized Difference Vegetation Index) from a single camera. It provides per-pixel vegetation health maps and zone-level stress classification — all on-device, no cloud required.

---

## How It Works

The NoIR camera (no infrared filter) paired with a Wratten 25A longpass filter separates light into two useful bands:

| Bayer Channel | What It Captures               |
|---------------|--------------------------------|
| Blue          | Near-infrared (NIR), 700-1000nm |
| Red           | Visible red, 580-700nm          |

NDVI is then computed per-pixel:

```
NDVI = (NIR - Red) / (NIR + Red)
```

Healthy vegetation reflects strongly in NIR and absorbs red light, producing high NDVI values (0.5 - 1.0). Stressed or bare areas produce low values (< 0.3).

---

## Hardware

| Component | Model | Purpose |
|-----------|-------|---------|
| Computer | Raspberry Pi Zero 2W | On-device NDVI computation |
| Camera | Arducam NoIR V3 (IMX708) | NIR-sensitive image capture |
| Filter | Wratten 25A red gel | Channel separation for NDVI |
| Storage | 32GB MicroSD | OS + data |

---

## Project Structure

```
cropvolare/
├── cropvolare/
│   ├── __init__.py
│   └── ndvi.py              # NDVI capture, compute, calibrate, visualize
├── scripts/
│   └── capture_ndvi.py      # CLI script to capture and compute NDVI
├── tests/
│   └── test_ndvi.py         # Unit tests (no hardware needed)
├── config/
│   └── default.json         # Default configuration
├── docs/                    # Documentation (coming soon)
├── requirements.txt         # Python dependencies
├── requirements-pi.txt      # Raspberry Pi dependencies
├── LICENSE
└── README.md
```

---

## Setup

### On Raspberry Pi

```bash
# Install system dependencies
sudo apt update
sudo apt install python3-picamera2 python3-opencv python3-numpy

# Clone the repo
git clone https://github.com/DinoMighty25/cropvolare.git
cd cropvolare
```

### On Desktop (for development/testing)

```bash
git clone https://github.com/DinoMighty25/cropvolare.git
cd cropvolare
pip install -r requirements.txt
```

---

## Usage

### Capture and compute NDVI (on Raspberry Pi)

```bash
python scripts/capture_ndvi.py
python scripts/capture_ndvi.py --output my_field.png --print-zones
```

### Use as a library

```python
from cropvolare.ndvi import (
    create_camera,
    capture_image,
    compute_ndvi_from_image,
    classify_zones,
    save_ndvi_image,
)

cam = create_camera()
image = capture_image(cam)
ndvi = compute_ndvi_from_image(image)

zones = classify_zones(ndvi, block_size=64)
stressed = [z for z in zones if z["status"] == "stressed"]
print(f"Stressed zones: {len(stressed)}")

save_ndvi_image(ndvi, "output/ndvi_map.png")
```

### Run tests

```bash
pytest tests/
```

---

## NDVI Value Reference

| NDVI Range | Interpretation |
|------------|----------------|
| 0.6 - 1.0 | Dense, healthy vegetation |
| 0.3 - 0.6 | Moderate vegetation / early stress |
| 0.1 - 0.3 | Sparse vegetation / significant stress |
| -0.1 - 0.1 | Bare soil |
| -1.0 - -0.1 | Water / non-vegetation |

---

## Roadmap

- [x] NDVI computation pipeline
- [x] Hardware reference-target calibration
- [x] Zone-level stress classification
- [ ] Edge-AI disease detection (MobileNetV2)
- [ ] LoRa connectivity
- [ ] SMS/WhatsApp farmer alerts
- [ ] Solar power management
- [ ] Field deployment tools

---

## License

MIT License. See [LICENSE](LICENSE) for details.
