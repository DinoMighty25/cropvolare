"""
EXIF GPS handling for geotagged drone photos.

Reading happens on the ground laptop (turn a folder of photos into NDVI records
with coordinates). Writing is used Pi-side by scripts/tag_gps.py to stamp a GPS
fix onto a captured JPEG. Both directions use piexif - pure-Python, no binaries,
Windows-friendly.

All coordinates are decimal degrees: latitude +N/-S, longitude +E/-W.
"""

from fractions import Fraction

try:
    import piexif
except ImportError:
    piexif = None


def _require_piexif():
    if piexif is None:
        raise RuntimeError("piexif not installed - run: pip install piexif")


def _to_rational(number):
    """Turn a non-negative float into a piexif (numerator, denominator) tuple."""
    frac = Fraction(number).limit_denominator(1000000)
    return (frac.numerator, frac.denominator)


def _decimal_to_dms(value):
    """Decimal degrees -> ((d,1),(m,1),(s,100)) EXIF rational triple.

    Sign is dropped here; the caller stores N/S or E/W as a separate ref.
    """
    value = abs(value)
    degrees = int(value)
    minutes_full = (value - degrees) * 60
    minutes = int(minutes_full)
    seconds = round((minutes_full - minutes) * 60, 5)
    return (
        (degrees, 1),
        (minutes, 1),
        _to_rational(seconds),
    )


def _dms_to_decimal(dms, ref):
    """((d,den),(m,den),(s,den)) + ref ('N'/'S'/'E'/'W') -> signed decimal degrees."""
    def _val(rational):
        num, den = rational
        return num / den if den else 0.0

    degrees = _val(dms[0])
    minutes = _val(dms[1])
    seconds = _val(dms[2])
    decimal = degrees + minutes / 60.0 + seconds / 3600.0

    if isinstance(ref, bytes):
        ref = ref.decode("ascii", "ignore")
    if ref in ("S", "W"):
        decimal = -decimal
    return decimal


def write_gps(path, lat, lon, alt=None, timestamp=None):
    """Insert/overwrite the GPS IFD on an existing JPEG, in place.

    timestamp, if given, is stored as a human-readable EXIF GPSDateStamp-style
    ASCII string (we keep it simple - the iso string the caller passes in).
    """
    _require_piexif()
    try:
        exif = piexif.load(path)
    except Exception:
        exif = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}

    gps = {
        piexif.GPSIFD.GPSLatitudeRef: "N" if lat >= 0 else "S",
        piexif.GPSIFD.GPSLatitude: _decimal_to_dms(lat),
        piexif.GPSIFD.GPSLongitudeRef: "E" if lon >= 0 else "W",
        piexif.GPSIFD.GPSLongitude: _decimal_to_dms(lon),
    }
    if alt is not None:
        gps[piexif.GPSIFD.GPSAltitudeRef] = 0 if alt >= 0 else 1
        gps[piexif.GPSIFD.GPSAltitude] = _to_rational(abs(alt))
    if timestamp is not None:
        gps[piexif.GPSIFD.GPSDateStamp] = str(timestamp)

    exif["GPS"] = gps
    piexif.insert(piexif.dump(exif), path)


def read_gps(path):
    """Return {'lat', 'lon', 'alt', 'timestamp'} in decimal degrees, or None.

    None means the file has no usable GPS IFD - callers should flag the photo as
    untagged rather than crash.
    """
    _require_piexif()
    try:
        exif = piexif.load(path)
    except Exception:
        return None

    gps = exif.get("GPS") or {}
    lat_dms = gps.get(piexif.GPSIFD.GPSLatitude)
    lon_dms = gps.get(piexif.GPSIFD.GPSLongitude)
    if not lat_dms or not lon_dms:
        return None

    lat = _dms_to_decimal(lat_dms, gps.get(piexif.GPSIFD.GPSLatitudeRef, "N"))
    lon = _dms_to_decimal(lon_dms, gps.get(piexif.GPSIFD.GPSLongitudeRef, "E"))

    alt = None
    alt_raw = gps.get(piexif.GPSIFD.GPSAltitude)
    if alt_raw:
        num, den = alt_raw
        alt = num / den if den else None
        if alt is not None and gps.get(piexif.GPSIFD.GPSAltitudeRef, 0) == 1:
            alt = -alt

    timestamp = gps.get(piexif.GPSIFD.GPSDateStamp)
    if isinstance(timestamp, bytes):
        timestamp = timestamp.decode("ascii", "ignore")

    return {"lat": lat, "lon": lon, "alt": alt, "timestamp": timestamp}


def has_gps(path):
    """Cheap predicate: does this file carry usable GPS coordinates?"""
    return read_gps(path) is not None
