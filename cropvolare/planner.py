"""
Survey planning: turn a field polygon into a lawnmower scan pattern.

Pure math, no I/O. Given a polygon of (lat, lon) points, altitude, camera FOV,
and desired side overlap, produce parallel flight lines clipped to the polygon
(boustrophedon order - alternating direction, ready to fly), plus stats
(distance, time, frame count, forward-overlap check) and exporters for DJI
waypoint apps (KML, Litchi CSV).

Uses the same equirectangular local projection as field.build_grid - accurate
over a single sub-km field.
"""

import math

from .field import EARTH_RADIUS_M

# Camera Module 3 (IMX708) standard lens
DEFAULT_HFOV_DEG = 66.0   # across-track (image width)
DEFAULT_VFOV_DEG = 41.0   # along-track (image height)


# --------------------------------------------------------------------------
# projection helpers
# --------------------------------------------------------------------------

def _projection(points_latlon):
    """Local meters frame about the polygon centroid; returns (to_xy, to_latlon)."""
    lat0 = sum(p[0] for p in points_latlon) / len(points_latlon)
    lon0 = sum(p[1] for p in points_latlon) / len(points_latlon)
    m_lat = math.pi / 180.0 * EARTH_RADIUS_M
    m_lon = m_lat * math.cos(math.radians(lat0))

    def to_xy(lat, lon):
        return ((lon - lon0) * m_lon, (lat - lat0) * m_lat)

    def to_latlon(x, y):
        return (lat0 + y / m_lat, lon0 + x / m_lon)

    return to_xy, to_latlon


def _rotate(pt, angle_rad):
    x, y = pt
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return (x * c - y * s, x * s + y * c)


def _longest_edge_angle(xy):
    """Angle (radians) of the polygon's longest edge - the default line heading."""
    best_len, best_angle = -1.0, 0.0
    n = len(xy)
    for i in range(n):
        x1, y1 = xy[i]
        x2, y2 = xy[(i + 1) % n]
        length = math.hypot(x2 - x1, y2 - y1)
        if length > best_len:
            best_len = length
            best_angle = math.atan2(y2 - y1, x2 - x1)
    return best_angle


def _scanline_segments(xy, y):
    """x-intervals where the horizontal line at `y` lies inside the polygon."""
    xs = []
    n = len(xy)
    for i in range(n):
        x1, y1 = xy[i]
        x2, y2 = xy[(i + 1) % n]
        if y1 == y2:
            continue  # horizontal edge: endpoints handled by neighbours
        if (y1 <= y < y2) or (y2 <= y < y1):
            t = (y - y1) / (y2 - y1)
            xs.append(x1 + t * (x2 - x1))
    xs.sort()
    return [(xs[i], xs[i + 1]) for i in range(0, len(xs) - 1, 2)]


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------

def ground_footprint(altitude_m, hfov_deg=DEFAULT_HFOV_DEG):
    """Width of the ground strip one frame covers, in meters."""
    return 2.0 * altitude_m * math.tan(math.radians(hfov_deg) / 2.0)


def survey_lines(polygon_latlon, altitude_m, side_overlap=0.75,
                 hfov_deg=DEFAULT_HFOV_DEG, heading_deg=None):
    """Lawnmower lines over the polygon, in boustrophedon (fly-ready) order.

    polygon_latlon: >=3 (lat, lon) vertices. Returns a list of lines, each a
    list of (lat, lon) endpoints, alternating direction so consecutive lines
    connect end-to-start. Line spacing = footprint * (1 - side_overlap).
    heading_deg (compass-style, 0=N... not needed for most uses): when None,
    lines run parallel to the polygon's longest edge.
    """
    if len(polygon_latlon) < 3:
        raise ValueError("polygon needs at least 3 points")
    spacing = ground_footprint(altitude_m, hfov_deg) * (1.0 - side_overlap)
    if spacing <= 0:
        raise ValueError("side_overlap must be < 1.0")

    to_xy, to_latlon = _projection(polygon_latlon)
    xy = [to_xy(lat, lon) for lat, lon in polygon_latlon]

    if heading_deg is None:
        angle = _longest_edge_angle(xy)
    else:
        # compass heading -> math angle of the line direction
        angle = math.radians(90.0 - heading_deg)

    # rotate so lines run along +x, scan in y
    rot = [_rotate(p, -angle) for p in xy]
    ys = [p[1] for p in rot]
    y_min, y_max = min(ys), max(ys)

    lines = []
    flip = False
    y = y_min + spacing / 2.0
    while y <= y_max:
        for x_a, x_b in _scanline_segments(rot, y):
            seg = [(x_a, y), (x_b, y)]
            if flip:
                seg.reverse()
            latlon = [to_latlon(*_rotate(p, angle)) for p in seg]
            lines.append(latlon)
        flip = not flip
        y += spacing
    return lines


def waypoints(lines):
    """Flatten boustrophedon lines into an ordered waypoint list."""
    return [pt for line in lines for pt in line]


def plan_stats(lines, altitude_m, speed_mps=4.0, interval_s=2.0,
               vfov_deg=DEFAULT_VFOV_DEG):
    """Distance / time / frame estimates + forward-overlap sanity check."""
    pts = waypoints(lines)
    if len(pts) < 2:
        return {"distance_m": 0.0, "time_min": 0.0, "est_frames": 0,
                "forward_overlap": None, "overlap_ok": None}
    to_xy, _ = _projection(pts)
    xy = [to_xy(lat, lon) for lat, lon in pts]
    dist = sum(math.hypot(xy[i + 1][0] - xy[i][0], xy[i + 1][1] - xy[i][1])
               for i in range(len(xy) - 1))
    time_s = dist / speed_mps
    forward_fp = 2.0 * altitude_m * math.tan(math.radians(vfov_deg) / 2.0)
    forward_overlap = 1.0 - (speed_mps * interval_s) / forward_fp
    return {
        "distance_m": round(dist, 1),
        "time_min": round(time_s / 60.0, 1),
        "est_frames": int(time_s / interval_s),
        "forward_overlap": round(forward_overlap, 2),
        # ODM-grade stitching wants >=0.75 forward overlap later; 0.5 is the
        # floor for even per-image coverage
        "overlap_ok": forward_overlap >= 0.5,
    }


# --------------------------------------------------------------------------
# exporters
# --------------------------------------------------------------------------

def to_kml(lines, name="cropvolare survey"):
    """One KML LineString following the whole boustrophedon path."""
    coords = " ".join(f"{lon:.7f},{lat:.7f},0"
                      for lat, lon in waypoints(lines))
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<kml xmlns="http://www.opengis.net/kml/2.2">\n'
        f"  <Document><name>{name}</name>\n"
        "    <Placemark><name>survey path</name>\n"
        "      <LineString><tessellate>1</tessellate>\n"
        f"        <coordinates>{coords}</coordinates>\n"
        "      </LineString>\n"
        "    </Placemark>\n"
        "  </Document>\n"
        "</kml>\n"
    )


def to_litchi_csv(lines, altitude_m):
    """Litchi Mission Hub CSV (import via https://flylitchi.com/hub).

    Minimal column set; Litchi fills defaults for the rest. Gimbal pitch is
    pinned to -90 (straight down) to match the nadir camera.
    """
    rows = ["latitude,longitude,altitude(m),heading(deg),curvesize(m),"
            "rotationdir,gimbalmode,gimbalpitchangle"]
    for lat, lon in waypoints(lines):
        rows.append(f"{lat:.7f},{lon:.7f},{altitude_m},0,0,0,2,-90")
    return "\n".join(rows) + "\n"
