"""GpsReader bookkeeping tests (no serial hardware - fixes injected)."""

from cropvolare.gpsread import GpsReader


def _fix(lat, lon, alt=50.0, t="2026-07-06T20:00:00"):
    return {"time": t, "lat": lat, "lon": lon, "alt": alt}


def test_latest_none_until_fix():
    r = GpsReader()
    assert r.latest() is None
    r._accept(_fix(40.0, -88.0))
    assert r.latest()["lat"] == 40.0


def test_track_and_backfill():
    r = GpsReader()
    for i in range(5):
        r._accept(_fix(40.0 + i * 1e-5, -88.0))
    assert len(r.track()) == 5
    # a client that saw the first 3 asks for the rest (WiFi-drop backfill)
    assert len(r.track(since=3)) == 2


def test_track_csv_written(tmp_path):
    path = str(tmp_path / "track.csv")
    r = GpsReader(track_path=path)
    r._accept(_fix(40.0, -88.0, alt=61.5))
    r._accept(_fix(40.00001, -88.00001, alt=None))
    lines = open(path).read().strip().splitlines()
    assert lines[0] == "time,lat,lon,alt"
    assert lines[1].endswith(",40.0,-88.0,61.5")
    assert lines[2].endswith(",40.00001,-88.00001,")  # alt empty, not 'None'
