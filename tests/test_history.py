"""Flight-history store + change-detection tests."""

from cropvolare import history


def _result(date, mean, patches=None, scope="field", flight_id=None):
    return {
        "flight_id": flight_id or date, "date": date, "scope": scope,
        "area_ha": 1.0, "n_frames": 50,
        "distribution": {"mean": mean, "pct_healthy": 60.0,
                         "pct_stressed": 10.0, "pct_severe": 5.0},
        "patches": patches or [],
    }


def test_record_and_load_roundtrip(tmp_path):
    d = str(tmp_path)
    assert history.record_flight(None, _result("2026-07-01", 0.5),
                                 history_dir=d) is None  # no field = no-op
    history.record_flight("yard", _result("2026-07-01", 0.5), history_dir=d)
    history.record_flight("yard", _result("2026-07-05", 0.55), history_dir=d)
    rows = history.load_history("yard", history_dir=d)
    assert [r["date"] for r in rows] == ["2026-07-01", "2026-07-05"]
    assert rows[1]["mean_ndvi"] == 0.55


def test_previous_excludes_current_flight(tmp_path):
    d = str(tmp_path)
    history.record_flight("yard", _result("2026-07-01", 0.5), history_dir=d)
    history.record_flight("yard", _result("2026-07-05", 0.55), history_dir=d)
    prev = history.previous("yard", exclude_flight="2026-07-05", history_dir=d)
    assert prev["date"] == "2026-07-01"


def test_reprocessing_same_flight_dedupes(tmp_path):
    d = str(tmp_path)
    history.record_flight("yard", _result("2026-07-01", 0.50, flight_id="A"),
                          history_dir=d)
    history.record_flight("yard", _result("2026-07-01", 0.55, flight_id="A"),
                          history_dir=d)   # re-run of the same flight
    rows = history.load_history("yard", history_dir=d)
    assert len(rows) == 1 and rows[0]["mean_ndvi"] == 0.55


def test_previous_none_for_first_flight(tmp_path):
    assert history.previous("new", history_dir=str(tmp_path)) is None


def test_trend_direction():
    prior = history.record_from_result(_result("2026-07-01", 0.40))
    up = history.compare(_result("2026-07-05", 0.50), prior)
    assert up["overall"] == "improving" and up["mean_ndvi_delta"] == 0.10
    down = history.compare(_result("2026-07-05", 0.30), prior)
    assert down["overall"] == "declining"
    flat = history.compare(_result("2026-07-05", 0.41), prior)
    assert flat["overall"] == "stable"


def test_compare_none_prior():
    assert history.compare(_result("2026-07-05", 0.5), None) is None


def test_patch_matching_new_worse_improved_resolved():
    # prior: two patches A (worsens) and B (resolves)
    A = {"lat": 40.0000, "lon": -88.0000, "area_ha": 0.1, "mean_ndvi": 0.25}
    B = {"lat": 40.0100, "lon": -88.0000, "area_ha": 0.1, "mean_ndvi": 0.25}
    prior = history.record_from_result(_result("2026-07-01", 0.5, [A, B]))

    # current: A worse (same spot, lower NDVI), B gone, C new far away
    A2 = {"lat": 40.00002, "lon": -88.00002, "area_ha": 0.12, "mean_ndvi": 0.10,
          "severity": 1}
    C = {"lat": 40.0500, "lon": -88.0500, "area_ha": 0.1, "mean_ndvi": 0.2,
         "severity": 1}
    cur = _result("2026-07-05", 0.45, [A2, C])
    trend = history.compare(cur, prior)

    assert trend["spatial"] is True
    assert len(trend["worsened"]) == 1
    assert abs(trend["worsened"][0]["prev_ndvi"] - 0.25) < 1e-9
    assert len(trend["new"]) == 1 and trend["new"][0]["lat"] == 40.05
    assert len(trend["resolved"]) == 1 and trend["resolved"][0]["lat"] == 40.01


def test_patch_matching_improved():
    A = {"lat": 40.0, "lon": -88.0, "area_ha": 0.1, "mean_ndvi": 0.15}
    prior = history.record_from_result(_result("2026-07-01", 0.4, [A]))
    A2 = {"lat": 40.00001, "lon": -88.0, "area_ha": 0.05, "mean_ndvi": 0.28,
          "severity": 1}
    trend = history.compare(_result("2026-07-05", 0.5, [A2]), prior)
    assert len(trend["improved"]) == 1
    assert trend["worsened"] == [] and trend["new"] == []
