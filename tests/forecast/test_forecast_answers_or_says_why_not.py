"""`/v1/forecast`: a range (p10 <= p50 <= p90) with its backtest, or an abstention with a reason; never a guess.
Trained on SYNTHETIC data (tests/forecast/conftest.py); no figure here describes a real market."""

import json

from fastapi.testclient import TestClient

from src.forecast import service


def _client(monkeypatch, models_dir):
    from src.api.main import app
    from src.config.settings import settings
    monkeypatch.setattr(settings, "models_dir", models_dir)
    service.reset_cache()
    return TestClient(app)


def test_the_backtest_is_against_the_naive_rule_and_recorded(trained):
    _models, meta = trained
    o = meta["backtest"]["overall"]
    assert {"mae_p50", "mae_persistence", "skill_vs_persistence", "coverage_p10_p90", "n"} <= set(o)
    assert meta["features_never_used"][0] == "min_price"
    assert meta["data_source"].startswith("SYNTHETIC")


def test_a_usable_group_gets_an_ordered_range(monkeypatch, trained):
    models, meta = trained
    usable = [g for g in meta["backtest"]["groups"] if g["usable"]]
    c = _client(monkeypatch, models)
    for g in usable[:2]:
        market = meta["markets"][f"{g['commodity']}|{g['state']}"][0]
        r = c.post("/v1/forecast", json={"commodity": g["commodity"], "state": g["state"], "market": market,
                                         "as_of": meta["trained_through"], "horizon_days": g["horizon_days"]}).json()
        assert r["abstain"] is False, r
        assert r["p10"] <= r["p50"] <= r["p90"]
        assert r["backtest"]["skill_vs_persistence"] > 0


def test_a_group_without_skill_abstains_and_still_shows_its_record(monkeypatch, trained):
    models, meta = trained
    meta_path = models / meta["model_file"].replace(".joblib", ".json")
    saved = meta_path.read_text()
    try:
        m = json.loads(saved)
        for g in m["backtest"]["groups"]:
            g["usable"] = False
        meta_path.write_text(json.dumps(m))
        c = _client(monkeypatch, models)
        g = m["backtest"]["groups"][0]
        r = c.post("/v1/forecast", json={"commodity": g["commodity"], "state": g["state"],
                                         "market": m["markets"][f"{g['commodity']}|{g['state']}"][0],
                                         "as_of": m["trained_through"], "horizon_days": g["horizon_days"]}).json()
        assert r["abstain"] is True and r["reason_code"] == "no_skill" and r["p50"] is None
        assert r["backtest"]["n"] == g["n"]
    finally:
        meta_path.write_text(saved)
        service.reset_cache()


def test_an_unknown_crop_mandi_or_horizon_is_refused_not_guessed(monkeypatch, trained):
    models, meta = trained
    c = _client(monkeypatch, models)
    ask = {"commodity": "Onion", "state": "Maharashtra", "market": "Lasalgaon", "as_of": meta["trained_through"],
           "horizon_days": 7}
    for change, code in (({"commodity": "Saffron"}, "not_trained_for"), ({"market": "Nowhere APMC"}, "not_trained_for"),
                         ({"horizon_days": 11}, "horizon_not_trained"), ({"as_of": "2031-01-01"}, "as_of_outside_history")):
        r = c.post("/v1/forecast", json={**ask, **change}).json()
        assert r["abstain"] is True and r["reason_code"] == code, (change, r)


def test_a_day_inside_the_training_data_is_refused_as_in_sample(monkeypatch, trained):
    """The served models were fitted on every sample up to `trained_through`, so a range for a day whose outcome is in
    that data would describe a known price and would look far better than a forecast."""
    import pandas as pd
    models, meta = trained
    early = (pd.Timestamp(meta["trained_through"]) - pd.Timedelta(days=30)).date().isoformat()
    r = _client(monkeypatch, models).post("/v1/forecast", json={
        "commodity": "Onion", "state": "Maharashtra", "market": "Lasalgaon", "as_of": early, "horizon_days": 7}).json()
    assert r["abstain"] is True and r["reason_code"] == "in_sample" and r["p50"] is None, r


def test_stale_price_age_uses_as_of_for_field_and_reason(monkeypatch, trained):
    import pandas as pd

    models, meta = trained
    as_of = (pd.Timestamp(meta["trained_through"]) + pd.Timedelta(days=31)).date().isoformat()
    r = _client(monkeypatch, models).post("/v1/forecast", json={
        "commodity": "Onion", "state": "Maharashtra", "market": "Lasalgaon",
        "as_of": as_of, "horizon_days": 7,
    }).json()

    assert r["abstain"] is True and r["reason_code"] == "STALE_PRICE", r
    assert r["days_since_last_price"] == 31
    assert r["reason"] == "The last price is 31 days old."


def test_the_cli_refuses_a_crop_that_is_not_in_the_data(tmp_path, synthetic):
    """`--crops` names are exact, as the data spells them (Agmarknet writes "Soyabean"); a typo is refused, not
    silently trained on nothing."""
    import pytest

    from src.forecast.train import main
    csv = tmp_path / "synthetic.csv"
    synthetic.to_csv(csv, index=False)
    with pytest.raises(SystemExit) as e:
        main(["--input", str(csv), "--crops", "Saffron", "--models-dir", str(tmp_path / "m")])
    assert "not in the data" in str(e.value)


def test_meta_says_what_the_model_is_and_is_not(monkeypatch, trained):
    models, _meta = trained
    m = _client(monkeypatch, models).get("/v1/forecast/meta").json()
    assert m["horizons_days"] == [7, 30] and "min_price" in m["features_never_used"]
    assert m["markets"]["Onion|Maharashtra"] == ["Lasalgaon", "Nashik", "Pimpalgaon"]   # AIAIC offers these mandis
    assert set(m["range_widening"]) == {"7", "30"}
