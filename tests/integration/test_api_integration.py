"""
The served API end to end, through HTTP (TestClient), against a model trained on SYNTHETIC prices in a temporary
directory (tests/integration/conftest.py). Nothing here needs a model in models/, so it runs the same on a clean
clone and in CI. No figure here describes a real market.
"""

import pytest
from datetime import datetime, timezone
from fastapi.testclient import TestClient

from src.forecast import service


def _generated_at(monkeypatch, date_text):
    generated = datetime.fromisoformat(date_text).replace(tzinfo=timezone.utc)

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return generated if tz else generated.replace(tzinfo=None)

    monkeypatch.setattr(service, "datetime", FixedDatetime)


@pytest.fixture
def client(monkeypatch, api_models):
    from src.api.main import app
    from src.config.settings import settings
    models, _meta = api_models
    monkeypatch.setattr(settings, "models_dir", models)
    service.reset_cache()
    yield TestClient(app)
    service.reset_cache()


def _ask(meta, group, **over):
    body = {"commodity": group["commodity"], "state": group["state"],
            "market": meta["markets"][f"{group['commodity']}|{group['state']}"][0],
            "as_of": meta["trained_through"], "horizon_days": group["horizon_days"]}
    body.update(over)
    return body


def test_health_and_ready_describe_the_served_forecast(client, api_models):
    _models, meta = api_models
    h = client.get("/v1/health").json()
    assert h["status"] == "ok" and h["model_loaded"] is True
    assert h["model_file"] == meta["model_file"] and h["trained_through"] == meta["trained_through"]
    assert client.get("/v1/ready").json() == {"ready": True, "reason": None}


def test_meta_says_what_the_model_never_reads(client):
    m = client.get("/v1/forecast/meta").json()
    assert {"min_price", "max_price"} <= set(m["features_never_used"])
    assert not {"min_price", "max_price"} & set(m["features"])


def test_a_usable_group_answers_with_an_ordered_range_and_its_backtest(client, api_models, monkeypatch):
    _models, meta = api_models
    _generated_at(monkeypatch, meta["trained_through"])
    usable = [g for g in meta["backtest"]["groups"] if g["usable"]]
    assert usable, "the synthetic data must give at least one usable group, or this test proves nothing"
    r = client.post("/v1/forecast", json=_ask(meta, usable[0]))
    assert r.status_code == 200
    out = r.json()
    assert out["abstain"] is False and out["p10"] <= out["p50"] <= out["p90"]
    assert out["backtest"]["usable"] is True and out["model"]["file"] == meta["model_file"]
    assert out["request_id"] and out["execution_status"] == "SUCCESS"


def test_the_same_question_gets_the_same_range(client, api_models, monkeypatch):
    _models, meta = api_models
    _generated_at(monkeypatch, meta["trained_through"])
    g = [g for g in meta["backtest"]["groups"] if g["usable"]][0]
    a = client.post("/v1/forecast", json=_ask(meta, g)).json()
    b = client.post("/v1/forecast", json=_ask(meta, g)).json()
    assert (a["p10"], a["p50"], a["p90"]) == (b["p10"], b["p50"], b["p90"])
    assert a["request_id"] != b["request_id"]


def test_an_untrained_crop_or_mandi_is_refused_not_guessed(client, api_models, monkeypatch):
    _models, meta = api_models
    _generated_at(monkeypatch, meta["trained_through"])
    g = meta["backtest"]["groups"][0]
    for over in ({"commodity": "Saffron"}, {"market": "Nowhere"}):
        out = client.post("/v1/forecast", json=_ask(meta, g, **over)).json()
        assert out["abstain"] is True and out["reason_code"] == "not_trained_for" and out["p50"] is None


def test_predict_stays_gone_when_a_model_is_loaded(client):
    assert client.post("/v1/predict", json={"commodity": "Onion", "state": "Maharashtra", "month": 8}).status_code == 410
