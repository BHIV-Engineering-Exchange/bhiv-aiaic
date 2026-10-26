"""Answer a forecast request from the latest trained bundle, or say plainly why not.

Abstains (answers with `abstain: true` and a reason code, never a guessed range) when:
- `not_trained_for`: the commodity, state or mandi is not in the model's closed lists;
- `horizon_not_trained`: the horizon is not one the model was trained for;
- `no_recent_price`: the mandi has no quote in the 7 days before `as_of`;
- `as_of_outside_history`: `as_of` is later than the model's data plus 7 days (it would forecast from stale prices);
- `no_skill`: in the walk-forward backtest the model did not beat "the price stays where it is" for this crop, state
  and horizon, or had too few test samples to say.
The backtest row is returned in every case, so a caller can show how the model has done.
"""

from __future__ import annotations

import json
import os
import uuid
from functools import lru_cache
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from src.forecast.series import ASOF_TOLERANCE_DAYS, Series, features_as_of
from src.forecast.train import _categorical, predict_quantiles


class ForecastUnavailable(Exception):
    """No trained forecast bundle is present."""


def _latest(models_dir: Path) -> Path:
    found = sorted(p for p in models_dir.glob("forecast_*.json"))
    if not found:
        # No server path in the message: it reaches callers in a 503 (review 2026-10-03).
        raise ForecastUnavailable("No trained forecast is loaded. Run: python -m src.forecast.train")
    return found[-1]


@lru_cache(maxsize=1)
def load_bundle(models_dir: str) -> Dict:
    meta_path = _latest(Path(models_dir))
    meta = json.loads(meta_path.read_text())
    bundle = joblib.load(Path(models_dir) / meta["model_file"])
    history = pd.read_csv(Path(models_dir) / meta["history_file"], parse_dates=["date"])
    return {"meta": meta, "bundle": bundle, "history": history}


def _norm(v: Optional[str]) -> str:
    return str(v or "").strip()


def _backtest_row(meta: Dict, commodity: str, state: str, horizon: int) -> Optional[Dict]:
    for g in meta["backtest"]["groups"]:
        if g["commodity"] == commodity and g["state"] == state and g["horizon_days"] == horizon:
            return g
    return None


KNOWN_UNKNOWNS = {
    "onion": ["arrivals data not available", "export policy not included", "weather not included"],
    "soybean": ["arrivals data not available", "global commodity prices not included"],
    "tur": ["arrivals data not available", "MSP season coverage partial"],
    "wheat": ["arrivals data not available"],
    "maize": ["arrivals data not available"],
    "gram": ["arrivals data not available"],
    "bajra": ["arrivals data not available"],
    "tomato": ["arrivals data not available", "no MSP exists for tomato", "high volatility crop"],
}


def _freshness(trained_through: str, generated_at: datetime) -> str:
    days = (generated_at.date() - pd.Timestamp(trained_through).date()).days
    if days < 7:
        return "fresh"
    if days < 30:
        return "aging"
    return "stale"


def _confidence(backtest: Optional[Dict], abstain: bool) -> str:
    if abstain or not backtest:
        return "low"
    skill = backtest.get("skill_vs_persistence", 0)
    coverage = backtest.get("coverage_p10_p90", 0)
    if skill > 0.15 and coverage > 0.80:
        value = "high"
    elif skill > 0.05 and coverage > 0.70:
        value = "medium"
    else:
        value = "low"
    return "medium" if value == "high" else value


def forecast(req: Dict, models_dir: str) -> Dict:
    """`req`: commodity, state, market, as_of (YYYY-MM-DD), horizon_days, optional history [{date, modal_price}]."""
    loaded = load_bundle(models_dir)
    meta, bundle = loaded["meta"], loaded["bundle"]
    generated_dt = datetime.now(timezone.utc)
    generated_at = generated_dt.isoformat().replace("+00:00", "Z")
    source_name = str(meta["data_source"]).replace("AIAIC pilot export", "AIAIC price export")
    if req.get("history") is not None:
        return {
            "abstain": True,
            "reason_code": "SYNTHETIC_INPUT",
            "reason": "Caller-supplied history is not accepted in production. Use the stored model history.",
            "confidence": "low",
            "read_these_first": ["This answer uses test data, not real market prices."],
            "sources_used": [{"source_name": source_name, "trained_through": meta["trained_through"],
                              "real_world_freshness": _freshness(meta["trained_through"], generated_dt),
                              "synthetic": True}],
            "p10": None,
            "p50": None,
            "p90": None,
        }
    commodity, state, market = _norm(req.get("commodity")), _norm(req.get("state")), _norm(req.get("market"))
    horizon = int(req.get("horizon_days") or 0)
    as_of = np.datetime64(pd.Timestamp(req.get("as_of")).normalize().date(), "D")
    target = as_of + np.timedelta64(horizon, "D")
    freshness = _freshness(meta["trained_through"], generated_dt)
    engine_version = os.environ.get("ENGINE_VERSION") or Path(meta["model_file"]).stem
    backtest = _backtest_row(meta, commodity, state, horizon)
    known_unknowns = [
        *KNOWN_UNKNOWNS.get(commodity.lower(), []),
        f"Model trained on data up to {meta['trained_through']}. Forecast patterns are based on historical prices, "
        "not recent market conditions.",
    ]
    read_these_first = (
        ["This forecast is not better than a simple guess."]
        if backtest and backtest.get("skill_vs_persistence", 0) < 0
        else []
    )
    base = {
        "decision_id": f"forecast-{uuid.uuid4().hex[:8]}",
        "generated_at": generated_at,
        "engine_version": engine_version,
        "commodity": commodity, "state": state, "market": market, "as_of": str(as_of), "horizon_days": horizon,
        "target_date": str(target), "unit": "INR per quintal (modal price)",
        "model": {"file": meta["model_file"], "trained_at": meta["trained_at"], "trained_through": meta["trained_through"],
                  "data_source": meta["data_source"], "data_hash": meta["data_hash"],
                  "backtest_skill": backtest.get("skill_vs_persistence") if backtest else None,
                  "backtest_coverage": backtest.get("coverage_p10_p90") if backtest else None},
        "backtest": backtest,
        "last_price": None,
        "last_price_date": None,
        "expected_change_pct": None,
        "direction": "UNCLEAR",
        "confidence": "low",
        "known_unknowns": known_unknowns,
        "read_these_first": read_these_first,
        "sources_used": [{"source_name": source_name, "trained_through": meta["trained_through"],
                          "real_world_freshness": freshness, "synthetic": False}],
    }

    def abstain(code: str, why: str) -> Dict:
        return {**base, "abstain": True, "reason_code": code, "reason": why, "p10": None, "p50": None, "p90": None}

    if commodity not in meta["commodities"] or state not in meta["states"] \
            or market not in meta["markets"].get(f"{commodity}|{state}", []):
        return abstain("not_trained_for", "This crop, state and mandi are not in what the model was trained on.")
    if horizon not in meta["horizons_days"]:
        return abstain("horizon_not_trained", f"Horizons trained: {meta['horizons_days']} days.")
    if target <= np.datetime64(meta["trained_through"], "D"):
        # The served models were fitted on every sample up to `trained_through`, so this day's outcome is in their
        # training data. A range here would describe a known price, and it would look far better than a forecast.
        return abstain("in_sample", f"This day's outcome ({target}) is inside the model's training data (to "
                                    f"{meta['trained_through']}), so a range here would not be a forecast.")
    latest = np.datetime64(meta["trained_through"], "D") + np.timedelta64(ASOF_TOLERANCE_DAYS, "D")
    if as_of > latest and as_of > np.datetime64(generated_dt.date(), "D"):
        return abstain("as_of_outside_history", f"The model's prices end on {meta['trained_through']}; send `history` "
                                                "to forecast from a later day.")
    hist = loaded["history"]
    g = hist[(hist["commodity"] == commodity) & (hist["state"] == state) & (hist["market"] == market)]
    g = g[g["date"] <= pd.Timestamp(as_of)]
    if g.empty:
        return abstain("no_recent_price", "No price for this mandi on or before that day.")
    last_price_date = g["date"].max().date().isoformat()
    last_price = float(g.loc[g["date"].idxmax(), "modal_price"])
    days_since_last = (generated_dt.date() - pd.Timestamp(last_price_date).date()).days
    base.update({"last_price": last_price, "last_price_date": last_price_date, "days_since_last_price": days_since_last})
    days_since_as_of = (pd.Timestamp(as_of).date() - pd.Timestamp(last_price_date).date()).days
    if days_since_as_of > 30:
        base["read_these_first"].append(f"The last price is {days_since_as_of} days old.")
        return {**base, "abstain": True, "reason_code": "STALE_PRICE",
                "reason": f"The last price is {days_since_as_of} days old.",
                "p10": None, "p50": None, "p90": None}
    bt = base["backtest"]
    if not bt or not bt.get("usable"):
        return abstain("no_skill", "In its backtest the model did not beat 'the price stays where it is' for this crop, "
                                   "state and horizon, or its range held too few of the real outcomes (under 70%), or "
                                   "there were too few test cases to say.")
    row = features_as_of((commodity, state, market), Series.from_frame(g), as_of, horizon)
    if row is None:
        return abstain("no_recent_price", f"The mandi has no quote in the {ASOF_TOLERANCE_DAYS} days before {as_of}.")
    X = _categorical(pd.DataFrame([row]), bundle["categories"])
    qhat = float((meta.get("range_widening") or {}).get(str(horizon), {}).get("qhat", 0.0))
    raw = predict_quantiles(bundle["models"], X)[0]
    p10, p50, p90 = (round(float(v), 2) for v in (raw[0] - qhat, raw[1], raw[2] + qhat))
    expected_change_pct = round((p50 - last_price) / last_price * 100, 2)
    direction = "UP" if expected_change_pct > 3 else "DOWN" if expected_change_pct < -3 else "FLAT"
    base.update({"confidence": _confidence(bt, False), "expected_change_pct": expected_change_pct,
                 "direction": direction})
    return {**base, "abstain": False, "p10": p10, "p50": p50, "p90": p90,
            "range_widened_by": qhat,
            "what_this_is": ("A range for the modal price at this mandi around the target date: p50 is the middle "
                             "estimate. In the backtest the outcome fell inside p10-p90 in the share of cases given as "
                             "backtest.coverage_p10_p90. It is a model estimate, not a promise.")}


def list_meta(models_dir: str) -> Dict:
    meta = load_bundle(models_dir)["meta"]
    keep = ("model_file", "trained_at", "trained_through", "backtest_cutoff", "data_source", "horizons_days",
            "quantiles", "features", "features_never_used", "commodities", "states", "markets", "backtest",
            "range_widening", "target_coverage_p10_p90")
    out = {k: meta[k] for k in keep}
    for k in ("market_last_quote", "market_identity"):      # models trained before 2026-10-02 do not have them
        if k in meta:
            out[k] = meta[k]
    return out


def reset_cache() -> None:
    load_bundle.cache_clear()


def usable_groups(models_dir: str) -> List[Dict]:
    return [g for g in load_bundle(models_dir)["meta"]["backtest"]["groups"] if g.get("usable")]
