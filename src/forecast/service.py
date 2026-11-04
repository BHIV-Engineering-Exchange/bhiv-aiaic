"""Answer a forecast request from the latest trained bundle, or say plainly why not.

Abstains (answers with `abstain: true` and a reason code, never a guessed range) when:
- `NOT_TRAINED_FOR`: the commodity, state or mandi is not in the model's closed lists;
- `HORIZON_NOT_TRAINED`: the horizon is not one the model was trained for;
- `NO_RECENT_PRICE`: the mandi has no quote in the 7 days before `as_of`;
- `AS_OF_OUTSIDE_HISTORY`: `as_of` is later than the model's data plus 7 days (it would forecast from stale prices);
- `PAST_TARGET`: the target date is before the forecast was generated;
- `NO_SKILL`: in the walk-forward backtest the model did not beat "the price stays where it is" for this crop, state
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


def _confidence(backtest: Optional[Dict], abstain: bool, stale: bool = False) -> str:
    # high: skill above 0.15 and coverage above 0.80. medium: skill above 0.05 and coverage above 0.70.
    # low: all other cases. Capped at medium when data is stale. Always low when abstain is true.
    # STALE_PRICE triggers when days_since_last_price is above 30.
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
    return "medium" if stale and value == "high" else value


def forecast(req: Dict, models_dir: str) -> Dict:
    """`req`: commodity, state, market, as_of (YYYY-MM-DD), horizon_days, optional history [{date, modal_price}]."""
    loaded = load_bundle(models_dir)
    meta, bundle, hist = loaded["meta"], loaded["bundle"], loaded["history"]
    generated_dt = datetime.now(timezone.utc)
    generated_at = generated_dt.isoformat().replace("+00:00", "Z")
    source_name = (str(meta["data_source"])
                   .replace("AIAIC pilot export", "AIAIC price export")
                   .replace("agmarknet_pilot_", "agmarknet_"))
    confidence_rule = ("high: skill above 0.15 and coverage above 0.80. medium: skill above 0.05 and coverage above "
                       "0.70. low: all other cases. Capped at medium when data is stale. Always low when abstain is "
                       "true. STALE_PRICE triggers when days_since_last_price is above 30.")
    if req.get("history") is not None:
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
        stored = hist[(hist["commodity"] == commodity) & (hist["state"] == state) & (hist["market"] == market)]
        stored = stored[stored["date"] <= pd.Timestamp(as_of)]
        stored_fields = {}
        if not stored.empty:
            stored_date = stored["date"].max().date().isoformat()
            stored_fields = {
                "last_price": float(stored.loc[stored["date"].idxmax(), "modal_price"]),
                "last_price_date": stored_date,
                "days_since_last_price": (pd.Timestamp(as_of).date() - pd.Timestamp(stored_date).date()).days,
            }
        read_these_first = ["This answer uses test data, not real market prices."]
        if freshness == "stale":
            read_these_first.append(
                "This forecast uses data that is more than 30 days old. Confidence is capped at medium."
            )
        return {
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
            "last_price": stored_fields.get("last_price"),
            "last_price_date": stored_fields.get("last_price_date"),
            "days_since_last_price": stored_fields.get("days_since_last_price"),
            "expected_change_pct": None, "direction": "UNCLEAR",
            "confidence": "low",
            "confidence_rule": confidence_rule,
            "known_unknowns": known_unknowns,
            "read_these_first": read_these_first,
            "sources_used": [{"source_name": source_name, "trained_through": meta["trained_through"],
                              "observed_on": meta["trained_through"],
                              "licence": "Government Open Data Licence - India (GODL-India)",
                              "real_world_freshness": freshness,
                              "synthetic": True}],
            "abstain": True,
            "replay": bool(req.get("replay", False)),
            "reason_code": "SYNTHETIC_INPUT",
            "reason": "Caller-supplied history is not accepted in production. Use the stored model history.",
            "p10": None,
            "p50": None,
            "p90": None,
            "range_widened_by": None,
            "what_this_is": None,
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
        "replay": bool(req.get("replay", False)),
        "model": {"file": meta["model_file"], "trained_at": meta["trained_at"], "trained_through": meta["trained_through"],
                  "data_source": meta["data_source"], "data_hash": meta["data_hash"],
                  "backtest_skill": backtest.get("skill_vs_persistence") if backtest else None,
                  "backtest_coverage": backtest.get("coverage_p10_p90") if backtest else None},
        "backtest": backtest,
        "last_price": None,
        "last_price_date": None,
        "days_since_last_price": None,
        "expected_change_pct": None,
        "direction": "UNCLEAR",
        "confidence": "low",
        "confidence_rule": confidence_rule,
        "known_unknowns": known_unknowns,
        "read_these_first": read_these_first,
        "sources_used": [{"source_name": source_name, "trained_through": meta["trained_through"],
                          "observed_on": meta["trained_through"],
                          "licence": "Government Open Data Licence - India (GODL-India)",
                          "real_world_freshness": freshness, "synthetic": False}],
        "abstain": None,
        "reason_code": None,
        "reason": None,
        "p10": None,
        "p50": None,
        "p90": None,
        "range_widened_by": None,
        "what_this_is": None,
    }

    def abstain(code: str, why: str) -> Dict:
        return {**base, "abstain": True, "reason_code": code, "reason": why, "p10": None, "p50": None, "p90": None,
                "range_widened_by": None, "what_this_is": None}

    if freshness == "stale":
        base["read_these_first"].append(
            "This forecast uses data that is more than 30 days old. Confidence is capped at medium."
        )
    if commodity not in meta["commodities"] or state not in meta["states"] \
            or market not in meta["markets"].get(f"{commodity}|{state}", []):
        return abstain("NOT_TRAINED_FOR", "This crop, state and mandi are not in what the model was trained on.")
    g = hist[(hist["commodity"] == commodity) & (hist["state"] == state) & (hist["market"] == market)]
    g = g[g["date"] <= pd.Timestamp(as_of)]
    if not g.empty:
        last_price_date = g["date"].max().date().isoformat()
        base.update({
            "last_price": float(g.loc[g["date"].idxmax(), "modal_price"]),
            "last_price_date": last_price_date,
            "days_since_last_price": (pd.Timestamp(as_of).date() - pd.Timestamp(last_price_date).date()).days,
        })
    if req.get("replay"):
        base["read_these_first"].append(
            "TEST REPLAY EXAMPLE: This is a replay example using historical data, not a live forecast."
        )
    if not req.get("replay") and target < np.datetime64(generated_dt.date(), "D"):
        return abstain("PAST_TARGET", "The target date is in the past.")
    if horizon not in meta["horizons_days"]:
        return abstain("HORIZON_NOT_TRAINED", f"Horizons trained: {meta['horizons_days']} days.")
    if target <= np.datetime64(meta["trained_through"], "D"):
        # The served models were fitted on every sample up to `trained_through`, so this day's outcome is in their
        # training data. A range here would describe a known price, and it would look far better than a forecast.
        return abstain("IN_SAMPLE", f"This day's outcome ({target}) is inside the model's training data (to "
                                    f"{meta['trained_through']}), so a range here would not be a forecast.")
    latest = np.datetime64(meta["trained_through"], "D") + np.timedelta64(ASOF_TOLERANCE_DAYS, "D")
    if as_of > latest and as_of > np.datetime64(generated_dt.date(), "D"):
        return abstain("AS_OF_OUTSIDE_HISTORY", f"The model's prices end on {meta['trained_through']}; send `history` "
                                                "to forecast from a later day.")
    if g.empty:
        return abstain("NO_RECENT_PRICE", "No price for this mandi on or before that day.")
    last_price = base["last_price"]
    days_since_last = base["days_since_last_price"]
    if days_since_last > 30:
        base["read_these_first"].append(f"The last price is {days_since_last} days old.")
        return {**base, "abstain": True, "reason_code": "STALE_PRICE",
                "reason": f"The last price is {days_since_last} days old.",
                "p10": None, "p50": None, "p90": None,
                "range_widened_by": None, "what_this_is": None}
    bt = base["backtest"]
    if not bt or not bt.get("usable"):
        return abstain("NO_SKILL", "In its backtest the model did not beat 'the price stays where it is' for this crop, "
                                   "state and horizon, or its range held too few of the real outcomes (under 70%), or "
                                   "there were too few test cases to say.")
    row = features_as_of((commodity, state, market), Series.from_frame(g), as_of, horizon)
    if row is None:
        return abstain("NO_RECENT_PRICE", f"The mandi has no quote in the {ASOF_TOLERANCE_DAYS} days before {as_of}.")
    X = _categorical(pd.DataFrame([row]), bundle["categories"])
    qhat = float((meta.get("range_widening") or {}).get(str(horizon), {}).get("qhat", 0.0))
    raw = predict_quantiles(bundle["models"], X)[0]
    p10, p50, p90 = (round(float(v), 2) for v in (raw[0] - qhat, raw[1], raw[2] + qhat))
    expected_change_pct = round((p50 - last_price) / last_price * 100, 2)
    direction = "UP" if expected_change_pct > 3 else "DOWN" if expected_change_pct < -3 else "FLAT"
    base.update({"confidence": _confidence(bt, False, freshness == "stale"), "expected_change_pct": expected_change_pct,
                 "direction": direction})
    return {**base, "abstain": False, "reason_code": None, "reason": None,
            "p10": p10, "p50": p50, "p90": p90,
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
