"""Train the forecast: three quantile models (p10, p50, p90), checked walk-forward against the naive rule.

    python -m src.forecast.train --horizons 7 14 30 [--source "Agmarknet via data.gov.in (CEDA)"]
        [--input rows.csv[.gz]] [--crops Onion Wheat] [--holdout-days 90] [--step-days 7] [--seed 42]
        [--models-dir models]

    # the configuration of the official backtest AIAIC measured (2026-09-28); it reproduces that table:
    python -m src.forecast.train --input agmarknet_mh_mp_2016_2026.csv.gz --crops Onion Soyabean Wheat \
        --horizons 7 30 --holdout-days 180 --step-days 7 --seed 42 --models-dir models/forecast_official \
        --source "Agmarknet via data.gov.in (Variety-wise Daily Market Prices), AIAIC export 2016-2026"

    # the pilot crops on AIAIC's export of 2026-10-02, one series per MANDI across Agmarknet's Nov-2025 renames
    # (read its MANIFEST.json first; name the file's sha256 in --source):
    python -m src.forecast.train --input agmarknet_pilot_mh_mp_2001_2026.csv.gz --market-column mandi \
        --commodity-column crop --crops onion soybean tur wheat --horizons 7 30 --holdout-days 180 \
        --step-days 7 --seed 42 --models-dir models/forecast_pilot \
        --source "Agmarknet via data.gov.in, AIAIC pilot export 2026-10-02 (sha256 <from MANIFEST.json>)"

Reads the PREPROCESSED dataset (the output of `run_pipeline.py --stage preprocess`, or any frame with commodity,
state, market, date, modal_price). Writes, in `models/` (or `--models-dir`):
    forecast_<UTC time>_<data hash>.joblib         the three models and the category lists
    forecast_<UTC time>_<data hash>.json           metadata: horizons, dates, closed lists, the backtest
    forecast_<UTC time>_<data hash>_history.csv.gz the last 400 days per mandi, so the service can answer alone

THE BACKTEST IS WALK-FORWARD. Models are fitted on samples whose TARGET date is on or before the cutoff, and scored
on samples ASKED after it, so no price from the scoring period is ever seen in training. The number that matters is
`skill_vs_persistence` = 1 - MAE(p50) / MAE("the price stays where it is"). Above 0 means the model helped; at or
below 0 the service ABSTAINS for that crop, state and horizon.

THE RANGE IS CALIBRATED (split conformal: "conformalized quantile regression", Romano, Patterson and Candes, 2019).
Raw quantile models are usually too narrow (on the synthetic test data, p10-p90 held 64% of outcomes, not 80%). The
scoring period is split in time: its first half CALIBRATES a widening `qhat` per horizon (the finite-sample 80%
quantile of max(p10 - y, y - p90)), and the second half, untouched, is where every reported number comes from.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.forecast.series import CATEGORICAL_FEATURES, FEATURES, daily_series, training_samples

logger = logging.getLogger(__name__)

QUANTILES = (0.1, 0.5, 0.9)
TARGET_COVERAGE = 0.8          # p10..p90
MIN_CALIBRATION = 20           # below this, a horizon's range is not widened, and the metadata says so
HOLDOUT_DAYS = 90
MIN_GROUP_TEST_SAMPLES = 20
#: A range that held fewer outcomes than this in the scored period is not shown, whatever its p50 did. Measured on the
#: official data (2026-09-28): Maharashtra onion at 30 days beat persistence slightly, but its "80%" range held 52%.
MIN_COVERAGE = 0.7
HISTORY_DAYS = 400


def _categorical(frame: pd.DataFrame, categories: Dict[str, List[str]]) -> pd.DataFrame:
    """The same category lists at training and at inference, so a code means the same mandi in both."""
    X = frame[FEATURES].copy()
    for c in CATEGORICAL_FEATURES:
        X[c] = pd.Categorical(X[c].astype(str), categories=categories[c])
    for c in FEATURES:
        if c not in CATEGORICAL_FEATURES:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
    return X


def _fit(X: pd.DataFrame, y: pd.Series, alpha: float, seed: int) -> lgb.LGBMRegressor:
    m = lgb.LGBMRegressor(objective="quantile", alpha=alpha, n_estimators=400, learning_rate=0.05, num_leaves=31,
                          min_child_samples=20, random_state=seed, verbose=-1)
    m.fit(X, y, categorical_feature=CATEGORICAL_FEATURES)
    return m


def predict_quantiles(models: Dict[float, lgb.LGBMRegressor], X: pd.DataFrame) -> np.ndarray:
    """(n, 3) array of p10, p50, p90, sorted per row so the range never crosses itself."""
    raw = np.column_stack([models[q].predict(X) for q in QUANTILES])
    return np.sort(raw, axis=1)


def conformal_widening(cal: pd.DataFrame, q: np.ndarray) -> Dict[int, Dict]:
    """Per horizon: the amount `qhat` to widen p10..p90 by so that about TARGET_COVERAGE of outcomes fall inside
    (split-conformal, finite-sample quantile). Horizons with fewer than MIN_CALIBRATION cases get 0 and say so."""
    y = cal["target"].to_numpy()
    scores = np.maximum(q[:, 0] - y, y - q[:, 2])
    out = {}
    for h in sorted(cal["horizon_days"].unique()):
        s = np.sort(scores[cal["horizon_days"].to_numpy() == h])
        n = len(s)
        if n < MIN_CALIBRATION:
            out[int(h)] = {"qhat": 0.0, "n_calibration": n, "note": "too few calibration cases; range NOT widened"}
            continue
        k = min(n - 1, int(np.ceil((n + 1) * TARGET_COVERAGE)) - 1)
        out[int(h)] = {"qhat": round(max(0.0, float(s[k])), 2), "n_calibration": n}
    return out


def widen(q: np.ndarray, horizons: np.ndarray, widening: Dict[int, Dict]) -> np.ndarray:
    """p10 - qhat, p50, p90 + qhat, per row's horizon."""
    w = np.array([widening.get(int(h), {}).get("qhat", 0.0) for h in horizons], dtype=float)
    return np.column_stack([q[:, 0] - w, q[:, 1], q[:, 2] + w])


def backtest_table(test: pd.DataFrame, q: np.ndarray, raw: Optional[np.ndarray] = None) -> Dict:
    """Overall and per (commodity, state, horizon): MAE of p50, MAE of persistence, skill, and how often the outcome
    fell inside p10..p90 (calibrated; and before widening, when `raw` is given)."""
    t = test.assign(p10=q[:, 0], p50=q[:, 1], p90=q[:, 2])
    t["err_model"] = (t["p50"] - t["target"]).abs()
    t["err_persist"] = (t["last_price"] - t["target"]).abs()
    t["inside"] = (t["target"] >= t["p10"]) & (t["target"] <= t["p90"])
    if raw is not None:
        t["inside_raw"] = (t["target"] >= raw[:, 0]) & (t["target"] <= raw[:, 2])

    def summarise(g: pd.DataFrame) -> Dict:
        mae, base = float(g["err_model"].mean()), float(g["err_persist"].mean())
        row = {"n": int(len(g)), "mae_p50": round(mae, 2), "mae_persistence": round(base, 2),
               "skill_vs_persistence": round(1.0 - mae / base, 4) if base > 0 else None,
               "coverage_p10_p90": round(float(g["inside"].mean()), 4)}
        if "inside_raw" in g:
            row["coverage_p10_p90_before_calibration"] = round(float(g["inside_raw"].mean()), 4)
        return row

    groups = []
    for (c, s, h), g in t.groupby(["commodity", "state", "horizon_days"], observed=True):
        row = {"commodity": c, "state": s, "horizon_days": int(h), **summarise(g)}
        row["usable"] = bool(row["n"] >= MIN_GROUP_TEST_SAMPLES and (row["skill_vs_persistence"] or 0) > 0
                             and row["coverage_p10_p90"] >= MIN_COVERAGE)
        groups.append(row)
    return {"overall": summarise(t), "groups": groups}


def walk_forward_split(samples: pd.DataFrame, holdout_days: int = HOLDOUT_DAYS):
    """(train, test, cutoff). Train: samples whose TARGET date is on or before the cutoff. Test: samples ASKED after
    it. No test-period price is ever inside a training FEATURE. A training TARGET is the nearest quote on or after its
    target day, up to TARGET_TOLERANCE_DAYS (3) later, so it can be a quote a few days past the cutoff. It can
    coincide with a scored target only for a horizon shorter than that tolerance, and every horizon used here is
    longer."""
    cutoff = samples["target_date"].max() - pd.Timedelta(days=holdout_days)
    return samples[samples["target_date"] <= cutoff], samples[samples["as_of"] > cutoff], cutoff


def train(df: pd.DataFrame, horizons: List[int], source: str, models_dir: Path, seed: int = 42,
          holdout_days: int = HOLDOUT_DAYS, step_days: int = 7, market_identity: Optional[str] = None) -> Dict:
    daily = daily_series(df)
    if daily.empty:
        raise RuntimeError("No usable rows (need commodity, state, market, date and a positive modal_price).")
    samples = training_samples(daily, horizons, step_days=step_days)
    if samples.empty:
        raise RuntimeError("No training samples: every mandi has too little history for these horizons.")
    train_s, test_s, cutoff = walk_forward_split(samples, holdout_days)
    if len(train_s) < 200 or len(test_s) < 50:
        raise RuntimeError(f"Too few samples for an honest backtest (train={len(train_s)}, test={len(test_s)}).")

    categories = {c: sorted(daily[c].astype(str).unique().tolist()) for c in CATEGORICAL_FEATURES}
    Xtr, ytr = _categorical(train_s, categories), train_s["target"]
    bt_models = {q: _fit(Xtr, ytr, q, seed) for q in QUANTILES}
    # The scoring period, split in time: the first half calibrates the range, the second half is scored untouched.
    test_s = test_s.sort_values("as_of").reset_index(drop=True)
    mid = test_s["as_of"].iloc[len(test_s) // 2]
    cal = test_s[test_s["as_of"] < mid].reset_index(drop=True)
    ev = test_s[test_s["as_of"] >= mid].reset_index(drop=True)
    widening = conformal_widening(cal, predict_quantiles(bt_models, _categorical(cal, categories)))
    raw_ev = predict_quantiles(bt_models, _categorical(ev, categories))
    backtest = backtest_table(ev, widen(raw_ev, ev["horizon_days"].to_numpy(), widening), raw=raw_ev)
    backtest["scored_from"] = str(mid.date())
    backtest["calibrated_on"] = (f"{cal['as_of'].min().date()} to {cal['as_of'].max().date()}" if len(cal)
                                 else "nothing")

    # The served models are REFITTED on every sample, the backtest period included. The backtest above is the record
    # of this PROCEDURE (the same features, settings and seed, fitted up to the cutoff), not of these exact fitted
    # models. Their range is widened by the `qhat` calibrated above. A day whose outcome is inside this training data
    # is refused by the service (`IN_SAMPLE`), since a range there would describe a known price.
    Xall = _categorical(samples, categories)
    served = {q: _fit(Xall, samples["target"], q, seed) for q in QUANTILES}

    data_hash = hashlib.sha256(pd.util.hash_pandas_object(daily, index=False).values.tobytes()).hexdigest()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = f"forecast_{stamp}_{data_hash[:8]}"
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": served, "categories": categories, "quantiles": QUANTILES}, models_dir / f"{stem}.joblib")

    keep_from = daily["date"].max() - pd.Timedelta(days=HISTORY_DAYS)
    daily[daily["date"] >= keep_from].to_csv(models_dir / f"{stem}_history.csv.gz", index=False, compression="gzip")

    markets = {f"{c}|{s}": sorted(g["market"].unique().tolist())
               for (c, s), g in daily.groupby(["commodity", "state"], observed=True)}
    # The last day each mandi quoted: a caller can tell a mandi that still reports from one that stopped (the
    # service refuses the latter with `NO_RECENT_PRICE`; this says so before anyone asks).
    last_quote = {f"{c}|{s}": {m: str(d.date()) for m, d in g.groupby("market", observed=True)["date"].max().items()}
                  for (c, s), g in daily.groupby(["commodity", "state"], observed=True)}
    meta = {
        "model_file": f"{stem}.joblib", "history_file": f"{stem}_history.csv.gz", "trained_at": stamp,
        "data_hash": data_hash, "data_source": source, "horizons_days": sorted(horizons),
        "trained_through": str(daily["date"].max().date()), "backtest_cutoff": str(cutoff.date()),
        "holdout_days": holdout_days, "step_days": step_days, "quantiles": list(QUANTILES), "features": FEATURES,
        "features_never_used": ["min_price", "max_price", "any price dated after as_of"],
        "commodities": categories["commodity"], "states": categories["state"], "markets": markets,
        "market_last_quote": last_quote,
        "market_identity": market_identity or "the published market name, as given (no identity applied)",
        "backtest": backtest, "range_widening": {str(h): v for h, v in widening.items()},
        "target_coverage_p10_p90": TARGET_COVERAGE, "random_seed": seed,
    }
    (models_dir / f"{stem}.json").write_text(json.dumps(meta, indent=2))
    logger.info("Forecast trained: %s | overall %s", stem, backtest["overall"])
    return meta


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Train the AIAIC price forecast (quantiles, walk-forward).")
    ap.add_argument("--horizons", type=int, nargs="+", default=[7, 14, 30])
    ap.add_argument("--source", default="see data/raw (state it: e.g. 'Agmarknet via data.gov.in')")
    ap.add_argument("--input", default=None, help="CSV of preprocessed rows (default: rerun ingest..preprocess)")
    ap.add_argument("--crops", nargs="+", default=None, help="Train only these commodities (exact names)")
    ap.add_argument("--holdout-days", type=int, default=HOLDOUT_DAYS, help="Days before the last target left for the backtest")
    ap.add_argument("--step-days", type=int, default=7, help="Days between two samples of one mandi")
    ap.add_argument("--seed", type=int, default=None, help="Default: settings.random_seed")
    ap.add_argument("--models-dir", default=None, help="Default: settings.models_dir")
    ap.add_argument("--market-column", default="market",
                    help="The column that names a mandi. Use 'mandi' with AIAIC's export: Agmarknet renamed about 700 "
                         "mandis 'X' -> 'X APMC' in Nov 2025, and `mandi` keeps each one a single series.")
    ap.add_argument("--commodity-column", default="commodity",
                    help="The column that names a crop. Use 'crop' with AIAIC's export: Agmarknet spelled whole tur "
                         "three ways since 2001, and `crop` keeps it one crop. --crops then takes crop keys "
                         "(onion soybean tur wheat).")
    args = ap.parse_args(argv)
    from src.config.settings import settings
    identity = None
    if args.input:
        cols = ("commodity", "state", "market", "date", "modal_price", args.market_column, args.commodity_column)
        df = pd.read_csv(args.input, usecols=lambda c: c in cols)
        used = []
        for col, as_ in ((args.market_column, "market"), (args.commodity_column, "commodity")):
            if col == as_:
                continue
            if col not in df.columns:
                raise SystemExit(f"--{as_}-column: no column '{col}' in {args.input}")
            df[as_] = df.pop(col)
            used.append(f"'{col}' for the {as_}")
        if used:
            identity = (f"the input's {' and '.join(used)} (one series per mandi and crop across Agmarknet's "
                        "renames), not the published names")
    else:
        from src.ingestion.pipeline import ingest_raw_dataset
        from src.preprocessing.pipeline import run_preprocessing
        from src.validation.pipeline import validate_dataset
        raw, m1 = ingest_raw_dataset()
        valid, _q, m2 = validate_dataset(raw, parent_hash=m1["content_hash"])
        df, _m3 = run_preprocessing(valid, parent_hash=m2["content_hash"])
    if args.crops:
        unknown = sorted(set(args.crops) - set(df["commodity"].astype(str).unique()))
        if unknown:
            raise SystemExit(f"--crops: not in the data: {unknown}. Names are exact, as the data spells them.")
        df = df[df["commodity"].isin(args.crops)]
    seed = settings.random_seed if args.seed is None else args.seed
    models_dir = Path(args.models_dir) if args.models_dir else settings.models_dir
    meta = train(df, args.horizons, args.source, models_dir, seed=seed, holdout_days=args.holdout_days,
                 step_days=args.step_days, market_identity=identity)
    print(json.dumps({k: meta[k] for k in ("model_file", "trained_through", "backtest_cutoff")}, indent=1))
    print(json.dumps(meta["backtest"]["overall"], indent=1))


if __name__ == "__main__":
    logging.basicConfig(level="INFO", format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    main()
