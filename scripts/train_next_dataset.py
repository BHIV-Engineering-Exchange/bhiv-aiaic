"""Train and evaluate the 17 October 2026 four-crop pilot.

The model choice is made only on the calibration half of the walk-forward
holdout. The scored half is untouched until the selected variant is fixed.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

from src.forecast.series import CATEGORICAL_FEATURES, NUMERIC_FEATURES, daily_series, training_samples
from src.forecast.train import QUANTILES, _fit, backtest_table, conformal_widening, walk_forward_split, widen

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "2026-10-17_next" / "pilot_v3_nashik_next"
OUTPUT_DIR = ROOT / "models" / "forecast_next_20261017"
RESULTS_PATH = ROOT / "experiments" / "pilot_v3_20261017_metrics.json"
HORIZONS = [7, 30]
HOLDOUT_DAYS = 180
STEP_DAYS = 7
SEED = 42
logger = logging.getLogger("train_next_dataset")


def _season(crop: str, timestamp: pd.Timestamp) -> str:
    first_month = 4 if crop == "gram" else 10
    year = timestamp.year if timestamp.month >= first_month else timestamp.year - 1
    return f"{year}-{(year + 1) % 100:02d}"


def load_samples() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    source = DATA_DIR / "agmarknet_pilot_mh_mp_2001_2026.csv.gz"
    msp_path = DATA_DIR / "msp_pilot_crops.csv"
    logger.info("Loading %s", source)
    raw = pd.read_csv(source, usecols=["state", "date", "modal_price", "mandi", "crop"])
    raw = raw.rename(columns={"mandi": "market", "crop": "commodity"})
    daily = daily_series(raw)
    samples = training_samples(daily, HORIZONS, step_days=STEP_DAYS)
    msp = pd.read_csv(msp_path)
    lookup = {
        (str(crop).strip().lower(), str(year).strip()): float(value)
        for crop, year, value in zip(
            msp["crop"], msp["marketing_year"], msp["msp_rs_per_quintal"]
        )
        if pd.notna(value)
    }
    samples["msp_gap_pct"] = [
        100.0 * (last / lookup[(crop, _season(crop, as_of))] - 1.0)
        if (crop, _season(crop, as_of)) in lookup
        else np.nan
        for crop, as_of, last in zip(
            samples["commodity"], samples["as_of"], samples["last_price"]
        )
    ]
    data_hash = hashlib.sha256(
        pd.util.hash_pandas_object(daily, index=False).values.tobytes()
    ).hexdigest()
    return daily, samples, data_hash


def make_X(frame: pd.DataFrame, numeric_features: List[str], categories: Dict[str, List[str]]) -> pd.DataFrame:
    columns = CATEGORICAL_FEATURES + numeric_features
    result = frame[columns].copy()
    for column in CATEGORICAL_FEATURES:
        result[column] = pd.Categorical(result[column].astype(str), categories=categories[column])
    for column in numeric_features:
        result[column] = pd.to_numeric(result[column], errors="coerce").astype(float)
    return result


def predict_frame(models, frame, numeric_features, categories, log_target):
    if frame.empty:
        return frame.assign(r10=np.nan, r50=np.nan, r90=np.nan)
    raw = np.column_stack(
        [models[q].predict(make_X(frame, numeric_features, categories)) for q in QUANTILES]
    )
    if log_target:
        raw = np.exp(np.clip(raw, -2.0, 2.0)) * frame["last_price"].to_numpy(float)[:, None]
    return frame.assign(r10=np.sort(raw, axis=1)[:, 0],
                        r50=np.sort(raw, axis=1)[:, 1],
                        r90=np.sort(raw, axis=1)[:, 2])


def run_variant(name, train_set, calibration, scored, categories, numeric_features, log_target=False, per_crop=False):
    calibration_parts, scored_parts = [], []
    crops = sorted(train_set["commodity"].unique()) if per_crop else [None]
    for crop in crops:
        train_part = train_set if crop is None else train_set[train_set["commodity"] == crop]
        calibration_part = calibration if crop is None else calibration[calibration["commodity"] == crop]
        scored_part = scored if crop is None else scored[scored["commodity"] == crop]
        y = np.log(train_part["target"] / train_part["last_price"]) if log_target else train_part["target"]
        models = {q: _fit(make_X(train_part, numeric_features, categories), y, q, SEED)
                  for q in QUANTILES}
        calibration_parts.append(predict_frame(models, calibration_part, numeric_features, categories, log_target))
        scored_parts.append(predict_frame(models, scored_part, numeric_features, categories, log_target))
    calibration_out = pd.concat(calibration_parts, ignore_index=True).dropna(subset=["r10"])
    scored_out = pd.concat(scored_parts, ignore_index=True).dropna(subset=["r10"])
    widening = conformal_widening(calibration_out, calibration_out[["r10", "r50", "r90"]].to_numpy())
    raw = scored_out[["r10", "r50", "r90"]].to_numpy()
    adjusted = widen(raw, scored_out["horizon_days"].to_numpy(), widening)
    scored_eval = scored_out.assign(
        p10=adjusted[:, 0], p50=adjusted[:, 1], p90=adjusted[:, 2]
    )
    return {
        "name": name,
        "calibration": calibration_out,
        "scored": scored_out,
        "scored_eval": scored_eval,
        "widening": widening,
        "backtest": backtest_table(scored_out, adjusted, raw=raw),
        "per_crop": per_crop,
        "log_target": log_target,
        "numeric_features": numeric_features,
    }


def calibration_score(result, crop: str) -> float:
    frame = result["calibration"]
    frame = frame[(frame["commodity"] == crop) & frame["horizon_days"].isin(HORIZONS)]
    if frame.empty:
        return float("-inf")
    persistence = (frame["last_price"] - frame["target"]).abs().mean()
    if persistence <= 0:
        return float("-inf")
    return float(1.0 - (frame["r50"] - frame["target"]).abs().mean() / persistence)


def serialise_result(result) -> dict:
    return {
        "name": result["name"],
        "backtest": result["backtest"],
        "range_widening": result["widening"],
        "per_crop": result["per_crop"],
        "log_target": result["log_target"],
        "numeric_features": result["numeric_features"],
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    daily, samples, data_hash = load_samples()
    train_set, holdout, cutoff = walk_forward_split(samples, HOLDOUT_DAYS)
    holdout = holdout.sort_values("as_of").reset_index(drop=True)
    midpoint = holdout["as_of"].iloc[len(holdout) // 2]
    calibration = holdout[holdout["as_of"] < midpoint].reset_index(drop=True)
    scored = holdout[holdout["as_of"] >= midpoint].reset_index(drop=True)
    categories = {c: sorted(daily[c].astype(str).unique().tolist()) for c in CATEGORICAL_FEATURES}
    base_features = list(NUMERIC_FEATURES)
    msp_features = base_features + ["msp_gap_pct"]
    variants = [
        ("pooled_level", base_features, False, False),
        ("pooled_log_change", base_features, True, False),
        ("per_crop_level", base_features, False, True),
        ("per_crop_log_change", base_features, True, True),
        ("per_crop_msp_level", msp_features, False, True),
        ("per_crop_msp_log_change", msp_features, True, True),
    ]
    results = {}
    for name, features, log_target, per_crop in variants:
        logger.info("Running %s", name)
        results[name] = run_variant(name, train_set, calibration, scored, categories, features, log_target, per_crop)

    chosen = {}
    selected_parts = []
    for crop in sorted(samples["commodity"].unique()):
        scores = {name: calibration_score(result, crop) for name, result in results.items()}
        selected = max(scores, key=scores.get)
        chosen[crop] = {"variant": selected, "calibration_skill": scores}
        selected_parts.append(
            results[selected]["scored_eval"][
                results[selected]["scored_eval"]["commodity"] == crop
            ]
        )
    selected_scored = pd.concat(selected_parts, ignore_index=True)
    selected_backtest = backtest_table(
        selected_scored,
        selected_scored[["p10", "p50", "p90"]].to_numpy(),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selected_models = {}
    selected_features = {}
    for crop, choice in chosen.items():
        variant = results[choice["variant"]]
        features = variant["numeric_features"]
        fit_frame = samples if not variant["per_crop"] else samples[samples["commodity"] == crop]
        y = np.log(fit_frame["target"] / fit_frame["last_price"]) if variant["log_target"] else fit_frame["target"]
        selected_models[crop] = {
            q: _fit(make_X(fit_frame, features, categories), y, q, SEED)
            for q in QUANTILES
        }
        selected_features[crop] = features
    joblib.dump(
        {
            "models": selected_models,
            "categories": categories,
            "quantiles": QUANTILES,
            "selected_features": selected_features,
            "chosen": chosen,
            "data_hash": data_hash,
        },
        OUTPUT_DIR / "pilot_v3_selected_models.joblib",
    )
    daily.to_csv(OUTPUT_DIR / "pilot_v3_history.csv.gz", index=False, compression="gzip")
    model_payload = {
        "data_hash": data_hash,
        "features": CATEGORICAL_FEATURES + msp_features,
        "model_file": "pilot_v3_selected_models.joblib",
        "history_file": "pilot_v3_history.csv.gz",
        "selection_rule": "For each crop, maximize calibration-half 7/30-day pooled absolute skill; scored half is not used.",
        "chosen": chosen,
        "note": "Experimental offline artifact; MSP feature availability is season-dependent.",
    }
    (OUTPUT_DIR / "SELECTION.json").write_text(json.dumps(model_payload, indent=2, default=str) + "\n")
    metrics = {
        "dataset": str(DATA_DIR),
        "manifest": str(DATA_DIR / "MANIFEST.json"),
        "msp_manifest": str(DATA_DIR / "MSP_MANIFEST.json"),
        "data_hash": data_hash,
        "rows": {"daily": len(daily), "samples": len(samples), "train": len(train_set),
                 "calibration": len(calibration), "scored": len(scored)},
        "split": {"cutoff": str(cutoff.date()), "calibration_through": str(calibration["as_of"].max().date()),
                  "scored_from": str(midpoint.date()), "holdout_days": HOLDOUT_DAYS, "step_days": STEP_DAYS},
        "selection": model_payload["selection_rule"],
        "chosen": chosen,
        "selected_backtest": selected_backtest,
        "variants": {name: serialise_result(result) for name, result in results.items()},
    }
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(metrics, indent=2, default=str) + "\n")
    logger.info("Wrote %s and %s", RESULTS_PATH, OUTPUT_DIR / "SELECTION.json")
    print(json.dumps({"data_hash": data_hash, "selected_backtest": selected_backtest, "chosen": chosen}, indent=2, default=str))


if __name__ == "__main__":
    main()
