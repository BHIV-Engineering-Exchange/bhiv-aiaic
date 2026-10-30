# Executive Assessment

**Current release:** `forecast-v2026.10.17`
**Model:** `forecast_20261016T062532Z_97c62dd7.joblib`

## What was built

This repository provides an experimental, leakage-safe mandi price forecast
service. The pilot_v2 data was retrained and evaluated with walk-forward
backtesting. Five experiment families were compared, with selection isolated
from the untouched scored half. The release serves calibrated p10/p50/p90
ranges and abstains when its evidence is insufficient.

## Assessment

**Data and reproducibility - PASS.** The model uses the Agmarknet export,
records its data hash and packages checksummed release assets.

**Forecast validity - PASS.** Features are derived only from prices available
on or before `as_of`. The retired `/v1/predict` endpoint is 410 and is not
part of the served contract.

**Evaluation - PASS with honest limits.** The release scores 18,719 cases,
with model MAE Rs 267.83/q versus persistence Rs 289.21/q, skill `0.0739`,
and 86.31% calibrated coverage. Fourteen of 16 groups are usable; the two
onion/Maharashtra groups abstain.

**API and operations - PASS.** `/v1/forecast`, `/v1/forecast/meta`,
`/v1/ready`, and `/v1/health` are implemented. Responses carry
`confidence`, `direction`, `known_unknowns`, model/backtest provenance, and
source freshness. Docker packaging verifies the release bundle.

**Verification - PASS.** The current suite reports **70 passed**. The only
reported results are existing dependency/deprecation and date-parsing
warnings.

## Integration boundary

AIAIC consumes the service through `ML_PRICE_API_URL` and identifies model
assets by the release tag. The service remains labelled experimental and is
not used for decisions. No TANTRA, reasoning engine, governance, runtime
contract, PIG, or Bucket code was modified.

## Remaining gaps

- No automated retraining or hot model reload exists.
- Agmarknet live API ingestion is not the deployed training path; releases use
  an exported, hashed dataset.
- Docker was packaged and release-smoke-tested, but the local development
  machine does not provide a full production VM integration test.
- Forecast quality remains crop/state/market dependent; usable-group status is
  the gate, not a guarantee of accuracy.

The historical same-day `/v1/predict` R2/MAE/MAPE numbers are leakage-contaminated
and are intentionally excluded from this assessment.
