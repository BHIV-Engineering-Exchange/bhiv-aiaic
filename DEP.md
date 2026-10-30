# DEP.md - Deployment and Execution Protocol

## Release status

The current served bundle is `forecast-v2026.10.17`, model
`forecast_20261016T062532Z_97c62dd7.joblib`. It was retrained from the pilot_v2
Agmarknet export, trained through 2026-08-26, and evaluated with 5 experiment
families before the release choice. It is experimental and is never an input
to an AIAIC decision.

The release backtest covers 18,719 cases: model MAE Rs 267.83/q versus
persistence MAE Rs 289.21/q, skill `0.0739`, and calibrated p10-p90 coverage
86.31%. Fourteen of 16 groups are usable. The two non-usable groups are onion
in Maharashtra at 7 and 30 days; the service abstains for them.

## 1. Components

| Component | Role |
|---|---|
| `src/forecast/train.py` | Quantile models, walk-forward split, baseline and calibration |
| `scripts/package_forecast_release.py` | Creates the bundle, checksums, metadata and smoke test |
| `models/release/` | Current model assets, `RELEASE.json`, and `SHA256SUMS` |
| `src/api/main.py` | FastAPI application |
| `src/api/forecast_routes.py` | `/v1/forecast` and `/v1/forecast/meta` |

`/v1/predict` is retired and returns 410. Its old same-day min/max design
leaked information about the modal-price target, so its old metrics are not
forecast-quality evidence.

## 2. Data and closed lists

The source is AIAIC's export of the official Agmarknet daily market prices
from data.gov.in. The model keys series on the normalized `mandi` and `crop`
identities and takes one price per crop, mandi and day across Agmarknet
renames. The release supports onion, soybean, tur and wheat in Maharashtra and
Madhya Pradesh, with 7- and 30-day horizons.

The model reads only history available on or before `as_of`: recent modal
price, lag/rolling features, calendar information, and market identity. It
does not read same-day min/max or any price after `as_of`.

## 3. Train, package and test

```bash
pip install -r requirements.txt
python -m src.forecast.train --help
python scripts/package_forecast_release.py --from <models-dir>
python -m pytest tests -q
```

The current verification result is **70 passed**. Training requires the
appropriate Agmarknet export and writes model metadata with the data hash,
closed lists, backtest, and calibration information.

## 4. Release procedure

1. Train and inspect the complete backtest table.
2. Package one bundle with `package_forecast_release.py`.
3. Verify `RELEASE.json`, `SHA256SUMS`, `/v1/ready`, and
   `/v1/forecast/meta`.
4. Create a dated tag in the form `forecast-vYYYY.MM.DD` and attach the
   release assets. The current tag is `forecast-v2026.10.17`.
5. Tell the AIAIC backend and deployment owner the exact tag. Model files are
   not copied into the AIAIC repository.

## 5. Docker

```bash
docker compose -f docker/docker-compose.yml up --build -d
curl http://127.0.0.1:8040/v1/ready
curl http://127.0.0.1:8040/v1/forecast/meta
docker compose -f docker/docker-compose.yml down
```

Port 8040 is local-only. On the AIAIC VM the shared Docker network uses
`http://aiaic-ml:8000`. MongoDB is optional and is not used by the forecast
service.

## 6. API contract

```http
POST /v1/forecast
Content-Type: application/json
```

```json
{"commodity":"onion","state":"Madhya Pradesh","market":"A lot","as_of":"2026-08-26","horizon_days":7}
```

The response contains the audit fields `request_id`, `execution_status`,
`decision_id`, `generated_at`, `engine_version`, `model`, `backtest`,
`sources_used`, `known_unknowns`, `read_these_first`, `confidence`, and
`direction`, plus `target_date`, `last_price`, `expected_change_pct`, and
`p10`/`p50`/`p90` when a forecast is usable. Abstentions return
`abstain: true`, `reason_code`, `reason`, and null quantiles. Common reason
codes are `not_trained_for`, `horizon_not_trained`, `in_sample`,
`as_of_outside_history`, `no_skill`, `no_recent_price`, `STALE_PRICE`, and
`SYNTHETIC_INPUT`.

`GET /v1/forecast/meta` exposes the model file, source, dates, closed lists,
market quote dates, features, and the full backtest.

## 7. Evidence and limitations

`experiments/` contains the pilot_v2 experiment log and 5 experiment families;
selection is made on calibration data and reported on untouched scored data.
`evidence_packet/` contains release and replay evidence. The forecast remains
experimental, has no automated retraining trigger, and depends on the
freshness and identity quality of Agmarknet data. See
`review_packet/known_limitations.md`.
