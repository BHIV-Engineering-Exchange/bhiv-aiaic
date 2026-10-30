# AIAIC ML Foundation - mandi price forecast service

**Owner:** Aryan Sawant
**Current release:** `forecast-v2026.10.17`
**Status:** Experimental. AIAIC may show ranges on the operator dashboard, but
the forecast is not an input to an AIAIC decision.

## What this project does

This service forecasts the future modal mandi price for a crop, state, market,
and horizon. It uses official Agmarknet daily price exports from data.gov.in,
walk-forward evaluation, three LightGBM quantile models (p10, p50, p90), and
split-conformal range calibration. It refuses to present a forecast when the
requested group has not demonstrated skill against the persistence baseline,
has insufficient scored cases, or has coverage below 70%.

The current release is trained through 2026-08-26 on the pilot_v2 export. It
supports these crops and states:

| Crop | States |
|---|---|
| onion | Madhya Pradesh, Maharashtra |
| soybean | Madhya Pradesh, Maharashtra |
| tur | Madhya Pradesh, Maharashtra |
| wheat | Madhya Pradesh, Maharashtra |

Supported forecast horizons are 7 and 30 days. The release has 16 evaluated
crop/state/horizon groups; 14 are usable and two onion/Maharashtra groups
abstain under the release rules. Overall backtest skill versus persistence is
`0.0739`, with 86.31% p10-p90 coverage (18,719 scored cases).

## Run it

```bash
pip install -r requirements.txt
python -m pytest tests -q                 # 70 tests
python -m src.forecast.train --help       # training options
python scripts/package_forecast_release.py --from <models-dir>
docker compose -f docker/docker-compose.yml up --build -d
curl http://127.0.0.1:8040/v1/ready
```

The release bundle is in `models/release/` and is checked by
`RELEASE.json`/`SHA256SUMS`. For the complete training and release procedure,
see `DEP.md`.

## API

| Endpoint | Method | Purpose |
|---|---|---|
| `/v1/health` | GET | Liveness and model status |
| `/v1/ready` | GET | 200 only when the release model is loaded |
| `/v1/forecast/meta` | GET | Release metadata, closed lists, and backtest |
| `/v1/forecast` | POST | Forecast range or an explicit abstention |
| `/v1/predict` | any | 410 Gone; retired because it leaked same-day min/max |

Example request:

```bash
curl -X POST http://127.0.0.1:8040/v1/forecast \
  -H "Content-Type: application/json" \
  -d '{"commodity":"onion","state":"Madhya Pradesh","market":"A lot","as_of":"2026-08-26","horizon_days":7}'
```

Successful responses include `request_id`, `execution_status`, `decision_id`,
`generated_at`, `engine_version`, the request fields, `target_date`, `p10`,
`p50`, `p90`, `unit`, `last_price`, `expected_change_pct`, `direction`,
`confidence`, `backtest`, `model`, `known_unknowns`, `read_these_first`, and
`sources_used`. An abstention keeps the same audit fields and returns
`abstain: true`, `reason_code`, `reason`, and null quantiles. The model never
uses same-day min/max prices or prices after `as_of`.

## Integration and limitations

AIAIC calls the service through `ML_PRICE_API_URL` (normally
`http://aiaic-ml:8000` inside the Docker network). New model bundles are
identified by a dated release tag, not copied into the AIAIC repository.
`confidence`, `direction`, and `known_unknowns` are advisory response fields;
they do not turn an experimental forecast into a decision.

The retired Test 1 pipeline remains in `src/{ingestion,validation,preprocessing,
features,training,inference}` for historical review and is not served. No
changes were made to TANTRA, the reasoning engine, governance, runtime
contracts, PIG, or Bucket.
