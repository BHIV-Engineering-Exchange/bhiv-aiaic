# Forecast release evidence packet

## Release

- Release: `forecast-v2026.10.17`
- Model: `forecast_20261016T062532Z_97c62dd7.joblib`
- Trained at: `2026-10-16T06:25:32Z`
- Trained through: `2026-08-26`
- Backtest cutoff: `2026-02-27`
- Data hash: `97c62dd7488d796fd29895aa54656f155032ad8c8ca276b829fc6ceba06854b0`
- Source: Agmarknet via data.gov.in, AIAIC pilot export 2026-10-02
- Horizons: 7 and 30 days

The release is experimental. It answers only for groups that beat persistence,
have at least 20 scored cases, and whose calibrated p10-p90 range covered at
least 70% of scored outcomes. Other requests abstain with a reason.

## Overall backtest

| Cases | Model MAE (Rs/q) | Persistence MAE (Rs/q) | Skill vs persistence | p10-p90 coverage |
|---:|---:|---:|---:|---:|
| 18,719 | 267.83 | 289.21 | 0.0739 | 86.31% |

Coverage before calibration was 81.80%.

## Usable groups

The release metadata contains 16 total groups; 14 are usable.

| Crop | State | Horizon | Cases | Skill | Coverage |
|---|---|---:|---:|---:|---:|
| Onion | Madhya Pradesh | 7 | 782 | 0.0238 | 78.39% |
| Onion | Madhya Pradesh | 30 | 563 | 0.0347 | 71.05% |
| Soybean | Madhya Pradesh | 7 | 1,591 | 0.1031 | 89.69% |
| Soybean | Madhya Pradesh | 30 | 1,171 | 0.0665 | 91.03% |
| Soybean | Maharashtra | 7 | 1,211 | 0.0588 | 85.38% |
| Soybean | Maharashtra | 30 | 920 | 0.0655 | 90.22% |
| Tur | Madhya Pradesh | 7 | 464 | 0.1406 | 80.82% |
| Tur | Madhya Pradesh | 30 | 330 | 0.0820 | 86.67% |
| Tur | Maharashtra | 7 | 1,163 | 0.0525 | 88.99% |
| Tur | Maharashtra | 30 | 833 | 0.0405 | 93.52% |
| Wheat | Madhya Pradesh | 7 | 2,997 | 0.1270 | 86.12% |
| Wheat | Madhya Pradesh | 30 | 2,225 | 0.2468 | 92.45% |
| Wheat | Maharashtra | 7 | 1,531 | 0.0424 | 87.46% |
| Wheat | Maharashtra | 30 | 1,148 | 0.1325 | 86.93% |

The two non-usable groups are onion/Maharashtra at 7 and 30 days. Their
backtest records remain in `RELEASE.json`; the service abstains for them.

## Release smoke test

The packaging smoke test loaded the release bundle and checked `/v1/ready`
and `/v1/forecast/meta`.

```json
{
  "ready": 200,
  "meta_model_file": "forecast_20261016T062532Z_97c62dd7.joblib",
  "forecast": {
    "commodity": "onion",
    "state": "Madhya Pradesh",
    "market": "A lot",
    "as_of": "2026-08-26",
    "horizon_days": 7,
    "target_date": "2026-09-02",
    "p10": 1742.09,
    "p50": 2503.87,
    "p90": 4217.38
  }
}
```

## Replay evidence

`scripts/replay_forecast_evidence.py` records each `/v1/forecast` request and
response in `review_packet/runtime_logs/`. Each record includes the original
request ID, model file, data hash, request body, response, replay request ID,
replay response, and exact equality checks for `p10`, `p50`, and `p90`.

The generated replay record is the runtime evidence for deterministic
quantile reproduction.
