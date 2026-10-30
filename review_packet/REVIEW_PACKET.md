# REVIEW PACKET - AIAIC ML Foundation

**Candidate:** Aryan Sawant
**Test:** Test 1 - ML Foundation & Prediction Service
**Current release:** `forecast-v2026.10.17`
**Status:** Complete for the current experimental forecast release

## Quick summary

The repository now serves `/v1/forecast`, not the retired leaky `/v1/predict`.
The current bundle was retrained from pilot_v2, evaluated through walk-forward
backtesting, and compared across 5 experiment families. The release has 16
crop/state/horizon groups, 14 usable groups, overall skill `0.0739` versus
persistence, and 86.31% calibrated p10-p90 coverage over 18,719 cases.

| Check | Current result |
|---|---|
| Source data | Agmarknet via data.gov.in; pilot_v2 export |
| Crops | onion, soybean, tur, wheat |
| States | Maharashtra, Madhya Pradesh |
| Horizons | 7 and 30 days |
| Release tag | `forecast-v2026.10.17` |
| API | `/v1/forecast`, `/v1/forecast/meta`, `/v1/ready` |
| Response additions | `confidence`, `direction`, `known_unknowns`, audit/source fields |
| Tests | **70 passed** |
| Retired API | `/v1/predict` returns 410 |

## Evidence index

| Path | Contents |
|---|---|
| `../models/release/RELEASE.json` | Release metadata and full backtest |
| `../models/release/SHA256SUMS` | Bundle checksums |
| `../experiments/` | pilot_v2 experiment log and run evidence |
| `../evidence_packet/` | Release smoke and replay evidence |
| `Executive_Assessment.md` | Current assessment |
| `Assignment_vs_Delivery.md` | Requirement mapping |
| `known_limitations.md` | Current limitations and assumptions |

## Key release numbers

- Model: `forecast_20261016T062532Z_97c62dd7.joblib`
- Trained through: 2026-08-26
- Scored cases: 18,719
- Model MAE: Rs 267.83/q
- Persistence MAE: Rs 289.21/q
- Skill versus persistence: `0.0739`
- Calibrated p10-p90 coverage: `86.31%`
- Usable groups: 14 of 16

The old `/v1/predict` R2, MAE, and MAPE figures are leakage-contaminated
historical figures and are not current model-quality claims.
