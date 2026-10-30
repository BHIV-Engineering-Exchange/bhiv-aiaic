# Assignment vs Delivery

**Current release:** `forecast-v2026.10.17`
**Verification:** 70 tests passed

| Requirement | Delivered | Evidence |
|---|---|---|
| Real agricultural data | Yes | Agmarknet/data.gov.in pilot_v2 export and manifest |
| Deterministic, replayable processing | Yes | Data hashes, release checksums and replay evidence |
| Leakage-safe forecasting | Yes | Features stop at `as_of`; no same-day min/max or future price |
| Walk-forward evaluation | Yes | `src/forecast/train.py`, `RELEASE.json` |
| Baseline comparison | Yes | Persistence MAE and `skill_vs_persistence` |
| Experiment comparison | Yes | `experiments/` contains 5 experiment families and selection log |
| Release packaging | Yes | `models/release/RELEASE.json` and `SHA256SUMS` |
| Supported crop/state contract | Yes | onion, soybean, tur, wheat in Maharashtra and Madhya Pradesh |
| Forecast API | Yes | `POST /v1/forecast` |
| Metadata API | Yes | `GET /v1/forecast/meta` |
| Health/readiness | Yes | `GET /v1/health` and `GET /v1/ready` |
| Explicit abstention | Yes | Closed-list, horizon, skill, freshness and input checks |
| Explainable response | Yes | `confidence`, `direction`, `known_unknowns`, `sources_used` |
| Docker deployment | Yes | `docker/Dockerfile` and `docker/docker-compose.yml` |
| Tests | Yes | 70 passed in the current environment |
| Documentation | Yes | README, DEP and this review packet |
| Integration boundary | Yes | AIAIC consumes the tagged bundle over HTTP; no AIAIC code changed |

The old `/v1/predict` endpoint is intentionally retired with 410 because its
same-day min/max inputs leaked the target. Its old metrics must not be used as
forecast accuracy evidence.
