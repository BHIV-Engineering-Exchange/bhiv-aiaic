# DEP.md — Deployment and Execution Protocol
# AIAIC ML Foundation — the mandi price FORECAST service

**Owner:** Aryan Sawant
**Consumer:** AIAIC (`/ml/price-model`, `/ml/price-range`): an experimental range on the operator dashboard, never
an input to any AIAIC decision.
**Revised:** 2026-10-02 (the forecast is the served model; `/v1/predict` is retired)

---

## 1. What runs

| Piece | Where | What it does |
|---|---|---|
| `src/forecast/train.py` | training machine | three LightGBM quantile models (p10, p50, p90), a walk-forward backtest against "the price stays where it is", split-conformal widening |
| `scripts/package_forecast_release.py` | training machine | copies ONE trained bundle to `models/release/`, writes `SHA256SUMS` and `RELEASE.json`, and proves the release answers |
| `docker/Dockerfile` | CI / VM | an image that carries its model; the build fails if a checksum does not match |
| `src/api/main.py` | container | `GET /v1/health`, `GET /v1/ready`, `GET /v1/forecast/meta`, `POST /v1/forecast` |

`/v1/predict` answers **410 Gone**. It read the same day's min and max price to predict that day's modal price, so
it described a price already known; its R² of 0.99 measured that, not forecasting skill. Its Test 1 code stays in
`src/training` and `src/inference`; nothing serves it.

---

## 2. Data

The training input is AIAIC's export of the official Agmarknet daily prices (data.gov.in "Variety-wise Daily Market
Prices", Government Open Data Licence - India). Read its `MANIFEST.json` first: it gives the period, the row counts,
the sha256 of every file, and what is NOT in the data (arrivals).

Key a series on the `mandi` and `crop` columns, not `market` and `commodity`. Agmarknet renamed about 700 mandis
"X" -> "X APMC" in November 2025 ("Indore" ends 2025-11-04, "Indore APMC" starts 2025-11-01). On the published name,
one mandi is two short series, and the dead name is listed as a mandi the model can forecast. `mandi_identity.csv`
lists every fold. Agmarknet also spelled whole tur three ways since 2001 ("Arhar (Tur/Red Gram)(Whole)" to
2025-11-04, "Arhar(Tur/Red Gram)(Whole)" 2025-11-01 to 2026-05-06, "Red gram/Arhar/Tur(whole)" from 2026-04-01);
`crop` is `tur` for all three. In a renaming week the SAME quote is published under both names; the trainer takes
one price per crop, mandi and day (the median), so the duplicates never count twice.

---

## 3. Train, package, check

```bash
pip install -r requirements.txt

# 1. train (about 7 minutes for the 2016-2026 crops on a laptop; longer from 2001)
python -m src.forecast.train --input agmarknet_pilot_mh_mp_2001_2026.csv.gz --market-column mandi \
    --commodity-column crop --crops onion soybean tur wheat --horizons 7 30 --holdout-days 180 \
    --step-days 7 --seed 42 --models-dir models/forecast_pilot \
    --source "Agmarknet via data.gov.in, AIAIC pilot export 2026-10-02 (sha256 <from MANIFEST.json>)"

# 2. package one bundle as the release, with checksums, and prove it answers
python scripts/package_forecast_release.py --from models/forecast_pilot

# 3. the tests (no model needed: they train on SYNTHETIC data in a temporary directory)
pytest tests -q
```

The backtest is the number that matters. `skill_vs_persistence = 1 - MAE(p50) / MAE(last price)`. Above 0, the
model beat "the price stays where it is". A crop, state and horizon is **usable** only with at least 20 scored cases,
skill above 0, and a p10-p90 range that held at least 70% of outcomes. Everything else abstains (`no_skill`).

---

## 4. Release (what to do once a model is built)

1. `python scripts/package_forecast_release.py --from <models dir>`: `models/release/` now holds the bundle,
   `SHA256SUMS` and `RELEASE.json`.
2. Create a GitHub release in this repository, tag `forecast-vYYYY.MM.DD`, and attach every file of
   `models/release/`. Paste `RELEASE.json`'s `backtest_overall` and `usable_groups` into the release notes, good or
   bad. The current served bundle is also committed under `models/release/`; raw training exports remain outside git.
3. Build and run the image locally, and check it (section 5).
4. Tell Hemanth (AIAIC backend) and Kaushlendra (CI/CD, VM) the tag. Kaushlendra's pipeline builds the image from
   that tag's assets and runs it beside AIAIC. Hemanth runs AIAIC's integration check against it. **Nobody copies
   model files into the AIAIC repository**, and AIAIC changes no code for a new model while the contract holds.

---

## 5. Docker

```bash
python scripts/package_forecast_release.py --from models/forecast_pilot   # if models/release/ is empty
docker compose -f docker/docker-compose.yml up --build -d
curl http://127.0.0.1:8040/v1/ready          # {"ready": true, "reason": null}
curl http://127.0.0.1:8040/v1/forecast/meta  # names the model in RELEASE.json
docker compose -f docker/docker-compose.yml down
```

- Port 8040 is bound to 127.0.0.1. On the AIAIC VM the service publishes no port: AIAIC reaches it on the shared
  docker network as `http://aiaic-ml:8000` (`ML_PRICE_API_URL`).
- The HEALTHCHECK calls `/v1/ready`, which is 200 only when the forecast model is loaded.
- MongoDB is optional and off (`--profile mongo`). The service never uses it.

---

## 6. API contract (what AIAIC calls)

`POST /v1/forecast`

```json
{"commodity": "Wheat", "state": "Madhya Pradesh", "market": "Indore", "as_of": "2026-08-26", "horizon_days": 30}
```

Optional: `history`, up to 400 `{"date", "modal_price"}` points for this mandi, oldest first, to forecast from a day
later than the model's data.

The answer is a range with its backtest, or an abstention with a reason, never a guess:

| `reason_code` | Meaning |
|---|---|
| `not_trained_for` | the crop, state or mandi is not in the model's closed lists |
| `horizon_not_trained` | the horizon is not one the model was trained for |
| `in_sample` | the target day is inside the training data, so a range would describe a known price |
| `as_of_outside_history` | `as_of` is more than 7 days after the model's data and no `history` was sent |
| `no_skill` | the backtest did not beat persistence for this crop, state and horizon, or its range held under 70% |
| `no_recent_price` | the mandi has no quote in the 7 days before `as_of` |

`GET /v1/forecast/meta`: the model file, data, dates, closed lists (crops, states, mandis), `market_last_quote` (the
last day each mandi quoted), the features it reads and the ones it never reads (`min_price`, `max_price`, any price
after `as_of`), and the full backtest.

---

## 7. The current official model (2026-10-02), from its metadata

`forecast_20261002T064450Z_9595e12e`: Agmarknet 2016-2026 (AIAIC export), Onion, Soyabean, Wheat; trained through
2026-08-26; walk-forward cutoff 2026-02-27; scored from 2026-05-23.

Overall: **skill -0.157** against persistence (p50 error Rs 307.51/q, persistence Rs 265.72/q), p10-p90 held 81.5%.
**4 of 12 groups are usable:**

| Crop | State | Days | Cases | p50 error (Rs/q) | Persistence error | Skill | p10-p90 held | Usable |
|---|---|---|---|---|---|---|---|---|
| Onion | Madhya Pradesh | 7 | 765 | 306 | 320 | +0.042 | 70% | yes |
| Onion | Madhya Pradesh | 30 | 528 | 501 | 543 | +0.079 | 65% | no |
| Onion | Maharashtra | 7 | 1,041 | 304 | 292 | -0.043 | 71% | no |
| Onion | Maharashtra | 30 | 741 | 624 | 635 | +0.018 | 52% | no |
| Soyabean | Madhya Pradesh | 7 | 1,667 | 522 | 388 | -0.345 | 81% | no |
| Soyabean | Madhya Pradesh | 30 | 1,147 | 683 | 507 | -0.347 | 85% | no |
| Soyabean | Maharashtra | 7 | 1,223 | 535 | 360 | -0.485 | 80% | no |
| Soyabean | Maharashtra | 30 | 896 | 695 | 501 | -0.386 | 81% | no |
| Wheat | Madhya Pradesh | 7 | 3,075 | 69 | 74 | +0.074 | 85% | yes |
| Wheat | Madhya Pradesh | 30 | 2,176 | 90 | 128 | +0.300 | 94% | yes |
| Wheat | Maharashtra | 7 | 1,504 | 85 | 82 | -0.041 | 83% | no |
| Wheat | Maharashtra | 30 | 1,109 | 123 | 146 | +0.157 | 90% | yes |

Update this section from `RELEASE.json` with every release. Never quote the retired `/v1/predict` figures (MAE 44.07,
MAPE 4.52%, R² 0.9946) as model quality.

---

## 8. Failure behaviour

| Scenario | Behaviour |
|---|---|
| No model in the models directory | `/v1/health` 200 `degraded`; `/v1/ready` 503; `/v1/forecast` 503 naming the train command; the container is unhealthy |
| A release file missing or altered | the Docker build fails at `sha256sum -c` |
| Invalid input (empty mandi, horizon outside 1-90, a date not YYYY-MM-DD) | 422 with Pydantic's detail |
| Untrained crop, mandi or horizon; no recent quote; no skill | 200 with `abstain: true`, a `reason_code` and the backtest row |
| `/v1/predict` | 410 Gone, pointing to `/v1/forecast` |
| Unhandled exception | 500 with clean JSON; traceback logged server-side |

---

## 9. Known limitations

- Overall the 2026-10-02 model does **not** beat persistence (skill -0.157); only 4 of 12 groups answer.
  Soybean is worse than persistence in both states.
- No arrivals: AIAIC holds no arrivals history (the daily price files carry no arrivals column).
- Prices are nominal rupees.
- One pooled model for every crop and state; a new model needs a service restart (no hot reload).
- The served models are refitted on all samples; the backtest records the procedure, not those exact fitted models.
- Forecast calls are not yet written to a replay record (Test 1's replay covered only `/v1/predict`).
