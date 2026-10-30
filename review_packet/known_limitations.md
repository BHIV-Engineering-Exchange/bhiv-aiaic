# Known Limitations and Assumptions

**Current release:** `forecast-v2026.10.17`
**Training data:** pilot_v2 Agmarknet export, through 2026-08-26

## Data assumptions

- Prices are treated as nominal INR per quintal modal prices.
- The model uses normalized `mandi` and `crop` identities to join series
  across Agmarknet market renames and takes one price per crop, mandi and day.
- The release is closed to onion, soybean, tur and wheat in Maharashtra and
  Madhya Pradesh. Other crops, states, or mandis abstain as
  `not_trained_for`.
- The source is an export from data.gov.in. It does not provide arrivals
  history, and source freshness depends on a later export and release.
- Early history can be sparse. A mandi with no recent quote abstains rather
  than receiving a synthetic value.

## Model and evaluation limitations

- The release supports only 7- and 30-day horizons.
- A group is usable only with at least 20 scored cases, positive skill against
  persistence, and p10-p90 coverage of at least 70%. The service therefore
  abstains for the two onion/Maharashtra groups.
- Backtest results are aggregate evidence, not a promise for an individual
  market or future regime.
- Five experiment families were evaluated on pilot_v2. Variant choice is
  based on calibration data; final numbers come from untouched scored data.
- The release uses historical prices and has no arrivals, weather, event, or
  other causal market features.
- There is no automated retraining trigger. A new model must be trained,
  packaged, checked, and released under a new tag.

## API and deployment limitations

- The model bundle is loaded and cached by the service; replacing it requires
  a restart. There is no hot `/reload` endpoint.
- Caller-supplied `history` is rejected in production as `SYNTHETIC_INPUT`;
  the service uses its stored release history.
- A request more than 7 days beyond the model history without an accepted
  current release abstains. Prices more than 30 days old trigger
  `STALE_PRICE`.
- `confidence` and `direction` are advisory summaries derived from the
  backtest and forecast range. `known_unknowns` is an explicit uncertainty
  list; none of these fields is a decision recommendation.
- The release is Docker-packaged and smoke-tested, but a full AIAIC VM
  integration depends on the consuming deployment and network.
- MongoDB is optional and is not used by the current forecast service.

## Retired endpoint

`/v1/predict` is permanently retired with 410. Its same-day min/max inputs
leaked the same-day modal-price target, so its old R2, MAE, and MAPE figures
are not forecast-quality evidence and must not be presented as current
metrics.
