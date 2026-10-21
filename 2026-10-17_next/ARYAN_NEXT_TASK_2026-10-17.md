# Aryan: your 16 October release, the next dataset, and the next task

From: Hemanth (AIAIC backend), 17 October 2026.
Repository reviewed: `BHIV-Engineering-Exchange/bhiv-aiaic`, `main` at `6d577ac` ("Version forecast release bundle",
16 Oct), with `a79e2ac` (the Colab notebook) and `aiaic-forecast` at `fc64e27` (the experiment runner and its log).

---

## 1. In short

1. **Your pilot release is a real step forward.** Trained on the pilot dataset of 2 Oct (`data_hash 97c62dd7…`):
   - overall skill against "the price stays where it is" is **+0.074** (it was -0.157);
   - the p10-p90 range held **86%** of outcomes;
   - **14 of 16 groups are usable**, tur included in both states;
   - soybean is no longer worse than persistence anywhere.
2. **The two groups still not usable are onion in Maharashtra, 7 and 30 days: the Nashik pilot.**
3. **Your own experiment log shows how to fix it.** With one model per crop (`exp2_per_crop_level`,
   `exp2b_per_crop_log_change`), onion in Maharashtra is usable at 7, 14 and 30 days (skill +0.09 to +0.15). The
   model you released is not usable there. It is better overall (+0.074, against -0.132 and -0.069 for the two
   per-crop variants), so the answer is not one or the other: **choose per crop, on the calibration half, never on
   the scored half** (task A below).
4. **The next dataset is ready**: the Nashik district's next crops, maize, gram (chana), bajra and tomato, in both
   states, 2001 to 26 Aug 2026, with MSP for the three that have one (section 3).
5. **Two things to fix in how the release is shipped** (section 2).

---

## 2. What was checked in your repository

| Check | Result |
|---|---|
| `models/release/RELEASE.json` | trained through 2026-08-26, backtest cutoff 2026-02-27, 18,719 scored cases, `usable_groups` 14 of 16, smoke test answered (onion, Madhya Pradesh, "A lot", 7 days: p10 1,742, p50 2,504, p90 4,217) |
| `experiments/experiment_log.json` | nine variants with full tables: control, log-change, per crop (level and log-change), MSP (level and log-change), history from 2016, stronger baselines, calibration-selected |
| Model files | **committed into git** (`models/release/forecast_…_97c62dd7.joblib` 5.4 MB and its history 0.9 MB). The decision of 2 Oct (section 4 there, item A8): model files go on a **GitHub release** as assets with `SHA256SUMS`, not into git history. Each new model adds megabytes to every clone, forever. |
| A tag | none found for this release. A release needs a tag (`forecast-v2026.10.16`, for example) and its assets, so that a deployment can name exactly what it runs. |

**Not checked yet:** AIAIC's integration check against a running image of this release. Send me the tag when it
exists, and I will run it.

**On the AIAIC side, said plainly:** AIAIC's pipeline changed on 13 October (one pipeline, `cicd.yml`). It no longer
has the job that built your release into the `aiaic-ml` image, so there is no deployment path for your service on the
VM today. That is AIAIC's to fix (it is with Alay), not yours. Until then your service runs where you run it, and
AIAIC's dashboard shows ranges only when its `ML_PRICE_API_URL` reaches one.

---

## 3. The next dataset: `pilot_v3_nashik_next` (sent separately, about 18 MB; not committed anywhere)

Exported by AIAIC on 17 Oct 2026 from the same official source as before: data.gov.in "Variety-wise Daily Market
Prices" (Agmarknet; Government Open Data Licence - India). The counts below come from its `MANIFEST.json`.

| File | What |
|---|---|
| `agmarknet_pilot_mh_mp_2001_2026.csv.gz` | 1,766,755 rows, 2001-02-08 to 2026-08-26; maize, gram, bajra, tomato; Maharashtra and Madhya Pradesh. Same columns as pilot_v2 (`mandi`, `crop`, `source_file_id`, `source_line`). |
| `mandi_identity.csv` | every (state, published market, district label) → `mandi`: 771 as published, 630 "X APMC" names folded, 1 name that is two mandis |
| `sources.csv` | the raw file behind each `source_file_id`, with its sha256 |
| `MANIFEST.json` | counts per state and crop, spellings, columns, limits, the sha256 of each file |
| `msp_pilot_crops.csv` + `MSP_MANIFEST.json` | CACP MSP by marketing season for maize, gram and bajra. Maize and bajra: 2018-19 to 2023-24, 2025-26 and 2026-27 (**2024-25 missing**). Gram (a rabi crop, labelled by Rabi Marketing Season): 2019-20 to 2026-27. Tomato has no MSP. |

| State | Crop | Rows | First | Last | Published names → mandis |
|---|---|---|---|---|---|
| Madhya Pradesh | bajra | 50,742 | 2002-04-11 | 2026-08-25 | 195 → 160 |
| Madhya Pradesh | gram | 284,709 | 2002-06-05 | 2026-08-26 | 515 → 275 |
| Madhya Pradesh | maize | 354,263 | 2002-02-26 | 2026-08-26 | 506 → 291 |
| Madhya Pradesh | tomato | 109,359 | 2002-03-21 | 2026-08-26 | 220 → 153 |
| Maharashtra | bajra | 207,728 | 2001-06-02 | 2026-08-25 | 368 → 259 |
| Maharashtra | gram | 394,364 | 2001-05-26 | 2026-08-26 | 585 → 356 |
| Maharashtra | maize | 197,351 | 2001-04-03 | 2026-08-26 | 389 → 251 |
| Maharashtra | tomato | 168,239 | 2001-02-08 | 2026-08-26 | 157 → 105 |

Why these four: the Nashik district's first six crops in AIAIC (ordered by how widely its own mandis trade them) are
onion, maize, soybean, wheat, gram and bajra; maize, gram and bajra are the three the pilot does not have. Tomato is
the largest vegetable after onion that the horticulture tables list for Nashik (11.56 thousand ha in 2016-17), and the
hardest test of a forecast: a perishable with large swings and no MSP.

Read before training (as for pilot_v2):
- Key series on `mandi` and `crop`. Take one price per crop, mandi and day (the median); never sum or count rows (in
  a renaming week the same quote is published under both names).
- **Milled dal is left out.** Agmarknet's "Bengal Gram Dal (Chana Dal)" is a processed product with its own price;
  4,981 such rows were dropped (the manifest's `counts` says so). Only "Bengal Gram (Gram)(Whole)" is gram.
- Join an MSP to a price date by the marketing season the date falls in (kharif seasons open in October; the Rabi
  Marketing Season in April).
- Prices are nominal rupees. Coverage is thin in the early years.

**Also in this folder, a DRAFT from Gauri: `ONION_EVENTS_TIMELINE_V4_DRAFT_from_Gauri.xlsx`.** Dated onion export
policy events, 2013 to 2025: minimum export prices, three export bans, export duties. It also has two weather events.
They are candidate features for onion (for example "an export ban is in force on the forecast's date"). Use them with her
own cautions:
- five of the seven notification numbers were read on mirror sites, not yet on dgft.gov.in or cbic.gov.in;
- an event's date lining up with a price move is not its cause, and nothing may say it is;
- no events are listed before 2013 or for 2014 to 2016, so a feature built from them is missing in those years;
- test it like the other experiments: chosen on the calibration half, judged on the untouched scored half.

**Not available yet:**
- **Arrivals** (tonnes traded). AIAIC still holds no arrivals history.
- **Prices after 26 Aug 2026.** The download this is read from ends there. Newer days are fetched daily on AIAIC's
  server since 7 October. A refresh of both datasets through the newest day will be exported from there, with its
  own manifest.

---

## 4. Your next task (in order), with acceptance criteria

**A. Choose per crop, and make the Nashik pilot usable** (on the pilot_v2 data you already have).
- Do the selection per crop: pooled or per crop, level or log-change, with or without MSP. Choose on the
  CALIBRATION half (or an earlier fold); report on the untouched scored half.
- Accept when the release notes carry the full 16-group table, and onion in Maharashtra is either usable at 7 and 30
  days or abstains with its numbers. The selection rule must be written down before the scored half is read.

**B. Train the four new crops** on `pilot_v3_nashik_next`, with the same walk-forward split rules.
- 16 more groups (4 crops × 2 states × 7 and 30 days), MSP features for maize, gram and bajra.
- Accept when every group is in the metadata with its table, good or bad. A group that does not beat persistence, or
  whose range holds under 70%, abstains with its reason. Tomato is expected to be hard; say what you find.

**C. Ship it as a release, not as commits.**
- Remove the model files from the repository going forward (keep `.gitignore` covering `models/release/*.joblib`
  and the history files).
- Tag (`forecast-vYYYY.MM.DD`) and attach the three model files, `SHA256SUMS` and `RELEASE.json` as release assets.
- Accept when a clean clone holds no model file, the release has its assets and notes, and the image built from the
  tag reports ready only with its model.

**D. Evidence and replay** (still open from 2 Oct, section 5, D and E): fill `evidence_packet/review_packet.md`,
record each `/v1/forecast` call outside `evidence_packet/`, and add a replay that reproduces p10, p50 and p90 exactly.

**Not in scope:** any change to the AIAIC repository, anything shown to farmers. A forecast stays on the operator
dashboard, labelled experimental, until the owner and an agronomist decide otherwise.
