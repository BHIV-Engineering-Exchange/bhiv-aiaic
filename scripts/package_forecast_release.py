"""Package ONE trained forecast as a release: the three files the service needs, their checksums, and a summary.

    python scripts/package_forecast_release.py --from models/forecast_pilot [--model forecast_<stamp>_<hash>.json]

Writes models/release/ (replacing what was there):
    forecast_<stamp>_<hash>.joblib / .json / _history.csv.gz    the bundle, byte for byte
    SHA256SUMS                                                  `sha256sum -c` format; the Docker build checks it
    RELEASE.json                                                model, data, backtest (overall + usable groups), files

Then it PROVES the release answers: it starts the API in-process on models/release and checks /v1/ready, that
/v1/forecast/meta names this model, and that one usable crop/state/horizon answers from the served history alone
(no caller-supplied prices), at a mandi that quoted in the last 7 days. It exits non-zero if any check fails.

Attach the files of models/release/ to a GitHub release (tag forecast-vYYYY.MM.DD); they are not committed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import date, datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RELEASE = REPO / "models" / "release"


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _smoke(meta: dict) -> dict:
    """The release, served in-process: ready, the right model, and one usable group answering from served history."""
    os.environ["AIAIC_MODELS_DIR"] = str(RELEASE)
    sys.path.insert(0, str(REPO))
    from fastapi.testclient import TestClient
    from src.api.main import app
    from src.config.settings import settings
    from src.forecast import service
    settings.models_dir = RELEASE
    service.reset_cache()
    c = TestClient(app)
    out = {"ready": c.get("/v1/ready").status_code}
    m = c.get("/v1/forecast/meta").json()
    out["meta_model_file"] = m.get("model_file")
    if out["ready"] != 200 or out["meta_model_file"] != meta["model_file"]:
        raise SystemExit(f"release does not serve its own model: {out}")
    usable = [g for g in meta["backtest"]["groups"] if g.get("usable")]
    if not usable:
        out["forecast"] = "no usable group: every forecast will abstain (NO_SKILL); say so in the release notes"
        return out
    as_of = meta["trained_through"]
    last = meta.get("market_last_quote") or {}
    for g in usable:
        key = f"{g['commodity']}|{g['state']}"
        recent = [mk for mk, d in sorted((last.get(key) or {}).items())
                  if (date.fromisoformat(as_of) - date.fromisoformat(d)).days <= 7] or meta["markets"][key]
        for market in recent[:40]:
            r = c.post("/v1/forecast", json={"commodity": g["commodity"], "state": g["state"], "market": market,
                                             "as_of": as_of, "horizon_days": g["horizon_days"]}).json()
            if r.get("reason_code") == "NO_RECENT_PRICE":
                continue
            if r.get("abstain") is not False or not (r["p10"] <= r["p50"] <= r["p90"]):
                raise SystemExit(f"a usable group did not answer: {r}")
            out["forecast"] = {k: r[k] for k in ("commodity", "state", "market", "as_of", "horizon_days",
                                                 "target_date", "p10", "p50", "p90")}
            return out
    raise SystemExit("no usable group had a mandi with a quote in the 7 days before trained_through")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--from", dest="src", required=True, help="the models directory the forecast was trained into")
    ap.add_argument("--model", default=None, help="its forecast_*.json (default: the newest in --from)")
    args = ap.parse_args(argv)
    src = Path(args.src)
    metas = sorted(src.glob("forecast_*.json"))
    meta_path = (src / args.model) if args.model else (metas[-1] if metas else None)
    if meta_path is None or not meta_path.is_file():
        raise SystemExit(f"no forecast_*.json in {src}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    files = [meta_path, src / meta["model_file"], src / meta["history_file"]]
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise SystemExit(f"the bundle is incomplete: {missing}")

    if RELEASE.exists():
        shutil.rmtree(RELEASE)
    RELEASE.mkdir(parents=True)
    for p in files:
        shutil.copy2(p, RELEASE / p.name)
    sums = {p.name: _sha256(RELEASE / p.name) for p in files}
    # Bytes, not text: on Windows write_text turns "\n" into "\r\n", and sha256sum then looks for "<name>\r".
    (RELEASE / "SHA256SUMS").write_bytes("".join(f"{h}  {n}\n" for n, h in sorted(sums.items())).encode("utf-8"))

    smoke = _smoke(meta)
    bt = meta["backtest"]
    release = {
        "model_file": meta["model_file"], "trained_at": meta["trained_at"], "trained_through": meta["trained_through"],
        "backtest_cutoff": meta["backtest_cutoff"], "data_source": meta["data_source"], "data_hash": meta["data_hash"],
        "market_identity": meta.get("market_identity"), "horizons_days": meta["horizons_days"],
        "backtest_overall": bt["overall"],
        "usable_groups": [{k: g[k] for k in ("commodity", "state", "horizon_days", "n", "skill_vs_persistence",
                                             "coverage_p10_p90")} for g in bt["groups"] if g.get("usable")],
        "groups_total": len(bt["groups"]),
        "files": {n: {"sha256": h, "bytes": (RELEASE / n).stat().st_size} for n, h in sorted(sums.items())},
        "smoke_test": smoke,
        "packaged_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "honesty": "Experimental. A forecast is answered only for usable groups (beat persistence, range held >= 70% "
                   "of outcomes, >= 20 cases); everything else abstains with a reason.",
    }
    (RELEASE / "RELEASE.json").write_bytes((json.dumps(release, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({k: release[k] for k in ("model_file", "trained_through", "backtest_overall", "usable_groups",
                                              "smoke_test")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
