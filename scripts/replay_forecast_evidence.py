"""Record and replay /v1/forecast calls for release evidence.

Usage:
    python scripts/replay_forecast_evidence.py
    python scripts/replay_forecast_evidence.py --base-url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "review_packet" / "runtime_logs"
DEFAULT_REQUEST = {
    "commodity": "onion",
    "state": "Madhya Pradesh",
    "market": "A lot",
    "as_of": "2026-08-26",
    "horizon_days": 7,
}
QUANTILES = ("p10", "p50", "p90")


def post_forecast(base_url: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    response = requests.post(
        f"{base_url.rstrip('/')}/v1/forecast",
        json=body,
        timeout=30,
    )
    response.raise_for_status()
    return response.status_code, response.json()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--request", dest="request_path", type=Path)
    args = parser.parse_args()

    body = DEFAULT_REQUEST
    if args.request_path:
        body = json.loads(args.request_path.read_text(encoding="utf-8"))

    recorded_status, original = post_forecast(args.base_url, body)
    replay_status, replayed = post_forecast(args.base_url, body)
    original_model = original.get("model") or {}
    replay_model = replayed.get("model") or {}
    quantile_checks = {
        key: original.get(key) == replayed.get(key) for key in QUANTILES
    }
    deterministic = all(quantile_checks.values())
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = OUTPUT_DIR / f"forecast_replay_{stamp}.json"
    record = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "endpoint": f"{args.base_url.rstrip('/')}/v1/forecast",
        "request_id": original.get("request_id"),
        "model_file": original_model.get("file"),
        "data_hash": original_model.get("data_hash"),
        "request_body": body,
        "response_status": recorded_status,
        "response": original,
        "replay_request_id": replayed.get("request_id"),
        "replay_model_file": replay_model.get("file"),
        "replay_data_hash": replay_model.get("data_hash"),
        "replay_response_status": replay_status,
        "replay_response": replayed,
        "quantile_checks": quantile_checks,
        "deterministic": deterministic,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output_path),
        "request_id": record["request_id"],
        "replay_request_id": record["replay_request_id"],
        "model_file": record["model_file"],
        "data_hash": record["data_hash"],
        "quantile_checks": quantile_checks,
        "deterministic": deterministic,
    }, indent=2))
    return 0 if deterministic else 1


if __name__ == "__main__":
    raise SystemExit(main())
