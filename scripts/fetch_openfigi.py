#!/usr/bin/env python3
"""Fetch resumable, raw OpenFIGI ticker mappings in unauthenticated batches."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.public_universe import parse_sp500_constituents


ENDPOINT = "https://api.openfigi.com/v3/mapping"


def mapping_payload(
    jobs: list[dict[str, str]], exchange_scope: str
) -> list[dict[str, str]]:
    """Build one OpenFIGI request with an explicit exchange policy."""

    if exchange_scope not in {"global", "us"}:
        raise ValueError(f"unsupported OpenFIGI exchange scope: {exchange_scope}")
    payload: list[dict[str, str]] = []
    for job in jobs:
        request = {
            "idType": "TICKER",
            "idValue": job["ticker"],
            "marketSecDes": "Equity",
        }
        if exchange_scope == "us":
            request["exchCode"] = "US"
        payload.append(request)
    return payload


def fetch(
    jobs: list[dict[str, str]], exchange_scope: str
) -> list[dict[str, object]]:
    """Submit one bounded OpenFIGI batch and decode its response."""

    payload = mapping_payload(jobs, exchange_scope)
    result = subprocess.run(
        [
            "curl",
            "-sS",
            "--fail-with-body",
            "--retry",
            "3",
            "-X",
            "POST",
            "-H",
            "Content-Type: application/json",
            "-A",
            "dyadic-dml-finance-research/1.0",
            "--data-binary",
            "@-",
            ENDPOINT,
        ],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=True,
        timeout=60,
    )
    return json.loads(result.stdout)


def main() -> None:
    """Fetch resumable OpenFIGI batches for the pinned index universe."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--universe-html", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--pause", type=float, default=3.0)
    parser.add_argument(
        "--exchange-scope",
        choices=["global", "us"],
        default="global",
        help=(
            "Query globally by default because share-class FIGIs are global; "
            "the downstream point-in-time SEC bridge supplies the U.S. gate"
        ),
    )
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 10:
        raise ValueError("the unauthenticated OpenFIGI limit is 10 jobs per request")

    securities = parse_sp500_constituents(args.universe_html)
    jobs = [
        {"ticker": row["TICKER"], "cik": row["CIK"]}
        for row in securities.select("TICKER", "CIK").to_dicts()
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    n_batches = (len(jobs) + args.batch_size - 1) // args.batch_size
    for batch_index in range(n_batches):
        output = args.output_dir / f"batch_{batch_index:03d}.json"
        if output.exists():
            continue
        batch = jobs[batch_index * args.batch_size : (batch_index + 1) * args.batch_size]
        for attempt in range(6):
            try:
                responses = fetch(batch, args.exchange_scope)
                break
            except subprocess.CalledProcessError:
                if attempt == 5:
                    raise
                time.sleep(10.0 * (attempt + 1))
        envelope = {
            "endpoint": ENDPOINT,
            "exchange_scope": args.exchange_scope,
            "jobs": batch,
            "responses": responses,
        }
        output.write_text(json.dumps(envelope, indent=2, sort_keys=True) + "\n")
        print(f"wrote {output} ({batch_index + 1}/{n_batches})", flush=True)
        if batch_index + 1 < n_batches:
            time.sleep(args.pause)


if __name__ == "__main__":
    main()
