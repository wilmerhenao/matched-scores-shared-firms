#!/usr/bin/env python3
"""Recover narrowly audited symbol histories from a public no-login endpoint.

This is a narrow fallback for symbols that Yahoo no longer serves after a
subsequent delisting.  It is not a second general-purpose price pipeline.
Raw responses remain under the ignored ``data_public`` tree and are not
redistributed.
"""

from __future__ import annotations

import argparse
import json
import ssl
import sys
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl
import certifi

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.provenance import record, sha256_file


API_TEMPLATE = (
    "https://stockanalysis.com/api/symbol/s/{ticker}/history"
    "?range=Max&period=Daily"
)

# Independent, public month-end anchors recorded from Digrin's price-history
# pages.  Both adjusted and unadjusted values agree to the displayed cent.
PUBLIC_ANCHORS = {
    "CMA": [
        ("2023-12-29", 50.75, 55.81),
        ("2024-12-31", 59.24, 61.85),
    ],
    "CTRA": [
        ("2023-12-29", 23.69, 25.52),
        ("2024-03-28", 26.09, 27.88),
    ],
    "CTLT": [
        ("2023-12-29", 44.93, 44.93),
        ("2024-12-16", 63.48, 63.48),
    ],
    # The immutable Kaggle snapshots intermittently contain an unadjusted XOM
    # series.  StockAnalysis supplies a complete adjusted history, while these
    # public Digrin month-end values provide a separate displayed-price check.
    "XOM": [
        ("2023-12-29", 92.02, 99.98),
        ("2024-12-31", 102.38, 107.57),
    ],
}
ANCHOR_URLS = {
    "CMA": "https://www.digrin.com/stocks/detail/CMA/price",
    "CTRA": "https://www.digrin.com/stocks/detail/CTRA/price",
    "CTLT": "https://www.digrin.com/stocks/detail/CTLT/price",
    "XOM": "https://www.digrin.com/stocks/detail/XOM/price",
}
MINIMUM_ROWS = {
    # Catalent stopped trading on its verified December 18, 2024 acquisition
    # date, so a 2023-through-2024 request is structurally shorter than 500
    # observations while still covering both analysis windows.
    "CTLT": 450,
}


def fetch(url: str, user_agent: str) -> bytes:
    """Fetch one public JSON endpoint with a research user agent."""

    request = urllib.request.Request(
        url,
        headers={"User-Agent": user_agent, "Accept": "application/json"},
    )
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(request, timeout=90, context=context) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status} for {url}")
        return response.read()


def main() -> None:
    """Build an audited fallback panel for the supported symbols."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=["CMA", "CTRA"])
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2025-01-10")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data_public/raw/stockanalysis_fallback_2023_2024",
    )
    parser.add_argument(
        "--user-agent",
        default="DyadicDML-Fin/0.1 academic-reproducibility",
    )
    args = parser.parse_args()
    tickers = sorted(set(ticker.upper() for ticker in args.tickers))
    unsupported = sorted(set(tickers) - set(PUBLIC_ANCHORS))
    if unsupported:
        raise ValueError(
            f"fallback is deliberately restricted to audited symbols: {unsupported}"
        )
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    if start >= end:
        raise ValueError("start must precede end")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    series: dict[str, pl.DataFrame] = {}
    raw_paths: list[Path] = []
    ticker_audits: list[dict[str, object]] = []
    for ticker in tickers:
        url = API_TEMPLATE.format(ticker=ticker)
        payload = fetch(url, args.user_agent)
        raw_path = args.output_dir / f"{ticker}.json"
        raw_path.write_bytes(payload)
        raw_paths.append(raw_path)
        parsed = json.loads(payload)
        if parsed.get("status") != 200 or not isinstance(parsed.get("data"), list):
            raise RuntimeError(f"invalid response schema for {ticker}")
        frame = (
            pl.DataFrame(parsed["data"])
            .with_columns(pl.col("t").str.to_date().alias("date"))
            .filter(pl.col("date").is_between(start, end, closed="left"))
            .select(
                "date",
                pl.col("a").cast(pl.Float64).alias("adjusted_close"),
                pl.col("c").cast(pl.Float64).alias("close"),
                pl.col("v").cast(pl.Float64).alias("volume"),
            )
            .sort("date")
        )
        minimum_rows = MINIMUM_ROWS.get(ticker, 500)
        if (
            frame.height < minimum_rows
            or frame.get_column("date").n_unique() != frame.height
        ):
            raise RuntimeError(f"daily coverage gate failed for {ticker}")
        if frame.select(
            pl.any_horizontal(
                pl.col("adjusted_close").is_null(),
                pl.col("volume").is_null(),
                pl.col("adjusted_close") <= 0,
                pl.col("volume") < 0,
            ).any()
        ).item():
            raise RuntimeError(f"non-finite or non-positive fallback values for {ticker}")

        anchor_checks = []
        for anchor_date, adjusted, unadjusted in PUBLIC_ANCHORS[ticker]:
            row = frame.filter(pl.col("date") == date.fromisoformat(anchor_date))
            if row.height != 1:
                raise RuntimeError(f"missing public anchor {ticker} {anchor_date}")
            observed_adjusted = float(row.get_column("adjusted_close")[0])
            observed_close = float(row.get_column("close")[0])
            passed = (
                round(observed_adjusted, 2) == adjusted
                and round(observed_close, 2) == unadjusted
            )
            if not passed:
                raise RuntimeError(
                    f"cross-source anchor mismatch for {ticker} {anchor_date}"
                )
            anchor_checks.append(
                {
                    "date": anchor_date,
                    "expected_adjusted": adjusted,
                    "observed_adjusted": observed_adjusted,
                    "expected_unadjusted": unadjusted,
                    "observed_unadjusted": observed_close,
                    "pass": passed,
                }
            )
        series[ticker] = frame
        ticker_audits.append(
            {
                "ticker": ticker,
                "endpoint": url,
                "rows": frame.height,
                "first_date": str(frame.get_column("date").min()),
                "last_date": str(frame.get_column("date").max()),
                "raw_sha256": sha256_file(raw_path),
                "independent_anchor_url": ANCHOR_URLS[ticker],
                "anchor_checks": anchor_checks,
            }
        )

    common_dates = set(series[tickers[0]].get_column("date").to_list())
    for ticker in tickers[1:]:
        common_dates &= set(series[ticker].get_column("date").to_list())
    minimum_common_rows = min(MINIMUM_ROWS.get(ticker, 500) for ticker in tickers)
    if len(common_dates) < minimum_common_rows:
        raise RuntimeError(
            "fallback symbols do not share their minimum required trading dates"
        )

    close = series[tickers[0]].select(
        "date", pl.col("adjusted_close").alias(tickers[0])
    )
    volume = series[tickers[0]].select("date", pl.col("volume").alias(tickers[0]))
    for ticker in tickers[1:]:
        close = close.join(
            series[ticker].select("date", pl.col("adjusted_close").alias(ticker)),
            on="date",
            how="full",
            coalesce=True,
            validate="1:1",
        )
        volume = volume.join(
            series[ticker].select("date", pl.col("volume").alias(ticker)),
            on="date",
            how="full",
            coalesce=True,
            validate="1:1",
        )
    close_path = args.output_dir / "fallback_close.parquet"
    volume_path = args.output_dir / "fallback_volume.parquet"
    audit_path = args.output_dir / "fallback_audit.json"
    close.sort("date").write_parquet(close_path)
    volume.sort("date").write_parquet(volume_path)
    audit = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "narrow recovery for symbols removed from Yahoo after later delisting",
        "tickers": tickers,
        "start_inclusive": args.start,
        "end_exclusive": args.end,
        "price_field": "adjusted close (a)",
        "upstream_attribution": (
            "StockAnalysis history pages attribute these prices to "
            "S&P Global Market Intelligence"
        ),
        "raw_responses_redistributed": False,
        "ticker_audits": ticker_audits,
        "close_sha256": sha256_file(close_path),
        "volume_sha256": sha256_file(volume_path),
    }
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    record(
        run_id=datetime.now(timezone.utc).strftime(
            "stockanalysis-fallback-%Y%m%dT%H%M%SZ"
        ),
        command=[sys.executable, *sys.argv],
        config={"tickers": tickers, "start": args.start, "end": args.end},
        inputs=[Path(__file__), PROJECT_ROOT / "src/provenance.py"],
        outputs=[*raw_paths, close_path, volume_path, audit_path],
        status="completed",
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
