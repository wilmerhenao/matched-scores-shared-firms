#!/usr/bin/env python3
"""Fetch resumable adjusted-close and volume batches for the validated firms."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import polars as pl
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.provenance import record


def extract_field(
    data: pd.DataFrame,
    field: str,
    requested: list[str],
) -> pd.DataFrame:
    """Return one yfinance field with flat ticker columns."""
    if data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        level = next(
            (
                index
                for index in range(data.columns.nlevels)
                if field in data.columns.get_level_values(index)
            ),
            None,
        )
        if level is None:
            return pd.DataFrame(index=data.index)
        frame = data.xs(field, axis=1, level=level, drop_level=True).copy()
        if isinstance(frame, pd.Series):
            frame = frame.to_frame()
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(-1)
    else:
        if field not in data.columns:
            return pd.DataFrame(index=data.index)
        frame = data[[field]].copy()
        if len(requested) != 1:
            raise RuntimeError(
                f"flat yfinance response for {len(requested)} requested tickers"
            )
        frame.columns = requested
    frame.columns = [str(column) for column in frame.columns]
    return frame


def download_fields(
    tickers: list[str],
    start: str,
    end: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download adjusted closes and volumes for one ticker batch."""

    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=len(tickers) > 1,
        timeout=60,
    )
    return (
        extract_field(data, "Close", tickers),
        extract_field(data, "Volume", tickers),
    )


def observed_tickers(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    requested: list[str],
) -> list[str]:
    """Return tickers with at least one observed close and volume."""

    return [
        ticker
        for ticker in requested
        if ticker in close.columns
        and ticker in volume.columns
        and int(close[ticker].notna().sum()) > 0
        and int(volume[ticker].notna().sum()) > 0
    ]


def replace_column(
    frame: pd.DataFrame,
    replacement: pd.DataFrame,
    ticker: str,
) -> pd.DataFrame:
    """Replace one ticker column with an audited fallback series."""

    if ticker not in replacement.columns:
        return frame
    return frame.drop(columns=[ticker], errors="ignore").join(
        replacement[[ticker]], how="outer"
    )


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one persisted price artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_fallback(
    close_path: Path | None,
    volume_path: Path | None,
    audit_path: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object] | None]:
    """Load and authenticate an optional close-volume fallback bundle."""

    provided = [path is not None for path in (close_path, volume_path, audit_path)]
    if any(provided) and not all(provided):
        raise ValueError("provide fallback close, volume, and audit together")
    if not any(provided):
        return pd.DataFrame(), pd.DataFrame(), None
    assert close_path is not None and volume_path is not None and audit_path is not None
    for path in (close_path, volume_path, audit_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("close_sha256") != sha256_file(close_path):
        raise RuntimeError("fallback close fingerprint mismatch")
    if audit.get("volume_sha256") != sha256_file(volume_path):
        raise RuntimeError("fallback volume fingerprint mismatch")

    def as_pandas(path: Path) -> pd.DataFrame:
        frame = pl.read_parquet(path).to_pandas()
        frame["date"] = pd.to_datetime(frame["date"])
        return frame.set_index("date").sort_index()

    return as_pandas(close_path), as_pandas(volume_path), audit


def main() -> None:
    """Download resumable price batches and record symbol-level coverage."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--firms", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2025-01-10")
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--pause", type=float, default=1.0)
    parser.add_argument("--individual-retries", type=int, default=3)
    parser.add_argument("--fallback-close", type=Path)
    parser.add_argument("--fallback-volume", type=Path)
    parser.add_argument("--fallback-audit", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tickers = sorted(
        pl.read_parquet(args.firms).get_column("PRICE_TICKER").unique().to_list()
    )
    fallback_close, fallback_volume, fallback_audit = load_fallback(
        args.fallback_close, args.fallback_volume, args.fallback_audit
    )
    n_batches = (len(tickers) + args.batch_size - 1) // args.batch_size
    for batch_index in range(n_batches):
        batch = tickers[
            batch_index * args.batch_size : (batch_index + 1) * args.batch_size
        ]
        close_path = args.output_dir / f"close_{batch_index:03d}.parquet"
        volume_path = args.output_dir / f"volume_{batch_index:03d}.parquet"
        audit_path = args.output_dir / f"audit_{batch_index:03d}.json"
        existing_count = sum(
            path.exists() for path in (close_path, volume_path, audit_path)
        )
        if existing_count not in (0, 3):
            raise RuntimeError(
                f"partial batch {batch_index}; use a fresh output directory"
            )
        if existing_count == 3:
            existing = json.loads(audit_path.read_text(encoding="utf-8"))
            expected = {
                "requested": batch,
                "start": args.start,
                "end": args.end,
            }
            actual = {key: existing.get(key) for key in expected}
            if actual != expected:
                raise RuntimeError(
                    f"batch {batch_index} does not match current request; "
                    "use a fresh output directory"
                )
            if not existing.get("missing"):
                continue
            print(
                f"retrying incomplete batch {batch_index + 1}/{n_batches}: "
                f"{existing['missing']}",
                flush=True,
            )

        close, volume = download_fields(batch, args.start, args.end)
        recovered_individually: list[str] = []
        recovered_from_fallback: list[str] = []
        missing = sorted(set(batch) - set(observed_tickers(close, volume, batch)))
        for ticker in missing:
            for attempt in range(args.individual_retries):
                single_close, single_volume = download_fields(
                    [ticker], args.start, args.end
                )
                if ticker in observed_tickers(
                    single_close, single_volume, [ticker]
                ):
                    close = replace_column(close, single_close, ticker)
                    volume = replace_column(volume, single_volume, ticker)
                    recovered_individually.append(ticker)
                    break
                if attempt + 1 < args.individual_retries:
                    time.sleep(args.pause)
        missing = sorted(set(batch) - set(observed_tickers(close, volume, batch)))
        for ticker in missing:
            if (
                ticker in fallback_close.columns
                and ticker in fallback_volume.columns
                and fallback_close[ticker].notna().sum() > 0
                and fallback_volume[ticker].notna().sum() > 0
            ):
                close = replace_column(close, fallback_close, ticker)
                volume = replace_column(volume, fallback_volume, ticker)
                recovered_from_fallback.append(ticker)

        if close.empty:
            raise RuntimeError(f"empty yfinance batch {batch_index}: {batch}")
        close = close.sort_index().reindex(columns=batch)
        volume = volume.sort_index().reindex(columns=batch)
        close.index.name = "date"
        volume.index.name = "date"
        close_frame = pl.from_pandas(close.reset_index()).with_columns(
            pl.col("date").cast(pl.Date)
        )
        volume_frame = pl.from_pandas(volume.reset_index()).with_columns(
            pl.col("date").cast(pl.Date)
        )
        # A batch is complete only when both adjusted close and volume exist.
        observed = observed_tickers(close, volume, batch)
        missing = sorted(set(batch) - set(observed))
        close_frame.write_parquet(close_path)
        volume_frame.write_parquet(volume_path)
        audit = {
            "schema_version": 2,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "batch_index": batch_index,
            "requested": batch,
            "observed": observed,
            "missing": missing,
            "recovered_individually": sorted(recovered_individually),
            "recovered_from_fallback": sorted(recovered_from_fallback),
            "fallback_audit_sha256": (
                sha256_file(args.fallback_audit)
                if recovered_from_fallback and args.fallback_audit is not None
                else None
            ),
            "fallback_purpose": (
                fallback_audit.get("purpose")
                if recovered_from_fallback and fallback_audit is not None
                else None
            ),
            "start": args.start,
            "end": args.end,
            "rows": close_frame.height,
        }
        audit_path.write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            f"wrote batch {batch_index + 1}/{n_batches}: "
            f"{len(observed)}/{len(batch)} tickers",
            flush=True,
        )
        if missing:
            raise RuntimeError(
                f"price completeness gate failed for batch {batch_index}: {missing}"
            )
        if batch_index + 1 < n_batches:
            time.sleep(args.pause)

    output_paths = [
        path
        for batch_index in range(n_batches)
        for path in (
            args.output_dir / f"close_{batch_index:03d}.parquet",
            args.output_dir / f"volume_{batch_index:03d}.parquet",
            args.output_dir / f"audit_{batch_index:03d}.json",
        )
    ]
    input_paths = [
        Path(__file__),
        PROJECT_ROOT / "src/provenance.py",
        args.firms,
    ]
    if args.fallback_close is not None:
        input_paths.extend(
            [
                PROJECT_ROOT / "scripts/fetch_stockanalysis_fallback.py",
                args.fallback_close,
                args.fallback_volume,
                args.fallback_audit,
            ]
        )
    record(
        run_id=datetime.now(timezone.utc).strftime(
            "yfinance-prices-%Y%m%dT%H%M%SZ"
        ),
        command=[sys.executable, *sys.argv],
        config={
            "start": args.start,
            "end": args.end,
            "batch_size": args.batch_size,
            "individual_retries": args.individual_retries,
            "requested_tickers": len(tickers),
            "strict_complete_panel": True,
        },
        inputs=input_paths,
        outputs=output_paths,
        status="completed",
    )


if __name__ == "__main__":
    main()
