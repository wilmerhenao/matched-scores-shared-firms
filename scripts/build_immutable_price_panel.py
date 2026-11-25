#!/usr/bin/env python3
"""Build a frozen, source-audited 2023--2024 price panel.

The panel keeps the already-frozen public yfinance batches for the original
linked universe and recovers historically named or delisted constituents from
checksum-pinned public snapshots.  Every recovered series must agree in daily
returns with an independently selected snapshot (or public endpoint), and
known dividend payers must actually contain an adjusted-price path.  Corporate
successor returns are never appended after a constituent's final trade.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from functools import reduce
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.provenance import record, sha256_file


PANEL_START = date(2023, 1, 3)
PANEL_END = date(2024, 12, 20)
PRE_START = date(2023, 1, 3)
PRE_END = date(2023, 12, 29)
Q1_START = date(2024, 1, 2)
Q1_END = date(2024, 3, 28)
PRIMARY_START = date(2024, 1, 2)
PRIMARY_END = date(2024, 12, 20)


@dataclass(frozen=True)
class SnapshotSpec:
    """Immutable identity and retrieval metadata for one public snapshot."""

    version: int
    canonical_file_name: str
    compressed_sha256: str
    compressed_size_bytes: int
    dataset_slug: str
    download_url: str
    download_command: str
    license_label: str
    upstream_file_name: str
    upstream_sha256: str
    upstream_size_bytes: int


EXPECTED_SNAPSHOT_VERSIONS = frozenset({700, 728, 750, 790, 854, 992, 1017, 1021})
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def load_snapshot_manifest(path: Path) -> dict[int, SnapshotSpec]:
    """Load and validate the checksum contract for Kaggle snapshots."""

    if not path.is_file():
        raise FileNotFoundError(f"missing Kaggle snapshot source manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(
        payload.get("files"), list
    ):
        raise RuntimeError("Kaggle snapshot source manifest has an invalid schema")
    snapshots: dict[int, SnapshotSpec] = {}
    for raw in payload["files"]:
        try:
            spec = SnapshotSpec(
                version=int(raw["version"]),
                canonical_file_name=str(raw["canonical_file_name"]),
                compressed_sha256=str(raw["compressed_sha256"]),
                compressed_size_bytes=int(raw["compressed_size_bytes"]),
                dataset_slug=str(raw["dataset_slug"]),
                download_url=str(raw["download_url"]),
                download_command=str(raw["download_command"]),
                license_label=str(raw["license_label"]),
                upstream_file_name=str(raw["upstream_file_name"]),
                upstream_sha256=str(raw["upstream_sha256"]),
                upstream_size_bytes=int(raw["upstream_size_bytes"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(
                "Kaggle snapshot source manifest has a malformed file record"
            ) from error
        expected_name = f"sp500_stocks_kaggle_v{spec.version:04d}.csv.gz"
        if (
            spec.version in snapshots
            or spec.canonical_file_name != expected_name
            or not SHA256_PATTERN.fullmatch(spec.compressed_sha256)
            or not SHA256_PATTERN.fullmatch(spec.upstream_sha256)
            or spec.compressed_size_bytes <= 0
            or spec.upstream_size_bytes <= 0
            or spec.dataset_slug != "andrewmvd/sp-500-stocks"
            or str(spec.version) not in spec.download_url
            or str(spec.version) not in spec.download_command
            or spec.upstream_file_name != "sp500_stocks.csv"
            or "CC0" not in spec.license_label
        ):
            raise RuntimeError(
                f"Kaggle snapshot source manifest record is invalid for v{spec.version}"
            )
        snapshots[spec.version] = spec
    if set(snapshots) != EXPECTED_SNAPSHOT_VERSIONS:
        raise RuntimeError(
            "Kaggle snapshot source manifest versions differ from the frozen contract: "
            f"observed={sorted(snapshots)}, expected={sorted(EXPECTED_SNAPSHOT_VERSIONS)}"
        )
    return snapshots


@dataclass(frozen=True)
class RecoverySpec:
    """Selection and cross-check contract for one recovered price history."""

    canonical_ticker: str
    source_kind: str
    source_alias: str
    source_version: int | None
    reference_alias: str
    reference_version: int
    expected_adjustment: bool
    verified_last_date: date
    corporate_action_status: str
    corporate_action_note: str
    action_evidence_id: str | None


def _kaggle(
    ticker: str,
    *,
    alias: str | None = None,
    version: int = 1021,
    reference_alias: str | None = None,
    reference_version: int = 790,
    adjusted: bool = False,
    last: date = PRIMARY_END,
    status: str = "survivor_to_primary_endpoint",
    note: str = "same issuer through the frozen primary endpoint",
    action_evidence_id: str | None = None,
) -> RecoverySpec:
    source_alias = alias or ticker
    return RecoverySpec(
        canonical_ticker=ticker,
        source_kind="kaggle_cc0_snapshot",
        source_alias=source_alias,
        source_version=version,
        reference_alias=reference_alias or source_alias,
        reference_version=reference_version,
        expected_adjustment=adjusted,
        verified_last_date=last,
        corporate_action_status=status,
        corporate_action_note=note,
        action_evidence_id=action_evidence_id,
    )


RECOVERY_SPECS = {
    spec.canonical_ticker: spec
    for spec in [
        _kaggle("ANSS"),
        _kaggle("APTV"),
        _kaggle("BF-B", adjusted=True),
        _kaggle("BK", adjusted=True),
        _kaggle("BLK", adjusted=True),
        _kaggle("BRK-B"),
        _kaggle(
            "CDAY",
            alias="DAY",
            reference_alias="CDAY",
            reference_version=700,
            action_evidence_id="CDAY_TO_DAY_2024",
        ),
        RecoverySpec(
            canonical_ticker="CTLT",
            source_kind="stockanalysis_s_and_p_global",
            source_alias="CTLT",
            source_version=None,
            reference_alias="CTLT",
            reference_version=1017,
            expected_adjustment=False,
            verified_last_date=date(2024, 12, 17),
            corporate_action_status="acquired_2024",
            corporate_action_note=(
                "Novo Holdings acquisition; December 18 zero-volume terminal "
                "marker excluded, final positive-volume trade December 17"
            ),
            action_evidence_id="CTLT_ACQUISITION_2024",
        ),
        _kaggle("DFS", adjusted=True),
        _kaggle("EA", adjusted=True),
        _kaggle("FI"),
        _kaggle(
            "FLT",
            alias="CPAY",
            reference_alias="FLT",
            reference_version=750,
            action_evidence_id="FLT_TO_CPAY_2024",
        ),
        _kaggle(
            "HES", adjusted=True, reference_alias="HES", reference_version=854
        ),
        _kaggle("HOLX"),
        _kaggle("IPG", adjusted=True),
        _kaggle("JNPR", adjusted=True),
        _kaggle("K", adjusted=True),
        _kaggle("MMC", adjusted=True),
        _kaggle(
            "MRO",
            version=992,
            adjusted=True,
            last=date(2024, 11, 21),
            status="acquired_2024",
            note="ConocoPhillips acquisition; final MRO trade November 21",
            action_evidence_id="MRO_ACQUISITION_2024",
        ),
        _kaggle("PARA", adjusted=True),
        _kaggle(
            "PEAK",
            alias="DOC",
            reference_alias="PEAK",
            reference_version=728,
            adjusted=True,
            note=(
                "Healthpeak CIK continuity; PEAK became DOC on March 4 after "
                "the merger, with backfilled returns checked against PEAK"
            ),
            action_evidence_id="PEAK_TO_DOC_2024",
        ),
        _kaggle(
            "PXD",
            version=790,
            reference_version=700,
            adjusted=True,
            last=date(2024, 5, 2),
            status="acquired_2024",
            note="Exxon Mobil acquisition; final PXD trade May 2",
            action_evidence_id="PXD_ACQUISITION_2024",
        ),
        _kaggle("TEL", adjusted=True),
        _kaggle("WBA", adjusted=True),
        _kaggle(
            "WRK",
            version=854,
            adjusted=True,
            last=date(2024, 7, 5),
            status="merged_2024",
            note="Smurfit Westrock combination; final WRK history July 5",
            action_evidence_id="WRK_MERGER_2024",
        ),
        RecoverySpec(
            canonical_ticker="XOM",
            source_kind="stockanalysis_s_and_p_global",
            source_alias="XOM",
            source_version=None,
            reference_alias="XOM",
            reference_version=790,
            expected_adjustment=True,
            verified_last_date=PRIMARY_END,
            corporate_action_status="survivor_to_primary_endpoint",
            corporate_action_note=(
                "complete public adjusted series used because adjacent Kaggle "
                "snapshots supplied unadjusted or null XOM values"
            ),
            action_evidence_id=None,
        ),
    ]
}


def load_action_evidence_manifest(
    path: Path, firms: pl.DataFrame
) -> tuple[str, dict[str, dict[str, Any]]]:
    """Validate ticker-alias evidence against the linked issuer universe."""

    if not path.is_file():
        raise FileNotFoundError(f"missing price alias/action evidence manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    authority_scope = payload.get("authority_scope")
    records = payload.get("records")
    if (
        payload.get("schema_version") != 1
        or not isinstance(authority_scope, str)
        or "not an authoritative" not in authority_scope
        or not isinstance(records, list)
    ):
        raise RuntimeError("price alias/action evidence manifest has an invalid schema")
    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(
            record.get("evidence_id"), str
        ):
            raise RuntimeError("price alias/action evidence manifest has a malformed record")
        evidence_id = record["evidence_id"]
        if evidence_id in by_id:
            raise RuntimeError(f"duplicate price alias/action evidence id: {evidence_id}")
        if (
            not isinstance(record.get("local_evidence"), list)
            or not record["local_evidence"]
            or not isinstance(record.get("public_context_sources"), list)
            or not record["public_context_sources"]
        ):
            raise RuntimeError(f"incomplete price alias/action evidence: {evidence_id}")
        by_id[evidence_id] = record

    expected_ids = {
        spec.action_evidence_id
        for spec in RECOVERY_SPECS.values()
        if spec.action_evidence_id is not None
    }
    if set(by_id) != expected_ids:
        raise RuntimeError(
            "price alias/action evidence ids differ from the recovery contract: "
            f"observed={sorted(by_id)}, expected={sorted(expected_ids)}"
        )
    firm_ciks = dict(
        zip(firms.get_column("PRICE_TICKER"), firms.get_column("CIK"), strict=True)
    )
    for spec in RECOVERY_SPECS.values():
        if spec.action_evidence_id is None:
            continue
        record = by_id[spec.action_evidence_id]
        if (
            record.get("canonical_ticker") != spec.canonical_ticker
            or record.get("source_alias") != spec.source_alias
            or record.get("corporate_action_status")
            != spec.corporate_action_status
            or record.get("verified_last_date") != str(spec.verified_last_date)
            or record.get("cik") != firm_ciks.get(spec.canonical_ticker)
        ):
            raise RuntimeError(
                "price alias/action evidence disagrees with the recovery contract: "
                f"{spec.action_evidence_id}"
            )
    return authority_scope, by_id


def snapshot_path(root: Path, spec: SnapshotSpec) -> Path:
    """Resolve the canonical cache path for one snapshot specification."""

    return root / spec.canonical_file_name


def _gzip_content_fingerprint(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size_bytes = 0
    with gzip.open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            size_bytes += len(block)
    return digest.hexdigest(), size_bytes


def verify_snapshot(
    path: Path,
    expected_sha256: str,
    *,
    expected_size_bytes: int | None = None,
    expected_upstream_sha256: str | None = None,
    expected_upstream_size_bytes: int | None = None,
    download_command: str | None = None,
) -> None:
    """Verify compressed and decompressed identities for one snapshot."""

    if not path.is_file():
        hint = f"; download with `{download_command}`" if download_command else ""
        raise FileNotFoundError(f"missing immutable snapshot {path}{hint}")
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise RuntimeError(
            f"immutable snapshot fingerprint mismatch for {path}: {observed}"
        )
    if expected_size_bytes is not None and path.stat().st_size != expected_size_bytes:
        raise RuntimeError(
            f"immutable snapshot size mismatch for {path}: {path.stat().st_size}"
        )
    upstream_expected = [
        expected_upstream_sha256 is not None,
        expected_upstream_size_bytes is not None,
    ]
    if any(upstream_expected) and not all(upstream_expected):
        raise ValueError("provide both upstream snapshot fingerprint fields")
    if all(upstream_expected):
        if path.suffix != ".gz":
            raise RuntimeError(f"upstream snapshot payload gate requires gzip: {path}")
        upstream_sha256, upstream_size_bytes = _gzip_content_fingerprint(path)
        if (
            upstream_sha256 != expected_upstream_sha256
            or upstream_size_bytes != expected_upstream_size_bytes
        ):
            raise RuntimeError(
                "immutable upstream snapshot payload mismatch for "
                f"{path}: sha256={upstream_sha256}, size={upstream_size_bytes}"
            )


def load_snapshot_symbols(path: Path, symbols: set[str]) -> dict[str, pl.DataFrame]:
    """Read selected symbols from one verified compressed snapshot."""

    frame = (
        pl.scan_csv(
            path,
            schema_overrides={
                "Adj Close": pl.Float64,
                "Close": pl.Float64,
                "Volume": pl.Float64,
            },
        )
        .filter(pl.col("Symbol").is_in(sorted(symbols)))
        .select(
            pl.col("Date").str.to_date().alias("date"),
            "Symbol",
            pl.col("Adj Close").alias("adjusted_close"),
            pl.col("Close").alias("close"),
            pl.col("Volume").alias("volume"),
        )
        .collect()
    )
    result: dict[str, pl.DataFrame] = {}
    for symbol in symbols:
        result[symbol] = frame.filter(pl.col("Symbol") == symbol).drop("Symbol").sort(
            "date"
        )
    return result


def load_stockanalysis(raw_path: Path, ticker: str) -> pl.DataFrame:
    """Parse one frozen StockAnalysis response into a canonical price frame."""

    parsed = json.loads(raw_path.read_text(encoding="utf-8"))
    if parsed.get("status") != 200 or not isinstance(parsed.get("data"), list):
        raise RuntimeError(f"invalid frozen StockAnalysis response for {ticker}")
    return (
        pl.DataFrame(parsed["data"])
        .select(
            pl.col("t").str.to_date().alias("date"),
            pl.col("a").cast(pl.Float64).alias("adjusted_close"),
            pl.col("c").cast(pl.Float64).alias("close"),
            pl.col("v").cast(pl.Float64).alias("volume"),
        )
        .sort("date")
    )


def _daily_returns(frame: pl.DataFrame, column: str = "adjusted_close") -> pl.DataFrame:
    return frame.sort("date").select(
        "date", (pl.col(column) / pl.col(column).shift(1) - 1.0).alias("return")
    )


def cross_source_return_check(
    selected: pl.DataFrame,
    reference: pl.DataFrame,
    *,
    tolerance: float,
    minimum_overlap: int = 200,
) -> tuple[int, float]:
    """Compare adjusted-return paths across independent public snapshots."""

    selected_returns = _daily_returns(selected).rename({"return": "selected_return"})
    reference_returns = _daily_returns(reference).rename({"return": "reference_return"})
    overlap = (
        selected_returns.join(reference_returns, on="date", how="inner", validate="1:1")
        .filter(pl.col("date").is_between(PANEL_START, PANEL_END))
        .drop_nulls()
    )
    if overlap.height < minimum_overlap:
        raise RuntimeError(
            f"adjusted-return cross-check has only {overlap.height} observations"
        )
    discrepancy = float(
        (overlap.get_column("selected_return") - overlap.get_column("reference_return"))
        .abs()
        .max()
    )
    if not np.isfinite(discrepancy) or discrepancy > tolerance:
        raise RuntimeError(
            f"adjusted-return cross-check failed: max difference {discrepancy:.9g} "
            f"> {tolerance:.9g}"
        )
    return overlap.height, discrepancy


def validate_source_frame(
    frame: pl.DataFrame,
    *,
    ticker: str,
    expected_adjustment: bool,
    verified_last_date: date,
) -> pl.DataFrame:
    """Enforce date, finiteness, adjustment, and terminal-trading gates."""

    if frame.get_column("date").n_unique() != frame.height or not frame.get_column(
        "date"
    ).is_sorted():
        raise RuntimeError(f"duplicate or unsorted source dates for {ticker}")
    active = frame.filter(pl.col("date").is_between(PANEL_START, verified_last_date))
    if active.is_empty():
        raise RuntimeError(f"empty source window for {ticker}")
    malformed = active.select(
        pl.any_horizontal(
            pl.col("adjusted_close").is_null(),
            pl.col("close").is_null(),
            pl.col("volume").is_null(),
            ~pl.col("adjusted_close").is_finite(),
            ~pl.col("close").is_finite(),
            ~pl.col("volume").is_finite(),
            pl.col("adjusted_close") <= 0,
            pl.col("close") <= 0,
            pl.col("volume") < 0,
        ).any()
    ).item()
    if malformed:
        raise RuntimeError(f"invalid active price or volume value for {ticker}")
    if expected_adjustment:
        ratio_sd = (active.get_column("adjusted_close") / active.get_column("close")).std()
        if ratio_sd is None or not np.isfinite(ratio_sd) or ratio_sd <= 1e-6:
            raise RuntimeError(
                f"expected dividend adjustment is absent from source for {ticker}"
            )
    return active


def _indexed_paths(paths: list[Path], prefix: str) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    for path in paths:
        match = re.fullmatch(rf"{re.escape(prefix)}_(\d+)", path.stem)
        if match is None:
            raise RuntimeError(f"unexpected {prefix} batch filename: {path.name}")
        key = match.group(1)
        if key in indexed:
            raise RuntimeError(f"duplicate {prefix} batch index: {key}")
        indexed[key] = path
    return indexed


def validate_price_batches(
    close_paths: list[Path],
    volume_paths: list[Path],
    audit_paths: list[Path],
    expected_tickers: set[str],
) -> list[dict[str, Any]]:
    """Validate frozen upstream batches without importing downstream designs."""

    close_by_index = _indexed_paths(close_paths, "close")
    volume_by_index = _indexed_paths(volume_paths, "volume")
    audit_by_index = _indexed_paths(audit_paths, "audit")
    indices = set(close_by_index)
    if not indices or indices != set(volume_by_index) or indices != set(audit_by_index):
        raise RuntimeError("price directory has incomplete close/volume/audit batches")

    records: list[dict[str, Any]] = []
    all_requested: list[str] = []
    request_ranges: set[tuple[Any, Any]] = set()
    for key in sorted(indices):
        audit = json.loads(audit_by_index[key].read_text(encoding="utf-8"))
        requested = list(audit.get("requested", []))
        observed = list(audit.get("observed", []))
        missing = list(audit.get("missing", []))
        schema_version = audit.get("schema_version")
        if schema_version not in {1, 2, 3}:
            raise RuntimeError(f"price audit {key} has an unsupported schema")
        if audit.get("batch_index") != int(key):
            raise RuntimeError(f"price audit {key} has the wrong batch index")
        if len(requested) != len(set(requested)):
            raise RuntimeError(f"price audit {key} repeats a ticker")
        if missing or observed != requested:
            raise RuntimeError(
                f"price completeness gate failed for batch {key}; "
                f"missing={missing}, observed_match={observed == requested}"
            )
        close_frame = pl.read_parquet(close_by_index[key])
        volume_frame = pl.read_parquet(volume_by_index[key])
        if schema_version == 3:
            if audit.get("close_sha256") != sha256_file(close_by_index[key]):
                raise RuntimeError(f"price audit {key} close fingerprint mismatch")
            if audit.get("volume_sha256") != sha256_file(volume_by_index[key]):
                raise RuntimeError(f"price audit {key} volume fingerprint mismatch")
            symbol_audit_name = audit.get("symbol_audit_file")
            if not isinstance(symbol_audit_name, str):
                raise RuntimeError(f"price audit {key} lacks a symbol audit")
            symbol_audit_path = audit_by_index[key].parent / symbol_audit_name
            if (
                not symbol_audit_path.is_file()
                or audit.get("symbol_audit_sha256")
                != sha256_file(symbol_audit_path)
            ):
                raise RuntimeError(f"price audit {key} symbol-audit fingerprint mismatch")
            symbol_audit = pl.read_parquet(symbol_audit_path)
            if (
                symbol_audit.get_column("PRICE_TICKER").n_unique()
                != symbol_audit.height
                or set(symbol_audit.get_column("PRICE_TICKER")) != set(requested)
            ):
                raise RuntimeError(f"price audit {key} symbol audit is not exhaustive")
        expected_columns = ["date", *requested]
        if close_frame.columns != expected_columns or volume_frame.columns != expected_columns:
            raise RuntimeError(f"price batch {key} columns do not match its audit")
        expected_rows = audit.get("rows")
        if close_frame.height != expected_rows or volume_frame.height != expected_rows:
            raise RuntimeError(f"price batch {key} row count does not match its audit")
        close_dates = close_frame.get_column("date")
        volume_dates = volume_frame.get_column("date")
        if (
            close_dates.n_unique() != close_frame.height
            or volume_dates.n_unique() != volume_frame.height
            or not close_dates.is_sorted()
            or not volume_dates.is_sorted()
            or not close_dates.equals(volume_dates)
        ):
            raise RuntimeError(f"price batch {key} has misaligned or duplicate dates")
        request_ranges.add((audit.get("start"), audit.get("end")))
        all_requested.extend(requested)
        records.append(audit)

    if len(request_ranges) != 1:
        raise RuntimeError("price batches use inconsistent request windows")
    if len(all_requested) != len(set(all_requested)):
        raise RuntimeError("a ticker appears in more than one price batch")
    if set(all_requested) != expected_tickers:
        missing = sorted(expected_tickers - set(all_requested))
        unexpected = sorted(set(all_requested) - expected_tickers)
        raise RuntimeError(
            f"price request universe mismatch; missing={missing}, unexpected={unexpected}"
        )
    return records


def load_wide_batches(paths: list[Path]) -> pl.DataFrame:
    """Join canonically indexed wide price batches on their date column."""

    frames = [pl.read_parquet(path) for path in paths]
    if not frames:
        raise ValueError("no price batches found")
    frame = reduce(
        lambda left, right: left.join(
            right, on="date", how="full", coalesce=True, validate="1:1"
        ),
        frames,
    )
    if len(frame.columns) != len(set(frame.columns)):
        raise ValueError("duplicate price columns across frozen batches")
    return frame.sort("date")


def _load_batch_directory(
    path: Path, *, require_complete: bool = True
) -> tuple[
    pl.DataFrame,
    pl.DataFrame,
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    close_paths = sorted(path.glob("close_*.parquet"))
    volume_paths = sorted(path.glob("volume_*.parquet"))
    audit_paths = sorted(path.glob("audit_*.json"))
    requested = {
        ticker
        for audit_path in audit_paths
        for ticker in json.loads(audit_path.read_text(encoding="utf-8")).get(
            "requested", []
        )
    }
    if require_complete:
        audits = validate_price_batches(
            close_paths, volume_paths, audit_paths, requested
        )
    else:
        audits = [
            json.loads(path.read_text(encoding="utf-8")) for path in audit_paths
        ]
        if any(
            audit.get("batch_index") != index
            for index, audit in enumerate(audits)
        ):
            raise RuntimeError("reference price batches are not canonically indexed")
    details: dict[str, dict[str, Any]] = {}
    for audit, close_path, volume_path, audit_path in zip(
        audits, close_paths, volume_paths, audit_paths, strict=True
    ):
        fallback = set(audit.get("recovered_from_fallback", []))
        for ticker in audit["requested"]:
            details[ticker] = {
                "source_kind": (
                    "stockanalysis_fallback_via_frozen_batch"
                    if ticker in fallback
                    else "yfinance_frozen_batch"
                ),
                "source_version": "yfinance_2023_2024_v2",
                "source_close_sha256": sha256_file(close_path),
                "source_volume_sha256": sha256_file(volume_path),
                "source_audit_sha256": sha256_file(audit_path),
                "source_created_at_utc": audit.get("created_at_utc"),
            }
    return (
        load_wide_batches(close_paths),
        load_wide_batches(volume_paths),
        audits,
        details,
    )


def _counts(frame: pl.DataFrame) -> tuple[int, int, int]:
    return (
        frame.filter(pl.col("date").is_between(PRE_START, PRE_END)).height,
        frame.filter(pl.col("date").is_between(Q1_START, Q1_END)).height,
        frame.filter(pl.col("date").is_between(PRIMARY_START, PRIMARY_END)).height,
    )


def _canonical_digest(frame: pl.DataFrame, columns: list[str]) -> str:
    payload = frame.select(columns).sort(columns).write_json().encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    """Assemble and audit the immutable 2023--2024 price panel."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--firms",
        type=Path,
        default=PROJECT_ROOT / "data_public/processed/ownership_q4_2023/firm_nodes.parquet",
    )
    parser.add_argument(
        "--base-price-dir",
        type=Path,
        default=PROJECT_ROOT / "data_public/raw/yfinance_2023_2024_v2",
    )
    parser.add_argument(
        "--base-reference-dir",
        type=Path,
        default=PROJECT_ROOT / "data_public/raw/yfinance_2023_2024",
    )
    parser.add_argument(
        "--base-fallback-dir",
        type=Path,
        default=PROJECT_ROOT / "data_public/raw/stockanalysis_fallback_2023_2024",
    )
    parser.add_argument(
        "--kaggle-root",
        type=Path,
        default=PROJECT_ROOT / "data_public/raw/kaggle_sp500_snapshots",
    )
    parser.add_argument(
        "--snapshot-manifest",
        type=Path,
        default=PROJECT_ROOT / "source_manifests/kaggle_sp500_snapshots.json",
    )
    parser.add_argument(
        "--action-evidence-manifest",
        type=Path,
        default=PROJECT_ROOT / "source_manifests/price_alias_actions.json",
    )
    parser.add_argument(
        "--stockanalysis-ctlt-dir",
        type=Path,
        default=PROJECT_ROOT / "data_public/raw/stockanalysis_ctlt_2023_2024",
    )
    parser.add_argument(
        "--stockanalysis-xom-dir",
        type=Path,
        default=PROJECT_ROOT / "data_public/raw/stockanalysis_xom_2023_2024",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data_public/raw/immutable_prices_2023_2024",
    )
    args = parser.parse_args()

    firms = pl.read_parquet(args.firms).sort("NODE_ID")
    if firms.get_column("PRICE_TICKER").n_unique() != firms.height:
        raise RuntimeError("price tickers are not unique across ownership firms")
    canonical_tickers = sorted(firms.get_column("PRICE_TICKER").to_list())
    snapshots = load_snapshot_manifest(args.snapshot_manifest)
    action_evidence_authority, action_evidence = load_action_evidence_manifest(
        args.action_evidence_manifest, firms
    )

    base_close, base_volume, base_audits, base_details = _load_batch_directory(
        args.base_price_dir
    )
    reference_close, _, reference_audits, reference_details = _load_batch_directory(
        args.base_reference_dir, require_complete=False
    )
    base_tickers = set(base_close.columns) - {"date"}
    recovered_tickers = set(canonical_tickers) - base_tickers
    if recovered_tickers != set(RECOVERY_SPECS):
        raise RuntimeError(
            "recovery contract does not exactly equal the missing base universe: "
            f"missing_specs={sorted(recovered_tickers - set(RECOVERY_SPECS))}, "
            f"unexpected_specs={sorted(set(RECOVERY_SPECS) - recovered_tickers)}"
        )

    needed_symbols: dict[int, set[str]] = {version: set() for version in snapshots}
    for spec in RECOVERY_SPECS.values():
        if spec.source_version is not None:
            needed_symbols[spec.source_version].add(spec.source_alias)
        needed_symbols[spec.reference_version].add(spec.reference_alias)
    # TMUS is the one base ticker absent from the earlier frozen yfinance run.
    needed_symbols[1021].add("TMUS")
    required_snapshot_versions = {
        version for version, symbols in needed_symbols.items() if symbols
    }
    if required_snapshot_versions != set(snapshots):
        raise RuntimeError(
            "Kaggle snapshot manifest is not the exact source/reference dependency set: "
            f"required={sorted(required_snapshot_versions)}, "
            f"manifest={sorted(snapshots)}"
        )

    snapshot_frames: dict[int, dict[str, pl.DataFrame]] = {}
    snapshot_inputs: list[Path] = []
    for version, symbols in needed_symbols.items():
        if not symbols:
            continue
        snapshot = snapshots[version]
        path = snapshot_path(args.kaggle_root, snapshot)
        verify_snapshot(
            path,
            snapshot.compressed_sha256,
            expected_size_bytes=snapshot.compressed_size_bytes,
            expected_upstream_sha256=snapshot.upstream_sha256,
            expected_upstream_size_bytes=snapshot.upstream_size_bytes,
            download_command=snapshot.download_command,
        )
        snapshot_frames[version] = load_snapshot_symbols(path, symbols)
        snapshot_inputs.append(path)

    ctlt_raw = args.stockanalysis_ctlt_dir / "CTLT.json"
    xom_raw = args.stockanalysis_xom_dir / "XOM.json"
    stockanalysis_frames = {
        "CTLT": load_stockanalysis(ctlt_raw, "CTLT"),
        "XOM": load_stockanalysis(xom_raw, "XOM"),
    }
    stockanalysis_audits = {
        ticker: json.loads(
            (directory / "fallback_audit.json").read_text(encoding="utf-8")
        )
        for ticker, directory in {
            "CTLT": args.stockanalysis_ctlt_dir,
            "XOM": args.stockanalysis_xom_dir,
        }.items()
    }
    for ticker, directory in {
        "CTLT": args.stockanalysis_ctlt_dir,
        "XOM": args.stockanalysis_xom_dir,
    }.items():
        audit = stockanalysis_audits[ticker]
        if audit.get("close_sha256") != sha256_file(directory / "fallback_close.parquet"):
            raise RuntimeError(f"StockAnalysis close fingerprint mismatch for {ticker}")
        if audit.get("volume_sha256") != sha256_file(directory / "fallback_volume.parquet"):
            raise RuntimeError(f"StockAnalysis volume fingerprint mismatch for {ticker}")
        if audit["ticker_audits"][0].get("raw_sha256") != sha256_file(
            directory / f"{ticker}.json"
        ):
            raise RuntimeError(f"StockAnalysis raw fingerprint mismatch for {ticker}")

    calendar = (
        base_close.select("date")
        .filter(pl.col("date").is_between(PANEL_START, PANEL_END))
        .sort("date")
    )
    if calendar.height != 496 or calendar.get_column("date").n_unique() != 496:
        raise RuntimeError("frozen 2023--December 20, 2024 calendar is not 496 dates")
    output_close = calendar
    output_volume = calendar
    symbol_audits: list[dict[str, Any]] = []

    reference_tickers = {
        ticker for audit in reference_audits for ticker in audit.get("observed", [])
    }
    fallback_reference_close_path = args.base_fallback_dir / "fallback_close.parquet"
    fallback_reference_close = pl.read_parquet(fallback_reference_close_path)
    fallback_reference_audit_path = args.base_fallback_dir / "fallback_audit.json"
    fallback_reference_audit = json.loads(
        fallback_reference_audit_path.read_text(encoding="utf-8")
    )
    if fallback_reference_audit.get("close_sha256") != sha256_file(
        fallback_reference_close_path
    ):
        raise RuntimeError("base StockAnalysis reference close fingerprint mismatch")
    for ticker in sorted(base_tickers):
        selected = base_close.select("date", ticker).rename({ticker: "adjusted_close"})
        if ticker in reference_tickers:
            reference = reference_close.select("date", ticker).rename(
                {ticker: "adjusted_close"}
            )
            reference_version = "yfinance_2023_2024"
            reference_sha256 = reference_details[ticker]["source_close_sha256"]
        elif ticker == "TMUS":
            reference = snapshot_frames[1021]["TMUS"]
            reference_version = "kaggle_cc0_v1021"
            reference_sha256 = snapshots[1021].upstream_sha256
        elif ticker in {"CMA", "CTRA"}:
            reference = fallback_reference_close.select("date", ticker).rename(
                {ticker: "adjusted_close"}
            )
            reference_version = "stockanalysis_fallback_2023_2024"
            reference_sha256 = fallback_reference_audit["close_sha256"]
        else:
            raise RuntimeError(f"no frozen adjusted-price reference for {ticker}")
        minimum_overlap = 50 if ticker in {"KVUE", "VLTO"} else 200
        overlap, discrepancy = cross_source_return_check(
            selected,
            reference,
            tolerance=1e-5,
            minimum_overlap=minimum_overlap,
        )
        selected_panel = calendar.join(selected, on="date", how="left", validate="1:1")
        selected_volume = calendar.join(
            base_volume.select("date", ticker).rename({ticker: "volume"}),
            on="date",
            how="left",
            validate="1:1",
        )
        output_close = output_close.with_columns(
            selected_panel.get_column("adjusted_close").alias(ticker)
        )
        output_volume = output_volume.with_columns(
            selected_volume.get_column("volume").alias(ticker)
        )
        finite = selected_panel.filter(pl.col("adjusted_close").is_not_null())
        pre_n, q1_n, primary_n = _counts(finite)
        detail = base_details[ticker]
        firm = firms.filter(pl.col("PRICE_TICKER") == ticker).row(0, named=True)
        symbol_audits.append(
            {
                "CIK": firm["CIK"],
                "PRICE_TICKER": ticker,
                "SOURCE_KIND": detail["source_kind"],
                "SOURCE_ALIAS": ticker,
                "SOURCE_VERSION": detail["source_version"],
                "SOURCE_RAW_SHA256": detail["source_audit_sha256"],
                "SOURCE_CLOSE_SHA256": detail["source_close_sha256"],
                "SOURCE_VOLUME_SHA256": detail["source_volume_sha256"],
                "REFERENCE_ALIAS": ticker,
                "REFERENCE_VERSION": reference_version,
                "REFERENCE_RAW_SHA256": reference_sha256,
                "RETURN_OVERLAP": overlap,
                "MAX_ABS_RETURN_DIFFERENCE": discrepancy,
                "ADJUSTMENT_CHECK": "frozen_cross_extraction_return_stability",
                "FIRST_DATE": str(finite.get_column("date").min()),
                "LAST_SOURCE_DATE": str(finite.get_column("date").max()),
                "VERIFIED_LAST_DATE": str(PRIMARY_END),
                "PRE_PRICE_OBSERVATIONS": pre_n,
                "Q1_PRICE_OBSERVATIONS": q1_n,
                "PRIMARY_PRICE_OBSERVATIONS": primary_n,
                "CORPORATE_ACTION_STATUS": "survivor_to_primary_endpoint",
                "CORPORATE_ACTION_NOTE": "same ticker through frozen primary endpoint",
                "CORPORATE_ACTION_EVIDENCE_ID": None,
                "CORPORATE_ACTION_EVIDENCE_SCOPE": None,
            }
        )

    for ticker in sorted(recovered_tickers):
        spec = RECOVERY_SPECS[ticker]
        if spec.source_kind == "kaggle_cc0_snapshot":
            assert spec.source_version is not None
            selected = snapshot_frames[spec.source_version][spec.source_alias]
            selected_sha256 = snapshots[spec.source_version].upstream_sha256
            source_version = f"kaggle_cc0_v{spec.source_version}"
            source_close_sha256 = selected_sha256
            source_volume_sha256 = selected_sha256
        else:
            selected = stockanalysis_frames[ticker]
            audit = stockanalysis_audits[ticker]
            selected_sha256 = audit["ticker_audits"][0]["raw_sha256"]
            source_version = audit["created_at_utc"]
            source_close_sha256 = audit["close_sha256"]
            source_volume_sha256 = audit["volume_sha256"]
        selected_source_last_date = selected.get_column("date").max()
        selected = validate_source_frame(
            selected,
            ticker=ticker,
            expected_adjustment=spec.expected_adjustment,
            verified_last_date=spec.verified_last_date,
        )
        reference = snapshot_frames[spec.reference_version][spec.reference_alias]
        tolerance = 5e-4 if spec.source_kind.startswith("stockanalysis") else 1e-5
        overlap, discrepancy = cross_source_return_check(
            selected,
            reference,
            tolerance=tolerance,
            minimum_overlap=200,
        )
        selected_panel = calendar.join(selected, on="date", how="left", validate="1:1")
        selected_panel = selected_panel.with_columns(
            pl.when(pl.col("date") <= spec.verified_last_date)
            .then(pl.col("adjusted_close"))
            .otherwise(None)
            .alias("adjusted_close"),
            pl.when(pl.col("date") <= spec.verified_last_date)
            .then(pl.col("volume"))
            .otherwise(None)
            .alias("volume"),
        )
        output_close = output_close.with_columns(
            selected_panel.get_column("adjusted_close").alias(ticker)
        )
        output_volume = output_volume.with_columns(
            selected_panel.get_column("volume").alias(ticker)
        )
        finite = selected_panel.filter(pl.col("adjusted_close").is_not_null())
        pre_n, q1_n, primary_n = _counts(finite)
        if pre_n != 250 or q1_n != 61:
            raise RuntimeError(
                f"recovered {ticker} does not have the complete pre/Q1 calendar"
            )
        if spec.corporate_action_status == "survivor_to_primary_endpoint" and primary_n != 246:
            raise RuntimeError(f"recovered survivor {ticker} lacks the primary calendar")
        firm = firms.filter(pl.col("PRICE_TICKER") == ticker).row(0, named=True)
        symbol_audits.append(
            {
                "CIK": firm["CIK"],
                "PRICE_TICKER": ticker,
                "SOURCE_KIND": spec.source_kind,
                "SOURCE_ALIAS": spec.source_alias,
                "SOURCE_VERSION": source_version,
                "SOURCE_RAW_SHA256": selected_sha256,
                "SOURCE_CLOSE_SHA256": source_close_sha256,
                "SOURCE_VOLUME_SHA256": source_volume_sha256,
                "REFERENCE_ALIAS": spec.reference_alias,
                "REFERENCE_VERSION": f"kaggle_cc0_v{spec.reference_version}",
                "REFERENCE_RAW_SHA256": snapshots[
                    spec.reference_version
                ].upstream_sha256,
                "RETURN_OVERLAP": overlap,
                "MAX_ABS_RETURN_DIFFERENCE": discrepancy,
                "ADJUSTMENT_CHECK": "_and_".join(
                    [
                        (
                            "adjustment_present"
                            if spec.expected_adjustment
                            else "nondividend"
                        ),
                        (
                            "cross_source_return_stable"
                            if spec.source_kind.startswith("stockanalysis")
                            else "cross_snapshot_return_stable"
                        ),
                    ]
                ),
                "FIRST_DATE": str(finite.get_column("date").min()),
                "LAST_SOURCE_DATE": str(selected_source_last_date),
                "VERIFIED_LAST_DATE": str(spec.verified_last_date),
                "PRE_PRICE_OBSERVATIONS": pre_n,
                "Q1_PRICE_OBSERVATIONS": q1_n,
                "PRIMARY_PRICE_OBSERVATIONS": primary_n,
                "CORPORATE_ACTION_STATUS": spec.corporate_action_status,
                "CORPORATE_ACTION_NOTE": spec.corporate_action_note,
                "CORPORATE_ACTION_EVIDENCE_ID": spec.action_evidence_id,
                "CORPORATE_ACTION_EVIDENCE_SCOPE": (
                    action_evidence_authority
                    if spec.action_evidence_id is not None
                    else None
                ),
            }
        )

    output_close = output_close.select("date", *canonical_tickers)
    output_volume = output_volume.select("date", *canonical_tickers)
    if output_close.width != 500 or output_volume.width != 500:
        raise RuntimeError("immutable panel does not contain all 499 firms")
    if output_close.get_column("date").n_unique() != output_close.height:
        raise RuntimeError("immutable panel repeats a date")
    if output_close.select(pl.exclude("date").is_null().all()).row(0).count(True):
        raise RuntimeError("immutable panel contains an entirely missing symbol")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    close_path = args.output_dir / "close_000.parquet"
    volume_path = args.output_dir / "volume_000.parquet"
    symbol_audit_path = args.output_dir / "symbol_audit.parquet"
    audit_path = args.output_dir / "audit_000.json"
    run_id = datetime.now(timezone.utc).strftime(
        "immutable-public-prices-%Y%m%dT%H%M%SZ"
    )
    output_close.write_parquet(close_path)
    output_volume.write_parquet(volume_path)
    symbol_audit = pl.DataFrame(
        symbol_audits, infer_schema_length=None
    ).sort("PRICE_TICKER")
    if symbol_audit.height != 499 or symbol_audit.get_column(
        "PRICE_TICKER"
    ).n_unique() != 499:
        raise RuntimeError("per-symbol price audit does not cover exactly 499 firms")
    symbol_audit.write_parquet(symbol_audit_path)
    audit = {
        "schema_version": 3,
        "producer_run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "batch_index": 0,
        "requested": canonical_tickers,
        "observed": canonical_tickers,
        "missing": [],
        "start": str(PANEL_START),
        "end": str(PANEL_END),
        "rows": output_close.height,
        "price_field": "adjusted close",
        "calendar_contract": {
            "pre_price_dates": 250,
            "q1_2024_price_dates": 61,
            "primary_2024_price_dates": 246,
        },
        "strict_per_symbol_source_audit": True,
        "corporate_successor_returns_appended": False,
        "kaggle_snapshot_source_manifest_sha256": sha256_file(
            args.snapshot_manifest
        ),
        "kaggle_snapshot_dataset_slug": snapshots[700].dataset_slug,
        "kaggle_snapshot_license_label": snapshots[700].license_label,
        "kaggle_snapshot_versions": sorted(snapshot_frames),
        "alias_action_evidence_manifest_sha256": sha256_file(
            args.action_evidence_manifest
        ),
        "alias_action_evidence_authority_scope": action_evidence_authority,
        "alias_action_evidence_ids": sorted(action_evidence),
        "symbol_audit_file": symbol_audit_path.name,
        "symbol_audit_sha256": sha256_file(symbol_audit_path),
        "close_sha256": sha256_file(close_path),
        "volume_sha256": sha256_file(volume_path),
        "ownership_universe_sha256": _canonical_digest(
            firms, ["CIK", "PRICE_TICKER"]
        ),
        "source_summary": (
            symbol_audit.group_by("SOURCE_KIND")
            .len()
            .sort("SOURCE_KIND")
            .to_dicts()
        ),
        "recovered_tickers": sorted(recovered_tickers),
        "verified_2024_exits": symbol_audit.filter(
            pl.col("CORPORATE_ACTION_STATUS") != "survivor_to_primary_endpoint"
        ).select(
            "PRICE_TICKER",
            "CORPORATE_ACTION_STATUS",
            "CORPORATE_ACTION_EVIDENCE_ID",
            "VERIFIED_LAST_DATE",
            "PRIMARY_PRICE_OBSERVATIONS",
        ).to_dicts(),
    }
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    input_paths = [
        Path(__file__),
        PROJECT_ROOT / "scripts/fetch_yfinance_prices.py",
        PROJECT_ROOT / "scripts/fetch_stockanalysis_fallback.py",
        PROJECT_ROOT / "src/provenance.py",
        args.snapshot_manifest,
        args.action_evidence_manifest,
        args.firms,
        *sorted(args.base_price_dir.glob("close_*.parquet")),
        *sorted(args.base_price_dir.glob("volume_*.parquet")),
        *sorted(args.base_price_dir.glob("audit_*.json")),
        *sorted(args.base_reference_dir.glob("close_*.parquet")),
        *sorted(args.base_reference_dir.glob("volume_*.parquet")),
        *sorted(args.base_reference_dir.glob("audit_*.json")),
        args.base_fallback_dir / "fallback_close.parquet",
        fallback_reference_audit_path,
        *snapshot_inputs,
        ctlt_raw,
        args.stockanalysis_ctlt_dir / "fallback_close.parquet",
        args.stockanalysis_ctlt_dir / "fallback_volume.parquet",
        args.stockanalysis_ctlt_dir / "fallback_audit.json",
        xom_raw,
        args.stockanalysis_xom_dir / "fallback_close.parquet",
        args.stockanalysis_xom_dir / "fallback_volume.parquet",
        args.stockanalysis_xom_dir / "fallback_audit.json",
    ]
    record(
        run_id=run_id,
        command=[sys.executable, *sys.argv],
        config={
            "panel_start": str(PANEL_START),
            "panel_end": str(PANEL_END),
            "requested_tickers": len(canonical_tickers),
            "snapshot_versions": sorted(
                version for version, symbols in needed_symbols.items() if symbols
            ),
            "return_stability_checks_fail_closed": True,
            "return_stability_comparison_scope": (
                "cross-extraction and cross-snapshot; CTLT/XOM use a "
                "StockAnalysis-to-Kaggle cross-source check"
            ),
            "corporate_successor_returns_appended": False,
        },
        inputs=input_paths,
        outputs=[close_path, volume_path, symbol_audit_path, audit_path],
        status="completed",
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
