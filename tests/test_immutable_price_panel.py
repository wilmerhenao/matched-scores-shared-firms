from __future__ import annotations

import gzip
import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from scripts.build_immutable_price_panel import (
    cross_source_return_check,
    load_action_evidence_manifest,
    load_snapshot_manifest,
    validate_source_frame,
    verify_snapshot,
)


def _source_frame(*, adjusted: bool) -> pl.DataFrame:
    dates = [date(2023, 1, 3) + timedelta(days=index) for index in range(260)]
    close = 100.0 * np.cumprod(1.0 + 0.001 * np.sin(np.arange(260)))
    if adjusted:
        # A piecewise adjustment factor mimics a dividend while preserving a
        # smooth adjusted-return path.
        factor = np.where(np.arange(260) < 130, 0.98, 1.0)
    else:
        factor = np.ones(260)
    return pl.DataFrame(
        {
            "date": dates,
            "adjusted_close": close * factor,
            "close": close,
            "volume": np.full(260, 1000.0),
        }
    )


def test_snapshot_fingerprint_gate_is_fail_closed(tmp_path: Path) -> None:
    snapshot = tmp_path / "sp500_stocks.csv"
    snapshot.write_text("immutable\n", encoding="utf-8")
    verify_snapshot(
        snapshot,
        "f71e36a5dac9d7d1ccc5ea816b533122cd38f79ab6e035968ae495f8a89c10d0",
    )
    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        verify_snapshot(snapshot, "0" * 64)


def test_compressed_snapshot_gate_checks_archive_and_upstream_payload(
    tmp_path: Path,
) -> None:
    payload = b"Date,Symbol,Adj Close,Close,High,Low,Open,Volume\n"
    snapshot = tmp_path / "snapshot.csv.gz"
    with snapshot.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as encoded:
            encoded.write(payload)
    verify_snapshot(
        snapshot,
        hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        expected_size_bytes=snapshot.stat().st_size,
        expected_upstream_sha256=hashlib.sha256(payload).hexdigest(),
        expected_upstream_size_bytes=len(payload),
    )
    with pytest.raises(RuntimeError, match="upstream snapshot payload mismatch"):
        verify_snapshot(
            snapshot,
            hashlib.sha256(snapshot.read_bytes()).hexdigest(),
            expected_size_bytes=snapshot.stat().st_size,
            expected_upstream_sha256="0" * 64,
            expected_upstream_size_bytes=len(payload),
        )


def test_tracked_snapshot_manifest_is_exact_and_fail_closed(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    source = project_root / "source_manifests/kaggle_sp500_snapshots.json"
    snapshots = load_snapshot_manifest(source)
    assert set(snapshots) == {700, 728, 750, 790, 854, 992, 1017, 1021}
    assert all(spec.canonical_file_name.endswith(".csv.gz") for spec in snapshots.values())

    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["files"] = payload["files"][:-1]
    malformed = tmp_path / "missing-version.json"
    malformed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="versions differ"):
        load_snapshot_manifest(malformed)


def test_tracked_alias_action_manifest_matches_hardcoded_contract() -> None:
    project_root = Path(__file__).resolve().parents[1]
    firms = pl.DataFrame(
        {
            "PRICE_TICKER": ["CDAY", "CTLT", "FLT", "MRO", "PEAK", "PXD", "WRK"],
            "CIK": [
                "0001725057",
                "0001596783",
                "0001175454",
                "0000101778",
                "0000765880",
                "0001038357",
                "0001732845",
            ],
        }
    )
    authority, records = load_action_evidence_manifest(
        project_root / "source_manifests/price_alias_actions.json", firms
    )
    assert "not an authoritative" in authority
    assert set(records) == {
        "CDAY_TO_DAY_2024",
        "CTLT_ACQUISITION_2024",
        "FLT_TO_CPAY_2024",
        "MRO_ACQUISITION_2024",
        "PEAK_TO_DOC_2024",
        "PXD_ACQUISITION_2024",
        "WRK_MERGER_2024",
    }


def test_expected_dividend_adjustment_cannot_equal_raw_close() -> None:
    unadjusted = _source_frame(adjusted=False)
    with pytest.raises(RuntimeError, match="expected dividend adjustment is absent"):
        validate_source_frame(
            unadjusted,
            ticker="DIV",
            expected_adjustment=True,
            verified_last_date=unadjusted.get_column("date").max(),
        )

    adjusted = _source_frame(adjusted=True)
    observed = validate_source_frame(
        adjusted,
        ticker="DIV",
        expected_adjustment=True,
        verified_last_date=adjusted.get_column("date").max(),
    )
    assert observed.height == adjusted.height


def test_cross_source_check_uses_returns_not_price_levels() -> None:
    selected = _source_frame(adjusted=True)
    reference = selected.with_columns(
        (7.0 * pl.col("adjusted_close")).alias("adjusted_close")
    )
    overlap, discrepancy = cross_source_return_check(
        selected, reference, tolerance=1e-12, minimum_overlap=200
    )
    assert overlap == selected.height - 1
    assert discrepancy < 1e-15

    corrupted = reference.with_columns(
        pl.when(pl.int_range(pl.len()) == 220)
        .then(1.01 * pl.col("adjusted_close"))
        .otherwise(pl.col("adjusted_close"))
        .alias("adjusted_close")
    )
    with pytest.raises(RuntimeError, match="cross-check failed"):
        cross_source_return_check(
            selected, corrupted, tolerance=1e-4, minimum_overlap=200
        )
