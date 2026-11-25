#!/usr/bin/env python3
"""Build the point-in-time continuous return-correlation application."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from functools import reduce
from pathlib import Path

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.provenance import record, sha256_file


FACTOR_COLUMNS = ["mkt_rf", "smb", "hml", "rmw", "cma"]


@dataclass(frozen=True)
class ReturnDesignSpec:
    """Frozen calendar, coverage, and interpretation rules for one design."""

    name: str
    role: str
    outcome_start: date
    outcome_end: date
    balanced_outcome: bool
    expected_outcome_dates: int
    minimum_outcome_dates: int
    expected_firms: int
    expected_exclusions: frozenset[str]
    default_output_directory: str
    outcome_label: str


RETURN_DESIGN_SPECS = {
    "primary_balanced_2024": ReturnDesignSpec(
        name="primary_balanced_2024",
        role="primary",
        outcome_start=date(2024, 1, 2),
        outcome_end=date(2024, 12, 20),
        balanced_outcome=True,
        expected_outcome_dates=246,
        minimum_outcome_dates=246,
        expected_firms=493,
        expected_exclusions=frozenset(
            {"CTLT", "KVUE", "MRO", "PXD", "VLTO", "WRK"}
        ),
        default_output_directory="continuous_design",
        outcome_label="balanced January 2--December 20, 2024 FF5-residual correlation",
    ),
    "partial_to_exit_2024": ReturnDesignSpec(
        name="partial_to_exit_2024",
        role="survival_selection_sensitivity",
        outcome_start=date(2024, 1, 2),
        outcome_end=date(2024, 12, 20),
        balanced_outcome=False,
        expected_outcome_dates=246,
        minimum_outcome_dates=200,
        expected_firms=495,
        expected_exclusions=frozenset({"KVUE", "PXD", "VLTO", "WRK"}),
        default_output_directory="continuous_design_partial_to_exit",
        outcome_label="verified partial-to-exit 2024 FF5-residual correlation",
    ),
    "balanced_q1_2024": ReturnDesignSpec(
        name="balanced_q1_2024",
        role="short_horizon_survival_selection_sensitivity",
        outcome_start=date(2024, 1, 2),
        outcome_end=date(2024, 3, 28),
        balanced_outcome=True,
        expected_outcome_dates=61,
        minimum_outcome_dates=61,
        expected_firms=497,
        expected_exclusions=frozenset({"KVUE", "VLTO"}),
        default_output_directory="continuous_design_q1_balanced",
        outcome_label="balanced calendar-Q1 2024 FF5-residual correlation",
    ),
}


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
) -> list[dict[str, object]]:
    """Require an exact, internally aligned set of complete price batches."""

    close_by_index = _indexed_paths(close_paths, "close")
    volume_by_index = _indexed_paths(volume_paths, "volume")
    audit_by_index = _indexed_paths(audit_paths, "audit")
    indices = set(close_by_index)
    if not indices or indices != set(volume_by_index) or indices != set(audit_by_index):
        raise RuntimeError("price directory has incomplete close/volume/audit batches")

    records: list[dict[str, object]] = []
    all_requested: list[str] = []
    request_ranges: set[tuple[object, object]] = set()
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
                or audit.get("symbol_audit_sha256") != sha256_file(symbol_audit_path)
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
    """Join nonoverlapping wide price batches on a unique date index."""

    frames = [pl.read_parquet(path) for path in paths]
    if not frames:
        raise ValueError("no price batches found")
    frame = reduce(
        lambda left, right: left.join(
            right, on="date", how="full", coalesce=True, validate="1:1"
        ),
        frames,
    )
    duplicate_columns = [
        name
        for name in frame.columns
        if name != "date" and frame.columns.count(name) > 1
    ]
    if duplicate_columns:
        raise ValueError(f"duplicate price columns: {duplicate_columns}")
    return frame.sort("date")


def daily_simple_returns(close: pl.DataFrame) -> pl.DataFrame:
    """Convert adjusted prices to arithmetic one-day returns.

    The Kenneth French factors and risk-free rate are arithmetic returns, so
    the dependent return must use the same unit before excess-return
    construction and factor regression.
    """

    expressions = [
        (pl.col(name) / pl.col(name).shift(1) - 1.0).alias(name)
        for name in close.columns
        if name != "date"
    ]
    return close.select("date", *expressions)


def validate_market_panels(
    close: pl.DataFrame,
    volume: pl.DataFrame,
    expected_tickers: set[str],
) -> None:
    """Reject malformed, infinite, or economically impossible market values."""

    if set(close.columns) != {"date", *expected_tickers}:
        raise RuntimeError("adjusted-close columns do not equal the ownership universe")
    if set(volume.columns) != {"date", *expected_tickers}:
        raise RuntimeError("volume columns do not equal the ownership universe")
    if not close.get_column("date").equals(volume.get_column("date")):
        raise RuntimeError("adjusted-close and volume dates disagree")
    for ticker in sorted(expected_tickers):
        close_values = close.get_column(ticker).to_numpy()
        volume_values = volume.get_column(ticker).to_numpy()
        if np.isinf(close_values).any() or np.isinf(volume_values).any():
            raise RuntimeError(f"infinite market value for {ticker}")
        if np.any(np.isfinite(close_values) & (close_values <= 0)):
            raise RuntimeError(f"nonpositive adjusted close for {ticker}")
        if np.any(np.isfinite(volume_values) & (volume_values < 0)):
            raise RuntimeError(f"negative volume for {ticker}")


def load_price_symbol_audit(
    price_dir: Path,
    audit_records: list[dict[str, object]],
    expected_tickers: set[str],
) -> tuple[pl.DataFrame, Path]:
    """Load the exhaustive immutable-price audit required by new designs."""

    if len(audit_records) != 1 or audit_records[0].get("schema_version") != 3:
        raise RuntimeError(
            "return designs require exactly one schema-3 immutable price batch"
        )
    symbol_audit_name = audit_records[0].get("symbol_audit_file")
    if not isinstance(symbol_audit_name, str):
        raise RuntimeError("immutable price batch lacks a symbol-audit path")
    path = price_dir / symbol_audit_name
    frame = pl.read_parquet(path).sort("PRICE_TICKER")
    required = {
        "CIK",
        "PRICE_TICKER",
        "SOURCE_KIND",
        "SOURCE_ALIAS",
        "SOURCE_VERSION",
        "SOURCE_RAW_SHA256",
        "REFERENCE_ALIAS",
        "REFERENCE_VERSION",
        "REFERENCE_RAW_SHA256",
        "RETURN_OVERLAP",
        "MAX_ABS_RETURN_DIFFERENCE",
        "ADJUSTMENT_CHECK",
        "VERIFIED_LAST_DATE",
        "PRE_PRICE_OBSERVATIONS",
        "Q1_PRICE_OBSERVATIONS",
        "PRIMARY_PRICE_OBSERVATIONS",
        "CORPORATE_ACTION_STATUS",
    }
    if not required.issubset(frame.columns):
        raise RuntimeError(
            "immutable symbol audit lacks required columns: "
            f"{sorted(required - set(frame.columns))}"
        )
    if (
        frame.get_column("PRICE_TICKER").n_unique() != frame.height
        or set(frame.get_column("PRICE_TICKER")) != expected_tickers
    ):
        raise RuntimeError("immutable symbol audit does not equal the firm universe")
    if frame.select(
        pl.any_horizontal(
            pl.col("SOURCE_RAW_SHA256").str.len_chars() != 64,
            pl.col("REFERENCE_RAW_SHA256").str.len_chars() != 64,
            pl.col("RETURN_OVERLAP") <= 0,
            ~pl.col("MAX_ABS_RETURN_DIFFERENCE").is_finite(),
        ).any()
    ).item():
        raise RuntimeError("immutable symbol audit contains an invalid provenance row")
    return frame, path


def evaluate_firm_coverage(
    *,
    firms: pl.DataFrame,
    close: pl.DataFrame,
    volume: pl.DataFrame,
    pre_diagnostics: dict[str, dict[str, float | int]],
    outcome_diagnostics: dict[str, dict[str, float | int]],
    symbol_audit: pl.DataFrame,
    spec: ReturnDesignSpec,
    outcome_dates: list[date],
    minimum_window_observations: int,
) -> pl.DataFrame:
    """Apply the named price-window rule and retain exact exclusion reasons."""

    if len(outcome_dates) != spec.expected_outcome_dates:
        raise RuntimeError(
            f"{spec.name} expected {spec.expected_outcome_dates} factor dates, "
            f"found {len(outcome_dates)}"
        )
    audit_lookup = {
        row["PRICE_TICKER"]: row for row in symbol_audit.to_dicts()
    }
    close_lookup = close.select("date", *firms.get_column("PRICE_TICKER").to_list())
    volume_lookup = volume.select("date", *firms.get_column("PRICE_TICKER").to_list())
    rows: list[dict[str, object]] = []
    for firm in firms.select("CIK", "PRICE_TICKER").sort("CIK").to_dicts():
        cik = firm["CIK"]
        ticker = firm["PRICE_TICKER"]
        source = audit_lookup[ticker]
        verified_last_date = date.fromisoformat(source["VERIFIED_LAST_DATE"])
        required_dates = (
            outcome_dates
            if spec.balanced_outcome
            else [observed for observed in outcome_dates if observed <= verified_last_date]
        )
        required_set = set(required_dates)
        close_values = (
            close_lookup.filter(pl.col("date").is_in(required_set))
            .sort("date")
            .get_column(ticker)
            .to_numpy()
        )
        volume_values = (
            volume_lookup.filter(pl.col("date").is_in(required_set))
            .sort("date")
            .get_column(ticker)
            .to_numpy()
        )
        complete_values = (
            np.isfinite(close_values)
            & np.isfinite(volume_values)
            & (close_values > 0)
            & (volume_values >= 0)
        )
        complete_dates = int(complete_values.sum())
        pre_n = int(pre_diagnostics.get(cik, {}).get("n_obs", 0))
        outcome_n = int(outcome_diagnostics.get(cik, {}).get("n_obs", 0))
        reasons: list[str] = []
        if pre_n < minimum_window_observations:
            reasons.append("pre_window_below_minimum")
        if spec.balanced_outcome and verified_last_date < spec.outcome_end:
            reasons.append("verified_history_ends_before_balanced_endpoint")
        if complete_dates != len(required_dates):
            reasons.append("outcome_adjusted_close_volume_calendar_incomplete")
        if len(required_dates) < spec.minimum_outcome_dates:
            reasons.append("outcome_window_below_minimum")
        if outcome_n != complete_dates:
            reasons.append("outcome_return_count_mismatch")
        rows.append(
            {
                **firm,
                "SOURCE_KIND": source["SOURCE_KIND"],
                "SOURCE_ALIAS": source["SOURCE_ALIAS"],
                "SOURCE_VERSION": source["SOURCE_VERSION"],
                "VERIFIED_LAST_DATE": source["VERIFIED_LAST_DATE"],
                "CORPORATE_ACTION_STATUS": source["CORPORATE_ACTION_STATUS"],
                "PRE_OBSERVATIONS": pre_n,
                "OUTCOME_OBSERVATIONS": outcome_n,
                "REQUIRED_OUTCOME_DATES": len(required_dates),
                "COMPLETE_OUTCOME_PRICE_VOLUME_DATES": complete_dates,
                "BALANCED_OUTCOME_REQUIRED": spec.balanced_outcome,
                "ELIGIBLE": not reasons,
                "EXCLUSION_REASON": ";".join(reasons),
            }
        )
    return pl.DataFrame(rows)


def identity_sha256(frame: pl.DataFrame, columns: list[str]) -> str:
    """Hash a canonically sorted subset of identity-defining columns."""

    payload = frame.select(columns).sort(columns).write_json().encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_ownership_graph(firms: pl.DataFrame, edges: pl.DataFrame) -> None:
    """Require a complete, node-aligned ownership graph before subsetting."""

    n_nodes = firms.height
    if firms.get_column("NODE_ID").to_list() != list(range(n_nodes)):
        raise RuntimeError("ownership node IDs are not contiguous and ordered")
    row, col = np.triu_indices(n_nodes, k=1)
    if edges.height != row.size:
        raise RuntimeError("ownership graph is not complete")
    if not np.array_equal(edges.get_column("NODE_I").to_numpy(), row) or not np.array_equal(
        edges.get_column("NODE_J").to_numpy(), col
    ):
        raise RuntimeError("ownership edge order does not match the node table")
    ciks = firms.get_column("CIK").to_numpy()
    if not np.array_equal(edges.get_column("CIK_I").to_numpy(), ciks[row]) or not np.array_equal(
        edges.get_column("CIK_J").to_numpy(), ciks[col]
    ):
        raise RuntimeError("ownership edge identities do not match the node table")
    treatment_columns = [
        "BINARY_COSINE",
        "BINARY_JACCARD",
        "PORTFOLIO_WEIGHT_COSINE",
    ]
    if not np.all(np.isfinite(edges.select(treatment_columns).to_numpy())):
        raise RuntimeError("ownership similarities contain nonfinite values")


def residualize_window(
    simple_returns: pl.DataFrame,
    factors: pl.DataFrame,
    ticker_to_cik: dict[str, str],
    start: date,
    end: date,
) -> tuple[pl.DataFrame, dict[str, dict[str, float | int]]]:
    """Residualize firm simple excess returns on daily FF5 factors by window."""

    joined = simple_returns.filter(pl.col("date").is_between(start, end)).join(
        factors, on="date", how="inner", validate="1:1"
    )
    output: dict[str, np.ndarray] = {}
    diagnostics: dict[str, dict[str, float | int]] = {}
    factor_values = joined.select(FACTOR_COLUMNS).to_numpy()
    rf = joined.get_column("rf").to_numpy()
    for ticker, cik in ticker_to_cik.items():
        if ticker not in joined.columns:
            continue
        stock_simple_return = joined.get_column(ticker).to_numpy()
        valid = (
            np.isfinite(stock_simple_return)
            & np.isfinite(rf)
            & np.all(np.isfinite(factor_values), axis=1)
        )
        residual = np.full(joined.height, np.nan, dtype=np.float64)
        if valid.sum() >= 30:
            if np.any(stock_simple_return[valid] <= -1.0):
                raise RuntimeError(
                    f"simple return at or below -100% prevents log momentum for {ticker}"
                )
            design = np.column_stack((np.ones(valid.sum()), factor_values[valid]))
            excess = stock_simple_return[valid] - rf[valid]
            coefficient, *_ = np.linalg.lstsq(design, excess, rcond=None)
            residual[valid] = excess - design @ coefficient
            diagnostics[cik] = {
                "n_obs": int(valid.sum()),
                "residual_mean": float(np.mean(residual[valid])),
                "residual_sd": float(np.std(residual[valid], ddof=1)),
                "arithmetic_excess_sd": float(np.std(excess, ddof=1)),
                "cumulative_log_return": float(
                    np.sum(np.log1p(stock_simple_return[valid]))
                ),
            }
        output[cik] = residual
    return pl.DataFrame({"date": joined.get_column("date"), **output}), diagnostics


def pair_correlations(
    panel: pl.DataFrame,
    node_ciks: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute pairwise-complete correlations and common-date counts."""

    values = panel.select(node_ciks).to_numpy()
    row, col = np.triu_indices(len(node_ciks), k=1)
    correlations = np.empty(row.size, dtype=np.float64)
    common = np.empty(row.size, dtype=np.int64)
    for edge, (left, right) in enumerate(zip(row, col)):
        valid = np.isfinite(values[:, left]) & np.isfinite(values[:, right])
        common[edge] = int(valid.sum())
        correlations[edge] = (
            np.corrcoef(values[valid, left], values[valid, right])[0, 1]
            if valid.sum() >= 3
            else np.nan
        )
    return row, col, correlations, common


def main() -> None:
    """Build a manifest-bound dyadic return design from frozen public inputs."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ownership-dir",
        type=Path,
        default=PROJECT_ROOT / "data_public/processed/ownership_q4_2023",
    )
    parser.add_argument(
        "--price-dir",
        type=Path,
        default=PROJECT_ROOT / "data_public/raw/immutable_prices_2023_2024",
    )
    parser.add_argument(
        "--factors",
        type=Path,
        default=PROJECT_ROOT / "data_public/raw/ff5_daily.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--design-spec",
        choices=sorted(RETURN_DESIGN_SPECS),
        default="primary_balanced_2024",
    )
    parser.add_argument("--minimum-window-observations", type=int, default=200)
    args = parser.parse_args()
    spec = RETURN_DESIGN_SPECS[args.design_spec]
    if args.minimum_window_observations != 200:
        raise ValueError("named return designs freeze the minimum window at 200")
    if args.output_dir is None:
        args.output_dir = (
            PROJECT_ROOT
            / "data_public/processed"
            / spec.default_output_directory
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    close_paths = sorted(args.price_dir.glob("close_*.parquet"))
    volume_paths = sorted(args.price_dir.glob("volume_*.parquet"))
    audit_paths = sorted(args.price_dir.glob("audit_*.json"))
    ownership_audit_path = args.ownership_dir / "audit_summary.json"
    ownership_audit = json.loads(ownership_audit_path.read_text(encoding="utf-8"))
    if ownership_audit.get("design_status") != "VALIDATED_FOR_ESTIMATION":
        raise RuntimeError("ownership inputs are not validated for estimation")
    firms = pl.read_parquet(args.ownership_dir / "firm_nodes.parquet").sort("NODE_ID")
    expected_tickers = set(firms.get_column("PRICE_TICKER").to_list())
    audit_records = validate_price_batches(
        close_paths, volume_paths, audit_paths, expected_tickers
    )
    symbol_audit, symbol_audit_path = load_price_symbol_audit(
        args.price_dir, audit_records, expected_tickers
    )
    firm_source_alignment = firms.select("CIK", "PRICE_TICKER").join(
        symbol_audit.select("CIK", "PRICE_TICKER"),
        on=["CIK", "PRICE_TICKER"],
        how="inner",
        validate="1:1",
    )
    if firm_source_alignment.height != firms.height:
        raise RuntimeError("immutable price identities do not align to ownership firms")
    if ownership_audit.get("mapped_firm_nodes") != firms.height:
        raise RuntimeError(
            "ownership node count does not match its validated audit"
        )
    close = load_wide_batches(close_paths)
    volume = load_wide_batches(volume_paths)
    validate_market_panels(close, volume, expected_tickers)
    simple_returns = daily_simple_returns(close)
    factors = pl.read_parquet(args.factors).select("date", *FACTOR_COLUMNS, "rf")
    outcome_dates = (
        factors.filter(
            pl.col("date").is_between(spec.outcome_start, spec.outcome_end)
        )
        .sort("date")
        .get_column("date")
        .to_list()
    )
    if len(outcome_dates) != spec.expected_outcome_dates:
        raise RuntimeError(
            f"factor calendar for {spec.name} has {len(outcome_dates)} dates; "
            f"expected {spec.expected_outcome_dates}"
        )
    ownership_edges = pl.read_parquet(
        args.ownership_dir / "ownership_edges.parquet"
    ).sort(["NODE_I", "NODE_J"])
    validate_ownership_graph(firms, ownership_edges)
    if ownership_audit.get("ownership_edges") != ownership_edges.height:
        raise RuntimeError(
            "ownership edge count does not match its validated audit"
        )
    ticker_to_cik = dict(
        zip(firms.get_column("PRICE_TICKER"), firms.get_column("CIK"))
    )

    pre_panel, pre_diagnostics = residualize_window(
        simple_returns,
        factors,
        ticker_to_cik,
        date(2023, 1, 3),
        date(2023, 12, 29),
    )
    outcome_panel, outcome_diagnostics = residualize_window(
        simple_returns,
        factors,
        ticker_to_cik,
        spec.outcome_start,
        spec.outcome_end,
    )
    coverage = evaluate_firm_coverage(
        firms=firms,
        close=close,
        volume=volume,
        pre_diagnostics=pre_diagnostics,
        outcome_diagnostics=outcome_diagnostics,
        symbol_audit=symbol_audit,
        spec=spec,
        outcome_dates=outcome_dates,
        minimum_window_observations=args.minimum_window_observations,
    )
    eligible = sorted(
        coverage.filter(pl.col("ELIGIBLE")).get_column("CIK").to_list()
    )
    exclusions = set(
        coverage.filter(~pl.col("ELIGIBLE")).get_column("PRICE_TICKER")
    )
    if len(eligible) != spec.expected_firms or exclusions != set(
        spec.expected_exclusions
    ):
        raise RuntimeError(
            f"{spec.name} fail-closed coverage mismatch: firms={len(eligible)} "
            f"expected={spec.expected_firms}, exclusions={sorted(exclusions)} "
            f"expected_exclusions={sorted(spec.expected_exclusions)}"
        )

    selected_firms = (
        firms.filter(pl.col("CIK").is_in(eligible))
        .sort("CIK")
        .rename({"NODE_ID": "OWNERSHIP_NODE_ID"})
        .with_row_index("NODE_ID")
    )
    node_ciks = selected_firms.get_column("CIK").to_list()
    row, col, pre_correlation, pre_common = pair_correlations(pre_panel, node_ciks)
    _, _, outcome_correlation, outcome_common = pair_correlations(outcome_panel, node_ciks)
    if np.min(pre_common) < args.minimum_window_observations:
        raise RuntimeError("at least one eligible pair fails the pre-period common-date gate")
    if spec.balanced_outcome and (
        np.min(outcome_common) != spec.expected_outcome_dates
        or np.max(outcome_common) != spec.expected_outcome_dates
    ):
        raise RuntimeError(
            "balanced outcome pairs do not share the exact frozen calendar"
        )
    if not spec.balanced_outcome and np.min(
        outcome_common
    ) < args.minimum_window_observations:
        raise RuntimeError("at least one eligible pair fails the outcome common-date gate")

    # Raw-return outcome is retained strictly as a robustness target.
    raw_outcome = simple_returns.filter(
        pl.col("date").is_between(spec.outcome_start, spec.outcome_end)
    ).select("date", *selected_firms.get_column("PRICE_TICKER").to_list())
    raw_outcome = raw_outcome.rename(
        dict(zip(selected_firms.get_column("PRICE_TICKER"), node_ciks))
    )
    _, _, raw_correlation, raw_common = pair_correlations(raw_outcome, node_ciks)
    if spec.balanced_outcome and (
        np.min(raw_common) != spec.expected_outcome_dates
        or np.max(raw_common) != spec.expected_outcome_dates
    ):
        raise RuntimeError("balanced raw-return pairs do not share the exact calendar")
    if not spec.balanced_outcome and np.min(
        raw_common
    ) < args.minimum_window_observations:
        raise RuntimeError("at least one eligible pair fails the raw-return common-date gate")

    endpoint = selected_firms.select(
        "CIK", "GICS_SECTOR", "GICS_SUBINDUSTRY"
    ).to_dicts()
    pre_vol = np.asarray(
        [pre_diagnostics[cik]["arithmetic_excess_sd"] for cik in node_ciks]
    )
    pre_momentum = np.asarray(
        [pre_diagnostics[cik]["cumulative_log_return"] for cik in node_ciks]
    )

    def fisher_z(correlation: np.ndarray) -> np.ndarray:
        return np.arctanh(np.clip(correlation, -0.999999, 0.999999))

    return_edges = pl.DataFrame(
        {
            "NODE_I": row,
            "NODE_J": col,
            "CIK_I": [node_ciks[index] for index in row],
            "CIK_J": [node_ciks[index] for index in col],
            "Y_FF5_FISHER_Z": fisher_z(outcome_correlation),
            "Y_RAW_FISHER_Z": fisher_z(raw_correlation),
            "PRE_FF5_FISHER_Z": fisher_z(pre_correlation),
            "PRE_COMMON_DATES": pre_common,
            "OUTCOME_COMMON_DATES": outcome_common,
            "RAW_OUTCOME_COMMON_DATES": raw_common,
            "PRE_VOL_MIN": np.minimum(pre_vol[row], pre_vol[col]),
            "PRE_VOL_MAX": np.maximum(pre_vol[row], pre_vol[col]),
            "PRE_VOL_ABS_DIFF": np.abs(pre_vol[row] - pre_vol[col]),
            "PRE_MOMENTUM_MIN": np.minimum(pre_momentum[row], pre_momentum[col]),
            "PRE_MOMENTUM_MAX": np.maximum(pre_momentum[row], pre_momentum[col]),
            "PRE_MOMENTUM_ABS_DIFF": np.abs(pre_momentum[row] - pre_momentum[col]),
            "SECTOR_PAIR": [
                " | ".join(sorted((endpoint[left]["GICS_SECTOR"], endpoint[right]["GICS_SECTOR"])))
                for left, right in zip(row, col)
            ],
        }
    )
    design = return_edges.join(
        ownership_edges.drop("NODE_I", "NODE_J"),
        on=["CIK_I", "CIK_J"],
        how="inner",
        validate="1:1",
    )
    if design.height != len(row):
        raise RuntimeError("ownership/return edge join is not complete")
    for source, destination in [
        ("BINARY_COSINE", "D_BINARY_COSINE_Z"),
        ("BINARY_JACCARD", "D_BINARY_JACCARD_Z"),
        ("PORTFOLIO_WEIGHT_COSINE", "D_PORTFOLIO_WEIGHT_COSINE_Z"),
    ]:
        mean = design.get_column(source).mean()
        standard_deviation = design.get_column(source).std(ddof=1)
        if (
            mean is None
            or standard_deviation is None
            or not np.isfinite(mean)
            or not np.isfinite(standard_deviation)
            or standard_deviation <= np.finfo(float).eps
        ):
            raise RuntimeError(f"cannot standardize ownership treatment {source}")
        design = design.with_columns(
            ((pl.col(source) - mean) / standard_deviation).alias(destination)
        )
    design = design.sort(["NODE_I", "NODE_J"])
    numeric_columns = [
        name for name, dtype in design.schema.items() if dtype.is_numeric()
    ]
    if not np.all(np.isfinite(design.select(numeric_columns).to_numpy())):
        raise RuntimeError("final dyadic design contains nonfinite numeric values")
    if design.get_column("SECTOR_PAIR").null_count() > 0:
        raise RuntimeError("final dyadic design contains a missing sector pair")

    output_paths = {
        "firm_nodes": args.output_dir / "firm_nodes.parquet",
        "dyadic_design": args.output_dir / "dyadic_design.parquet",
        "pre_residuals": args.output_dir / "pre_residuals.parquet",
        "outcome_residuals": args.output_dir / "outcome_residuals.parquet",
        "outcome_raw_returns": args.output_dir / "outcome_raw_returns.parquet",
        "firm_coverage": args.output_dir / "firm_coverage.parquet",
        "audit": args.output_dir / "audit_summary.json",
    }
    selected_firms.write_parquet(output_paths["firm_nodes"])
    design.write_parquet(output_paths["dyadic_design"])
    pre_panel.select("date", *node_ciks).write_parquet(output_paths["pre_residuals"])
    outcome_panel.select("date", *node_ciks).write_parquet(output_paths["outcome_residuals"])
    raw_outcome.write_parquet(output_paths["outcome_raw_returns"])
    coverage.write_parquet(output_paths["firm_coverage"])
    selected_tickers = sorted(selected_firms.get_column("PRICE_TICKER").to_list())
    primary_delta = {
        "primary_balanced_2024": [],
        "partial_to_exit_2024": ["CTLT", "MRO"],
        "balanced_q1_2024": ["CTLT", "MRO", "PXD", "WRK"],
    }[spec.name]
    if not set(primary_delta).issubset(selected_tickers):
        raise RuntimeError("declared node additions are absent from the named design")
    outcome_calendar_sha256 = hashlib.sha256(
        "\n".join(str(observed) for observed in outcome_dates).encode("utf-8")
    ).hexdigest()
    source_counts = (
        coverage.group_by("SOURCE_KIND")
        .len()
        .sort("SOURCE_KIND")
        .to_dicts()
    )
    audit = {
        "schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "design_status": (
            "VALIDATED_FOR_ESTIMATION"
            if spec.role == "primary"
            else "VALIDATED_SENSITIVITY_DESIGN"
        ),
        "design_spec": spec.name,
        "design_role": spec.role,
        "is_primary_default": spec.role == "primary",
        "balanced_outcome_calendar": spec.balanced_outcome,
        "pre_window": ["2023-01-03", "2023-12-29"],
        "outcome_window": [str(spec.outcome_start), str(spec.outcome_end)],
        "outcome_calendar_dates": spec.expected_outcome_dates,
        "outcome_calendar_sha256": outcome_calendar_sha256,
        "minimum_outcome_dates": spec.minimum_outcome_dates,
        "minimum_window_observations": args.minimum_window_observations,
        "ownership_firms": firms.height,
        "eligible_firms": selected_firms.height,
        "eligible_edges": design.height,
        "eligible_tickers": selected_tickers,
        "node_identity_sha256": identity_sha256(
            selected_firms, ["CIK", "PRICE_TICKER"]
        ),
        "edge_identity_sha256": identity_sha256(
            design, ["CIK_I", "CIK_J"]
        ),
        "added_tickers_relative_to_primary_balanced_2024": primary_delta,
        "price_batches": len(audit_records),
        "price_requested_tickers": len(expected_tickers),
        "price_source_counts": source_counts,
        "price_batch_schema": 3,
        "download_missing": [],
        "coverage_exclusions": coverage.filter(~pl.col("ELIGIBLE")).select(
            "CIK",
            "PRICE_TICKER",
            "SOURCE_ALIAS",
            "VERIFIED_LAST_DATE",
            "PRE_OBSERVATIONS",
            "OUTCOME_OBSERVATIONS",
            "REQUIRED_OUTCOME_DATES",
            "COMPLETE_OUTCOME_PRICE_VOLUME_DATES",
            "EXCLUSION_REASON",
        ).to_dicts(),
        "annual_exit_histories_coverage_only_below_200": symbol_audit.filter(
            pl.col("PRICE_TICKER").is_in(["PXD", "WRK"])
        ).select(
            "PRICE_TICKER",
            "VERIFIED_LAST_DATE",
            pl.col("PRIMARY_PRICE_OBSERVATIONS").alias(
                "ANNUAL_OUTCOME_PRICE_DATES"
            ),
            "CORPORATE_ACTION_STATUS",
        ).to_dicts(),
        "pre_common_dates_min": int(pre_common.min()),
        "outcome_common_dates_min": int(outcome_common.min()),
        "outcome_common_dates_max": int(outcome_common.max()),
        "raw_outcome_common_dates_min": int(raw_common.min()),
        "raw_outcome_common_dates_max": int(raw_common.max()),
        "numeric_design_all_finite": True,
        "outcome": spec.outcome_label,
        "outcome_estimand_note": (
            "common January 2--December 20 calendar; conditions on observed "
            "history through the frozen endpoint"
            if spec.name == "primary_balanced_2024"
            else (
                "firm-specific verified endpoints with at least 200 dates; "
                "pairwise correlations use the shared history through the "
                "earlier endpoint"
                if spec.name == "partial_to_exit_2024"
                else (
                    "shorter-horizon calendar-Q1 correlation; includes every "
                    "mapped 2024 exit and is not the annual primary estimand"
                )
            )
        ),
        "primary_treatment": "raw Q4-2023 binary reporting-filer cosine",
        "return_convention": (
            "simple adjusted-price returns minus arithmetic daily RF; "
            "arithmetic FF5 factors"
        ),
        "momentum_control": "sum(log1p(simple adjusted-price return))",
        "interpretation": (
            "descriptive association; downstream reporting uses a fixed +0.10 "
            "raw-treatment increment; not causal"
        ),
        "factor_sha256": sha256_file(args.factors),
        "ownership_audit_sha256": sha256_file(ownership_audit_path),
        "price_batch_audit_sha256": sha256_file(audit_paths[0]),
        "price_symbol_audit_sha256": sha256_file(symbol_audit_path),
    }
    output_paths["audit"].write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    run_id = datetime.now(timezone.utc).strftime(
        f"continuous-design-{spec.name}-%Y%m%dT%H%M%SZ"
    )
    record(
        run_id=run_id,
        command=[sys.executable, *sys.argv],
        config={
            "pre_window": ["2023-01-03", "2023-12-29"],
            "design_spec": spec.name,
            "design_role": spec.role,
            "outcome_window": [str(spec.outcome_start), str(spec.outcome_end)],
            "balanced_outcome_calendar": spec.balanced_outcome,
            "outcome_calendar_dates": spec.expected_outcome_dates,
            "minimum_outcome_dates": spec.minimum_outcome_dates,
            "minimum_window_observations": args.minimum_window_observations,
            "outcome": spec.outcome_label,
            "primary_treatment": "raw binary reporting-filer cosine",
            "reporting_increment": 0.10,
            "return_convention": "arithmetic simple returns and factors",
            "momentum_control": "cumulative log return from simple returns",
        },
        inputs=[
            Path(__file__),
            PROJECT_ROOT / "src/provenance.py",
            PROJECT_ROOT / "scripts/build_immutable_price_panel.py",
            PROJECT_ROOT / "scripts/fetch_stockanalysis_fallback.py",
            ownership_audit_path,
            args.ownership_dir / "firm_nodes.parquet",
            args.ownership_dir / "ownership_edges.parquet",
            args.factors,
            *close_paths,
            *volume_paths,
            *audit_paths,
            symbol_audit_path,
        ],
        outputs=list(output_paths.values()),
        status="completed",
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
