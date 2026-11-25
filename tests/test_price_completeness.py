from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import polars as pl
import pytest

from experiments.build_return_design import (
    ReturnDesignSpec,
    daily_simple_returns,
    evaluate_firm_coverage,
    residualize_window,
    validate_market_panels,
    validate_ownership_graph,
    validate_price_batches,
)
from scripts.fetch_yfinance_prices import (
    extract_field,
    observed_tickers,
    replace_column,
)


def test_extract_field_flattens_yfinance_multiindex() -> None:
    dates = pd.date_range("2024-01-02", periods=2)
    columns = pd.MultiIndex.from_product(
        [["Close", "Volume"], ["AAA", "BBB"]], names=["Price", "Ticker"]
    )
    data = pd.DataFrame(
        [[10.0, 20.0, 100.0, 200.0], [11.0, 21.0, 110.0, 210.0]],
        index=dates,
        columns=columns,
    )
    close = extract_field(data, "Close", ["AAA", "BBB"])
    volume = extract_field(data, "Volume", ["AAA", "BBB"])
    assert close.columns.tolist() == ["AAA", "BBB"]
    assert volume.columns.tolist() == ["AAA", "BBB"]
    assert close.loc[dates[1], "BBB"] == 21.0


def test_completeness_requires_both_close_and_volume() -> None:
    index = pd.date_range("2024-01-02", periods=2)
    close = pd.DataFrame({"AAA": [10.0, 11.0], "BBB": [20.0, 21.0]}, index=index)
    volume = pd.DataFrame(
        {"AAA": [100.0, 110.0], "BBB": [np.nan, np.nan]}, index=index
    )
    assert observed_tickers(close, volume, ["AAA", "BBB"]) == ["AAA"]


def test_individual_retry_replaces_failed_column() -> None:
    index = pd.date_range("2024-01-02", periods=2)
    bulk = pd.DataFrame({"AAA": [10.0, 11.0], "BBB": [np.nan, np.nan]}, index=index)
    retry = pd.DataFrame({"BBB": [20.0, 21.0]}, index=index)
    repaired = replace_column(bulk, retry, "BBB")
    assert repaired.columns.tolist() == ["AAA", "BBB"]
    assert repaired["BBB"].tolist() == [20.0, 21.0]


def test_price_batch_gate_requires_exact_audit_alignment(tmp_path) -> None:
    dates = pl.Series("date", ["2024-01-02", "2024-01-03"]).str.to_date()
    close_path = tmp_path / "close_000.parquet"
    volume_path = tmp_path / "volume_000.parquet"
    audit_path = tmp_path / "audit_000.json"
    pl.DataFrame({"date": dates, "AAA": [10.0, 11.0]}).write_parquet(close_path)
    pl.DataFrame({"date": dates, "AAA": [100.0, 110.0]}).write_parquet(
        volume_path
    )
    audit = {
        "schema_version": 2,
        "batch_index": 0,
        "requested": ["AAA"],
        "observed": ["AAA"],
        "missing": [],
        "start": "2024-01-01",
        "end": "2024-01-04",
        "rows": 2,
    }
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    records = validate_price_batches(
        [close_path], [volume_path], [audit_path], {"AAA"}
    )
    assert records == [audit]

    audit["observed"] = []
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(RuntimeError, match="price completeness gate failed"):
        validate_price_batches(
            [close_path], [volume_path], [audit_path], {"AAA"}
        )


def test_market_panel_gate_rejects_nonpositive_close() -> None:
    dates = pl.Series("date", ["2024-01-02", "2024-01-03"]).str.to_date()
    close = pl.DataFrame({"date": dates, "AAA": [10.0, 0.0]})
    volume = pl.DataFrame({"date": dates, "AAA": [100.0, 110.0]})
    with pytest.raises(RuntimeError, match="nonpositive adjusted close"):
        validate_market_panels(close, volume, {"AAA"})


def test_balanced_and_partial_coverage_rules_are_distinct() -> None:
    outcome_dates = [date(2024, 1, 2) + timedelta(days=index) for index in range(5)]
    firms = pl.DataFrame(
        {
            "CIK": ["1", "2", "3"],
            "PRICE_TICKER": ["AAA", "EXIT", "SHORT"],
        }
    )
    close = pl.DataFrame(
        {
            "date": outcome_dates,
            "AAA": [10.0] * 5,
            "EXIT": [20.0, 20.1, 20.2, None, None],
            "SHORT": [30.0] * 5,
        }
    )
    volume = close.select(
        "date",
        pl.lit(100.0).alias("AAA"),
        pl.when(pl.int_range(pl.len()) < 3)
        .then(100.0)
        .otherwise(None)
        .alias("EXIT"),
        pl.lit(100.0).alias("SHORT"),
    )
    symbol_audit = pl.DataFrame(
        {
            "PRICE_TICKER": ["AAA", "EXIT", "SHORT"],
            "SOURCE_KIND": ["frozen"] * 3,
            "SOURCE_ALIAS": ["AAA", "EXIT", "SHORT"],
            "SOURCE_VERSION": ["v1"] * 3,
            "VERIFIED_LAST_DATE": [
                str(outcome_dates[-1]),
                str(outcome_dates[2]),
                str(outcome_dates[-1]),
            ],
            "CORPORATE_ACTION_STATUS": ["survivor", "acquired", "survivor"],
        }
    )
    pre = {"1": {"n_obs": 4}, "2": {"n_obs": 4}, "3": {"n_obs": 1}}
    outcome = {"1": {"n_obs": 5}, "2": {"n_obs": 3}, "3": {"n_obs": 5}}
    balanced = ReturnDesignSpec(
        "balanced",
        "test",
        outcome_dates[0],
        outcome_dates[-1],
        True,
        5,
        5,
        1,
        frozenset(),
        "unused",
        "test",
    )
    partial = ReturnDesignSpec(
        "partial",
        "test",
        outcome_dates[0],
        outcome_dates[-1],
        False,
        5,
        3,
        2,
        frozenset(),
        "unused",
        "test",
    )
    balanced_coverage = evaluate_firm_coverage(
        firms=firms,
        close=close,
        volume=volume,
        pre_diagnostics=pre,
        outcome_diagnostics=outcome,
        symbol_audit=symbol_audit,
        spec=balanced,
        outcome_dates=outcome_dates,
        minimum_window_observations=2,
    )
    partial_coverage = evaluate_firm_coverage(
        firms=firms,
        close=close,
        volume=volume,
        pre_diagnostics=pre,
        outcome_diagnostics=outcome,
        symbol_audit=symbol_audit,
        spec=partial,
        outcome_dates=outcome_dates,
        minimum_window_observations=2,
    )
    assert balanced_coverage.filter(pl.col("ELIGIBLE"))["PRICE_TICKER"].to_list() == [
        "AAA"
    ]
    assert set(
        partial_coverage.filter(pl.col("ELIGIBLE"))["PRICE_TICKER"].to_list()
    ) == {"AAA", "EXIT"}


def test_ownership_graph_gate_requires_canonical_edge_order() -> None:
    firms = pl.DataFrame(
        {
            "NODE_ID": [0, 1, 2],
            "CIK": ["1", "2", "3"],
        }
    )
    edges = pl.DataFrame(
        {
            "NODE_I": [0, 0, 1],
            "NODE_J": [1, 2, 2],
            "CIK_I": ["1", "1", "2"],
            "CIK_J": ["2", "3", "3"],
            "BINARY_COSINE": [0.1, 0.2, 0.3],
            "BINARY_JACCARD": [0.1, 0.2, 0.3],
            "PORTFOLIO_WEIGHT_COSINE": [0.1, 0.2, 0.3],
        }
    )
    validate_ownership_graph(firms, edges)
    with pytest.raises(RuntimeError, match="edge order"):
        validate_ownership_graph(firms, edges.reverse())


def test_daily_simple_returns_are_arithmetic_and_scale_invariant() -> None:
    dates = pl.Series("date", ["2024-01-02", "2024-01-03", "2024-01-04"]).str.to_date()
    close = pl.DataFrame({"date": dates, "AAA": [100.0, 110.0, 99.0]})
    scaled = close.with_columns((7.0 * pl.col("AAA")).alias("AAA"))

    observed = daily_simple_returns(close).get_column("AAA").to_numpy()
    observed_scaled = daily_simple_returns(scaled).get_column("AAA").to_numpy()

    assert np.isnan(observed[0])
    np.testing.assert_allclose(observed[1:], [0.10, -0.10], atol=1e-15)
    np.testing.assert_allclose(observed_scaled, observed, equal_nan=True, atol=1e-15)
    assert not np.isclose(observed[1], np.log(1.10))


def test_ff5_residualization_keeps_arithmetic_units_and_log_momentum() -> None:
    n_dates = 48
    dates = [date(2024, 1, 2) + timedelta(days=index) for index in range(n_dates)]
    index = np.arange(n_dates, dtype=np.float64)
    factor_values = np.column_stack(
        (
            0.012 * np.sin(0.31 * index),
            0.009 * np.cos(0.23 * index),
            0.004 * ((index % 5.0) - 2.0),
            0.006 * np.sin(0.17 * index + 0.4),
            0.005 * np.cos(0.11 * index - 0.2),
        )
    )
    rf = 0.00015 + 0.00004 * np.sin(0.19 * index)
    beta = np.asarray([0.7, -0.4, 0.25, 0.5, -0.3])
    intended_simple_returns = rf + 0.0008 + factor_values @ beta

    prices = np.empty(n_dates, dtype=np.float64)
    prices[0] = 100.0
    for position in range(1, n_dates):
        prices[position] = prices[position - 1] * (1.0 + intended_simple_returns[position])

    close = pl.DataFrame({"date": dates, "AAA": prices})
    factors = pl.DataFrame(
        {
            "date": dates,
            **{
                name: factor_values[:, column]
                for column, name in enumerate(["mkt_rf", "smb", "hml", "rmw", "cma"])
            },
            "rf": rf,
        }
    )
    simple_returns = daily_simple_returns(close)
    residuals, diagnostics = residualize_window(
        simple_returns,
        factors,
        {"AAA": "0000000001"},
        dates[0],
        dates[-1],
    )

    residual_values = residuals.get_column("0000000001").to_numpy()
    valid_residuals = residual_values[np.isfinite(residual_values)]
    assert np.max(np.abs(valid_residuals)) < 1e-12
    diagnostic = diagnostics["0000000001"]
    assert diagnostic["n_obs"] == n_dates - 1
    np.testing.assert_allclose(
        diagnostic["arithmetic_excess_sd"],
        np.std(intended_simple_returns[1:] - rf[1:], ddof=1),
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        diagnostic["cumulative_log_return"],
        np.sum(np.log1p(intended_simple_returns[1:])),
        rtol=0.0,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        diagnostic["cumulative_log_return"],
        np.log(prices[-1] / prices[0]),
        rtol=0.0,
        atol=1e-14,
    )

    mixed_log_returns = close.select(
        "date", pl.col("AAA").log().diff().alias("AAA")
    )
    mixed_residuals, _ = residualize_window(
        mixed_log_returns,
        factors,
        {"AAA": "0000000001"},
        dates[0],
        dates[-1],
    )
    mixed_values = mixed_residuals.get_column("0000000001").to_numpy()
    mixed_valid = mixed_values[np.isfinite(mixed_values)]
    assert np.max(np.abs(mixed_valid)) > 1e-7
