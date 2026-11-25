from __future__ import annotations

import numpy as np
import pytest

from experiments.run_full_graph_ols_diagnostic import (
    REDUNDANT_CONTROL_NAMES,
    full_graph_ols_diagnostic,
    two_sided_normal_power,
)


def synthetic_complete_graph(seed: int = 7):
    rng = np.random.default_rng(seed)
    n_nodes = 20
    row, col = np.triu_indices(n_nodes, k=1)
    node_x = rng.normal(size=(n_nodes, 2))
    pair_x = rng.normal(size=(row.size, 2))
    x = node_x[row] + node_x[col] + pair_x
    treatment_node = rng.normal(size=n_nodes)
    d = 0.3 * x[:, 0] - 0.2 * x[:, 1]
    d += treatment_node[row] + treatment_node[col] + rng.normal(size=row.size)
    outcome_node = rng.normal(size=n_nodes)
    y = 0.6 * d + 0.4 * x[:, 0] + 0.1 * x[:, 1]
    y += outcome_node[row] + outcome_node[col] + rng.normal(size=row.size)
    return y, d, x, row, col, n_nodes


def test_full_graph_ols_reports_exact_overlap_and_first_order_meats() -> None:
    y, d, x, row, col, n_nodes = synthetic_complete_graph()
    result = full_graph_ols_diagnostic(
        y,
        d,
        x,
        row,
        col,
        n_nodes=n_nodes,
        reporting_increment=0.1,
    )
    assert abs(float(result["score_moment"])) < 1e-12
    assert result["asa_cr0_meat"] == pytest.approx(
        result["node_meat"] - result["edge_diagonal_meat"]
    )
    estimates = result["estimates"]
    assert estimates["asa_dyadic_cr0"]["standard_error"] > 0.0
    assert estimates["chiang_first_order"]["standard_error"] > 0.0
    assert estimates["iid_hc0"]["standard_error"] > 0.0
    assert result["jacobian_mean_residual_treatment_square"] > 0.0
    assert result["reporting_increment_in_residual_treatment_sd"] > 0.0


def test_full_graph_gate_rejects_incomplete_edges() -> None:
    y, d, x, row, col, n_nodes = synthetic_complete_graph()
    with pytest.raises(ValueError, match="complete undirected"):
        full_graph_ols_diagnostic(
            y[:-1],
            d[:-1],
            x[:-1],
            row[:-1],
            col[:-1],
            n_nodes=n_nodes,
            reporting_increment=0.1,
        )


def test_power_is_monotone_in_signal_to_noise() -> None:
    low = two_sided_normal_power(0.01, 0.02, 0.05)
    high = two_sided_normal_power(0.02, 0.01, 0.05)
    assert 0.0 < low < high < 1.0


def test_ols_redundancy_contract_tracks_only_retained_endpoint_summaries() -> None:
    assert REDUNDANT_CONTROL_NAMES == (
        "numeric:PRE_VOL_ABS_DIFF",
        "numeric:PRE_MOMENTUM_ABS_DIFF",
        "numeric:LOG_HOLDERS_ABS_DIFF",
    )
