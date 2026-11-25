from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from experiments.run_hgb_learner_audit import (
    DEFAULT_SWAP_SALTS,
    HGB_SOURCE_DEPENDENCIES,
    HGB_RANDOM_STATE,
    hgb_factory,
    run_seed,
    summarize,
)


def linear_factory() -> LinearRegression:
    return LinearRegression()


def synthetic_design(n_nodes: int = 30, seed: int = 20260809):
    rng = np.random.default_rng(seed)
    row, col = np.triu_indices(n_nodes, k=1)
    node_x = rng.normal(size=(n_nodes, 3))
    x = np.column_stack(
        (
            node_x[row, 0] + node_x[col, 0],
            np.abs(node_x[row, 1] - node_x[col, 1]),
            node_x[row, 2] * node_x[col, 2],
        )
    )
    d = 0.4 * x[:, 0] - 0.2 * x[:, 1] + rng.normal(size=row.size)
    node_error = rng.normal(size=n_nodes)
    y = 0.5 * d + 0.3 * x[:, 2] + node_error[row] + node_error[col]
    return y, d, x, row.astype(np.int64), col.astype(np.int64)


def test_hgb_factory_is_frozen_to_declared_settings() -> None:
    params = hgb_factory().get_params()
    assert params["max_iter"] == 100
    assert params["learning_rate"] == 0.05
    assert params["max_leaf_nodes"] == 15
    assert params["min_samples_leaf"] == 40
    assert params["l2_regularization"] == 1.0
    assert params["early_stopping"] is False
    assert params["random_state"] == HGB_RANDOM_STATE


def test_hgb_provenance_closes_over_local_source_dependencies() -> None:
    project_root = HGB_SOURCE_DEPENDENCIES[0].parents[1]
    relative_paths = {
        str(path.relative_to(project_root)) for path in HGB_SOURCE_DEPENDENCIES
    }
    assert relative_paths == {
        "experiments/application_design_contract.py",
        "experiments/run_hgb_learner_audit.py",
        "experiments/run_continuous_dml.py",
        "src/dyadic_dml.py",
        "src/provenance.py",
        "pyproject.toml",
    }
    assert all(path.is_file() for path in HGB_SOURCE_DEPENDENCIES)


def test_run_seed_matches_scores_rows_and_exports_score_residual_diagnostics() -> None:
    n_nodes = 30
    y, d, x, row, col = synthetic_design(n_nodes)
    rows = run_seed(
        y,
        d,
        x,
        row,
        col,
        n_nodes,
        seed=11,
        swap_salts=DEFAULT_SWAP_SALTS,
        learner_factory=linear_factory,
    )

    assert len(rows) == 2 + len(DEFAULT_SWAP_SALTS)
    assert {item["method"] for item in rows} == {
        "strict",
        "full_leaky",
        "equal_size_leaky",
    }
    assert len({item["score_edge_fingerprints_json"] for item in rows}) == 1
    strict = next(item for item in rows if item["method"] == "strict")
    equal_size = [item for item in rows if item["method"] == "equal_size_leaky"]
    for item in equal_size:
        assert item["train_edges_min"] == strict["train_edges_min"]
        assert item["train_edges_max"] == strict["train_edges_max"]
        assert item["cross_boundary_train_fraction"] > 0.0
    full = next(item for item in rows if item["method"] == "full_leaky")
    assert full["train_edges_min"] > strict["train_edges_max"]
    assert strict["cross_boundary_train_fraction"] == 0.0
    assert strict["valid_under_maintained_assumptions"] is True
    assert strict["standard_error"] == strict["mechanical_dyadic_se"]
    assert strict["t_statistic"] is not None
    assert strict["p_value_normal"] is not None
    for item in rows:
        assert item["n_scored"] == strict["n_scored"]
        assert abs(float(item["moment"])) < 1e-12
        assert np.isclose(
            item["jacobian"], item["score_treatment_residual_mse"], atol=1e-15
        )
        assert item["score_outcome_residual_mse"] > 0.0
        assert item["score_treatment_residual_sd"] > 0.0
        assert item["mechanical_dyadic_se"] > 0.0
        assert item["designated_primary_result"] is False
        assert item["valid_primary_inference"] is False
        if item["method"] != "strict":
            assert item["valid_under_maintained_assumptions"] is False
            assert item["standard_error_estimation_scale"] is None
            assert item["standard_error"] is None
            assert item["t_statistic"] is None
            assert item["p_value_normal"] is None

    summary = summarize(rows, (11,), DEFAULT_SWAP_SALTS)
    assert len(summary["cells"]) == len(rows)
    assert len(summary["paired_differences_from_strict"]) == 1 + len(
        DEFAULT_SWAP_SALTS
    )
    strict_cell = next(
        item for item in summary["cells"] if item["method"] == "strict"
    )
    assert strict_cell["standard_error"] == strict_cell[
        "mechanical_repeated_split_dyadic_se"
    ]
    for cell in summary["cells"]:
        assert cell["designated_primary_result"] is False
        if cell["method"] != "strict":
            assert cell["standard_error"] is None
            assert cell["t_statistic"] is None
            assert cell["p_value_normal"] is None
            assert cell["mechanical_repeated_split_dyadic_se"] > 0.0
    for paired in summary["paired_differences_from_strict"]:
        assert paired["n_paired_partitions"] == 1
        assert "median_score_outcome_residual_mse_difference" in paired
        assert "median_score_treatment_residual_mse_difference" in paired
        assert "median_jacobian_difference" in paired
        assert 0.0 <= paired["tau_difference_fraction_positive"] <= 1.0


def test_hgb_run_seed_rejects_invalid_partition_randomization() -> None:
    y, d, x, row, col = synthetic_design(30)
    with pytest.raises(ValueError, match="seed must be nonnegative"):
        run_seed(
            y,
            d,
            x,
            row,
            col,
            30,
            seed=-1,
            learner_factory=linear_factory,
        )
    with pytest.raises(ValueError, match="nonempty, unique, and nonnegative"):
        run_seed(
            y,
            d,
            x,
            row,
            col,
            30,
            seed=11,
            swap_salts=(1, 1),
            learner_factory=linear_factory,
        )
