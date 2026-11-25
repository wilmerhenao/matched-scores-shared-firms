from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from experiments.application_design_contract import ApplicationDesignError
from experiments.run_continuous_dml import PRIMARY_N_FOLDS, PRIMARY_SEEDS
from experiments.run_ridge_equal_count_audit import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SWAP_SALTS,
    SOURCE_DEPENDENCIES,
    application_ridge_factory,
    run_seed,
    summarize,
    validate_execution_role,
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


def test_provenance_closes_over_every_local_source_dependency() -> None:
    root = SOURCE_DEPENDENCIES[0].parents[1]
    assert {str(path.relative_to(root)) for path in SOURCE_DEPENDENCIES} == {
        "experiments/application_design_contract.py",
        "experiments/run_continuous_dml.py",
        "experiments/run_ridge_equal_count_audit.py",
        "src/dyadic_dml.py",
        "src/provenance.py",
        "pyproject.toml",
    }
    assert all(path.is_file() for path in SOURCE_DEPENDENCIES)


def test_application_factory_is_the_frozen_primary_ridge_pipeline() -> None:
    pipeline = application_ridge_factory()
    assert list(pipeline.named_steps) == ["preprocess", "ridge"]
    ridge = pipeline.named_steps["ridge"]
    assert ridge.alpha == 1.0
    assert ridge.solver == "lsqr"
    assert ridge.tol == 1e-12


def test_ridge_audit_matches_scores_and_training_counts_and_suppresses_inference() -> None:
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
    assert len({item["score_edge_fingerprints_json"] for item in rows}) == 1
    assert len({item["score_edge_counts_json"] for item in rows}) == 1
    strict = next(item for item in rows if item["method"] == "strict")
    full = next(item for item in rows if item["method"] == "full_leaky")
    equal = [item for item in rows if item["method"] == "equal_size_leaky"]
    assert full["train_edges_min"] > strict["train_edges_max"]
    assert strict["cross_boundary_train_fraction"] == 0.0
    assert strict["standard_error"] is not None
    assert strict["p_value_normal"] is not None
    assert strict["valid_under_maintained_assumptions"] is True
    for item in equal:
        assert item["train_edge_counts_json"] == strict["train_edge_counts_json"]
        assert item["cross_boundary_train_fraction"] > 0.0
        assert item["train_edge_fingerprints_json"] is not None
    assert len({item["train_edge_fingerprints_json"] for item in equal}) == len(equal)
    for item in [full, *equal]:
        assert item["standard_error_estimation_scale"] is None
        assert item["standard_error"] is None
        assert item["t_statistic"] is None
        assert item["p_value_normal"] is None
        assert item["confidence_interval_low"] is None
        assert item["confidence_interval_high"] is None
        assert item["valid_under_maintained_assumptions"] is False
        assert item["inference_role"] == "unsupported_endpoint_exposure_diagnostic"
    for item in rows:
        assert item["n_folds"] == PRIMARY_N_FOLDS
        assert item["score_outcome_residual_mse"] > 0.0
        assert item["score_treatment_residual_mse"] > 0.0
        assert item["valid_primary_inference"] is False
        assert item["designated_primary_result"] is False
        assert item["diagnostic_only"] is True

    summary = summarize(rows, (11,), DEFAULT_SWAP_SALTS)
    assert len(summary["cells"]) == len(rows)
    assert len(summary["paired_differences_from_strict"]) == 1 + len(
        DEFAULT_SWAP_SALTS
    )
    strict_cell = next(
        item for item in summary["cells"] if item["method"] == "strict"
    )
    assert strict_cell["standard_error"] is not None
    for cell in summary["cells"]:
        if cell["method"] != "strict":
            assert cell["standard_error"] is None
            assert cell["t_statistic"] is None
            assert cell["p_value_normal"] is None
    for paired in summary["paired_differences_from_strict"]:
        assert "mean_tau_difference" in paired
        assert "median_tau_difference" in paired
        assert "mean_score_outcome_residual_mse_difference" in paired
        assert "median_score_treatment_residual_mse_difference" in paired


def test_primary_output_role_gate_rejects_partial_grid(tmp_path) -> None:
    primary = SimpleNamespace(
        design_status="VALIDATED_FOR_ESTIMATION",
        design_role="primary",
        is_primary_default=True,
        design_spec="primary_balanced_2024",
    )
    with pytest.raises(ValueError, match="exact PRIMARY_SEEDS"):
        validate_execution_role(
            primary,
            allow_sensitivity_design=False,
            output_dir=DEFAULT_OUTPUT_DIR,
            seeds=(PRIMARY_SEEDS[0],),
            swap_salts=DEFAULT_SWAP_SALTS,
        )
    validate_execution_role(
        primary,
        allow_sensitivity_design=False,
        output_dir=tmp_path,
        seeds=(PRIMARY_SEEDS[0],),
        swap_salts=(DEFAULT_SWAP_SALTS[0],),
    )


def test_sensitivity_role_requires_flag_and_distinct_output(tmp_path) -> None:
    sensitivity = SimpleNamespace(
        design_status="VALIDATED_SENSITIVITY_ONLY",
        design_role="partial_to_exit_sensitivity",
        is_primary_default=False,
        design_spec="partial_to_exit_2024",
    )
    with pytest.raises(ApplicationDesignError, match="--allow-sensitivity-design"):
        validate_execution_role(
            sensitivity,
            allow_sensitivity_design=False,
            output_dir=tmp_path,
            seeds=(11,),
            swap_salts=(DEFAULT_SWAP_SALTS[0],),
        )
    with pytest.raises(ApplicationDesignError, match="primary default output"):
        validate_execution_role(
            sensitivity,
            allow_sensitivity_design=True,
            output_dir=DEFAULT_OUTPUT_DIR,
            seeds=PRIMARY_SEEDS,
            swap_salts=DEFAULT_SWAP_SALTS,
        )
    validate_execution_role(
        sensitivity,
        allow_sensitivity_design=True,
        output_dir=tmp_path,
        seeds=(11,),
        swap_salts=(DEFAULT_SWAP_SALTS[0],),
    )
