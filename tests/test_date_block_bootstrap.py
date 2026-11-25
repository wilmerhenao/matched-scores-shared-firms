from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from experiments.run_continuous_dml import (
    PRIMARY_ALPHA,
    PRIMARY_N_FOLDS,
    PRIMARY_OUTCOME,
    PRIMARY_REPORTING_INCREMENT,
    PRIMARY_TREATMENT,
    treatment_reporting_scale,
)
from experiments.run_date_block_bootstrap import (
    REPRODUCIBILITY_NODE_FOLD_SEED,
    circular_block_indices,
    select_reproducibility_estimate,
    summarize_block_draws,
    validate_block_inputs,
)


def test_circular_blocks_are_reproducible_bounded_and_contiguous() -> None:
    first = circular_block_indices(10, 4, np.random.default_rng(123))
    second = circular_block_indices(10, 4, np.random.default_rng(123))
    assert np.array_equal(first, second)
    assert first.shape == (10,)
    assert np.all((0 <= first) & (first < 10))
    for index in range(1, first.size):
        if index % 4:
            assert first[index] == (first[index - 1] + 1) % 10

    with pytest.raises(ValueError, match="at least two dates"):
        circular_block_indices(1, 1, np.random.default_rng(1))
    with pytest.raises(ValueError, match="block_length"):
        circular_block_indices(10, 11, np.random.default_rng(1))


def test_seed_29_reference_selection_is_exact_and_unique() -> None:
    row = {
        "specification": "primary_seed_sensitivity",
        "seed": REPRODUCIBILITY_NODE_FOLD_SEED,
        "split": "strict_node",
        "variance": "dyadic",
        "ridge_alpha": PRIMARY_ALPHA,
        "treatment": PRIMARY_TREATMENT,
        "outcome": PRIMARY_OUTCOME,
        "n_folds": PRIMARY_N_FOLDS,
        "reporting_increment": PRIMARY_REPORTING_INCREMENT,
        "effect_unit": treatment_reporting_scale(PRIMARY_TREATMENT).effect_unit,
        "valid_under_maintained_assumptions": True,
        "contributes_to_designated_primary_aggregate": True,
        "designated_primary_result": False,
        "valid_primary_inference": False,
        "theta": 0.01,
        "standard_error": 0.02,
        "theta_estimation_scale": 0.1,
        "standard_error_estimation_scale": 0.2,
        "n_nodes": 10,
        "n_edges": 45,
    }
    estimates = pl.DataFrame([row])
    selected = select_reproducibility_estimate(estimates)
    assert selected["seed"] == 29
    assert selected["theta"] == 0.01
    assert selected["theta_estimation_scale"] == 0.1

    with pytest.raises(ValueError, match="exactly one"):
        select_reproducibility_estimate(pl.concat([estimates, estimates]))


def test_block_summary_contains_no_gaussian_mixture() -> None:
    draws = pl.DataFrame(
        {
            "block_length": [5, 5, 5, 20, 20, 20],
            "repetition": [0, 1, 2, 0, 1, 2],
            "theta_effect_per_0_10_raw": [
                0.01,
                0.02,
                0.03,
                -0.01,
                0.00,
                0.01,
            ],
            "conditional_dyadic_se_per_0_10_raw": [
                0.04,
                0.05,
                0.06,
                0.03,
                0.04,
                0.05,
            ],
        }
    )
    summaries = summarize_block_draws(draws, [20, 5], repetitions=3)
    assert [item["block_length"] for item in summaries] == [5, 20]
    assert summaries[0]["theta_median"] == 0.02
    assert summaries[0]["conditional_dyadic_se_median"] == 0.05
    assert summaries[0]["theta_sample_sd_descriptive"] == pytest.approx(0.01)
    assert summaries[0]["conditional_interval_crosses_zero_fraction"] == 1.0
    assert (
        summaries[0]["conditional_interval_excludes_zero_positive_fraction"]
        == 0.0
    )
    expected_node_variance = np.mean(np.asarray([0.04, 0.05, 0.06]) ** 2)
    expected_date_variance = np.var([0.01, 0.02, 0.03], ddof=1)
    assert summaries[0][
        "exploratory_mean_conditional_node_variance_component"
    ] == pytest.approx(expected_node_variance)
    assert summaries[0]["exploratory_date_variance_component"] == pytest.approx(
        expected_date_variance
    )
    assert summaries[0]["exploratory_total_variance"] == pytest.approx(
        expected_node_variance + expected_date_variance
    )
    assert "not a formal" in summaries[0]["exploratory_total_variance_status"]
    assert not any(
        "combined" in key
        for summary in summaries
        for key in summary
    )


def test_seed_29_selector_rejects_a_reporting_scale_mismatch() -> None:
    reporting = treatment_reporting_scale(PRIMARY_TREATMENT)
    estimates = pl.DataFrame(
        [
            {
                "specification": "primary_seed_sensitivity",
                "seed": REPRODUCIBILITY_NODE_FOLD_SEED,
                "split": "strict_node",
                "variance": "dyadic",
                "ridge_alpha": PRIMARY_ALPHA,
                "treatment": PRIMARY_TREATMENT,
                "outcome": PRIMARY_OUTCOME,
                "n_folds": PRIMARY_N_FOLDS,
                "reporting_increment": PRIMARY_REPORTING_INCREMENT,
                "effect_unit": reporting.effect_unit,
                "valid_under_maintained_assumptions": True,
                "contributes_to_designated_primary_aggregate": True,
                "designated_primary_result": False,
                "valid_primary_inference": False,
                "theta": 0.01,
                "standard_error": 0.02,
                "theta_estimation_scale": 0.09,
                "standard_error_estimation_scale": 0.2,
                "n_nodes": 10,
                "n_edges": 45,
            }
        ]
    )
    with pytest.raises(ValueError, match="theta violates"):
        select_reproducibility_estimate(estimates)


def test_block_input_gate_reproduces_the_stored_outcome() -> None:
    firms = pl.DataFrame(
        {"NODE_ID": [0, 1, 2], "CIK": ["A", "B", "C"]}
    )
    start = date(2024, 1, 2)
    residual_values = np.asarray(
        [
            [0.1, 0.4, -0.2],
            [0.2, 0.1, 0.3],
            [-0.1, 0.3, 0.2],
            [0.4, -0.2, 0.1],
            [0.0, 0.2, -0.1],
        ]
    )
    residuals = pl.DataFrame(
        {
            "date": [start + timedelta(days=index) for index in range(5)],
            "A": residual_values[:, 0],
            "B": residual_values[:, 1],
            "C": residual_values[:, 2],
        }
    )
    row, col = np.triu_indices(3, k=1)
    correlation = np.corrcoef(residual_values, rowvar=False)
    design = pl.DataFrame(
        {
            "NODE_I": row,
            "NODE_J": col,
            PRIMARY_OUTCOME: np.arctanh(correlation[row, col]),
        }
    )
    _, observed, observed_row, observed_col = validate_block_inputs(
        design, residuals, firms
    )
    assert np.array_equal(observed, residual_values)
    assert np.array_equal(observed_row, row)
    assert np.array_equal(observed_col, col)

    corrupted = design.with_columns((pl.col(PRIMARY_OUTCOME) + 0.01))
    with pytest.raises(ValueError, match="do not reproduce"):
        validate_block_inputs(corrupted, residuals, firms)
