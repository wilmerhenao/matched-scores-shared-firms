from __future__ import annotations

import polars as pl
import pytest

from experiments.run_canonical_simulation import generate_plr_dgp
from experiments.run_continuous_dml import (
    PRIMARY_ALPHA,
    PRIMARY_N_FOLDS,
    PRIMARY_SEEDS,
    RIDGE_SOLVER_TOLERANCE,
    ridge_factory,
)
from experiments.run_repeated_split_calibration import (
    N_FOLDS,
    RIDGE_ALPHA,
    calibration_tasks,
    one_replication,
    summarize_replications,
)
from src.dyadic_dml import fit_plr_dml, make_node_folds


def test_application_calibration_seed_set_is_fixed_but_dimensions_are_bound() -> None:
    assert len(PRIMARY_SEEDS) == 25
    assert len(set(PRIMARY_SEEDS)) == 25
    tasks = calibration_tasks(
        n_nodes=43,
        n_controls=17,
        replications=3,
        seed=20260808,
        fold_seeds=(11, 29, 47),
    )
    assert len(tasks) == 3
    assert {(item[0], item[1]) for item in tasks} == {(43, 17)}
    assert all(item[4] == (11, 29, 47) for item in tasks)
    assert len({item[3] for item in tasks}) == 3
    with pytest.raises(ValueError, match="two nodes per score fold"):
        calibration_tasks(
            n_nodes=2 * N_FOLDS - 1,
            n_controls=17,
            replications=2,
            seed=1,
            fold_seeds=(11,),
        )
    with pytest.raises(ValueError, match="nonnegative"):
        calibration_tasks(
            n_nodes=30,
            n_controls=17,
            replications=2,
            seed=-1,
            fold_seeds=(11,),
        )


def test_small_repeated_replication_applies_dispersion_rule() -> None:
    result = one_replication((30, 17, 0, 20260809, (11, 29, 47)))
    assert result["n_splits"] == 3
    assert result["n_scored_per_split"] > 0
    assert float(result["max_abs_split_moment"]) < 1e-12
    for split in ("strict", "oracle"):
        assert float(result[f"{split}_se_dyadic"]) > 0.0
        assert float(result[f"{split}_aggregate_to_run_se_ratio"]) > 0.0
        assert float(result[f"{split}_theta_min"]) <= float(
            result[f"{split}_theta"]
        ) <= float(result[f"{split}_theta_max"])


def test_learned_arm_exactly_reuses_application_ridge_rule() -> None:
    assert N_FOLDS == PRIMARY_N_FOLDS
    assert RIDGE_ALPHA == PRIMARY_ALPHA
    pipeline = ridge_factory(RIDGE_ALPHA)
    assert pipeline.get_params()["ridge__solver"] == "lsqr"
    assert pipeline.get_params()["ridge__tol"] == RIDGE_SOLVER_TOLERANCE

    n_nodes = 30
    n_controls = 17
    dgp_seed = 20260809
    fold_seed = 11
    y, d, x, row, col = generate_plr_dgp(n_nodes, n_controls, dgp_seed)
    folds = make_node_folds(n_nodes, N_FOLDS, seed=fold_seed)
    direct = fit_plr_dml(
        y,
        d,
        x,
        row,
        col,
        n_nodes,
        n_folds=N_FOLDS,
        fold_assignment=folds,
        outcome_factory=lambda: ridge_factory(RIDGE_ALPHA),
        treatment_factory=lambda: ridge_factory(RIDGE_ALPHA),
    )
    calibrated = one_replication(
        (n_nodes, n_controls, 0, dgp_seed, (fold_seed,))
    )
    assert float(calibrated["strict_theta"]) == pytest.approx(
        direct.theta, abs=1e-14
    )
    assert float(calibrated["strict_se_dyadic"]) == pytest.approx(
        direct.se_dyadic, abs=1e-14
    )


def test_repeated_summary_reports_coverage_and_empirical_criticals() -> None:
    rows = [
        one_replication((30, 17, repetition, 100 + repetition, (11, 29, 47)))
        for repetition in range(3)
    ]
    summary = summarize_replications(pl.DataFrame(rows))
    assert summary["replications"] == 3
    for split in ("strict", "oracle"):
        cell = summary[split]
        assert 0.0 <= cell["coverage90"] <= 1.0
        assert 0.0 <= cell["coverage95"] <= 1.0
        assert cell["coverage90_successes"] in range(4)
        assert cell["coverage95_successes"] in range(4)
        assert cell["mean_mcse"] > 0.0
        assert cell["bias_mcse"] == cell["mean_mcse"]
        assert cell["rejection5"] == pytest.approx(1.0 - cell["coverage95"])
        assert cell["abs_t_quantile95"] >= cell["abs_t_quantile90"]


def test_repeated_summary_fails_closed_on_duplicate_seed_or_invalid_se() -> None:
    rows = [
        one_replication((30, 17, repetition, 100 + repetition, (11,)))
        for repetition in range(2)
    ]
    duplicate = pl.DataFrame([rows[0], {**rows[1], "dgp_seed": rows[0]["dgp_seed"]}])
    with pytest.raises(ValueError, match="duplicate dgp_seed"):
        summarize_replications(duplicate)

    invalid = pl.DataFrame(
        [rows[0], {**rows[1], "strict_se_dyadic": float("nan")}]
    )
    with pytest.raises(ValueError, match="estimates or SEs are invalid"):
        summarize_replications(invalid)
