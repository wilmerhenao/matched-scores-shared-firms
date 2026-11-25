from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from experiments.run_continuous_dml import (
    CONTINUOUS_DML_SCHEMA_VERSION,
    MEASUREMENT_CELLS,
    NUMERIC_CONTROLS,
    PRIMARY_ALPHA,
    PRIMARY_N_FOLDS,
    PRIMARY_OUTCOME,
    PRIMARY_REPORTING_INCREMENT,
    PRIMARY_SEEDS,
    PRIMARY_TREATMENT,
    REPEATED_SPLIT_FORMULA,
    RIDGE_SENSITIVITY_ALPHAS,
    SECTOR_PAIR_CATEGORIES,
    SECTOR_PAIR_REFERENCE,
    build_repeated_split_aggregates,
    design_matrix,
    node_variance_rows,
    repeated_split_summary,
    result_rows,
    ridge_factory,
    ridge_sensitivity_configurations,
    treatment_reporting_scale,
)
from src.dyadic_dml import fit_plr_dml


def repeated_split_frame(seeds: tuple[int, ...]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "specification": "primary_seed_sensitivity",
                "split": "strict_node",
                "variance": "dyadic",
                "inference_role": "maintained_assumptions_inference",
                "valid_under_maintained_assumptions": True,
                "contributes_to_designated_primary_aggregate": False,
                "designated_primary_result": False,
                "valid_primary_inference": False,
                "ridge_alpha": 1.0,
                "treatment": "D",
                "outcome": "Y",
                "reporting_increment": 0.1,
                "effect_unit": "effect of +0.10 D",
                "estimation_unit": "effect of +1.00 D",
                "n_nodes": 20,
                "n_edges": 190,
                "n_folds": 5,
                "seed": seed,
                "theta": theta,
                "standard_error": standard_error,
                "theta_estimation_scale": theta / 0.1,
                "standard_error_estimation_scale": standard_error / 0.1,
                "n_scored": 30,
            }
            for seed, theta, standard_error in zip(
                seeds,
                (0.4, 1.0, 1.6),
                (0.2, 0.3, 0.4),
                strict=True,
            )
        ]
    )


def test_repeated_split_summary_uses_fixed_seed_set_and_total_variance_rule():
    seeds = (11, 29, 47)
    summary = repeated_split_summary(
        repeated_split_frame(seeds),
        analysis_group="test",
        expected_seeds=seeds,
    )

    expected_variance = np.median(
        np.array([0.2, 0.3, 0.4]) ** 2
        + (np.array([0.4, 1.0, 1.6]) - 1.0) ** 2
    )
    assert summary["theta"] == 1.0
    assert np.isclose(summary["standard_error"], np.sqrt(expected_variance))
    assert summary["theta_estimation_scale"] == 10.0
    assert np.isclose(
        summary["standard_error"],
        0.1 * summary["standard_error_estimation_scale"],
    )
    assert summary["aggregation_formula"] == REPEATED_SPLIT_FORMULA
    assert summary["n_splits"] == 3
    assert summary["valid_under_maintained_assumptions"] is True
    assert summary["designated_primary_result"] is False
    assert summary["valid_primary_inference"] is False


def test_exactly_the_headline_aggregate_is_designated_primary() -> None:
    seeds = (11, 29, 47)
    primary = repeated_split_frame(seeds).with_columns(
        pl.lit(PRIMARY_TREATMENT).alias("treatment"),
        pl.lit(PRIMARY_OUTCOME).alias("outcome"),
        pl.lit(True).alias("contributes_to_designated_primary_aggregate"),
    )
    headline = repeated_split_summary(
        primary,
        analysis_group="primary_split_by_variance",
        expected_seeds=seeds,
    )
    reused_sensitivity = repeated_split_summary(
        primary,
        analysis_group="strict_k_sensitivity",
        expected_seeds=seeds,
    )

    assert CONTINUOUS_DML_SCHEMA_VERSION == 4
    assert headline["valid_under_maintained_assumptions"] is True
    assert headline["designated_primary_result"] is True
    assert headline["valid_primary_inference"] is True
    assert headline["inference_role"] == "designated_primary_inference"
    assert reused_sensitivity["valid_under_maintained_assumptions"] is True
    assert reused_sensitivity["designated_primary_result"] is False
    assert reused_sensitivity["valid_primary_inference"] is False
    assert (
        reused_sensitivity["contributes_to_designated_primary_aggregate"] is False
    )
    assert reused_sensitivity["inference_role"] == (
        "maintained_assumptions_sensitivity"
    )


def _aggregate_dummy_result(estimator: str, n_folds: int, seed: int) -> SimpleNamespace:
    return SimpleNamespace(
        estimator=estimator,
        theta=0.40 + seed / 10_000.0,
        se_iid=0.20,
        se_dyadic=0.30,
        n_nodes=20,
        n_edges=190,
        n_scored=30,
        n_folds=n_folds,
        moment=0.0,
        jacobian=0.25,
        diagnostics={
            "score_definition": "synthetic aggregate regression",
            "score_edge_fingerprints": ["synthetic"],
        },
    )


def _full_synthetic_result_frame() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for seed in PRIMARY_SEEDS:
        for estimator in (
            "plr",
            "plr_matched_score_leaky_complement_diagnostic",
            "plr_random_edge_diagnostic",
        ):
            rows.extend(
                result_rows(
                    _aggregate_dummy_result(estimator, PRIMARY_N_FOLDS, seed),
                    seed=seed,
                    alpha=PRIMARY_ALPHA,
                    treatment=PRIMARY_TREATMENT,
                    outcome=PRIMARY_OUTCOME,
                    specification="primary_seed_sensitivity",
                )
            )
        for n_folds in (2, 10):
            rows.extend(
                result_rows(
                    _aggregate_dummy_result("plr", n_folds, seed),
                    seed=seed,
                    alpha=PRIMARY_ALPHA,
                    treatment=PRIMARY_TREATMENT,
                    outcome=PRIMARY_OUTCOME,
                    specification="fold_count_sensitivity",
                )
            )
        for alpha in RIDGE_SENSITIVITY_ALPHAS:
            if alpha == PRIMARY_ALPHA:
                continue
            rows.extend(
                result_rows(
                    _aggregate_dummy_result("plr", PRIMARY_N_FOLDS, seed),
                    seed=seed,
                    alpha=alpha,
                    treatment=PRIMARY_TREATMENT,
                    outcome=PRIMARY_OUTCOME,
                    specification="ridge_penalty_sensitivity",
                )
            )
        for specification, treatment, outcome in MEASUREMENT_CELLS:
            rows.extend(
                result_rows(
                    _aggregate_dummy_result("plr", PRIMARY_N_FOLDS, seed),
                    seed=seed,
                    alpha=PRIMARY_ALPHA,
                    treatment=treatment,
                    outcome=outcome,
                    specification=specification,
                )
            )
    return pl.DataFrame(rows)


def test_full_aggregate_construction_separates_source_contribution_and_run_role() -> None:
    results = _full_synthetic_result_frame()
    primary = build_repeated_split_aggregates(results, design_is_primary=True)
    sensitivity = build_repeated_split_aggregates(results, design_is_primary=False)

    assert primary.height == sensitivity.height == 28
    assert primary.filter(pl.col("designated_primary_result")).height == 1
    assert sensitivity.filter(pl.col("designated_primary_result")).height == 0
    assert primary.filter(
        pl.col("contributes_to_designated_primary_aggregate")
    ).height == 1
    assert sensitivity.filter(
        pl.col("contributes_to_designated_primary_aggregate")
    ).height == 1

    for frame in (primary, sensitivity):
        alpha_one_ridge = frame.filter(
            (pl.col("analysis_group") == "ridge_penalty_sensitivity")
            & (pl.col("ridge_alpha") == PRIMARY_ALPHA)
        )
        k_five = frame.filter(
            (pl.col("analysis_group") == "strict_k_sensitivity")
            & (pl.col("n_folds") == PRIMARY_N_FOLDS)
        )
        assert alpha_one_ridge.height == 2
        assert k_five.height == 2
        assert not alpha_one_ridge.get_column(
            "contributes_to_designated_primary_aggregate"
        ).any()
        assert not k_five.get_column(
            "contributes_to_designated_primary_aggregate"
        ).any()


def test_repeated_split_summary_rejects_seed_substitution():
    frame = repeated_split_frame((11, 29, 47)).with_columns(
        pl.when(pl.col("seed") == 47)
        .then(pl.lit(71))
        .otherwise(pl.col("seed"))
        .alias("seed")
    )
    with np.testing.assert_raises(AssertionError):
        repeated_split_summary(
            frame,
            analysis_group="test",
            expected_seeds=(11, 29, 47),
        )


def _dummy_result() -> SimpleNamespace:
    return SimpleNamespace(
        estimator="plr",
        theta=0.5,
        se_iid=0.2,
        se_dyadic=0.3,
        n_nodes=2,
        n_edges=1,
        n_scored=1,
        n_folds=2,
        moment=1e-15,
        jacobian=-0.25,
        diagnostics={
            "score_definition": "test score",
            "score_edge_fingerprints": ["abc", "def"],
            "node_variance_contributions": [0.01, 0.08],
        },
    )


def test_raw_primary_is_reported_per_point_one_without_changing_t_statistics():
    rows = result_rows(
        _dummy_result(),
        seed=11,
        alpha=PRIMARY_ALPHA,
        treatment=PRIMARY_TREATMENT,
        outcome="Y",
        specification="test",
    )
    iid = next(row for row in rows if row["variance"] == "iid")
    dyadic = next(row for row in rows if row["variance"] == "dyadic")

    assert PRIMARY_TREATMENT == "BINARY_COSINE"
    assert iid["reporting_increment"] == PRIMARY_REPORTING_INCREMENT == 0.1
    assert iid["theta"] == pytest.approx(0.05)
    assert iid["standard_error"] == pytest.approx(0.02)
    assert iid["theta_estimation_scale"] == pytest.approx(0.5)
    assert iid["standard_error_estimation_scale"] == pytest.approx(0.2)
    assert iid["t_statistic"] == pytest.approx(2.5)
    assert dyadic["t_statistic"] == pytest.approx(0.5 / 0.3)
    assert dyadic["ci_lower_95"] == pytest.approx(0.05 - 1.96 * 0.03)
    assert dyadic["ci_upper_95"] == pytest.approx(0.05 + 1.96 * 0.03)
    assert iid["valid_under_maintained_assumptions"] is False
    assert dyadic["valid_under_maintained_assumptions"] is True
    assert dyadic["designated_primary_result"] is False
    assert dyadic["valid_primary_inference"] is False


def test_primary_partitions_only_contribute_to_the_reported_aggregate() -> None:
    result = _dummy_result()
    result.n_folds = 5
    rows = result_rows(
        result,
        seed=11,
        alpha=PRIMARY_ALPHA,
        treatment=PRIMARY_TREATMENT,
        outcome=PRIMARY_OUTCOME,
        specification="primary_seed_sensitivity",
    )
    iid = next(row for row in rows if row["variance"] == "iid")
    dyadic = next(row for row in rows if row["variance"] == "dyadic")
    assert iid["contributes_to_designated_primary_aggregate"] is False
    assert dyadic["contributes_to_designated_primary_aggregate"] is True
    assert dyadic["designated_primary_result"] is False


def test_node_variance_output_uses_reported_scale_but_preserves_raw_audit_values():
    detail, concentration = node_variance_rows(
        _dummy_result(),
        seed=11,
        alpha=PRIMARY_ALPHA,
        treatment=PRIMARY_TREATMENT,
        outcome="Y",
        specification="test",
        node_labels={0: "a", 1: "b"},
    )

    assert sum(row["variance_contribution"] for row in detail) == pytest.approx(
        (0.1 * 0.3) ** 2
    )
    assert sum(
        row["variance_contribution_estimation_scale"] for row in detail
    ) == pytest.approx(0.3**2)
    assert concentration["dyadic_variance"] == pytest.approx((0.1 * 0.3) ** 2)
    assert concentration["dyadic_variance_estimation_scale"] == pytest.approx(0.3**2)
    assert sum(row["variance_share"] for row in detail) == pytest.approx(1.0)


def test_all_registered_inferential_treatments_are_raw_similarity_columns():
    for treatment in (
        "BINARY_COSINE",
        "BINARY_JACCARD",
        "PORTFOLIO_WEIGHT_COSINE",
    ):
        assert treatment_reporting_scale(treatment).reporting_increment == 0.1
    for prohibited in (
        "D_BINARY_COSINE_Z",
        "D_BINARY_JACCARD_Z",
        "D_PORTFOLIO_WEIGHT_COSINE_Z",
    ):
        with pytest.raises(ValueError, match="no reporting scale"):
            treatment_reporting_scale(prohibited)


def test_feature_contract_excludes_adjusted_price_level_liquidity_controls():
    assert len(NUMERIC_CONTROLS) == 11
    assert not any("DOLLAR_VOLUME" in name for name in NUMERIC_CONTROLS)
    assert len(NUMERIC_CONTROLS) + len(SECTOR_PAIR_CATEGORIES) - 1 == 76


def test_ridge_sensitivity_uses_all_fixed_seeds_and_reuses_alpha_one_primary():
    configurations = ridge_sensitivity_configurations()
    nonprimary_alphas = set(RIDGE_SENSITIVITY_ALPHAS) - {PRIMARY_ALPHA}
    assert len(configurations) == len(nonprimary_alphas) * len(PRIMARY_SEEDS)
    assert {float(config["alpha"]) for config in configurations} == nonprimary_alphas
    for alpha in nonprimary_alphas:
        assert {
            int(config["seed"])
            for config in configurations
            if float(config["alpha"]) == alpha
        } == set(PRIMARY_SEEDS)
    assert {str(config["treatment"]) for config in configurations} == {
        "BINARY_COSINE"
    }


def _preprocessing_frame(*, perturb_scored_rows: bool) -> pl.DataFrame:
    n_rows = 6
    data: dict[str, object] = {}
    for offset, column in enumerate(NUMERIC_CONTROLS):
        values = np.arange(n_rows, dtype=float) + offset
        if perturb_scored_rows:
            values[3:] += 10_000.0 + offset
        data[column] = values
    sector_pairs = list(SECTOR_PAIR_CATEGORIES[:n_rows])
    if perturb_scored_rows:
        sector_pairs[3:] = list(SECTOR_PAIR_CATEGORIES[-3:])
    data["SECTOR_PAIR"] = sector_pairs
    return pl.DataFrame(data)


def test_scored_rows_cannot_change_training_preprocessing_parameters():
    baseline, names = design_matrix(
        _preprocessing_frame(perturb_scored_rows=False)
    )
    perturbed, perturbed_names = design_matrix(
        _preprocessing_frame(perturb_scored_rows=True)
    )
    assert names == perturbed_names
    assert len(names) == len(NUMERIC_CONTROLS) + len(SECTOR_PAIR_CATEGORIES) - 1
    assert f"SECTOR_PAIR_{SECTOR_PAIR_REFERENCE}" not in names
    np.testing.assert_array_equal(baseline[:3], perturbed[:3])

    outcome = np.array([0.0, 1.0, 2.0])
    baseline_model = ridge_factory(1.0).fit(baseline[:3], outcome)
    perturbed_model = ridge_factory(1.0).fit(perturbed[:3], outcome)
    baseline_scaler = baseline_model.named_steps["preprocess"].named_transformers_[
        "numeric"
    ]
    perturbed_scaler = perturbed_model.named_steps[
        "preprocess"
    ].named_transformers_["numeric"]
    np.testing.assert_array_equal(baseline_scaler.mean_, perturbed_scaler.mean_)
    np.testing.assert_array_equal(baseline_scaler.scale_, perturbed_scaler.scale_)
    np.testing.assert_allclose(
        baseline_model.named_steps["ridge"].coef_,
        perturbed_model.named_steps["ridge"].coef_,
    )


def test_raw_and_globally_standardized_affine_fits_preserve_t_and_score_edges():
    rng = np.random.default_rng(20260809)
    n_nodes = 25
    row, col = np.triu_indices(n_nodes, k=1)
    x = rng.normal(size=(row.size, len(NUMERIC_CONTROLS) + 2))
    x[:, -2:] = rng.integers(0, 2, size=(row.size, 2))
    raw = np.clip(
        0.52 + 0.025 * x[:, 0] - 0.015 * x[:, 1] + rng.normal(0, 0.04, row.size),
        0.05,
        0.95,
    )
    raw_sd = float(np.std(raw, ddof=1))
    standardized = (raw - float(np.mean(raw))) / raw_sd
    outcome = 0.08 * raw + 0.03 * x[:, 2] + rng.normal(0, 0.05, row.size)
    factory = lambda: ridge_factory(1.0)

    raw_fit = fit_plr_dml(
        outcome,
        raw,
        x,
        row,
        col,
        n_nodes,
        directed=False,
        n_folds=5,
        seed=11,
        outcome_factory=factory,
        treatment_factory=factory,
    )
    standardized_fit = fit_plr_dml(
        outcome,
        standardized,
        x,
        row,
        col,
        n_nodes,
        directed=False,
        n_folds=5,
        seed=11,
        outcome_factory=factory,
        treatment_factory=factory,
    )

    assert raw_fit.diagnostics["score_edge_fingerprints"] == (
        standardized_fit.diagnostics["score_edge_fingerprints"]
    )
    assert raw_fit.n_scored == standardized_fit.n_scored
    assert raw_fit.t_dyadic == pytest.approx(
        standardized_fit.t_dyadic, rel=1e-9, abs=1e-11
    )
    assert raw_fit.t_iid == pytest.approx(
        standardized_fit.t_iid, rel=1e-9, abs=1e-11
    )
    assert 0.1 * raw_fit.theta == pytest.approx(
        (0.1 / raw_sd) * standardized_fit.theta,
        rel=1e-9,
        abs=1e-11,
    )
    assert 0.1 * raw_fit.se_dyadic == pytest.approx(
        (0.1 / raw_sd) * standardized_fit.se_dyadic,
        rel=1e-9,
        abs=1e-11,
    )
