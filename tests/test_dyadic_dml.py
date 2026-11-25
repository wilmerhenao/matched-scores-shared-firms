from __future__ import annotations

import math

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression

from src.dyadic_dml import (
    _FoldScores,
    _score_edge_fingerprint,
    _weighted_projection,
    aggregate_repeated_splits,
    dyadic_variance_component,
    fit_edge_split_plr_diagnostic,
    fit_logit_dml,
    fit_matched_score_equal_size_maximal_leak_plr_diagnostic,
    fit_matched_score_leaky_complement_plr_diagnostic,
    fit_plr_dml,
    fold_edge_masks,
    make_node_folds,
)


def complete_undirected(n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(n_nodes, k=1)


def plr_fixture(n_nodes: int = 15, seed: int = 7):
    rng = np.random.default_rng(seed)
    row, col = complete_undirected(n_nodes)
    node_x = rng.normal(size=(n_nodes, 2))
    x = np.column_stack(
        (
            node_x[row, 0] + node_x[col, 0],
            np.abs(node_x[row, 1] - node_x[col, 1]),
        )
    )
    d = 1.5 + 0.7 * x[:, 0] - 0.4 * x[:, 1] + rng.normal(scale=0.7, size=row.size)
    node_error = rng.normal(scale=0.25, size=n_nodes)
    y = 1.25 * d + 0.3 * x[:, 0] + node_error[row] + node_error[col]
    return y, d, x, row, col


def linear_factory():
    return LinearRegression()


def test_one_endpoint_edges_are_excluded_from_both_masks():
    folds = np.array([0, 0, 1, 1])
    row = np.array([0, 0, 2, 0])
    col = np.array([1, 2, 3, 3])
    train, score = fold_edge_masks(row, col, folds, fold=0)
    assert score.tolist() == [True, False, False, False]
    assert train.tolist() == [False, False, True, False]
    assert not np.any(train & score)


def test_edge_identity_fingerprint_is_order_invariant_and_direction_aware():
    row = np.array([4, 2, 3, 0, 8])
    col = np.array([1, 5, 0, 2, 6])
    selected = np.array([True, False, True, True, True])
    order = np.array([3, 0, 4, 1, 2])

    undirected = _score_edge_fingerprint(row, col, selected, directed=False)
    undirected_reordered = _score_edge_fingerprint(
        row[order], col[order], selected[order], directed=False
    )
    undirected_reversed = _score_edge_fingerprint(
        col, row, selected, directed=False
    )
    directed = _score_edge_fingerprint(row, col, selected, directed=True)
    directed_reordered = _score_edge_fingerprint(
        row[order], col[order], selected[order], directed=True
    )
    directed_reversed = _score_edge_fingerprint(
        col, row, selected, directed=True
    )

    assert undirected == undirected_reordered == undirected_reversed
    assert directed == directed_reordered
    assert directed != directed_reversed


def test_plr_solves_fold_normalized_moment_and_counts_edges():
    n_nodes = 15
    y, d, x, row, col = plr_fixture(n_nodes)
    folds = make_node_folds(n_nodes, 4, seed=19)
    result = fit_plr_dml(
        y,
        d,
        x,
        row,
        col,
        n_nodes,
        n_folds=4,
        fold_assignment=folds,
        outcome_factory=linear_factory,
        treatment_factory=linear_factory,
    )
    expected = sum(math.comb(int(np.sum(folds == k)), 2) for k in range(4))
    assert result.n_scored == expected
    assert abs(result.moment) < 1e-12
    assert result.se_dyadic > 0
    assert result.se_iid > 0
    for audit in result.fold_audits:
        assert audit.score_nodes == audit.n_nodes
        assert audit.train_nodes + audit.score_nodes == n_nodes


def test_node_relabeling_equivariance_with_transformed_partition():
    n_nodes = 15
    y, d, x, row, col = plr_fixture(n_nodes)
    folds = make_node_folds(n_nodes, 4, seed=4)
    original = fit_plr_dml(
        y,
        d,
        x,
        row,
        col,
        n_nodes,
        n_folds=4,
        fold_assignment=folds,
        outcome_factory=linear_factory,
        treatment_factory=linear_factory,
    )

    rng = np.random.default_rng(91)
    new_label = rng.permutation(n_nodes)
    transformed_folds = np.empty_like(folds)
    transformed_folds[new_label] = folds
    order = rng.permutation(row.size)
    relabeled = fit_plr_dml(
        y[order],
        d[order],
        x[order],
        new_label[row[order]],
        new_label[col[order]],
        n_nodes,
        n_folds=4,
        fold_assignment=transformed_folds,
        outcome_factory=linear_factory,
        treatment_factory=linear_factory,
    )
    assert np.isclose(relabeled.theta, original.theta, atol=1e-12)
    assert np.isclose(relabeled.se_dyadic, original.se_dyadic, atol=1e-12)
    assert np.isclose(relabeled.se_iid, original.se_iid, atol=1e-12)


def test_undirected_endpoint_reversal_is_invariant():
    n_nodes = 15
    y, d, x, row, col = plr_fixture(n_nodes)
    folds = make_node_folds(n_nodes, 3, seed=8)
    kwargs = dict(
        n_nodes=n_nodes,
        n_folds=3,
        fold_assignment=folds,
        outcome_factory=linear_factory,
        treatment_factory=linear_factory,
    )
    first = fit_plr_dml(y, d, x, row, col, **kwargs)
    reverse = np.arange(row.size) % 2 == 0
    row_swapped = np.where(reverse, col, row)
    col_swapped = np.where(reverse, row, col)
    second = fit_plr_dml(y, d, x, row_swapped, col_swapped, **kwargs)
    assert np.isclose(second.theta, first.theta, atol=1e-12)
    assert np.isclose(second.se_dyadic, first.se_dyadic, atol=1e-12)
    assert second.diagnostics["score_edge_fingerprints"] == first.diagnostics[
        "score_edge_fingerprints"
    ]


def test_unique_undirected_variance_equals_symmetric_directed_duplication():
    n_nodes = 8
    row, col = complete_undirected(n_nodes)
    rng = np.random.default_rng(12)
    psi = rng.normal(size=row.size)
    nodes = np.arange(n_nodes, dtype=np.int64)
    undirected = [_FoldScores(0, nodes, row, col, psi)]
    directed = [
        _FoldScores(
            0,
            nodes,
            np.concatenate((row, col)),
            np.concatenate((col, row)),
            np.concatenate((psi, psi)),
        )
    ]
    gamma_u, _, _ = dyadic_variance_component(undirected, n_nodes, False)
    gamma_d, _, _ = dyadic_variance_component(directed, n_nodes, True)
    assert np.isclose(gamma_u, gamma_d, atol=1e-14)


def test_vectorized_variance_matches_brute_force_node_sums():
    n_nodes = 9
    row, col = complete_undirected(n_nodes)
    rng = np.random.default_rng(22)
    psi = rng.normal(size=row.size)
    item = _FoldScores(0, np.arange(n_nodes), row, col, psi)
    gamma, iid, _ = dyadic_variance_component([item], n_nodes, False)

    sums = np.zeros(n_nodes)
    for edge, value in enumerate(psi):
        sums[row[edge]] += value
        sums[col[edge]] += value
    edge_count = math.comb(n_nodes, 2)
    brute_gamma = (n_nodes - 1) * np.sum(sums**2) / edge_count**2
    brute_iid = np.sum(psi**2) / edge_count**2
    assert np.isclose(gamma, brute_gamma, atol=1e-14)
    assert np.isclose(iid, brute_iid, atol=1e-14)


def test_weighted_projection_keeps_intercept_unpenalized():
    rng = np.random.default_rng(33)
    x = rng.normal(size=(600, 3))
    d = 2.7 + x @ np.array([0.8, -0.5, 0.2]) + rng.normal(scale=0.2, size=600)
    weights = np.exp(0.2 * x[:, 0])
    intercept, gamma, diagnostics = _weighted_projection(
        d, x, weights, alpha=None, seed=1
    )
    residual = d - intercept - x @ gamma
    assert abs(np.average(residual, weights=weights)) < 1e-12
    assert abs(float(diagnostics["weighted_intercept_moment"])) < 1e-12
    assert float(diagnostics["max_weighted_slope_moment"]) < 1e-12


def test_logit_solves_strict_fold_moment_with_intercept_projection():
    n_nodes = 24
    rng = np.random.default_rng(2026)
    row, col = complete_undirected(n_nodes)
    node_x = rng.normal(size=(n_nodes, 2))
    x = np.column_stack(
        (
            node_x[row, 0] + node_x[col, 0],
            np.abs(node_x[row, 1] - node_x[col, 1]),
        )
    )
    d = 0.8 + 0.5 * x[:, 0] - 0.3 * x[:, 1] + rng.normal(size=row.size)
    probability = 1.0 / (1.0 + np.exp(-(-0.4 + 0.6 * d + 0.25 * x[:, 0])))
    y = rng.binomial(1, probability).astype(float)
    folds = make_node_folds(n_nodes, 4, seed=5)

    result = fit_logit_dml(
        y,
        d,
        x,
        row,
        col,
        n_nodes,
        directed=False,
        n_folds=4,
        fold_assignment=folds,
        outcome_c=10.0,
        gamma_alpha=None,
    )

    expected = sum(math.comb(int(np.sum(folds == k)), 2) for k in range(4))
    assert result.n_scored == expected
    assert abs(result.moment) < 1e-9
    assert result.jacobian > 0
    assert result.se_dyadic > 0
    assert result.se_iid > 0
    for diagnostic in result.diagnostics["projection"]:
        assert abs(float(diagnostic["weighted_intercept_moment"])) < 1e-10


def test_edge_split_diagnostic_scores_each_edge_once_and_solves_moment():
    n_nodes = 15
    y, d, x, row, col = plr_fixture(n_nodes)
    result = fit_edge_split_plr_diagnostic(
        y,
        d,
        x,
        row,
        col,
        n_nodes,
        n_folds=5,
        seed=9,
        outcome_factory=linear_factory,
        treatment_factory=linear_factory,
    )
    assert result.n_scored == math.comb(n_nodes, 2)
    assert abs(result.moment) < 1e-12
    assert result.se_iid > 0
    assert result.se_dyadic > 0
    assert result.diagnostics["valid_dyadic_crossfit"] is False


def test_matched_score_leaky_complement_holds_score_sample_and_moment_fixed():
    n_nodes = 15
    n_folds = 4
    y, d, x, row, col = plr_fixture(n_nodes)
    folds = make_node_folds(n_nodes, n_folds, seed=31)

    def zero_factory():
        return DummyRegressor(strategy="constant", constant=0.0)

    kwargs = dict(
        n_nodes=n_nodes,
        n_folds=n_folds,
        fold_assignment=folds,
        outcome_factory=zero_factory,
        treatment_factory=zero_factory,
    )
    strict = fit_plr_dml(y, d, x, row, col, **kwargs)
    leaky = fit_matched_score_leaky_complement_plr_diagnostic(
        y, d, x, row, col, **kwargs
    )

    assert leaky.n_scored == strict.n_scored
    assert leaky.diagnostics["score_edge_fingerprints"] == strict.diagnostics[
        "score_edge_fingerprints"
    ]
    assert np.isclose(leaky.theta, strict.theta, atol=1e-14)
    assert np.isclose(leaky.jacobian, strict.jacobian, atol=1e-14)
    assert np.isclose(leaky.moment, strict.moment, atol=1e-14)
    assert np.isclose(leaky.se_dyadic, strict.se_dyadic, atol=1e-14)
    assert np.isclose(leaky.se_iid, strict.se_iid, atol=1e-14)


def test_matched_score_leaky_complement_exposes_exact_endpoint_leakage():
    n_nodes = 15
    n_folds = 4
    y, d, x, row, col = plr_fixture(n_nodes)
    folds = make_node_folds(n_nodes, n_folds, seed=13)
    result = fit_matched_score_leaky_complement_plr_diagnostic(
        y,
        d,
        x,
        row,
        col,
        n_nodes,
        n_folds=n_folds,
        fold_assignment=folds,
        outcome_factory=linear_factory,
        treatment_factory=linear_factory,
    )

    assert result.diagnostics["valid_dyadic_crossfit"] is False
    assert result.diagnostics["diagnostic_only"] is True
    assert result.diagnostics["matched_score_to_strict"] is True
    assert result.diagnostics["endpoint_leakage_observed"] is True
    for audit, leakage in zip(
        result.fold_audits, result.diagnostics["leakage_by_fold"], strict=True
    ):
        m = audit.n_nodes
        assert audit.n_score_edges == math.comb(m, 2)
        assert audit.n_train_edges == math.comb(n_nodes, 2) - math.comb(m, 2)
        assert leakage["strict_train_edges"] == math.comb(n_nodes - m, 2)
        assert leakage["leaking_train_edges"] == m * (n_nodes - m)
        assert leakage["overlap_nodes"] == m

    contributions = np.asarray(
        result.diagnostics["node_variance_contributions"], dtype=float
    )
    assert contributions.shape == (n_nodes,)
    assert np.all(contributions >= 0.0)
    assert np.isclose(contributions.sum(), result.se_dyadic**2, atol=1e-14)


def test_equal_size_maximal_leak_matches_strict_rows_and_scores_deterministically():
    n_nodes = 30
    n_folds = 5
    y, d, x, row, col = plr_fixture(n_nodes)
    folds = make_node_folds(n_nodes, n_folds, seed=17)
    kwargs = dict(
        n_nodes=n_nodes,
        n_folds=n_folds,
        seed=17,
        fold_assignment=folds,
        outcome_factory=linear_factory,
        treatment_factory=linear_factory,
    )
    strict = fit_plr_dml(y, d, x, row, col, **kwargs)
    first = fit_matched_score_equal_size_maximal_leak_plr_diagnostic(
        y, d, x, row, col, sampling_salt=101, **kwargs
    )
    repeated = fit_matched_score_equal_size_maximal_leak_plr_diagnostic(
        y, d, x, row, col, sampling_salt=101, **kwargs
    )
    alternate = fit_matched_score_equal_size_maximal_leak_plr_diagnostic(
        y, d, x, row, col, sampling_salt=103, **kwargs
    )
    order = np.random.default_rng(109).permutation(row.size)
    reordered = fit_matched_score_equal_size_maximal_leak_plr_diagnostic(
        y[order],
        d[order],
        x[order],
        row[order],
        col[order],
        sampling_salt=101,
        **kwargs,
    )

    assert first.n_scored == strict.n_scored
    assert first.diagnostics["score_edge_fingerprints"] == strict.diagnostics[
        "score_edge_fingerprints"
    ]
    assert first.diagnostics["train_edge_fingerprints"] == repeated.diagnostics[
        "train_edge_fingerprints"
    ]
    assert first.diagnostics["train_edge_fingerprints"] != alternate.diagnostics[
        "train_edge_fingerprints"
    ]
    assert first.diagnostics["score_edge_fingerprints"] == reordered.diagnostics[
        "score_edge_fingerprints"
    ]
    assert first.diagnostics["train_edge_fingerprints"] == reordered.diagnostics[
        "train_edge_fingerprints"
    ]
    assert np.isclose(first.theta, reordered.theta, atol=1e-12)
    assert np.isclose(first.se_dyadic, reordered.se_dyadic, atol=1e-12)
    assert [item.n_train_edges for item in first.fold_audits] == [
        item.n_train_edges for item in strict.fold_audits
    ]
    assert first.diagnostics["matched_training_size_to_strict"] is True
    assert first.diagnostics["endpoint_leakage_observed"] is True
    for audit, leakage in zip(
        first.fold_audits, first.diagnostics["leakage_by_fold"], strict=True
    ):
        m = audit.n_nodes
        strict_count = math.comb(n_nodes - m, 2)
        cross_count = m * (n_nodes - m)
        assert leakage["strict_train_edges"] == strict_count
        assert leakage["selected_train_edges"] == strict_count
        assert leakage["eligible_cross_boundary_edges"] == cross_count
        assert leakage["selected_cross_boundary_edges"] == min(
            strict_count, cross_count
        )
        assert leakage["selected_outside_outside_edges"] == max(
            strict_count - cross_count, 0
        )
        assert leakage["overlap_nodes"] == m

    diagnostics = first.diagnostics
    assert np.isclose(
        first.jacobian, diagnostics["score_treatment_residual_mse"], atol=1e-15
    )
    assert np.isclose(
        diagnostics["score_treatment_residual_rms"],
        np.sqrt(first.jacobian),
        atol=1e-15,
    )
    assert diagnostics["score_outcome_residual_mse"] > 0.0
    assert diagnostics["score_treatment_residual_sd"] > 0.0
    assert abs(first.moment) < 1e-12


def test_equal_size_maximal_leak_samples_cross_edges_when_budget_is_smaller():
    n_nodes = 20
    n_folds = 2
    y, d, x, row, col = plr_fixture(n_nodes)
    folds = make_node_folds(n_nodes, n_folds, seed=23)
    result = fit_matched_score_equal_size_maximal_leak_plr_diagnostic(
        y,
        d,
        x,
        row,
        col,
        n_nodes,
        n_folds=n_folds,
        seed=23,
        sampling_salt=107,
        fold_assignment=folds,
        outcome_factory=linear_factory,
        treatment_factory=linear_factory,
    )
    for leakage in result.diagnostics["leakage_by_fold"]:
        assert leakage["eligible_cross_boundary_edges"] > leakage[
            "strict_train_edges"
        ]
        assert leakage["selected_cross_boundary_edges"] == leakage[
            "strict_train_edges"
        ]
        assert leakage["selected_outside_outside_edges"] == 0


def test_repeated_split_aggregation_includes_between_split_instability():
    estimates = np.array([0.4, 1.0, 1.6])
    standard_errors = np.array([0.2, 0.3, 0.4])
    theta, standard_error = aggregate_repeated_splits(estimates, standard_errors)

    expected_variance = np.median(
        standard_errors**2 + (estimates - np.median(estimates)) ** 2
    )
    assert theta == 1.0
    assert np.isclose(standard_error, np.sqrt(expected_variance), atol=1e-15)
    assert standard_error > np.median(standard_errors)


def test_repeated_split_aggregation_rejects_invalid_inputs():
    with np.testing.assert_raises(ValueError):
        aggregate_repeated_splits([], [])
    with np.testing.assert_raises(ValueError):
        aggregate_repeated_splits([1.0, 2.0], [0.1])
    with np.testing.assert_raises(ValueError):
        aggregate_repeated_splits([1.0], [0.0])
