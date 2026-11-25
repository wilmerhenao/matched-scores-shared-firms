"""Canonical double/debiased ML primitives for dyadic data.

The implementation follows the node-disjoint cross-fitting construction of
Chiang, Ma, Rodrigue, and Sasaki.  It deliberately supports both directed
dyads and unique undirected edges, uses fold-normalized moments, and exposes
the exact scored-edge counts needed to audit the design.

This module replaces the divergent experimental engines.  It does not
subsample nuisance-training edges implicitly.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.base import clone
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
EstimatorFactory = Callable[[], Any]


@dataclass(frozen=True)
class FoldAudit:
    """Counts and diagnostics for one node fold."""

    fold: int
    n_nodes: int
    n_train_edges: int
    n_score_edges: int
    train_nodes: int
    score_nodes: int


@dataclass
class DyadicDMLResult:
    """Stable result schema shared by PLR and logit estimators."""

    estimator: str
    directed: bool
    theta: float
    se_dyadic: float
    se_iid: float
    jacobian: float
    moment: float
    n_nodes: int
    n_edges: int
    n_scored: int
    n_folds: int
    fold_audits: list[FoldAudit]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def t_dyadic(self) -> float:
        """Return the coefficient divided by its node-overlap standard error."""

        return self.theta / self.se_dyadic

    @property
    def t_iid(self) -> float:
        """Return the coefficient divided by the independent-edge diagnostic SE."""

        return self.theta / self.se_iid

    def as_dict(self) -> dict[str, Any]:
        """Serialize the estimate, fold audits, diagnostics, and derived statistics."""

        result = asdict(self)
        result["t_dyadic"] = self.t_dyadic
        result["t_iid"] = self.t_iid
        return result


@dataclass(frozen=True)
class _FoldScores:
    fold: int
    nodes: IntArray
    row: IntArray
    col: IntArray
    psi: FloatArray

    @property
    def edge_count(self) -> int:
        return int(self.psi.size)


def _fold_normalized_score_residual_diagnostics(
    fold_payload: Sequence[
        tuple[int, IntArray, IntArray, IntArray, FloatArray, FloatArray]
    ],
) -> dict[str, float]:
    """Summarize observed prediction residuals with equal score-fold weights.

    These quantities are score-sample predictive residual diagnostics for
    ``Y - g_hat(X)`` and ``D - m_hat(X)``.  They are not errors relative to the
    unknown true nuisance functions, which are unavailable in an application.
    """

    if not fold_payload:
        raise ValueError("at least one scored fold is required")
    fold_count = len(fold_payload)
    outcome_residual_mse = 0.0
    treatment_residual_mse = 0.0
    outcome_mean = 0.0
    treatment_mean = 0.0
    for _, _, _, _, y_tilde, d_tilde in fold_payload:
        outcome_residual_mse += float(y_tilde @ y_tilde) / y_tilde.size
        treatment_residual_mse += float(d_tilde @ d_tilde) / d_tilde.size
        outcome_mean += float(np.mean(y_tilde))
        treatment_mean += float(np.mean(d_tilde))
    outcome_residual_mse /= fold_count
    treatment_residual_mse /= fold_count
    outcome_mean /= fold_count
    treatment_mean /= fold_count
    outcome_variance = max(outcome_residual_mse - outcome_mean**2, 0.0)
    treatment_variance = max(treatment_residual_mse - treatment_mean**2, 0.0)
    return {
        "score_outcome_residual_mse": outcome_residual_mse,
        "score_treatment_residual_mse": treatment_residual_mse,
        "score_outcome_residual_mean": outcome_mean,
        "score_treatment_residual_mean": treatment_mean,
        "score_outcome_residual_sd": float(np.sqrt(outcome_variance)),
        "score_treatment_residual_sd": float(np.sqrt(treatment_variance)),
        "score_treatment_residual_rms": float(np.sqrt(treatment_residual_mse)),
    }


def make_node_folds(n_nodes: int, n_folds: int, seed: int) -> IntArray:
    """Return a balanced random node-to-fold assignment."""

    if n_nodes < 2:
        raise ValueError("at least two nodes are required")
    if not 2 <= n_folds <= n_nodes:
        raise ValueError("n_folds must lie in [2, n_nodes]")
    rng = np.random.default_rng(seed)
    assignment = np.empty(n_nodes, dtype=np.int64)
    for fold, nodes in enumerate(np.array_split(rng.permutation(n_nodes), n_folds)):
        assignment[nodes] = fold
    return assignment


def _validate_inputs(
    y: ArrayLike,
    d: ArrayLike,
    x: ArrayLike,
    row: ArrayLike,
    col: ArrayLike,
    n_nodes: int,
    fold_assignment: ArrayLike,
) -> tuple[FloatArray, FloatArray, FloatArray, IntArray, IntArray, IntArray]:
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    d_arr = np.asarray(d, dtype=np.float64).reshape(-1)
    x_arr = np.asarray(x, dtype=np.float64)
    row_arr = np.asarray(row, dtype=np.int64).reshape(-1)
    col_arr = np.asarray(col, dtype=np.int64).reshape(-1)
    folds = np.asarray(fold_assignment, dtype=np.int64).reshape(-1)

    if x_arr.ndim != 2:
        raise ValueError("x must be a two-dimensional design matrix")
    lengths = {y_arr.size, d_arr.size, x_arr.shape[0], row_arr.size, col_arr.size}
    if len(lengths) != 1:
        raise ValueError("y, d, x, row, and col must contain the same edge count")
    if folds.size != n_nodes:
        raise ValueError("fold_assignment must contain one entry per node")
    if y_arr.size == 0:
        raise ValueError("at least one edge is required")
    if np.any(row_arr == col_arr):
        raise ValueError("self edges are not allowed")
    if np.any(row_arr < 0) or np.any(col_arr < 0):
        raise ValueError("node indices must be nonnegative")
    if np.any(row_arr >= n_nodes) or np.any(col_arr >= n_nodes):
        raise ValueError("node index exceeds n_nodes")
    unique_folds = np.unique(folds)
    if not np.array_equal(unique_folds, np.arange(unique_folds.size)):
        raise ValueError("fold labels must be contiguous and start at zero")
    if not np.all(np.isfinite(y_arr)) or not np.all(np.isfinite(d_arr)):
        raise ValueError("y and d must be finite")
    if not np.all(np.isfinite(x_arr)):
        raise ValueError("x must be finite")
    return y_arr, d_arr, x_arr, row_arr, col_arr, folds


def fold_edge_masks(
    row: ArrayLike,
    col: ArrayLike,
    fold_assignment: ArrayLike,
    fold: int,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Return strict complement-training and within-fold scoring masks.

    An edge with exactly one endpoint in ``fold`` is deliberately in neither
    mask.  This is the central invariant that the legacy empirical engines
    violated.
    """

    row_arr = np.asarray(row, dtype=np.int64)
    col_arr = np.asarray(col, dtype=np.int64)
    folds = np.asarray(fold_assignment, dtype=np.int64)
    row_fold = folds[row_arr]
    col_fold = folds[col_arr]
    train = (row_fold != fold) & (col_fold != fold)
    score = (row_fold == fold) & (col_fold == fold)
    return train, score


def _fold_audit(
    fold: int,
    folds: IntArray,
    row: IntArray,
    col: IntArray,
    train: NDArray[np.bool_],
    score: NDArray[np.bool_],
) -> FoldAudit:
    nodes = np.flatnonzero(folds == fold)
    score_nodes = np.unique(np.concatenate((row[score], col[score])))
    train_nodes = np.unique(np.concatenate((row[train], col[train])))
    if np.intersect1d(score_nodes, train_nodes).size:
        raise AssertionError("nuisance training and scoring share at least one node")
    return FoldAudit(
        fold=fold,
        n_nodes=int(nodes.size),
        n_train_edges=int(train.sum()),
        n_score_edges=int(score.sum()),
        train_nodes=int(train_nodes.size),
        score_nodes=int(score_nodes.size),
    )


def _dyadic_variance_decomposition(
    fold_scores: Sequence[_FoldScores],
    n_nodes: int,
    directed: bool,
) -> tuple[float, float, FloatArray, FloatArray]:
    """Compute moment variances and each node's contribution to ``Gamma``."""

    if not fold_scores:
        raise ValueError("at least one scored fold is required")
    k_total = len(fold_scores)
    gamma = 0.0
    iid_variance = 0.0
    global_node_sums = np.zeros(n_nodes, dtype=np.float64)
    node_gamma_contributions = np.zeros(n_nodes, dtype=np.float64)

    for item in fold_scores:
        m = int(item.nodes.size)
        edge_count = item.edge_count
        expected = m * (m - 1) if directed else m * (m - 1) // 2
        if edge_count != expected:
            kind = "directed" if directed else "undirected"
            raise ValueError(
                f"fold {item.fold} has {edge_count} scored edges; "
                f"a complete {kind} fold with {m} nodes requires {expected}"
            )

        node_sum = np.zeros(n_nodes, dtype=np.float64)
        np.add.at(node_sum, item.row, item.psi)
        np.add.at(node_sum, item.col, item.psi)
        global_node_sums += node_sum

        # For directed data node_sum is A_i+B_i.  For unique undirected data
        # it is the incident-edge sum S_i.  The same expression applies when
        # edge_count is the appropriate complete-fold count.
        fold_node_gamma = (m - 1) * node_sum**2 / float(edge_count**2)
        gamma += float(fold_node_gamma.sum())
        node_gamma_contributions += fold_node_gamma / k_total
        iid_variance += float(item.psi @ item.psi) / float(edge_count**2)

    gamma /= k_total
    iid_variance /= k_total**2
    return gamma, iid_variance, global_node_sums, node_gamma_contributions


def dyadic_variance_component(
    fold_scores: Sequence[_FoldScores],
    n_nodes: int,
    directed: bool,
) -> tuple[float, float, FloatArray]:
    """Compute dyadic and IID moment variances from fold scores.

    Returns ``(Gamma, iid_moment_variance, node_score_sums)``.  ``Gamma`` is
    the Chiang fold-averaged dyadic component used as ``Gamma / n_nodes`` in
    the estimator variance.  The IID value is already the variance of the
    fold-averaged moment.
    """

    gamma, iid_variance, node_sums, _ = _dyadic_variance_decomposition(
        fold_scores, n_nodes, directed
    )
    return gamma, iid_variance, node_sums


def _score_edge_fingerprint(
    row: IntArray,
    col: IntArray,
    selected: NDArray[np.bool_],
    directed: bool,
) -> str:
    """Hash selected canonical endpoint identities, independent of edge-row order.

    Undirected endpoints are represented as ``(min(i, j), max(i, j))``;
    directed endpoints retain their ``(i, j)`` orientation.  Sorting the pair
    records before hashing makes the audit fingerprint invariant to how the
    edge observations happen to be ordered in the input arrays.
    """

    row_arr = np.asarray(row, dtype=np.int64).reshape(-1)
    col_arr = np.asarray(col, dtype=np.int64).reshape(-1)
    selected_arr = np.asarray(selected, dtype=np.bool_).reshape(-1)
    if row_arr.size != col_arr.size or row_arr.size != selected_arr.size:
        raise ValueError("row, col, and selected must have equal length")

    selected_row = row_arr[selected_arr]
    selected_col = col_arr[selected_arr]
    if directed:
        first = selected_row
        second = selected_col
    else:
        first = np.minimum(selected_row, selected_col)
        second = np.maximum(selected_row, selected_col)

    order = np.lexsort((second, first))
    pairs = np.column_stack((first[order], second[order])).astype(
        "<i8", copy=False
    )
    canonical_pairs = np.ascontiguousarray(pairs)
    digest = hashlib.sha256()
    digest.update(b"dyadic-edge-identities-v1\0")
    digest.update(b"directed\0" if directed else b"undirected\0")
    digest.update(np.asarray([canonical_pairs.shape[0]], dtype="<u8").tobytes())
    digest.update(canonical_pairs.tobytes())
    return digest.hexdigest()


def aggregate_repeated_splits(
    estimates: ArrayLike, standard_errors: ArrayLike
) -> tuple[float, float]:
    """Median-aggregate repeated sample splits and their uncertainty.

    This is the median repeated-splitting rule of Chernozhukov et al. (2018):
    if ``theta_s`` and ``se_s`` are the estimate and estimator-scale standard
    error from split ``s``, return

    ``median(theta_s)`` and
    ``sqrt(median(se_s**2 + (theta_s - median(theta_s))**2))``.

    The second term propagates split-to-split instability; taking the median
    of the run-specific standard errors alone would omit it.
    """

    theta = np.asarray(estimates, dtype=np.float64).reshape(-1)
    se = np.asarray(standard_errors, dtype=np.float64).reshape(-1)
    if theta.size == 0:
        raise ValueError("at least one repeated-split estimate is required")
    if theta.size != se.size:
        raise ValueError("estimates and standard_errors must have equal length")
    if not np.all(np.isfinite(theta)) or not np.all(np.isfinite(se)):
        raise ValueError("repeated-split estimates and standard errors must be finite")
    if np.any(se <= 0.0):
        raise ValueError("repeated-split standard errors must be positive")
    theta_median = float(np.median(theta))
    variance = float(np.median(se**2 + (theta - theta_median) ** 2))
    if variance <= 0.0 or not np.isfinite(variance):
        raise ValueError("repeated-split aggregate variance is not positive and finite")
    return theta_median, float(np.sqrt(variance))


def fit_plr_dml(
    y: ArrayLike,
    d: ArrayLike,
    x: ArrayLike,
    row: ArrayLike,
    col: ArrayLike,
    n_nodes: int,
    *,
    directed: bool = False,
    n_folds: int = 5,
    seed: int = 42,
    fold_assignment: ArrayLike | None = None,
    outcome_factory: EstimatorFactory | None = None,
    treatment_factory: EstimatorFactory | None = None,
) -> DyadicDMLResult:
    """Fit partially linear dyadic DML with strict node cross-fitting."""

    folds = (
        make_node_folds(n_nodes, n_folds, seed)
        if fold_assignment is None
        else np.asarray(fold_assignment, dtype=np.int64)
    )
    y_arr, d_arr, x_arr, row_arr, col_arr, folds = _validate_inputs(
        y, d, x, row, col, n_nodes, folds
    )
    n_folds_actual = int(np.unique(folds).size)
    if n_folds_actual != n_folds:
        raise ValueError("n_folds does not match fold_assignment")

    outcome_factory = outcome_factory or (lambda: LinearRegression())
    treatment_factory = treatment_factory or (lambda: LinearRegression())

    fold_payload: list[tuple[int, IntArray, IntArray, IntArray, FloatArray, FloatArray]] = []
    audits: list[FoldAudit] = []
    score_edge_fingerprints: list[str] = []
    numerator = 0.0
    denominator = 0.0

    for fold in range(n_folds_actual):
        train, score = fold_edge_masks(row_arr, col_arr, folds, fold)
        audit = _fold_audit(fold, folds, row_arr, col_arr, train, score)
        audits.append(audit)
        score_edge_fingerprints.append(
            _score_edge_fingerprint(row_arr, col_arr, score, directed)
        )
        if audit.n_score_edges == 0 or audit.n_train_edges == 0:
            raise ValueError(f"fold {fold} has an empty train or score edge set")

        g_model = clone(outcome_factory())
        m_model = clone(treatment_factory())
        g_model.fit(x_arr[train], y_arr[train])
        m_model.fit(x_arr[train], d_arr[train])

        y_tilde = y_arr[score] - np.asarray(g_model.predict(x_arr[score])).reshape(-1)
        d_tilde = d_arr[score] - np.asarray(m_model.predict(x_arr[score])).reshape(-1)
        edge_count = audit.n_score_edges
        numerator += float(y_tilde @ d_tilde) / edge_count
        denominator += float(d_tilde @ d_tilde) / edge_count
        nodes = np.flatnonzero(folds == fold)
        fold_payload.append(
            (fold, nodes, row_arr[score], col_arr[score], y_tilde, d_tilde)
        )

    numerator /= n_folds_actual
    denominator /= n_folds_actual
    if denominator <= np.finfo(float).eps:
        raise ValueError("residualized treatment has no identifying variation")
    theta = numerator / denominator
    score_residual_diagnostics = _fold_normalized_score_residual_diagnostics(
        fold_payload
    )

    score_items: list[_FoldScores] = []
    moment = 0.0
    for fold, nodes, row_k, col_k, y_tilde, d_tilde in fold_payload:
        psi = (y_tilde - theta * d_tilde) * d_tilde
        moment += float(psi.mean())
        score_items.append(_FoldScores(fold, nodes, row_k, col_k, psi))
    moment /= n_folds_actual

    gamma, iid_moment_variance, node_sums, node_gamma = _dyadic_variance_decomposition(
        score_items, n_nodes, directed
    )
    se_dyadic = np.sqrt(gamma / n_nodes) / denominator
    se_iid = np.sqrt(iid_moment_variance) / denominator
    n_scored = int(sum(item.edge_count for item in score_items))

    return DyadicDMLResult(
        estimator="plr",
        directed=directed,
        theta=float(theta),
        se_dyadic=float(se_dyadic),
        se_iid=float(se_iid),
        jacobian=float(denominator),
        moment=float(moment),
        n_nodes=n_nodes,
        n_edges=int(y_arr.size),
        n_scored=n_scored,
        n_folds=n_folds_actual,
        fold_audits=audits,
        diagnostics={
            "valid_dyadic_crossfit": True,
            "training_scheme": "strict_node_complement",
            "score_definition": "within_node_fold",
            "moment_weighting": "equal_fold_average",
            **score_residual_diagnostics,
            "gamma": float(gamma),
            "iid_moment_variance": float(iid_moment_variance),
            "max_abs_node_score": float(np.max(np.abs(node_sums))),
            "node_variance_contributions": (
                node_gamma / n_nodes / denominator**2
            ).tolist(),
            "fold_assignment": folds.tolist(),
            "score_edge_fingerprints": score_edge_fingerprints,
        },
    )


def fit_matched_score_leaky_complement_plr_diagnostic(
    y: ArrayLike,
    d: ArrayLike,
    x: ArrayLike,
    row: ArrayLike,
    col: ArrayLike,
    n_nodes: int,
    *,
    directed: bool = False,
    n_folds: int = 5,
    seed: int = 42,
    fold_assignment: ArrayLike | None = None,
    outcome_factory: EstimatorFactory | None = None,
    treatment_factory: EstimatorFactory | None = None,
) -> DyadicDMLResult:
    """Fit the matched-score, leaky-complement PLR diagnostic.

    Scoring observations and fold normalization are exactly those of
    :func:`fit_plr_dml`: fold ``k`` scores only edges whose two endpoints are
    in node fold ``k``.  The deliberate violation is confined to nuisance
    training, which uses every edge outside that score set.  Cross-boundary
    training edges therefore contain score-fold endpoints.  This estimator
    isolates endpoint leakage while holding the scored sample fixed; it is a
    diagnostic comparator and is not valid dyadic cross-fitting.
    """

    folds = (
        make_node_folds(n_nodes, n_folds, seed)
        if fold_assignment is None
        else np.asarray(fold_assignment, dtype=np.int64)
    )
    y_arr, d_arr, x_arr, row_arr, col_arr, folds = _validate_inputs(
        y, d, x, row, col, n_nodes, folds
    )
    n_folds_actual = int(np.unique(folds).size)
    if n_folds_actual != n_folds:
        raise ValueError("n_folds does not match fold_assignment")

    outcome_factory = outcome_factory or (lambda: LinearRegression())
    treatment_factory = treatment_factory or (lambda: LinearRegression())

    fold_payload: list[tuple[int, IntArray, IntArray, IntArray, FloatArray, FloatArray]] = []
    audits: list[FoldAudit] = []
    leakage_audits: list[dict[str, int]] = []
    score_edge_fingerprints: list[str] = []
    numerator = 0.0
    denominator = 0.0

    for fold in range(n_folds_actual):
        strict_train, score = fold_edge_masks(row_arr, col_arr, folds, fold)
        train = ~score
        nodes = np.flatnonzero(folds == fold)
        score_nodes = np.unique(np.concatenate((row_arr[score], col_arr[score])))
        train_nodes = np.unique(np.concatenate((row_arr[train], col_arr[train])))
        overlap_nodes = np.intersect1d(score_nodes, train_nodes)
        incident_to_score_fold = (folds[row_arr] == fold) | (folds[col_arr] == fold)
        leaking_train = train & incident_to_score_fold

        audit = FoldAudit(
            fold=fold,
            n_nodes=int(nodes.size),
            n_train_edges=int(train.sum()),
            n_score_edges=int(score.sum()),
            train_nodes=int(train_nodes.size),
            score_nodes=int(score_nodes.size),
        )
        audits.append(audit)
        score_edge_fingerprints.append(
            _score_edge_fingerprint(row_arr, col_arr, score, directed)
        )
        leakage_audits.append(
            {
                "fold": fold,
                "strict_train_edges": int(strict_train.sum()),
                "complement_train_edges": int(train.sum()),
                "leaking_train_edges": int(leaking_train.sum()),
                "overlap_nodes": int(overlap_nodes.size),
            }
        )
        if audit.n_score_edges == 0 or audit.n_train_edges == 0:
            raise ValueError(f"fold {fold} has an empty train or score edge set")

        g_model = clone(outcome_factory())
        m_model = clone(treatment_factory())
        g_model.fit(x_arr[train], y_arr[train])
        m_model.fit(x_arr[train], d_arr[train])

        y_tilde = y_arr[score] - np.asarray(g_model.predict(x_arr[score])).reshape(-1)
        d_tilde = d_arr[score] - np.asarray(m_model.predict(x_arr[score])).reshape(-1)
        edge_count = audit.n_score_edges
        numerator += float(y_tilde @ d_tilde) / edge_count
        denominator += float(d_tilde @ d_tilde) / edge_count
        fold_payload.append(
            (fold, nodes, row_arr[score], col_arr[score], y_tilde, d_tilde)
        )

    numerator /= n_folds_actual
    denominator /= n_folds_actual
    if denominator <= np.finfo(float).eps:
        raise ValueError("residualized treatment has no identifying variation")
    theta = numerator / denominator
    score_residual_diagnostics = _fold_normalized_score_residual_diagnostics(
        fold_payload
    )

    score_items: list[_FoldScores] = []
    moment = 0.0
    for fold, nodes, row_k, col_k, y_tilde, d_tilde in fold_payload:
        psi = (y_tilde - theta * d_tilde) * d_tilde
        moment += float(psi.mean())
        score_items.append(_FoldScores(fold, nodes, row_k, col_k, psi))
    moment /= n_folds_actual

    gamma, iid_moment_variance, node_sums, node_gamma = _dyadic_variance_decomposition(
        score_items, n_nodes, directed
    )
    se_dyadic = np.sqrt(gamma / n_nodes) / denominator
    se_iid = np.sqrt(iid_moment_variance) / denominator
    n_scored = int(sum(item.edge_count for item in score_items))
    leakage_observed = any(item["leaking_train_edges"] > 0 for item in leakage_audits)

    return DyadicDMLResult(
        estimator="plr_matched_score_leaky_complement_diagnostic",
        directed=directed,
        theta=float(theta),
        se_dyadic=float(se_dyadic),
        se_iid=float(se_iid),
        jacobian=float(denominator),
        moment=float(moment),
        n_nodes=n_nodes,
        n_edges=int(y_arr.size),
        n_scored=n_scored,
        n_folds=n_folds_actual,
        fold_audits=audits,
        diagnostics={
            "valid_dyadic_crossfit": False,
            "diagnostic_only": True,
            "purpose": "matched-score endpoint-leakage diagnostic only",
            "training_scheme": "all_non_score_edges",
            "score_definition": "within_node_fold",
            "moment_weighting": "equal_fold_average",
            "matched_score_to_strict": True,
            "endpoint_leakage_observed": leakage_observed,
            "leakage_by_fold": leakage_audits,
            **score_residual_diagnostics,
            "gamma": float(gamma),
            "iid_moment_variance": float(iid_moment_variance),
            "max_abs_node_score": float(np.max(np.abs(node_sums))),
            "node_variance_contributions": (
                node_gamma / n_nodes / denominator**2
            ).tolist(),
            "fold_assignment": folds.tolist(),
            "score_edge_fingerprints": score_edge_fingerprints,
        },
    )


def fit_matched_score_equal_size_maximal_leak_plr_diagnostic(
    y: ArrayLike,
    d: ArrayLike,
    x: ArrayLike,
    row: ArrayLike,
    col: ArrayLike,
    n_nodes: int,
    *,
    directed: bool = False,
    n_folds: int = 5,
    seed: int = 42,
    sampling_salt: int = 20260809,
    fold_assignment: ArrayLike | None = None,
    outcome_factory: EstimatorFactory | None = None,
    treatment_factory: EstimatorFactory | None = None,
) -> DyadicDMLResult:
    """Fit an equal-training-size, maximally endpoint-leaky PLR diagnostic.

    The score folds are identical to :func:`fit_plr_dml`.  In each fold, the
    diagnostic uses exactly as many nuisance-training edges as the strict fit,
    but admits as many cross-boundary edges as that fixed row budget permits.
    When every cross-boundary edge fits in the budget, all are retained and a
    deterministic sample of outside--outside edges fills the remaining slots.
    Otherwise, a deterministic sample of cross-boundary edges fills the entire
    budget.  Sampling depends only on ``seed``, fold number, and
    ``sampling_salt``; it never depends on outcomes, treatments, or controls.

    This diagnostic removes total nuisance-training row count as an explanation
    for a strict-versus-leaky difference.  It deliberately violates node-
    disjoint cross-fitting and is never valid primary inference.
    """

    if seed < 0 or sampling_salt < 0:
        raise ValueError("seed and sampling_salt must be nonnegative")
    folds = (
        make_node_folds(n_nodes, n_folds, seed)
        if fold_assignment is None
        else np.asarray(fold_assignment, dtype=np.int64)
    )
    y_arr, d_arr, x_arr, row_arr, col_arr, folds = _validate_inputs(
        y, d, x, row, col, n_nodes, folds
    )
    n_folds_actual = int(np.unique(folds).size)
    if n_folds_actual != n_folds:
        raise ValueError("n_folds does not match fold_assignment")

    outcome_factory = outcome_factory or (lambda: LinearRegression())
    treatment_factory = treatment_factory or (lambda: LinearRegression())

    fold_payload: list[
        tuple[int, IntArray, IntArray, IntArray, FloatArray, FloatArray]
    ] = []
    audits: list[FoldAudit] = []
    leakage_audits: list[dict[str, int]] = []
    score_edge_fingerprints: list[str] = []
    train_edge_fingerprints: list[str] = []
    numerator = 0.0
    denominator = 0.0

    for fold in range(n_folds_actual):
        strict_train, score = fold_edge_masks(row_arr, col_arr, folds, fold)
        row_in_fold = folds[row_arr] == fold
        col_in_fold = folds[col_arr] == fold
        cross_boundary = row_in_fold ^ col_in_fold
        strict_indices = np.flatnonzero(strict_train)
        cross_indices = np.flatnonzero(cross_boundary)
        if directed:
            strict_order = np.lexsort(
                (col_arr[strict_indices], row_arr[strict_indices])
            )
            cross_order = np.lexsort(
                (col_arr[cross_indices], row_arr[cross_indices])
            )
        else:
            strict_low = np.minimum(row_arr[strict_indices], col_arr[strict_indices])
            strict_high = np.maximum(row_arr[strict_indices], col_arr[strict_indices])
            cross_low = np.minimum(row_arr[cross_indices], col_arr[cross_indices])
            cross_high = np.maximum(row_arr[cross_indices], col_arr[cross_indices])
            strict_order = np.lexsort((strict_high, strict_low))
            cross_order = np.lexsort((cross_high, cross_low))
        strict_indices = strict_indices[strict_order]
        cross_indices = cross_indices[cross_order]
        strict_count = int(strict_indices.size)
        cross_count = int(cross_indices.size)
        if strict_count == 0 or int(score.sum()) == 0:
            raise ValueError(f"fold {fold} has an empty train or score edge set")

        rng = np.random.default_rng(
            np.random.SeedSequence([int(seed), int(fold), int(sampling_salt)])
        )
        if cross_count <= strict_count:
            selected_cross = cross_indices
            outside_needed = strict_count - cross_count
            selected_outside = rng.choice(
                strict_indices, size=outside_needed, replace=False
            )
        else:
            selected_cross = rng.choice(
                cross_indices, size=strict_count, replace=False
            )
            selected_outside = np.empty(0, dtype=np.int64)

        train = np.zeros(row_arr.size, dtype=np.bool_)
        train[selected_cross] = True
        train[selected_outside] = True
        if int(train.sum()) != strict_count:
            raise AssertionError(
                "equal-size leakage diagnostic does not match strict train count"
            )
        if np.any(train & score):
            raise AssertionError("nuisance training and scoring edges overlap")

        nodes = np.flatnonzero(folds == fold)
        score_nodes = np.unique(np.concatenate((row_arr[score], col_arr[score])))
        train_nodes = np.unique(np.concatenate((row_arr[train], col_arr[train])))
        overlap_nodes = np.intersect1d(score_nodes, train_nodes)
        audit = FoldAudit(
            fold=fold,
            n_nodes=int(nodes.size),
            n_train_edges=int(train.sum()),
            n_score_edges=int(score.sum()),
            train_nodes=int(train_nodes.size),
            score_nodes=int(score_nodes.size),
        )
        audits.append(audit)
        score_edge_fingerprints.append(
            _score_edge_fingerprint(row_arr, col_arr, score, directed)
        )
        train_edge_fingerprints.append(
            _score_edge_fingerprint(row_arr, col_arr, train, directed)
        )
        leakage_audits.append(
            {
                "fold": fold,
                "strict_train_edges": strict_count,
                "selected_train_edges": int(train.sum()),
                "eligible_cross_boundary_edges": cross_count,
                "selected_cross_boundary_edges": int(selected_cross.size),
                "selected_outside_outside_edges": int(selected_outside.size),
                "overlap_nodes": int(overlap_nodes.size),
                "sampling_salt": int(sampling_salt),
            }
        )

        g_model = clone(outcome_factory())
        m_model = clone(treatment_factory())
        g_model.fit(x_arr[train], y_arr[train])
        m_model.fit(x_arr[train], d_arr[train])

        y_tilde = y_arr[score] - np.asarray(g_model.predict(x_arr[score])).reshape(-1)
        d_tilde = d_arr[score] - np.asarray(m_model.predict(x_arr[score])).reshape(-1)
        edge_count = audit.n_score_edges
        numerator += float(y_tilde @ d_tilde) / edge_count
        denominator += float(d_tilde @ d_tilde) / edge_count
        fold_payload.append(
            (fold, nodes, row_arr[score], col_arr[score], y_tilde, d_tilde)
        )

    numerator /= n_folds_actual
    denominator /= n_folds_actual
    if denominator <= np.finfo(float).eps:
        raise ValueError("residualized treatment has no identifying variation")
    theta = numerator / denominator
    score_residual_diagnostics = _fold_normalized_score_residual_diagnostics(
        fold_payload
    )

    score_items: list[_FoldScores] = []
    moment = 0.0
    for fold, nodes, row_k, col_k, y_tilde, d_tilde in fold_payload:
        psi = (y_tilde - theta * d_tilde) * d_tilde
        moment += float(psi.mean())
        score_items.append(_FoldScores(fold, nodes, row_k, col_k, psi))
    moment /= n_folds_actual

    gamma, iid_moment_variance, node_sums, node_gamma = (
        _dyadic_variance_decomposition(score_items, n_nodes, directed)
    )
    se_dyadic = np.sqrt(gamma / n_nodes) / denominator
    se_iid = np.sqrt(iid_moment_variance) / denominator
    n_scored = int(sum(item.edge_count for item in score_items))
    leakage_observed = any(
        item["selected_cross_boundary_edges"] > 0 for item in leakage_audits
    )

    return DyadicDMLResult(
        estimator="plr_matched_score_equal_size_maximal_leak_diagnostic",
        directed=directed,
        theta=float(theta),
        se_dyadic=float(se_dyadic),
        se_iid=float(se_iid),
        jacobian=float(denominator),
        moment=float(moment),
        n_nodes=n_nodes,
        n_edges=int(y_arr.size),
        n_scored=n_scored,
        n_folds=n_folds_actual,
        fold_audits=audits,
        diagnostics={
            "valid_dyadic_crossfit": False,
            "diagnostic_only": True,
            "purpose": (
                "matched-score equal-training-size maximal endpoint-leakage "
                "diagnostic only"
            ),
            "training_scheme": "equal_size_maximal_cross_boundary",
            "score_definition": "within_node_fold",
            "moment_weighting": "equal_fold_average",
            "matched_score_to_strict": True,
            "matched_training_size_to_strict": True,
            "endpoint_leakage_observed": leakage_observed,
            "sampling_seed": int(seed),
            "sampling_salt": int(sampling_salt),
            "leakage_by_fold": leakage_audits,
            **score_residual_diagnostics,
            "gamma": float(gamma),
            "iid_moment_variance": float(iid_moment_variance),
            "max_abs_node_score": float(np.max(np.abs(node_sums))),
            "node_variance_contributions": (
                node_gamma / n_nodes / denominator**2
            ).tolist(),
            "fold_assignment": folds.tolist(),
            "score_edge_fingerprints": score_edge_fingerprints,
            "train_edge_fingerprints": train_edge_fingerprints,
        },
    )


def fit_edge_split_plr_diagnostic(
    y: ArrayLike,
    d: ArrayLike,
    x: ArrayLike,
    row: ArrayLike,
    col: ArrayLike,
    n_nodes: int,
    *,
    n_folds: int = 5,
    seed: int = 42,
    fold_assignment: ArrayLike | None = None,
    outcome_factory: EstimatorFactory | None = None,
    treatment_factory: EstimatorFactory | None = None,
) -> DyadicDMLResult:
    """Conventional random-edge cross-fit diagnostic for undirected data.

    This intentionally permits endpoint leakage and is not valid dyadic DML.
    It exists only to populate the split-by-variance diagnostic table.  Each
    edge is scored once and receives equal weight in the estimating moment.
    """

    placeholder_node_folds = np.zeros(n_nodes, dtype=np.int64)
    y_arr, d_arr, x_arr, row_arr, col_arr, _ = _validate_inputs(
        y, d, x, row, col, n_nodes, placeholder_node_folds
    )
    expected_edges = n_nodes * (n_nodes - 1) // 2
    if y_arr.size != expected_edges:
        raise ValueError("edge diagnostic requires one complete undirected graph")
    if fold_assignment is None:
        rng = np.random.default_rng(seed)
        edge_folds = np.empty(y_arr.size, dtype=np.int64)
        for fold, indices in enumerate(np.array_split(rng.permutation(y_arr.size), n_folds)):
            edge_folds[indices] = fold
    else:
        edge_folds = np.asarray(fold_assignment, dtype=np.int64).reshape(-1)
    if edge_folds.size != y_arr.size:
        raise ValueError("edge fold assignment must contain one label per edge")
    if not np.array_equal(np.unique(edge_folds), np.arange(n_folds)):
        raise ValueError("edge fold labels must be contiguous and match n_folds")

    outcome_factory = outcome_factory or (lambda: LinearRegression())
    treatment_factory = treatment_factory or (lambda: LinearRegression())
    y_tilde = np.empty(y_arr.size, dtype=np.float64)
    d_tilde = np.empty(d_arr.size, dtype=np.float64)
    fold_counts: list[int] = []
    for fold in range(n_folds):
        score = edge_folds == fold
        train = ~score
        g_model = clone(outcome_factory())
        m_model = clone(treatment_factory())
        g_model.fit(x_arr[train], y_arr[train])
        m_model.fit(x_arr[train], d_arr[train])
        y_tilde[score] = y_arr[score] - np.asarray(
            g_model.predict(x_arr[score])
        ).reshape(-1)
        d_tilde[score] = d_arr[score] - np.asarray(
            m_model.predict(x_arr[score])
        ).reshape(-1)
        fold_counts.append(int(score.sum()))

    denominator = float(d_tilde @ d_tilde / y_arr.size)
    if denominator <= np.finfo(float).eps:
        raise ValueError("residualized treatment has no identifying variation")
    theta = float((y_tilde @ d_tilde / y_arr.size) / denominator)
    psi = (y_tilde - theta * d_tilde) * d_tilde
    node_sums = np.zeros(n_nodes, dtype=np.float64)
    np.add.at(node_sums, row_arr, psi)
    np.add.at(node_sums, col_arr, psi)
    gamma = (n_nodes - 1) * float(node_sums @ node_sums) / y_arr.size**2
    iid_moment_variance = float(psi @ psi) / y_arr.size**2
    return DyadicDMLResult(
        estimator="plr_random_edge_diagnostic",
        directed=False,
        theta=theta,
        se_dyadic=float(np.sqrt(gamma / n_nodes) / denominator),
        se_iid=float(np.sqrt(iid_moment_variance) / denominator),
        jacobian=denominator,
        moment=float(np.mean(psi)),
        n_nodes=n_nodes,
        n_edges=int(y_arr.size),
        n_scored=int(y_arr.size),
        n_folds=n_folds,
        fold_audits=[],
        diagnostics={
            "valid_dyadic_crossfit": False,
            "purpose": "split-by-variance diagnostic only",
            "edge_fold_counts": fold_counts,
            "gamma": gamma,
            "iid_moment_variance": iid_moment_variance,
            "edge_fold_assignment": edge_folds.tolist(),
        },
    )


def _weighted_projection(
    d: FloatArray,
    x: FloatArray,
    weights: FloatArray,
    *,
    alpha: float | None,
    seed: int,
) -> tuple[float, FloatArray, dict[str, float | int]]:
    """Weighted projection of D on an unpenalized intercept and X.

    Slopes may be selected with a fixed Lasso penalty.  Selection uses
    weighted centering, so the intercept is never penalized.  The selected
    model is then refitted by exact weighted least squares.
    """

    del seed  # reserved for future node-level tuning
    w = np.maximum(np.asarray(weights, dtype=np.float64), 1e-12)
    w_sum = float(w.sum())
    x_bar = (w[:, None] * x).sum(axis=0) / w_sum
    d_bar = float(w @ d / w_sum)
    x_centered = x - x_bar
    d_centered = d - d_bar

    if alpha is None or alpha <= 0.0:
        support = np.ones(x.shape[1], dtype=bool)
    else:
        selector = Lasso(alpha=alpha, fit_intercept=False, max_iter=20_000)
        selector.fit(x_centered * np.sqrt(w)[:, None], d_centered * np.sqrt(w))
        support = np.abs(selector.coef_) > 1e-10

    gamma = np.zeros(x.shape[1], dtype=np.float64)
    if support.any():
        z = np.column_stack((np.ones(d.size), x[:, support]))
    else:
        z = np.ones((d.size, 1), dtype=np.float64)
    zw = z * np.sqrt(w)[:, None]
    dw = d * np.sqrt(w)
    coef, *_ = np.linalg.lstsq(zw, dw, rcond=None)
    intercept = float(coef[0])
    if support.any():
        gamma[support] = coef[1:]

    residual = d - intercept - x @ gamma
    intercept_moment = float(w @ residual / w_sum)
    slope_moment = float(np.max(np.abs((x * w[:, None]).T @ residual / w_sum)))
    diagnostics: dict[str, float | int] = {
        "support": int(support.sum()),
        "weighted_intercept_moment": intercept_moment,
        "max_weighted_slope_moment": slope_moment,
    }
    return intercept, gamma, diagnostics


def _unpenalized_logit_refit(
    design: FloatArray, outcome: FloatArray
) -> tuple[float, FloatArray]:
    """Maximum-likelihood logit refit with an explicit unpenalized intercept."""

    augmented = np.column_stack((np.ones(outcome.size), design))

    def objective(coefficient: FloatArray) -> tuple[float, FloatArray]:
        index = augmented @ coefficient
        value = float(np.sum(np.logaddexp(0.0, index) - outcome * index))
        gradient = augmented.T @ (expit(index) - outcome)
        return value, np.asarray(gradient, dtype=np.float64)

    fitted = minimize(
        objective,
        np.zeros(augmented.shape[1], dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 10_000, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not fitted.success:
        raise RuntimeError(f"unpenalized logit refit failed: {fitted.message}")
    coefficient = np.asarray(fitted.x, dtype=np.float64)
    return float(coefficient[0]), coefficient[1:]


def fit_logit_dml(
    y: ArrayLike,
    d: ArrayLike,
    x: ArrayLike,
    row: ArrayLike,
    col: ArrayLike,
    n_nodes: int,
    *,
    directed: bool = True,
    n_folds: int = 5,
    seed: int = 42,
    fold_assignment: ArrayLike | None = None,
    outcome_c: float = 1.0,
    gamma_alpha: float | None = None,
) -> DyadicDMLResult:
    """Fit high-dimensional logit dyadic DML with strict node folds.

    ``gamma_alpha=None`` uses a full weighted projection.  A positive value
    performs fixed-penalty slope selection followed by an unpenalized
    weighted refit.  Data-driven penalty selection must be implemented with
    node-level inner folds before it can be used for headline results.
    """

    folds = (
        make_node_folds(n_nodes, n_folds, seed)
        if fold_assignment is None
        else np.asarray(fold_assignment, dtype=np.int64)
    )
    y_arr, d_arr, x_arr, row_arr, col_arr, folds = _validate_inputs(
        y, d, x, row, col, n_nodes, folds
    )
    if not np.array_equal(np.unique(y_arr), np.array([0.0, 1.0])):
        raise ValueError("logit y must contain both 0 and 1")
    n_folds_actual = int(np.unique(folds).size)
    if n_folds_actual != n_folds:
        raise ValueError("n_folds does not match fold_assignment")

    # (outcome intercept, training theta, outcome slopes,
    #  treatment intercept, treatment slopes)
    nuisances: list[tuple[float, float, FloatArray, float, FloatArray]] = []
    audits: list[FoldAudit] = []
    projection_diagnostics: list[dict[str, float | int]] = []

    for fold in range(n_folds_actual):
        train, score = fold_edge_masks(row_arr, col_arr, folds, fold)
        audit = _fold_audit(fold, folds, row_arr, col_arr, train, score)
        audits.append(audit)
        if audit.n_score_edges == 0 or audit.n_train_edges == 0:
            raise ValueError(f"fold {fold} has an empty train or score edge set")

        y_train = y_arr[train]
        if np.unique(y_train).size != 2:
            raise ValueError(f"fold {fold} nuisance outcome contains one class")
        selector = LogisticRegression(
            C=outcome_c,
            solver="liblinear",
            l1_ratio=1.0,
            fit_intercept=True,
            max_iter=5_000,
            random_state=seed + fold,
        )
        # Select controls jointly with D.  D is always retained in the refit;
        # including it during selection avoids choosing X from a misspecified
        # outcome regression that omits the target regressor.
        selector_design = np.column_stack((d_arr[train], x_arr[train]))
        selector.fit(selector_design, y_train)
        support = np.abs(selector.coef_.reshape(-1)[1:]) > 1e-10
        selected_x = x_arr[train][:, support]
        refit_design = np.column_stack((d_arr[train], selected_x))
        outcome_intercept, refit_slopes = _unpenalized_logit_refit(
            refit_design, y_train
        )
        theta_train = float(refit_slopes[0])
        beta = np.zeros(x_arr.shape[1], dtype=np.float64)
        beta[support] = refit_slopes[1:]

        probability = expit(outcome_intercept + theta_train * d_arr[train] + x_arr[train] @ beta)
        weights = np.maximum(probability * (1.0 - probability), 1e-8)
        gamma_intercept, gamma, gamma_diag = _weighted_projection(
            d_arr[train], x_arr[train], weights, alpha=gamma_alpha, seed=seed + fold
        )
        gamma_diag["outcome_support"] = int(support.sum())
        projection_diagnostics.append(gamma_diag)
        nuisances.append(
            (outcome_intercept, theta_train, beta, gamma_intercept, gamma)
        )

    def score_and_derivative(theta: float) -> tuple[float, float]:
        score_value = 0.0
        derivative = 0.0
        for fold in range(n_folds_actual):
            _, score = fold_edge_masks(row_arr, col_arr, folds, fold)
            alpha_y, _, beta, alpha_d, gamma = nuisances[fold]
            probability = expit(alpha_y + theta * d_arr[score] + x_arr[score] @ beta)
            residual_d = d_arr[score] - alpha_d - x_arr[score] @ gamma
            psi = (y_arr[score] - probability) * residual_d
            dpsi = -probability * (1.0 - probability) * d_arr[score] * residual_d
            score_value += float(psi.mean())
            derivative += float(dpsi.mean())
        return score_value / n_folds_actual, derivative / n_folds_actual

    theta = float(np.mean([item[1] for item in nuisances]))
    for _ in range(100):
        moment, derivative = score_and_derivative(theta)
        if abs(derivative) < 1e-12:
            raise RuntimeError("logit score Jacobian is numerically zero")
        step = -moment / derivative
        # Backtracking avoids a rare separation-driven Newton explosion.
        scale = 1.0
        candidate = theta + step
        while scale > 1e-6:
            candidate_moment, _ = score_and_derivative(candidate)
            if abs(candidate_moment) <= abs(moment):
                break
            scale *= 0.5
            candidate = theta + scale * step
        theta = candidate
        if abs(scale * step) < 1e-10:
            break

    moment, derivative = score_and_derivative(theta)
    jacobian = -derivative
    if jacobian <= np.finfo(float).eps:
        raise ValueError("logit Jacobian is not positive")

    score_items: list[_FoldScores] = []
    for fold in range(n_folds_actual):
        _, score = fold_edge_masks(row_arr, col_arr, folds, fold)
        nodes = np.flatnonzero(folds == fold)
        alpha_y, _, beta, alpha_d, gamma = nuisances[fold]
        probability = expit(alpha_y + theta * d_arr[score] + x_arr[score] @ beta)
        residual_d = d_arr[score] - alpha_d - x_arr[score] @ gamma
        psi = (y_arr[score] - probability) * residual_d
        score_items.append(
            _FoldScores(fold, nodes, row_arr[score], col_arr[score], psi)
        )

    gamma_value, iid_moment_variance, node_sums = dyadic_variance_component(
        score_items, n_nodes, directed
    )
    se_dyadic = np.sqrt(gamma_value / n_nodes) / jacobian
    se_iid = np.sqrt(iid_moment_variance) / jacobian

    return DyadicDMLResult(
        estimator="logit",
        directed=directed,
        theta=float(theta),
        se_dyadic=float(se_dyadic),
        se_iid=float(se_iid),
        jacobian=float(jacobian),
        moment=float(moment),
        n_nodes=n_nodes,
        n_edges=int(y_arr.size),
        n_scored=int(sum(item.edge_count for item in score_items)),
        n_folds=n_folds_actual,
        fold_audits=audits,
        diagnostics={
            "gamma": float(gamma_value),
            "iid_moment_variance": float(iid_moment_variance),
            "max_abs_node_score": float(np.max(np.abs(node_sums))),
            "fold_assignment": folds.tolist(),
            "projection": projection_diagnostics,
        },
    )


__all__ = [
    "DyadicDMLResult",
    "FoldAudit",
    "aggregate_repeated_splits",
    "dyadic_variance_component",
    "fit_edge_split_plr_diagnostic",
    "fit_logit_dml",
    "fit_matched_score_equal_size_maximal_leak_plr_diagnostic",
    "fit_matched_score_leaky_complement_plr_diagnostic",
    "fit_plr_dml",
    "fold_edge_masks",
    "make_node_folds",
]
