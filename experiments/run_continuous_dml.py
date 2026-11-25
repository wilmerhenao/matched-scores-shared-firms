#!/usr/bin/env python3
"""Run the validated continuous application and split-by-variance audit.

The primary application uses 25 fixed final-run node partitions and reports the
median repeated-splitting aggregate.  A matched-score leaky-complement
estimator and a conventional random-edge estimator are retained solely as
diagnostics; neither is promoted to valid dyadic cross-fitting.  The full
leaky-complement arm changes both endpoint exposure and nuisance-training row
count; the separate equal-row-count learner audit controls the latter.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import norm
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dyadic_dml import (
    aggregate_repeated_splits,
    fit_edge_split_plr_diagnostic,
    fit_matched_score_leaky_complement_plr_diagnostic,
    fit_plr_dml,
)
from src.provenance import record


NUMERIC_CONTROLS = [
    "PRE_FF5_FISHER_Z",
    "PRE_VOL_MIN",
    "PRE_VOL_MAX",
    "PRE_VOL_ABS_DIFF",
    "PRE_MOMENTUM_MIN",
    "PRE_MOMENTUM_MAX",
    "PRE_MOMENTUM_ABS_DIFF",
    "LOG_HOLDERS_MIN",
    "LOG_HOLDERS_MAX",
    "LOG_HOLDERS_ABS_DIFF",
    "SAME_GICS_SUBINDUSTRY",
]

# The 11-sector GICS taxonomy is external to this analysis sample.  Enumerating
# all unordered pairs fixes both the categorical vocabulary and its reference
# level before any node partition is observed.
GICS_SECTORS = (
    "Communication Services",
    "Consumer Discretionary",
    "Consumer Staples",
    "Energy",
    "Financials",
    "Health Care",
    "Industrials",
    "Information Technology",
    "Materials",
    "Real Estate",
    "Utilities",
)
SECTOR_PAIR_CATEGORIES = tuple(
    f"{left} | {right}"
    for left, right in combinations_with_replacement(GICS_SECTORS, 2)
)
SECTOR_PAIR_REFERENCE = SECTOR_PAIR_CATEGORIES[0]

# Fixed before the final rerun.  The original five partitions remain
# in the set, including the historically reported partition with seed 29.
PRIMARY_SEEDS = (
    11,
    29,
    47,
    71,
    101,
    103,
    107,
    109,
    113,
    127,
    131,
    137,
    139,
    149,
    151,
    157,
    163,
    167,
    173,
    179,
    181,
    191,
    193,
    197,
    199,
)
PRIMARY_N_FOLDS = 5
K_SENSITIVITY = (2, 5, 10)
PRIMARY_ALPHA = 1.0
PRIMARY_TREATMENT = "BINARY_COSINE"
PRIMARY_OUTCOME = "Y_FF5_FISHER_Z"
PRIMARY_REPORTING_INCREMENT = 0.10
RIDGE_SOLVER_TOLERANCE = 1e-12
RIDGE_SENSITIVITY_ALPHAS = (0.0, 0.1, 1.0, 10.0, 100.0)
MEASUREMENT_CELLS = (
    ("treatment_robustness", "BINARY_JACCARD", PRIMARY_OUTCOME),
    ("treatment_robustness", "PORTFOLIO_WEIGHT_COSINE", PRIMARY_OUTCOME),
    ("raw_return_outcome_robustness", PRIMARY_TREATMENT, "Y_RAW_FISHER_Z"),
)
CONTINUOUS_DML_SCHEMA_VERSION = 4
DEFAULT_DESIGN = (
    PROJECT_ROOT / "data_public/processed/continuous_design/dyadic_design.parquet"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data_public/processed/continuous_results"
DEFAULT_MANIFEST = PROJECT_ROOT / "results/MANIFEST.jsonl"
REPEATED_SPLIT_FORMULA = (
    "theta_tilde=median_s(theta_s); "
    "se_tilde=sqrt(median_s(se_s^2+(theta_s-theta_tilde)^2))"
)

_SPLIT_BY_ESTIMATOR = {
    "plr": "strict_node",
    "plr_matched_score_leaky_complement_diagnostic": (
        "matched_score_leaky_complement_diagnostic"
    ),
    "plr_random_edge_diagnostic": "random_edge_diagnostic",
}


@dataclass(frozen=True)
class TreatmentReportingScale:
    """Separate the scale fitted by DML from the effect scale reported."""

    reporting_increment: float
    effect_unit: str
    estimation_unit: str


_TREATMENT_REPORTING_SCALES = {
    PRIMARY_TREATMENT: TreatmentReportingScale(
        reporting_increment=PRIMARY_REPORTING_INCREMENT,
        effect_unit="effect of a +0.10 raw binary reporting-filer cosine increase",
        estimation_unit="effect of a +1.00 raw binary reporting-filer cosine increase",
    ),
    "BINARY_JACCARD": TreatmentReportingScale(
        reporting_increment=0.10,
        effect_unit="effect of a +0.10 raw binary reporting-filer Jaccard increase",
        estimation_unit="effect of a +1.00 raw binary reporting-filer Jaccard increase",
    ),
    "PORTFOLIO_WEIGHT_COSINE": TreatmentReportingScale(
        reporting_increment=0.10,
        effect_unit="effect of a +0.10 raw portfolio-weight cosine increase",
        estimation_unit="effect of a +1.00 raw portfolio-weight cosine increase",
    ),
}


def treatment_reporting_scale(treatment: str) -> TreatmentReportingScale:
    """Return an explicit, fail-closed reporting convention for a treatment."""

    try:
        scale = _TREATMENT_REPORTING_SCALES[treatment]
    except KeyError as error:
        raise ValueError(f"no reporting scale is registered for {treatment}") from error
    if not np.isfinite(scale.reporting_increment) or scale.reporting_increment <= 0:
        raise ValueError(f"invalid reporting increment for {treatment}")
    return scale


def _contributes_to_designated_primary_aggregate(
    *,
    specification: str,
    split: str,
    variance: str,
    alpha: float,
    treatment: str,
    outcome: str,
    n_folds: int,
) -> bool:
    return (
        specification == "primary_seed_sensitivity"
        and split == "strict_node"
        and variance == "dyadic"
        and alpha == PRIMARY_ALPHA
        and treatment == PRIMARY_TREATMENT
        and outcome == PRIMARY_OUTCOME
        and n_folds == PRIMARY_N_FOLDS
    )


def _is_designated_primary_aggregate(
    *, analysis_group: str, metadata: dict[str, object]
) -> bool:
    return analysis_group == "primary_split_by_variance" and (
        _contributes_to_designated_primary_aggregate(
            specification=str(metadata["specification"]),
            split=str(metadata["split"]),
            variance=str(metadata["variance"]),
            alpha=float(metadata["ridge_alpha"]),
            treatment=str(metadata["treatment"]),
            outcome=str(metadata["outcome"]),
            n_folds=int(metadata["n_folds"]),
        )
    )


def ridge_sensitivity_configurations() -> list[dict[str, object]]:
    """Return non-primary ridge cells; alpha=1 reuses the primary strict fits."""

    return [
        {
            "specification": "ridge_penalty_sensitivity",
            "seed": seed,
            "alpha": alpha,
            "treatment": PRIMARY_TREATMENT,
            "outcome": PRIMARY_OUTCOME,
        }
        for alpha in RIDGE_SENSITIVITY_ALPHAS
        if alpha != PRIMARY_ALPHA
        for seed in PRIMARY_SEEDS
    ]


def design_matrix(frame: pl.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Encode controls using numeric columns and a fixed sector-pair vocabulary."""

    numeric = frame.select(NUMERIC_CONTROLS).to_numpy()
    sector_pairs = frame.get_column("SECTOR_PAIR").to_numpy()
    unexpected = sorted(set(sector_pairs) - set(SECTOR_PAIR_CATEGORIES))
    if unexpected:
        raise ValueError(f"sector pairs fall outside the fixed GICS schema: {unexpected}")
    sector = np.column_stack(
        [sector_pairs == category for category in SECTOR_PAIR_CATEGORIES[1:]]
    ).astype(float, copy=False)
    # Numeric scaling is deliberately *not* performed here: it is fitted anew
    # on each nuisance training fold by ``ridge_factory`` below.  Categorical
    # columns use the fixed schema above and therefore have no learned state.
    names = [f"numeric:{name}" for name in NUMERIC_CONTROLS]
    names.extend(f"SECTOR_PAIR_{category}" for category in SECTOR_PAIR_CATEGORIES[1:])
    matrix = np.column_stack((numeric, sector))
    if not np.all(np.isfinite(matrix)):
        raise ValueError("control matrix contains non-finite values")
    return matrix, names


def ridge_factory(alpha: float):
    """Construct a fold-local preprocessor and ridge nuisance learner."""

    numeric_columns = list(range(len(NUMERIC_CONTROLS)))
    categorical_columns = slice(len(NUMERIC_CONTROLS), None)
    preprocessing = ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric_columns),
            ("sector_pair", "passthrough", categorical_columns),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("preprocess", preprocessing),
            (
                "ridge",
                Ridge(
                    alpha=alpha,
                    solver="lsqr",
                    tol=RIDGE_SOLVER_TOLERANCE,
                ),
            ),
        ]
    )


def result_rows(
    result,
    *,
    seed: int,
    alpha: float,
    treatment: str,
    outcome: str,
    specification: str,
) -> list[dict[str, object]]:
    """Convert one fitted estimator into dyadic and IID reporting rows."""

    try:
        split = _SPLIT_BY_ESTIMATOR[result.estimator]
    except KeyError as error:
        raise ValueError(f"unrecognized PLR estimator: {result.estimator}") from error
    score_fingerprints = result.diagnostics.get("score_edge_fingerprints")
    score_fingerprints_json = (
        json.dumps(score_fingerprints, separators=(",", ":"))
        if score_fingerprints is not None
        else None
    )
    reporting = treatment_reporting_scale(treatment)
    increment = reporting.reporting_increment
    theta = increment * result.theta
    rows = []
    for variance, estimation_standard_error in [
        ("iid", result.se_iid),
        ("dyadic", result.se_dyadic),
    ]:
        standard_error = increment * estimation_standard_error
        statistic = theta / standard_error
        valid_under_maintained_assumptions = (
            split == "strict_node" and variance == "dyadic"
        )
        contributes_to_primary = _contributes_to_designated_primary_aggregate(
            specification=specification,
            split=split,
            variance=variance,
            alpha=alpha,
            treatment=treatment,
            outcome=outcome,
            n_folds=result.n_folds,
        )
        if valid_under_maintained_assumptions:
            inference_role = "maintained_assumptions_inference"
        elif split == "strict_node":
            inference_role = "iid_variance_diagnostic_only"
        elif split == "matched_score_leaky_complement_diagnostic":
            inference_role = "endpoint_leakage_diagnostic_only"
        else:
            inference_role = "conventional_random_edge_diagnostic_only"
        rows.append(
            {
                "specification": specification,
                "split": split,
                "variance": variance,
                "inference_role": inference_role,
                "valid_under_maintained_assumptions": (
                    valid_under_maintained_assumptions
                ),
                "contributes_to_designated_primary_aggregate": (
                    contributes_to_primary
                ),
                # No single partition is the designated reported result.
                "designated_primary_result": False,
                "valid_primary_inference": False,
                "seed": seed,
                "ridge_alpha": alpha,
                "treatment": treatment,
                "outcome": outcome,
                "reporting_increment": increment,
                "effect_unit": reporting.effect_unit,
                "estimation_unit": reporting.estimation_unit,
                "theta": theta,
                "standard_error": standard_error,
                "t_statistic": statistic,
                "p_value_normal": float(2 * norm.sf(abs(statistic))),
                "ci_lower_95": theta - 1.96 * standard_error,
                "ci_upper_95": theta + 1.96 * standard_error,
                "theta_estimation_scale": result.theta,
                "standard_error_estimation_scale": estimation_standard_error,
                "ci_lower_95_estimation_scale": (
                    result.theta - 1.96 * estimation_standard_error
                ),
                "ci_upper_95_estimation_scale": (
                    result.theta + 1.96 * estimation_standard_error
                ),
                "n_nodes": result.n_nodes,
                "n_edges": result.n_edges,
                "n_scored": result.n_scored,
                "n_folds": result.n_folds,
                "moment": result.moment,
                "jacobian": result.jacobian,
                "score_definition": result.diagnostics.get("score_definition"),
                "score_fold_fingerprints_json": score_fingerprints_json,
            }
        )
    return rows


def node_variance_rows(
    result,
    *,
    seed: int,
    alpha: float,
    treatment: str,
    outcome: str,
    specification: str,
    node_labels: dict[int, str],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Return node-level dyadic-variance shares and concentration diagnostics."""

    raw_contributions = result.diagnostics.get("node_variance_contributions")
    if raw_contributions is None:
        raise ValueError(
            f"{result.estimator} does not expose node variance contributions"
        )
    contributions = np.asarray(raw_contributions, dtype=float).reshape(-1)
    if contributions.size != result.n_nodes:
        raise ValueError("node variance contribution count does not equal n_nodes")
    if not np.all(np.isfinite(contributions)) or np.any(contributions < 0.0):
        raise ValueError("node variance contributions must be finite and nonnegative")
    total = float(contributions.sum())
    if total <= 0.0:
        raise ValueError("node variance contributions have zero total")
    if not np.isclose(total, result.se_dyadic**2, rtol=1e-10, atol=1e-15):
        raise AssertionError("node contributions do not sum to dyadic variance")

    split = _SPLIT_BY_ESTIMATOR[result.estimator]
    reporting = treatment_reporting_scale(treatment)
    increment = reporting.reporting_increment
    reported_contributions = contributions * increment**2
    reported_total = float(reported_contributions.sum())
    if not np.isclose(
        reported_total,
        (increment * result.se_dyadic) ** 2,
        rtol=1e-10,
        atol=1e-15,
    ):
        raise AssertionError("reported node contributions do not sum to dyadic variance")
    shares = contributions / total
    order = np.argsort(-shares, kind="stable")
    ranks = np.empty(result.n_nodes, dtype=np.int64)
    ranks[order] = np.arange(1, result.n_nodes + 1)
    detail: list[dict[str, object]] = []
    for node in range(result.n_nodes):
        detail.append(
            {
                "specification": specification,
                "split": split,
                "variance": "dyadic",
                "seed": seed,
                "n_folds": result.n_folds,
                "ridge_alpha": alpha,
                "treatment": treatment,
                "outcome": outcome,
                "reporting_increment": increment,
                "effect_unit": reporting.effect_unit,
                "estimation_unit": reporting.estimation_unit,
                "node_id": node,
                "node_label": node_labels.get(node),
                "variance_contribution": float(reported_contributions[node]),
                "variance_contribution_estimation_scale": float(
                    contributions[node]
                ),
                "variance_share": float(shares[node]),
                "variance_share_rank": int(ranks[node]),
            }
        )

    hhi = float(shares @ shares)
    concentration: dict[str, object] = {
        "specification": specification,
        "split": split,
        "variance": "dyadic",
        "seed": seed,
        "n_folds": result.n_folds,
        "ridge_alpha": alpha,
        "treatment": treatment,
        "outcome": outcome,
        "reporting_increment": increment,
        "effect_unit": reporting.effect_unit,
        "estimation_unit": reporting.estimation_unit,
        "n_nodes": result.n_nodes,
        "dyadic_variance": float((increment * result.se_dyadic) ** 2),
        "contribution_sum": reported_total,
        "dyadic_variance_estimation_scale": float(result.se_dyadic**2),
        "contribution_sum_estimation_scale": total,
        "max_share": float(shares[order[0]]),
        "top_5_share": float(shares[order[: min(5, result.n_nodes)]].sum()),
        "top_10_share": float(shares[order[: min(10, result.n_nodes)]].sum()),
        "herfindahl_share": hhi,
        "effective_nodes": float(1.0 / hhi),
        "largest_node_id": int(order[0]),
        "largest_node_label": node_labels.get(int(order[0])),
    }
    return detail, concentration


def repeated_split_summary(
    frame: pl.DataFrame,
    *,
    analysis_group: str,
    expected_seeds: tuple[int, ...],
) -> dict[str, object]:
    """Aggregate one estimator/variance/K cell over fixed final-run partitions."""

    if frame.height != len(expected_seeds):
        raise AssertionError(
            f"{analysis_group} contains {frame.height} rows; "
            f"expected {len(expected_seeds)}"
        )
    actual_seeds = tuple(sorted(frame.get_column("seed").to_list()))
    if actual_seeds != tuple(sorted(expected_seeds)):
        raise AssertionError(
            f"{analysis_group} seed set differs from the fixed final-run seed set"
        )
    invariant_columns = [
        "specification",
        "split",
        "variance",
        "inference_role",
        "valid_under_maintained_assumptions",
        "contributes_to_designated_primary_aggregate",
        "ridge_alpha",
        "treatment",
        "outcome",
        "reporting_increment",
        "effect_unit",
        "estimation_unit",
        "n_nodes",
        "n_edges",
        "n_folds",
    ]
    metadata: dict[str, object] = {}
    for column in invariant_columns:
        values = frame.get_column(column).unique().to_list()
        if len(values) != 1:
            raise AssertionError(f"{analysis_group} varies in {column}: {values}")
        metadata[column] = values[0]

    expected_source_contribution = _contributes_to_designated_primary_aggregate(
        specification=str(metadata["specification"]),
        split=str(metadata["split"]),
        variance=str(metadata["variance"]),
        alpha=float(metadata["ridge_alpha"]),
        treatment=str(metadata["treatment"]),
        outcome=str(metadata["outcome"]),
        n_folds=int(metadata["n_folds"]),
    )
    if bool(metadata["contributes_to_designated_primary_aggregate"]) != (
        expected_source_contribution
    ):
        raise AssertionError(
            f"{analysis_group} has an invalid primary-aggregate contribution flag"
        )
    # A row can be reused in a sensitivity aggregate without making that
    # sensitivity aggregate part of the designated headline cell.
    metadata["contributes_to_designated_primary_aggregate"] = (
        analysis_group == "primary_split_by_variance"
        and expected_source_contribution
    )

    estimates = frame.get_column("theta").to_numpy()
    standard_errors = frame.get_column("standard_error").to_numpy()
    theta, standard_error = aggregate_repeated_splits(estimates, standard_errors)
    estimation_scale_estimates = frame.get_column("theta_estimation_scale").to_numpy()
    estimation_scale_standard_errors = frame.get_column(
        "standard_error_estimation_scale"
    ).to_numpy()
    estimation_scale_theta, estimation_scale_standard_error = (
        aggregate_repeated_splits(
            estimation_scale_estimates, estimation_scale_standard_errors
        )
    )
    increment = float(metadata["reporting_increment"])
    if not np.isclose(theta, increment * estimation_scale_theta):
        raise AssertionError("aggregate theta does not match its reporting increment")
    if not np.isclose(
        standard_error, increment * estimation_scale_standard_error
    ):
        raise AssertionError(
            "aggregate standard error does not match its reporting increment"
        )
    statistic = theta / standard_error
    designated_primary_result = _is_designated_primary_aggregate(
        analysis_group=analysis_group, metadata=metadata
    )
    if designated_primary_result:
        aggregate_inference_role = "designated_primary_inference"
    elif bool(metadata["valid_under_maintained_assumptions"]):
        aggregate_inference_role = "maintained_assumptions_sensitivity"
    else:
        aggregate_inference_role = str(metadata["inference_role"])
    return {
        "analysis_group": analysis_group,
        **metadata,
        "inference_role": aggregate_inference_role,
        "designated_primary_result": designated_primary_result,
        # Backward-compatible name, now restricted to exactly one aggregate.
        "valid_primary_inference": designated_primary_result,
        "aggregation": "median_repeated_splitting",
        "aggregation_formula": REPEATED_SPLIT_FORMULA,
        "n_splits": len(expected_seeds),
        "seeds_json": json.dumps(list(expected_seeds), separators=(",", ":")),
        "theta": theta,
        "standard_error": standard_error,
        "t_statistic": statistic,
        "p_value_normal": float(2 * norm.sf(abs(statistic))),
        "ci_lower_95": theta - 1.96 * standard_error,
        "ci_upper_95": theta + 1.96 * standard_error,
        "theta_estimation_scale": estimation_scale_theta,
        "standard_error_estimation_scale": estimation_scale_standard_error,
        "ci_lower_95_estimation_scale": (
            estimation_scale_theta - 1.96 * estimation_scale_standard_error
        ),
        "ci_upper_95_estimation_scale": (
            estimation_scale_theta + 1.96 * estimation_scale_standard_error
        ),
        "theta_min": float(np.min(estimates)),
        "theta_max": float(np.max(estimates)),
        "theta_standard_deviation": float(np.std(estimates, ddof=1)),
        "run_standard_error_median": float(np.median(standard_errors)),
        "run_standard_error_min": float(np.min(standard_errors)),
        "run_standard_error_max": float(np.max(standard_errors)),
        "max_abs_split_deviation": float(np.max(np.abs(estimates - theta))),
        "n_scored_min": int(frame.get_column("n_scored").min()),
        "n_scored_max": int(frame.get_column("n_scored").max()),
    }


def build_repeated_split_aggregates(
    results: pl.DataFrame, *, design_is_primary: bool
) -> pl.DataFrame:
    """Build every repeated-split cell and enforce its reporting role."""

    aggregate_rows: list[dict[str, object]] = []
    primary_base = results.filter(
        (pl.col("specification") == "primary_seed_sensitivity")
        & (pl.col("treatment") == PRIMARY_TREATMENT)
        & (pl.col("outcome") == PRIMARY_OUTCOME)
        & (pl.col("n_folds") == PRIMARY_N_FOLDS)
    )
    for split in _SPLIT_BY_ESTIMATOR.values():
        for variance in ("iid", "dyadic"):
            aggregate_rows.append(
                repeated_split_summary(
                    primary_base.filter(
                        (pl.col("split") == split)
                        & (pl.col("variance") == variance)
                    ),
                    analysis_group="primary_split_by_variance",
                    expected_seeds=PRIMARY_SEEDS,
                )
            )
    for n_folds in K_SENSITIVITY:
        source_specification = (
            "primary_seed_sensitivity"
            if n_folds == PRIMARY_N_FOLDS
            else "fold_count_sensitivity"
        )
        for variance in ("iid", "dyadic"):
            aggregate_rows.append(
                repeated_split_summary(
                    results.filter(
                        (pl.col("specification") == source_specification)
                        & (pl.col("split") == "strict_node")
                        & (pl.col("variance") == variance)
                        & (pl.col("n_folds") == n_folds)
                        & (pl.col("treatment") == PRIMARY_TREATMENT)
                        & (pl.col("outcome") == PRIMARY_OUTCOME)
                    ),
                    analysis_group="strict_k_sensitivity",
                    expected_seeds=PRIMARY_SEEDS,
                )
            )
    for alpha in RIDGE_SENSITIVITY_ALPHAS:
        if alpha == PRIMARY_ALPHA:
            ridge_source = primary_base.filter(
                pl.col("split") == "strict_node"
            ).with_columns(
                pl.lit("ridge_penalty_sensitivity").alias("specification"),
                pl.lit(False).alias(
                    "contributes_to_designated_primary_aggregate"
                ),
            )
        else:
            ridge_source = results.filter(
                (pl.col("specification") == "ridge_penalty_sensitivity")
                & (pl.col("split") == "strict_node")
                & (pl.col("ridge_alpha") == alpha)
                & (pl.col("n_folds") == PRIMARY_N_FOLDS)
                & (pl.col("treatment") == PRIMARY_TREATMENT)
                & (pl.col("outcome") == PRIMARY_OUTCOME)
            )
        for variance in ("iid", "dyadic"):
            aggregate_rows.append(
                repeated_split_summary(
                    ridge_source.filter(pl.col("variance") == variance),
                    analysis_group="ridge_penalty_sensitivity",
                    expected_seeds=PRIMARY_SEEDS,
                )
            )
    for specification, treatment, outcome in MEASUREMENT_CELLS:
        for variance in ("iid", "dyadic"):
            aggregate_rows.append(
                repeated_split_summary(
                    results.filter(
                        (pl.col("specification") == specification)
                        & (pl.col("split") == "strict_node")
                        & (pl.col("variance") == variance)
                        & (pl.col("n_folds") == PRIMARY_N_FOLDS)
                        & (pl.col("treatment") == treatment)
                        & (pl.col("outcome") == outcome)
                    ),
                    analysis_group="measurement_robustness",
                    expected_seeds=PRIMARY_SEEDS,
                )
            )

    aggregates = pl.DataFrame(aggregate_rows)
    if not design_is_primary:
        aggregates = aggregates.with_columns(
            pl.lit(False).alias("designated_primary_result"),
            pl.lit(False).alias("valid_primary_inference"),
            pl.when(pl.col("valid_under_maintained_assumptions"))
            .then(pl.lit("maintained_assumptions_sensitivity"))
            .otherwise(pl.col("inference_role"))
            .alias("inference_role"),
        )
    aggregates = aggregates.sort(
        [
            "analysis_group",
            "specification",
            "ridge_alpha",
            "n_folds",
            "split",
            "variance",
            "treatment",
            "outcome",
        ]
    )
    designated = aggregates.filter(pl.col("designated_primary_result"))
    expected_designated_count = 1 if design_is_primary else 0
    if designated.height != expected_designated_count:
        raise AssertionError(
            "DML output has the wrong designated-primary aggregate count for its "
            "application-design role"
        )
    if design_is_primary:
        designated_row = designated.row(0, named=True)
        if not _is_designated_primary_aggregate(
            analysis_group=str(designated_row["analysis_group"]),
            metadata=designated_row,
        ):
            raise AssertionError("DML designated-primary aggregate has the wrong cell")
    return aggregates


def node_label_map(frame: pl.DataFrame, n_nodes: int) -> dict[int, str]:
    """Recover the unique CIK label attached to every numeric node id."""

    if not {"CIK_I", "CIK_J"}.issubset(frame.columns):
        return {}
    endpoints = pl.concat(
        [
            frame.select(
                pl.col("NODE_I").alias("node_id"), pl.col("CIK_I").alias("node_label")
            ),
            frame.select(
                pl.col("NODE_J").alias("node_id"), pl.col("CIK_J").alias("node_label")
            ),
        ]
    ).unique()
    counts = endpoints.group_by("node_id").len()
    if counts.filter(pl.col("len") != 1).height:
        raise ValueError("a node id maps to multiple CIK labels")
    labels = {
        int(item["node_id"]): str(item["node_label"])
        for item in endpoints.iter_rows(named=True)
    }
    if set(labels) != set(range(n_nodes)):
        raise ValueError("CIK endpoint labels do not cover every node")
    return labels


def main() -> None:
    # Imported lazily because the binding module imports this file's feature
    # contract when validating the encoded application matrix.
    """Run the primary DML specification and its declared diagnostic grid."""

    from experiments.application_design_contract import (
        bind_application_design,
        design_spec_run_token,
        validate_driver_design_role,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--design",
        type=Path,
        default=DEFAULT_DESIGN,
    )
    parser.add_argument("--design-audit", type=Path, default=None)
    parser.add_argument(
        "--manifest", type=Path, default=DEFAULT_MANIFEST
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--allow-sensitivity-design",
        action="store_true",
        help=(
            "allow a validated nonprimary design; requires a nondefault output directory"
        ),
    )
    args = parser.parse_args()
    design_binding = bind_application_design(
        args.design, args.manifest, audit_path=args.design_audit
    )
    validate_driver_design_role(
        design_binding,
        allow_sensitivity_design=args.allow_sensitivity_design,
        output_dir=args.output_dir,
        primary_output_dir=DEFAULT_OUTPUT_DIR,
        driver_name="continuous DML",
    )
    design_is_primary = design_binding.is_primary_default
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = pl.read_parquet(args.design).sort(["NODE_I", "NODE_J"])
    row = frame.get_column("NODE_I").to_numpy()
    col = frame.get_column("NODE_J").to_numpy()
    if row.size == 0 or np.any(row < 0) or np.any(col < 0) or np.any(row >= col):
        raise ValueError("design must contain unique undirected edges with NODE_I < NODE_J")
    n_nodes = int(max(row.max(), col.max()) + 1)
    expected_edges = n_nodes * (n_nodes - 1) // 2
    unique_edges = frame.select("NODE_I", "NODE_J").unique().height
    if frame.height != expected_edges or unique_edges != expected_edges:
        raise ValueError("validated application is not a complete undirected graph")
    x, feature_names = design_matrix(frame)
    node_labels = node_label_map(frame, n_nodes)
    if (
        n_nodes != design_binding.n_nodes
        or frame.height != design_binding.n_edges
        or x.shape[1] != design_binding.n_controls
    ):
        raise RuntimeError("loaded DML design disagrees with its manifest binding")

    rows: list[dict[str, object]] = []
    influence_rows: list[dict[str, object]] = []
    concentration_rows: list[dict[str, object]] = []
    primary_y = frame.get_column(PRIMARY_OUTCOME).to_numpy()
    treatment_descriptives: dict[str, dict[str, float]] = {}
    for treatment in _TREATMENT_REPORTING_SCALES:
        values = frame.get_column(treatment).to_numpy()
        if not np.all(np.isfinite(values)):
            raise ValueError(f"raw similarity treatment {treatment} is non-finite")
        if np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError(f"raw similarity treatment {treatment} must lie in [0, 1]")
        sample_standard_deviation = float(np.std(values, ddof=1))
        if sample_standard_deviation <= 0.0:
            raise ValueError(f"raw similarity treatment {treatment} is constant")
        treatment_descriptives[treatment] = {
            "sample_mean": float(np.mean(values)),
            "sample_standard_deviation": sample_standard_deviation,
            "sample_minimum": float(np.min(values)),
            "sample_maximum": float(np.max(values)),
        }
    primary_d = frame.get_column(PRIMARY_TREATMENT).to_numpy()

    # Strict and matched-score estimators share identical node folds and score
    # edges.  The full-complement arm changes both endpoint exposure and the
    # nuisance-training row count.  The separate equal-row-count HGB audit
    # isolates endpoint exposure from training-sample size.
    for seed in PRIMARY_SEEDS:
        factory = lambda: ridge_factory(PRIMARY_ALPHA)
        strict = fit_plr_dml(
            primary_y,
            primary_d,
            x,
            row,
            col,
            n_nodes,
            directed=False,
            n_folds=PRIMARY_N_FOLDS,
            seed=seed,
            outcome_factory=factory,
            treatment_factory=factory,
        )
        rows.extend(
            result_rows(
                strict,
                seed=seed,
                alpha=PRIMARY_ALPHA,
                treatment=PRIMARY_TREATMENT,
                outcome=PRIMARY_OUTCOME,
                specification="primary_seed_sensitivity",
            )
        )
        detail, concentration = node_variance_rows(
            strict,
            seed=seed,
            alpha=PRIMARY_ALPHA,
            treatment=PRIMARY_TREATMENT,
            outcome=PRIMARY_OUTCOME,
            specification="primary_seed_sensitivity",
            node_labels=node_labels,
        )
        influence_rows.extend(detail)
        concentration_rows.append(concentration)

        matched = fit_matched_score_leaky_complement_plr_diagnostic(
            primary_y,
            primary_d,
            x,
            row,
            col,
            n_nodes,
            directed=False,
            n_folds=PRIMARY_N_FOLDS,
            seed=seed,
            outcome_factory=factory,
            treatment_factory=factory,
        )
        if strict.n_scored != matched.n_scored:
            raise AssertionError("strict and matched-score estimators score different counts")
        if (
            strict.diagnostics["score_edge_fingerprints"]
            != matched.diagnostics["score_edge_fingerprints"]
        ):
            raise AssertionError("strict and matched-score score fingerprints differ")
        if not matched.diagnostics.get("endpoint_leakage_observed", False):
            raise AssertionError("matched-score diagnostic did not expose endpoint leakage")
        rows.extend(
            result_rows(
                matched,
                seed=seed,
                alpha=PRIMARY_ALPHA,
                treatment=PRIMARY_TREATMENT,
                outcome=PRIMARY_OUTCOME,
                specification="primary_seed_sensitivity",
            )
        )
        detail, concentration = node_variance_rows(
            matched,
            seed=seed,
            alpha=PRIMARY_ALPHA,
            treatment=PRIMARY_TREATMENT,
            outcome=PRIMARY_OUTCOME,
            specification="primary_seed_sensitivity",
            node_labels=node_labels,
        )
        influence_rows.extend(detail)
        concentration_rows.append(concentration)

        conventional = fit_edge_split_plr_diagnostic(
            primary_y,
            primary_d,
            x,
            row,
            col,
            n_nodes,
            n_folds=PRIMARY_N_FOLDS,
            seed=seed,
            outcome_factory=factory,
            treatment_factory=factory,
        )
        rows.extend(
            result_rows(
                conventional,
                seed=seed,
                alpha=PRIMARY_ALPHA,
                treatment=PRIMARY_TREATMENT,
                outcome=PRIMARY_OUTCOME,
                specification="primary_seed_sensitivity",
            )
        )

    # K=5 is the primary block above and is reused in the aggregate table.
    # K=2 and K=10 use the same fixed final-run 25 seeds.
    for n_folds in (2, 10):
        for seed in PRIMARY_SEEDS:
            factory = lambda: ridge_factory(PRIMARY_ALPHA)
            strict = fit_plr_dml(
                primary_y,
                primary_d,
                x,
                row,
                col,
                n_nodes,
                directed=False,
                n_folds=n_folds,
                seed=seed,
                outcome_factory=factory,
                treatment_factory=factory,
            )
            rows.extend(
                result_rows(
                    strict,
                    seed=seed,
                    alpha=PRIMARY_ALPHA,
                    treatment=PRIMARY_TREATMENT,
                    outcome=PRIMARY_OUTCOME,
                    specification="fold_count_sensitivity",
                )
            )
            detail, concentration = node_variance_rows(
                strict,
                seed=seed,
                alpha=PRIMARY_ALPHA,
                treatment=PRIMARY_TREATMENT,
                outcome=PRIMARY_OUTCOME,
                specification="fold_count_sensitivity",
                node_labels=node_labels,
            )
            influence_rows.extend(detail)
            concentration_rows.append(concentration)

    # Ridge-penalty and measurement robustness both use all 25 fixed final-run
    # partitions.  Alpha=1 reuses the strict primary fits in the aggregate.
    configurations = ridge_sensitivity_configurations()
    for specification, treatment, outcome in MEASUREMENT_CELLS:
        for seed in PRIMARY_SEEDS:
            configurations.append(
                {
                    "specification": specification,
                    "seed": seed,
                    "alpha": PRIMARY_ALPHA,
                    "treatment": treatment,
                    "outcome": outcome,
                }
            )

    for config in configurations:
        seed = int(config["seed"])
        alpha = float(config["alpha"])
        treatment = str(config["treatment"])
        outcome = str(config["outcome"])
        factory = lambda alpha=alpha: ridge_factory(alpha)
        y = frame.get_column(outcome).to_numpy()
        d = frame.get_column(treatment).to_numpy()
        strict = fit_plr_dml(
            y,
            d,
            x,
            row,
            col,
            n_nodes,
            directed=False,
            n_folds=PRIMARY_N_FOLDS,
            seed=seed,
            outcome_factory=factory,
            treatment_factory=factory,
        )
        rows.extend(
            result_rows(
                strict,
                seed=seed,
                alpha=alpha,
                treatment=treatment,
                outcome=outcome,
                specification=str(config["specification"]),
            )
        )
        detail, concentration = node_variance_rows(
            strict,
            seed=seed,
            alpha=alpha,
            treatment=treatment,
            outcome=outcome,
            specification=str(config["specification"]),
            node_labels=node_labels,
        )
        influence_rows.extend(detail)
        concentration_rows.append(concentration)

    results = pl.DataFrame(rows).with_columns(
        pl.lit(design_binding.design_spec).alias("application_design_spec"),
        pl.lit(design_binding.design_role).alias("application_design_role"),
        pl.lit(design_binding.is_primary_default).alias(
            "application_design_is_primary_default"
        ),
    )
    if not design_is_primary:
        results = results.with_columns(
            pl.when(pl.col("valid_under_maintained_assumptions"))
            .then(pl.lit("maintained_assumptions_sensitivity"))
            .otherwise(pl.col("inference_role"))
            .alias("inference_role")
        )
    results = results.sort(
        [
            "specification",
            "treatment",
            "outcome",
            "n_folds",
            "seed",
            "split",
            "variance",
        ]
    )
    influence = pl.DataFrame(influence_rows).with_columns(
        pl.lit(design_binding.design_spec).alias("application_design_spec"),
        pl.lit(design_binding.design_role).alias("application_design_role"),
        pl.lit(design_binding.is_primary_default).alias(
            "application_design_is_primary_default"
        ),
    ).sort(
        ["specification", "n_folds", "seed", "split", "variance_share_rank"]
    )
    concentration = pl.DataFrame(concentration_rows).with_columns(
        pl.lit(design_binding.design_spec).alias("application_design_spec"),
        pl.lit(design_binding.design_role).alias("application_design_role"),
        pl.lit(design_binding.is_primary_default).alias(
            "application_design_is_primary_default"
        ),
    ).sort(
        ["specification", "n_folds", "seed", "split"]
    )

    aggregates = build_repeated_split_aggregates(
        results, design_is_primary=design_is_primary
    ).with_columns(
        pl.lit(design_binding.design_spec).alias("application_design_spec"),
        pl.lit(design_binding.design_role).alias("application_design_role"),
        pl.lit(design_binding.is_primary_default).alias(
            "application_design_is_primary_default"
        ),
    )
    expected_designated_count = 1 if design_is_primary else 0

    current_design_binding = bind_application_design(
        args.design, args.manifest, audit_path=args.design_audit
    )
    if current_design_binding != design_binding:
        raise RuntimeError("application design binding changed during continuous DML")

    result_path = args.output_dir / "estimates.parquet"
    csv_path = args.output_dir / "estimates.csv"
    aggregate_path = args.output_dir / "repeated_split_aggregates.parquet"
    aggregate_csv_path = args.output_dir / "repeated_split_aggregates.csv"
    influence_path = args.output_dir / "node_variance_contributions.parquet"
    influence_csv_path = args.output_dir / "node_variance_contributions.csv"
    concentration_path = args.output_dir / "node_influence_concentration.parquet"
    concentration_csv_path = args.output_dir / "node_influence_concentration.csv"
    summary_path = args.output_dir / "summary.json"
    feature_path = args.output_dir / "feature_manifest.json"
    results.write_parquet(result_path)
    results.write_csv(csv_path)
    aggregates.write_parquet(aggregate_path)
    aggregates.write_csv(aggregate_csv_path)
    influence.write_parquet(influence_path)
    influence.write_csv(influence_csv_path)
    concentration.write_parquet(concentration_path)
    concentration.write_csv(concentration_csv_path)

    primary_base = results.filter(
        (pl.col("specification") == "primary_seed_sensitivity")
        & (pl.col("treatment") == PRIMARY_TREATMENT)
        & (pl.col("outcome") == PRIMARY_OUTCOME)
        & (pl.col("n_folds") == PRIMARY_N_FOLDS)
    )
    primary_aggregate = aggregates.filter(
        (pl.col("analysis_group") == "primary_split_by_variance")
        & (pl.col("split") == "strict_node")
        & (pl.col("variance") == "dyadic")
    ).row(0, named=True)
    primary_runs = primary_base.filter(
        (pl.col("split") == "strict_node") & (pl.col("variance") == "dyadic")
    )
    primary_concentration = concentration.filter(
        (pl.col("specification") == "primary_seed_sensitivity")
        & (pl.col("split") == "strict_node")
        & (pl.col("n_folds") == PRIMARY_N_FOLDS)
    )
    ridge_aggregates = aggregates.filter(
        (pl.col("analysis_group") == "ridge_penalty_sensitivity")
        & (pl.col("split") == "strict_node")
        & (pl.col("variance") == "dyadic")
        & (pl.col("treatment") == PRIMARY_TREATMENT)
        & (pl.col("outcome") == PRIMARY_OUTCOME)
    ).sort("ridge_alpha")
    if ridge_aggregates.height != len(RIDGE_SENSITIVITY_ALPHAS):
        raise AssertionError("ridge sensitivity aggregate is incomplete")
    primary_reporting = treatment_reporting_scale(PRIMARY_TREATMENT)
    summary = {
        "schema_version": CONTINUOUS_DML_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "application_design_binding": design_binding.as_dict(),
        "run_role": (
            "primary_application" if design_is_primary else "application_sensitivity"
        ),
        "n_nodes": n_nodes,
        "n_edges": frame.height,
        "n_controls": x.shape[1],
        "primary_treatment": {
            "column": PRIMARY_TREATMENT,
            "preprocessing": "none; raw design column passed directly to DML",
            "reporting_increment": primary_reporting.reporting_increment,
            "effect_unit": primary_reporting.effect_unit,
            "estimation_unit": primary_reporting.estimation_unit,
        },
        "treatment_descriptive_statistics": {
            "status": (
                "post-estimation description only; no statistic is supplied "
                "to a nuisance learner"
            ),
            "raw_columns": treatment_descriptives,
        },
        "fixed_final_run_seeds": list(PRIMARY_SEEDS),
        "repeated_split_aggregation": {
            "name": "median repeated-splitting aggregate",
            "formula": REPEATED_SPLIT_FORMULA,
            "reference": (
                "Chernozhukov et al. (2018), Econometrics Journal, Section 3.4"
            ),
        },
        "primary": {
            "n_splits": int(primary_aggregate["n_splits"]),
            "n_folds": int(primary_aggregate["n_folds"]),
            "theta": float(primary_aggregate["theta"]),
            "standard_error": float(primary_aggregate["standard_error"]),
            "t_statistic": float(primary_aggregate["t_statistic"]),
            "p_value_normal": float(primary_aggregate["p_value_normal"]),
            "ci_lower_95": float(primary_aggregate["ci_lower_95"]),
            "ci_upper_95": float(primary_aggregate["ci_upper_95"]),
            "reporting_increment": float(
                primary_aggregate["reporting_increment"]
            ),
            "effect_unit": str(primary_aggregate["effect_unit"]),
            "theta_estimation_scale": float(
                primary_aggregate["theta_estimation_scale"]
            ),
            "standard_error_estimation_scale": float(
                primary_aggregate["standard_error_estimation_scale"]
            ),
            "ci_lower_95_estimation_scale": float(
                primary_aggregate["ci_lower_95_estimation_scale"]
            ),
            "ci_upper_95_estimation_scale": float(
                primary_aggregate["ci_upper_95_estimation_scale"]
            ),
            "theta_min": float(primary_aggregate["theta_min"]),
            "theta_max": float(primary_aggregate["theta_max"]),
            # Backward-compatible keys for schema version 1 consumers.
            "theta_median": float(primary_aggregate["theta"]),
            "se_dyadic_median": float(
                primary_runs.get_column("standard_error").median()
            ),
            "n_scored_min": int(primary_aggregate["n_scored_min"]),
            "n_scored_max": int(primary_aggregate["n_scored_max"]),
        },
        "ridge_penalty_sensitivity": {
            "status": "25-split diagnostic for each alpha; alpha=1 reuses primary fits",
            "alphas": list(RIDGE_SENSITIVITY_ALPHAS),
            "n_splits_per_alpha": len(PRIMARY_SEEDS),
            "dyadic_cells": [
                {
                    "ridge_alpha": float(cell["ridge_alpha"]),
                    "theta": float(cell["theta"]),
                    "standard_error": float(cell["standard_error"]),
                    "t_statistic": float(cell["t_statistic"]),
                    "p_value_normal": float(cell["p_value_normal"]),
                }
                for cell in ridge_aggregates.iter_rows(named=True)
            ],
        },
        "individual_influence_concentration": {
            "measure": "share of estimated dyadic variance by node",
            "max_share_median": float(
                primary_concentration.get_column("max_share").median()
            ),
            "max_share_worst_partition": float(
                primary_concentration.get_column("max_share").max()
            ),
            "top_10_share_median": float(
                primary_concentration.get_column("top_10_share").median()
            ),
            "effective_nodes_median": float(
                primary_concentration.get_column("effective_nodes").median()
            ),
        },
        "measurement_robustness_uses_repeated_split_aggregation": True,
        "ridge_penalty_sensitivity_uses_repeated_split_aggregation": True,
        "interpretation": (
            "descriptive association for a +0.10 raw binary reporting-filer "
            "cosine increase"
        ),
        "strict_node_dyadic_rows_valid_under_maintained_assumptions": True,
        "individual_rows_are_designated_primary_results": False,
        "designated_primary_aggregate_count": expected_designated_count,
        "matched_score_rows_valid_for_inference": False,
        "random_edge_rows_valid_for_inference": False,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    feature_path.write_text(
        json.dumps(
            {
                "numeric_controls": NUMERIC_CONTROLS,
                "encoded_features": feature_names,
                "n_features": len(feature_names),
                "sector_pair_schema_source": "fixed 11-sector GICS taxonomy",
                "sector_pair_categories": list(SECTOR_PAIR_CATEGORIES),
                "sector_pair_reference": SECTOR_PAIR_REFERENCE,
                "sector_pair_vocabulary_fitted_on_analysis_graph": False,
                "primary_treatment": PRIMARY_TREATMENT,
                "primary_treatment_preprocessing": "none",
                "primary_reporting_increment": PRIMARY_REPORTING_INCREMENT,
                "treatment_reporting_scales": {
                    treatment: {
                        "reporting_increment": scale.reporting_increment,
                        "effect_unit": scale.effect_unit,
                        "estimation_unit": scale.estimation_unit,
                    }
                    for treatment, scale in _TREATMENT_REPORTING_SCALES.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    run_id = datetime.now(timezone.utc).strftime(
        f"continuous-dml-{design_spec_run_token(design_binding)}-%Y%m%dT%H%M%SZ"
    )
    record(
        run_id=run_id,
        command=[sys.executable, *sys.argv],
        config={
            "primary_n_folds": PRIMARY_N_FOLDS,
            "primary_seeds": list(PRIMARY_SEEDS),
            "k_sensitivity": list(K_SENSITIVITY),
            "primary_alpha": PRIMARY_ALPHA,
            "primary_treatment": PRIMARY_TREATMENT,
            "primary_treatment_preprocessing": "none",
            "primary_reporting_increment": PRIMARY_REPORTING_INCREMENT,
            "primary_effect_unit": primary_reporting.effect_unit,
            "primary_outcome": PRIMARY_OUTCOME,
            "ridge_solver": "lsqr",
            "ridge_solver_tolerance": RIDGE_SOLVER_TOLERANCE,
            "sector_pair_schema_source": "fixed 11-sector GICS taxonomy",
            "sector_pair_reference": SECTOR_PAIR_REFERENCE,
            "sector_pair_vocabulary_fitted_on_analysis_graph": False,
            "ridge_sensitivity_alphas": list(RIDGE_SENSITIVITY_ALPHAS),
            "ridge_sensitivity_seeds": list(PRIMARY_SEEDS),
            "ridge_sensitivity_aggregation": "median_repeated_splitting",
            "repeated_split_formula": REPEATED_SPLIT_FORMULA,
            "matched_score_comparator_status": "diagnostic_only",
            "random_edge_comparator_status": "diagnostic_only",
            "application_design_binding": design_binding.as_dict(),
        },
        inputs=[
            args.design,
            design_binding.audit_path,
            Path(__file__),
            PROJECT_ROOT / "experiments/application_design_contract.py",
            PROJECT_ROOT / "src/dyadic_dml.py",
            PROJECT_ROOT / "src/provenance.py",
            PROJECT_ROOT / "pyproject.toml",
        ],
        outputs=[
            result_path,
            csv_path,
            aggregate_path,
            aggregate_csv_path,
            influence_path,
            influence_csv_path,
            concentration_path,
            concentration_csv_path,
            summary_path,
            feature_path,
        ],
        status="completed",
        manifest_path=args.manifest,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(aggregates)


if __name__ == "__main__":
    main()
