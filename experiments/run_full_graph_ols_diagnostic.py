#!/usr/bin/env python3
"""Conventional full-graph OLS diagnostic with dyadic sandwiches.

This is intentionally not a replacement for the strict dyadic-DML primary
fit.  It reports a fixed-dimensional linear-projection baseline using both
the exact cross-sectional Aronow--Samii--Assenova CR0 overlap meat and the
first-order complete-graph node aggregation used by the paper's Chiang-style
diagnostic.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import norm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_continuous_dml import (
    PRIMARY_OUTCOME,
    PRIMARY_REPORTING_INCREMENT,
    PRIMARY_TREATMENT,
    design_matrix,
    treatment_reporting_scale,
)
from experiments.application_design_contract import (
    DEFAULT_MANIFEST,
    bind_application_design,
    design_spec_run_token,
    validate_driver_design_role,
)
from src.provenance import record


REDUNDANT_CONTROL_NAMES = (
    "numeric:PRE_VOL_ABS_DIFF",
    "numeric:PRE_MOMENTUM_ABS_DIFF",
    "numeric:LOG_HOLDERS_ABS_DIFF",
)
DEFAULT_DESIGN = (
    PROJECT_ROOT / "data_public/processed/continuous_design/dyadic_design.parquet"
)
DEFAULT_OLS_OUTPUT_DIR = (
    PROJECT_ROOT / "data_public/processed/full_graph_ols_diagnostic"
)


def two_sided_normal_power(effect: float, standard_error: float, alpha: float) -> float:
    """Return normal-reference power for a two-sided test at a fixed effect."""

    if standard_error <= 0.0:
        raise ValueError("standard error must be positive")
    critical = float(norm.ppf(1.0 - alpha / 2.0))
    noncentrality = abs(effect) / standard_error
    return float(
        norm.cdf(-critical - noncentrality)
        + 1.0
        - norm.cdf(critical - noncentrality)
    )


def infer_n_nodes(row: np.ndarray, col: np.ndarray) -> int:
    """Infer a contiguous node count from nonempty edge endpoints."""

    if row.size == 0 or col.size != row.size:
        raise ValueError("edge endpoints must be nonempty and equally sized")
    return int(max(np.max(row), np.max(col)) + 1)


def full_graph_ols_diagnostic(
    y: np.ndarray,
    d: np.ndarray,
    x: np.ndarray,
    row: np.ndarray,
    col: np.ndarray,
    *,
    n_nodes: int,
    reporting_increment: float,
) -> dict[str, object]:
    """Fit OLS and return scalar FWL forms of three CR0 variances."""

    y = np.asarray(y, dtype=np.float64).reshape(-1)
    d = np.asarray(d, dtype=np.float64).reshape(-1)
    x = np.asarray(x, dtype=np.float64)
    row = np.asarray(row, dtype=np.int64).reshape(-1)
    col = np.asarray(col, dtype=np.int64).reshape(-1)
    n_edges = y.size
    if d.size != n_edges or x.shape[0] != n_edges:
        raise ValueError("outcome, treatment, and controls have different row counts")
    if row.size != n_edges or col.size != n_edges:
        raise ValueError("endpoint arrays do not match the design")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(d)):
        raise ValueError("outcome and treatment must be finite")
    if x.ndim != 2 or not np.all(np.isfinite(x)):
        raise ValueError("controls must be a finite matrix")
    if reporting_increment <= 0.0 or not np.isfinite(reporting_increment):
        raise ValueError("reporting increment must be finite and positive")
    expected_row, expected_col = np.triu_indices(n_nodes, k=1)
    if not np.array_equal(row, expected_row) or not np.array_equal(col, expected_col):
        raise ValueError("OLS diagnostic requires a sorted complete undirected graph")

    controls = np.column_stack((np.ones(n_edges), x))
    control_rank = int(np.linalg.matrix_rank(controls))
    if control_rank != controls.shape[1]:
        raise ValueError("OLS controls are rank deficient after redundancy removal")
    treatment_projection = np.linalg.lstsq(controls, d, rcond=None)[0]
    d_tilde = d - controls @ treatment_projection
    denominator_sum = float(d_tilde @ d_tilde)
    if denominator_sum <= np.finfo(float).eps:
        raise ValueError("residualized treatment has no identifying variation")

    augmented = np.column_stack((d, controls))
    augmented_rank = int(np.linalg.matrix_rank(augmented))
    if augmented_rank != augmented.shape[1]:
        raise ValueError("treatment is collinear with the OLS controls")
    coefficients = np.linalg.lstsq(augmented, y, rcond=None)[0]
    theta = float(coefficients[0])
    fwl_theta = float(d_tilde @ y / denominator_sum)
    if not np.isclose(theta, fwl_theta, rtol=1e-10, atol=1e-12):
        raise AssertionError("full OLS and FWL treatment coefficients disagree")
    residual = y - augmented @ coefficients
    score = d_tilde * residual
    node_sums = np.bincount(row, weights=score, minlength=n_nodes)
    node_sums += np.bincount(col, weights=score, minlength=n_nodes)
    node_meat = float(node_sums @ node_sums)
    edge_diagonal_meat = float(score @ score)
    asa_meat = node_meat - edge_diagonal_meat
    if asa_meat <= 0.0:
        raise ValueError("Aronow--Samii--Assenova CR0 meat is not positive")

    variances = {
        "asa_dyadic_cr0": asa_meat / denominator_sum**2,
        "chiang_first_order": (
            (n_nodes - 1.0) / n_nodes * node_meat / denominator_sum**2
        ),
        "iid_hc0": edge_diagonal_meat / denominator_sum**2,
    }
    reported_theta = reporting_increment * theta
    alpha = 0.05
    target_power = 0.80
    mde_multiplier = float(
        norm.ppf(1.0 - alpha / 2.0) + norm.ppf(target_power)
    )
    estimates: dict[str, dict[str, float]] = {}
    for name, variance in variances.items():
        if variance <= 0.0 or not np.isfinite(variance):
            raise ValueError(f"{name} variance is not finite and positive")
        standard_error = reporting_increment * float(np.sqrt(variance))
        statistic = reported_theta / standard_error
        estimates[name] = {
            "theta": reported_theta,
            "standard_error": standard_error,
            "t_statistic": statistic,
            "p_value_normal": float(2.0 * norm.sf(abs(statistic))),
            "ci_lower_95": reported_theta - 1.96 * standard_error,
            "ci_upper_95": reported_theta + 1.96 * standard_error,
            "mde_80_power_two_sided_5pct": mde_multiplier * standard_error,
            "normal_power_at_observed_effect_two_sided_5pct": (
                two_sided_normal_power(reported_theta, standard_error, alpha)
            ),
            "standard_error_estimation_scale": standard_error
            / reporting_increment,
        }

    treatment_sd = float(np.std(d, ddof=1))
    residual_treatment_sd = float(np.std(d_tilde, ddof=1))
    outcome_sd = float(np.std(y, ddof=1))
    return {
        "diagnostic_only": True,
        "target": (
            "fixed-dimensional full-graph linear projection; equals the paper's "
            "semiparametric target only under the corresponding linear nuisance restrictions"
        ),
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "n_controls_after_redundancy_removal": x.shape[1],
        "control_rank_with_intercept": control_rank,
        "augmented_rank": augmented_rank,
        "reporting_increment": reporting_increment,
        "theta_estimation_scale": theta,
        "jacobian_mean_residual_treatment_square": denominator_sum / n_edges,
        "treatment_mean": float(np.mean(d)),
        "treatment_sd": treatment_sd,
        "residual_treatment_sd": residual_treatment_sd,
        "reporting_increment_in_treatment_sd": reporting_increment / treatment_sd,
        "reporting_increment_in_residual_treatment_sd": (
            reporting_increment / residual_treatment_sd
        ),
        "outcome_sd": outcome_sd,
        "reported_effect_in_outcome_sd": reported_theta / outcome_sd,
        "score_moment": float(np.mean(score)),
        "node_meat": node_meat,
        "edge_diagonal_meat": edge_diagonal_meat,
        "asa_cr0_meat": asa_meat,
        "variance_formulae": {
            "asa_dyadic_cr0": (
                "(sum_i S_i^2 - sum_e q_e^2)/(sum_e dtilde_e^2)^2"
            ),
            "chiang_first_order": (
                "((N-1)/N)*sum_i S_i^2/(sum_e dtilde_e^2)^2"
            ),
            "iid_hc0": "sum_e q_e^2/(sum_e dtilde_e^2)^2",
        },
        "estimates": estimates,
    }


def main() -> None:
    """Run the full-graph linear-projection diagnostic with overlap variances."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--design",
        type=Path,
        default=DEFAULT_DESIGN,
    )
    parser.add_argument("--design-audit", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OLS_OUTPUT_DIR,
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
        primary_output_dir=DEFAULT_OLS_OUTPUT_DIR,
        driver_name="full-graph OLS diagnostic",
    )

    frame = pl.read_parquet(args.design).sort(["NODE_I", "NODE_J"])
    x_full, feature_names = design_matrix(frame)
    missing = sorted(set(REDUNDANT_CONTROL_NAMES) - set(feature_names))
    if missing:
        raise ValueError(f"expected redundant controls are absent: {missing}")
    keep = [
        index
        for index, name in enumerate(feature_names)
        if name not in REDUNDANT_CONTROL_NAMES
    ]
    x = x_full[:, keep]
    retained_names = [feature_names[index] for index in keep]
    row = frame.get_column("NODE_I").to_numpy()
    col = frame.get_column("NODE_J").to_numpy()
    n_nodes = infer_n_nodes(row, col)
    if (
        n_nodes != design_binding.n_nodes
        or frame.height != design_binding.n_edges
        or x_full.shape[1] != design_binding.n_controls
    ):
        raise RuntimeError("loaded OLS design disagrees with its manifest binding")
    reporting = treatment_reporting_scale(PRIMARY_TREATMENT)
    if reporting.reporting_increment != PRIMARY_REPORTING_INCREMENT:
        raise AssertionError("primary reporting scale changed unexpectedly")
    result = full_graph_ols_diagnostic(
        frame.get_column(PRIMARY_OUTCOME).to_numpy(),
        frame.get_column(PRIMARY_TREATMENT).to_numpy(),
        x,
        row,
        col,
        n_nodes=n_nodes,
        reporting_increment=reporting.reporting_increment,
    )
    summary = {
        "schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "application_design_binding": design_binding.as_dict(),
        "estimator": "full-graph OLS via Frisch--Waugh--Lovell",
        "treatment": PRIMARY_TREATMENT,
        "outcome": PRIMARY_OUTCOME,
        "effect_unit": reporting.effect_unit,
        "estimation_unit": reporting.estimation_unit,
        "removed_exactly_redundant_controls": list(REDUNDANT_CONTROL_NAMES),
        "retained_control_names": retained_names,
        "reference_mapping": {
            "asa_dyadic_cr0": (
                "Aronow, Samii, and Assenova (2015) cross-sectional "
                "dyadic CR0 overlap sandwich"
            ),
            "chiang_first_order": (
                "complete-graph first-order node-overlap aggregation for this "
                "full-graph OLS diagnostic"
            ),
        },
        **result,
    }
    current_design_binding = bind_application_design(
        args.design, args.manifest, audit_path=args.design_audit
    )
    if current_design_binding != design_binding:
        raise RuntimeError("application design binding changed during OLS diagnostic")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    run_id = datetime.now(timezone.utc).strftime(
        f"full-ols-{design_spec_run_token(design_binding)}-%Y%m%dT%H%M%SZ"
    )
    record(
        run_id=run_id,
        command=[sys.executable, *sys.argv],
        config={
            "treatment": PRIMARY_TREATMENT,
            "outcome": PRIMARY_OUTCOME,
            "reporting_increment": reporting.reporting_increment,
            "removed_exactly_redundant_controls": list(REDUNDANT_CONTROL_NAMES),
            "diagnostic_only": True,
            "application_design_binding": design_binding.as_dict(),
        },
        inputs=[
            args.design,
            design_binding.audit_path,
            Path(__file__),
            PROJECT_ROOT / "experiments/application_design_contract.py",
            PROJECT_ROOT / "experiments/run_continuous_dml.py",
            PROJECT_ROOT / "src/provenance.py",
            PROJECT_ROOT / "pyproject.toml",
        ],
        outputs=[summary_path],
        status="completed",
        manifest_path=args.manifest,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
