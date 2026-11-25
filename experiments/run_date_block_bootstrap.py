#!/usr/bin/env python3
"""Common-date circular-block sensitivity for the continuous application."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
from threadpoolctl import threadpool_limits

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_continuous_dml import (
    PRIMARY_ALPHA,
    PRIMARY_N_FOLDS,
    PRIMARY_OUTCOME,
    PRIMARY_REPORTING_INCREMENT,
    PRIMARY_TREATMENT,
    design_matrix,
    ridge_factory,
    treatment_reporting_scale,
)
from experiments.application_design_contract import (
    DEFAULT_MANIFEST,
    bind_application_design,
    bind_design_consumer_output,
    design_spec_run_token,
    validate_driver_design_role,
)
from src.dyadic_dml import fit_plr_dml, make_node_folds
from src.provenance import record


REPRODUCIBILITY_NODE_FOLD_SEED = 29
DEFAULT_RESAMPLING_SEED = 20260808
DATE_BLOCK_SCHEMA_VERSION = 4
DEFAULT_DESIGN_DIR = PROJECT_ROOT / "data_public/processed/continuous_design"
DEFAULT_ESTIMATES = (
    PROJECT_ROOT / "data_public/processed/continuous_results/estimates.parquet"
)
DEFAULT_DATE_OUTPUT_DIR = PROJECT_ROOT / "data_public/processed/date_block_bootstrap"


def circular_block_indices(
    n_dates: int, block_length: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw one circular moving-block resample of a fixed date index."""

    if n_dates < 2:
        raise ValueError("circular block resampling requires at least two dates")
    if not 1 <= block_length <= n_dates:
        raise ValueError("block_length must lie in [1, n_dates]")
    n_blocks = (n_dates + block_length - 1) // block_length
    starts = rng.integers(0, n_dates, size=n_blocks)
    offsets = np.arange(block_length)
    return ((starts[:, None] + offsets[None, :]) % n_dates).reshape(-1)[:n_dates]


def select_reproducibility_estimate(estimates: pl.DataFrame) -> dict[str, object]:
    """Select the unique maintained-assumptions seed-29 reference estimate."""

    reporting = treatment_reporting_scale(PRIMARY_TREATMENT)
    required = {
        "specification",
        "seed",
        "split",
        "variance",
        "ridge_alpha",
        "treatment",
        "outcome",
        "n_folds",
        "reporting_increment",
        "effect_unit",
        "valid_under_maintained_assumptions",
        "contributes_to_designated_primary_aggregate",
        "designated_primary_result",
        "valid_primary_inference",
        "theta",
        "standard_error",
        "theta_estimation_scale",
        "standard_error_estimation_scale",
        "n_nodes",
        "n_edges",
    }
    missing = sorted(required - set(estimates.columns))
    if missing:
        raise ValueError(f"continuous-DML estimates lack reporting fields: {missing}")
    selected = estimates.filter(
        (pl.col("specification") == "primary_seed_sensitivity")
        & (pl.col("seed") == REPRODUCIBILITY_NODE_FOLD_SEED)
        & (pl.col("split") == "strict_node")
        & (pl.col("variance") == "dyadic")
        & (pl.col("ridge_alpha") == PRIMARY_ALPHA)
        & (pl.col("treatment") == PRIMARY_TREATMENT)
        & (pl.col("outcome") == PRIMARY_OUTCOME)
        & (pl.col("n_folds") == PRIMARY_N_FOLDS)
        & (pl.col("reporting_increment") == reporting.reporting_increment)
        & (pl.col("effect_unit") == reporting.effect_unit)
        & pl.col("valid_under_maintained_assumptions")
        & ~pl.col("designated_primary_result")
        & ~pl.col("valid_primary_inference")
    )
    if selected.height != 1:
        raise ValueError(
            "expected exactly one strict/dyadic seed-29 reproducibility estimate"
        )
    row = selected.row(0, named=True)
    numeric = [
        float(row[key])
        for key in (
            "theta",
            "standard_error",
            "theta_estimation_scale",
            "standard_error_estimation_scale",
            "n_nodes",
            "n_edges",
        )
    ]
    if not np.all(np.isfinite(numeric)) or float(row["standard_error"]) <= 0:
        raise ValueError("seed-29 reproducibility estimate is invalid")
    if not np.isclose(
        float(row["theta"]),
        reporting.reporting_increment * float(row["theta_estimation_scale"]),
    ):
        raise ValueError("seed-29 theta violates the +0.10 reporting scale")
    if not np.isclose(
        float(row["standard_error"]),
        reporting.reporting_increment
        * float(row["standard_error_estimation_scale"]),
    ):
        raise ValueError("seed-29 standard error violates the +0.10 reporting scale")
    return row


def validate_block_inputs(
    design: pl.DataFrame,
    residuals: pl.DataFrame,
    firms: pl.DataFrame,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Validate a complete design and its balanced common-date residual panel."""

    n_nodes = firms.height
    if firms.get_column("NODE_ID").to_list() != list(range(n_nodes)):
        raise ValueError("firm node IDs are not contiguous and ordered")
    node_ciks = firms.get_column("CIK").to_list()
    if residuals.columns != ["date", *node_ciks]:
        raise ValueError("residual columns do not match the ordered firm nodes")
    dates = residuals.get_column("date")
    if dates.n_unique() != residuals.height or not dates.is_sorted():
        raise ValueError("outcome residual dates are duplicate or unordered")
    residual_values = residuals.select(node_ciks).to_numpy()
    if not np.all(np.isfinite(residual_values)):
        raise ValueError("block sensitivity requires a common balanced date panel")

    row, col = np.triu_indices(n_nodes, k=1)
    if design.height != row.size:
        raise ValueError("date sensitivity requires a complete undirected design")
    row_matches = np.array_equal(row, design.get_column("NODE_I").to_numpy())
    col_matches = np.array_equal(col, design.get_column("NODE_J").to_numpy())
    if not row_matches or not col_matches:
        raise ValueError("residual and dyadic edge order disagree")
    baseline_correlation = np.corrcoef(residual_values, rowvar=False)
    baseline_y = np.arctanh(
        np.clip(baseline_correlation[row, col], -0.999999, 0.999999)
    )
    if not np.allclose(
        baseline_y,
        design.get_column(PRIMARY_OUTCOME).to_numpy(),
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError("outcome residuals do not reproduce the design outcome")
    return node_ciks, residual_values, row, col


def summarize_block_draws(
    draws: pl.DataFrame,
    block_lengths: list[int],
    repetitions: int,
) -> list[dict[str, object]]:
    """Summarize block draws without promoting sensitivities to confidence intervals.

    Besides the original point-distribution summaries, this records (i) how
    often the already-computed conditional node interval crosses zero and
    (ii) an explicitly exploratory law-of-total-variance calculation.  The
    latter is a two-layer sensitivity, not a theorem-backed joint interval.
    """

    summaries: list[dict[str, object]] = []
    for block_length in sorted(block_lengths):
        subset = draws.filter(pl.col("block_length") == block_length)
        if subset.height != repetitions:
            raise ValueError(
                f"block length {block_length} has {subset.height} draws; "
                f"expected {repetitions}"
            )
        theta = subset.get_column("theta_effect_per_0_10_raw").to_numpy()
        conditional_se = subset.get_column(
            "conditional_dyadic_se_per_0_10_raw"
        ).to_numpy()
        if not np.all(np.isfinite(theta)) or not np.all(
            np.isfinite(conditional_se)
        ):
            raise ValueError("date-resampling outputs must be finite")
        if np.any(conditional_se <= 0):
            raise ValueError("conditional dyadic standard errors must be positive")
        critical_value = 1.96
        conditional_lower = theta - critical_value * conditional_se
        conditional_upper = theta + critical_value * conditional_se
        conditional_crosses_zero = (conditional_lower <= 0.0) & (
            conditional_upper >= 0.0
        )
        date_variance = float(np.var(theta, ddof=1))
        mean_conditional_node_variance = float(np.mean(conditional_se**2))
        exploratory_total_variance = (
            mean_conditional_node_variance + date_variance
        )
        exploratory_total_se = float(np.sqrt(exploratory_total_variance))
        theta_mean = float(np.mean(theta))
        summaries.append(
            {
                "block_length": block_length,
                "repetitions": repetitions,
                "theta_mean": theta_mean,
                "theta_median": float(np.median(theta)),
                "theta_sample_sd_descriptive": float(np.sqrt(date_variance)),
                "theta_percentile_2_5": float(np.quantile(theta, 0.025)),
                "theta_percentile_97_5": float(np.quantile(theta, 0.975)),
                "conditional_dyadic_se_min": float(np.min(conditional_se)),
                "conditional_dyadic_se_median": float(np.median(conditional_se)),
                "conditional_dyadic_se_max": float(np.max(conditional_se)),
                "conditional_interval_critical_value": critical_value,
                "conditional_interval_crosses_zero_fraction": float(
                    np.mean(conditional_crosses_zero)
                ),
                "conditional_interval_excludes_zero_positive_fraction": float(
                    np.mean(conditional_lower > 0.0)
                ),
                "conditional_interval_lower_median": float(
                    np.median(conditional_lower)
                ),
                "conditional_interval_upper_median": float(
                    np.median(conditional_upper)
                ),
                "exploratory_date_variance_component": date_variance,
                "exploratory_mean_conditional_node_variance_component": (
                    mean_conditional_node_variance
                ),
                "exploratory_total_variance": exploratory_total_variance,
                "exploratory_total_standard_error": exploratory_total_se,
                "exploratory_node_variance_share": float(
                    mean_conditional_node_variance / exploratory_total_variance
                ),
                "exploratory_total_variance_center": theta_mean,
                "exploratory_total_variance_normal_range_lower": (
                    theta_mean - critical_value * exploratory_total_se
                ),
                "exploratory_total_variance_normal_range_upper": (
                    theta_mean + critical_value * exploratory_total_se
                ),
                "exploratory_total_variance_status": (
                    "law-of-total-variance sensitivity only; not a formal "
                    "joint node-time confidence interval"
                ),
            }
        )
    return summaries


def main() -> None:
    """Run common-date block sensitivities around a fixed node partition."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--design-dir",
        type=Path,
        default=DEFAULT_DESIGN_DIR,
    )
    parser.add_argument("--design-audit", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--estimates",
        type=Path,
        default=DEFAULT_ESTIMATES,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DATE_OUTPUT_DIR,
    )
    parser.add_argument(
        "--allow-sensitivity-design",
        action="store_true",
        help=(
            "allow a validated nonprimary design; requires a nondefault output directory"
        ),
    )
    parser.add_argument("--repetitions", type=int, default=500)
    parser.add_argument("--block-lengths", type=int, nargs="+", default=[5, 20])
    parser.add_argument("--seed", type=int, default=DEFAULT_RESAMPLING_SEED)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.repetitions < 2:
        raise ValueError("at least two date-resampling repetitions are required")
    if len(args.block_lengths) != len(set(args.block_lengths)):
        raise ValueError("block lengths must be unique")
    if args.workers < 1:
        raise ValueError("workers must be positive")

    design_path = args.design_dir / "dyadic_design.parquet"
    residual_path = args.design_dir / "outcome_residuals.parquet"
    firm_path = args.design_dir / "firm_nodes.parquet"
    design_binding = bind_application_design(
        design_path, args.manifest, audit_path=args.design_audit
    )
    validate_driver_design_role(
        design_binding,
        allow_sensitivity_design=args.allow_sensitivity_design,
        output_dir=args.output_dir,
        primary_output_dir=DEFAULT_DATE_OUTPUT_DIR,
        driver_name="date-block sensitivity",
    )
    estimate_binding = bind_design_consumer_output(
        design_binding,
        args.estimates,
        run_prefix="continuous-dml-",
    )
    design = pl.read_parquet(design_path).sort(["NODE_I", "NODE_J"])
    residuals = pl.read_parquet(residual_path)
    firms = pl.read_parquet(firm_path).sort("NODE_ID")
    _, residual_values, row, col = validate_block_inputs(
        design, residuals, firms
    )
    n_nodes = firms.height
    for block_length in args.block_lengths:
        if not 1 <= block_length <= residual_values.shape[0]:
            raise ValueError("every block length must lie in [1, n_dates]")
    x, _ = design_matrix(design)
    if (
        n_nodes != design_binding.n_nodes
        or design.height != design_binding.n_edges
        or x.shape[1] != design_binding.n_controls
    ):
        raise RuntimeError("loaded date-block design disagrees with its manifest binding")
    d = design.get_column(PRIMARY_TREATMENT).to_numpy()
    if not np.all(np.isfinite(d)) or np.any(d < 0.0) or np.any(d > 1.0):
        raise ValueError(
            "raw binary reporting-filer cosine must be finite and lie in [0, 1]"
        )
    reporting = treatment_reporting_scale(PRIMARY_TREATMENT)
    if reporting.reporting_increment != PRIMARY_REPORTING_INCREMENT:
        raise AssertionError("primary reporting increment is internally inconsistent")
    estimate_rows = pl.read_parquet(args.estimates)
    canonical = select_reproducibility_estimate(estimate_rows)
    if (
        int(canonical["n_nodes"]) != n_nodes
        or int(canonical["n_edges"]) != design.height
    ):
        raise ValueError("seed-29 estimate does not match the design dimensions")
    node_folds = make_node_folds(
        n_nodes, PRIMARY_N_FOLDS, seed=REPRODUCIBILITY_NODE_FOLD_SEED
    )

    tasks: list[tuple[int, int, np.ndarray]] = []
    for block_length in sorted(args.block_lengths):
        # Derive one stream per block length so reordering the CLI arguments
        # cannot change either block-specific sensitivity distribution.
        rng = np.random.default_rng(
            np.random.SeedSequence([args.seed, block_length])
        )
        for repetition in range(args.repetitions):
            indices = circular_block_indices(
                residual_values.shape[0], block_length, rng
            )
            tasks.append((block_length, repetition, indices))

    def one_draw(task: tuple[int, int, np.ndarray]) -> dict[str, object]:
        block_length, repetition, indices = task
        correlation = np.corrcoef(residual_values[indices], rowvar=False)
        y = np.arctanh(np.clip(correlation[row, col], -0.999999, 0.999999))
        local_factory = lambda: ridge_factory(PRIMARY_ALPHA)
        estimate = fit_plr_dml(
            y,
            d,
            x,
            row,
            col,
            n_nodes,
            directed=False,
            n_folds=PRIMARY_N_FOLDS,
            fold_assignment=node_folds,
            outcome_factory=local_factory,
            treatment_factory=local_factory,
        )
        return {
            "block_length": block_length,
            "repetition": repetition,
            "treatment": PRIMARY_TREATMENT,
            "reporting_increment": reporting.reporting_increment,
            "effect_unit": reporting.effect_unit,
            "theta_effect_per_0_10_raw": (
                reporting.reporting_increment * estimate.theta
            ),
            "conditional_dyadic_se_per_0_10_raw": (
                reporting.reporting_increment * estimate.se_dyadic
            ),
            "theta_estimation_scale_per_1_00_raw": estimate.theta,
            "conditional_dyadic_se_estimation_scale_per_1_00_raw": (
                estimate.se_dyadic
            ),
        }

    rows: list[dict[str, object]] = []
    with threadpool_limits(limits=1):
        if args.workers == 1:
            for completed, task in enumerate(tasks, start=1):
                rows.append(one_draw(task))
                if completed % 20 == 0 or completed == len(tasks):
                    print(f"completed {completed}/{len(tasks)}", flush=True)
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(one_draw, task) for task in tasks]
                for completed, future in enumerate(as_completed(futures), start=1):
                    rows.append(future.result())
                    if completed % 20 == 0 or completed == len(tasks):
                        print(f"completed {completed}/{len(tasks)}", flush=True)

    draws = pl.DataFrame(rows).with_columns(
        pl.lit(design_binding.design_spec).alias("application_design_spec"),
        pl.lit(design_binding.design_role).alias("application_design_role"),
        pl.lit(design_binding.is_primary_default).alias(
            "application_design_is_primary_default"
        ),
    ).sort(["block_length", "repetition"])
    by_block = summarize_block_draws(
        draws, args.block_lengths, args.repetitions
    )

    current_design_binding = bind_application_design(
        design_path, args.manifest, audit_path=args.design_audit
    )
    current_estimate_binding = bind_design_consumer_output(
        current_design_binding,
        args.estimates,
        run_prefix="continuous-dml-",
    )
    if (
        current_design_binding != design_binding
        or current_estimate_binding != estimate_binding
    ):
        raise RuntimeError("date-block input binding changed during execution")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    draws_path = args.output_dir / "draws.parquet"
    summary_path = args.output_dir / "summary.json"
    draws.write_parquet(draws_path)
    summary = {
        "schema_version": DATE_BLOCK_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sensitivity_only": True,
        "application_design_binding": design_binding.as_dict(),
        "continuous_dml_estimate_binding": estimate_binding.as_dict(),
        "treatment": PRIMARY_TREATMENT,
        "treatment_preprocessing": "none; raw design column passed directly to DML",
        "reporting_increment": reporting.reporting_increment,
        "effect_unit": reporting.effect_unit,
        "estimation_unit": reporting.estimation_unit,
        "reproducibility_node_fold_seed": REPRODUCIBILITY_NODE_FOLD_SEED,
        "reproducibility_theta": canonical["theta"],
        "reproducibility_conditional_dyadic_se": canonical["standard_error"],
        "reproducibility_theta_estimation_scale": canonical[
            "theta_estimation_scale"
        ],
        "reproducibility_conditional_dyadic_se_estimation_scale": canonical[
            "standard_error_estimation_scale"
        ],
        "n_dates": residual_values.shape[0],
        "n_nodes": n_nodes,
        "method": (
            "common-date circular block resampling of the fixed FF5-residual "
            "panel with nuisance refit at node-fold seed 29; effects and "
            "conditional standard errors reported per +0.10 raw cosine"
        ),
        "inference_status": (
            "block-only sensitivity distribution; percentile ranges are not "
            "confidence intervals; theta sample SDs are descriptive summaries "
            "only; draw-wise conditional intervals and total-variance ranges "
            "are exploratory sensitivities, not formal joint node-time inference"
        ),
        "by_block_length": by_block,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    run_id = datetime.now(timezone.utc).strftime(
        f"date-block-{design_spec_run_token(design_binding)}-%Y%m%dT%H%M%SZ"
    )
    record(
        run_id=run_id,
        command=[sys.executable, *sys.argv],
        config={
            "repetitions": args.repetitions,
            "block_lengths": args.block_lengths,
            "seed": args.seed,
            "workers": args.workers,
            "node_fold_seed": REPRODUCIBILITY_NODE_FOLD_SEED,
            "treatment": PRIMARY_TREATMENT,
            "treatment_preprocessing": "none",
            "reporting_increment": reporting.reporting_increment,
            "effect_unit": reporting.effect_unit,
            "sensitivity_only": True,
            "application_design_binding": design_binding.as_dict(),
            "continuous_dml_estimate_binding": estimate_binding.as_dict(),
        },
        inputs=[
            design_path,
            design_binding.audit_path,
            residual_path,
            firm_path,
            args.estimates,
            Path(__file__),
            PROJECT_ROOT / "experiments/application_design_contract.py",
            PROJECT_ROOT / "experiments/run_continuous_dml.py",
            PROJECT_ROOT / "src/dyadic_dml.py",
            PROJECT_ROOT / "src/provenance.py",
            PROJECT_ROOT / "pyproject.toml",
        ],
        outputs=[draws_path, summary_path],
        status="completed",
        manifest_path=args.manifest,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
