#!/usr/bin/env python3
"""Monte Carlo calibration of the exact 25-partition reporting rule.

The canonical simulation reports one node partition per replication.  This
diagnostic instead applies the application's fixed 25 fold seeds, including
the split-dispersion term, to strict learned and matched-oracle fits at node
and feature dimensions bound to the current validated application design.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import norm
from threadpoolctl import threadpool_limits

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.application_design_contract import (
    DEFAULT_DESIGN,
    DEFAULT_MANIFEST,
    bind_application_design,
    design_spec_run_token,
    validate_driver_design_role,
)
from experiments.run_canonical_simulation import (
    THETA_ZERO,
    generate_plr_dgp,
    oracle_factories,
)
from experiments.run_continuous_dml import (
    NUMERIC_CONTROLS,
    PRIMARY_ALPHA,
    PRIMARY_N_FOLDS,
    PRIMARY_SEEDS,
    RIDGE_SOLVER_TOLERANCE,
    ridge_factory,
)
from src.dyadic_dml import aggregate_repeated_splits, fit_plr_dml, make_node_folds
from src.provenance import record


N_FOLDS = PRIMARY_N_FOLDS
RIDGE_ALPHA = PRIMARY_ALPHA
RIDGE_SOLVER = "lsqr"
DEFAULT_REPLICATIONS = 500
DEFAULT_SEED = 20260808
DEFAULT_CALIBRATION_OUTPUT_DIR = (
    PROJECT_ROOT / "data_public/processed/repeated_split_calibration"
)


def positive_int(value: str) -> int:
    """Parse a strictly positive command-line integer."""

    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line interface for repeated-split calibration."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--design-audit", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--allow-sensitivity-design",
        action="store_true",
        help=(
            "allow a validated nonprimary design; requires a nondefault output directory"
        ),
    )
    parser.add_argument("--workers", type=positive_int, default=2)
    parser.add_argument(
        "--replications", type=positive_int, default=DEFAULT_REPLICATIONS
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_CALIBRATION_OUTPUT_DIR,
    )
    return parser


def calibration_tasks(
    *,
    n_nodes: int,
    n_controls: int,
    replications: int,
    seed: int,
    fold_seeds: tuple[int, ...],
) -> list[tuple[int, int, int, int, tuple[int, ...]]]:
    """Construct reporting-rule tasks from manifest-bound design dimensions."""

    if n_nodes < 2 * N_FOLDS:
        raise ValueError("calibration requires at least two nodes per score fold")
    if n_controls < len(NUMERIC_CONTROLS) or replications <= 0:
        raise ValueError(
            "calibration controls must contain the application numeric block and "
            "replications must be positive"
        )
    if seed < 0:
        raise ValueError("calibration seed must be nonnegative")
    if (
        not fold_seeds
        or len(fold_seeds) != len(set(fold_seeds))
        or any(fold_seed < 0 for fold_seed in fold_seeds)
    ):
        raise ValueError("fold seeds must be nonempty, unique, and nonnegative")
    dgp_seeds = [
        int(child.generate_state(1)[0])
        for child in np.random.SeedSequence(seed).spawn(replications)
    ]
    if len(dgp_seeds) != len(set(dgp_seeds)):
        raise RuntimeError("spawned DGP seeds are not unique")
    return [
        (n_nodes, n_controls, repetition, dgp_seed, fold_seeds)
        for repetition, dgp_seed in enumerate(dgp_seeds)
    ]


def one_replication(
    payload: tuple[int, int, int, int, tuple[int, ...]]
) -> dict[str, float | int | str]:
    """Apply the exact 25-partition reporting rule to one simulated graph."""

    n_nodes, n_controls, repetition, dgp_seed, fold_seeds = payload
    if n_nodes < 2 * N_FOLDS:
        raise ValueError("calibration requires at least two nodes per score fold")
    if n_controls < len(NUMERIC_CONTROLS):
        raise ValueError("calibration is missing the application numeric control block")
    if repetition < 0 or dgp_seed < 0:
        raise ValueError("repetition and DGP seed must be nonnegative")
    if (
        len(fold_seeds) < 1
        or len(set(fold_seeds)) != len(fold_seeds)
        or any(fold_seed < 0 for fold_seed in fold_seeds)
    ):
        raise ValueError("fold seeds must be nonempty, unique, and nonnegative")
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        y, d, x, row, col = generate_plr_dgp(
            n_nodes, n_controls, dgp_seed
        )
        oracle_outcome_factory, oracle_treatment_factory = oracle_factories(
            n_controls
        )
        split_values: dict[str, list[float]] = {
            "strict_theta": [],
            "strict_se": [],
            "oracle_theta": [],
            "oracle_se": [],
        }
        n_scored: int | None = None
        max_abs_moment = 0.0
        factory = lambda: ridge_factory(RIDGE_ALPHA)
        for fold_seed in fold_seeds:
            node_folds = make_node_folds(n_nodes, N_FOLDS, seed=fold_seed)
            strict = fit_plr_dml(
                y,
                d,
                x,
                row,
                col,
                n_nodes,
                directed=False,
                n_folds=N_FOLDS,
                fold_assignment=node_folds,
                outcome_factory=factory,
                treatment_factory=factory,
            )
            oracle = fit_plr_dml(
                y,
                d,
                x,
                row,
                col,
                n_nodes,
                directed=False,
                n_folds=N_FOLDS,
                fold_assignment=node_folds,
                outcome_factory=oracle_outcome_factory,
                treatment_factory=oracle_treatment_factory,
            )
            if strict.n_scored != oracle.n_scored:
                raise AssertionError("strict and oracle scored-pair counts differ")
            for name, fit in (("strict", strict), ("oracle", oracle)):
                numerical = np.asarray(
                    [fit.theta, fit.se_dyadic, fit.jacobian, fit.moment], dtype=float
                )
                if not np.all(np.isfinite(numerical)) or fit.se_dyadic <= 0.0:
                    raise RuntimeError(
                        f"{name} repeated-split fit produced invalid numerical output"
                    )
            if n_scored is None:
                n_scored = strict.n_scored
            elif strict.n_scored != n_scored:
                raise AssertionError("balanced partitions changed scored-pair count")
            split_values["strict_theta"].append(strict.theta)
            split_values["strict_se"].append(strict.se_dyadic)
            split_values["oracle_theta"].append(oracle.theta)
            split_values["oracle_se"].append(oracle.se_dyadic)
            max_abs_moment = max(
                max_abs_moment, abs(strict.moment), abs(oracle.moment)
            )

    result: dict[str, float | int | str] = {
        "n_nodes": n_nodes,
        "n_controls": n_controls,
        "n_folds": N_FOLDS,
        "n_splits": len(fold_seeds),
        "repetition": repetition,
        "dgp_seed": dgp_seed,
        "n_scored_per_split": int(n_scored or 0),
        "max_abs_split_moment": max_abs_moment,
        "runtime_seconds": time.perf_counter() - started,
        "status": "completed",
    }
    for split in ("strict", "oracle"):
        theta_values = np.asarray(split_values[f"{split}_theta"])
        se_values = np.asarray(split_values[f"{split}_se"])
        theta, standard_error = aggregate_repeated_splits(theta_values, se_values)
        result[f"{split}_theta"] = theta
        result[f"{split}_se_dyadic"] = standard_error
        result[f"{split}_theta_min"] = float(np.min(theta_values))
        result[f"{split}_theta_max"] = float(np.max(theta_values))
        result[f"{split}_theta_split_sd"] = float(
            np.std(theta_values, ddof=1) if theta_values.size > 1 else 0.0
        )
        result[f"{split}_run_se_median"] = float(np.median(se_values))
        result[f"{split}_aggregate_to_run_se_ratio"] = float(
            standard_error / np.median(se_values)
        )
    return result


def summarize_replications(frame: pl.DataFrame) -> dict[str, object]:
    """Summarize learned and oracle calibration across simulated graphs."""

    if frame.height < 2:
        raise ValueError("at least two replications are required for a summary")
    invariant_columns = ["n_nodes", "n_controls", "n_folds", "n_splits"]
    invariant: dict[str, int] = {}
    for column in invariant_columns:
        values = frame.get_column(column).unique().to_list()
        if len(values) != 1:
            raise ValueError(f"repeated-split replications vary in {column}")
        invariant[column] = int(values[0])
    for column in ("repetition", "dgp_seed"):
        if frame.get_column(column).n_unique() != frame.height:
            raise ValueError(f"repeated-split calibration contains duplicate {column}")
    result: dict[str, object] = {
        **invariant,
        "replications": frame.height,
        "runtime_seconds_total": float(frame.get_column("runtime_seconds").sum()),
        "runtime_seconds_median": float(frame.get_column("runtime_seconds").median()),
        "max_abs_split_moment": float(
            frame.get_column("max_abs_split_moment").max()
        ),
    }
    z90 = float(norm.ppf(0.95))
    z95 = float(norm.ppf(0.975))
    for split in ("strict", "oracle"):
        theta = frame.get_column(f"{split}_theta").to_numpy()
        standard_error = frame.get_column(f"{split}_se_dyadic").to_numpy()
        if (
            not np.all(np.isfinite(theta))
            or not np.all(np.isfinite(standard_error))
            or np.any(standard_error <= 0.0)
        ):
            raise ValueError(f"{split} calibration estimates or SEs are invalid")
        empirical_sd = float(np.std(theta, ddof=1))
        if not np.isfinite(empirical_sd) or empirical_sd <= 0.0:
            raise ValueError(f"{split} empirical sampling SD is not positive")
        standardized = np.abs(theta - THETA_ZERO) / standard_error
        covered90 = standardized <= z90
        covered95 = standardized <= z95
        coverage90 = float(np.mean(covered90))
        coverage95 = float(np.mean(covered95))
        mean_mcse = empirical_sd / np.sqrt(frame.height)
        result[split] = {
            "mean": float(np.mean(theta)),
            "bias": float(np.mean(theta) - THETA_ZERO),
            "mean_mcse": float(mean_mcse),
            "bias_mcse": float(mean_mcse),
            "empirical_sd": empirical_sd,
            "mean_se": float(np.mean(standard_error)),
            "se_over_sd": float(np.mean(standard_error) / empirical_sd),
            "coverage90": coverage90,
            "coverage90_successes": int(np.sum(covered90)),
            "coverage95": coverage95,
            "coverage95_successes": int(np.sum(covered95)),
            "rejection5": float(1.0 - coverage95),
            "coverage90_mcse": float(
                np.sqrt(coverage90 * (1.0 - coverage90) / frame.height)
            ),
            "coverage95_mcse": float(
                np.sqrt(coverage95 * (1.0 - coverage95) / frame.height)
            ),
            "abs_t_quantile90": float(np.quantile(standardized, 0.90)),
            "abs_t_quantile95": float(np.quantile(standardized, 0.95)),
            "aggregate_to_run_se_ratio_median": float(
                frame.get_column(
                    f"{split}_aggregate_to_run_se_ratio"
                ).median()
            ),
        }
    return result


def main() -> None:
    """Run the application-matched repeated-split calibration experiment."""

    parser = build_parser()
    args = parser.parse_args()
    if args.replications < 2:
        parser.error("--replications must be at least two")
    fold_seeds = tuple(int(seed) for seed in PRIMARY_SEEDS)
    if len(fold_seeds) != 25:
        raise AssertionError("application reporting rule no longer uses 25 seeds")
    design_binding = bind_application_design(
        args.design, args.manifest, audit_path=args.design_audit
    )
    validate_driver_design_role(
        design_binding,
        allow_sensitivity_design=args.allow_sensitivity_design,
        output_dir=args.output_dir,
        primary_output_dir=DEFAULT_CALIBRATION_OUTPUT_DIR,
        driver_name="repeated-split calibration",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tasks = calibration_tasks(
        n_nodes=design_binding.n_nodes,
        n_controls=design_binding.n_controls,
        replications=args.replications,
        seed=args.seed,
        fold_seeds=fold_seeds,
    )
    rows: list[dict[str, float | int | str]] = []
    if args.workers == 1:
        for completed, task in enumerate(tasks, start=1):
            rows.append(one_replication(task))
            if completed % 5 == 0 or completed == len(tasks):
                print(f"completed {completed}/{len(tasks)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(one_replication, task) for task in tasks]
            for completed, future in enumerate(as_completed(futures), start=1):
                rows.append(future.result())
                if completed % 5 == 0 or completed == len(tasks):
                    print(f"completed {completed}/{len(tasks)}", flush=True)

    per_replication = pl.DataFrame(rows).with_columns(
        pl.lit(design_binding.design_spec).alias("application_design_spec"),
        pl.lit(design_binding.design_role).alias("application_design_role"),
        pl.lit(design_binding.is_primary_default).alias(
            "application_design_is_primary_default"
        ),
    ).sort("repetition")
    if (
        bind_application_design(
            args.design, args.manifest, audit_path=args.design_audit
        )
        != design_binding
    ):
        raise RuntimeError("application design binding changed during calibration")
    cell = summarize_replications(per_replication)
    per_path = args.output_dir / "per_replication.parquet"
    summary_path = args.output_dir / "summary.json"
    per_replication.write_parquet(per_path)
    summary = {
        "schema_version": 3,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "diagnostic_only": True,
        "theta_zero": THETA_ZERO,
        "learned_nuisance_rule": {
            "factory": "experiments.run_continuous_dml.ridge_factory",
            "alpha": RIDGE_ALPHA,
            "solver": RIDGE_SOLVER,
            "solver_tolerance": RIDGE_SOLVER_TOLERANCE,
            "preprocessing": (
                "fold-local StandardScaler on the application numeric block; "
                "fixed sector-pair indicators passed through"
            ),
        },
        "fold_seeds": list(fold_seeds),
        "aggregation": (
            "median(theta_s); sqrt(median(se_s^2+(theta_s-median(theta_s))^2))"
        ),
        "dgp": (
            "canonical jointly exchangeable Gaussian PLR at application-bound "
            "node and feature dimensions; this is an algorithm-matched calibration, "
            "not a claim that the DGP reproduces the empirical covariate distribution"
        ),
        "application_design_binding": design_binding.as_dict(),
        "cell": cell,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    run_id = datetime.now(timezone.utc).strftime(
        "repeated-split-calibration-"
        f"{design_spec_run_token(design_binding)}-%Y%m%dT%H%M%SZ"
    )
    record(
        run_id=run_id,
        command=[sys.executable, *sys.argv],
        config={
            "n_nodes": design_binding.n_nodes,
            "n_controls": design_binding.n_controls,
            "n_folds": N_FOLDS,
            "n_splits": len(fold_seeds),
            "fold_seeds": list(fold_seeds),
            "ridge_alpha": RIDGE_ALPHA,
            "ridge_solver": RIDGE_SOLVER,
            "ridge_solver_tolerance": RIDGE_SOLVER_TOLERANCE,
            "ridge_preprocessing": summary["learned_nuisance_rule"][
                "preprocessing"
            ],
            "replications": args.replications,
            "workers": args.workers,
            "seed": args.seed,
            "diagnostic_only": True,
            "application_design_binding": design_binding.as_dict(),
        },
        inputs=[
            design_binding.design_path,
            design_binding.audit_path,
            Path(__file__),
            PROJECT_ROOT / "experiments/application_design_contract.py",
            PROJECT_ROOT / "experiments/build_return_design.py",
            PROJECT_ROOT / "experiments/run_canonical_simulation.py",
            PROJECT_ROOT / "experiments/run_continuous_dml.py",
            PROJECT_ROOT / "src/dyadic_dml.py",
            PROJECT_ROOT / "src/provenance.py",
            PROJECT_ROOT / "pyproject.toml",
        ],
        outputs=[per_path, summary_path],
        status="completed",
        manifest_path=args.manifest,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
