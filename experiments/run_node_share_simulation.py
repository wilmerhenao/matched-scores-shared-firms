#!/usr/bin/env python3
"""Paired Monte Carlo sensitivity over node-innovation variance shares.

This diagnostic keeps the canonical linear nuisance design fixed and changes
only the fraction of each treatment and outcome innovation attributable to
node terms.  Common seeds, node folds, controls, and primitive Gaussian draws
are reused across shares so cell differences have a paired interpretation.
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
from sklearn.linear_model import Ridge
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
    ar1_gaussian,
    nuisance_coefficients,
    oracle_factories,
)
from src.dyadic_dml import fit_plr_dml, make_node_folds
from src.provenance import record


N_FOLDS = 5
RIDGE_ALPHA = 1.0
RIDGE_SOLVER = "lsqr"
RIDGE_SOLVER_TOLERANCE = 1e-4
DEFAULT_REPLICATIONS = 1000
DEFAULT_SEED = 20260808
DEFAULT_NODE_SHARES = (0.10, 0.25, 0.50, 2.0 / 3.0, 1.00)
DEFAULT_NODE_SHARE_OUTPUT_DIR = (
    PROJECT_ROOT / "data_public/processed/node_share_simulation"
)


def positive_int(value: str) -> int:
    """Parse a strictly positive command-line integer."""

    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def validate_node_shares(values: list[float] | tuple[float, ...]) -> tuple[float, ...]:
    """Validate and order the nondegenerate node-share grid."""

    shares = tuple(float(value) for value in values)
    if not shares:
        raise ValueError("at least one node-innovation share is required")
    if not np.all(np.isfinite(shares)):
        raise ValueError("node-innovation shares must be finite")
    if any(value <= 0.0 or value > 1.0 for value in shares):
        raise ValueError("node-innovation shares must lie in (0, 1]")
    if len(set(shares)) != len(shares):
        raise ValueError("node-innovation shares must be unique")
    return tuple(sorted(shares))


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line interface for the dependence-grid study."""

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
    parser.add_argument(
        "--node-shares",
        type=float,
        nargs="+",
        default=list(DEFAULT_NODE_SHARES),
        help=(
            "marginal share of each innovation's variance contributed jointly "
            "by its two endpoint terms"
        ),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_NODE_SHARE_OUTPUT_DIR,
    )
    return parser


def simulation_tasks(
    *,
    n_nodes: int,
    n_controls: int,
    replications: int,
    node_shares: tuple[float, ...],
    seed: int,
) -> list[tuple[int, int, int, int, float]]:
    """Construct paired tasks from dimensions bound to the application design."""

    if n_nodes < 2 * N_FOLDS:
        raise ValueError("simulation requires at least two nodes per score fold")
    if n_controls <= 0 or replications <= 0:
        raise ValueError("simulation controls and replications must be positive")
    if seed < 0:
        raise ValueError("simulation seed must be nonnegative")
    node_shares = validate_node_shares(node_shares)
    paired_seeds = [
        int(child.generate_state(1)[0])
        for child in np.random.SeedSequence(seed).spawn(replications)
    ]
    if len(paired_seeds) != len(set(paired_seeds)):
        raise RuntimeError("spawned paired simulation seeds are not unique")
    return [
        (n_nodes, n_controls, repetition, paired_seeds[repetition], node_share)
        for node_share in node_shares
        for repetition in range(replications)
    ]


def generate_node_share_dgp(
    n_nodes: int,
    n_controls: int,
    seed: int,
    node_share: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Canonical PLR DGP with a configurable innovation node share.

    For either innovation, ``lambda=node_share`` parameterizes

    ``sqrt(lambda/2) * (a_i+a_j) + sqrt(1-lambda) * a_ij``.

    Thus ``lambda`` is the marginal endpoint share in each innovation, the
    innovation has unit marginal variance, and incident edges have innovation
    covariance ``lambda/2``.  Treatment and outcome innovations use independent
    primitive draws.  ``lambda=1`` is the node-only boundary (and maximal only
    within this additive innovation family); ``lambda=0`` is excluded because
    it removes the nondegenerate first-order node component of the oracle score.
    """

    share = validate_node_shares([node_share])[0]
    rng = np.random.default_rng(seed)
    row, col = np.triu_indices(n_nodes, k=1)
    node_x = ar1_gaussian(rng, n_nodes, n_controls)
    pair_x = ar1_gaussian(rng, row.size, n_controls)
    x = (node_x[row] + node_x[col] + pair_x) / np.sqrt(3.0)

    beta, gamma = nuisance_coefficients(n_controls)
    treatment_node = rng.normal(size=n_nodes)
    treatment_pair = rng.normal(size=row.size)
    outcome_node = rng.normal(size=n_nodes)
    outcome_pair = rng.normal(size=row.size)
    node_scale = np.sqrt(share / 2.0)
    pair_scale = np.sqrt(1.0 - share)
    v = node_scale * (treatment_node[row] + treatment_node[col])
    v += pair_scale * treatment_pair
    epsilon = node_scale * (outcome_node[row] + outcome_node[col])
    epsilon += pair_scale * outcome_pair
    d = x @ beta + v
    y = THETA_ZERO * d + x @ gamma + epsilon
    return y, d, x, row.astype(np.int64), col.astype(np.int64)


def oracle_score_covariance(node_share: float) -> float:
    """Incident-edge covariance of the unit-variance oracle PLR score."""

    share = validate_node_shares([node_share])[0]
    return float(share**2 / 4.0)


def analytical_oracle_moment_iid_sd_ratio(
    n_nodes: int, n_folds: int, node_share: float
) -> float:
    """IID-formula SD divided by the exact oracle-moment SD at ``theta_zero``.

    This uses the balanced fold sizes produced by ``numpy.array_split`` and
    the equal-fold truth-score moment.  It is an analytical mechanism check,
    not the finite-sample SD of the ratio estimator and not a correction for
    learned nuisances.
    """

    if n_folds < 2 or n_nodes < 2 * n_folds:
        raise ValueError("n_folds must be at least two with two nodes per fold")
    covariance = oracle_score_covariance(node_share)
    fold_sizes = [part.size for part in np.array_split(np.arange(n_nodes), n_folds)]
    iid_variance = 0.0
    true_variance = 0.0
    for size in fold_sizes:
        edge_count = size * (size - 1) // 2
        iid_component = 1.0 / edge_count
        incident_pair_count = size * (size - 1) * (size - 2) / 2.0
        dependence_component = (
            2.0 * incident_pair_count * covariance / edge_count**2
        )
        iid_variance += iid_component / n_folds**2
        true_variance += (iid_component + dependence_component) / n_folds**2
    return float(np.sqrt(iid_variance / true_variance))


def one_replication(
    payload: tuple[int, int, int, int, float]
) -> dict[str, float | int | str]:
    """Fit strict and oracle estimators for one paired node-share draw."""

    n_nodes, n_controls, repetition, seed, node_share = payload
    if n_nodes < 2 * N_FOLDS or n_controls <= 0:
        raise ValueError("replication dimensions do not support nonempty score folds")
    if repetition < 0 or seed < 0:
        raise ValueError("repetition and paired seed must be nonnegative")
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        y, d, x, row, col = generate_node_share_dgp(
            n_nodes, n_controls, seed, node_share
        )
        node_folds = make_node_folds(n_nodes, N_FOLDS, seed=seed + 101)
        factory = lambda: Ridge(
            alpha=RIDGE_ALPHA,
            solver=RIDGE_SOLVER,
            tol=RIDGE_SOLVER_TOLERANCE,
        )
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
        oracle_outcome_factory, oracle_treatment_factory = oracle_factories(
            n_controls
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
        for name, fit in (("strict", strict), ("oracle", oracle)):
            numerical = np.asarray(
                [fit.theta, fit.se_dyadic, fit.se_iid, fit.jacobian, fit.moment],
                dtype=float,
            )
            if (
                not np.all(np.isfinite(numerical))
                or fit.se_dyadic <= 0.0
                or fit.se_iid <= 0.0
            ):
                raise RuntimeError(
                    f"{name} node-share fit produced invalid numerical output"
                )
    return {
        "n_nodes": n_nodes,
        "n_controls": n_controls,
        "n_folds": N_FOLDS,
        "node_innovation_share": node_share,
        "oracle_score_incident_covariance": oracle_score_covariance(node_share),
        "repetition": repetition,
        "paired_seed": seed,
        "strict_theta": strict.theta,
        "strict_se_dyadic": strict.se_dyadic,
        "strict_se_iid": strict.se_iid,
        "strict_n_scored": strict.n_scored,
        "strict_moment": strict.moment,
        "oracle_theta": oracle.theta,
        "oracle_se_dyadic": oracle.se_dyadic,
        "oracle_se_iid": oracle.se_iid,
        "oracle_n_scored": oracle.n_scored,
        "oracle_moment": oracle.moment,
        "runtime_seconds": time.perf_counter() - started,
        "status": "completed",
    }


def summarize_share(frame: pl.DataFrame) -> dict[str, float | int]:
    """Summarize calibration for one node-driven variance share."""

    if frame.height < 2:
        raise ValueError("at least two replications are required for a summary")
    invariant = ["n_nodes", "n_controls", "n_folds", "node_innovation_share"]
    values: dict[str, float | int] = {}
    for column in invariant:
        unique = frame.get_column(column).unique().to_list()
        if len(unique) != 1:
            raise ValueError(f"simulation cell varies in {column}")
        values[column] = unique[0]
    for column in ("repetition", "paired_seed"):
        if frame.get_column(column).n_unique() != frame.height:
            raise ValueError(f"node-share cell contains duplicate {column}")
    node_share = float(values["node_innovation_share"])
    result: dict[str, float | int] = {
        **values,
        "replications": frame.height,
        "runtime_seconds_total": float(frame.get_column("runtime_seconds").sum()),
        "runtime_seconds_median": float(frame.get_column("runtime_seconds").median()),
        "oracle_score_incident_covariance": oracle_score_covariance(node_share),
        "analytical_oracle_moment_iid_sd_to_exact_sd_ratio": (
            analytical_oracle_moment_iid_sd_ratio(
                int(values["n_nodes"]), int(values["n_folds"]), node_share
            )
        ),
    }
    z90 = float(norm.ppf(0.95))
    z95 = float(norm.ppf(0.975))
    for split in ("strict", "oracle"):
        theta = frame.get_column(f"{split}_theta").to_numpy()
        if not np.all(np.isfinite(theta)):
            raise ValueError(f"{split} node-share estimates are not finite")
        empirical_sd = float(np.std(theta, ddof=1))
        if not np.isfinite(empirical_sd) or empirical_sd <= 0.0:
            raise ValueError(f"{split} empirical sampling SD is not positive")
        result[f"{split}_mean"] = float(np.mean(theta))
        result[f"{split}_bias"] = float(np.mean(theta) - THETA_ZERO)
        result[f"{split}_mean_mcse"] = float(empirical_sd / np.sqrt(frame.height))
        result[f"{split}_bias_mcse"] = float(empirical_sd / np.sqrt(frame.height))
        result[f"{split}_empirical_sd"] = empirical_sd
        scored_counts = frame.get_column(f"{split}_n_scored").unique().to_list()
        if len(scored_counts) != 1 or int(scored_counts[0]) <= 0:
            raise ValueError(f"{split} scored-pair count varies or is empty")
        result[f"{split}_n_scored"] = int(scored_counts[0])
        for variance in ("dyadic", "iid"):
            standard_error = frame.get_column(f"{split}_se_{variance}").to_numpy()
            if (
                not np.all(np.isfinite(standard_error))
                or np.any(standard_error <= 0.0)
            ):
                raise ValueError(f"{split}/{variance} standard errors are invalid")
            standardized = np.abs(theta - THETA_ZERO) / standard_error
            for level, critical in ((90, z90), (95, z95)):
                covered = standardized <= critical
                coverage = float(np.mean(covered))
                result[f"{split}_{variance}_coverage{level}"] = coverage
                result[f"{split}_{variance}_coverage{level}_successes"] = int(
                    np.sum(covered)
                )
                result[f"{split}_{variance}_coverage{level}_mcse"] = float(
                    np.sqrt(coverage * (1.0 - coverage) / frame.height)
                )
            result[f"{split}_{variance}_rejection5"] = float(
                1.0 - result[f"{split}_{variance}_coverage95"]
            )
            result[f"{split}_{variance}_mean_se"] = float(
                np.mean(standard_error)
            )
            result[f"{split}_{variance}_se_over_sd"] = float(
                np.mean(standard_error) / empirical_sd
            )
            result[f"{split}_{variance}_abs_t_quantile90"] = float(
                np.quantile(standardized, 0.90)
            )
            result[f"{split}_{variance}_abs_t_quantile95"] = float(
                np.quantile(standardized, 0.95)
            )
    return result


def main() -> None:
    """Run the application-sized node-dependence calibration grid."""

    parser = build_parser()
    args = parser.parse_args()
    if args.replications < 2:
        parser.error("--replications must be at least two")
    node_shares = validate_node_shares(args.node_shares)
    design_binding = bind_application_design(
        args.design, args.manifest, audit_path=args.design_audit
    )
    validate_driver_design_role(
        design_binding,
        allow_sensitivity_design=args.allow_sensitivity_design,
        output_dir=args.output_dir,
        primary_output_dir=DEFAULT_NODE_SHARE_OUTPUT_DIR,
        driver_name="node-share simulation",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tasks = simulation_tasks(
        n_nodes=design_binding.n_nodes,
        n_controls=design_binding.n_controls,
        replications=args.replications,
        node_shares=node_shares,
        seed=args.seed,
    )
    rows: list[dict[str, float | int | str]] = []
    if args.workers == 1:
        for completed, task in enumerate(tasks, start=1):
            rows.append(one_replication(task))
            if completed % 20 == 0 or completed == len(tasks):
                print(f"completed {completed}/{len(tasks)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(one_replication, task) for task in tasks]
            for completed, future in enumerate(as_completed(futures), start=1):
                rows.append(future.result())
                if completed % 20 == 0 or completed == len(tasks):
                    print(f"completed {completed}/{len(tasks)}", flush=True)

    per_replication = pl.DataFrame(rows).with_columns(
        pl.lit(design_binding.design_spec).alias("application_design_spec"),
        pl.lit(design_binding.design_role).alias("application_design_role"),
        pl.lit(design_binding.is_primary_default).alias(
            "application_design_is_primary_default"
        ),
    ).sort(
        ["node_innovation_share", "repetition"]
    )
    if per_replication.height != args.replications * len(node_shares):
        raise RuntimeError("node-share simulation returned the wrong row count")
    reference_pairs: list[tuple[int, int]] | None = None
    for node_share in node_shares:
        share_pairs = (
            per_replication.filter(pl.col("node_innovation_share") == node_share)
            .select("repetition", "paired_seed")
            .sort("repetition")
            .iter_rows()
        )
        current_pairs = [(int(rep), int(seed)) for rep, seed in share_pairs]
        if len(current_pairs) != args.replications:
            raise RuntimeError("node-share simulation cell has the wrong row count")
        if reference_pairs is None:
            reference_pairs = current_pairs
        elif current_pairs != reference_pairs:
            raise RuntimeError("node-share cells do not share the same paired seeds")
    if (
        bind_application_design(
            args.design, args.manifest, audit_path=args.design_audit
        )
        != design_binding
    ):
        raise RuntimeError("application design binding changed during simulation")
    cells = [
        summarize_share(
            per_replication.filter(pl.col("node_innovation_share") == node_share)
        )
        for node_share in node_shares
    ]
    per_path = args.output_dir / "per_replication.parquet"
    summary_path = args.output_dir / "summary.json"
    per_replication.write_parquet(per_path)
    summary = {
        "schema_version": 3,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "diagnostic_only": True,
        "theta_zero": THETA_ZERO,
        "n_nodes": design_binding.n_nodes,
        "n_controls": design_binding.n_controls,
        "n_folds": N_FOLDS,
        "ridge_alpha": RIDGE_ALPHA,
        "ridge_solver": RIDGE_SOLVER,
        "ridge_solver_tolerance": RIDGE_SOLVER_TOLERANCE,
        "paired_across_node_shares": True,
        "node_share_parameterization": (
            "sqrt(lambda/2)*(a_i+a_j)+sqrt(1-lambda)*a_ij, separately "
            "for independent treatment and outcome innovations"
        ),
        "node_share_interpretation": (
            "lambda is the marginal endpoint share in each innovation; lambda=1 "
            "is the node-only boundary and is maximal only within this additive "
            "innovation family; lambda=0 is excluded because its oracle score has "
            "a degenerate first-order node projection"
        ),
        "analytical_ratio_interpretation": (
            "IID-formula SD divided by exact SD of the equal-fold oracle truth-score "
            "moment at theta_zero; not the finite-sample SD ratio of theta_hat"
        ),
        "controls_dgp": "unchanged canonical AR(1) Gaussian node-plus-pair design",
        "application_design_binding": design_binding.as_dict(),
        "cells": cells,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    run_id = datetime.now(timezone.utc).strftime(
        f"node-share-sim-{design_spec_run_token(design_binding)}-%Y%m%dT%H%M%SZ"
    )
    record(
        run_id=run_id,
        command=[sys.executable, *sys.argv],
        config={
            "n_nodes": design_binding.n_nodes,
            "n_controls": design_binding.n_controls,
            "n_folds": N_FOLDS,
            "ridge_alpha": RIDGE_ALPHA,
            "ridge_solver": RIDGE_SOLVER,
            "ridge_solver_tolerance": RIDGE_SOLVER_TOLERANCE,
            "replications": args.replications,
            "workers": args.workers,
            "seed": args.seed,
            "node_innovation_shares": list(node_shares),
            "paired_across_node_shares": True,
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
