#!/usr/bin/env python3
"""Canonical PLR Monte Carlo using the audited strict and edge diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl
from scipy.stats import norm
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dyadic_dml import (
    fit_edge_split_plr_diagnostic,
    fit_matched_score_equal_size_maximal_leak_plr_diagnostic,
    fit_matched_score_leaky_complement_plr_diagnostic,
    fit_plr_dml,
    make_node_folds,
)
from src.provenance import record


THETA_ZERO = 0.5
CANONICAL_OUTPUT_SCHEMA_VERSION = 2
EQUAL_SIZE_LEAK_SAMPLING_SALT = 20260809
REJECTION5_DEFINITION = (
    "false rejection probability for the two-sided H0: theta = theta_zero "
    "at alpha = 0.05"
)
SIMULATION_ARMS = (
    "oracle",
    "strict",
    "matched_leaky",
    "equal_size_leaky",
    "edge",
)
PILOT_CELLS = ((100, 20, 30), (250, 50, 30))
DEFAULT_NONPILOT_CELLS = (
    (100, 20, 200),
    (250, 50, 200),
    (500, 50, 150),
    (500, 100, 100),
    (1000, 50, 75),
)


def positive_int(value: str) -> int:
    """Parse a strictly positive integer for replication-count overrides."""

    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line interface for the canonical simulation."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument(
        "--replications-per-cell",
        type=positive_int,
        default=None,
        help=(
            "override every nonpilot cell's replication count; "
            "the legacy cell-specific defaults are used when omitted"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data_public/processed/canonical_simulation",
    )
    return parser


def simulation_cells(
    *, pilot: bool, replications_per_cell: int | None
) -> list[tuple[int, int, int]]:
    """Resolve simulation cells without changing pilot or legacy defaults."""

    if pilot:
        return list(PILOT_CELLS)
    if replications_per_cell is None:
        return list(DEFAULT_NONPILOT_CELLS)
    if replications_per_cell <= 0:
        raise ValueError("replications_per_cell must be positive")
    return [
        (n_nodes, n_controls, replications_per_cell)
        for n_nodes, n_controls, _ in DEFAULT_NONPILOT_CELLS
    ]


class KnownLinearRegressor(RegressorMixin, BaseEstimator):
    """Scikit-learn-compatible oracle with fixed, known coefficients."""

    def __init__(self, coefficients: tuple[float, ...]):
        self.coefficients = coefficients

    def fit(self, x: np.ndarray, y: np.ndarray) -> "KnownLinearRegressor":
        """Validate the feature dimension and retain the fixed coefficients."""

        del y
        if x.shape[1] != len(self.coefficients):
            raise ValueError("oracle coefficient and feature dimensions disagree")
        self.n_features_in_ = x.shape[1]
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the known linear nuisance function at supplied rows."""

        if x.shape[1] != len(self.coefficients):
            raise ValueError("oracle coefficient and feature dimensions disagree")
        return x @ np.asarray(self.coefficients, dtype=np.float64)


def nuisance_coefficients(n_controls: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact treatment and structural-outcome DGP coefficients."""

    sparsity = min(8, n_controls)
    beta = np.zeros(n_controls)
    gamma = np.zeros(n_controls)
    beta[:sparsity] = 0.45 * (-0.65) ** np.arange(sparsity)
    gamma[:sparsity] = 0.35 * (0.60) ** np.arange(sparsity)
    return beta, gamma


def oracle_factories(
    n_controls: int,
) -> tuple[
    Callable[[], KnownLinearRegressor], Callable[[], KnownLinearRegressor]
]:
    """Return known E[Y|X] and E[D|X] estimator factories for this DGP."""

    beta, gamma = nuisance_coefficients(n_controls)
    outcome = tuple((THETA_ZERO * beta + gamma).tolist())
    treatment = tuple(beta.tolist())
    return (
        lambda: KnownLinearRegressor(outcome),
        lambda: KnownLinearRegressor(treatment),
    )


def ar1_gaussian(
    rng: np.random.Generator, n_rows: int, n_columns: int, rho: float = 0.35
) -> np.ndarray:
    """Generate Gaussian features with AR(1) column covariance."""

    innovation = rng.normal(size=(n_rows, n_columns))
    values = np.empty_like(innovation)
    values[:, 0] = innovation[:, 0]
    scale = np.sqrt(1.0 - rho**2)
    for column in range(1, n_columns):
        values[:, column] = rho * values[:, column - 1] + scale * innovation[:, column]
    return values


def generate_plr_dgp(
    n_nodes: int, n_controls: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Jointly exchangeable undirected PLR with node-sharing score dependence."""

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
    v = (treatment_node[row] + treatment_node[col] + treatment_pair) / np.sqrt(3.0)
    epsilon = (outcome_node[row] + outcome_node[col] + outcome_pair) / np.sqrt(3.0)
    d = x @ beta + v
    y = THETA_ZERO * d + x @ gamma + epsilon
    return y, d, x, row.astype(np.int64), col.astype(np.int64)


def one_replication(payload: tuple[int, int, int, int]) -> dict[str, float | int | str]:
    """Run all matched-score arms for one seeded canonical graph."""

    n_nodes, n_controls, repetition, seed = payload
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        y, d, x, row, col = generate_plr_dgp(n_nodes, n_controls, seed)
        factory = lambda: Ridge(alpha=1.0, solver="lsqr")
        node_folds = make_node_folds(n_nodes, 5, seed=seed + 101)
        strict = fit_plr_dml(
            y,
            d,
            x,
            row,
            col,
            n_nodes,
            directed=False,
            n_folds=5,
            seed=seed + 101,
            fold_assignment=node_folds,
            outcome_factory=factory,
            treatment_factory=factory,
        )
        oracle_outcome_factory, oracle_treatment_factory = oracle_factories(n_controls)
        oracle = fit_plr_dml(
            y,
            d,
            x,
            row,
            col,
            n_nodes,
            directed=False,
            n_folds=5,
            seed=seed + 101,
            fold_assignment=node_folds,
            outcome_factory=oracle_outcome_factory,
            treatment_factory=oracle_treatment_factory,
        )
        matched_leaky = fit_matched_score_leaky_complement_plr_diagnostic(
            y,
            d,
            x,
            row,
            col,
            n_nodes,
            directed=False,
            n_folds=5,
            seed=seed + 101,
            fold_assignment=node_folds,
            outcome_factory=factory,
            treatment_factory=factory,
        )
        equal_size_leaky = (
            fit_matched_score_equal_size_maximal_leak_plr_diagnostic(
                y,
                d,
                x,
                row,
                col,
                n_nodes,
                directed=False,
                n_folds=5,
                seed=seed + 101,
                sampling_salt=EQUAL_SIZE_LEAK_SAMPLING_SALT,
                fold_assignment=node_folds,
                outcome_factory=factory,
                treatment_factory=factory,
            )
        )
        for comparison in (oracle, matched_leaky, equal_size_leaky):
            if (
                comparison.diagnostics["score_edge_fingerprints"]
                != strict.diagnostics["score_edge_fingerprints"]
            ):
                raise AssertionError("canonical matched-score arms score different edges")
        if [item.n_train_edges for item in equal_size_leaky.fold_audits] != [
            item.n_train_edges for item in strict.fold_audits
        ]:
            raise AssertionError(
                "equal-size leakage arm does not match strict training counts"
            )
        edge = fit_edge_split_plr_diagnostic(
            y,
            d,
            x,
            row,
            col,
            n_nodes,
            n_folds=5,
            seed=seed + 202,
            outcome_factory=factory,
            treatment_factory=factory,
        )
    return {
        "n_nodes": n_nodes,
        "n_controls": n_controls,
        "repetition": repetition,
        "seed": seed,
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
        "matched_leaky_theta": matched_leaky.theta,
        "matched_leaky_se_dyadic": matched_leaky.se_dyadic,
        "matched_leaky_se_iid": matched_leaky.se_iid,
        "matched_leaky_n_scored": matched_leaky.n_scored,
        "matched_leaky_moment": matched_leaky.moment,
        "equal_size_leaky_theta": equal_size_leaky.theta,
        "equal_size_leaky_se_dyadic": equal_size_leaky.se_dyadic,
        "equal_size_leaky_se_iid": equal_size_leaky.se_iid,
        "equal_size_leaky_n_scored": equal_size_leaky.n_scored,
        "equal_size_leaky_moment": equal_size_leaky.moment,
        "edge_theta": edge.theta,
        "edge_se_dyadic": edge.se_dyadic,
        "edge_se_iid": edge.se_iid,
        "edge_n_scored": edge.n_scored,
        "edge_moment": edge.moment,
        "runtime_seconds": time.perf_counter() - started,
        "status": "completed",
    }


def summarize_cell(frame: pl.DataFrame) -> dict[str, float | int]:
    """Summarize bias, dispersion, coverage, and rejection for one cell."""

    z90 = float(norm.ppf(0.95))
    z95 = float(norm.ppf(0.975))
    result: dict[str, float | int] = {
        "n_nodes": int(frame.get_column("n_nodes")[0]),
        "n_controls": int(frame.get_column("n_controls")[0]),
        "replications": frame.height,
        "runtime_seconds_total": float(frame.get_column("runtime_seconds").sum()),
        "runtime_seconds_median": float(frame.get_column("runtime_seconds").median()),
    }
    for split in SIMULATION_ARMS:
        theta = frame.get_column(f"{split}_theta").to_numpy()
        empirical_sd = float(np.std(theta, ddof=1))
        result[f"{split}_mean"] = float(np.mean(theta))
        result[f"{split}_bias"] = float(np.mean(theta) - THETA_ZERO)
        result[f"{split}_empirical_sd"] = empirical_sd
        result[f"{split}_n_scored"] = int(frame.get_column(f"{split}_n_scored")[0])
        for variance in ["dyadic", "iid"]:
            standard_error = frame.get_column(f"{split}_se_{variance}").to_numpy()
            covered90 = np.abs(theta - THETA_ZERO) <= z90 * standard_error
            covered95 = np.abs(theta - THETA_ZERO) <= z95 * standard_error
            coverage90 = float(np.mean(covered90))
            coverage95 = float(np.mean(covered95))
            false_rejection5 = 1.0 - coverage95
            result[f"{split}_{variance}_mean_se"] = float(np.mean(standard_error))
            result[f"{split}_{variance}_se_over_sd"] = float(
                np.mean(standard_error) / empirical_sd
            )
            result[f"{split}_{variance}_coverage90"] = coverage90
            result[f"{split}_{variance}_coverage95"] = coverage95
            # The DGP sets theta to THETA_ZERO, so this is false rejection of
            # H0: theta = theta_zero, not rejection under a zero-effect null.
            result[f"{split}_{variance}_rejection5"] = false_rejection5
            result[f"{split}_{variance}_coverage90_mcse"] = float(
                np.sqrt(coverage90 * (1.0 - coverage90) / frame.height)
            )
            result[f"{split}_{variance}_coverage95_mcse"] = float(
                np.sqrt(coverage95 * (1.0 - coverage95) / frame.height)
            )
            result[f"{split}_{variance}_rejection5_mcse"] = float(
                np.sqrt(
                    false_rejection5
                    * (1.0 - false_rejection5)
                    / frame.height
                )
            )
    return result


def main() -> None:
    """Execute the canonical simulation grid and write audited outputs."""

    parser = build_parser()
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cells = simulation_cells(
        pilot=args.pilot,
        replications_per_cell=args.replications_per_cell,
    )
    seed_sequence = np.random.SeedSequence(20260808)
    tasks: list[tuple[int, int, int, int]] = []
    child_seeds = seed_sequence.spawn(sum(cell[2] for cell in cells))
    offset = 0
    for n_nodes, n_controls, repetitions in cells:
        for repetition in range(repetitions):
            seed = int(child_seeds[offset].generate_state(1)[0])
            tasks.append((n_nodes, n_controls, repetition, seed))
            offset += 1

    rows: list[dict[str, float | int | str]] = []
    if args.workers == 1:
        for completed, task in enumerate(tasks, start=1):
            rows.append(one_replication(task))
            if completed % 20 == 0 or completed == len(tasks):
                print(f"completed {completed}/{len(tasks)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(one_replication, task): task for task in tasks}
            completed = 0
            for future in as_completed(futures):
                rows.append(future.result())
                completed += 1
                if completed % 20 == 0 or completed == len(tasks):
                    print(f"completed {completed}/{len(tasks)}", flush=True)

    per_replication = pl.DataFrame(rows).sort(
        ["n_nodes", "n_controls", "repetition"]
    )
    summaries = []
    for n_nodes, n_controls, _ in cells:
        summaries.append(
            summarize_cell(
                per_replication.filter(
                    (pl.col("n_nodes") == n_nodes)
                    & (pl.col("n_controls") == n_controls)
                )
            )
        )
    per_path = args.output_dir / ("pilot_per_replication.parquet" if args.pilot else "per_replication.parquet")
    summary_path = args.output_dir / ("pilot_summary.json" if args.pilot else "summary.json")
    per_replication.write_parquet(per_path)
    summary = {
        "schema_version": CANONICAL_OUTPUT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "theta_zero": THETA_ZERO,
        "rejection5_definition": REJECTION5_DEFINITION,
        "dgp": "jointly exchangeable undirected partially linear model",
        "n_folds": 5,
        "ridge_alpha": 1.0,
        "equal_size_leak_sampling_salt": EQUAL_SIZE_LEAK_SAMPLING_SALT,
        "oracle_nuisance": "known E[Y|X] and E[D|X] under the unchanged DGP",
        "cells": summaries,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    run_id = datetime.now(timezone.utc).strftime(
        ("canonical-sim-pilot-%Y%m%dT%H%M%SZ" if args.pilot else "canonical-sim-%Y%m%dT%H%M%SZ")
    )
    record(
        run_id=run_id,
        command=[sys.executable, *sys.argv],
        config={
            "cells": cells,
            "workers": args.workers,
            "theta_zero": THETA_ZERO,
            "replications_per_cell_override": args.replications_per_cell,
        },
        inputs=[Path(__file__), PROJECT_ROOT / "src/dyadic_dml.py"],
        outputs=[per_path, summary_path],
        status="completed",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
