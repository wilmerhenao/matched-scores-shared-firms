#!/usr/bin/env python3
"""Manifest-bound ridge audit of equal-count endpoint exposure.

This diagnostic reuses the frozen application controls, ridge pipeline, node
partitions, and reporting unit.  It compares identical score edges under:

* strict outside--outside nuisance training;
* the full non-score complement; and
* three equal-row-count, maximally endpoint-exposed training samples.

Only the strict arm satisfies the maintained node-disjoint cross-fitting rule.
Inference-shaped fields are therefore null for both exposed arms.  Their point
estimates and observed score-residual MSEs are descriptive diagnostics only.
"""

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
from threadpoolctl import threadpool_limits

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.application_design_contract import (  # noqa: E402
    DEFAULT_MANIFEST,
    bind_application_design,
    design_spec_run_token,
    validate_driver_design_role,
)
from experiments.run_continuous_dml import (  # noqa: E402
    PRIMARY_ALPHA,
    PRIMARY_N_FOLDS,
    PRIMARY_OUTCOME,
    PRIMARY_REPORTING_INCREMENT,
    PRIMARY_SEEDS,
    PRIMARY_TREATMENT,
    RIDGE_SOLVER_TOLERANCE,
    design_matrix,
    ridge_factory,
)
from src.dyadic_dml import (  # noqa: E402
    aggregate_repeated_splits,
    fit_matched_score_equal_size_maximal_leak_plr_diagnostic,
    fit_matched_score_leaky_complement_plr_diagnostic,
    fit_plr_dml,
    make_node_folds,
)
from src.provenance import record  # noqa: E402


DEFAULT_SWAP_SALTS = (20260809, 20260810, 20260811)
DEFAULT_DESIGN = (
    PROJECT_ROOT / "data_public/processed/continuous_design/dyadic_design.parquet"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "data_public/processed/ridge_equal_count_audit"
)
SOURCE_DEPENDENCIES = (
    Path(__file__).resolve(),
    PROJECT_ROOT / "experiments/application_design_contract.py",
    PROJECT_ROOT / "experiments/run_continuous_dml.py",
    PROJECT_ROOT / "src/dyadic_dml.py",
    PROJECT_ROOT / "src/provenance.py",
    PROJECT_ROOT / "pyproject.toml",
)
LearnerFactory = Callable[[], object]


def application_ridge_factory() -> object:
    """Return the exact fold-local ridge pipeline from the primary driver."""

    return ridge_factory(PRIMARY_ALPHA)


def load_design(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Load the canonical complete, unique, undirected application graph."""

    frame = pl.read_parquet(path).sort(["NODE_I", "NODE_J"])
    row = frame.get_column("NODE_I").to_numpy().astype(np.int64, copy=False)
    col = frame.get_column("NODE_J").to_numpy().astype(np.int64, copy=False)
    if row.size == 0 or np.any(row < 0) or np.any(row >= col):
        raise ValueError("design must contain sorted unique edges with NODE_I < NODE_J")
    n_nodes = int(max(row.max(), col.max()) + 1)
    expected_edges = n_nodes * (n_nodes - 1) // 2
    if frame.height != expected_edges:
        raise ValueError("ridge equal-count audit requires a complete graph")
    if frame.select("NODE_I", "NODE_J").unique().height != expected_edges:
        raise ValueError("ridge equal-count audit design contains duplicate edges")
    x, feature_names = design_matrix(frame)
    y = frame.get_column(PRIMARY_OUTCOME).to_numpy()
    d = frame.get_column(PRIMARY_TREATMENT).to_numpy()
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(d)):
        raise ValueError("ridge equal-count outcome or treatment is non-finite")
    return y, d, x, row, col, n_nodes, len(feature_names)


def _fold_values(result, method: str) -> tuple[list[int], list[int], list[int]]:
    score_counts = [int(item.n_score_edges) for item in result.fold_audits]
    train_counts = [int(item.n_train_edges) for item in result.fold_audits]
    if method == "strict":
        cross_counts = [0] * len(train_counts)
    elif method == "full_leaky":
        cross_counts = [
            int(item["leaking_train_edges"])
            for item in result.diagnostics["leakage_by_fold"]
        ]
    elif method == "equal_size_leaky":
        cross_counts = [
            int(item["selected_cross_boundary_edges"])
            for item in result.diagnostics["leakage_by_fold"]
        ]
    else:  # pragma: no cover - internal method labels are fixed
        raise ValueError(f"unknown method: {method}")
    return score_counts, train_counts, cross_counts


def result_row(result, *, method: str, seed: int, swap_salt: int | None) -> dict:
    """Convert one engine result while suppressing exposed-arm inference."""

    valid_crossfit = method == "strict"
    diagnostics = result.diagnostics
    if bool(diagnostics["valid_dyadic_crossfit"]) != valid_crossfit:
        raise AssertionError("engine validity flag disagrees with the audit arm")
    score_counts, train_counts, cross_counts = _fold_values(result, method)
    tau = float(PRIMARY_REPORTING_INCREMENT * result.theta)
    standard_error = float(PRIMARY_REPORTING_INCREMENT * result.se_dyadic)
    statistic = tau / standard_error
    numerical = np.asarray(
        [
            result.theta,
            tau,
            result.jacobian,
            result.moment,
            diagnostics["score_outcome_residual_mse"],
            diagnostics["score_treatment_residual_mse"],
        ],
        dtype=float,
    )
    if (
        not np.all(np.isfinite(numerical))
        or result.jacobian <= 0.0
        or standard_error <= 0.0
    ):
        raise RuntimeError(f"{method} ridge audit fit has invalid numerical output")
    if method != "strict" and not diagnostics.get(
        "endpoint_leakage_observed", False
    ):
        raise AssertionError(f"{method} arm did not expose score-fold endpoints")

    score_fingerprints = list(diagnostics["score_edge_fingerprints"])
    train_fingerprints = diagnostics.get("train_edge_fingerprints")
    return {
        "method": method,
        "seed": int(seed),
        "swap_salt": swap_salt,
        "theta_estimation_scale": float(result.theta),
        "tau": tau,
        "standard_error_estimation_scale": (
            float(result.se_dyadic) if valid_crossfit else None
        ),
        "standard_error": standard_error if valid_crossfit else None,
        "t_statistic": statistic if valid_crossfit else None,
        "p_value_normal": (
            float(2.0 * norm.sf(abs(statistic))) if valid_crossfit else None
        ),
        "confidence_interval_low": (
            tau - 1.96 * standard_error if valid_crossfit else None
        ),
        "confidence_interval_high": (
            tau + 1.96 * standard_error if valid_crossfit else None
        ),
        "jacobian": float(result.jacobian),
        "moment": float(result.moment),
        "score_outcome_residual_mse": float(
            diagnostics["score_outcome_residual_mse"]
        ),
        "score_treatment_residual_mse": float(
            diagnostics["score_treatment_residual_mse"]
        ),
        "score_outcome_residual_sd": float(
            diagnostics["score_outcome_residual_sd"]
        ),
        "score_treatment_residual_sd": float(
            diagnostics["score_treatment_residual_sd"]
        ),
        "n_nodes": int(result.n_nodes),
        "n_edges": int(result.n_edges),
        "n_scored": int(result.n_scored),
        "n_folds": int(result.n_folds),
        "score_edge_counts_json": json.dumps(score_counts, separators=(",", ":")),
        "train_edge_counts_json": json.dumps(train_counts, separators=(",", ":")),
        "cross_boundary_train_edge_counts_json": json.dumps(
            cross_counts, separators=(",", ":")
        ),
        "score_edge_fingerprints_json": json.dumps(
            score_fingerprints, separators=(",", ":")
        ),
        "train_edge_fingerprints_json": (
            json.dumps(list(train_fingerprints), separators=(",", ":"))
            if train_fingerprints is not None
            else None
        ),
        "train_edges_min": min(train_counts),
        "train_edges_max": max(train_counts),
        "cross_boundary_train_fraction": float(
            sum(cross_counts) / sum(train_counts)
        ),
        "valid_dyadic_crossfit": valid_crossfit,
        "valid_under_maintained_assumptions": valid_crossfit,
        "inference_role": (
            "maintained_assumptions_diagnostic"
            if valid_crossfit
            else "unsupported_endpoint_exposure_diagnostic"
        ),
        "valid_primary_inference": False,
        "designated_primary_result": False,
        "diagnostic_only": True,
    }


def run_seed(
    y: np.ndarray,
    d: np.ndarray,
    x: np.ndarray,
    row: np.ndarray,
    col: np.ndarray,
    n_nodes: int,
    *,
    seed: int,
    swap_salts: tuple[int, ...] = DEFAULT_SWAP_SALTS,
    learner_factory: LearnerFactory = application_ridge_factory,
) -> list[dict]:
    """Run all matched-score ridge arms for one primary node partition."""

    if n_nodes < 2 * PRIMARY_N_FOLDS:
        raise ValueError("ridge equal-count audit needs two nodes per score fold")
    if seed < 0:
        raise ValueError("ridge equal-count seed must be nonnegative")
    if (
        not swap_salts
        or len(swap_salts) != len(set(swap_salts))
        or any(salt < 0 for salt in swap_salts)
    ):
        raise ValueError("swap salts must be nonempty, unique, and nonnegative")

    folds = make_node_folds(n_nodes, PRIMARY_N_FOLDS, seed=seed)
    common = {
        "n_folds": PRIMARY_N_FOLDS,
        "seed": seed,
        "fold_assignment": folds,
        "outcome_factory": learner_factory,
        "treatment_factory": learner_factory,
    }
    strict = fit_plr_dml(y, d, x, row, col, n_nodes, **common)
    full = fit_matched_score_leaky_complement_plr_diagnostic(
        y, d, x, row, col, n_nodes, **common
    )
    equal = [
        fit_matched_score_equal_size_maximal_leak_plr_diagnostic(
            y,
            d,
            x,
            row,
            col,
            n_nodes,
            sampling_salt=salt,
            **common,
        )
        for salt in swap_salts
    ]

    expected_scores = strict.diagnostics["score_edge_fingerprints"]
    expected_score_counts = [item.n_score_edges for item in strict.fold_audits]
    expected_train_counts = [item.n_train_edges for item in strict.fold_audits]
    for comparison in (full, *equal):
        if comparison.diagnostics["score_edge_fingerprints"] != expected_scores:
            raise AssertionError("ridge audit arms do not score identical edges")
        if [item.n_score_edges for item in comparison.fold_audits] != expected_score_counts:
            raise AssertionError("ridge audit arms have different score counts")
    if any(
        item.n_train_edges <= strict_item.n_train_edges
        for item, strict_item in zip(full.fold_audits, strict.fold_audits, strict=True)
    ):
        raise AssertionError("full complement does not exceed strict training counts")
    for comparison in equal:
        if [item.n_train_edges for item in comparison.fold_audits] != expected_train_counts:
            raise AssertionError("equal-count ridge arm does not match strict row counts")

    equal_fingerprints = [
        tuple(comparison.diagnostics["train_edge_fingerprints"])
        for comparison in equal
    ]
    if len(equal_fingerprints) != len(set(equal_fingerprints)):
        raise AssertionError("distinct salts produced duplicate equal-count samples")

    rows = [result_row(strict, method="strict", seed=seed, swap_salt=None)]
    rows.append(result_row(full, method="full_leaky", seed=seed, swap_salt=None))
    rows.extend(
        result_row(
            result,
            method="equal_size_leaky",
            seed=seed,
            swap_salt=salt,
        )
        for salt, result in zip(swap_salts, equal, strict=True)
    )
    return rows


def _run_seed_from_path(
    design_path: str, seed: int, swap_salts: tuple[int, ...]
) -> list[dict]:
    y, d, x, row, col, n_nodes, _ = load_design(Path(design_path))
    with threadpool_limits(limits=1):
        return run_seed(
            y, d, x, row, col, n_nodes, seed=seed, swap_salts=swap_salts
        )


def _summary_cell(rows: list[dict], seeds: tuple[int, ...]) -> dict:
    if tuple(sorted(int(item["seed"]) for item in rows)) != tuple(sorted(seeds)):
        raise AssertionError("summary cell does not contain the requested seeds")
    if len({(item["method"], item["swap_salt"]) for item in rows}) != 1:
        raise AssertionError("summary cell mixes methods or salts")
    valid = bool(rows[0]["valid_under_maintained_assumptions"])
    if any(bool(item["valid_under_maintained_assumptions"]) != valid for item in rows):
        raise AssertionError("summary cell mixes inference roles")
    tau = np.asarray([float(item["tau"]) for item in rows])
    theta = np.asarray([float(item["theta_estimation_scale"]) for item in rows])
    result = {
        "method": rows[0]["method"],
        "swap_salt": rows[0]["swap_salt"],
        "n_partitions": len(rows),
        "theta_median": float(np.median(theta)),
        "tau_median": float(np.median(tau)),
        "tau_mean": float(np.mean(tau)),
        "tau_minimum": float(np.min(tau)),
        "tau_maximum": float(np.max(tau)),
        "tau_partition_sd": float(np.std(tau, ddof=1) if tau.size > 1 else 0.0),
        "score_outcome_residual_mse_median": float(
            np.median([float(item["score_outcome_residual_mse"]) for item in rows])
        ),
        "score_treatment_residual_mse_median": float(
            np.median(
                [float(item["score_treatment_residual_mse"]) for item in rows]
            )
        ),
        "standard_error": None,
        "t_statistic": None,
        "p_value_normal": None,
        "valid_under_maintained_assumptions": valid,
        "inference_role": rows[0]["inference_role"],
        "valid_primary_inference": False,
        "designated_primary_result": False,
        "diagnostic_only": True,
    }
    if valid:
        se = np.asarray([float(item["standard_error"]) for item in rows])
        aggregate_tau, aggregate_se = aggregate_repeated_splits(tau, se)
        statistic = aggregate_tau / aggregate_se
        result.update(
            {
                "tau_median": aggregate_tau,
                "standard_error": aggregate_se,
                "t_statistic": statistic,
                "p_value_normal": float(2.0 * norm.sf(abs(statistic))),
            }
        )
    return result


def summarize(
    rows: list[dict], seeds: tuple[int, ...], swap_salts: tuple[int, ...]
) -> dict:
    """Return cell summaries and paired descriptive shifts from strict."""

    expected_keys = {
        (seed, method, salt)
        for seed in seeds
        for method, salt in (
            ("strict", None),
            ("full_leaky", None),
            *(("equal_size_leaky", value) for value in swap_salts),
        )
    }
    actual_keys = {
        (int(item["seed"]), str(item["method"]), item["swap_salt"])
        for item in rows
    }
    if actual_keys != expected_keys or len(rows) != len(expected_keys):
        raise AssertionError("ridge audit rows do not form the requested complete grid")

    cells = []
    for method, salt in (
        ("strict", None),
        ("full_leaky", None),
        *(("equal_size_leaky", value) for value in swap_salts),
    ):
        cells.append(
            _summary_cell(
                [
                    item
                    for item in rows
                    if item["method"] == method and item["swap_salt"] == salt
                ],
                seeds,
            )
        )

    strict = {
        int(item["seed"]): item for item in rows if item["method"] == "strict"
    }
    paired = []
    for method, salt in (
        ("full_leaky", None),
        *(("equal_size_leaky", value) for value in swap_salts),
    ):
        comparison = [
            item
            for item in rows
            if item["method"] == method and item["swap_salt"] == salt
        ]
        cell = {
            "method": method,
            "swap_salt": salt,
            "n_paired_partitions": len(comparison),
            "uncertainty_interpretation": (
                "descriptive variation over fixed algorithmic partitions; not a "
                "confidence interval or independent sampling uncertainty"
            ),
        }
        for output, column in (
            ("tau", "tau"),
            ("score_outcome_residual_mse", "score_outcome_residual_mse"),
            ("score_treatment_residual_mse", "score_treatment_residual_mse"),
        ):
            differences = np.asarray(
                [
                    float(item[column])
                    - float(strict[int(item["seed"])][column])
                    for item in comparison
                ]
            )
            cell.update(
                {
                    f"mean_{output}_difference": float(np.mean(differences)),
                    f"median_{output}_difference": float(np.median(differences)),
                    f"minimum_{output}_difference": float(np.min(differences)),
                    f"maximum_{output}_difference": float(np.max(differences)),
                    f"{output}_difference_sd": float(
                        np.std(differences, ddof=1)
                        if differences.size > 1
                        else 0.0
                    ),
                    f"{output}_difference_fraction_positive": float(
                        np.mean(differences > 0.0)
                    ),
                }
            )
        paired.append(cell)
    return {"cells": cells, "paired_differences_from_strict": paired}


def positive_int(value: str) -> int:
    """Parse a strictly positive command-line integer."""

    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def validate_execution_role(
    binding,
    *,
    allow_sensitivity_design: bool,
    output_dir: Path,
    seeds: tuple[int, ...],
    swap_salts: tuple[int, ...],
) -> None:
    """Protect the primary output path from partial or sensitivity runs."""

    validate_driver_design_role(
        binding,
        allow_sensitivity_design=allow_sensitivity_design,
        output_dir=output_dir,
        primary_output_dir=DEFAULT_OUTPUT_DIR,
        driver_name="ridge equal-count audit",
    )
    if output_dir.resolve() == DEFAULT_OUTPUT_DIR.resolve() and (
        seeds != PRIMARY_SEEDS or swap_salts != DEFAULT_SWAP_SALTS
    ):
        raise ValueError(
            "primary ridge equal-count output requires the exact PRIMARY_SEEDS "
            "and DEFAULT_SWAP_SALTS grid"
        )


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line interface for the ridge exposure audit."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--design-audit", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--allow-sensitivity-design", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(PRIMARY_SEEDS))
    parser.add_argument(
        "--swap-salts", type=int, nargs="+", default=list(DEFAULT_SWAP_SALTS)
    )
    parser.add_argument("--workers", type=positive_int, default=1)
    return parser


def main() -> None:
    """Run strict and row-count-matched ridge exposure comparisons."""

    args = build_parser().parse_args()
    started = time.perf_counter()
    design_binding = bind_application_design(
        args.design, args.manifest, audit_path=args.design_audit
    )
    seeds = tuple(int(item) for item in args.seeds)
    swap_salts = tuple(int(item) for item in args.swap_salts)
    if not seeds or len(seeds) != len(set(seeds)) or any(seed < 0 for seed in seeds):
        raise ValueError("ridge audit seeds must be nonempty, unique, and nonnegative")
    if (
        not swap_salts
        or len(swap_salts) != len(set(swap_salts))
        or any(salt < 0 for salt in swap_salts)
    ):
        raise ValueError("swap salts must be nonempty, unique, and nonnegative")
    validate_execution_role(
        design_binding,
        allow_sensitivity_design=args.allow_sensitivity_design,
        output_dir=args.output_dir,
        seeds=seeds,
        swap_salts=swap_salts,
    )

    y, d, x, row, col, n_nodes, n_features = load_design(args.design)
    if (
        n_nodes != design_binding.n_nodes
        or row.size != design_binding.n_edges
        or n_features != design_binding.n_controls
    ):
        raise RuntimeError("loaded ridge design disagrees with its manifest binding")

    rows: list[dict] = []
    if args.workers == 1:
        with threadpool_limits(limits=1):
            for completed, seed in enumerate(seeds, start=1):
                rows.extend(
                    run_seed(
                        y,
                        d,
                        x,
                        row,
                        col,
                        n_nodes,
                        seed=seed,
                        swap_salts=swap_salts,
                    )
                )
                print(f"completed {completed}/{len(seeds)} seeds", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    _run_seed_from_path, str(args.design), seed, swap_salts
                ): seed
                for seed in seeds
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                rows.extend(future.result())
                print(f"completed {completed}/{len(seeds)} seeds", flush=True)

    rows.sort(
        key=lambda item: (
            int(item["seed"]),
            str(item["method"]),
            -1 if item["swap_salt"] is None else int(item["swap_salt"]),
        )
    )
    for item in rows:
        item["application_design_spec"] = design_binding.design_spec
        item["application_design_role"] = design_binding.design_role
        item["application_design_is_primary_default"] = (
            design_binding.is_primary_default
        )

    summary = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "diagnostic_only",
        "runtime_seconds": float(time.perf_counter() - started),
        "application_design_binding": design_binding.as_dict(),
        "learner": {
            "factory": "experiments.run_continuous_dml.ridge_factory",
            "class": "sklearn.linear_model.Ridge",
            "alpha": PRIMARY_ALPHA,
            "solver": "lsqr",
            "tolerance": RIDGE_SOLVER_TOLERANCE,
            "preprocessing": (
                "fold-local StandardScaler for canonical numeric controls; fixed "
                "sector-pair indicators passed through"
            ),
        },
        "design": {
            "n_nodes": n_nodes,
            "n_edges": int(row.size),
            "n_features": n_features,
            "n_folds": PRIMARY_N_FOLDS,
            "seeds": list(seeds),
            "swap_salts": list(swap_salts),
            "treatment": PRIMARY_TREATMENT,
            "outcome": PRIMARY_OUTCOME,
            "reporting_increment": PRIMARY_REPORTING_INCREMENT,
            "score_edges": "identical within-node-fold edges in every arm",
            "equal_count_rule": (
                "maximize cross-boundary rows subject to each fold's exact strict "
                "training-row count; sampling uses only seed, fold, and salt"
            ),
            "residual_mse_interpretation": (
                "equal-fold-weighted observed score residual MSE; not error against "
                "an unknown true nuisance function"
            ),
            "unsupported_inference_fields": (
                "standard error, t, p, and confidence-interval fields are null for "
                "both exposed arms"
            ),
        },
        **summarize(rows, seeds, swap_salts),
    }

    current_binding = bind_application_design(
        args.design, args.manifest, audit_path=args.design_audit
    )
    if current_binding != design_binding:
        raise RuntimeError("application design binding changed during ridge audit")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = args.output_dir / "estimates.parquet"
    csv_path = args.output_dir / "estimates.csv"
    summary_path = args.output_dir / "summary.json"
    pl.DataFrame(rows).write_parquet(parquet_path)
    pl.DataFrame(rows).write_csv(csv_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    run_id = datetime.now(timezone.utc).strftime(
        "ridge-equal-count-"
        f"{design_spec_run_token(design_binding)}-%Y%m%dT%H%M%SZ"
    )
    record(
        run_id=run_id,
        command=[sys.executable, *sys.argv],
        config={
            "seeds": list(seeds),
            "swap_salts": list(swap_salts),
            "workers": args.workers,
            "runtime_seconds": summary["runtime_seconds"],
            "learner": summary["learner"],
            "application_design_binding": design_binding.as_dict(),
        },
        inputs=[args.design, design_binding.audit_path, *SOURCE_DEPENDENCIES],
        outputs=[parquet_path, csv_path, summary_path],
        status="completed",
        manifest_path=args.manifest,
    )
    print(json.dumps({"run_id": run_id, **summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
