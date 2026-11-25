#!/usr/bin/env python3
"""Matched-score learner audit with a fixed nonlinear tree ensemble.

This diagnostic holds node folds and scored pairs fixed while comparing:

* strict outside--outside nuisance training;
* the full non-score complement, including cross-boundary edges; and
* equal-row-count, maximally leaky training for three deterministic samples.

The empirical target remains descriptive and the leaky arms are diagnostic
only.  This driver is separate from the canonical ridge application so adding
a flexible learner cannot silently redefine the primary specification.
Reported score-sample MSEs are prediction-residual MSEs against observed
outcomes and treatments, not errors against unknown true nuisance functions.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_continuous_dml import (  # noqa: E402
    PRIMARY_OUTCOME,
    PRIMARY_REPORTING_INCREMENT,
    PRIMARY_SEEDS,
    PRIMARY_TREATMENT,
    design_matrix,
)
from experiments.application_design_contract import (  # noqa: E402
    DEFAULT_MANIFEST,
    bind_application_design,
    design_spec_run_token,
    validate_driver_design_role,
)
from src.dyadic_dml import (  # noqa: E402
    aggregate_repeated_splits,
    fit_matched_score_equal_size_maximal_leak_plr_diagnostic,
    fit_matched_score_leaky_complement_plr_diagnostic,
    fit_plr_dml,
    make_node_folds,
)
from src.provenance import record  # noqa: E402


N_FOLDS = 5
HGB_RANDOM_STATE = 20260808
DEFAULT_SWAP_SALTS = (20260809, 20260810, 20260811)
DEFAULT_DESIGN = (
    PROJECT_ROOT / "data_public/processed/continuous_design/dyadic_design.parquet"
)
DEFAULT_HGB_OUTPUT_DIR = PROJECT_ROOT / "data_public/processed/hgb_learner_audit"
HGB_SOURCE_DEPENDENCIES = (
    Path(__file__).resolve(),
    PROJECT_ROOT / "experiments/application_design_contract.py",
    PROJECT_ROOT / "experiments/run_continuous_dml.py",
    PROJECT_ROOT / "src/dyadic_dml.py",
    PROJECT_ROOT / "src/provenance.py",
    PROJECT_ROOT / "pyproject.toml",
)
LearnerFactory = Callable[[], object]


def hgb_factory() -> HistGradientBoostingRegressor:
    """Return the frozen nonlinear nuisance learner used by this audit."""

    return HistGradientBoostingRegressor(
        max_iter=100,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=HGB_RANDOM_STATE,
    )


def load_design(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Load and validate the canonical complete undirected application design."""

    frame = pl.read_parquet(path).sort(["NODE_I", "NODE_J"])
    row = frame.get_column("NODE_I").to_numpy().astype(np.int64, copy=False)
    col = frame.get_column("NODE_J").to_numpy().astype(np.int64, copy=False)
    if row.size == 0 or np.any(row < 0) or np.any(row >= col):
        raise ValueError("design must contain sorted unique undirected edges")
    n_nodes = int(max(row.max(), col.max()) + 1)
    expected_edges = n_nodes * (n_nodes - 1) // 2
    if frame.height != expected_edges:
        raise ValueError("HGB audit requires a complete undirected graph")
    if frame.select("NODE_I", "NODE_J").unique().height != expected_edges:
        raise ValueError("HGB audit design contains duplicate edges")
    x, feature_names = design_matrix(frame)
    y = frame.get_column(PRIMARY_OUTCOME).to_numpy()
    d = frame.get_column(PRIMARY_TREATMENT).to_numpy()
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(d)):
        raise ValueError("HGB audit outcome or treatment contains non-finite values")
    return y, d, x, row, col, n_nodes, len(feature_names)


def _training_diagnostics(result, method: str) -> tuple[list[int], list[int]]:
    training_counts = [item.n_train_edges for item in result.fold_audits]
    if method == "strict":
        cross_counts = [0 for _ in result.fold_audits]
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
    else:  # pragma: no cover - internal call sites are fixed
        raise ValueError(f"unknown HGB audit method: {method}")
    return training_counts, cross_counts


def result_row(result, *, method: str, seed: int, swap_salt: int | None) -> dict:
    """Convert one native-scale estimator result to an auditable row."""

    training_counts, cross_counts = _training_diagnostics(result, method)
    tau = PRIMARY_REPORTING_INCREMENT * result.theta
    mechanical_standard_error = PRIMARY_REPORTING_INCREMENT * result.se_dyadic
    valid_crossfit = method == "strict"
    statistic = tau / mechanical_standard_error
    diagnostics = result.diagnostics
    numerical = np.asarray(
        [
            result.theta,
            result.se_dyadic,
            tau,
            mechanical_standard_error,
            result.jacobian,
            result.moment,
            diagnostics["score_outcome_residual_mse"],
            diagnostics["score_treatment_residual_mse"],
        ],
        dtype=float,
    )
    if (
        not np.all(np.isfinite(numerical))
        or result.se_dyadic <= 0.0
        or result.jacobian <= 0.0
    ):
        raise RuntimeError(f"{method} HGB audit fit produced invalid numerical output")
    return {
        "method": method,
        "seed": int(seed),
        "swap_salt": swap_salt,
        "theta_estimation_scale": result.theta,
        "mechanical_dyadic_se_estimation_scale": result.se_dyadic,
        "standard_error_estimation_scale": (
            result.se_dyadic if valid_crossfit else None
        ),
        "tau": tau,
        "mechanical_dyadic_se": mechanical_standard_error,
        "standard_error": mechanical_standard_error if valid_crossfit else None,
        "t_statistic": statistic if valid_crossfit else None,
        "p_value_normal": (
            float(2.0 * norm.sf(abs(statistic))) if valid_crossfit else None
        ),
        "jacobian": result.jacobian,
        "score_outcome_residual_mse": diagnostics[
            "score_outcome_residual_mse"
        ],
        "score_treatment_residual_mse": diagnostics[
            "score_treatment_residual_mse"
        ],
        "score_outcome_residual_sd": diagnostics["score_outcome_residual_sd"],
        "score_treatment_residual_sd": diagnostics[
            "score_treatment_residual_sd"
        ],
        "score_treatment_residual_rms": diagnostics[
            "score_treatment_residual_rms"
        ],
        "n_nodes": result.n_nodes,
        "n_edges": result.n_edges,
        "n_scored": result.n_scored,
        "n_folds": result.n_folds,
        "train_edges_min": min(training_counts),
        "train_edges_max": max(training_counts),
        "cross_boundary_train_edges_min": min(cross_counts),
        "cross_boundary_train_edges_max": max(cross_counts),
        "cross_boundary_train_fraction": float(
            sum(cross_counts) / sum(training_counts)
        ),
        "moment": result.moment,
        "score_edge_fingerprints_json": json.dumps(
            diagnostics["score_edge_fingerprints"], separators=(",", ":")
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
    learner_factory: LearnerFactory = hgb_factory,
) -> list[dict]:
    """Run every matched-score learner arm for one fixed node partition."""

    if n_nodes < 2 * N_FOLDS:
        raise ValueError("HGB audit requires at least two nodes per score fold")
    if seed < 0:
        raise ValueError("HGB audit seed must be nonnegative")
    if (
        not swap_salts
        or len(swap_salts) != len(set(swap_salts))
        or any(salt < 0 for salt in swap_salts)
    ):
        raise ValueError("HGB swap salts must be nonempty, unique, and nonnegative")
    folds = make_node_folds(n_nodes, N_FOLDS, seed=seed)
    strict = fit_plr_dml(
        y,
        d,
        x,
        row,
        col,
        n_nodes,
        n_folds=N_FOLDS,
        seed=seed,
        fold_assignment=folds,
        outcome_factory=learner_factory,
        treatment_factory=learner_factory,
    )
    full_leaky = fit_matched_score_leaky_complement_plr_diagnostic(
        y,
        d,
        x,
        row,
        col,
        n_nodes,
        n_folds=N_FOLDS,
        seed=seed,
        fold_assignment=folds,
        outcome_factory=learner_factory,
        treatment_factory=learner_factory,
    )
    equal_size = [
        fit_matched_score_equal_size_maximal_leak_plr_diagnostic(
            y,
            d,
            x,
            row,
            col,
            n_nodes,
            n_folds=N_FOLDS,
            seed=seed,
            sampling_salt=salt,
            fold_assignment=folds,
            outcome_factory=learner_factory,
            treatment_factory=learner_factory,
        )
        for salt in swap_salts
    ]

    strict_scores = strict.diagnostics["score_edge_fingerprints"]
    for comparison in (full_leaky, *equal_size):
        if comparison.diagnostics["score_edge_fingerprints"] != strict_scores:
            raise AssertionError("HGB learner arms do not score identical edges")
    strict_train_counts = [item.n_train_edges for item in strict.fold_audits]
    for comparison in equal_size:
        if [item.n_train_edges for item in comparison.fold_audits] != strict_train_counts:
            raise AssertionError("equal-size HGB arm does not match strict row counts")

    rows = [result_row(strict, method="strict", seed=seed, swap_salt=None)]
    rows.append(
        result_row(full_leaky, method="full_leaky", seed=seed, swap_salt=None)
    )
    rows.extend(
        result_row(
            result,
            method="equal_size_leaky",
            seed=seed,
            swap_salt=salt,
        )
        for salt, result in zip(swap_salts, equal_size, strict=True)
    )
    return rows


def _run_seed_from_path(
    design_path: str, seed: int, swap_salts: tuple[int, ...]
) -> list[dict]:
    y, d, x, row, col, n_nodes, _ = load_design(Path(design_path))
    with threadpool_limits(limits=1):
        return run_seed(
            y,
            d,
            x,
            row,
            col,
            n_nodes,
            seed=seed,
            swap_salts=swap_salts,
        )


def aggregate_cell(rows: list[dict], expected_seeds: tuple[int, ...]) -> dict:
    """Median-aggregate one method/salt cell over the requested partitions."""

    if len(rows) != len(expected_seeds):
        raise AssertionError("HGB aggregate has the wrong number of seed rows")
    if tuple(sorted(int(item["seed"]) for item in rows)) != tuple(
        sorted(expected_seeds)
    ):
        raise AssertionError("HGB aggregate seed set differs from requested seeds")
    methods = {str(item["method"]) for item in rows}
    salts = {item["swap_salt"] for item in rows}
    if len(methods) != 1 or len(salts) != 1:
        raise AssertionError("HGB aggregate mixes methods or swap salts")
    tau = np.asarray([float(item["tau"]) for item in rows])
    mechanical_se = np.asarray(
        [float(item["mechanical_dyadic_se"]) for item in rows]
    )
    if (
        not np.all(np.isfinite(tau))
        or not np.all(np.isfinite(mechanical_se))
        or np.any(mechanical_se <= 0.0)
    ):
        raise ValueError("HGB aggregate contains invalid estimates or mechanical SEs")
    aggregate_tau, mechanical_aggregate_se = aggregate_repeated_splits(
        tau, mechanical_se
    )
    valid_crossfit_values = {
        bool(item["valid_under_maintained_assumptions"]) for item in rows
    }
    if len(valid_crossfit_values) != 1:
        raise AssertionError("HGB aggregate mixes inference-validity roles")
    valid_crossfit = valid_crossfit_values.pop()
    if valid_crossfit != (rows[0]["method"] == "strict"):
        raise AssertionError("HGB method and inference-validity role disagree")
    inference_roles = {str(item["inference_role"]) for item in rows}
    if len(inference_roles) != 1:
        raise AssertionError("HGB aggregate mixes inference roles")
    standard_error = mechanical_aggregate_se if valid_crossfit else None
    statistic = aggregate_tau / mechanical_aggregate_se if valid_crossfit else None
    diagnostic_names = (
        "jacobian",
        "score_outcome_residual_mse",
        "score_treatment_residual_mse",
        "score_outcome_residual_sd",
        "score_treatment_residual_sd",
        "score_treatment_residual_rms",
        "cross_boundary_train_fraction",
    )
    result = {
        "method": rows[0]["method"],
        "swap_salt": rows[0]["swap_salt"],
        "n_splits": len(rows),
        "tau": aggregate_tau,
        "mechanical_repeated_split_dyadic_se": mechanical_aggregate_se,
        "standard_error": standard_error,
        "t_statistic": statistic,
        "p_value_normal": (
            float(2.0 * norm.sf(abs(statistic))) if statistic is not None else None
        ),
        "tau_min": float(np.min(tau)),
        "tau_max": float(np.max(tau)),
        "tau_split_sd": float(np.std(tau, ddof=1) if tau.size > 1 else 0.0),
        "tau_q25": float(np.quantile(tau, 0.25)),
        "tau_q75": float(np.quantile(tau, 0.75)),
        "valid_dyadic_crossfit": valid_crossfit,
        "valid_under_maintained_assumptions": valid_crossfit,
        "inference_role": rows[0]["inference_role"],
        "valid_primary_inference": False,
        "designated_primary_result": False,
        "diagnostic_only": True,
    }
    for name in diagnostic_names:
        result[f"{name}_median"] = float(
            np.median([float(item[name]) for item in rows])
        )
    return result


def summarize(rows: list[dict], seeds: tuple[int, ...], swap_salts: tuple[int, ...]) -> dict:
    """Summarize fixed learner cells and paired endpoint-exposure shifts."""

    if not seeds or len(seeds) != len(set(seeds)) or any(seed < 0 for seed in seeds):
        raise ValueError("HGB summary seeds must be nonempty, unique, and nonnegative")
    if (
        not swap_salts
        or len(swap_salts) != len(set(swap_salts))
        or any(salt < 0 for salt in swap_salts)
    ):
        raise ValueError(
            "HGB summary swap salts must be nonempty, unique, and nonnegative"
        )
    cells = []
    for method, salt in (
        ("strict", None),
        ("full_leaky", None),
        *(("equal_size_leaky", salt) for salt in swap_salts),
    ):
        subset = [
            item
            for item in rows
            if item["method"] == method and item["swap_salt"] == salt
        ]
        cells.append(aggregate_cell(subset, seeds))

    strict_by_seed = {
        int(item["seed"]): item for item in rows if item["method"] == "strict"
    }
    paired = []
    for method, salt in (
        ("full_leaky", None),
        *(("equal_size_leaky", salt) for salt in swap_salts),
    ):
        subset = [
            item
            for item in rows
            if item["method"] == method and item["swap_salt"] == salt
        ]
        if tuple(sorted(int(item["seed"]) for item in subset)) != tuple(
            sorted(seeds)
        ):
            raise AssertionError("paired HGB comparison has the wrong seed set")
        cell: dict[str, object] = {
            "method": method,
            "swap_salt": salt,
            "n_paired_partitions": len(subset),
            "uncertainty_interpretation": (
                "descriptive variation over fixed algorithmic node partitions; "
                "not independent sampling uncertainty and not a confidence interval"
            ),
        }
        for output_name, column in (
            ("tau", "tau"),
            ("score_outcome_residual_mse", "score_outcome_residual_mse"),
            ("score_treatment_residual_mse", "score_treatment_residual_mse"),
            ("jacobian", "jacobian"),
        ):
            differences = np.asarray(
                [
                    float(item[column])
                    - float(strict_by_seed[int(item["seed"])][column])
                    for item in subset
                ]
            )
            if not np.all(np.isfinite(differences)):
                raise ValueError(f"paired HGB {output_name} differences are invalid")
            cell.update(
                {
                    f"mean_{output_name}_difference": float(np.mean(differences)),
                    f"median_{output_name}_difference": float(np.median(differences)),
                    f"mean_abs_{output_name}_difference": float(
                        np.mean(np.abs(differences))
                    ),
                    f"minimum_{output_name}_difference": float(
                        np.min(differences)
                    ),
                    f"maximum_{output_name}_difference": float(
                        np.max(differences)
                    ),
                    f"{output_name}_difference_sd": float(
                        np.std(differences, ddof=1)
                        if differences.size > 1
                        else 0.0
                    ),
                    f"{output_name}_difference_q25": float(
                        np.quantile(differences, 0.25)
                    ),
                    f"{output_name}_difference_q75": float(
                        np.quantile(differences, 0.75)
                    ),
                    f"{output_name}_difference_fraction_positive": float(
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


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line interface for the HGB exposure audit."""

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
        default=DEFAULT_HGB_OUTPUT_DIR,
    )
    parser.add_argument(
        "--allow-sensitivity-design",
        action="store_true",
        help=(
            "allow a validated nonprimary design; requires a nondefault output directory"
        ),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(PRIMARY_SEEDS))
    parser.add_argument(
        "--swap-salts", type=int, nargs="+", default=list(DEFAULT_SWAP_SALTS)
    )
    parser.add_argument("--workers", type=positive_int, default=1)
    return parser


def main() -> None:
    """Run strict, full-complement, and equal-count HGB nuisance fits."""

    args = build_parser().parse_args()
    design_binding = bind_application_design(
        args.design, args.manifest, audit_path=args.design_audit
    )
    validate_driver_design_role(
        design_binding,
        allow_sensitivity_design=args.allow_sensitivity_design,
        output_dir=args.output_dir,
        primary_output_dir=DEFAULT_HGB_OUTPUT_DIR,
        driver_name="HGB learner audit",
    )
    seeds = tuple(int(item) for item in args.seeds)
    swap_salts = tuple(int(item) for item in args.swap_salts)
    if len(seeds) != len(set(seeds)) or any(item < 0 for item in seeds):
        raise ValueError("HGB audit seeds must be unique and nonnegative")
    if len(swap_salts) != len(set(swap_salts)) or any(
        item < 0 for item in swap_salts
    ):
        raise ValueError("HGB swap salts must be unique and nonnegative")
    if not seeds or not swap_salts:
        raise ValueError("HGB audit requires at least one seed and swap salt")

    y, d, x, row, col, n_nodes, n_features = load_design(args.design)
    if (
        n_nodes != design_binding.n_nodes
        or row.size != design_binding.n_edges
        or n_features != design_binding.n_controls
    ):
        raise RuntimeError("loaded HGB design disagrees with its manifest binding")

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
    expected_row_count = len(seeds) * (2 + len(swap_salts))
    if len(rows) != expected_row_count:
        raise RuntimeError("HGB audit returned the wrong number of rows")
    row_keys = [
        (int(item["seed"]), str(item["method"]), item["swap_salt"])
        for item in rows
    ]
    if len(row_keys) != len(set(row_keys)):
        raise RuntimeError("HGB audit returned duplicate method/seed/salt rows")
    for item in rows:
        item["application_design_spec"] = design_binding.design_spec
        item["application_design_role"] = design_binding.design_role
        item["application_design_is_primary_default"] = (
            design_binding.is_primary_default
        )
    summary = {
        "schema_version": 3,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "diagnostic_only",
        "application_design_binding": design_binding.as_dict(),
        "learner": {
            "class": "HistGradientBoostingRegressor",
            "parameters": hgb_factory().get_params(deep=False),
            "preprocessing": (
                "none; tree splits use the canonical numeric and fixed one-hot design"
            ),
        },
        "design": {
            "n_nodes": n_nodes,
            "n_edges": n_nodes * (n_nodes - 1) // 2,
            "n_features": n_features,
            "n_folds": N_FOLDS,
            "seeds": list(seeds),
            "swap_salts": list(swap_salts),
            "treatment": PRIMARY_TREATMENT,
            "outcome": PRIMARY_OUTCOME,
            "reporting_increment": PRIMARY_REPORTING_INCREMENT,
            "score_edges": "identical within-node-fold edges for every arm",
            "equal_size_rule": (
                "maximize cross-boundary training edges subject to exact strict "
                "training-row count; sample only by seed, fold, and salt"
            ),
            "score_residual_diagnostics": (
                "equal-fold-weighted MSE, mean, SD, and RMS of observed "
                "score-sample prediction residuals; not errors against unknown "
                "true nuisance functions"
            ),
            "unsupported_inference_fields": (
                "full-leaky and equal-size-leaky point estimates are descriptive; "
                "their inference-named SE, t, and p fields are null. The raw engine "
                "SE is retained only under an explicitly mechanical field name."
            ),
            "partition_variation": (
                "ranges, SDs, IQRs, and sign fractions over the 25 fixed partitions "
                "are algorithmic stability diagnostics, not confidence intervals"
            ),
        },
        **summarize(rows, seeds, swap_salts),
    }

    current_design_binding = bind_application_design(
        args.design, args.manifest, audit_path=args.design_audit
    )
    if current_design_binding != design_binding:
        raise RuntimeError("application design binding changed during HGB audit")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = args.output_dir / "estimates.parquet"
    csv_path = args.output_dir / "estimates.csv"
    summary_path = args.output_dir / "summary.json"
    pl.DataFrame(rows).write_parquet(parquet_path)
    pl.DataFrame(rows).write_csv(csv_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    run_id = datetime.now(timezone.utc).strftime(
        f"hgb-learner-audit-{design_spec_run_token(design_binding)}-%Y%m%dT%H%M%SZ"
    )
    record(
        run_id=run_id,
        command=[sys.executable, *sys.argv],
        config={
            "seeds": list(seeds),
            "swap_salts": list(swap_salts),
            "workers": args.workers,
            "learner": summary["learner"],
            "application_design_binding": design_binding.as_dict(),
        },
        inputs=[args.design, design_binding.audit_path, *HGB_SOURCE_DEPENDENCIES],
        outputs=[parquet_path, csv_path, summary_path],
        status="completed",
        manifest_path=args.manifest,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
