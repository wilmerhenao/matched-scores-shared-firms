from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from experiments.application_design_contract import (
    ApplicationDesignError,
    FEATURE_CONTRACT_PATH,
    bind_application_design,
    bind_design_consumer_output,
    design_spec_run_token,
    validate_driver_design_role,
)
from experiments.run_canonical_simulation import (
    THETA_ZERO,
    generate_plr_dgp,
    nuisance_coefficients,
)
from experiments.run_continuous_dml import (
    NUMERIC_CONTROLS,
    PRIMARY_OUTCOME,
    PRIMARY_TREATMENT,
    SECTOR_PAIR_CATEGORIES,
    SECTOR_PAIR_REFERENCE,
)
from experiments.run_node_share_simulation import (
    DEFAULT_NODE_SHARES,
    N_FOLDS,
    analytical_oracle_moment_iid_sd_ratio,
    generate_node_share_dgp,
    one_replication,
    oracle_score_covariance,
    simulation_tasks,
    summarize_share,
    validate_node_shares,
)


def _fingerprint(path: Path) -> dict[str, int | str]:
    payload = path.read_bytes()
    return {"sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}


def _write_design_binding_fixture(
    root: Path,
    *,
    n_nodes: int,
    complete: bool = True,
    design_directory: str = "continuous_design",
    audit_relative: str | None = None,
    run_id: str = "continuous-design-20260809T040000Z",
    recorded_at_utc: str | None = None,
    append_manifest: bool = False,
    design_status: str = "VALIDATED_FOR_ESTIMATION",
    design_spec: str | None = None,
    design_role: str | None = None,
    is_primary_default: bool | None = None,
) -> tuple[Path, Path]:
    design_dir = root / "data_public/processed" / design_directory
    design_dir.mkdir(parents=True, exist_ok=True)
    (root / "experiments").mkdir(exist_ok=True)
    (root / "src").mkdir(exist_ok=True)
    (root / "results").mkdir(exist_ok=True)
    builder = root / "experiments/build_return_design.py"
    provenance = root / "src/provenance.py"
    upstream = root / "data_public/raw/source.txt"
    upstream.parent.mkdir(exist_ok=True)
    builder.write_text("# frozen design builder\n", encoding="utf-8")
    provenance.write_text("# frozen provenance helper\n", encoding="utf-8")
    upstream.write_text("frozen upstream input\n", encoding="utf-8")

    row, col = np.triu_indices(n_nodes, k=1)
    if not complete:
        row = row[:-1]
        col = col[:-1]
    edge_count = row.size
    values: dict[str, object] = {
        "NODE_I": row.astype(np.int64),
        "NODE_J": col.astype(np.int64),
        "CIK_I": [f"{index:010d}" for index in row],
        "CIK_J": [f"{index:010d}" for index in col],
        PRIMARY_TREATMENT: np.linspace(0.15, 0.75, edge_count),
        PRIMARY_OUTCOME: np.linspace(-0.25, 0.35, edge_count),
        "SECTOR_PAIR": [SECTOR_PAIR_REFERENCE] * edge_count,
    }
    for offset, name in enumerate(NUMERIC_CONTROLS, start=1):
        values[name] = np.linspace(0.0, float(offset), edge_count)
    frame = pl.DataFrame(values)
    design = design_dir / "dyadic_design.parquet"
    audit = (
        root / audit_relative
        if audit_relative is not None
        else design_dir / "audit_summary.json"
    )
    audit.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(design)
    audit.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "design_status": design_status,
                "design_spec": design_spec or design_directory,
                "design_role": design_role
                or (
                    "primary"
                    if design_status == "VALIDATED_FOR_ESTIMATION"
                    else "sensitivity"
                ),
                "is_primary_default": (
                    design_status == "VALIDATED_FOR_ESTIMATION"
                    if is_primary_default is None
                    else is_primary_default
                ),
                "eligible_firms": n_nodes,
                "eligible_edges": edge_count,
                "numeric_design_all_finite": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = root / "results/MANIFEST.jsonl"
    record = {
        "schema_version": 1,
        "run_id": run_id,
        "recorded_at_utc": (
            recorded_at_utc or datetime.now(timezone.utc).isoformat()
        ),
        "status": "completed",
        "inputs": {
            "data_public/raw/source.txt": _fingerprint(upstream),
            "experiments/build_return_design.py": _fingerprint(builder),
            "src/provenance.py": _fingerprint(provenance),
        },
        "outputs": {
            design.relative_to(root).as_posix(): _fingerprint(design),
            audit.relative_to(root).as_posix(): _fingerprint(audit),
        },
    }
    with manifest.open(
        "a" if append_manifest else "w", encoding="utf-8"
    ) as stream:
        stream.write(json.dumps(record) + "\n")
    return design, manifest


def test_default_grid_is_valid_and_contains_mechanism_reference_share() -> None:
    assert validate_node_shares(DEFAULT_NODE_SHARES) == DEFAULT_NODE_SHARES
    assert 0.25 in DEFAULT_NODE_SHARES
    assert DEFAULT_NODE_SHARES[-1] == 1.0
    assert 0.90 not in DEFAULT_NODE_SHARES
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        validate_node_shares([0.0])
    with pytest.raises(ValueError, match="unique"):
        validate_node_shares([0.25, 0.25])


def test_two_thirds_share_reproduces_canonical_dgp_numerically() -> None:
    canonical = generate_plr_dgp(20, 5, 123)
    sensitivity = generate_node_share_dgp(20, 5, 123, 2.0 / 3.0)
    for expected, observed in zip(canonical, sensitivity, strict=True):
        assert np.allclose(expected, observed, rtol=1e-15, atol=1e-15)


def test_common_seed_holds_controls_and_edges_fixed_across_shares() -> None:
    low = generate_node_share_dgp(20, 5, 321, 0.10)
    high = generate_node_share_dgp(20, 5, 321, 1.00)
    assert np.array_equal(low[2], high[2])
    assert np.array_equal(low[3], high[3])
    assert np.array_equal(low[4], high[4])
    assert not np.array_equal(low[0], high[0])
    assert not np.array_equal(low[1], high[1])


def test_score_covariance_and_analytical_ratio_are_calibrated() -> None:
    assert oracle_score_covariance(2.0 / 3.0) == pytest.approx(1.0 / 9.0)
    assert oracle_score_covariance(1.0) == pytest.approx(0.25)
    n_nodes = 137
    ratios = [
        analytical_oracle_moment_iid_sd_ratio(n_nodes, 5, share)
        for share in DEFAULT_NODE_SHARES
    ]
    assert all(left > right for left, right in zip(ratios, ratios[1:]))
    assert 0.0 < analytical_oracle_moment_iid_sd_ratio(n_nodes, 5, 1.0) < 1.0
    with pytest.raises(ValueError, match="two nodes per fold"):
        analytical_oracle_moment_iid_sd_ratio(2 * N_FOLDS - 1, N_FOLDS, 0.5)


def test_lambda_one_is_a_regular_node_only_innovation_boundary() -> None:
    y, d, x, _, _ = generate_node_share_dgp(50, 8, 20260809, 1.0)
    beta, gamma = nuisance_coefficients(8)
    v = d - x @ beta
    epsilon = y - THETA_ZERO * d - x @ gamma
    assert np.all(np.isfinite(v))
    assert np.all(np.isfinite(epsilon))
    assert np.var(v) > 0.5
    assert np.var(epsilon) > 0.5
    assert np.var(v * epsilon) > 0.25


def test_application_binding_derives_dimensions_and_exact_producer(tmp_path: Path) -> None:
    design, manifest = _write_design_binding_fixture(tmp_path, n_nodes=9)
    sensitivity_record = {
        "schema_version": 1,
        "run_id": "continuous-design-balanced_q1_2024-20260809T050000Z",
        "recorded_at_utc": "2026-08-09T05:00:00+00:00",
        "status": "completed",
        "inputs": {},
        "outputs": {
            "data_public/processed/continuous_design_q1_balanced/"
            "dyadic_design.parquet": {
                "sha256": "not-the-primary-design",
                "size_bytes": 1,
            },
            "data_public/processed/continuous_design_q1_balanced/"
            "audit_summary.json": {
                "sha256": "not-the-primary-audit",
                "size_bytes": 1,
            },
        },
    }
    with manifest.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(sensitivity_record) + "\n")
    binding = bind_application_design(
        design,
        manifest,
        project_root=tmp_path,
        feature_contract_path=FEATURE_CONTRACT_PATH,
    )
    expected_controls = len(NUMERIC_CONTROLS) + len(SECTOR_PAIR_CATEGORIES) - 1
    assert binding.n_nodes == 9
    assert binding.n_edges == 36
    assert binding.n_controls == expected_controls
    assert binding.producer_run_id == "continuous-design-20260809T040000Z"
    assert binding.design_sha256 == _fingerprint(design)["sha256"]
    assert binding.feature_contract_sha256 == _fingerprint(FEATURE_CONTRACT_PATH)[
        "sha256"
    ]


def test_sensitivity_binding_selects_its_own_producer_and_audit(tmp_path: Path) -> None:
    root = tmp_path / "multi_spec"
    _write_design_binding_fixture(
        root,
        n_nodes=9,
        recorded_at_utc="2026-08-09T04:00:00+00:00",
    )
    sensitivity_design, manifest = _write_design_binding_fixture(
        root,
        n_nodes=11,
        design_directory="continuous_design_q1_balanced",
        run_id="continuous-design-balanced_q1_2024-20260809T050000Z",
        recorded_at_utc="2026-08-09T05:00:00+00:00",
        append_manifest=True,
        design_status="VALIDATED_SENSITIVITY_DESIGN",
    )
    inferred = bind_application_design(
        sensitivity_design,
        manifest,
        project_root=root,
        feature_contract_path=FEATURE_CONTRACT_PATH,
    )
    explicit = bind_application_design(
        sensitivity_design,
        manifest,
        audit_path=sensitivity_design.parent / "audit_summary.json",
        project_root=root,
        feature_contract_path=FEATURE_CONTRACT_PATH,
    )
    assert inferred == explicit
    assert inferred.producer_run_id.endswith("20260809T050000Z")
    assert (inferred.n_nodes, inferred.n_edges) == (11, 55)
    assert inferred.design_status == "VALIDATED_SENSITIVITY_DESIGN"
    assert inferred.design_role == "sensitivity"
    assert inferred.is_primary_default is False


def test_driver_role_gate_requires_opt_in_and_distinct_sensitivity_output(
    tmp_path: Path,
) -> None:
    primary_root = tmp_path / "primary"
    primary_design, primary_manifest = _write_design_binding_fixture(
        primary_root, n_nodes=8
    )
    primary = bind_application_design(
        primary_design,
        primary_manifest,
        project_root=primary_root,
        feature_contract_path=FEATURE_CONTRACT_PATH,
    )
    primary_output = primary_root / "results/primary"
    validate_driver_design_role(
        primary,
        allow_sensitivity_design=False,
        output_dir=primary_output,
        primary_output_dir=primary_output,
        driver_name="test driver",
    )

    sensitivity_root = tmp_path / "sensitivity"
    sensitivity_design, sensitivity_manifest = _write_design_binding_fixture(
        sensitivity_root,
        n_nodes=8,
        design_directory="continuous_design_partial_to_exit",
        design_status="VALIDATED_SENSITIVITY_DESIGN",
        design_spec="partial_to_exit_2024",
        design_role="survival_selection_sensitivity",
    )
    sensitivity = bind_application_design(
        sensitivity_design,
        sensitivity_manifest,
        project_root=sensitivity_root,
        feature_contract_path=FEATURE_CONTRACT_PATH,
    )
    default_output = sensitivity_root / "results/primary"
    sensitivity_output = sensitivity_root / "results/partial_to_exit"
    with pytest.raises(ApplicationDesignError, match="--allow-sensitivity-design"):
        validate_driver_design_role(
            sensitivity,
            allow_sensitivity_design=False,
            output_dir=sensitivity_output,
            primary_output_dir=default_output,
            driver_name="test driver",
        )
    with pytest.raises(ApplicationDesignError, match="primary default output"):
        validate_driver_design_role(
            sensitivity,
            allow_sensitivity_design=True,
            output_dir=default_output,
            primary_output_dir=default_output,
            driver_name="test driver",
        )
    validate_driver_design_role(
        sensitivity,
        allow_sensitivity_design=True,
        output_dir=sensitivity_output,
        primary_output_dir=default_output,
        driver_name="test driver",
    )
    assert design_spec_run_token(sensitivity) == "partial_to_exit_2024"


@pytest.mark.parametrize(
    ("design_status", "design_role", "is_primary_default"),
    [
        ("VALIDATED_FOR_ESTIMATION", "sensitivity", True),
        ("VALIDATED_FOR_ESTIMATION", "primary", False),
        ("VALIDATED_SENSITIVITY_DESIGN", "primary", False),
        ("VALIDATED_SENSITIVITY_DESIGN", "diagnostic", False),
        ("VALIDATED_SENSITIVITY_DESIGN", "sensitivity", True),
    ],
)
def test_application_binding_rejects_inconsistent_audit_roles(
    tmp_path: Path,
    design_status: str,
    design_role: str,
    is_primary_default: bool,
) -> None:
    design, manifest = _write_design_binding_fixture(
        tmp_path,
        n_nodes=8,
        design_status=design_status,
        design_role=design_role,
        is_primary_default=is_primary_default,
    )
    with pytest.raises(ApplicationDesignError, match="validated .* design must"):
        bind_application_design(
            design,
            manifest,
            project_root=tmp_path,
            feature_contract_path=FEATURE_CONTRACT_PATH,
        )


def test_downstream_output_binding_requires_the_same_design_and_current_files(
    tmp_path: Path,
) -> None:
    design, manifest = _write_design_binding_fixture(tmp_path, n_nodes=8)
    binding = bind_application_design(
        design,
        manifest,
        project_root=tmp_path,
        feature_contract_path=FEATURE_CONTRACT_PATH,
    )
    estimate = tmp_path / "data_public/processed/continuous_results/estimates.parquet"
    estimate.parent.mkdir(parents=True, exist_ok=True)
    estimate.write_bytes(b"frozen DML estimate")
    producer = {
        "schema_version": 1,
        "run_id": "continuous-dml-continuous_design-20260809T060000Z",
        "recorded_at_utc": "2026-08-09T06:00:00+00:00",
        "status": "completed",
        "inputs": {
            binding.design_manifest_key: _fingerprint(binding.design_path),
            binding.audit_manifest_key: _fingerprint(binding.audit_path),
        },
        "outputs": {
            estimate.relative_to(tmp_path).as_posix(): _fingerprint(estimate),
        },
    }
    with manifest.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(producer) + "\n")

    downstream = bind_design_consumer_output(
        binding, estimate, run_prefix="continuous-dml-"
    )
    assert downstream.output_sha256 == _fingerprint(estimate)["sha256"]
    assert downstream.producer_run_id == producer["run_id"]
    assert downstream.design_sha256 == binding.design_sha256

    estimate.write_bytes(b"replaced estimate")
    with pytest.raises(ApplicationDesignError, match="no current"):
        bind_design_consumer_output(binding, estimate, run_prefix="continuous-dml-")


def test_custom_audit_path_must_match_the_same_manifest_producer(tmp_path: Path) -> None:
    root = tmp_path / "custom_audit"
    custom_audit = "data_public/processed/design_audits/q1_audit.json"
    design, manifest = _write_design_binding_fixture(
        root,
        n_nodes=10,
        design_directory="continuous_design_q1_custom",
        audit_relative=custom_audit,
        design_status="VALIDATED_SENSITIVITY_DESIGN",
    )
    binding = bind_application_design(
        design,
        manifest,
        audit_path=root / custom_audit,
        project_root=root,
        feature_contract_path=FEATURE_CONTRACT_PATH,
    )
    assert binding.audit_path == (root / custom_audit).resolve()

    unrecorded_audit = root / "data_public/processed/design_audits/copy.json"
    unrecorded_audit.write_bytes(binding.audit_path.read_bytes())
    with pytest.raises(ApplicationDesignError, match="no completed"):
        bind_application_design(
            design,
            manifest,
            audit_path=unrecorded_audit,
            project_root=root,
            feature_contract_path=FEATURE_CONTRACT_PATH,
        )


def test_application_binding_rejects_ambiguous_or_duplicate_manifest_records(
    tmp_path: Path,
) -> None:
    tied_root = tmp_path / "tied"
    design, manifest = _write_design_binding_fixture(
        tied_root,
        n_nodes=8,
        run_id="continuous-design-primary-a",
        recorded_at_utc="2026-08-09T04:00:00+00:00",
    )
    _write_design_binding_fixture(
        tied_root,
        n_nodes=8,
        run_id="continuous-design-primary-b",
        recorded_at_utc="2026-08-09T04:00:00+00:00",
        append_manifest=True,
    )
    with pytest.raises(ApplicationDesignError, match="ambiguous"):
        bind_application_design(
            design,
            manifest,
            project_root=tied_root,
            feature_contract_path=FEATURE_CONTRACT_PATH,
        )

    duplicate_root = tmp_path / "duplicate"
    design, manifest = _write_design_binding_fixture(duplicate_root, n_nodes=8)
    first_record = manifest.read_text(encoding="utf-8")
    with manifest.open("a", encoding="utf-8") as stream:
        stream.write(first_record)
    with pytest.raises(ApplicationDesignError, match="duplicate manifest run_id"):
        bind_application_design(
            design,
            manifest,
            project_root=duplicate_root,
            feature_contract_path=FEATURE_CONTRACT_PATH,
        )


def test_application_binding_rejects_duplicate_output_fingerprint_keys(
    tmp_path: Path,
) -> None:
    root = tmp_path / "duplicate_output_key"
    design, manifest = _write_design_binding_fixture(root, n_nodes=8)
    record = json.loads(manifest.read_text(encoding="utf-8"))
    design_key = design.relative_to(root).as_posix()
    output_items: list[str] = []
    for key, fingerprint in record["outputs"].items():
        encoded = f"{json.dumps(key)}:{json.dumps(fingerprint)}"
        output_items.append(encoded)
        if key == design_key:
            output_items.append(encoded)
    record_without_outputs = {
        key: value for key, value in record.items() if key != "outputs"
    }
    malformed = (
        json.dumps(record_without_outputs)[:-1]
        + ',"outputs":{'
        + ",".join(output_items)
        + "}}\n"
    )
    manifest.write_text(malformed, encoding="utf-8")

    with pytest.raises(ApplicationDesignError, match="duplicate JSON key"):
        bind_application_design(
            design,
            manifest,
            project_root=root,
            feature_contract_path=FEATURE_CONTRACT_PATH,
        )


def test_application_binding_fails_closed_for_stale_or_malformed_design(
    tmp_path: Path,
) -> None:
    stale_root = tmp_path / "stale"
    design, manifest = _write_design_binding_fixture(stale_root, n_nodes=8)
    design.write_bytes(b"not the recorded parquet")
    with pytest.raises(ApplicationDesignError, match="stale"):
        bind_application_design(
            design,
            manifest,
            project_root=stale_root,
            feature_contract_path=FEATURE_CONTRACT_PATH,
        )

    upstream_root = tmp_path / "upstream"
    design, manifest = _write_design_binding_fixture(upstream_root, n_nodes=8)
    (upstream_root / "data_public/raw/source.txt").write_text(
        "new upstream input\n", encoding="utf-8"
    )
    with pytest.raises(ApplicationDesignError, match="stale"):
        bind_application_design(
            design,
            manifest,
            project_root=upstream_root,
            feature_contract_path=FEATURE_CONTRACT_PATH,
        )

    malformed_root = tmp_path / "malformed"
    design, manifest = _write_design_binding_fixture(
        malformed_root, n_nodes=8, complete=False
    )
    with pytest.raises(ApplicationDesignError, match="every undirected edge"):
        bind_application_design(
            design,
            manifest,
            project_root=malformed_root,
            feature_contract_path=FEATURE_CONTRACT_PATH,
        )


def test_node_share_tasks_use_bound_dimensions_without_copied_constants() -> None:
    tasks = simulation_tasks(
        n_nodes=43,
        n_controls=17,
        replications=3,
        node_shares=(0.10, 0.50),
        seed=20260808,
    )
    assert len(tasks) == 6
    assert {(item[0], item[1]) for item in tasks} == {(43, 17)}
    assert [item[3] for item in tasks[:3]] == [item[3] for item in tasks[3:]]
    with pytest.raises(ValueError, match="two nodes per score fold"):
        simulation_tasks(
            n_nodes=2 * N_FOLDS - 1,
            n_controls=17,
            replications=2,
            node_shares=(0.25,),
            seed=1,
        )
    with pytest.raises(ValueError, match="nonnegative"):
        simulation_tasks(
            n_nodes=30,
            n_controls=17,
            replications=2,
            node_shares=(0.25,),
            seed=-1,
        )


def test_small_replication_shares_scores_and_solves_moments() -> None:
    result = one_replication((30, 8, 0, 20260809, 0.25))
    assert result["strict_n_scored"] == result["oracle_n_scored"]
    assert abs(float(result["strict_moment"])) < 1e-12
    assert abs(float(result["oracle_moment"])) < 1e-12


def test_summary_reports_learned_and_oracle_calibration() -> None:
    rows = [
        one_replication((30, 8, repetition, 100 + repetition, 0.25))
        for repetition in range(3)
    ]
    summary = summarize_share(pl.DataFrame(rows))
    assert summary["replications"] == 3
    assert summary["node_innovation_share"] == 0.25
    assert 0.0 < summary["analytical_oracle_moment_iid_sd_to_exact_sd_ratio"] < 1.0
    for split in ("strict", "oracle"):
        assert summary[f"{split}_mean_mcse"] > 0.0
        assert summary[f"{split}_bias_mcse"] == summary[f"{split}_mean_mcse"]
        for variance in ("dyadic", "iid"):
            assert 0.0 <= summary[f"{split}_{variance}_coverage95"] <= 1.0
            assert summary[f"{split}_{variance}_coverage95_successes"] in range(4)
            assert summary[f"{split}_{variance}_rejection5"] == pytest.approx(
                1.0 - summary[f"{split}_{variance}_coverage95"]
            )
            assert summary[f"{split}_{variance}_abs_t_quantile95"] >= 0.0


def test_node_share_summary_fails_closed_on_duplicate_seed_or_invalid_se() -> None:
    rows = [
        one_replication((30, 8, repetition, 100 + repetition, 0.25))
        for repetition in range(2)
    ]
    duplicate = pl.DataFrame(
        [rows[0], {**rows[1], "paired_seed": rows[0]["paired_seed"]}]
    )
    with pytest.raises(ValueError, match="duplicate paired_seed"):
        summarize_share(duplicate)

    invalid = pl.DataFrame(
        [rows[0], {**rows[1], "strict_se_dyadic": float("nan")}]
    )
    with pytest.raises(ValueError, match="standard errors are invalid"):
        summarize_share(invalid)
