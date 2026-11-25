"""Fail-closed binding to the current validated application design.

Simulation drivers that describe a cell as "application-sized" must not carry
copied node or feature counts.  This module binds them to the exact current
continuous-design artifact, its producing manifest record, and the feature
construction contract used by the empirical DML application.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from experiments.run_continuous_dml import (  # noqa: E402
    NUMERIC_CONTROLS,
    PRIMARY_OUTCOME,
    PRIMARY_TREATMENT,
    SECTOR_PAIR_CATEGORIES,
    design_matrix,
    node_label_map,
)
from src.provenance import sha256_file  # noqa: E402


DEFAULT_DESIGN = (
    PROJECT_ROOT / "data_public/processed/continuous_design/dyadic_design.parquet"
)
DEFAULT_MANIFEST = PROJECT_ROOT / "results/MANIFEST.jsonl"
FEATURE_CONTRACT_PATH = PROJECT_ROOT / "experiments/run_continuous_dml.py"
REQUIRED_DESIGN_CODE_INPUTS = (
    "experiments/build_return_design.py",
    "src/provenance.py",
)


class ApplicationDesignError(RuntimeError):
    """Raised when an application-sized simulation cannot bind a valid design."""


@dataclass(frozen=True)
class ApplicationDesignBinding:
    """Immutable provenance and dimension contract for one application design."""

    design_path: Path
    design_manifest_key: str
    design_sha256: str
    design_size_bytes: int
    audit_path: Path
    audit_manifest_key: str
    audit_sha256: str
    audit_size_bytes: int
    manifest_path: Path
    producer_run_id: str
    producer_recorded_at_utc: str
    design_status: str
    design_spec: str
    design_role: str
    is_primary_default: bool
    n_nodes: int
    n_edges: int
    n_controls: int
    feature_names_sha256: str
    feature_contract_path: Path
    feature_contract_sha256: str

    def as_dict(self) -> dict[str, str | int | bool]:
        """Return the exact binding recorded in summaries and run provenance."""

        return {
            "design_path": self.design_manifest_key,
            "design_sha256": self.design_sha256,
            "design_size_bytes": self.design_size_bytes,
            "design_audit_path": self.audit_manifest_key,
            "design_audit_sha256": self.audit_sha256,
            "design_audit_size_bytes": self.audit_size_bytes,
            "manifest_path": _display_path(self.manifest_path, PROJECT_ROOT),
            "producer_run_id": self.producer_run_id,
            "producer_recorded_at_utc": self.producer_recorded_at_utc,
            "design_status": self.design_status,
            "design_spec": self.design_spec,
            "design_role": self.design_role,
            "is_primary_default": self.is_primary_default,
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "n_controls": self.n_controls,
            "feature_names_sha256": self.feature_names_sha256,
            "feature_contract_path": _display_path(
                self.feature_contract_path, PROJECT_ROOT
            ),
            "feature_contract_sha256": self.feature_contract_sha256,
        }


@dataclass(frozen=True)
class DesignConsumerBinding:
    """Exact manifest binding for a downstream output of one design."""

    output_path: Path
    output_manifest_key: str
    output_sha256: str
    output_size_bytes: int
    producer_run_id: str
    producer_recorded_at_utc: str
    design_sha256: str
    design_audit_sha256: str

    def as_dict(self) -> dict[str, str | int]:
        """Serialize one downstream-output binding for provenance records."""

        return {
            "output_path": self.output_manifest_key,
            "output_sha256": self.output_sha256,
            "output_size_bytes": self.output_size_bytes,
            "producer_run_id": self.producer_run_id,
            "producer_recorded_at_utc": self.producer_recorded_at_utc,
            "design_sha256": self.design_sha256,
            "design_audit_sha256": self.design_audit_sha256,
        }


def _display_path(path: Path, root: Path) -> str:
    return os.path.relpath(path.resolve(), root.resolve()).replace(os.sep, "/")


def _fingerprint(path: Path) -> dict[str, str | int]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ApplicationDesignError(f"required file is missing: {resolved}")
    return {
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


class _DuplicateJSONKey(ValueError):
    pass


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateJSONKey(key)
        value[key] = item
    return value


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ApplicationDesignError(f"provenance manifest is missing: {path}")
    records: list[dict[str, Any]] = []
    seen_run_ids: set[str] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line, object_pairs_hook=_reject_duplicate_json_keys)
        except _DuplicateJSONKey as error:
            raise ApplicationDesignError(
                f"duplicate JSON key at {path}:{line_number}: {error}"
            ) from error
        except json.JSONDecodeError as error:
            raise ApplicationDesignError(
                f"invalid JSON at {path}:{line_number}: {error}"
            ) from error
        if not isinstance(record, dict):
            raise ApplicationDesignError(
                f"manifest line {line_number} must contain a JSON object"
            )
        run_id = record.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ApplicationDesignError(
                f"manifest line {line_number} lacks a nonempty run_id"
            )
        if run_id in seen_run_ids:
            raise ApplicationDesignError(f"duplicate manifest run_id: {run_id}")
        seen_run_ids.add(run_id)
        records.append(record)
    return records


def _record_time(record: dict[str, Any]) -> datetime:
    try:
        value = datetime.fromisoformat(str(record["recorded_at_utc"]))
    except (KeyError, TypeError, ValueError) as error:
        raise ApplicationDesignError(
            f"design run {record.get('run_id', '<unknown>')} has an invalid timestamp"
        ) from error
    if value.tzinfo is None:
        raise ApplicationDesignError(
            f"design run {record.get('run_id', '<unknown>')} timestamp lacks a timezone"
        )
    return value


def _latest_design_record(
    records: list[dict[str, Any]],
    *,
    design_key: str,
    audit_key: str,
) -> dict[str, Any]:
    """Return the newest completed producer of the requested design pair.

    The manifest can contain newer continuous-design runs for sensitivity
    specifications.  Those runs must not supersede the default primary design
    merely because they were recorded later.
    """

    candidates = [
        record
        for record in records
        if record.get("status") == "completed"
        and str(record.get("run_id", "")).startswith("continuous-design-")
        and isinstance(record.get("outputs"), dict)
        and design_key in record["outputs"]
        and audit_key in record["outputs"]
    ]
    if not candidates:
        raise ApplicationDesignError(
            "manifest has no completed continuous-design producer for "
            f"{design_key} and {audit_key}"
        )
    latest_time = max(_record_time(record) for record in candidates)
    latest = [record for record in candidates if _record_time(record) == latest_time]
    if len(latest) != 1:
        run_ids = sorted(str(record.get("run_id")) for record in latest)
        raise ApplicationDesignError(
            "ambiguous continuous-design producers share the latest timestamp: "
            f"{run_ids}"
        )
    return latest[0]


def _require_recorded_fingerprint(
    record: dict[str, Any],
    *,
    collection: str,
    key: str,
    current: dict[str, str | int],
) -> None:
    values = record.get(collection)
    if not isinstance(values, dict):
        raise ApplicationDesignError(
            f"design run {record.get('run_id')} lacks a {collection} dictionary"
        )
    recorded = values.get(key)
    if not isinstance(recorded, dict):
        raise ApplicationDesignError(
            f"design run {record.get('run_id')} does not record {collection[:-1]} {key}"
        )
    expected = {
        "sha256": recorded.get("sha256"),
        "size_bytes": recorded.get("size_bytes"),
    }
    if expected != current:
        raise ApplicationDesignError(
            f"latest design run {record.get('run_id')} is stale for {key}"
        )


def _validate_all_recorded_files(
    record: dict[str, Any], *, project_root: Path
) -> None:
    """Require every recorded design input and output to remain current."""

    for collection in ("inputs", "outputs"):
        values = record.get(collection)
        if not isinstance(values, dict) or not values:
            raise ApplicationDesignError(
                f"design run {record.get('run_id')} lacks recorded {collection}"
            )
        for key in values:
            raw_path = Path(key)
            path = raw_path if raw_path.is_absolute() else project_root / raw_path
            current = _fingerprint(path)
            _require_recorded_fingerprint(
                record,
                collection=collection,
                key=key,
                current=current,
            )


def _validate_manifest_binding(
    *,
    project_root: Path,
    manifest_path: Path,
    design_path: Path,
    audit_path: Path,
) -> tuple[dict[str, Any], str, str, dict[str, str | int], dict[str, str | int]]:
    design_key = _display_path(design_path, project_root)
    audit_key = _display_path(audit_path, project_root)
    design_fingerprint = _fingerprint(design_path)
    audit_fingerprint = _fingerprint(audit_path)
    producer = _latest_design_record(
        _read_manifest(manifest_path),
        design_key=design_key,
        audit_key=audit_key,
    )
    _require_recorded_fingerprint(
        producer,
        collection="outputs",
        key=design_key,
        current=design_fingerprint,
    )
    _require_recorded_fingerprint(
        producer,
        collection="outputs",
        key=audit_key,
        current=audit_fingerprint,
    )
    for code_key in REQUIRED_DESIGN_CODE_INPUTS:
        current = _fingerprint(project_root / code_key)
        _require_recorded_fingerprint(
            producer,
            collection="inputs",
            key=code_key,
            current=current,
        )
    # This catches an apparently current design whose upstream ownership,
    # price, factor, builder, or companion outputs have since been replaced.
    _validate_all_recorded_files(producer, project_root=project_root)
    return producer, design_key, audit_key, design_fingerprint, audit_fingerprint


def _load_audit(path: Path) -> dict[str, Any]:
    try:
        audit = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ApplicationDesignError(f"invalid design audit {path}: {error}") from error
    if not isinstance(audit, dict):
        raise ApplicationDesignError("design audit must contain a JSON object")
    supported_statuses = {
        "VALIDATED_FOR_ESTIMATION",
        "VALIDATED_SENSITIVITY_DESIGN",
    }
    if audit.get("design_status") not in supported_statuses:
        raise ApplicationDesignError(
            "design audit is not marked as a validated primary or sensitivity design"
        )
    design_spec = audit.get("design_spec")
    design_role = audit.get("design_role")
    is_primary_default = audit.get("is_primary_default")
    if not isinstance(design_spec, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*", design_spec
    ):
        raise ApplicationDesignError("design audit has an invalid design_spec")
    if not isinstance(design_role, str) or not design_role:
        raise ApplicationDesignError("design audit has an invalid design_role")
    if not isinstance(is_primary_default, bool):
        raise ApplicationDesignError(
            "design audit is_primary_default must be an explicit boolean"
        )
    if audit["design_status"] == "VALIDATED_FOR_ESTIMATION":
        if design_role != "primary" or is_primary_default is not True:
            raise ApplicationDesignError(
                "validated estimation design must have role=primary and "
                "is_primary_default=true"
            )
    elif design_role == "primary" or "sensitivity" not in design_role.lower() or (
        is_primary_default is not False
    ):
        raise ApplicationDesignError(
            "validated sensitivity design must have a sensitivity role and "
            "is_primary_default=false"
        )
    return audit


def design_spec_run_token(binding: ApplicationDesignBinding) -> str:
    """Return the validated design spec as a collision-resistant run-ID token."""

    # ``_load_audit`` has already restricted this value to a portable token.
    return binding.design_spec.lower()


def validate_driver_design_role(
    binding: ApplicationDesignBinding,
    *,
    allow_sensitivity_design: bool,
    output_dir: Path,
    primary_output_dir: Path,
    driver_name: str,
) -> None:
    """Keep primary defaults from being overwritten by sensitivity executions."""

    is_primary = (
        binding.design_status == "VALIDATED_FOR_ESTIMATION"
        and binding.design_role == "primary"
        and binding.is_primary_default is True
    )
    if is_primary:
        return
    if not allow_sensitivity_design:
        raise ApplicationDesignError(
            f"{driver_name} received sensitivity design {binding.design_spec!r}; "
            "pass --allow-sensitivity-design and a distinct --output-dir"
        )
    if output_dir.resolve() == primary_output_dir.resolve():
        raise ApplicationDesignError(
            f"{driver_name} cannot write sensitivity design {binding.design_spec!r} "
            "to its primary default output directory"
        )


def _load_and_validate_frame(path: Path) -> tuple[pl.DataFrame, int, int, int, str]:
    try:
        frame = pl.read_parquet(path)
    except Exception as error:
        raise ApplicationDesignError(f"cannot read application design: {error}") from error
    required = {
        "NODE_I",
        "NODE_J",
        "CIK_I",
        "CIK_J",
        PRIMARY_TREATMENT,
        PRIMARY_OUTCOME,
        "SECTOR_PAIR",
        *NUMERIC_CONTROLS,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ApplicationDesignError(
            f"application design is missing required columns: {missing}"
        )
    if not frame.schema["NODE_I"].is_integer() or not frame.schema[
        "NODE_J"
    ].is_integer():
        raise ApplicationDesignError("NODE_I and NODE_J must be integer columns")
    row = frame.get_column("NODE_I").to_numpy().astype(np.int64, copy=False)
    col = frame.get_column("NODE_J").to_numpy().astype(np.int64, copy=False)
    if row.size == 0 or np.any(row < 0) or np.any(row >= col):
        raise ApplicationDesignError(
            "design must contain canonical undirected edges with 0 <= NODE_I < NODE_J"
        )
    n_nodes = int(max(row.max(), col.max()) + 1)
    expected_row, expected_col = np.triu_indices(n_nodes, k=1)
    if not np.array_equal(row, expected_row) or not np.array_equal(col, expected_col):
        raise ApplicationDesignError(
            "design must contain every undirected edge exactly once in canonical order"
        )
    n_edges = int(row.size)
    if n_nodes < 5:
        raise ApplicationDesignError("application design has fewer than five nodes")

    labels = node_label_map(frame, n_nodes)
    if len(labels) != n_nodes:
        raise ApplicationDesignError("application design lacks a unique CIK per node")
    for column in (PRIMARY_TREATMENT, PRIMARY_OUTCOME):
        values = frame.get_column(column).to_numpy()
        if not np.all(np.isfinite(values)):
            raise ApplicationDesignError(f"application column {column} is non-finite")
    treatment = frame.get_column(PRIMARY_TREATMENT).to_numpy()
    if np.any(treatment < 0.0) or np.any(treatment > 1.0):
        raise ApplicationDesignError(
            f"raw primary treatment {PRIMARY_TREATMENT} must lie in [0, 1]"
        )

    try:
        x, feature_names = design_matrix(frame)
    except (TypeError, ValueError) as error:
        raise ApplicationDesignError(
            f"application feature-construction contract failed: {error}"
        ) from error
    expected_names = [f"numeric:{name}" for name in NUMERIC_CONTROLS]
    expected_names.extend(
        f"SECTOR_PAIR_{category}" for category in SECTOR_PAIR_CATEGORIES[1:]
    )
    if feature_names != expected_names:
        raise ApplicationDesignError(
            "application feature names differ from the shared fixed contract"
        )
    if x.shape != (n_edges, len(expected_names)) or not np.all(np.isfinite(x)):
        raise ApplicationDesignError(
            "application feature matrix has the wrong dimensions or non-finite values"
        )
    encoded = json.dumps(feature_names, ensure_ascii=True, separators=(",", ":"))
    feature_names_sha256 = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return frame, n_nodes, n_edges, len(feature_names), feature_names_sha256


def bind_application_design(
    design_path: Path = DEFAULT_DESIGN,
    manifest_path: Path = DEFAULT_MANIFEST,
    *,
    audit_path: Path | None = None,
    project_root: Path = PROJECT_ROOT,
    feature_contract_path: Path = FEATURE_CONTRACT_PATH,
) -> ApplicationDesignBinding:
    """Bind and validate the newest manifest-backed producer of this design.

    The newest completed ``continuous-design`` record that produced the exact
    requested design and audit files must fingerprint both, and its
    recorded builder/provenance code must still match the current files.
    Structural graph, audit, and feature-schema checks are then applied before
    dimensions are returned.  The audit defaults to the design's sibling
    ``audit_summary.json`` but can be supplied explicitly for a custom layout.
    """

    root = project_root.resolve()
    design = design_path.resolve()
    manifest = manifest_path.resolve()
    audit = (
        audit_path.resolve()
        if audit_path is not None
        else design.parent / "audit_summary.json"
    )
    producer, design_key, audit_key, design_fp, audit_fp = (
        _validate_manifest_binding(
            project_root=root,
            manifest_path=manifest,
            design_path=design,
            audit_path=audit,
        )
    )
    audit_payload = _load_audit(audit)
    _, n_nodes, n_edges, n_controls, feature_names_sha256 = (
        _load_and_validate_frame(design)
    )
    if audit_payload.get("eligible_firms") != n_nodes:
        raise ApplicationDesignError(
            "design audit eligible_firms does not match the graph node count"
        )
    if audit_payload.get("eligible_edges") != n_edges:
        raise ApplicationDesignError(
            "design audit eligible_edges does not match the graph edge count"
        )
    if audit_payload.get("numeric_design_all_finite") is not True:
        raise ApplicationDesignError(
            "design audit does not affirm finite numeric design values"
        )
    contract = feature_contract_path.resolve()
    contract_fp = _fingerprint(contract)
    return ApplicationDesignBinding(
        design_path=design,
        design_manifest_key=design_key,
        design_sha256=str(design_fp["sha256"]),
        design_size_bytes=int(design_fp["size_bytes"]),
        audit_path=audit,
        audit_manifest_key=audit_key,
        audit_sha256=str(audit_fp["sha256"]),
        audit_size_bytes=int(audit_fp["size_bytes"]),
        manifest_path=manifest,
        producer_run_id=str(producer["run_id"]),
        producer_recorded_at_utc=str(producer["recorded_at_utc"]),
        design_status=str(audit_payload["design_status"]),
        design_spec=str(audit_payload.get("design_spec", "unspecified")),
        design_role=str(
            audit_payload.get(
                "design_role",
                (
                    "primary"
                    if audit_payload["design_status"]
                    == "VALIDATED_FOR_ESTIMATION"
                    else "sensitivity"
                ),
            )
        ),
        is_primary_default=bool(
            audit_payload.get(
                "is_primary_default",
                audit_payload["design_status"] == "VALIDATED_FOR_ESTIMATION",
            )
        ),
        n_nodes=n_nodes,
        n_edges=n_edges,
        n_controls=n_controls,
        feature_names_sha256=feature_names_sha256,
        feature_contract_path=contract,
        feature_contract_sha256=str(contract_fp["sha256"]),
    )


def bind_design_consumer_output(
    design_binding: ApplicationDesignBinding,
    output_path: Path,
    *,
    run_prefix: str,
) -> DesignConsumerBinding:
    """Bind a current downstream output to the exact application design.

    This is used by multi-stage drivers such as the date sensitivity, which
    consumes both the return design and a DML result produced from that design.
    Every file recorded by the selected producer must still match its manifest
    fingerprint, so the check fails before expensive work begins.
    """

    if not run_prefix:
        raise ApplicationDesignError("consumer run prefix must be nonempty")
    project_root = design_binding.manifest_path.parent.parent.resolve()
    output = output_path.resolve()
    output_key = _display_path(output, project_root)
    output_fp = _fingerprint(output)
    expected_design = {
        "sha256": design_binding.design_sha256,
        "size_bytes": design_binding.design_size_bytes,
    }
    expected_audit = {
        "sha256": design_binding.audit_sha256,
        "size_bytes": design_binding.audit_size_bytes,
    }
    candidates: list[dict[str, Any]] = []
    for record in _read_manifest(design_binding.manifest_path):
        if record.get("status") != "completed" or not str(
            record.get("run_id", "")
        ).startswith(run_prefix):
            continue
        inputs = record.get("inputs")
        outputs = record.get("outputs")
        if not isinstance(inputs, dict) or not isinstance(outputs, dict):
            continue
        if outputs.get(output_key) != output_fp:
            continue
        if inputs.get(design_binding.design_manifest_key) != expected_design:
            continue
        if inputs.get(design_binding.audit_manifest_key) != expected_audit:
            continue
        candidates.append(record)
    if not candidates:
        raise ApplicationDesignError(
            f"no current {run_prefix} producer binds {output_key} to design "
            f"{design_binding.design_spec}"
        )
    latest_time = max(_record_time(record) for record in candidates)
    latest = [record for record in candidates if _record_time(record) == latest_time]
    if len(latest) != 1:
        raise ApplicationDesignError(
            f"ambiguous {run_prefix} producers share the latest timestamp"
        )
    producer = latest[0]
    _validate_all_recorded_files(producer, project_root=project_root)
    return DesignConsumerBinding(
        output_path=output,
        output_manifest_key=output_key,
        output_sha256=str(output_fp["sha256"]),
        output_size_bytes=int(output_fp["size_bytes"]),
        producer_run_id=str(producer["run_id"]),
        producer_recorded_at_utc=str(producer["recorded_at_utc"]),
        design_sha256=design_binding.design_sha256,
        design_audit_sha256=design_binding.audit_sha256,
    )


__all__ = [
    "ApplicationDesignBinding",
    "ApplicationDesignError",
    "DEFAULT_DESIGN",
    "DEFAULT_MANIFEST",
    "DesignConsumerBinding",
    "FEATURE_CONTRACT_PATH",
    "bind_application_design",
    "bind_design_consumer_output",
    "design_spec_run_token",
    "validate_driver_design_role",
]
