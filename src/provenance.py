"""Small, dependency-free provenance recorder for validated experiments."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = PROJECT_ROOT / "results" / "MANIFEST.jsonl"


def sha256_file(path: str | Path) -> str:
    """Return the hexadecimal SHA-256 digest of one regular file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(paths: Iterable[str | Path]) -> dict[str, dict[str, Any]]:
    """Fingerprint paths with project-relative names, sizes, and SHA-256 digests."""

    result: dict[str, dict[str, Any]] = {}
    for raw_path in paths:
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        # Keep anonymous release manifests machine-independent.  Inputs from
        # sibling repositories remain explicit ``../`` relative paths without
        # embedding a workstation username.
        display_path = os.path.relpath(path, PROJECT_ROOT)
        result[display_path] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    return result


def record(
    *,
    run_id: str,
    command: list[str],
    config: dict[str, Any],
    inputs: Iterable[str | Path],
    outputs: Iterable[str | Path],
    status: str,
    manifest_path: str | Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    """Append one self-contained JSON record and return it."""

    output_paths = [Path(path) for path in outputs]
    if status == "completed":
        missing = [str(path) for path in output_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"completed run is missing outputs: {missing}")

    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "command": command,
        "config": config,
        "inputs": fingerprint(inputs),
        "outputs": fingerprint(output_paths) if status == "completed" else {},
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "executable": sys.executable,
            "aws_job_name": os.environ.get("TRAINING_JOB_NAME"),
        },
    }
    destination = Path(manifest_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
    return payload


__all__ = ["fingerprint", "record", "sha256_file"]
