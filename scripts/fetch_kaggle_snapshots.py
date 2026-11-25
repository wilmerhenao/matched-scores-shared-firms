#!/usr/bin/env python3
"""Download and verify the exact Kaggle price snapshots used by the panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import BinaryIO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "source_manifests/kaggle_sp500_snapshots.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data_public/raw/kaggle_sp500_snapshots"


def sha256_stream(handle: BinaryIO) -> tuple[str, int]:
    """Return the digest and byte count of an open binary stream."""

    digest = hashlib.sha256()
    size = 0
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
        size += len(block)
    return digest.hexdigest(), size


def sha256_file(path: Path) -> tuple[str, int]:
    """Return the digest and byte count of one regular file."""

    with path.open("rb") as handle:
        return sha256_stream(handle)


def deterministic_gzip(source: Path, destination: Path) -> None:
    """Write the frozen ``gzip -n -9`` transport representation."""

    partial = destination.with_suffix(destination.suffix + ".part")
    with partial.open("wb") as output_handle:
        subprocess.run(
            ["gzip", "-n", "-9", "-c", str(source)],
            stdout=output_handle,
            check=True,
        )
    partial.replace(destination)


def load_manifest(path: Path) -> list[dict[str, object]]:
    """Load and minimally validate the generated snapshot source contract."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("files")
    if payload.get("schema_version") != 1 or not isinstance(records, list):
        raise RuntimeError("invalid Kaggle snapshot manifest")
    return records


def fetch_snapshot(record: dict[str, object], output_dir: Path) -> Path:
    """Download, compress, and verify one versioned Kaggle CSV snapshot."""

    version = int(record["version"])
    dataset_slug = str(record["dataset_slug"])
    destination = output_dir / str(record["canonical_file_name"])
    if destination.is_file():
        observed_hash, observed_size = sha256_file(destination)
        if (
            observed_hash == record["compressed_sha256"]
            and observed_size == int(record["compressed_size_bytes"])
        ):
            return destination
        raise RuntimeError(f"existing snapshot fails its checksum contract: {destination}")

    with tempfile.TemporaryDirectory(prefix=f"snapshot-v{version}-") as temporary:
        temporary_path = Path(temporary)
        subprocess.run(
            [
                "kaggle",
                "datasets",
                "download",
                f"{dataset_slug}/{version}",
                "--path",
                str(temporary_path),
                "--unzip",
            ],
            check=True,
        )
        candidates = list(temporary_path.rglob(str(record["upstream_file_name"])))
        if len(candidates) != 1:
            raise RuntimeError(
                f"expected one upstream CSV for Kaggle version {version}; found {candidates}"
            )
        upstream_hash, upstream_size = sha256_file(candidates[0])
        if (
            upstream_hash != record["upstream_sha256"]
            or upstream_size != int(record["upstream_size_bytes"])
        ):
            raise RuntimeError(f"upstream Kaggle payload changed for version {version}")
        deterministic_gzip(candidates[0], destination)

    compressed_hash, compressed_size = sha256_file(destination)
    if (
        compressed_hash != record["compressed_sha256"]
        or compressed_size != int(record["compressed_size_bytes"])
    ):
        raise RuntimeError(
            "deterministic gzip output differs from the frozen transport hash; "
            "the decompressed payload was verified, so retain the CSV and use a "
            "platform gzip compatible with RFC 1952 level-nine output"
        )
    return destination


def main() -> int:
    """Download all requested snapshot versions and print verified paths."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--versions", type=int, nargs="*", default=None)
    args = parser.parse_args()
    records = load_manifest(args.manifest)
    requested = set(args.versions) if args.versions else {int(row["version"]) for row in records}
    selected = [row for row in records if int(row["version"]) in requested]
    if {int(row["version"]) for row in selected} != requested:
        raise ValueError("one or more requested versions are absent from the manifest")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for record in selected:
        print(fetch_snapshot(record, args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
