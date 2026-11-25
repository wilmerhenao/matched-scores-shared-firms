#!/usr/bin/env python3
"""Verify the integrity and documentation of the Matched Scores code release."""

from __future__ import annotations

import argparse
import ast
import hashlib
import subprocess
import sys
from pathlib import Path

from release_manifest import FILE_HASHES


ROOT = Path(__file__).resolve().parent
DOCUMENTED_TREES = ("src", "scripts", "experiments")


def sha256_file(path: Path) -> str:
    """Return the hexadecimal SHA-256 digest of one released file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_hashes() -> list[str]:
    """Return integrity failures for files recorded by the release builder."""

    failures: list[str] = []
    for relative, expected in sorted(FILE_HASHES.items()):
        path = ROOT / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
        elif sha256_file(path) != expected:
            failures.append(f"SHA-256 mismatch: {relative}")
    return failures


def _public_definition(node: ast.AST, parents: tuple[ast.AST, ...]) -> bool:
    """Identify top-level public callables and methods of public classes."""

    name = getattr(node, "name", "")
    if not name or name.startswith("_"):
        return False
    if not parents:
        return True
    return len(parents) == 1 and isinstance(parents[0], ast.ClassDef) and not (
        parents[0].name.startswith("_")
    )


def undocumented_public_definitions(path: Path) -> list[str]:
    """List reviewer-facing functions and classes that lack docstrings."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    failures: list[str] = []

    def visit(node: ast.AST, parents: tuple[ast.AST, ...] = ()) -> None:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if _public_definition(node, parents) and ast.get_docstring(node) is None:
                failures.append(f"{path.relative_to(ROOT)}:{node.lineno}:{node.name}")
            next_parents = (*parents, node)
        else:
            next_parents = parents
        for child in ast.iter_child_nodes(node):
            visit(child, next_parents)

    visit(tree)
    return failures


def verify_documentation() -> list[str]:
    """Return documentation failures across released production modules."""

    failures: list[str] = []
    for tree in DOCUMENTED_TREES:
        for path in sorted((ROOT / tree).rglob("*.py")):
            failures.extend(undocumented_public_definitions(path))
    for path in (ROOT / "reviewer_instructions.py", ROOT / "verify_release.py"):
        failures.extend(undocumented_public_definitions(path))
    return failures


def run_tests() -> int:
    """Run the released unit suite with the active Python interpreter."""

    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=ROOT,
        check=False,
    )
    return int(completed.returncode)


def main() -> int:
    """Verify hashes and documentation, and optionally execute unit tests."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests", action="store_true", help="also run pytest")
    args = parser.parse_args()

    failures = verify_hashes()
    failures.extend(
        f"undocumented public definition: {item}"
        for item in verify_documentation()
    )
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print(f"PASS: {len(FILE_HASHES)} released files match their SHA-256 digests")
    print("PASS: every reviewer-facing function and class is documented")
    if args.tests:
        return run_tests()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
