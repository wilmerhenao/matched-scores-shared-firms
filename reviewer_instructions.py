#!/usr/bin/env python3
"""Print and prepare the source-only Matched Scores replication environment.

The release repository intentionally contains Python and Markdown files only.
This helper materializes the two JSON source contracts and ``pyproject.toml``
that the pipeline expects, then prints the complete command sequence.  It does
not download data, run experiments, or modify any frozen result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
PYPROJECT_TEXT = """[project]
name = "matched-scores-shared-firms"
version = "0.1.0"
description = "Source-only replication code for Matched Scores, Shared Firms"
requires-python = ">=3.11"
dependencies = [
  "beautifulsoup4==4.14.3",
  "certifi==2026.5.20",
  "matplotlib==3.10.8",
  "numpy==2.4.2",
  "pandas==2.2.3",
  "polars==1.39.3",
  "pyarrow==23.0.1",
  "scikit-learn==1.8.0",
  "scipy==1.17.1",
  "threadpoolctl==3.6.0",
  "yfinance==1.2.0",
]

[project.optional-dependencies]
test = ["pytest==9.0.2"]

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
"""

INSTRUCTIONS = r"""
MATCHED SCORES: SOURCE-ONLY REPLICATION
================================

Run commands from the checkout root. Unit tests and the canonical simulation
use synthetic inputs. The empirical workflow, node-share simulation, and
repeated-split simulation require validated application inputs.

EMPIRICAL REPLAY LIMITATION: the historical Yahoo baseline and reference
batches are not distributed, and the commands below do not recreate both
original acquisition sequences. Read README.md before the price-panel step.

1. Materialize the generated support files and create the environment:

   python3 reviewer_instructions.py --prepare
   python3.13 -m venv .venv
   .venv/bin/pip install -e '.[test]'
   .venv/bin/python -B verify_release.py --tests

2. Fetch the small public inputs. The SEC requires a descriptive user agent:

   .venv/bin/python scripts/fetch_public_sources.py \
     --sec-user-agent 'Researcher Name researcher@example.org'
   .venv/bin/python scripts/fetch_openfigi.py \
     --universe-html data_public/raw/sp500_wikipedia_oldid_1192456039.html \
     --output-dir data_public/raw/openfigi_2023_sp500
   .venv/bin/python experiments/build_public_ownership.py

3. Reconstruct the public price inputs. Configure the Kaggle CLI first:

   .venv/bin/python scripts/fetch_kaggle_snapshots.py
   .venv/bin/python scripts/fetch_stockanalysis_fallback.py \
     --tickers CMA CTRA \
     --output-dir data_public/raw/stockanalysis_fallback_2023_2024
   .venv/bin/python scripts/fetch_stockanalysis_fallback.py \
     --tickers CTLT \
     --output-dir data_public/raw/stockanalysis_ctlt_2023_2024
   .venv/bin/python scripts/fetch_stockanalysis_fallback.py \
     --tickers XOM \
     --output-dir data_public/raw/stockanalysis_xom_2023_2024

   The immutable panel also requires historical Yahoo baseline and reference
   batches under data_public/raw/yfinance_2023_2024_v2/ and
   data_public/raw/yfinance_2023_2024/. These are not supplied by this release.
   The original baseline requested 473 tickers, with 26 separately specified
   recoveries. Downloading the full 499-firm universe is not an equivalent
   replay. Restore verified baseline inputs or validate a reconstruction of
   their original acquisition before proceeding. The downloader's flags are
   available with `python scripts/fetch_yfinance_prices.py --help`.
   Public market-data endpoints are mutable, so a fresh run need not match
   the submission vintage; do not disable failed provenance/coverage gates.

4. Build the primary design and empirical results:

   .venv/bin/python scripts/build_immutable_price_panel.py
   .venv/bin/python experiments/build_return_design.py
   .venv/bin/python experiments/run_continuous_dml.py
   .venv/bin/python experiments/run_full_graph_ols_diagnostic.py
   .venv/bin/python experiments/run_ridge_equal_count_audit.py --workers 4
   .venv/bin/python experiments/run_hgb_learner_audit.py --workers 4
   .venv/bin/python experiments/run_date_block_bootstrap.py --workers 4

5. Run truth-known calibration experiments (canonical is data-free; the
   other two require the validated application design from step 4):

   .venv/bin/python experiments/run_canonical_simulation.py \
     --replications-per-cell 1000 --workers 8
   .venv/bin/python experiments/run_node_share_simulation.py \
     --replications 1000 --workers 8
   .venv/bin/python experiments/run_repeated_split_calibration.py \
     --replications 500 --workers 8

The canonical simulation is the longest local job. Reduce worker counts if
memory is constrained. Every production command writes an append-only record
to results/MANIFEST.jsonl containing its configuration and file fingerprints.
""".strip()


def _write_exact(path: Path, content: str) -> None:
    """Create one support file or verify that an existing copy is identical."""

    payload = content.encode("utf-8")
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise RuntimeError(f"refusing to overwrite a differing support file: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _json_text(value: dict[str, Any]) -> str:
    """Serialize a generated source contract in stable, readable JSON form."""

    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def prepare() -> None:
    """Materialize dependency metadata and source contracts from Python data."""

    from support.source_contracts import (
        KAGGLE_SNAPSHOT_MANIFEST,
        PRICE_ALIAS_ACTIONS_MANIFEST,
    )

    _write_exact(ROOT / "pyproject.toml", PYPROJECT_TEXT)
    _write_exact(
        ROOT / "source_manifests/kaggle_sp500_snapshots.json",
        _json_text(KAGGLE_SNAPSHOT_MANIFEST),
    )
    _write_exact(
        ROOT / "source_manifests/price_alias_actions.json",
        _json_text(PRICE_ALIAS_ACTIONS_MANIFEST),
    )


def main() -> int:
    """Parse the optional preparation flag and print the replication protocol."""

    parser = argparse.ArgumentParser(
        description="Print the Matched Scores source-only replication instructions."
    )
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="materialize pyproject.toml and the two JSON source contracts",
    )
    args = parser.parse_args()
    if args.prepare:
        prepare()
        print("Prepared pyproject.toml and source_manifests/*.json.\n")
    print(INSTRUCTIONS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
