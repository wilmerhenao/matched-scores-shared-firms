#!/usr/bin/env python3
"""Fetch the small set of public, non-market inputs and build FF5 factors."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import polars as pl


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data_public/raw"
SOURCES = {
    "universe": {
        "url": "https://en.wikipedia.org/w/index.php?title=List_of_S%26P_500_companies&oldid=1192456039",
        "filename": "sp500_wikipedia_oldid_1192456039.html",
        "sha256": "78ef4c31ae2d3bd60c8aaebdd3d32342a48e83ced9a753fe3cb76d2f66129abb",
    },
    "dera": {
        "url": "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/2023q4_form13f.zip",
        "filename": "2023q4_form13f.zip",
        "sha256": "ea5e5a9a0124a3d068d723cff502dc33b4ecbf7d0bf1b1ddad9e216a815ed6d9",
    },
    "official_13f": {
        "url": "https://www.sec.gov/file/13flist2023q4",
        "filename": "sec_13f_list_2023q4.pdf",
        "sha256": "74708cba1b3e2a5dafc2736ad3590c110da6b746cac9e77d2c3ef8dcb75daf33",
    },
    "sec_company_tickers": {
        "url": "https://www.sec.gov/files/company_tickers.json",
        "filename": "sec_company_tickers_2026-08-08.json",
        "sha256": "6dd9c4363c5a95d43f4d8e8f8279f9ae6538d10d295bbdeebe5a433ec954bf6d",
    },
    "ff5": {
        "url": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
        "filename": "ff5_daily_csv.zip",
        "sha256": None,
    },
}


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a downloaded source file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path, user_agent: str) -> None:
    """Download one URL atomically unless the destination already exists."""

    if destination.exists():
        return
    partial = destination.with_suffix(destination.suffix + ".part")
    subprocess.run(
        [
            "curl",
            "-L",
            "--fail-with-body",
            "--retry",
            "3",
            "-A",
            user_agent,
            "-o",
            str(partial),
            url,
        ],
        check=True,
    )
    partial.replace(destination)


def parse_ff5(archive: Path, output: Path) -> None:
    """Extract daily Fama--French factors and write an arithmetic-return parquet."""

    with zipfile.ZipFile(archive) as bundle:
        member = next(name for name in bundle.namelist() if name.lower().endswith(".csv"))
        text = bundle.read(member).decode("latin-1")
    rows: list[dict[str, object]] = []
    for row in csv.reader(io.StringIO(text)):
        if len(row) < 7 or re.fullmatch(r"\d{8}", row[0].strip()) is None:
            continue
        values = [float(value.strip()) / 100.0 for value in row[1:7]]
        rows.append(
            dict(
                date=row[0].strip(),
                mkt_rf=values[0],
                smb=values[1],
                hml=values[2],
                rmw=values[3],
                cma=values[4],
                rf=values[5],
            )
        )
    factors = pl.DataFrame(rows).with_columns(pl.col("date").str.to_date("%Y%m%d"))
    if factors.height < 15_000:
        raise ValueError("Fama--French daily archive is implausibly short")
    factors.write_parquet(output)


def main() -> None:
    """Fetch, verify, and audit the small public-source bundle."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--sec-user-agent",
        required=True,
        help="SEC-compliant identity, e.g. 'Researcher name email@example.org'",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit: dict[str, object] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {},
    }
    for name, source in SOURCES.items():
        destination = args.output_dir / str(source["filename"])
        user_agent = (
            args.sec_user_agent if "sec.gov" in str(source["url"]) else "DyadicDML-Fin/1.0"
        )
        download(str(source["url"]), destination, user_agent)
        observed = sha256(destination)
        expected = source["sha256"]
        if expected is not None and observed != expected:
            raise RuntimeError(
                f"checksum mismatch for {name}: expected {expected}, observed {observed}"
            )
        audit["sources"][name] = {
            "url": source["url"],
            "filename": source["filename"],
            "sha256": observed,
            "size_bytes": destination.stat().st_size,
        }
    factor_output = args.output_dir / "ff5_daily.parquet"
    parse_ff5(args.output_dir / str(SOURCES["ff5"]["filename"]), factor_output)
    audit["ff5_parquet"] = {
        "filename": factor_output.name,
        "sha256": sha256(factor_output),
        "rows": pl.read_parquet(factor_output).height,
    }
    (args.output_dir / "public_source_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
