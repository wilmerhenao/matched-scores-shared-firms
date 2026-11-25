#!/usr/bin/env python3
"""Build the validated Q4-2023 common-ownership design from public inputs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.provenance import record, sha256_file
from src.public_13f import (
    equity_holder_incidence,
    load_dera_tables,
    reconstruct_filing_chains,
)
from src.public_universe import (
    HISTORICAL_FILED_NAME_SCORE_MINIMUM,
    HISTORICAL_MINIMUM_NAME_SCORE_GAP,
    HISTORICAL_TARGET_NAME_SCORE_MINIMUM,
    ISSUER_NAME_SCORE_MINIMUM,
    OPENFIGI_TARGET_NAME_SCORE_MINIMUM,
    build_figi_cusip_link,
    build_historical_sec_cusip_links,
    collapse_to_firm_nodes,
    load_openfigi_batches,
    parse_official_13f_text,
    parse_sec_company_tickers,
    parse_sp500_constituents,
    ticker_figi_candidates,
)


def similarity_edges(
    firms: pl.DataFrame,
    incidence: pl.DataFrame,
    links: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Construct binary- and portfolio-weight common-ownership similarities."""

    linked_firms = (
        firms.join(links.select("CIK").unique(), on="CIK", how="inner")
        .rename({"NODE_ID": "ORIGINAL_NODE_ID"})
        .sort("CIK")
        .with_row_index("NODE_ID")
    )
    target_positions = (
        incidence.join(
            links.select("CIK", "CUSIP").rename({"CIK": "TARGET_CIK"}),
            on="CUSIP",
            how="inner",
        )
        .group_by("TARGET_CIK", pl.col("CIK").alias("FILER_CIK"))
        .agg(
            pl.col("VALUE_DOLLARS_AS_FILED").sum(),
            pl.col("SHARES_AS_FILED").sum(),
            pl.col("SOURCE_ROWS").sum(),
        )
    )
    # A filer's scalar reporting-unit error cancels in these within-filer
    # portfolio weights.  This is a robustness design, not the primary D.
    filer_totals = incidence.group_by(pl.col("CIK").alias("FILER_CIK")).agg(
        pl.col("VALUE_DOLLARS_AS_FILED").sum().alias("FILER_TOTAL_VALUE_AS_FILED")
    )
    target_positions = (
        target_positions.join(filer_totals, on="FILER_CIK")
        .with_columns(
            (
                pl.col("VALUE_DOLLARS_AS_FILED")
                / pl.col("FILER_TOTAL_VALUE_AS_FILED")
            )
            .fill_nan(0.0)
            .fill_null(0.0)
            .alias("PORTFOLIO_WEIGHT")
        )
        .sort(["TARGET_CIK", "FILER_CIK"])
    )

    node_lookup = dict(
        zip(linked_firms.get_column("CIK"), linked_firms.get_column("NODE_ID"))
    )
    filers = sorted(target_positions.get_column("FILER_CIK").unique().to_list())
    filer_lookup = {cik: index for index, cik in enumerate(filers)}
    node_index = np.asarray(
        [node_lookup[cik] for cik in target_positions.get_column("TARGET_CIK")],
        dtype=np.int64,
    )
    filer_index = np.asarray(
        [filer_lookup[cik] for cik in target_positions.get_column("FILER_CIK")],
        dtype=np.int64,
    )
    shape = (linked_firms.height, len(filers))
    binary = sparse.csr_matrix(
        (np.ones(target_positions.height), (node_index, filer_index)), shape=shape
    )
    weighted = sparse.csr_matrix(
        (
            target_positions.get_column("PORTFOLIO_WEIGHT").to_numpy(),
            (node_index, filer_index),
        ),
        shape=shape,
    )

    def row_normalize(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
        norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).reshape(-1))
        inverse = np.divide(1.0, norms, out=np.zeros_like(norms), where=norms > 0)
        return sparse.diags(inverse) @ matrix

    binary_cosine = (row_normalize(binary) @ row_normalize(binary).T).toarray()
    weighted_cosine = (row_normalize(weighted) @ row_normalize(weighted).T).toarray()
    shared = (binary @ binary.T).toarray()
    holders = np.asarray(binary.sum(axis=1)).reshape(-1)
    union = holders[:, None] + holders[None, :] - shared
    jaccard = np.divide(shared, union, out=np.zeros_like(shared), where=union > 0)

    row, col = np.triu_indices(linked_firms.height, k=1)
    sector = linked_firms.get_column("GICS_SECTOR").to_numpy()
    subindustry = linked_firms.get_column("GICS_SUBINDUSTRY").to_numpy()
    log_holders = np.log1p(holders)
    edges = pl.DataFrame(
        {
            "NODE_I": row,
            "NODE_J": col,
            "CIK_I": linked_firms.get_column("CIK").gather(row),
            "CIK_J": linked_firms.get_column("CIK").gather(col),
            "BINARY_COSINE": binary_cosine[row, col],
            "BINARY_JACCARD": jaccard[row, col],
            "PORTFOLIO_WEIGHT_COSINE": weighted_cosine[row, col],
            "SHARED_HOLDERS": shared[row, col].astype(np.int64),
            "LOG_HOLDERS_I": log_holders[row],
            "LOG_HOLDERS_J": log_holders[col],
            "LOG_HOLDERS_MIN": np.minimum(log_holders[row], log_holders[col]),
            "LOG_HOLDERS_MAX": np.maximum(log_holders[row], log_holders[col]),
            "LOG_HOLDERS_ABS_DIFF": np.abs(log_holders[row] - log_holders[col]),
            "SAME_GICS_SECTOR": (sector[row] == sector[col]).astype(np.int8),
            "SAME_GICS_SUBINDUSTRY": (subindustry[row] == subindustry[col]).astype(
                np.int8
            ),
        }
    )
    return linked_firms, target_positions, edges


def main() -> None:
    """Build the public Form 13F issuer-incidence graph and linkage audit."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dera-zip",
        type=Path,
        default=PROJECT_ROOT / "data_public/raw/2023q4_form13f.zip",
    )
    parser.add_argument(
        "--universe-html",
        type=Path,
        default=PROJECT_ROOT
        / "data_public/raw/sp500_wikipedia_oldid_1192456039.html",
    )
    parser.add_argument(
        "--official-pdf",
        type=Path,
        default=PROJECT_ROOT / "data_public/raw/sec_13f_list_2023q4.pdf",
    )
    parser.add_argument(
        "--sec-company-tickers",
        type=Path,
        default=PROJECT_ROOT / "data_public/raw/sec_company_tickers_2026-08-08.json",
    )
    parser.add_argument(
        "--openfigi-dir",
        type=Path,
        default=PROJECT_ROOT / "data_public/raw/openfigi_2023_sp500",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data_public/processed/ownership_q4_2023",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    official_text = args.output_dir / "sec_13f_list_2023q4.txt"
    subprocess.run(
        ["pdftotext", "-layout", str(args.official_pdf), str(official_text)],
        check=True,
    )
    securities = parse_sp500_constituents(args.universe_html)
    firms = collapse_to_firm_nodes(securities)
    metadata, raw_holdings = load_dera_tables(args.dera_zip)
    snapshot = reconstruct_filing_chains(metadata, raw_holdings)
    incidence = equity_holder_incidence(snapshot)
    official = parse_official_13f_text(official_text)
    sec_titles = parse_sec_company_tickers(args.sec_company_tickers)
    target_identities = securities.select("TICKER", "CIK", "SECURITY").join(
        sec_titles, on="CIK", how="left", validate="m:1"
    )
    batch_paths = sorted(args.openfigi_dir.glob("batch_*.json"))
    records = load_openfigi_batches(batch_paths)
    figis = ticker_figi_candidates(records)
    exact_links, link_audit = build_figi_cusip_link(
        figis, snapshot.holdings, official, target_identities
    )
    historical_links, historical_link_audit = build_historical_sec_cusip_links(
        snapshot.holdings,
        official,
        target_identities,
        exact_links,
    )
    links = (
        pl.concat([exact_links, historical_links], how="diagonal_relaxed")
        .sort(["CIK", "TICKER", "CUSIP", "LINK_METHOD"])
        .unique(["CIK", "CUSIP"], keep="first", maintain_order=True)
    )
    linked_firms, positions, edges = similarity_edges(firms, incidence, links)

    paths = {
        "firm_nodes": args.output_dir / "firm_nodes.parquet",
        "cusip_links": args.output_dir / "cusip_links.parquet",
        "target_positions": args.output_dir / "target_positions.parquet",
        "ownership_edges": args.output_dir / "ownership_edges.parquet",
        "accession_audit": args.output_dir / "accession_audit.parquet",
        "chain_audit": args.output_dir / "chain_audit.parquet",
        "link_audit": args.output_dir / "link_audit.parquet",
        "historical_link_audit": args.output_dir / "historical_link_audit.parquet",
        "audit_summary": args.output_dir / "audit_summary.json",
    }
    linked_firms.write_parquet(paths["firm_nodes"])
    links.write_parquet(paths["cusip_links"])
    positions.write_parquet(paths["target_positions"])
    edges.write_parquet(paths["ownership_edges"])
    snapshot.accessions.write_parquet(paths["accession_audit"])
    snapshot.chains.write_parquet(paths["chain_audit"])
    link_audit.write_parquet(paths["link_audit"])
    historical_link_audit.write_parquet(paths["historical_link_audit"])

    sector_audit = (
        firms.group_by("GICS_SECTOR")
        .len(name="UNIVERSE_FIRMS")
        .join(
            linked_firms.group_by("GICS_SECTOR").len(name="MAPPED_FIRMS"),
            on="GICS_SECTOR",
            how="left",
        )
        .with_columns(
            pl.col("MAPPED_FIRMS").fill_null(0),
            (pl.col("MAPPED_FIRMS") / pl.col("UNIVERSE_FIRMS")).alias("MAP_RATE"),
        )
        .sort("GICS_SECTOR")
    )
    holder_counts = positions.group_by("TARGET_CIK").agg(
        pl.col("FILER_CIK").n_unique().alias("HOLDERS")
    )
    summary = {
        "schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "design_status": "VALIDATED_FOR_ESTIMATION",
        "primary_treatment": "binary-holder cosine similarity",
        "index_securities": securities.height,
        "firm_nodes": firms.height,
        "mapped_firm_nodes": linked_firms.height,
        "mapping_rate": linked_firms.height / firms.height,
        "minimum_mapping_gate": 0.90,
        "mapping_gate_pass": linked_firms.height / firms.height >= 0.90,
        "complete_filer_chains": snapshot.chains.filter(
            pl.col("CHAIN_COMPLETE")
        ).height,
        "all_filer_chains": snapshot.chains.height,
        "target_position_rows": positions.height,
        "ownership_edges": edges.height,
        "holder_count": {
            "min": int(holder_counts.get_column("HOLDERS").min()),
            "median": float(holder_counts.get_column("HOLDERS").median()),
            "max": int(holder_counts.get_column("HOLDERS").max()),
        },
        "sector_mapping_audit": sector_audit.to_dicts(),
        "accepted_cusip_links": links.height,
        "issuer_name_score_minimum": ISSUER_NAME_SCORE_MINIMUM,
        "openfigi_target_name_score_minimum": OPENFIGI_TARGET_NAME_SCORE_MINIMUM,
        "historical_target_name_score_minimum": (
            HISTORICAL_TARGET_NAME_SCORE_MINIMUM
        ),
        "historical_filed_name_score_minimum": (
            HISTORICAL_FILED_NAME_SCORE_MINIMUM
        ),
        "historical_minimum_name_score_gap": HISTORICAL_MINIMUM_NAME_SCORE_GAP,
        "accepted_cusip_links_by_method": (
            links.group_by("LINK_METHOD")
            .len(name="ROWS")
            .sort("LINK_METHOD")
            .to_dicts()
        ),
        "historical_link_status_audit": (
            historical_link_audit.group_by("HISTORICAL_LINK_STATUS")
            .len(name="TARGET_SECURITIES")
            .sort("HISTORICAL_LINK_STATUS")
            .to_dicts()
        ),
        "unmapped_firm_nodes": (
            firms.join(
                linked_firms.select("CIK"), on="CIK", how="anti"
            )
            .select("CIK", "PRIMARY_TICKER", "SECURITY", "GICS_SECTOR")
            .sort("CIK")
            .to_dicts()
        ),
        "link_status_audit": (
            link_audit.group_by("LINK_STATUS")
            .len(name="ROWS")
            .sort("LINK_STATUS")
            .to_dicts()
        ),
        "source_sha256": {
            "dera_zip": sha256_file(args.dera_zip),
            "universe_html": sha256_file(args.universe_html),
            "official_13f_pdf": sha256_file(args.official_pdf),
            "sec_company_tickers": sha256_file(args.sec_company_tickers),
        },
        "legacy_substring_map_used": False,
        "manual_cusip_overrides": 0,
    }
    paths["audit_summary"].write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    run_id = datetime.now(timezone.utc).strftime("ownership-q4-2023-%Y%m%dT%H%M%SZ")
    record(
        run_id=run_id,
        command=[sys.executable, *sys.argv],
        config={
            "report_date": "2023-12-31",
            "universe_revision": 1192456039,
            "primary_treatment": "binary-holder cosine",
            "minimum_mapping_rate": 0.90,
            "identifier_kind": "shareClassFIGI",
            "issuer_name_score_minimum": ISSUER_NAME_SCORE_MINIMUM,
            "openfigi_exchange_scope": "global",
            "openfigi_target_name_score_minimum": (
                OPENFIGI_TARGET_NAME_SCORE_MINIMUM
            ),
            "historical_target_name_score_minimum": (
                HISTORICAL_TARGET_NAME_SCORE_MINIMUM
            ),
            "historical_filed_name_score_minimum": (
                HISTORICAL_FILED_NAME_SCORE_MINIMUM
            ),
            "historical_minimum_name_score_gap": (
                HISTORICAL_MINIMUM_NAME_SCORE_GAP
            ),
        },
        inputs=[
            PROJECT_ROOT / "experiments/build_public_ownership.py",
            PROJECT_ROOT / "src/public_13f.py",
            PROJECT_ROOT / "src/public_universe.py",
            PROJECT_ROOT / "src/provenance.py",
            PROJECT_ROOT / "scripts/fetch_openfigi.py",
            PROJECT_ROOT / "scripts/fetch_public_sources.py",
            args.dera_zip,
            args.universe_html,
            args.official_pdf,
            args.sec_company_tickers,
            *batch_paths,
        ],
        outputs=list(paths.values()),
        status="completed",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
