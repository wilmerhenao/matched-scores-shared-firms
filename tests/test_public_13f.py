from __future__ import annotations

from datetime import date

import polars as pl

from src.public_13f import (
    audit_accessions,
    equity_holder_incidence,
    reconstruct_filing_chains,
)


def synthetic_filings() -> tuple[pl.DataFrame, pl.DataFrame]:
    metadata = pl.DataFrame(
        {
            "ACCESSION_NUMBER": ["O1", "N1", "O2", "R2", "N2", "N3", "O4"],
            "FILING_DATE": [
                date(2024, 1, 10),
                date(2024, 2, 10),
                date(2024, 1, 11),
                date(2024, 2, 1),
                date(2024, 2, 11),
                date(2024, 2, 12),
                date(2024, 1, 12),
            ],
            "SUBMISSIONTYPE": [
                "13F-HR",
                "13F-HR/A",
                "13F-HR",
                "13F-HR/A",
                "13F-HR/A",
                "13F-HR/A",
                "13F-HR",
            ],
            "CIK": ["0000000001", "0000000001", "0000000002", "0000000002", "0000000002", "0000000003", "0000000004"],
            "PERIODOFREPORT": ["31-DEC-2023"] * 7,
            "ISAMENDMENT": ["N", "Y", "N", "Y", "Y", "Y", "N"],
            "AMENDMENTNO": [None, 1, None, 1, 2, 1, None],
            "AMENDMENTTYPE": [
                None,
                "NEW HOLDINGS",
                None,
                "RESTATEMENT",
                "NEW HOLDINGS",
                "NEW HOLDINGS",
                None,
            ],
            "FILINGMANAGER_NAME": ["A", "A", "B", "B", "B", "C", "D"],
            "TABLEENTRYTOTAL": [2, 1, 1, 2, 1, 1, 2],
            "TABLEVALUETOTAL": [30, 30, 40, 110, 60, 70, 80],
            "ISCONFIDENTIALOMITTED": ["N"] * 7,
        }
    )
    accessions = ["O1", "O1", "N1", "O2", "R2", "R2", "N2", "N3", "O4"]
    holdings = pl.DataFrame(
        {
            "ACCESSION_NUMBER": accessions,
            "NAMEOFISSUER": ["X"] * len(accessions),
            "TITLEOFCLASS": ["COM"] * len(accessions),
            "CUSIP": [
                "000000001",
                "000000002",
                "000000003",
                "000000004",
                "000000005",
                "000000006",
                "000000007",
                "000000008",
                "000000009",
            ],
            "FIGI": [None] * len(accessions),
            "VALUE": [10, 20, 30, 40, 50, 60, 60, 70, 80],
            "SSHPRNAMT": [1.0] * len(accessions),
            "SSHPRNAMTTYPE": ["SH"] * len(accessions),
            "PUTCALL": [None] * len(accessions),
        }
    )
    return metadata, holdings


def test_accession_audit_detects_entry_and_value_mismatch():
    metadata, holdings = synthetic_filings()
    audit = audit_accessions(metadata, holdings)
    o4 = audit.filter(pl.col("ACCESSION_NUMBER") == "O4").row(0, named=True)
    assert not o4["ENTRY_TOTAL_MATCH"]
    assert o4["VALUE_TOTAL_MATCH"]
    assert o4["VALUE_AUDIT_CLASS"] == "EXACT"


def test_reconstruction_appends_new_holdings_and_replaces_restatement():
    metadata, holdings = synthetic_filings()
    snapshot = reconstruct_filing_chains(metadata, holdings)
    by_cik = (
        snapshot.holdings.group_by("CIK")
        .agg(pl.col("ACCESSION_NUMBER").unique().sort())
        .sort("CIK")
    )
    assert by_cik.to_dicts() == [
        {"CIK": "0000000001", "ACCESSION_NUMBER": ["N1", "O1"]},
        {"CIK": "0000000002", "ACCESSION_NUMBER": ["N2", "R2"]},
    ]
    incomplete = snapshot.chains.filter(~pl.col("CHAIN_COMPLETE"))
    assert set(incomplete.get_column("CHAIN_REASON")) == {
        "ORPHAN_NEW_HOLDINGS",
        "ENTRY_TOTAL_MISMATCH",
    }


def test_incidence_aggregates_duplicate_filer_cusip_rows():
    metadata, holdings = synthetic_filings()
    duplicate = holdings.filter(pl.col("ACCESSION_NUMBER") == "O1").head(1)
    holdings = pl.concat([holdings, duplicate])
    metadata = metadata.with_columns(
        pl.when(pl.col("ACCESSION_NUMBER") == "O1")
        .then(pl.lit(3))
        .otherwise(pl.col("TABLEENTRYTOTAL"))
        .alias("TABLEENTRYTOTAL"),
        pl.when(pl.col("ACCESSION_NUMBER") == "O1")
        .then(pl.lit(40))
        .otherwise(pl.col("TABLEVALUETOTAL"))
        .alias("TABLEVALUETOTAL"),
    )
    snapshot = reconstruct_filing_chains(metadata, holdings)
    incidence = equity_holder_incidence(snapshot)
    row = incidence.filter(
        (pl.col("CIK") == "0000000001") & (pl.col("CUSIP") == "000000001")
    ).row(0, named=True)
    assert row["HOLDS"] == 1
    assert row["SOURCE_ROWS"] == 2
    assert row["VALUE_DOLLARS_AS_FILED"] == 20


def test_incidence_excludes_only_zero_share_zero_value_rows():
    metadata, holdings = synthetic_filings()
    zero_rows = pl.DataFrame(
        {
            "ACCESSION_NUMBER": ["O1", "O1", "O1"],
            "NAMEOFISSUER": ["ZERO", "VALUE ONLY", "SHARES ONLY"],
            "TITLEOFCLASS": ["COM", "COM", "COM"],
            "CUSIP": ["000000010", "000000011", "000000012"],
            "FIGI": [None, None, None],
            "VALUE": [0, 5, 0],
            "SSHPRNAMT": [0.0, 0.0, 2.0],
            "SSHPRNAMTTYPE": ["SH", "SH", "SH"],
            "PUTCALL": [None, None, None],
        }
    )
    holdings = pl.concat([holdings, zero_rows])
    metadata = metadata.with_columns(
        pl.when(pl.col("ACCESSION_NUMBER") == "O1")
        .then(pl.lit(5))
        .otherwise(pl.col("TABLEENTRYTOTAL"))
        .alias("TABLEENTRYTOTAL"),
        pl.when(pl.col("ACCESSION_NUMBER") == "O1")
        .then(pl.lit(35))
        .otherwise(pl.col("TABLEVALUETOTAL"))
        .alias("TABLEVALUETOTAL"),
    )

    incidence = equity_holder_incidence(
        reconstruct_filing_chains(metadata, holdings)
    ).filter(pl.col("CIK") == "0000000001")

    assert "000000010" not in incidence.get_column("CUSIP").to_list()
    assert {"000000011", "000000012"}.issubset(
        set(incidence.get_column("CUSIP").to_list())
    )


def test_valid_restatement_repairs_an_incomplete_original():
    metadata, holdings = synthetic_filings()
    metadata = metadata.with_columns(
        pl.when(pl.col("ACCESSION_NUMBER") == "O2")
        .then(pl.lit(2))
        .otherwise(pl.col("TABLEENTRYTOTAL"))
        .alias("TABLEENTRYTOTAL")
    )
    snapshot = reconstruct_filing_chains(metadata, holdings)
    chain = snapshot.chains.filter(pl.col("CIK") == "0000000002").row(0, named=True)
    assert chain["CHAIN_COMPLETE"]
    assert chain["CHAIN_REASON"] == "OK"
    assert set(chain["ACTIVE_ACCESSIONS"]) == {"R2", "N2"}
