"""Auditable reconstruction of a public SEC Form 13F quarterly snapshot.

The SEC structured-data ZIP stores every filing accession separately.  A
RESTATEMENT replaces the filing it amends, whereas a NEW HOLDINGS amendment
adds only the omitted holdings.  Treating the latest accession as a complete
replacement therefore deletes valid positions.  This module makes the chain
semantics explicit and retains an accession-level audit table.

The post-January-2023 ``VALUE`` field is reported in dollars.  Some filers
nevertheless submit inconsistent summary totals.  The primary ownership
design uses binary holder incidence, so it never rescales position values or
silently guesses a unit correction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import polars as pl


REPORT_DATE_Q4_2023 = "31-DEC-2023"
EVENT_ORIGINAL = "ORIGINAL"
EVENT_RESTATEMENT = "RESTATEMENT"
EVENT_NEW_HOLDINGS = "NEW HOLDINGS"


@dataclass(frozen=True)
class FilingSnapshot:
    """Reconstructed filing data and the audit records used to build it."""

    holdings: pl.DataFrame
    chains: pl.DataFrame
    accessions: pl.DataFrame


def _read_tsv(
    archive: ZipFile,
    name: str,
    *,
    columns: list[str],
    schema_overrides: dict[str, pl.DataType] | None = None,
) -> pl.DataFrame:
    with archive.open(name) as handle:
        return pl.read_csv(
            handle,
            separator="\t",
            columns=columns,
            schema_overrides=schema_overrides,
            null_values=[""],
            infer_schema_length=10_000,
        )


def load_dera_tables(
    zip_path: str | Path,
    *,
    report_date: str = REPORT_DATE_Q4_2023,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load filing metadata and holdings for one report date from an SEC ZIP."""

    with ZipFile(zip_path) as archive:
        submission = _read_tsv(
            archive,
            "SUBMISSION.tsv",
            columns=[
                "ACCESSION_NUMBER",
                "FILING_DATE",
                "SUBMISSIONTYPE",
                "CIK",
                "PERIODOFREPORT",
            ],
            schema_overrides={
                "ACCESSION_NUMBER": pl.String,
                "FILING_DATE": pl.String,
                "SUBMISSIONTYPE": pl.String,
                "CIK": pl.String,
                "PERIODOFREPORT": pl.String,
            },
        )
        cover = _read_tsv(
            archive,
            "COVERPAGE.tsv",
            columns=[
                "ACCESSION_NUMBER",
                "ISAMENDMENT",
                "AMENDMENTNO",
                "AMENDMENTTYPE",
                "FILINGMANAGER_NAME",
            ],
            schema_overrides={
                "ACCESSION_NUMBER": pl.String,
                "ISAMENDMENT": pl.String,
                "AMENDMENTNO": pl.Int64,
                "AMENDMENTTYPE": pl.String,
                "FILINGMANAGER_NAME": pl.String,
            },
        )
        summary = _read_tsv(
            archive,
            "SUMMARYPAGE.tsv",
            columns=[
                "ACCESSION_NUMBER",
                "TABLEENTRYTOTAL",
                "TABLEVALUETOTAL",
                "ISCONFIDENTIALOMITTED",
            ],
            schema_overrides={
                "ACCESSION_NUMBER": pl.String,
                "TABLEENTRYTOTAL": pl.Int64,
                "TABLEVALUETOTAL": pl.Int64,
                "ISCONFIDENTIALOMITTED": pl.String,
            },
        )
        holdings = _read_tsv(
            archive,
            "INFOTABLE.tsv",
            columns=[
                "ACCESSION_NUMBER",
                "NAMEOFISSUER",
                "TITLEOFCLASS",
                "CUSIP",
                "FIGI",
                "VALUE",
                "SSHPRNAMT",
                "SSHPRNAMTTYPE",
                "PUTCALL",
            ],
            schema_overrides={
                "ACCESSION_NUMBER": pl.String,
                "NAMEOFISSUER": pl.String,
                "TITLEOFCLASS": pl.String,
                "CUSIP": pl.String,
                "FIGI": pl.String,
                "VALUE": pl.Int64,
                "SSHPRNAMT": pl.Float64,
                "SSHPRNAMTTYPE": pl.String,
                "PUTCALL": pl.String,
            },
        )

    metadata = (
        submission.filter(
            (pl.col("PERIODOFREPORT") == report_date)
            & pl.col("SUBMISSIONTYPE").is_in(["13F-HR", "13F-HR/A"])
        )
        .join(cover, on="ACCESSION_NUMBER", how="left", validate="1:1")
        .join(summary, on="ACCESSION_NUMBER", how="left", validate="1:1")
        .with_columns(
            pl.col("FILING_DATE").str.to_date("%d-%b-%Y", strict=True),
            pl.col("CIK").str.strip_chars().str.zfill(10),
        )
    )
    accession_ids = metadata.get_column("ACCESSION_NUMBER")
    holdings = holdings.filter(pl.col("ACCESSION_NUMBER").is_in(accession_ids)).with_columns(
        pl.col("CUSIP").str.strip_chars().str.to_uppercase()
    )
    return metadata, holdings


def audit_accessions(metadata: pl.DataFrame, holdings: pl.DataFrame) -> pl.DataFrame:
    """Compare every accession's item rows and values with its SEC summary."""

    aggregate = holdings.group_by("ACCESSION_NUMBER").agg(
        pl.len().alias("OBSERVED_ENTRY_TOTAL"),
        pl.col("VALUE").sum().alias("OBSERVED_VALUE_TOTAL"),
    )
    audited = metadata.join(aggregate, on="ACCESSION_NUMBER", how="left").with_columns(
        pl.col("OBSERVED_ENTRY_TOTAL").fill_null(0),
        pl.col("OBSERVED_VALUE_TOTAL").fill_null(0),
    )
    return audited.with_columns(
        (pl.col("TABLEENTRYTOTAL") == pl.col("OBSERVED_ENTRY_TOTAL")).alias(
            "ENTRY_TOTAL_MATCH"
        ),
        (pl.col("TABLEVALUETOTAL") == pl.col("OBSERVED_VALUE_TOTAL")).alias(
            "VALUE_TOTAL_MATCH"
        ),
        pl.when(pl.col("TABLEVALUETOTAL") == pl.col("OBSERVED_VALUE_TOTAL"))
        .then(pl.lit("EXACT"))
        .when(
            (pl.col("TABLEVALUETOTAL") - 1000 * pl.col("OBSERVED_VALUE_TOTAL"))
            .abs()
            <= 1
        )
        .then(pl.lit("SUMMARY_X1000"))
        .when(
            (1000 * pl.col("TABLEVALUETOTAL") - pl.col("OBSERVED_VALUE_TOTAL"))
            .abs()
            <= 1
        )
        .then(pl.lit("ITEMS_X1000"))
        .otherwise(pl.lit("MISMATCH"))
        .alias("VALUE_AUDIT_CLASS"),
    )


def _event_type(row: dict[str, object]) -> str:
    amendment_type = str(row.get("AMENDMENTTYPE") or "").strip().upper()
    if amendment_type in {EVENT_RESTATEMENT, EVENT_NEW_HOLDINGS}:
        return amendment_type
    submission_type = str(row.get("SUBMISSIONTYPE") or "").strip().upper()
    is_amendment = str(row.get("ISAMENDMENT") or "").strip().upper()
    if submission_type == "13F-HR" and is_amendment != "Y":
        return EVENT_ORIGINAL
    return "UNKNOWN"


def reconstruct_filing_chains(
    metadata: pl.DataFrame,
    holdings: pl.DataFrame,
    *,
    require_entry_match: bool = True,
) -> FilingSnapshot:
    """Apply original/restatement/new-holdings semantics by filer and period.

    Accessions whose information-table row count disagrees with the SEC
    summary are excluded by default.  Value-total mismatches are recorded but
    do not invalidate the binary-incidence design.
    """

    audited = audit_accessions(metadata, holdings).with_columns(
        pl.struct(pl.all()).map_elements(
            _event_type, return_dtype=pl.String
        ).alias("EVENT_TYPE")
    )
    rows = audited.sort(
        ["CIK", "PERIODOFREPORT", "FILING_DATE", "AMENDMENTNO", "ACCESSION_NUMBER"],
        nulls_last=True,
    ).to_dicts()

    active_by_chain: dict[tuple[str, str], list[str]] = {}
    audit_by_chain: dict[tuple[str, str], dict[str, object]] = {}
    for row in rows:
        key = (str(row["CIK"]), str(row["PERIODOFREPORT"]))
        state = audit_by_chain.setdefault(
            key,
            {
                "CIK": key[0],
                "PERIODOFREPORT": key[1],
                "FILINGMANAGER_NAME": row.get("FILINGMANAGER_NAME"),
                "N_EVENTS": 0,
                "N_NEW_HOLDINGS": 0,
                "CHAIN_COMPLETE": True,
                "CHAIN_REASON": "OK",
            },
        )
        state["N_EVENTS"] = int(state["N_EVENTS"]) + 1
        entry_valid = bool(row.get("ENTRY_TOTAL_MATCH")) or not require_entry_match
        confidential = str(row.get("ISCONFIDENTIALOMITTED") or "").strip().upper() == "Y"
        valid_accession = entry_valid and not confidential
        event = str(row["EVENT_TYPE"])
        accession = str(row["ACCESSION_NUMBER"])

        if event == EVENT_ORIGINAL:
            if valid_accession:
                active_by_chain[key] = [accession]
                state["CHAIN_COMPLETE"] = True
                state["CHAIN_REASON"] = "OK"
            else:
                active_by_chain[key] = []
                state["CHAIN_COMPLETE"] = False
                state["CHAIN_REASON"] = (
                    "CONFIDENTIAL_OMISSION" if confidential else "ENTRY_TOTAL_MISMATCH"
                )
        elif event == EVENT_RESTATEMENT:
            # A restatement contains the complete replacement table and can
            # therefore repair an incomplete or absent original accession.
            if valid_accession:
                active_by_chain[key] = [accession]
                state["CHAIN_COMPLETE"] = True
                state["CHAIN_REASON"] = "OK"
            else:
                active_by_chain[key] = []
                state["CHAIN_COMPLETE"] = False
                state["CHAIN_REASON"] = (
                    "CONFIDENTIAL_OMISSION" if confidential else "ENTRY_TOTAL_MISMATCH"
                )
        elif event == EVENT_NEW_HOLDINGS:
            state["N_NEW_HOLDINGS"] = int(state["N_NEW_HOLDINGS"]) + 1
            if not valid_accession:
                state["CHAIN_COMPLETE"] = False
                state["CHAIN_REASON"] = (
                    "CONFIDENTIAL_OMISSION" if confidential else "ENTRY_TOTAL_MISMATCH"
                )
            elif key not in active_by_chain or not active_by_chain[key]:
                state["CHAIN_COMPLETE"] = False
                state["CHAIN_REASON"] = "ORPHAN_NEW_HOLDINGS"
            else:
                active_by_chain[key].append(accession)
        else:
            state["CHAIN_COMPLETE"] = False
            state["CHAIN_REASON"] = "UNKNOWN_EVENT"

    chain_rows: list[dict[str, object]] = []
    valid_active: list[str] = []
    accession_to_cik: dict[str, str] = {}
    for key, state in audit_by_chain.items():
        active = active_by_chain.get(key, [])
        complete = bool(state["CHAIN_COMPLETE"]) and bool(active)
        state["CHAIN_COMPLETE"] = complete
        if not active and state["CHAIN_REASON"] == "OK":
            state["CHAIN_REASON"] = "NO_ACTIVE_ACCESSION"
        state["ACTIVE_ACCESSIONS"] = active
        state["N_ACTIVE_ACCESSIONS"] = len(active)
        chain_rows.append(state)
        if complete:
            valid_active.extend(active)
            accession_to_cik.update({accession: key[0] for accession in active})

    chains = pl.DataFrame(chain_rows)
    if valid_active:
        final_holdings = (
            holdings.filter(pl.col("ACCESSION_NUMBER").is_in(valid_active))
            .with_columns(
                pl.col("ACCESSION_NUMBER")
                .replace_strict(accession_to_cik)
                .alias("CIK")
            )
        )
    else:
        final_holdings = holdings.head(0).with_columns(
            pl.lit(None, dtype=pl.String).alias("CIK")
        )
    return FilingSnapshot(final_holdings, chains, audited)


def equity_holder_incidence(snapshot: FilingSnapshot) -> pl.DataFrame:
    """Return one auditable binary holder-incidence row per filer and CUSIP."""

    return (
        snapshot.holdings.filter(
            pl.col("PUTCALL").is_null()
            & (pl.col("SSHPRNAMTTYPE") == "SH")
            & pl.col("CUSIP").is_not_null()
            & pl.col("CUSIP").str.contains(r"^[0-9A-Z]{9}$")
            & (
                (pl.col("SSHPRNAMT") > 0)
                | (pl.col("VALUE") > 0)
            )
        )
        .group_by("CIK", "CUSIP")
        .agg(
            pl.col("VALUE").sum().alias("VALUE_DOLLARS_AS_FILED"),
            pl.col("SSHPRNAMT").sum().alias("SHARES_AS_FILED"),
            pl.len().alias("SOURCE_ROWS"),
        )
        .with_columns(pl.lit(1, dtype=pl.Int8).alias("HOLDS"))
        .sort(["CIK", "CUSIP"])
    )


__all__ = [
    "FilingSnapshot",
    "audit_accessions",
    "equity_holder_incidence",
    "load_dera_tables",
    "reconstruct_filing_chains",
]
