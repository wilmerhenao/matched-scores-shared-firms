from __future__ import annotations

import json

import polars as pl

from experiments.build_public_ownership import similarity_edges
from scripts.fetch_openfigi import mapping_payload
from src.public_universe import (
    build_figi_cusip_link,
    build_historical_sec_cusip_links,
    collapse_to_firm_nodes,
    cusip_check_digit_is_valid,
    load_openfigi_batches,
    parse_official_13f_text,
    ticker_figi_candidates,
)


def test_cusip_check_digit_examples():
    assert cusip_check_digit_is_valid("037833100")  # Apple
    assert cusip_check_digit_is_valid("02079K305")  # Alphabet class A
    assert not cusip_check_digit_is_valid("037833101")


def test_official_list_parser_handles_optional_star_and_status_alignment(tmp_path):
    filler = [
        f"{index:06d} 10 0         ISSUER {index:06d}                    COM"
        for index in range(10_000)
    ]
    filler.extend(
        [
            "06682J 10 0         BANZAI INTERNATIONAL INC          COM CL A           ADDED",
            "C00948 95 6         AGRIFORCE GROWING SYSTEMS LT      PUT                DELETED",
            "H11356 10 4 *       BUNGE GLOBAL SA                   COM SHS            ADDED",
        ]
    )
    path = tmp_path / "official.txt"
    path.write_text("\n".join(filler) + "\n")
    parsed = parse_official_13f_text(path)

    banzai = parsed.filter(pl.col("CUSIP") == "06682J100").row(0, named=True)
    assert banzai["ISSUER_NAME"] == "BANZAI INTERNATIONAL INC"
    assert banzai["ISSUER_DESCRIPTION"] == "COM CL A"
    assert banzai["STATUS"] == "ADDED"
    assert not banzai["HAS_LISTED_OPTION"]
    deleted = parsed.filter(pl.col("CUSIP") == "C00948956").row(0, named=True)
    assert deleted["STATUS"] == "DELETED"
    bunge = parsed.filter(pl.col("CUSIP") == "H11356104").row(0, named=True)
    assert bunge["HAS_LISTED_OPTION"]


def test_share_classes_collapse_by_cik_with_deterministic_primary():
    securities = pl.DataFrame(
        {
            "TICKER": ["GOOGL", "GOOG", "AAPL"],
            "PRICE_TICKER": ["GOOGL", "GOOG", "AAPL"],
            "CIK": ["1", "1", "2"],
            "SECURITY": ["Alphabet A", "Alphabet C", "Apple"],
            "GICS_SECTOR": ["Tech"] * 3,
            "GICS_SUBINDUSTRY": ["Interactive Media", "Interactive Media", "Hardware"],
        }
    )
    firms = collapse_to_firm_nodes(securities)
    alphabet = firms.filter(pl.col("CIK") == "1").row(0, named=True)
    assert alphabet["PRIMARY_TICKER"] == "GOOG"
    assert alphabet["INDEX_TICKERS"] == ["GOOG", "GOOGL"]
    assert alphabet["N_INDEX_SECURITIES"] == 2


def test_exact_figi_bridge_rejects_ambiguous_or_unofficial_cusips(tmp_path):
    envelope = {
        "jobs": [{"ticker": "AAPL", "cik": "1"}],
        "responses": [
            {
                "data": [
                    {
                        "figi": "BBG000B9XRY4",
                        "compositeFIGI": "BBG000B9XRY4",
                        "shareClassFIGI": "BBG001S5N8V8",
                        "name": "APPLE INC",
                        "ticker": "AAPL",
                        "exchCode": "US",
                        "marketSector": "Equity",
                        "securityType2": "Common Stock",
                    }
                ]
            }
        ],
    }
    path = tmp_path / "batch.json"
    path.write_text(json.dumps(envelope))
    candidates = ticker_figi_candidates(load_openfigi_batches([path]))
    holdings = pl.DataFrame(
        {
            "FIGI": ["BBG000B9XRY4", "BBG001S5N8V8"],
            "CUSIP": ["037833100", "037833100"],
            "NAMEOFISSUER": ["APPLE INC", "APPLE INC"],
        }
    )
    official = pl.DataFrame(
        {
            "CUSIP": ["037833100"],
            "ISSUER_NAME": ["APPLE INC"],
            "ISSUER_DESCRIPTION": ["COM"],
            "STATUS": [""],
        }
    )
    identities = pl.DataFrame(
        {
            "TICKER": ["AAPL"],
            "CIK": ["1"],
            "SECURITY": ["Apple"],
            "SEC_TITLE": [["APPLE INC"]],
        }
    )
    accepted, audit = build_figi_cusip_link(
        candidates, holdings, official, identities
    )
    assert accepted.select("CIK", "CUSIP").to_dicts() == [
        {"CIK": "1", "CUSIP": "037833100"}
    ]
    assert audit.get_column("CUSIP_CHECK_VALID").all()
    assert accepted.get_column("FIGI_KIND").to_list() == ["shareClassFIGI"]
    audit_keys = ["CIK", "TICKER", "CUSIP", "FIGI_KIND", "FIGI"]
    assert audit.select(audit_keys).equals(audit.sort(audit_keys).select(audit_keys))


def test_global_ticker_query_retains_non_us_venue_share_class(tmp_path):
    jobs = [{"ticker": "HES", "cik": "1"}]
    assert "exchCode" not in mapping_payload(jobs, "global")[0]
    assert mapping_payload(jobs, "us")[0]["exchCode"] == "US"
    envelope = {
        "jobs": jobs,
        "responses": [
            {
                "data": [
                    {
                        "figi": "HESS_EXCHANGE",
                        "compositeFIGI": "HESS_COMPOSITE",
                        "shareClassFIGI": "HESS_SHARE_CLASS",
                        "name": "HESS CORP",
                        "ticker": "HES",
                        "exchCode": "UA",
                        "marketSector": "Equity",
                        "securityType2": "Common Stock",
                    }
                ]
            }
        ],
    }
    path = tmp_path / "batch.json"
    path.write_text(json.dumps(envelope))
    candidates = ticker_figi_candidates(load_openfigi_batches([path]))
    assert candidates.filter(
        pl.col("FIGI_KIND") == "shareClassFIGI"
    ).get_column("FIGI").to_list() == ["HESS_SHARE_CLASS"]


def test_identity_gate_rejects_exchange_figi_and_ticker_reuse():
    candidates = pl.DataFrame(
        {
            "TICKER": ["AAPL", "AON", "PARA"],
            "CIK": ["1", "2", "3"],
            "FIGI": ["APPLE_SHARE", "AON_EXCHANGE", "BANZAI_SHARE"],
            "FIGI_KIND": ["shareClassFIGI", "figi", "shareClassFIGI"],
            "OPENFIGI_NAME": ["APPLE INC", "AON PLC", "BANZAI INTERNATIONAL INC"],
        }
    )
    holdings = pl.DataFrame(
        {
            "FIGI": ["APPLE_SHARE", "AON_EXCHANGE", "BANZAI_SHARE"],
            "CUSIP": ["037833100", "724078100", "06682J100"],
            "NAMEOFISSUER": ["APPLE INC", "PIPER SANDLER COMPANIES", "BANZAI INTERNATIONAL INC"],
        }
    )
    official = pl.DataFrame(
        {
            "CUSIP": ["037833100", "724078100", "06682J100"],
            "ISSUER_NAME": ["APPLE INC", "PIPER SANDLER COMPANIES", "BANZAI INTERNATIONAL INC"],
            "ISSUER_DESCRIPTION": ["COM", "COM", "COM CL A"],
            "STATUS": ["", "", "ADDED"],
        }
    )
    identities = pl.DataFrame(
        {
            "TICKER": ["AAPL", "AON", "PARA"],
            "CIK": ["1", "2", "3"],
            "SECURITY": ["Apple", "Aon", "Paramount Global"],
            "SEC_TITLE": [["APPLE INC"], ["AON PLC"], []],
        },
        schema_overrides={"SEC_TITLE": pl.List(pl.String)},
    )

    accepted, audit = build_figi_cusip_link(
        candidates, holdings, official, identities
    )
    assert accepted.select("TICKER", "CUSIP").to_dicts() == [
        {"TICKER": "AAPL", "CUSIP": "037833100"}
    ]
    statuses = dict(zip(audit.get_column("TICKER"), audit.get_column("LINK_STATUS")))
    assert "NON_SHARE_CLASS_FIGI" in statuses["AON"]
    assert "FILED_TARGET_NAME_MISMATCH" in statuses["AON"]
    assert "OPENFIGI_TARGET_NAME_MISMATCH" in statuses["PARA"]


def test_historical_sec_gate_accepts_unique_issue_and_explicit_b_class():
    issue_rows = [
        ("723787107", "PIONEER NAT RES CO", "COM", "PXD_FIGI"),
        ("723787907", "PIONEER NAT RES CO", "CALL", None),
        ("115637100", "BROWN FORMAN CORP", "CL A", "BF_A_FIGI"),
        ("115637209", "BROWN FORMAN CORP", "CL B", "BF_B_FIGI"),
        ("92556H107", "PARAMOUNT GLOBAL", "CLASS A COM", "PARA_A_FIGI"),
        ("92556H206", "PARAMOUNT GLOBAL", "CLASS B COM", "PARA_B_FIGI"),
        ("50540R409", "LABORATORY CORP AMER HLDGS", "COM", "LH_FIGI"),
        ("451107106", "IDACORP INC", "COM", "IDA_FIGI"),
        ("96145D105", "WESTROCK CO", "COM", "WRK_FIGI"),
        ("96145W103", "WESTROCK COFFEE CO", "COM", "WEST_FIGI"),
    ]
    holdings = pl.DataFrame(
        {
            "ACCESSION_NUMBER": [f"A{index}" for index in range(len(issue_rows))],
            "CIK": [f"F{index}" for index in range(len(issue_rows))],
            "CUSIP": [row[0] for row in issue_rows],
            "NAMEOFISSUER": [row[1] for row in issue_rows],
            "TITLEOFCLASS": [row[2] for row in issue_rows],
            "FIGI": [row[3] for row in issue_rows],
            "VALUE": [100] * len(issue_rows),
            "SSHPRNAMT": [10.0] * len(issue_rows),
            "SSHPRNAMTTYPE": ["SH"] * len(issue_rows),
            "PUTCALL": [None] * len(issue_rows),
        }
    )
    official = pl.DataFrame(
        {
            "CUSIP": [row[0] for row in issue_rows],
            "ISSUER_NAME": [row[1] for row in issue_rows],
            "ISSUER_DESCRIPTION": [row[2] for row in issue_rows],
            "STATUS": [""] * len(issue_rows),
        }
    )
    identities = pl.DataFrame(
        {
            "TICKER": ["PXD", "BF.B", "PARA", "LH", "WRK", "AAPL"],
            "CIK": ["1", "2", "3", "4", "5", "6"],
            "SECURITY": [
                "Pioneer Natural Resources",
                "Brown-Forman",
                "Paramount Global",
                "LabCorp",
                "WestRock",
                "Apple",
            ],
            "SEC_TITLE": [[], [], [], [], [], ["APPLE INC"]],
        },
        schema_overrides={"SEC_TITLE": pl.List(pl.String)},
    )
    exact = pl.DataFrame(
        {"TICKER": ["AAPL"], "CIK": ["6"], "CUSIP": ["037833100"]}
    )

    accepted, audit = build_historical_sec_cusip_links(
        holdings, official, identities, exact
    )
    assert accepted.select("TICKER", "CUSIP").to_dicts() == [
        {"TICKER": "PXD", "CUSIP": "723787107"},
        {"TICKER": "BF.B", "CUSIP": "115637209"},
        {"TICKER": "WRK", "CUSIP": "96145D105"},
    ]
    statuses = dict(
        zip(audit.get_column("TICKER"), audit.get_column("HISTORICAL_LINK_STATUS"))
    )
    assert statuses["AAPL"] == "SKIPPED_EXACT_FIGI_LINK"
    assert statuses["PARA"] == "MULTIPLE_RETAINED_EQUITY_ISSUES"
    assert statuses["LH"] == "TARGET_NAME_BELOW_MINIMUM"


def test_similarity_positions_have_deterministic_key_order():
    firms = pl.DataFrame(
        {
            "NODE_ID": [0, 1, 2],
            "CIK": ["3", "1", "2"],
            "GICS_SECTOR": ["A", "A", "B"],
            "GICS_SUBINDUSTRY": ["A1", "A2", "B1"],
        }
    )
    links = pl.DataFrame(
        {
            "CIK": ["2", "3", "1"],
            "CUSIP": ["CUSIP0002", "CUSIP0003", "CUSIP0001"],
        }
    )
    incidence = pl.DataFrame(
        {
            "CIK": ["F2", "F1", "F1", "F2"],
            "CUSIP": ["CUSIP0003", "CUSIP0002", "CUSIP0001", "CUSIP0001"],
            "VALUE_DOLLARS_AS_FILED": [20, 10, 30, 40],
            "SHARES_AS_FILED": [2.0, 1.0, 3.0, 4.0],
            "SOURCE_ROWS": [1, 1, 1, 1],
        }
    )

    _, positions, _ = similarity_edges(firms, incidence, links)
    keys = ["TARGET_CIK", "FILER_CIK"]
    assert positions.select(keys).equals(positions.sort(keys).select(keys))
