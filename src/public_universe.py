"""Point-in-time public firm universe and FIGI-to-CUSIP linkage helpers."""

from __future__ import annotations

import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

import polars as pl
from bs4 import BeautifulSoup


# Global, issuer-independent normalizations for abbreviations commonly used in
# SEC information tables.  These are deliberately not issuer-specific aliases
# or manual CUSIP overrides.
_LEGAL_NAME_TOKENS = {
    "A",
    "AG",
    "B",
    "C",
    "CLASS",
    "CO",
    "COM",
    "COMMON",
    "COMPANIES",
    "COMPANY",
    "CORP",
    "CORPORATION",
    "COS",
    "DE",
    "DEL",
    "GROUP",
    "HOLDING",
    "HOLDINGS",
    "INC",
    "INCORPORATED",
    "LIMITED",
    "LLC",
    "LP",
    "LTD",
    "NA",
    "NEW",
    "NV",
    "ORDINARY",
    "PLC",
    "SA",
    "SE",
    "SHARE",
    "SHARES",
    "STOCK",
    "THE",
}
_SEC_NAME_ABBREVIATIONS = {
    "AIRLS": "AIRLINES",
    "AMER": "AMERICAN",
    "APT": "APARTMENT",
    "ASSOCIATES": "ASSOC",
    "BAN": "BANK",
    "BANCORPORATION": "BANCORP",
    "CHEMS": "CHEMICALS",
    "CMNTYS": "COMMUNITIES",
    "ELEC": "ELECTRIC",
    "FINL": "FINANCIAL",
    "HLDGS": "HOLDINGS",
    "HLTH": "HEALTH",
    "INDS": "INDUSTRIES",
    "INDUSTRIES": "INDUSTRY",
    "INFORMATION": "INFO",
    "INTL": "INTERNATIONAL",
    "INVS": "INVESTMENT",
    "INVT": "INVESTMENT",
    "LABORATORIES": "LABORATORY",
    "LABS": "LABORATORIES",
    "MATLS": "MATERIALS",
    "MGMT": "MANAGEMENT",
    "MTR": "MOTOR",
    "NATL": "NATIONAL",
    "PPTY": "PROPERTY",
    "PPTYS": "PROPERTIES",
    "PRODS": "PRODUCTS",
    "PROPERTIES": "PROPERTY",
    "PWR": "POWER",
    "RESH": "RESEARCH",
    "RES": "RESOURCES",
    "RESOURCES": "RESOURCE",
    "RLTY": "REALTY",
    "SERVICES": "SERVICE",
    "SOLUTIONS": "SOLUTION",
    "SVCS": "SERVICES",
    "SYS": "SYSTEMS",
    "SYSTEM": "SYSTEMS",
    "TECH": "TECHNOLOGY",
    "TECHNOLOGIES": "TECHNOLOGY",
    "TRANS": "TRANSPORT",
    "TR": "TRUST",
    "WHSL": "WHOLESALE",
    "WKS": "WORKS",
}
ISSUER_NAME_SCORE_MINIMUM = 0.60
OPENFIGI_TARGET_NAME_SCORE_MINIMUM = 0.85
HISTORICAL_TARGET_NAME_SCORE_MINIMUM = 0.85
HISTORICAL_FILED_NAME_SCORE_MINIMUM = 0.85
HISTORICAL_MINIMUM_NAME_SCORE_GAP = 0.05


def parse_sp500_constituents(html_path: str | Path) -> pl.DataFrame:
    """Parse the constituents table from a checksum-pinned Wikipedia revision."""

    soup = BeautifulSoup(Path(html_path).read_text(encoding="utf-8"), "html.parser")
    table = soup.find("table", id="constituents")
    if table is None:
        raise ValueError("constituents table not found")
    rows: list[dict[str, str]] = []
    for table_row in table.find_all("tr")[1:]:
        cells = [cell.get_text(" ", strip=True) for cell in table_row.find_all(["th", "td"])]
        if len(cells) < 8:
            continue
        rows.append(
            {
                "TICKER": cells[0],
                "SECURITY": cells[1],
                "GICS_SECTOR": cells[2],
                "GICS_SUBINDUSTRY": cells[3],
                "HEADQUARTERS": cells[4],
                "DATE_ADDED": cells[5] or None,
                "CIK": cells[6].zfill(10),
                "FOUNDED": cells[7],
            }
        )
    frame = pl.DataFrame(rows).with_columns(
        pl.col("DATE_ADDED").str.to_date("%Y-%m-%d", strict=False),
        pl.col("TICKER").str.replace_all("\\.", "-").alias("PRICE_TICKER"),
    )
    if frame.height < 450 or frame.get_column("CIK").n_unique() < 450:
        raise ValueError("parsed S&P universe is implausibly small")
    if frame.select(pl.struct(["TICKER", "CIK"]).n_unique()).item() != frame.height:
        raise ValueError("constituents table has duplicate ticker-CIK rows")
    return frame.sort("TICKER")


def collapse_to_firm_nodes(securities: pl.DataFrame) -> pl.DataFrame:
    """Collapse multiple index share classes to one CIK-level firm node.

    The lexicographically first ticker is the pre-specified return-security
    representative.  Ownership positions remain aggregated across every
    mapped share class for that CIK.
    """

    firms = (
        securities.sort(["CIK", "TICKER"])
        .group_by("CIK", maintain_order=True)
        .agg(
            pl.first("TICKER").alias("PRIMARY_TICKER"),
            pl.first("PRICE_TICKER").alias("PRICE_TICKER"),
            pl.first("SECURITY").alias("SECURITY"),
            pl.first("GICS_SECTOR").alias("GICS_SECTOR"),
            pl.first("GICS_SUBINDUSTRY").alias("GICS_SUBINDUSTRY"),
            pl.col("TICKER").alias("INDEX_TICKERS"),
            pl.len().alias("N_INDEX_SECURITIES"),
        )
        .sort("CIK")
    )
    return firms.with_row_index("NODE_ID")


def parse_sec_company_tickers(json_path: str | Path) -> pl.DataFrame:
    """Parse a checksum-pinned SEC company-ticker snapshot into CIK titles."""

    payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    rows = [
        {
            "CIK": str(item["cik_str"]).zfill(10),
            "SEC_TITLE": str(item["title"]).strip(),
        }
        for item in payload.values()
        if isinstance(item, dict) and item.get("cik_str") is not None and item.get("title")
    ]
    if len(rows) < 8_000:
        raise ValueError("SEC company-ticker snapshot is implausibly small")
    return (
        pl.DataFrame(rows)
        .group_by("CIK")
        .agg(pl.col("SEC_TITLE").unique().sort())
        .sort("CIK")
    )


def _normalized_issuer_tokens(value: str | None) -> tuple[str, ...]:
    text = (
        unicodedata.normalize("NFKD", value or "")
        .encode("ascii", "ignore")
        .decode("ascii")
        .upper()
        .replace("&", " AND ")
    )
    tokens = []
    for token in re.findall(r"[A-Z0-9]+", text):
        if token in _LEGAL_NAME_TOKENS:
            continue
        tokens.append(_SEC_NAME_ABBREVIATIONS.get(token, token))
    return tuple(sorted(tokens))


def issuer_name_score(left: str | None, right: str | None) -> float:
    """Return a deterministic token-order-invariant issuer concordance score."""

    left_tokens = _normalized_issuer_tokens(left)
    right_tokens = _normalized_issuer_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    left_set = set(left_tokens)
    right_set = set(right_tokens)
    overlap = len(left_set & right_set)
    containment = overlap / min(len(left_set), len(right_set))
    jaccard = overlap / len(left_set | right_set)
    sequence = SequenceMatcher(
        None, " ".join(left_tokens), " ".join(right_tokens)
    ).ratio()
    return float(max(sequence, 0.9 * containment + 0.1 * jaccard))


def parse_official_13f_text(text_path: str | Path) -> pl.DataFrame:
    """Parse text produced by ``pdftotext -layout`` on the SEC official list."""

    rows: list[dict[str, str | bool]] = []
    for line in Path(text_path).read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.match(
            r"^\s*([A-Z0-9]{6})\s+([A-Z0-9]{2})\s+([A-Z0-9])"
            r"(?:\s+(\*))?\s+(.*)$",
            line,
        )
        if match is None:
            continue
        fields = re.split(r"\s{2,}", match.group(5).strip())
        if len(fields) < 2:
            continue
        status = fields[-1] if fields[-1] in {"ADDED", "DELETED"} else ""
        description_fields = fields[1:-1] if status else fields[1:]
        rows.append(
            {
                "CUSIP": "".join(match.group(index) for index in (1, 2, 3)),
                "HAS_LISTED_OPTION": match.group(4) == "*",
                "ISSUER_NAME": fields[0],
                "ISSUER_DESCRIPTION": " ".join(description_fields),
                "STATUS": status,
            }
        )
    frame = pl.DataFrame(rows)
    if frame.height < 10_000:
        raise ValueError("parsed official 13F list is implausibly small")
    return frame.unique("CUSIP", keep="first", maintain_order=True)


def cusip_check_digit_is_valid(cusip: str) -> bool:
    """Validate a nine-character CUSIP check digit."""

    if re.fullmatch(r"[0-9A-Z*@#]{9}", cusip) is None:
        return False
    values: list[int] = []
    for character in cusip[:8]:
        if character.isdigit():
            values.append(int(character))
        elif "A" <= character <= "Z":
            values.append(ord(character) - ord("A") + 10)
        else:
            values.append({"*": 36, "@": 37, "#": 38}[character])
    total = 0
    for index, value in enumerate(values):
        product = value * (2 if index % 2 else 1)
        total += product // 10 + product % 10
    return str((10 - total % 10) % 10) == cusip[-1]


def load_openfigi_batches(paths: Iterable[str | Path]) -> list[dict[str, object]]:
    """Load resumable batch envelopes produced by ``fetch_openfigi.py``."""

    records: list[dict[str, object]] = []
    for path in sorted(Path(value) for value in paths):
        envelope = json.loads(path.read_text(encoding="utf-8"))
        jobs = envelope["jobs"]
        responses = envelope["responses"]
        if len(jobs) != len(responses):
            raise ValueError(f"OpenFIGI job/response mismatch in {path}")
        records.extend({"job": job, "response": response} for job, response in zip(jobs, responses))
    return records


def ticker_figi_candidates(batch_records: list[dict[str, object]]) -> pl.DataFrame:
    """Select global common-stock FIGIs for each requested index security.

    A share-class FIGI is global.  Restricting a ticker query to the current
    U.S. venue can therefore erase a valid historical security after a ticker
    change or delisting.  The downstream bridge accepts only an exact
    share-class FIGI observed in the point-in-time 13F filings and applies a
    strong issuer-name gate, so exchange and composite FIGIs remain audit-only.
    """

    rows: list[dict[str, str]] = []
    for record in batch_records:
        job = record["job"]
        response = record["response"]
        if not isinstance(job, dict) or not isinstance(response, dict):
            continue
        for item in response.get("data", []):
            if item.get("marketSector") != "Equity":
                continue
            if item.get("securityType2") not in {"Common Stock", "REIT"}:
                continue
            for figi_kind in ("figi", "compositeFIGI", "shareClassFIGI"):
                figi = item.get(figi_kind)
                if figi:
                    rows.append(
                        {
                            "TICKER": str(job["ticker"]),
                            "CIK": str(job["cik"]),
                            "FIGI": str(figi),
                            "FIGI_KIND": figi_kind,
                            "OPENFIGI_NAME": str(item.get("name") or ""),
                        }
                    )
    if not rows:
        return pl.DataFrame(
            schema={
                "TICKER": pl.String,
                "CIK": pl.String,
                "FIGI": pl.String,
                "FIGI_KIND": pl.String,
                "OPENFIGI_NAME": pl.String,
            }
        )
    # Keep the identifier kind explicit.  OpenFIGI can return the same value
    # under more than one kind, and the strict link gate below accepts only the
    # share-class kind.
    return pl.DataFrame(rows).unique(["TICKER", "CIK", "FIGI", "FIGI_KIND"])


def build_figi_cusip_link(
    figi_candidates: pl.DataFrame,
    holdings: pl.DataFrame,
    official_list: pl.DataFrame,
    target_identities: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Link target CIKs to filed CUSIPs through exact FIGIs.

    Returns ``(accepted_links, audit)``.  A CUSIP is accepted only when it has
    an exact share-class FIGI, a valid check digit, appears on the Q4 official
    list without DELETED status, agrees with the pinned target issuer identity,
    and maps to exactly one target CIK.  Exchange-level and composite FIGIs are
    retained in the audit but never accepted.
    """

    observed = (
        holdings.filter(pl.col("FIGI").is_not_null())
        .group_by("FIGI", "CUSIP")
        .agg(
            pl.len().alias("FIGI_CUSIP_SOURCE_ROWS"),
            pl.col("NAMEOFISSUER").mode().first().alias("FILING_ISSUER_NAME"),
        )
    )
    joined = (
        figi_candidates.join(observed, on="FIGI", how="inner")
        .join(
            official_list.select(
                "CUSIP", "ISSUER_NAME", "ISSUER_DESCRIPTION", "STATUS"
            ),
            on="CUSIP",
            how="left",
        )
        .join(
            target_identities.select("TICKER", "CIK", "SECURITY", "SEC_TITLE"),
            on=["TICKER", "CIK"],
            how="left",
            validate="m:1",
        )
    )

    def identity_names(row: dict[str, object]) -> list[str]:
        values: list[str] = []
        security = row.get("SECURITY")
        if security:
            values.append(str(security))
        titles = row.get("SEC_TITLE")
        if isinstance(titles, list):
            values.extend(str(title) for title in titles if title)
        elif titles:
            values.append(str(titles))
        return values

    def openfigi_target_score(row: dict[str, object]) -> float:
        names = identity_names(row)
        return max(
            (
                issuer_name_score(str(row.get("OPENFIGI_NAME") or ""), target)
                for target in names
            ),
            default=0.0,
        )

    def filed_target_score(row: dict[str, object]) -> float:
        names = identity_names(row)
        filed_names = [
            str(row.get("FILING_ISSUER_NAME") or ""),
            str(row.get("ISSUER_NAME") or ""),
        ]
        return max(
            (
                issuer_name_score(filed, target)
                for filed in filed_names
                for target in names
            ),
            default=0.0,
        )

    audit = joined.with_columns(
        (pl.col("FIGI_KIND") == "shareClassFIGI").alias("SHARE_CLASS_FIGI"),
        pl.struct("OPENFIGI_NAME", "SECURITY", "SEC_TITLE")
        .map_elements(openfigi_target_score, return_dtype=pl.Float64)
        .alias("OPENFIGI_TARGET_NAME_SCORE"),
        pl.struct("FILING_ISSUER_NAME", "ISSUER_NAME", "SECURITY", "SEC_TITLE")
        .map_elements(filed_target_score, return_dtype=pl.Float64)
        .alias("FILED_TARGET_NAME_SCORE"),
        pl.struct("FILING_ISSUER_NAME", "ISSUER_NAME")
        .map_elements(
            lambda row: issuer_name_score(
                str(row.get("FILING_ISSUER_NAME") or ""),
                str(row.get("ISSUER_NAME") or ""),
            ),
            return_dtype=pl.Float64,
        )
        .alias("FILING_OFFICIAL_NAME_SCORE"),
        pl.col("CUSIP")
        .map_elements(cusip_check_digit_is_valid, return_dtype=pl.Boolean)
        .alias("CUSIP_CHECK_VALID"),
        pl.col("ISSUER_NAME").is_not_null().alias("ON_OFFICIAL_LIST"),
        (pl.col("STATUS").fill_null("") != "DELETED").alias("NOT_DELETED"),
    ).with_columns(
        (
            pl.col("OPENFIGI_TARGET_NAME_SCORE")
            >= OPENFIGI_TARGET_NAME_SCORE_MINIMUM
        ).alias(
            "OPENFIGI_TARGET_NAME_CONCORDANT"
        ),
        (pl.col("FILED_TARGET_NAME_SCORE") >= ISSUER_NAME_SCORE_MINIMUM).alias(
            "FILED_TARGET_NAME_CONCORDANT"
        ),
        (pl.col("FILING_OFFICIAL_NAME_SCORE") >= ISSUER_NAME_SCORE_MINIMUM).alias(
            "FILING_OFFICIAL_NAME_CONCORDANT"
        ),
    )
    preliminary = (
        pl.col("SHARE_CLASS_FIGI")
        & pl.col("CUSIP_CHECK_VALID")
        & pl.col("ON_OFFICIAL_LIST")
        & pl.col("NOT_DELETED")
        & pl.col("OPENFIGI_TARGET_NAME_CONCORDANT")
        & pl.col("FILED_TARGET_NAME_CONCORDANT")
        & pl.col("FILING_OFFICIAL_NAME_CONCORDANT")
    )
    target_counts = (
        audit.filter(preliminary)
        .group_by("CUSIP")
        .agg(pl.col("CIK").n_unique().alias("CUSIP_TARGET_CIKS"))
    )
    audit = audit.join(target_counts, on="CUSIP", how="left").with_columns(
        pl.col("CUSIP_TARGET_CIKS").fill_null(0)
    )

    def rejection_reasons(row: dict[str, object]) -> str:
        gates = [
            ("SHARE_CLASS_FIGI", "NON_SHARE_CLASS_FIGI"),
            ("CUSIP_CHECK_VALID", "INVALID_CUSIP_CHECK_DIGIT"),
            ("ON_OFFICIAL_LIST", "NOT_ON_OFFICIAL_LIST"),
            ("NOT_DELETED", "DELETED_ON_OFFICIAL_LIST"),
            ("OPENFIGI_TARGET_NAME_CONCORDANT", "OPENFIGI_TARGET_NAME_MISMATCH"),
            ("FILED_TARGET_NAME_CONCORDANT", "FILED_TARGET_NAME_MISMATCH"),
            ("FILING_OFFICIAL_NAME_CONCORDANT", "FILING_OFFICIAL_NAME_MISMATCH"),
        ]
        reasons = [reason for column, reason in gates if not bool(row.get(column))]
        if int(row.get("CUSIP_TARGET_CIKS") or 0) != 1:
            reasons.append("NONUNIQUE_TARGET_CIK")
        return "ACCEPTED" if not reasons else "|".join(reasons)

    audit = (
        audit.with_columns(
            pl.struct(
                "SHARE_CLASS_FIGI",
                "CUSIP_CHECK_VALID",
                "ON_OFFICIAL_LIST",
                "NOT_DELETED",
                "OPENFIGI_TARGET_NAME_CONCORDANT",
                "FILED_TARGET_NAME_CONCORDANT",
                "FILING_OFFICIAL_NAME_CONCORDANT",
                "CUSIP_TARGET_CIKS",
            )
            .map_elements(rejection_reasons, return_dtype=pl.String)
            .alias("LINK_STATUS"),
            pl.lit("OPENFIGI_SHARE_CLASS_EXACT").alias("LINK_METHOD"),
            pl.lit(
                "GLOBAL_OPENFIGI_TICKER_QUERY+SEC_DERA_EXACT_FIGI+"
                "SEC_OFFICIAL_13F_LIST"
            ).alias("LINK_PROVENANCE"),
        )
        .sort(["CIK", "TICKER", "CUSIP", "FIGI_KIND", "FIGI"])
    )
    accepted = (
        audit.filter(pl.col("LINK_STATUS") == "ACCEPTED")
        .sort(["CIK", "CUSIP", "FIGI_CUSIP_SOURCE_ROWS"], descending=[False, False, True])
        .unique(["CIK", "CUSIP"], keep="first", maintain_order=True)
    )
    return accepted, audit


def _identity_names(row: dict[str, object]) -> list[str]:
    values: list[str] = []
    security = row.get("SECURITY")
    if security:
        values.append(str(security))
    titles = row.get("SEC_TITLE")
    if isinstance(titles, list):
        values.extend(str(title) for title in titles if title)
    elif titles:
        values.append(str(titles))
    return values


_NON_COMMON_EQUITY_MARKERS = re.compile(
    r"\b(?:CALL|PUT|NOTE|NOTES|BOND|BONDS|DEBT|DEB|DEBENTURE|"
    r"PFD|PREFERRED|PREF|WARRANT|WARRANTS|WT|RIGHT|RIGHTS|UNIT|UNITS)\b"
)
_COMMON_EQUITY_MARKERS = re.compile(
    r"\b(?:COM|COMMON|SH|SHS|SHARE|SHARES|ORD|ORDINARY|CL|CLASS)\b"
)


def _looks_like_common_equity(value: str | None) -> bool:
    text = unicodedata.normalize("NFKD", value or "").encode(
        "ascii", "ignore"
    ).decode("ascii").upper()
    if not text or _NON_COMMON_EQUITY_MARKERS.search(text):
        return False
    return _COMMON_EQUITY_MARKERS.search(text) is not None


def _ticker_class_suffix(ticker: str | None) -> str | None:
    match = re.search(r"[.\-]([A-Z])$", str(ticker or "").upper())
    return match.group(1) if match else None


def _issue_class_letter(*values: str | None) -> str | None:
    for value in values:
        match = re.search(
            r"\b(?:CL|CLASS)\s*[- ]?([A-Z])\b", str(value or "").upper()
        )
        if match:
            return match.group(1)
    return None


def build_historical_sec_cusip_links(
    holdings: pl.DataFrame,
    official_list: pl.DataFrame,
    target_identities: pl.DataFrame,
    exact_links: pl.DataFrame,
    *,
    target_name_score_minimum: float = HISTORICAL_TARGET_NAME_SCORE_MINIMUM,
    filed_name_score_minimum: float = HISTORICAL_FILED_NAME_SCORE_MINIMUM,
    minimum_name_score_gap: float = HISTORICAL_MINIMUM_NAME_SCORE_GAP,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Recover historical CUSIPs without ticker substrings or manual aliases.

    This fallback is considered only after the exact global share-class FIGI
    bridge.  It selects an SEC official-list issuer group only when (i) the
    best target-name score clears a strong global threshold, (ii) it is
    separated from the runner-up issuer group, (iii) filed and official issuer
    names agree, and (iv) exactly one observed common-equity issue remains.
    An explicit ticker class suffix such as ``.B`` is allowed to resolve a
    multi-class issuer; an unsuffixed ambiguous target is rejected.
    """

    observed = (
        holdings.filter(
            pl.col("PUTCALL").is_null()
            & (pl.col("SSHPRNAMTTYPE") == "SH")
            & pl.col("CUSIP").is_not_null()
            & (
                (pl.col("SSHPRNAMT") > 0)
                | (pl.col("VALUE") > 0)
            )
        )
        .group_by("CUSIP")
        .agg(
            pl.len().alias("HISTORICAL_SOURCE_ROWS"),
            pl.col("CIK").n_unique().alias("HISTORICAL_SOURCE_FILERS"),
            pl.col("NAMEOFISSUER")
            .drop_nulls()
            .mode()
            .sort()
            .first()
            .alias("FILING_ISSUER_NAME"),
            pl.col("TITLEOFCLASS")
            .drop_nulls()
            .mode()
            .sort()
            .first()
            .alias("FILING_TITLE_OF_CLASS"),
            pl.col("FIGI")
            .drop_nulls()
            .mode()
            .sort()
            .first()
            .alias("HISTORICAL_FILED_FIGI"),
        )
        .join(
            official_list.select(
                "CUSIP", "ISSUER_NAME", "ISSUER_DESCRIPTION", "STATUS"
            ),
            on="CUSIP",
            how="inner",
            validate="1:1",
        )
        .with_columns(
            pl.col("CUSIP")
            .map_elements(cusip_check_digit_is_valid, return_dtype=pl.Boolean)
            .alias("CUSIP_CHECK_VALID"),
            (pl.col("STATUS").fill_null("") != "DELETED").alias("NOT_DELETED"),
            pl.col("ISSUER_DESCRIPTION")
            .map_elements(_looks_like_common_equity, return_dtype=pl.Boolean)
            .alias("OFFICIAL_COMMON_EQUITY"),
            pl.col("FILING_TITLE_OF_CLASS")
            .map_elements(_looks_like_common_equity, return_dtype=pl.Boolean)
            .alias("FILED_COMMON_EQUITY"),
            pl.struct("FILING_ISSUER_NAME", "ISSUER_NAME")
            .map_elements(
                lambda row: issuer_name_score(
                    str(row.get("FILING_ISSUER_NAME") or ""),
                    str(row.get("ISSUER_NAME") or ""),
                ),
                return_dtype=pl.Float64,
            )
            .alias("FILING_OFFICIAL_NAME_SCORE"),
        )
        .filter(
            pl.col("CUSIP_CHECK_VALID")
            & pl.col("NOT_DELETED")
            & pl.col("OFFICIAL_COMMON_EQUITY")
            & pl.col("FILED_COMMON_EQUITY")
            & (pl.col("FILING_OFFICIAL_NAME_SCORE") >= filed_name_score_minimum)
        )
        .sort(["ISSUER_NAME", "CUSIP"])
    )

    exact_keys = set()
    if exact_links.height:
        exact_keys = {
            (str(row["TICKER"]), str(row["CIK"]))
            for row in exact_links.select("TICKER", "CIK").unique().to_dicts()
        }
    issue_rows = observed.to_dicts()
    accepted_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    targets = target_identities.unique(["TICKER", "CIK"], keep="first").sort(
        ["CIK", "TICKER"]
    )

    for target in targets.to_dicts():
        ticker = str(target["TICKER"])
        cik = str(target["CIK"])
        base_audit: dict[str, object] = {
            "TICKER": ticker,
            "CIK": cik,
            "SECURITY": target.get("SECURITY"),
            "EXPLICIT_CLASS_SUFFIX": _ticker_class_suffix(ticker),
            "TARGET_NAME_SCORE_MINIMUM": target_name_score_minimum,
            "FILED_NAME_SCORE_MINIMUM": filed_name_score_minimum,
            "MINIMUM_NAME_SCORE_GAP": minimum_name_score_gap,
            "LINK_METHOD": "SEC_OFFICIAL_FILED_NAME_HISTORICAL",
            "LINK_PROVENANCE": (
                "SEC_Q4_2023_OFFICIAL_LIST+SEC_DERA_Q4_2023+PINNED_TARGET_IDENTITY"
            ),
        }
        if (ticker, cik) in exact_keys:
            audit_rows.append(
                {
                    **base_audit,
                    "HISTORICAL_LINK_STATUS": "SKIPPED_EXACT_FIGI_LINK",
                    "TOP_ISSUER_NAME": None,
                    "TOP_TARGET_NAME_SCORE": None,
                    "RUNNER_UP_TARGET_NAME_SCORE": None,
                    "TARGET_NAME_SCORE_GAP": None,
                    "RETAINED_EQUITY_ISSUES": 0,
                    "RETAINED_CUSIPS": [],
                    "SELECTED_CUSIP": None,
                }
            )
            continue

        names = _identity_names(target)
        group_scores: dict[str, float] = {}
        for issue in issue_rows:
            issuer = str(issue["ISSUER_NAME"])
            group_scores[issuer] = max(
                group_scores.get(issuer, 0.0),
                max(
                    (issuer_name_score(issuer, name) for name in names),
                    default=0.0,
                ),
            )
        ranked = sorted(group_scores.items(), key=lambda item: (-item[1], item[0]))
        if not ranked:
            audit_rows.append(
                {
                    **base_audit,
                    "HISTORICAL_LINK_STATUS": "NO_OBSERVED_OFFICIAL_EQUITY",
                    "TOP_ISSUER_NAME": None,
                    "TOP_TARGET_NAME_SCORE": None,
                    "RUNNER_UP_TARGET_NAME_SCORE": None,
                    "TARGET_NAME_SCORE_GAP": None,
                    "RETAINED_EQUITY_ISSUES": 0,
                    "RETAINED_CUSIPS": [],
                    "SELECTED_CUSIP": None,
                }
            )
            continue

        top_issuer, top_score = ranked[0]
        runner_score = ranked[1][1] if len(ranked) > 1 else None
        score_gap = top_score - runner_score if runner_score is not None else None
        status = "ACCEPTED"
        if top_score < target_name_score_minimum:
            status = "TARGET_NAME_BELOW_MINIMUM"
        elif score_gap is not None and score_gap + 1e-12 < minimum_name_score_gap:
            status = "TARGET_NAME_NOT_UNIQUELY_SEPARATED"

        retained = [issue for issue in issue_rows if issue["ISSUER_NAME"] == top_issuer]
        suffix = _ticker_class_suffix(ticker)
        if suffix is not None:
            retained = [
                issue
                for issue in retained
                if _issue_class_letter(
                    str(issue.get("ISSUER_DESCRIPTION") or ""),
                    str(issue.get("FILING_TITLE_OF_CLASS") or ""),
                )
                == suffix
            ]
            if status == "ACCEPTED" and not retained:
                status = "NO_EXPLICIT_CLASS_MATCH"

        retained = sorted(retained, key=lambda issue: str(issue["CUSIP"]))
        retained_cusips = sorted({str(issue["CUSIP"]) for issue in retained})
        if status == "ACCEPTED" and len(retained_cusips) != 1:
            status = (
                "NO_RETAINED_EQUITY_ISSUE"
                if not retained_cusips
                else "MULTIPLE_RETAINED_EQUITY_ISSUES"
            )
        selected_cusip = retained_cusips[0] if status == "ACCEPTED" else None
        audit_rows.append(
            {
                **base_audit,
                "HISTORICAL_LINK_STATUS": status,
                "TOP_ISSUER_NAME": top_issuer,
                "TOP_TARGET_NAME_SCORE": top_score,
                "RUNNER_UP_TARGET_NAME_SCORE": runner_score,
                "TARGET_NAME_SCORE_GAP": score_gap,
                "RETAINED_EQUITY_ISSUES": len(retained_cusips),
                "RETAINED_CUSIPS": retained_cusips,
                "SELECTED_CUSIP": selected_cusip,
            }
        )
        if selected_cusip is None:
            continue
        issue = next(row for row in retained if row["CUSIP"] == selected_cusip)
        accepted_rows.append(
            {
                "TICKER": ticker,
                "CIK": cik,
                "CUSIP": selected_cusip,
                "SECURITY": target.get("SECURITY"),
                "SEC_TITLE": target.get("SEC_TITLE"),
                "ISSUER_NAME": issue.get("ISSUER_NAME"),
                "ISSUER_DESCRIPTION": issue.get("ISSUER_DESCRIPTION"),
                "FILING_ISSUER_NAME": issue.get("FILING_ISSUER_NAME"),
                "FILING_TITLE_OF_CLASS": issue.get("FILING_TITLE_OF_CLASS"),
                "HISTORICAL_FILED_FIGI": issue.get("HISTORICAL_FILED_FIGI"),
                "HISTORICAL_SOURCE_ROWS": issue.get("HISTORICAL_SOURCE_ROWS"),
                "HISTORICAL_SOURCE_FILERS": issue.get("HISTORICAL_SOURCE_FILERS"),
                "TOP_TARGET_NAME_SCORE": top_score,
                "RUNNER_UP_TARGET_NAME_SCORE": runner_score,
                "TARGET_NAME_SCORE_GAP": score_gap,
                "FILING_OFFICIAL_NAME_SCORE": issue.get(
                    "FILING_OFFICIAL_NAME_SCORE"
                ),
                "LINK_STATUS": "ACCEPTED",
                "LINK_METHOD": "SEC_OFFICIAL_FILED_NAME_HISTORICAL",
                "LINK_PROVENANCE": (
                    "SEC_Q4_2023_OFFICIAL_LIST+SEC_DERA_Q4_2023+"
                    "PINNED_TARGET_IDENTITY"
                ),
            }
        )

    if accepted_rows:
        accepted = pl.DataFrame(accepted_rows).sort(["CIK", "TICKER", "CUSIP"])
    else:
        accepted = pl.DataFrame(
            schema={
                "TICKER": pl.String,
                "CIK": pl.String,
                "CUSIP": pl.String,
                "LINK_STATUS": pl.String,
                "LINK_METHOD": pl.String,
                "LINK_PROVENANCE": pl.String,
            }
        )
    audit = pl.DataFrame(audit_rows).sort(["CIK", "TICKER"])
    return accepted, audit


__all__ = [
    "ISSUER_NAME_SCORE_MINIMUM",
    "OPENFIGI_TARGET_NAME_SCORE_MINIMUM",
    "HISTORICAL_TARGET_NAME_SCORE_MINIMUM",
    "HISTORICAL_FILED_NAME_SCORE_MINIMUM",
    "HISTORICAL_MINIMUM_NAME_SCORE_GAP",
    "build_figi_cusip_link",
    "build_historical_sec_cusip_links",
    "collapse_to_firm_nodes",
    "cusip_check_digit_is_valid",
    "issuer_name_score",
    "load_openfigi_batches",
    "parse_official_13f_text",
    "parse_sec_company_tickers",
    "parse_sp500_constituents",
    "ticker_figi_candidates",
]
