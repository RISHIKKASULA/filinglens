"""Corpus manifest: EDGAR fetch, cache, and accession/CIK/fiscal-period pinning.

The manifest (corpus.yaml) pins CIK + accession number + fiscal-period end date per
company at first fetch; everything downstream keys off the pinned accession, never
"latest", so runs reproduce. Cache layout (gitignored):
data/cache/{cik}/{accession}/{filing.html, text.txt, companyfacts.json}.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

# v0.1 corpus (frozen): 10 large-cap, plain us-gaap filers; financial sector excluded.
V01_TICKERS: tuple[str, ...] = (
    "AAPL",
    "MSFT",
    "NVDA",
    "KO",
    "PG",
    "JNJ",
    "WMT",
    "XOM",
    "COST",
    "CAT",
)

# SEC fair-access identity: declared User-Agent with a real contact email.
EDGAR_IDENTITY = "filinglens research contact.rishikkasula@gmail.com"

DEFAULT_MANIFEST_PATH = Path("corpus.yaml")
DEFAULT_KPIS_PATH = Path("kpis.yaml")
DEFAULT_CACHE_DIR = Path("data/cache")

# An annual duration fact must span roughly one fiscal year (52/53-week years included);
# this excludes the quarterly durations that share the same period end.
_ANNUAL_DURATION_DAYS = (300, 400)


class CompanyPin(BaseModel):
    """One company's pinned filing identity."""

    ticker: str
    cik: int
    accession: str
    fiscal_period_end: dt.date
    form: str = "10-K"
    filed: dt.date


class Manifest(BaseModel):
    version: str = "0.1"
    companies: list[CompanyPin] = []

    def get(self, ticker: str) -> CompanyPin | None:
        for pin in self.companies:
            if pin.ticker == ticker.upper():
                return pin
        return None


class KpiSpec(BaseModel):
    """A KPI with its frozen ordered us-gaap tag-fallback chain."""

    name: str
    unit: str
    tags: list[str]


class XbrlFact(BaseModel):
    """A resolved XBRL ground-truth fact for one company x KPI."""

    kpi: str
    tag: str
    unit: str
    value: float
    start: dt.date | None
    end: dt.date
    accession: str
    fiscal_year: int
    fiscal_period: str
    form: str


def load_manifest(path: Path = DEFAULT_MANIFEST_PATH) -> Manifest:
    if not path.exists():
        return Manifest()
    raw = yaml.safe_load(path.read_text())
    return Manifest.model_validate(raw)


def save_manifest(manifest: Manifest, path: Path = DEFAULT_MANIFEST_PATH) -> None:
    header = (
        "# Corpus manifest (frozen policy, architecture.md §1).\n"
        "# Pinned at first fetch: CIK + accession + fiscal-period end per company.\n"
        '# Graders and prompts key off the pinned accession, never "latest".\n'
    )
    body = yaml.safe_dump(
        json.loads(manifest.model_dump_json()), sort_keys=False, default_flow_style=False
    )
    path.write_text(header + body)


def load_kpis(path: Path = DEFAULT_KPIS_PATH) -> list[KpiSpec]:
    raw = yaml.safe_load(path.read_text())
    return [
        KpiSpec(name=name, unit=spec["unit"], tags=list(spec["tags"]))
        for name, spec in raw["kpis"].items()
    ]


def cache_dir_for(pin: CompanyPin, cache_root: Path = DEFAULT_CACHE_DIR) -> Path:
    return cache_root / str(pin.cik) / pin.accession


def _is_annual_duration(rec: dict[str, Any]) -> bool:
    if "start" not in rec:
        return True  # instant fact (e.g. Assets)
    start = dt.date.fromisoformat(rec["start"])
    end = dt.date.fromisoformat(rec["end"])
    lo, hi = _ANNUAL_DURATION_DAYS
    return lo <= (end - start).days <= hi


def resolve_fact(companyfacts: dict[str, Any], pin: CompanyPin, kpi: KpiSpec) -> XbrlFact | None:
    """Resolve a KPI's ground truth from raw SEC companyfacts JSON.

    Walks the frozen tag-fallback chain; the first tag with a fact reported in the pinned
    accession, for the pinned fiscal-period end, over an annual duration, wins. Returns
    None if no tag resolves (`no-ground-truth`).
    """
    gaap = companyfacts.get("facts", {}).get("us-gaap", {})
    period_end = pin.fiscal_period_end.isoformat()
    for tag in kpi.tags:
        units = gaap.get(tag, {}).get("units", {})
        for rec in units.get(kpi.unit, []):
            if (
                rec.get("accn") == pin.accession
                and rec.get("end") == period_end
                and rec.get("form") == pin.form
                and _is_annual_duration(rec)
            ):
                return XbrlFact(
                    kpi=kpi.name,
                    tag=tag,
                    unit=kpi.unit,
                    value=float(rec["val"]),
                    start=dt.date.fromisoformat(rec["start"]) if "start" in rec else None,
                    end=dt.date.fromisoformat(rec["end"]),
                    accession=rec["accn"],
                    fiscal_year=int(rec.get("fy", 0)),
                    fiscal_period=str(rec.get("fp", "")),
                    form=str(rec.get("form", "")),
                )
    return None


# ---------------------------------------------------------------------------
# Network plumbing via edgartools (not exercised in CI; CI uses fixtures).
# ---------------------------------------------------------------------------


def _set_identity() -> None:  # pragma: no cover
    from edgar import set_identity

    set_identity(EDGAR_IDENTITY)


def pin_company(ticker: str) -> CompanyPin:  # pragma: no cover
    """Pin a company's most recent 10-K (annual period, as filed)."""
    from edgar import Company

    _set_identity()
    company = Company(ticker)
    filing = company.get_filings(form="10-K").latest()
    return CompanyPin(
        ticker=ticker.upper(),
        cik=int(company.cik),
        accession=str(filing.accession_no),
        fiscal_period_end=dt.date.fromisoformat(str(filing.period_of_report)),
        form="10-K",
        filed=filing.filing_date,
    )


def pin_corpus(
    tickers: tuple[str, ...] = V01_TICKERS,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> Manifest:  # pragma: no cover
    """Pin any not-yet-pinned tickers into the manifest. Existing pins are never changed."""
    manifest = load_manifest(manifest_path)
    for ticker in tickers:
        if manifest.get(ticker) is None:
            pin = pin_company(ticker)
            manifest.companies.append(pin)
            print(
                f"pinned {pin.ticker}: CIK {pin.cik}, {pin.accession}, "
                f"period end {pin.fiscal_period_end}"
            )
        else:
            print(f"already pinned {ticker.upper()} (pins are immutable once written)")
    save_manifest(manifest, manifest_path)
    return manifest


def fetch_artifacts(
    pin: CompanyPin, cache_root: Path = DEFAULT_CACHE_DIR
) -> Path:  # pragma: no cover
    """Cache-first download of filing.html, text.txt, companyfacts.json for a pin."""
    from edgar import Company
    from edgar.httprequests import download_json

    _set_identity()
    cache = cache_dir_for(pin, cache_root)
    cache.mkdir(parents=True, exist_ok=True)

    html_path = cache / "filing.html"
    text_path = cache / "text.txt"
    facts_path = cache / "companyfacts.json"

    if not (html_path.exists() and text_path.exists()):
        filings = Company(pin.cik).get_filings(form=pin.form, accession_number=pin.accession)
        filing = filings.latest()
        html_path.write_text(filing.html())
        text_path.write_text(filing.text())
        print(f"fetched filing {pin.ticker} {pin.accession}")
    if not facts_path.exists():
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{pin.cik:010d}.json"
        facts_path.write_text(json.dumps(download_json(url)))
        print(f"fetched companyfacts {pin.ticker} CIK {pin.cik}")
    return cache


def load_companyfacts(pin: CompanyPin, cache_root: Path = DEFAULT_CACHE_DIR) -> dict[str, Any]:
    path = cache_dir_for(pin, cache_root) / "companyfacts.json"
    result: dict[str, Any] = json.loads(path.read_text())
    return result


def load_filing_text(pin: CompanyPin, cache_root: Path = DEFAULT_CACHE_DIR) -> str:
    return (cache_dir_for(pin, cache_root) / "text.txt").read_text()
