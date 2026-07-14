import datetime as dt
from pathlib import Path
from typing import Any

from filinglens import corpus

PIN = corpus.CompanyPin(
    ticker="FAKE",
    cik=999999,
    accession="0000999999-25-000001",
    fiscal_period_end=dt.date(2025, 6, 30),
    form="10-K",
    filed=dt.date(2025, 8, 1),
)


def _rec(val: float, **overrides: Any) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "start": "2024-07-01",
        "end": "2025-06-30",
        "val": val,
        "accn": PIN.accession,
        "fy": 2025,
        "fp": "FY",
        "form": "10-K",
        "filed": "2025-08-01",
    }
    rec.update(overrides)
    return rec


def _facts(gaap: dict[str, Any]) -> dict[str, Any]:
    return {"cik": PIN.cik, "entityName": "Fake Corp", "facts": {"us-gaap": gaap}}


REVENUE_KPI = corpus.KpiSpec(
    name="revenue",
    unit="USD",
    tags=[
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
    ],
)
ASSETS_KPI = corpus.KpiSpec(name="total_assets", unit="USD", tags=["Assets"])


def test_manifest_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "corpus.yaml"
    corpus.save_manifest(corpus.Manifest(companies=[PIN]), path)
    loaded = corpus.load_manifest(path)
    assert loaded.companies == [PIN]
    assert loaded.get("fake") == PIN
    assert loaded.get("MISSING") is None


def test_load_manifest_missing_file_is_empty(tmp_path: Path) -> None:
    assert corpus.load_manifest(tmp_path / "nope.yaml").companies == []


def test_load_kpis_frozen_v01() -> None:
    kpis = corpus.load_kpis(Path(__file__).parent.parent / "kpis.yaml")
    assert [k.name for k in kpis] == [
        "revenue",
        "net_income",
        "eps_diluted",
        "total_assets",
        "operating_cash_flow",
    ]
    assert kpis[0].tags[0] == "RevenueFromContractWithCustomerExcludingAssessedTax"
    assert kpis[2].unit == "USD/shares"


def test_resolve_fact_first_tag_wins() -> None:
    facts = _facts(
        {
            "RevenueFromContractWithCustomerExcludingAssessedTax": {
                "units": {"USD": [_rec(100.0)]}
            },
            "Revenues": {"units": {"USD": [_rec(999.0)]}},
        }
    )
    fact = corpus.resolve_fact(facts, PIN, REVENUE_KPI)
    assert fact is not None
    assert fact.tag == "RevenueFromContractWithCustomerExcludingAssessedTax"
    assert fact.value == 100.0


def test_resolve_fact_falls_back_down_the_chain() -> None:
    facts = _facts({"Revenues": {"units": {"USD": [_rec(42.0)]}}})
    fact = corpus.resolve_fact(facts, PIN, REVENUE_KPI)
    assert fact is not None
    assert fact.tag == "Revenues"
    assert fact.end == PIN.fiscal_period_end


def test_resolve_fact_filters_wrong_accession_period_and_form() -> None:
    facts = _facts(
        {
            "Revenues": {
                "units": {
                    "USD": [
                        _rec(1.0, accn="0000999999-24-000001"),
                        _rec(2.0, start="2023-07-01", end="2024-06-30"),
                        _rec(3.0, form="10-Q"),
                    ]
                }
            }
        }
    )
    assert corpus.resolve_fact(facts, PIN, REVENUE_KPI) is None


def test_resolve_fact_excludes_quarterly_duration_same_end() -> None:
    facts = _facts({"Revenues": {"units": {"USD": [_rec(25.0, start="2025-04-01"), _rec(100.0)]}}})
    fact = corpus.resolve_fact(facts, PIN, REVENUE_KPI)
    assert fact is not None
    assert fact.value == 100.0


def test_resolve_fact_instant_fact_has_no_start() -> None:
    rec = _rec(5000.0)
    del rec["start"]
    facts = _facts({"Assets": {"units": {"USD": [rec]}}})
    fact = corpus.resolve_fact(facts, PIN, ASSETS_KPI)
    assert fact is not None
    assert fact.start is None
    assert fact.value == 5000.0


def test_resolve_fact_no_ground_truth() -> None:
    assert corpus.resolve_fact(_facts({}), PIN, REVENUE_KPI) is None


def test_cache_layout() -> None:
    assert corpus.cache_dir_for(PIN, Path("data/cache")) == Path(
        "data/cache/999999/0000999999-25-000001"
    )


def test_load_companyfacts_and_text_from_cache(tmp_path: Path) -> None:
    cache = corpus.cache_dir_for(PIN, tmp_path)
    cache.mkdir(parents=True)
    (cache / "companyfacts.json").write_text('{"facts": {}}')
    (cache / "text.txt").write_text("Total revenue was $100")
    assert corpus.load_companyfacts(PIN, tmp_path) == {"facts": {}}
    assert corpus.load_filing_text(PIN, tmp_path) == "Total revenue was $100"
