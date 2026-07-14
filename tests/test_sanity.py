import datetime as dt
from pathlib import Path

from filinglens import corpus, sanity

FACT = corpus.XbrlFact(
    kpi="revenue",
    tag="Revenues",
    unit="USD",
    value=391_035_000_000.0,
    start=dt.date(2024, 7, 1),
    end=dt.date(2025, 6, 30),
    accession="0000999999-25-000001",
    fiscal_year=2025,
    fiscal_period="FY",
    form="10-K",
)


def test_search_patterns_monetary_reported_and_millions() -> None:
    assert sanity.search_patterns(391_035_000_000.0, "USD") == ["391,035,000,000", "391,035"]


def test_search_patterns_eps_two_decimals() -> None:
    assert sanity.search_patterns(6.08, "USD/shares") == ["6.08"]


def test_search_patterns_dedupes_identical_scales() -> None:
    assert sanity.search_patterns(1_000_000.0, "USD") == ["1,000,000", "1"]


def test_find_candidate_lines_matches_and_caps() -> None:
    text = "\n".join(
        ["preamble", "Total net sales 391,035 383,285", "no numbers here"]
        + ["Total revenue was 391,035 again"] * 10
    )
    lines = sanity.find_candidate_lines(text, ["391,035"], max_lines=3)
    assert [line.line_no for line in lines] == [2, 4, 5]
    assert lines[0].snippet == "Total net sales 391,035 383,285"


def test_find_candidate_lines_snippet_trims_long_lines() -> None:
    text = ("x" * 200) + " 391,035 " + ("y" * 200)
    lines = sanity.find_candidate_lines(text, ["391,035"])
    assert len(lines) == 1
    assert "391,035" in lines[0].snippet
    assert len(lines[0].snippet) <= 112  # width + ellipses


def test_find_candidate_lines_no_match() -> None:
    assert sanity.find_candidate_lines("nothing to see", ["391,035"]) == []


def test_build_pairs_from_cache(tmp_path: Path) -> None:
    pin = corpus.CompanyPin(
        ticker="FAKE",
        cik=999999,
        accession="0000999999-25-000001",
        fiscal_period_end=dt.date(2025, 6, 30),
        form="10-K",
        filed=dt.date(2025, 8, 1),
    )
    cache = corpus.cache_dir_for(pin, tmp_path)
    cache.mkdir(parents=True)
    (cache / "companyfacts.json").write_text(
        """
        {"facts": {"us-gaap": {"Revenues": {"units": {"USD": [
            {"start": "2024-07-01", "end": "2025-06-30", "val": 391035000000,
             "accn": "0000999999-25-000001", "fy": 2025, "fp": "FY", "form": "10-K"}
        ]}}}}}
        """
    )
    (cache / "text.txt").write_text("Total net sales 391,035\n")
    kpi = corpus.KpiSpec(name="revenue", unit="USD", tags=["Revenues"])
    missing_kpi = corpus.KpiSpec(name="total_assets", unit="USD", tags=["Assets"])
    pairs = sanity.build_pairs([pin], [kpi, missing_kpi], tmp_path)
    assert len(pairs) == 2
    assert pairs[0].fact is not None and pairs[0].candidates[0].line_no == 1
    assert pairs[1].fact is None and pairs[1].candidates == []


def test_render_report_snapshot() -> None:
    """The sanity output is the §0 gate artifact; its format is snapshot-tested."""
    pairs = [
        sanity.SanityPair(
            ticker="FAKE",
            kpi="revenue",
            fact=FACT,
            patterns=["391,035,000,000", "391,035"],
            candidates=[sanity.CandidateLine(line_no=42, snippet="Total net sales 391,035")],
        ),
        sanity.SanityPair(ticker="FAKE", kpi="total_assets", fact=None, patterns=[], candidates=[]),
        sanity.SanityPair(
            ticker="FAKE",
            kpi="eps_diluted",
            fact=FACT.model_copy(
                update={"kpi": "eps_diluted", "unit": "USD/shares", "value": 6.08}
            ),
            patterns=["6.08"],
            candidates=[],
        ),
    ]
    expected = "\n".join(
        [
            "=" * 100,
            "[1/3] FAKE — revenue",
            "  XBRL: 391,035,000,000.00 USD  [Revenues]",
            "        period 2024-07-01 → 2025-06-30 (FY2025)  accession 0000999999-25-000001",
            '  search: "391,035,000,000" | "391,035"',
            "  text L42: Total net sales 391,035",
            "=" * 100,
            "[2/3] FAKE — total_assets",
            "  XBRL: no-ground-truth (no tag in the chain resolved for the pinned period)",
            "=" * 100,
            "[3/3] FAKE — eps_diluted",
            "  XBRL: 6.08 USD/shares  [Revenues]",
            "        period 2024-07-01 → 2025-06-30 (FY2025)  accession 0000999999-25-000001",
            '  search: "6.08"',
            "  text: NO MATCH — figure not found in filing text at searched scales",
            "=" * 100,
            "1/3 pairs have candidate lines. Gate (§0): >= 9/10 clean matches on MANUAL review "
            "(right period, right scale, figure findable in text).",
        ]
    )
    assert sanity.render_report(pairs) == expected
