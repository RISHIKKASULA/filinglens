import pytest

from filinglens import sections

# A mini 10-K shape: a table of contents that lists every item first, prose that
# cross-references "Item 8" mid-line, then the real body headings. The slicer must land on
# the body Item 8 (not the TOC entry, not the cross-reference) and stop at Item 9.
MINI_10K = """\
PART II
  Item 7.    Management's Discussion and Analysis                     21
  Item 8.    Financial Statements and Supplementary Data              28
  Item 9.    Changes in and Disagreements with Accountants            52
  Item 9A.   Controls and Procedures                                  52

Item 7. Management's Discussion and Analysis
Refer to the consolidated financial statements in Part II, Item 8 of this Form 10-K.
Revenue discussion goes here.

Item 8. Financial Statements and Supplementary Data
Consolidated Statements of Operations
Total net sales 416,161 391,035 383,285
Net income 112,010 93,736 96,995

Item 9. Changes in and Disagreements with Accountants
None.

Item 9A. Controls and Procedures
Effective.
"""


def test_slice_item8_picks_body_not_toc_or_crossref() -> None:
    s = sections.slice_item(MINI_10K, "8")
    assert s.found
    assert s.text.startswith("Item 8. Financial Statements")
    assert "416,161" in s.text and "112,010" in s.text
    # Bounded by Item 9: excludes the MD&A above and the controls section below.
    assert "Revenue discussion" not in s.text
    assert "Item 9." not in s.text
    assert "Effective." not in s.text


def test_slice_item8_line_bounds() -> None:
    s = sections.slice_item(MINI_10K, "8")
    lines = MINI_10K.splitlines()
    assert lines[s.start_line - 1].startswith("Item 8.")
    assert lines[s.end_line - 1].startswith("Item 9.")


def test_slice_item7_stops_at_item8() -> None:
    s = sections.slice_item(MINI_10K, "7")
    assert s.found
    assert "Revenue discussion goes here." in s.text
    assert "Financial Statements" not in s.text.split("\n", 1)[1]


def test_slice_missing_item_falls_back_to_whole_doc() -> None:
    s = sections.slice_item(MINI_10K, "12")
    assert not s.found
    assert s.text == MINI_10K
    assert s.start_line == 0 and s.end_line == 0


def test_slice_is_case_insensitive_uppercase_heading() -> None:
    text = "ITEM 8. FINANCIAL STATEMENTS\nrevenue 100\nITEM 9. OTHER\nx"
    s = sections.slice_item(text, "8")
    assert s.found
    assert "revenue 100" in s.text
    assert "OTHER" not in s.text


def test_statements_section_head_trims_to_budget() -> None:
    s = sections.statements_section(MINI_10K, char_budget=20)
    assert s.found
    assert len(s.text) == 20


def test_statements_section_under_budget_is_untrimmed() -> None:
    full = sections.slice_item(MINI_10K, "8")
    s = sections.statements_section(MINI_10K, char_budget=10_000)
    assert s.text == full.text


def test_item_key_ordering() -> None:
    assert sections._item_key("8") == (8, "")
    assert sections._item_key("9A") == (9, "A")
    assert sections._item_key("8") < sections._item_key("9A")
    assert sections._item_key("9") < sections._item_key("9A")


def test_item_key_rejects_malformed_token() -> None:
    with pytest.raises(ValueError, match="not an item token"):
        sections._item_key("nope")


# --- incorporation by reference (ADR-002) -------------------------------------------

# The NVDA/XOM shape: Item 8 answers with a cross-reference and the statements are filed
# elsewhere — here under Item 15, past a financial TOC and an auditor's report. The
# decoys are deliberate: a TOC entry ending in a page number, a statement index ending in
# dates, and a prose mention mid-sentence. Only the real heading is followed by figures.
BY_REFERENCE_10K = """\
PART II
  Item 8.    Financial Statements and Supplementary Data              28

Financial Table of Contents
  Statement of Income                                                  7
  Balance Sheet                                                        7

Item 7. Management's Discussion and Analysis
As discussed in our Consolidated Statement of Income, revenue grew.

Item 8. Financial Statements and Supplementary Data

The information required by this Item is set forth in our Consolidated Financial
Statements and Notes thereto included in this Annual Report on Form 10-K.

Item 9. Changes in and Disagreements with Accountants
None.

Item 15. Exhibits and Financial Statement Schedules

Consolidated Statements of Income for the years ended January 25, 2026, January 26, 2025
Consolidated Balance Sheets as of January 25, 2026 and January 26, 2025

Report of Independent Registered Public Accounting Firm
We have audited the accompanying consolidated balance sheets.

Consolidated Statements of Income
(In millions, except per share data)
Year Ended
Revenue                                       215,938        130,497
Net income                                    120,067         72,880

Consolidated Balance Sheets
Total assets                                  206,803        111,601
"""


def test_item8_stub_is_what_the_item_slicer_returns() -> None:
    # The slicer is not wrong — the stub genuinely *is* Item 8. This is the bug's root:
    # a correct slice of a section that does not hold the statements.
    s = sections.slice_item(BY_REFERENCE_10K, "8")
    assert s.found
    assert "information required by this Item" in s.text
    assert "215,938" not in s.text
    assert len(s.text) < 300


def test_statements_section_falls_back_to_the_real_statements() -> None:
    ctx = sections.statements_section(BY_REFERENCE_10K).text
    assert "215,938" in ctx  # revenue
    assert "120,067" in ctx  # net income
    assert "206,803" in ctx  # total assets
    assert "information required by this Item" not in ctx  # not the stub


def test_fallback_anchors_on_the_statement_not_the_decoys() -> None:
    ctx = sections.statements_section(BY_REFERENCE_10K).text
    assert ctx.startswith("Consolidated Statements of Income")
    # the financial-TOC entry, the statement index, and the prose mention are all above
    # the anchor and must not be swept in
    assert "Financial Table of Contents" not in ctx
    assert "for the years ended" not in ctx
    assert "As discussed in our" not in ctx


def test_toc_entry_with_a_page_number_is_not_a_statement_heading() -> None:
    lines = ["  Statement of Income                        7", "  Balance Sheet         7"]
    assert sections.find_statements_start(lines) is None


def test_statement_index_entry_with_trailing_dates_is_rejected() -> None:
    lines = [
        "Consolidated Statements of Income for the years ended January 25, 2026",
        "Consolidated Balance Sheets as of January 25, 2026 and January 26, 2025",
    ]
    assert sections.find_statements_start(lines) is None


def test_prose_cross_reference_is_rejected() -> None:
    lines = ["As discussed in our Consolidated Statement of Income, revenue grew 12,345."]
    assert sections.find_statements_start(lines) is None


def test_heading_without_following_figures_is_rejected() -> None:
    # A heading alone proves nothing; the statement is where the numbers are.
    lines = ["Consolidated Statements of Income", "", "See the following pages."]
    assert sections.find_statements_start(lines) is None


def test_heading_with_following_figures_is_accepted() -> None:
    lines = ["Consolidated Statements of Income", "(In millions)", "Revenue  215,938"]
    assert sections.find_statements_start(lines) == 0


@pytest.mark.parametrize(
    "heading",
    [
        "Consolidated Statements of Income",
        "CONSOLIDATED STATEMENT OF INCOME",
        "Consolidated Statements of Operations",
        "Statement of Earnings",
        "   Consolidated Statements of Income   ",
    ],
)
def test_statement_heading_spellings(heading: str) -> None:
    assert sections.find_statements_start([heading, "Revenue 215,938"]) == 0


def test_normal_filing_does_not_use_the_fallback() -> None:
    # A filer whose Item 8 holds its statements must be untouched by the ADR-002 path.
    ctx = sections.statements_section(MINI_10K).text
    assert ctx.startswith("Item 8. Financial Statements")


# --- the guard (ADR-002) ------------------------------------------------------------


def test_validate_statements_passes_a_real_section() -> None:
    text = "Item 8. Financial Statements\nConsolidated Statements of Income\nRevenue 1,234\n"
    text += "x" * 6000
    assert sections.validate_statements(text, label="OK").text


def test_validate_statements_raises_on_a_stub() -> None:
    stub = (
        "Item 8. Financial Statements and Supplementary Data\n"
        "The information required by this Item is set forth elsewhere.\n"
        "Item 9. Changes\n"
    )
    with pytest.raises(sections.StatementsSliceError, match="empty context"):
        sections.validate_statements(stub, label="NVDA")


def test_validate_statements_names_the_filing_and_the_size() -> None:
    stub = "Item 8. Financials\nSee elsewhere.\nItem 9. Changes\n"
    with pytest.raises(sections.StatementsSliceError) as exc:
        sections.validate_statements(stub, label="XOM")
    assert "XOM" in str(exc.value)
    assert str(sections.MIN_STATEMENTS_CHARS) in str(exc.value)
