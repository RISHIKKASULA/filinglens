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
