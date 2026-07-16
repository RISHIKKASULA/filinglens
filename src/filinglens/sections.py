"""Item-heading section slicing for 10-K filing text (architecture.md §5).

Cell A of the strategy grid feeds the model the Item 8 statements section ("Financial
Statements and Supplementary Data"), trimmed to a context budget. Slicing is a
line-anchored Item-heading regex over the extracted text: the real body heading is the
*last* occurrence of ``Item N`` (the table of contents lists it first, prose cross-refs
like "Part II, Item 8" are not line-anchored), and the section ends at the first heading
for a higher-numbered item. When no Item 8 heading resolves the slice is marked not-found
and the whole document is returned — the BM25 retrieval arm absorbs messy documents.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

# A heading line: optional indent, then "Item <num><letter>" followed by a period, colon,
# dash, or end-of-token. Anchored to line start so prose cross-references ("...in Part II,
# Item 8 of this Form 10-K") never match.
_HEADING = re.compile(r"^[ \t]*Item[ \t]+(\d{1,2})([A-Za-z]?)[ \t]*[.:—-]", re.IGNORECASE)

# Item 8 of a 10-K is the financial statements. Kept as a constant so callers read clearly.
STATEMENTS_ITEM = "8"


class SectionSlice(BaseModel):
    """A sliced section (or the whole-doc fallback when the heading was not found)."""

    item: str
    found: bool
    text: str
    start_line: int  # 1-indexed; 0 when not found (whole-doc fallback)
    end_line: int  # 1-indexed, exclusive; 0 when not found


class _Heading(BaseModel):
    key: tuple[int, str]
    line_no: int  # 0-indexed into the lines list


def _item_key(item: str) -> tuple[int, str]:
    """Order key for an item token like "8", "9A": (number, uppercased letter)."""
    m = re.fullmatch(r"(\d{1,2})([A-Za-z]?)", item.strip())
    if m is None:
        raise ValueError(f"not an item token: {item!r}")
    return int(m.group(1)), m.group(2).upper()


def _headings(lines: list[str]) -> list[_Heading]:
    found: list[_Heading] = []
    for i, line in enumerate(lines):
        m = _HEADING.match(line)
        if m is not None:
            found.append(_Heading(key=(int(m.group(1)), m.group(2).upper()), line_no=i))
    return found


def slice_item(text: str, item: str = STATEMENTS_ITEM) -> SectionSlice:
    """Slice the named Item's section out of filing text.

    The section starts at the last line-anchored heading for ``item`` (skipping the table
    of contents, which lists every item before the body) and ends just before the first
    heading for any higher-numbered item that follows it. Falls back to the whole document
    (``found=False``) when no such heading is present.
    """
    target = _item_key(item)
    lines = text.splitlines()
    headings = _headings(lines)

    starts = [h for h in headings if h.key == target]
    if not starts:
        return SectionSlice(item=item, found=False, text=text, start_line=0, end_line=0)
    start = starts[-1].line_no

    end = len(lines)
    for h in headings:
        if h.line_no > start and h.key > target:
            end = h.line_no
            break

    return SectionSlice(
        item=item,
        found=True,
        text="\n".join(lines[start:end]).strip(),
        start_line=start + 1,
        end_line=end + 1,
    )


def statements_section(text: str, char_budget: int = 40_000) -> SectionSlice:
    """The Item 8 statements section, head-trimmed to a context budget.

    The primary statements (income statement, balance sheet, cash flows) sit at the top of
    Item 8, right after its index, so head-trimming keeps them while bounding the prompt.
    The default budget stays well inside the frozen num_ctx of 16384 tokens; on the pinned
    AAPL and MSFT filings all five KPI figures survive the trim. The returned slice's
    ``text`` is at most ``char_budget`` characters.
    """
    sliced = slice_item(text, STATEMENTS_ITEM)
    if len(sliced.text) > char_budget:
        sliced = sliced.model_copy(update={"text": sliced.text[:char_budget]})
    return sliced
