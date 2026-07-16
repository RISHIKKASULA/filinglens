"""Item-heading section slicing for 10-K filing text (architecture.md §5).

Cell A of the strategy grid feeds the model the Item 8 statements section ("Financial
Statements and Supplementary Data"), trimmed to a context budget. Slicing is a
line-anchored Item-heading regex over the extracted text: the real body heading is the
*last* occurrence of ``Item N`` (the table of contents lists it first, prose cross-refs
like "Part II, Item 8" are not line-anchored), and the section ends at the first heading
for a higher-numbered item. When no Item 8 heading resolves the slice is marked not-found
and the whole document is returned — the BM25 retrieval arm absorbs messy documents.

**Incorporation by reference (ADR-002).** Item 8 is not always where the statements are.
NVDA and XOM answer Item 8 with a one-paragraph cross-reference — "The information
required by this Item is set forth in our Consolidated Financial Statements ... included
in this Annual Report" — and file the statements elsewhere: NVDA under Item 15, XOM in a
Financial Section that follows Item 16 entirely. Slicing Item 8 on those filings returns
a 207- and 706-character stub, so cell A handed the model an empty context and every
answer was graded a model failure. §5's prescribed whole-doc fallback does not help, as
it only fires when the heading is *absent* and head-trimming a whole 10-K yields its
table of contents.

So a stub falls back to locating the primary statements directly, by their own heading.
The discriminator is that a real statement heading stands alone on its line and is
followed within a few lines by thousands-separated figures; an index entry carries a
trailing page number, a statement index carries trailing dates, and a prose
cross-reference sits inside a sentence. Filers whose Item 8 already contains the
statements are untouched by this path.
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

# The primary statement's own heading, standing alone on its line (filers pad the line
# with whitespace). Requiring end-of-line is what rejects the decoys: a financial-TOC
# entry ends in a page number ("Statement of Income      7"), a statement index ends in
# dates ("Consolidated Statements of Income for the years ended January 25, 2026..."),
# and a prose cross-reference is embedded mid-sentence.
_STATEMENT_HEADING = re.compile(
    r"^[ \t]*(?:consolidated[ \t]+)?statements?[ \t]+of[ \t]+"
    r"(?:income|operations|earnings)[ \t]*$",
    re.IGNORECASE,
)

# A statement row carries thousands-separated figures. Used to confirm a candidate
# heading is the statement itself and not another list of its name.
_FIGURE = re.compile(r"\d{1,3},\d{3}")

# How far past a candidate heading to look for those figures. The real statement reaches
# its first numeric row within a few lines (scale marker, column headers, then rows).
_FIGURE_LOOKAHEAD = 20

# A statements context below this is not a statements section — it is a stub, and a stub
# silently turns every model answer into a fake failure. Callers use `validate_statements`
# to fail loudly rather than let that reach the grid (ADR-002).
MIN_STATEMENTS_CHARS = 5_000


class StatementsSliceError(RuntimeError):
    """A filing whose statements context is too small to contain its statements."""


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


def find_statements_start(lines: list[str]) -> int | None:
    """Index of the line where the primary statements begin, or None.

    A candidate is a standalone statement heading; it is accepted only if
    thousands-separated figures follow within ``_FIGURE_LOOKAHEAD`` lines, which is what
    separates the statement itself from a financial-TOC entry or a statement index that
    merely names it. The first accepted candidate wins: the income statement leads the
    primary statements, and the balance sheet and cash flows follow it contiguously.
    """
    for i, line in enumerate(lines):
        if _STATEMENT_HEADING.match(line) is None:
            continue
        window = lines[i + 1 : i + 1 + _FIGURE_LOOKAHEAD]
        if any(_FIGURE.search(w) for w in window):
            return i
    return None


def _has_statements(text: str) -> bool:
    """Whether a slice actually contains a primary statement, not just a pointer to one."""
    return find_statements_start(text.splitlines()) is not None


def statements_section(text: str, char_budget: int = 40_000) -> SectionSlice:
    """The statements section, head-trimmed to a context budget.

    Normally this is the Item 8 slice: the primary statements (income statement, balance
    sheet, cash flows) sit at the top of Item 8, right after its index, so head-trimming
    keeps them while bounding the prompt. When Item 8 is only a cross-reference — the
    incorporation-by-reference filers of ADR-002 — the statements are found by their own
    heading wherever they were actually filed, and the slice runs from there.

    The default budget stays well inside the frozen num_ctx of 16384 tokens. The returned
    slice's ``text`` is at most ``char_budget`` characters.
    """
    sliced = slice_item(text, STATEMENTS_ITEM)

    if not _has_statements(sliced.text):
        lines = text.splitlines()
        start = find_statements_start(lines)
        if start is not None:
            sliced = SectionSlice(
                item=STATEMENTS_ITEM,
                found=True,
                text="\n".join(lines[start:]).strip(),
                start_line=start + 1,
                end_line=len(lines) + 1,
            )

    if len(sliced.text) > char_budget:
        sliced = sliced.model_copy(update={"text": sliced.text[:char_budget]})
    return sliced


def validate_statements(
    text: str, label: str = "filing", min_chars: int = MIN_STATEMENTS_CHARS
) -> SectionSlice:
    """The statements section, or raise if it is too small to be one.

    The failure this guards against is silent and one-directional: a stub context makes a
    model look wrong for honestly reporting it cannot find a figure that was never in
    front of it, and those fake failures land on one arm of the ablation. Better to stop
    the run than to publish that as a measurement (ADR-002).
    """
    sliced = statements_section(text)
    if len(sliced.text) < min_chars:
        raise StatementsSliceError(
            f"{label}: statements context is {len(sliced.text)} chars "
            f"(minimum {min_chars}). Item 8 is likely a cross-reference and the "
            f"statements were not located — refusing to grade a model on an empty context."
        )
    return sliced
