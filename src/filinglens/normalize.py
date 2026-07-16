"""Parse and normalize raw model output into an absolute-USD figure (architecture.md §3).

The model is asked for ``{kpi, value, scale, currency, fiscal_period_end}`` (the contract
frozen in prompts.py). This module turns raw model text into that structure and applies
the frozen normalization: ``value x scale -> absolute USD``. Parsing is deliberately
lenient — the free-form arm of the 2x2 ablation (§5) asks for bare JSON and gets prose,
code fences, and thousands separators — because a parse that fails where a human could
read the answer would inflate `format-failure` and understate model accuracy.

What this module does NOT do is decide correctness; that is grade.py. The split matters:
normalization is a pure function of the model's output, while grading needs the XBRL fact.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from typing import Any

from pydantic import BaseModel

# Frozen scale vocabulary (§3). value x factor -> absolute units.
SCALE_FACTORS: dict[str, float] = {
    "units": 1.0,
    "thousands": 1e3,
    "millions": 1e6,
    "billions": 1e9,
}

# Words a model may use in the `scale` field instead of the enum. Mapped rather than
# rejected: the figure is right and the intent is unambiguous, so rejecting it would be a
# grader artifact, not a model error.
#
# Order matters and is load-bearing. These are scanned in sequence against free text, so
# the magnitude words must come before the units synonyms: "USD millions" has to read as
# millions, not as units via the bare "usd". Scanning a dict in insertion order made that
# a silent 1e6 error in the model's favour or against it, depending on phrasing.
_SCALE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("billions", "billions"),
    ("billion", "billions"),
    ("bn", "billions"),
    ("b", "billions"),
    ("millions", "millions"),
    ("million", "millions"),
    ("mm", "millions"),
    ("m", "millions"),
    ("thousands", "thousands"),
    ("thousand", "thousands"),
    ("k", "thousands"),
    ("units", "units"),
    ("unit", "units"),
    ("ones", "units"),
    ("one", "units"),
    ("actual", "units"),
    ("absolute", "units"),
    ("dollars", "units"),
    ("usd", "units"),
)

# Phrases that mark a genuine refusal ("the figure isn't here") rather than a malformed
# answer. Kept narrow and matched only when no figure parsed, so a model that answers and
# hedges ("I could not find it stated directly, but total net sales were 416,161") is
# graded on its figure, not on the hedge.
_REFUSAL_MARKERS: tuple[str, ...] = (
    "cannot find",
    "can't find",
    "could not find",
    "couldn't find",
    "not found",
    "unable to find",
    "unable to determine",
    "cannot determine",
    "not provided",
    "not present",
    "not available",
    "not stated",
    "not included",
    "does not contain",
    "doesn't contain",
    "no information",
    "insufficient information",
    "i cannot",
    "i can't",
    "i'm sorry",
    "i am sorry",
)

_JSON_OBJECT_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)
_NUMBER_RE = re.compile(r"-?\(?\$?\s*-?[\d,]+(?:\.\d+)?\)?")


class Extraction(BaseModel):
    """A parsed model answer, before normalization. Field values are as the model gave them."""

    kpi: str | None = None
    value: float
    scale: str
    currency: str
    fiscal_period_end: dt.date | None = None


class ParseResult(BaseModel):
    """Outcome of parsing raw model text.

    Exactly one of ``extraction`` / ``refused`` / neither is meaningful: an extraction
    means a figure was recovered; ``refused`` means the model declined; neither means the
    output was unparseable (a `format-failure` in §4 terms).
    """

    extraction: Extraction | None = None
    refused: bool = False

    @property
    def is_format_failure(self) -> bool:
        return self.extraction is None and not self.refused


def parse_number(raw: Any) -> float | None:
    """Parse a number from a model's `value` field, JSON number or human-typed string.

    Handles the accounting negative-parentheses convention (§3), thousands separators,
    and a leading currency symbol. Returns None if no number is recoverable.
    """
    if isinstance(raw, bool):  # bool is an int subclass; never a figure
        return None
    if isinstance(raw, int | float):
        return float(raw)
    if not isinstance(raw, str):
        return None

    text = raw.strip()
    if not text:
        return None
    # Parentheses mean negative in financial statements: (1,234) == -1234.
    negative = "(" in text and ")" in text
    cleaned = text.replace("(", "").replace(")", "")
    cleaned = cleaned.replace("$", "").replace(",", "").replace("_", "").strip()
    # A trailing/leading minus survives; strip stray spaces inside the sign.
    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if match is None:
        return None
    value = float(match.group())
    if negative:
        value = -abs(value)
    return value


def parse_scale(raw: Any, fallback: str = "units") -> str:
    """Map a model's `scale` field onto the frozen enum, tolerating common synonyms.

    Handles both the bare enum and the free text models actually emit ("in millions",
    "USD millions", "$ in thousands"), scanning _SCALE_PATTERNS in its declared order so
    the magnitude wins over an incidental currency word.
    """
    if not isinstance(raw, str):
        return fallback
    key = raw.strip().lower().rstrip(".")
    if key in SCALE_FACTORS:
        return key
    for word, canonical in _SCALE_PATTERNS:
        if re.search(rf"\b{re.escape(word)}\b", key):
            return canonical
    return fallback


def parse_date(raw: Any) -> dt.date | None:
    """Parse `fiscal_period_end`. ISO first, then the formats models actually emit."""
    if isinstance(raw, dt.date):
        return raw
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text:
        return None
    iso = re.search(r"\d{4}-\d{2}-\d{2}", text)
    if iso:
        try:
            return dt.date.fromisoformat(iso.group())
        except ValueError:
            pass
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%m/%d/%Y", "%d %B %Y", "%Y/%m/%d"):
        try:
            return dt.datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _is_refusal(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _REFUSAL_MARKERS)


def _candidate_objects(text: str) -> list[dict[str, Any]]:
    """JSON objects in raw model text, best candidate first.

    Tries the whole string (the schema-constrained arm returns exactly one object), then
    braces-delimited spans (the free-form arm wraps JSON in prose or ``` fences).
    """
    candidates: list[dict[str, Any]] = []
    stripped = text.strip()
    # Strip code fences if the whole body is fenced.
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, re.DOTALL)
    if fenced:
        stripped = fenced.group(1).strip()
    for blob in (stripped, *_JSON_OBJECT_RE.findall(text)):
        try:
            obj = json.loads(blob)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(obj, dict) and obj not in candidates:
            candidates.append(obj)
    return candidates


def parse_output(text: str) -> ParseResult:
    """Parse raw model text into a ParseResult.

    A model answer counts as an extraction only if a numeric value is recoverable — the
    field the verdict turns on. Missing `scale` defaults to ``units`` (an unstated scale
    means the figure is as-printed); missing `currency` defaults to USD (the corpus is
    US filers reporting in USD, so absence is not evidence of a foreign currency).
    """
    if not text or not text.strip():
        return ParseResult(refused=False)  # empty output is a format-failure, not a refusal

    for obj in _candidate_objects(text):
        if "value" not in obj:
            continue
        value = parse_number(obj["value"])
        if value is None:
            continue
        return ParseResult(
            extraction=Extraction(
                kpi=obj.get("kpi") if isinstance(obj.get("kpi"), str) else None,
                value=value,
                scale=parse_scale(obj.get("scale")),
                currency=str(obj.get("currency") or "USD").strip().upper(),
                fiscal_period_end=parse_date(obj.get("fiscal_period_end")),
            )
        )

    # No usable JSON. A refusal is a model behaviour worth distinguishing from a mangled
    # answer (§7 separates `refusal` from `format-failure`), so check markers before
    # falling back to scraping a bare number out of prose.
    if _is_refusal(text):
        return ParseResult(refused=True)
    return ParseResult(refused=False)


def normalize_value(extraction: Extraction) -> float:
    """Apply the frozen normalization: value x scale -> absolute units (§3)."""
    return extraction.value * SCALE_FACTORS[extraction.scale]
