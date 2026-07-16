"""Grade normalized model answers against XBRL ground truth (architecture.md §3, §4).

The company's own XBRL fact is the auto-grader. Per item (company x KPI x model x cell):
raw output -> parse (normalize.py) -> normalize -> verdict. Verdicts are the frozen four
of §4 plus the `no-ground-truth` exclusion, and the tolerances are the frozen ones of §3:

- monetary KPIs: relative error <= 0.5% (filing text rounds to millions; XBRL is exact)
- per-share KPIs: absolute error <= $0.005 (one half-cent)
- fiscal period end within +/-7 days of the pinned period end, else `wrong-period`
  regardless of value match — a right number for the wrong year is wrong.

The grader assigns only the labels §3 dictates deterministically (`wrong-period`,
`wrong-concept` for non-USD, `refusal`, `format-failure`). `scale-error` and
`hallucination` need the filing context in front of a human and are hand-labeled in the
Day D review loop (§7); this module leaves them to that pass rather than guessing.
"""

from __future__ import annotations

import datetime as dt
from enum import StrEnum
from typing import Any

from pydantic import BaseModel

from filinglens import normalize
from filinglens.corpus import XbrlFact
from filinglens.normalize import Extraction

# Frozen match tolerances (§3).
REL_TOLERANCE = 0.005  # 0.5% relative, monetary KPIs
EPS_ABS_TOLERANCE = 0.005  # half a cent, per-share KPIs
PERIOD_WINDOW_DAYS = 7

# §3's tolerances are inclusive ("<= 0.5%", "<= $0.005"), but neither 0.005 nor most
# decimal figures are exactly representable in binary: 10.005 - 10.0 evaluates to
# 0.005000000000000782, so an EPS answer exactly a half-cent off — which the frozen rule
# calls correct — would be graded incorrect. This guard makes the boundary land where §3
# says it lands. It is ~1e-7 of a cent, orders of magnitude below any distinction that
# matters in a financial statement, so it can only fix representation error and cannot
# widen the tolerance in any way a reader would notice.
_FLOAT_GUARD = 1e-9

# Per-share KPIs are graded on absolute error. Keyed off the XBRL unit rather than the KPI
# name so the deepening roadmap's eps_basic (§2) inherits the rule for free.
PER_SHARE_UNIT = "USD/shares"

REPORTING_CURRENCY = "USD"


class Verdict(StrEnum):
    """The frozen §4 verdict set. `NO_GROUND_TRUTH` items are excluded from accuracy."""

    CORRECT = "correct"
    INCORRECT = "incorrect"
    FORMAT_FAILURE = "format-failure"
    REFUSAL = "refusal"
    NO_GROUND_TRUTH = "no-ground-truth"


class FailureLabel(StrEnum):
    """The frozen §7 taxonomy. Hand-labeled for every incorrect item in v0.1."""

    WRONG_PERIOD = "wrong-period"
    SCALE_ERROR = "scale-error"
    WRONG_CONCEPT = "wrong-concept"
    HALLUCINATION = "hallucination"
    REFUSAL = "refusal"
    FORMAT_FAILURE = "format-failure"


class GradedItem(BaseModel):
    """One graded grid cell, carrying enough evidence to hand-check the verdict later."""

    ticker: str
    kpi: str
    model: str
    cell: str
    verdict: Verdict
    # Only set where §3 dictates it deterministically; None means "for the Day D pass".
    auto_label: FailureLabel | None = None
    predicted: float | None = None  # normalized to absolute USD
    truth: float | None = None
    rel_error: float | None = None
    abs_error: float | None = None
    predicted_period_end: dt.date | None = None
    truth_period_end: dt.date | None = None
    currency: str | None = None
    raw_text: str = ""

    @property
    def scored(self) -> bool:
        """Whether this item counts toward accuracy (§4: no-ground-truth is excluded)."""
        return self.verdict is not Verdict.NO_GROUND_TRUTH


def within_period_window(
    predicted: dt.date | None, truth: dt.date, window_days: int = PERIOD_WINDOW_DAYS
) -> bool:
    """Whether a predicted period end falls within +/-`window_days` of the pinned end.

    A missing date is not within the window. §3 makes the period a condition of
    correctness, so an answer that never states its period has not met it — the harness
    cannot confirm the figure is not the prior-year column, which is the single most
    likely error mode for this task (§7).
    """
    if predicted is None:
        return False
    return abs((predicted - truth).days) <= window_days


def value_matches(predicted: float, truth: float, unit: str) -> bool:
    """Whether a normalized value matches the XBRL fact under the frozen §3 tolerance."""
    if unit == PER_SHARE_UNIT:
        return abs(predicted - truth) <= EPS_ABS_TOLERANCE + _FLOAT_GUARD
    if truth == 0:
        return predicted == 0
    return abs(predicted - truth) / abs(truth) <= REL_TOLERANCE + _FLOAT_GUARD


def _errors(predicted: float, truth: float) -> tuple[float | None, float]:
    abs_error = abs(predicted - truth)
    rel_error = abs_error / abs(truth) if truth != 0 else None
    return rel_error, abs_error


def grade(
    raw_text: str,
    fact: XbrlFact | None,
    *,
    ticker: str,
    kpi: str,
    model: str,
    cell: str,
) -> GradedItem:
    """Grade one raw model answer against its XBRL fact.

    ``fact is None`` means no tag in the frozen chain resolved for the pinned period: the
    item is `no-ground-truth` and excluded from scoring (§2, §4). That exclusion is
    decided before the model's output is even considered — an item with no ground truth
    is unscoreable regardless of what the model said.
    """
    base: dict[str, Any] = {
        "ticker": ticker,
        "kpi": kpi,
        "model": model,
        "cell": cell,
        "raw_text": raw_text,
    }

    if fact is None:
        return GradedItem(verdict=Verdict.NO_GROUND_TRUTH, **base)

    parsed = normalize.parse_output(raw_text)
    if parsed.refused:
        return GradedItem(
            verdict=Verdict.REFUSAL,
            auto_label=FailureLabel.REFUSAL,
            truth=fact.value,
            truth_period_end=fact.end,
            **base,
        )
    if parsed.extraction is None:
        return GradedItem(
            verdict=Verdict.FORMAT_FAILURE,
            auto_label=FailureLabel.FORMAT_FAILURE,
            truth=fact.value,
            truth_period_end=fact.end,
            **base,
        )

    return _grade_extraction(parsed.extraction, fact, base)


def _grade_extraction(extraction: Extraction, fact: XbrlFact, base: dict[str, Any]) -> GradedItem:
    predicted = normalize.normalize_value(extraction)
    rel_error, abs_error = _errors(predicted, fact.value)
    common: dict[str, Any] = {
        "predicted": predicted,
        "truth": fact.value,
        "rel_error": rel_error,
        "abs_error": abs_error,
        "predicted_period_end": extraction.fiscal_period_end,
        "truth_period_end": fact.end,
        "currency": extraction.currency,
        **base,
    }

    # Currency is checked before value: a figure reported in the wrong currency is a
    # different concept, not a near-miss on magnitude (§3).
    if extraction.currency != REPORTING_CURRENCY:
        return GradedItem(
            verdict=Verdict.INCORRECT, auto_label=FailureLabel.WRONG_CONCEPT, **common
        )

    # Period gates the value: right number, wrong year is wrong (§3).
    if not within_period_window(extraction.fiscal_period_end, fact.end):
        return GradedItem(verdict=Verdict.INCORRECT, auto_label=FailureLabel.WRONG_PERIOD, **common)

    if value_matches(predicted, fact.value, fact.unit):
        return GradedItem(verdict=Verdict.CORRECT, **common)
    return GradedItem(verdict=Verdict.INCORRECT, **common)


def accuracy(items: list[GradedItem]) -> float | None:
    """Share of scored items graded `correct`. None when nothing is scoreable.

    Format failures and refusals count against accuracy: they are ways of failing to
    deliver the figure, which is what the harness measures (§4). Only `no-ground-truth`
    is excluded, because there the harness has nothing to grade against.
    """
    scored = [item for item in items if item.scored]
    if not scored:
        return None
    return sum(item.verdict is Verdict.CORRECT for item in scored) / len(scored)
