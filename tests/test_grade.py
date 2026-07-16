import datetime as dt
from typing import Any

import pytest

from filinglens import grade
from filinglens.corpus import XbrlFact
from filinglens.grade import FailureLabel, Verdict

PERIOD_END = dt.date(2025, 9, 27)


def _fact(value: float = 416_161_000_000.0, unit: str = "USD", kpi: str = "revenue") -> XbrlFact:
    return XbrlFact(
        kpi=kpi,
        tag="RevenueFromContractWithCustomerExcludingAssessedTax",
        unit=unit,
        value=value,
        start=dt.date(2024, 9, 29),
        end=PERIOD_END,
        accession="0000320193-25-000079",
        fiscal_year=2025,
        fiscal_period="FY",
        form="10-K",
    )


def _eps_fact(value: float = 7.46) -> XbrlFact:
    return _fact(value=value, unit="USD/shares", kpi="eps_diluted")


def _answer(
    value: str | float = 416161,
    scale: str = "millions",
    currency: str = "USD",
    period: str = "2025-09-27",
) -> str:
    return (
        f'{{"kpi": "revenue", "value": {value}, "scale": "{scale}", '
        f'"currency": "{currency}", "fiscal_period_end": "{period}"}}'
    )


# A sentinel, so `fact=None` (the no-ground-truth case) stays distinguishable from
# "caller didn't pass one" — a plain None default silently swallows it.
_DEFAULT: Any = object()


def _grade(raw: str, fact: XbrlFact | None = _DEFAULT) -> grade.GradedItem:
    return grade.grade(
        raw,
        _fact() if fact is _DEFAULT else fact,
        ticker="AAPL",
        kpi="revenue",
        model="stub",
        cell="A1",
    )


# --- the happy path -----------------------------------------------------------------


def test_correct_answer_rounded_to_millions() -> None:
    # The filing text prints 416,161 (millions); XBRL holds the exact 416,161,000,000.
    item = _grade(_answer())
    assert item.verdict is Verdict.CORRECT
    assert item.auto_label is None
    assert item.predicted == 416_161_000_000.0
    assert item.rel_error == 0.0


def test_correct_answer_at_units_scale() -> None:
    item = _grade(_answer(value=416161000000, scale="units"))
    assert item.verdict is Verdict.CORRECT


# --- value tolerance: 0.5% relative, monetary (§3) ----------------------------------


def test_value_matches_inside_relative_tolerance() -> None:
    assert grade.value_matches(1004.0, 1000.0, "USD")


def test_value_matches_exactly_at_the_relative_boundary() -> None:
    # 0.5% of 1000 is exactly 5; the frozen rule is <=, so this is a match.
    assert grade.value_matches(1005.0, 1000.0, "USD")


def test_value_matches_just_outside_the_relative_boundary() -> None:
    assert not grade.value_matches(1006.0, 1000.0, "USD")


def test_relative_tolerance_absorbs_millions_rounding_at_apple_scale() -> None:
    # The §3 rationale: text rounded to millions must clear the bar against exact XBRL.
    assert grade.value_matches(416_161_000_000.0, 416_160_501_234.0, "USD")


def test_value_matches_zero_truth_requires_exact_zero() -> None:
    assert grade.value_matches(0.0, 0.0, "USD")
    assert not grade.value_matches(1.0, 0.0, "USD")


def test_zero_truth_grades_without_dividing_by_zero() -> None:
    item = _grade(_answer(value=0, scale="units"), fact=_fact(value=0.0))
    assert item.verdict is Verdict.CORRECT
    assert item.rel_error is None
    assert item.abs_error == 0.0


# --- EPS tolerance: $0.005 absolute (§3, §10 boundary cases) ------------------------


def test_eps_matches_inside_half_cent() -> None:
    assert grade.value_matches(7.462, 7.46, "USD/shares")


def test_eps_matches_exactly_at_the_half_cent_boundary() -> None:
    # §10 calls for the exact-boundary case: 0.005 off, and <= makes it a match.
    assert grade.value_matches(10.005, 10.0, "USD/shares")
    assert grade.value_matches(9.995, 10.0, "USD/shares")


def test_eps_just_outside_half_cent_is_not_a_match() -> None:
    assert not grade.value_matches(10.006, 10.0, "USD/shares")


def test_eps_uses_absolute_not_relative_tolerance() -> None:
    # 7.50 vs 7.46 is well inside 0.5% relative but is four cents wrong: EPS is graded
    # absolutely precisely so near-misses like this stay incorrect.
    assert not grade.value_matches(7.50, 7.46, "USD/shares")
    item = grade.grade(
        _answer(value=7.50, scale="units"),
        _eps_fact(),
        ticker="AAPL",
        kpi="eps_diluted",
        model="stub",
        cell="A1",
    )
    assert item.verdict is Verdict.INCORRECT


def test_eps_correct_at_units_scale() -> None:
    item = grade.grade(
        _answer(value=7.46, scale="units"),
        _eps_fact(),
        ticker="AAPL",
        kpi="eps_diluted",
        model="stub",
        cell="A1",
    )
    assert item.verdict is Verdict.CORRECT


# --- period window: +/-7 days (§3, §10 edges) ---------------------------------------


@pytest.mark.parametrize("offset", [0, 1, -1, 7, -7])
def test_period_window_includes_the_edges(offset: int) -> None:
    assert grade.within_period_window(PERIOD_END + dt.timedelta(days=offset), PERIOD_END)


@pytest.mark.parametrize("offset", [8, -8, 365, -365])
def test_period_window_excludes_beyond_the_edges(offset: int) -> None:
    assert not grade.within_period_window(PERIOD_END + dt.timedelta(days=offset), PERIOD_END)


def test_period_window_rejects_missing_date() -> None:
    assert not grade.within_period_window(None, PERIOD_END)


def test_right_number_wrong_year_is_wrong_period() -> None:
    # The §3 rule that gives the taxonomy its sharpest label: the prior-year column.
    item = _grade(_answer(period="2024-09-28"))
    assert item.verdict is Verdict.INCORRECT
    assert item.auto_label is FailureLabel.WRONG_PERIOD


def test_period_gates_the_value_even_when_the_figure_matches() -> None:
    item = _grade(_answer(period="2024-09-28"))
    assert item.rel_error == 0.0  # the figure is right...
    assert item.verdict is Verdict.INCORRECT  # ...and it is still wrong.


def test_missing_period_is_wrong_period() -> None:
    raw = '{"kpi": "revenue", "value": 416161, "scale": "millions", "currency": "USD"}'
    item = _grade(raw)
    assert item.verdict is Verdict.INCORRECT
    assert item.auto_label is FailureLabel.WRONG_PERIOD


def test_period_inside_window_grades_on_value() -> None:
    item = _grade(_answer(period="2025-09-30"))  # 3 days off, inside the window
    assert item.verdict is Verdict.CORRECT


# --- currency: non-USD is wrong-concept (§3) ----------------------------------------


def test_non_usd_currency_is_wrong_concept() -> None:
    item = _grade(_answer(currency="EUR"))
    assert item.verdict is Verdict.INCORRECT
    assert item.auto_label is FailureLabel.WRONG_CONCEPT


def test_currency_is_checked_before_value_and_period() -> None:
    # Wrong currency on an otherwise-perfect answer is a concept error, not a near-miss.
    item = _grade(_answer(currency="EUR", period="2024-09-28"))
    assert item.auto_label is FailureLabel.WRONG_CONCEPT


def test_currency_case_is_normalized() -> None:
    assert _grade(_answer(currency="usd")).verdict is Verdict.CORRECT


# --- the remaining §7 labels --------------------------------------------------------


def test_refusal() -> None:
    item = _grade("I cannot find the revenue figure in the provided text.")
    assert item.verdict is Verdict.REFUSAL
    assert item.auto_label is FailureLabel.REFUSAL
    assert item.predicted is None
    assert item.truth == 416_161_000_000.0  # truth recorded for the review loop


def test_format_failure() -> None:
    item = _grade("Revenue was very strong this year.")
    assert item.verdict is Verdict.FORMAT_FAILURE
    assert item.auto_label is FailureLabel.FORMAT_FAILURE
    assert item.predicted is None


def test_scale_error_is_incorrect_and_left_for_hand_labeling() -> None:
    # Right digits, wrong scale: 416,161 thousands, not millions. §7 hand-labels this
    # as scale-error in the Day D pass; the grader must not guess the label.
    item = _grade(_answer(scale="thousands"))
    assert item.verdict is Verdict.INCORRECT
    assert item.auto_label is None
    assert item.predicted == 416_161_000.0


def test_hallucination_is_incorrect_and_left_for_hand_labeling() -> None:
    # A figure that appears nowhere in context is only knowable by reading the context,
    # so the grader records `incorrect` and the review loop supplies the label.
    item = _grade(_answer(value=999999))
    assert item.verdict is Verdict.INCORRECT
    assert item.auto_label is None


def test_wrong_concept_line_item_is_left_for_hand_labeling() -> None:
    # Grabbing gross profit instead of revenue is indistinguishable from any other wrong
    # number without the filing in front of a human.
    item = _grade(_answer(value=180683))
    assert item.verdict is Verdict.INCORRECT
    assert item.auto_label is None


def test_every_frozen_taxonomy_label_exists() -> None:
    assert {label.value for label in FailureLabel} == {
        "wrong-period",
        "scale-error",
        "wrong-concept",
        "hallucination",
        "refusal",
        "format-failure",
    }


# --- no-ground-truth exclusion accounting (§2, §4, §10) -----------------------------


def test_no_ground_truth_when_no_tag_resolves() -> None:
    item = _grade(_answer(), fact=None)
    assert item.verdict is Verdict.NO_GROUND_TRUTH
    assert not item.scored
    assert item.predicted is None
    assert item.truth is None


def test_no_ground_truth_even_when_the_model_answered_well() -> None:
    # Unscoreable is decided by the absence of ground truth, not by the answer.
    assert _grade(_answer(), fact=None).verdict is Verdict.NO_GROUND_TRUTH


def test_no_ground_truth_beats_a_refusal() -> None:
    assert _grade("I cannot find it.", fact=None).verdict is Verdict.NO_GROUND_TRUTH


# --- accuracy accounting (§4) -------------------------------------------------------


def _item(verdict: Verdict) -> grade.GradedItem:
    return grade.GradedItem(ticker="AAPL", kpi="revenue", model="stub", cell="A1", verdict=verdict)


def test_accuracy_excludes_no_ground_truth_from_the_denominator() -> None:
    items = [
        _item(Verdict.CORRECT),
        _item(Verdict.INCORRECT),
        _item(Verdict.NO_GROUND_TRUTH),
        _item(Verdict.NO_GROUND_TRUTH),
    ]
    assert grade.accuracy(items) == 0.5  # 1 of 2 scored, not 1 of 4


def test_accuracy_counts_failures_and_refusals_against_the_model() -> None:
    items = [
        _item(Verdict.CORRECT),
        _item(Verdict.REFUSAL),
        _item(Verdict.FORMAT_FAILURE),
        _item(Verdict.INCORRECT),
    ]
    assert grade.accuracy(items) == 0.25


def test_accuracy_is_none_when_nothing_is_scoreable() -> None:
    assert grade.accuracy([_item(Verdict.NO_GROUND_TRUTH)]) is None
    assert grade.accuracy([]) is None


def test_accuracy_all_correct() -> None:
    assert grade.accuracy([_item(Verdict.CORRECT), _item(Verdict.CORRECT)]) == 1.0
