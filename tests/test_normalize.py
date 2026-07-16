import datetime as dt

import pytest

from filinglens import normalize
from filinglens.normalize import Extraction


def _json(value: str = "416161", scale: str = "millions", currency: str = "USD") -> str:
    return (
        f'{{"kpi": "revenue", "value": {value}, "scale": "{scale}", '
        f'"currency": "{currency}", "fiscal_period_end": "2025-09-27"}}'
    )


# --- scale arithmetic (§3) ---------------------------------------------------------


@pytest.mark.parametrize(
    ("scale", "expected"),
    [
        ("units", 416_161.0),
        ("thousands", 416_161_000.0),
        ("millions", 416_161_000_000.0),
        ("billions", 416_161_000_000_000.0),
    ],
)
def test_normalize_value_applies_every_frozen_scale(scale: str, expected: float) -> None:
    extraction = Extraction(value=416_161.0, scale=scale, currency="USD")
    assert normalize.normalize_value(extraction) == expected


def test_scale_factors_are_the_frozen_vocabulary() -> None:
    assert set(normalize.SCALE_FACTORS) == {"units", "thousands", "millions", "billions"}


# --- number parsing, incl. the parentheses-negative convention (§3) ------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (416161, 416_161.0),
        (7.46, 7.46),
        ("416161", 416_161.0),
        ("416,161", 416_161.0),
        ("$416,161", 416_161.0),
        ("(1,234)", -1234.0),  # accounting negative
        ("(1,234.5)", -1234.5),
        ("$(1,234)", -1234.0),
        ("-1234", -1234.0),
        ("  7.46  ", 7.46),
    ],
)
def test_parse_number(raw: object, expected: float) -> None:
    assert normalize.parse_number(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", "n/a", None, True, False, [1], {"a": 1}])
def test_parse_number_rejects_non_numbers(raw: object) -> None:
    assert normalize.parse_number(raw) is None


def test_parse_number_bool_is_not_a_figure() -> None:
    # bool is an int subclass; a naive isinstance check would read True as 1.0.
    assert normalize.parse_number(True) is None


# --- scale parsing ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("millions", "millions"),
        ("Millions", "millions"),
        ("MILLIONS.", "millions"),
        ("million", "millions"),
        ("in millions", "millions"),
        ("$ in thousands", "thousands"),
        ("USD millions", "millions"),
        ("thousand", "thousands"),
        ("billion", "billions"),
        ("bn", "billions"),
        ("units", "units"),
        ("actual", "units"),
        ("dollars", "units"),
    ],
)
def test_parse_scale_maps_synonyms(raw: str, expected: str) -> None:
    assert normalize.parse_scale(raw) == expected


def test_parse_scale_falls_back_when_unrecognized() -> None:
    assert normalize.parse_scale("furlongs") == "units"
    assert normalize.parse_scale(None) == "units"
    assert normalize.parse_scale("furlongs", fallback="millions") == "millions"


# --- date parsing -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2025-09-27", dt.date(2025, 9, 27)),
        ("Period ending 2025-09-27", dt.date(2025, 9, 27)),
        ("September 27, 2025", dt.date(2025, 9, 27)),
        ("Sep 27, 2025", dt.date(2025, 9, 27)),
        ("09/27/2025", dt.date(2025, 9, 27)),
        ("2025/09/27", dt.date(2025, 9, 27)),
    ],
)
def test_parse_date(raw: str, expected: dt.date) -> None:
    assert normalize.parse_date(raw) == expected


@pytest.mark.parametrize("raw", ["", "fiscal 2025", "not a date", None, 2025])
def test_parse_date_rejects_unparseable(raw: object) -> None:
    assert normalize.parse_date(raw) is None


@pytest.mark.parametrize("raw", ["2025-13-45", "2025-02-30", "0000-00-00"])
def test_parse_date_rejects_iso_shaped_but_invalid_dates(raw: str) -> None:
    # Matches the ISO regex but is not a real date; must not raise out of the parser.
    assert normalize.parse_date(raw) is None


def test_parse_date_passes_through_date_objects() -> None:
    assert normalize.parse_date(dt.date(2025, 9, 27)) == dt.date(2025, 9, 27)


# --- output parsing: the free-form arm's real-world messiness (§5) -------------------


def test_parse_output_clean_json() -> None:
    result = normalize.parse_output(_json())
    assert result.extraction is not None
    assert result.extraction.value == 416_161.0
    assert result.extraction.scale == "millions"
    assert result.extraction.currency == "USD"
    assert result.extraction.fiscal_period_end == dt.date(2025, 9, 27)
    assert not result.is_format_failure


def test_parse_output_strips_code_fences() -> None:
    result = normalize.parse_output(f"```json\n{_json()}\n```")
    assert result.extraction is not None
    assert result.extraction.value == 416_161.0


def test_parse_output_finds_json_embedded_in_prose() -> None:
    result = normalize.parse_output(f"Sure! Here is the figure:\n{_json()}\nHope that helps.")
    assert result.extraction is not None
    assert result.extraction.value == 416_161.0


def test_parse_output_tolerates_quoted_number_with_separators() -> None:
    result = normalize.parse_output(_json(value='"416,161"'))
    assert result.extraction is not None
    assert result.extraction.value == 416_161.0


def test_parse_output_defaults_missing_currency_to_usd() -> None:
    # The corpus is US filers reporting in USD; silence is not evidence of a foreign
    # currency, so absence must not be graded as wrong-concept.
    result = normalize.parse_output('{"value": 100, "scale": "millions"}')
    assert result.extraction is not None
    assert result.extraction.currency == "USD"


def test_parse_output_defaults_missing_scale_to_units() -> None:
    result = normalize.parse_output('{"value": 100, "currency": "USD"}')
    assert result.extraction is not None
    assert result.extraction.scale == "units"


def test_parse_output_missing_date_is_still_an_extraction() -> None:
    # Grading, not parsing, decides what a missing period costs.
    result = normalize.parse_output('{"value": 100, "scale": "millions", "currency": "USD"}')
    assert result.extraction is not None
    assert result.extraction.fiscal_period_end is None


@pytest.mark.parametrize(
    "raw",
    [
        "I cannot find the revenue in the provided text.",
        "The figure is not provided in this excerpt.",
        "I'm sorry, the total revenue is not stated in the filing text.",
        "Unable to determine the requested figure.",
    ],
)
def test_parse_output_detects_refusal(raw: str) -> None:
    result = normalize.parse_output(raw)
    assert result.refused
    assert result.extraction is None
    assert not result.is_format_failure


def test_refusal_markers_do_not_override_an_actual_answer() -> None:
    # A hedged but answered response is an answer: it must be graded on its figure.
    raw = f"I could not find it stated directly, but based on the statements:\n{_json()}"
    result = normalize.parse_output(raw)
    assert result.extraction is not None
    assert result.extraction.value == 416_161.0
    assert not result.refused


@pytest.mark.parametrize(
    "raw",
    ["", "   ", "Revenue was strong this year.", "{broken json", '{"kpi": "revenue"}'],
)
def test_parse_output_format_failure(raw: str) -> None:
    result = normalize.parse_output(raw)
    assert result.is_format_failure
    assert result.extraction is None
    assert not result.refused


def test_parse_output_ignores_json_without_a_value() -> None:
    # The value is the field the verdict turns on; an object lacking it is unusable.
    result = normalize.parse_output('{"kpi": "revenue", "scale": "millions"}')
    assert result.is_format_failure


def test_parse_output_skips_json_whose_value_is_not_a_number() -> None:
    result = normalize.parse_output('{"kpi": "revenue", "value": "n/a", "scale": "millions"}')
    assert result.is_format_failure


def test_parse_output_falls_through_to_a_later_usable_object() -> None:
    # The free-form arm sometimes emits a preamble object before the real answer; a
    # leading unusable object must not mask a good one behind it.
    raw = '{"note": "here is the figure", "value": null}\n' + _json()
    result = normalize.parse_output(raw)
    assert result.extraction is not None
    assert result.extraction.value == 416_161.0
