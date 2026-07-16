import json

import pytest

from filinglens import prompts


def test_every_v01_kpi_has_a_description() -> None:
    assert set(prompts.KPI_DESCRIPTIONS) == {
        "revenue",
        "net_income",
        "eps_diluted",
        "total_assets",
        "operating_cash_flow",
    }


def test_schema_mirrors_the_normalization_contract() -> None:
    # The §3 contract: value in the printed scale, scale enum, currency, period end.
    props = prompts.EXTRACTION_SCHEMA["properties"]
    assert set(props) == {"kpi", "value", "scale", "currency", "fiscal_period_end"}
    assert props["scale"]["enum"] == ["units", "thousands", "millions", "billions"]
    assert set(prompts.EXTRACTION_SCHEMA["required"]) == set(props)


def test_schema_is_json_serializable() -> None:
    # Ollama takes this over the wire as `format=`.
    assert json.loads(json.dumps(prompts.EXTRACTION_SCHEMA)) == prompts.EXTRACTION_SCHEMA


@pytest.mark.parametrize("structured", [True, False])
def test_prompt_carries_the_kpi_and_the_context(structured: bool) -> None:
    prompt = prompts.build_prompt("revenue", "Total revenue 281,724", structured=structured)
    assert "revenue" in prompt
    assert prompts.KPI_DESCRIPTIONS["revenue"] in prompt
    assert "Total revenue 281,724" in prompt


def test_freeform_and_structured_templates_differ() -> None:
    freeform = prompts.build_prompt("revenue", "ctx", structured=False)
    structured = prompts.build_prompt("revenue", "ctx", structured=True)
    assert freeform != structured
    # Only the free-form arm has to police its own output shape; the schema does it in A2/B2.
    assert "nothing else" in freeform


@pytest.mark.parametrize("kpi", list(prompts.KPI_DESCRIPTIONS))
def test_every_kpi_builds_in_both_modes(kpi: str) -> None:
    assert prompts.build_prompt(kpi, "ctx", structured=False)
    assert prompts.build_prompt(kpi, "ctx", structured=True)


def test_unknown_kpi_is_rejected() -> None:
    with pytest.raises(KeyError):
        prompts.build_prompt("ebitda", "ctx", structured=False)
