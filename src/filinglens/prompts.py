"""Frozen extraction prompts and output schema (architecture.md §3, §5).

One prompt template per output mode — free-form (the model is asked for JSON, parsed
leniently downstream) and schema-constrained (Ollama enforces EXTRACTION_SCHEMA). Prompt
sensitivity is a documented non-goal for v0.1 (limitation e); these templates are frozen
at build time. The required output shape is the normalization contract from §3.
"""

from __future__ import annotations

from typing import Any

# Human-readable description of each KPI, so the model knows which line item to find.
KPI_DESCRIPTIONS: dict[str, str] = {
    "revenue": "total revenue (also called total net sales or total revenues) for the fiscal year",
    "net_income": "net income (net earnings) attributable to the company for the fiscal year",
    "eps_diluted": "diluted earnings per share for the fiscal year",
    "total_assets": "total assets as of the fiscal year-end balance sheet date",
    "operating_cash_flow": "net cash provided by operating activities for the fiscal year",
}

# The structured-output JSON schema (Ollama `format=`). Mirrors the §3 normalization
# contract: value in the scale as printed, scale as an enum, currency, and period end.
EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "kpi": {"type": "string"},
        "value": {"type": "number"},
        "scale": {"type": "string", "enum": ["units", "thousands", "millions", "billions"]},
        "currency": {"type": "string"},
        "fiscal_period_end": {"type": "string"},
    },
    "required": ["kpi", "value", "scale", "currency", "fiscal_period_end"],
}

_FIELDS_GUIDE = (
    "Return these fields:\n"
    '- "kpi": the KPI name, exactly "{kpi}".\n'
    '- "value": the number as printed in the filing (no thousands separators).\n'
    '- "scale": one of "units", "thousands", "millions", "billions" — the scale the figure '
    'is reported in (statements often say "in millions").\n'
    '- "currency": the ISO currency code, e.g. "USD".\n'
    '- "fiscal_period_end": the period-end date this figure covers, as YYYY-MM-DD.'
)

_FREEFORM_TEMPLATE = """\
You are extracting one figure from an SEC 10-K filing. Extract the {kpi}: {description}.

Use ONLY the filing text below. Respond with a single JSON object and nothing else — no
prose, no code fences.

{fields}

Filing text:
\"\"\"
{context}
\"\"\"
"""

_STRUCTURED_TEMPLATE = """\
Extract the {kpi} ({description}) from the SEC 10-K filing text below. Use ONLY this text.

{fields}

Filing text:
\"\"\"
{context}
\"\"\"
"""


def build_prompt(kpi: str, context: str, structured: bool) -> str:
    """Build the extraction prompt for a KPI over a context block.

    ``structured`` selects the schema-constrained template (the shape is also enforced by
    Ollama's ``format``); otherwise the free-form template asks for a bare JSON object.
    """
    if kpi not in KPI_DESCRIPTIONS:
        raise KeyError(f"no prompt description for KPI {kpi!r}")
    template = _STRUCTURED_TEMPLATE if structured else _FREEFORM_TEMPLATE
    fields = _FIELDS_GUIDE.format(kpi=kpi)
    return template.format(
        kpi=kpi, description=KPI_DESCRIPTIONS[kpi], fields=fields, context=context
    )
