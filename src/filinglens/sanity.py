"""Ground-truth sanity gate (architecture.md §0, blocking, manual).

For each company x KPI, print side by side: the XBRL fact (value, unit, fiscal period,
accession) and the candidate line(s) from the filing text containing that figure,
string-searched at reported and millions-rounded scale. The 10 pairs are reviewed
manually; the result is recorded in docs/decisions.md as ADR-001.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from filinglens import corpus

_MAX_CANDIDATE_LINES = 4
_LINE_SNIPPET_WIDTH = 110


class CandidateLine(BaseModel):
    line_no: int
    snippet: str


class SanityPair(BaseModel):
    ticker: str
    kpi: str
    fact: corpus.XbrlFact | None
    patterns: list[str]
    candidates: list[CandidateLine]


def search_patterns(value: float, unit: str) -> list[str]:
    """Figure formats to string-search: reported scale and millions-rounded scale."""
    patterns: list[str] = []
    if unit == "USD/shares":
        patterns.append(f"{value:.2f}")
    else:
        if value == int(value):
            patterns.append(f"{int(value):,}")
        millions = f"{round(value / 1e6):,}"
        if millions not in patterns:
            patterns.append(millions)
    return patterns


def _snippet(line: str, pattern: str) -> str:
    stripped = " ".join(line.split())
    if len(stripped) <= _LINE_SNIPPET_WIDTH:
        return stripped
    pos = stripped.find(pattern)
    start = max(0, pos - _LINE_SNIPPET_WIDTH // 2)
    end = start + _LINE_SNIPPET_WIDTH
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(stripped) else ""
    return f"{prefix}{stripped[start:end]}{suffix}"


def find_candidate_lines(
    text: str, patterns: list[str], max_lines: int = _MAX_CANDIDATE_LINES
) -> list[CandidateLine]:
    """Lines of filing text containing any search pattern, first matches first."""
    candidates: list[CandidateLine] = []
    seen: set[int] = set()
    for line_no, line in enumerate(text.splitlines(), start=1):
        for pattern in patterns:
            if pattern in line and line_no not in seen:
                seen.add(line_no)
                candidates.append(CandidateLine(line_no=line_no, snippet=_snippet(line, pattern)))
                break
        if len(candidates) >= max_lines:
            break
    return candidates


def build_pairs(
    pins: list[corpus.CompanyPin],
    kpis: list[corpus.KpiSpec],
    cache_root: Path = corpus.DEFAULT_CACHE_DIR,
) -> list[SanityPair]:
    pairs: list[SanityPair] = []
    for pin in pins:
        facts_json = corpus.load_companyfacts(pin, cache_root)
        text = corpus.load_filing_text(pin, cache_root)
        for kpi in kpis:
            fact = corpus.resolve_fact(facts_json, pin, kpi)
            patterns = search_patterns(fact.value, fact.unit) if fact else []
            candidates = find_candidate_lines(text, patterns) if patterns else []
            pairs.append(
                SanityPair(
                    ticker=pin.ticker,
                    kpi=kpi.name,
                    fact=fact,
                    patterns=patterns,
                    candidates=candidates,
                )
            )
    return pairs


def render_report(pairs: list[SanityPair]) -> str:
    """The gate artifact: 10 XBRL-fact vs filing-text pairs for manual review."""
    rule = "=" * 100
    out: list[str] = []
    matched = 0
    for i, pair in enumerate(pairs, start=1):
        out.append(rule)
        out.append(f"[{i}/{len(pairs)}] {pair.ticker} — {pair.kpi}")
        if pair.fact is None:
            out.append(
                "  XBRL: no-ground-truth (no tag in the chain resolved for the pinned period)"
            )
            continue
        f = pair.fact
        period = f"{f.start} → {f.end}" if f.start else f"as of {f.end}"
        out.append(f"  XBRL: {f.value:,.2f} {f.unit}  [{f.tag}]")
        out.append(
            f"        period {period} ({f.fiscal_period}{f.fiscal_year})  accession {f.accession}"
        )
        out.append("  search: " + " | ".join(f'"{p}"' for p in pair.patterns))
        if pair.candidates:
            matched += 1
            for cand in pair.candidates:
                out.append(f"  text L{cand.line_no}: {cand.snippet}")
        else:
            out.append("  text: NO MATCH — figure not found in filing text at searched scales")
    out.append(rule)
    out.append(
        f"{matched}/{len(pairs)} pairs have candidate lines. "
        "Gate (§0): >= 9/10 clean matches on MANUAL review "
        "(right period, right scale, figure findable in text)."
    )
    return "\n".join(out)
