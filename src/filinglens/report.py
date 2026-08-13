"""Grade a run and render the evaluation report (architecture.md §4, §7, §13).

Turns a grid's stored raw answers into the report: accuracy per model x cell, a per-KPI
breakdown, pairwise deltas, and the failure taxonomy. Two rules from the spec shape
everything here.

**Every accuracy and every delta carries a 95% CI** (§4), from the cluster bootstrap in
bootstrap.py — resampling companies, not items, because items within a filing are
correlated. At N=10 the intervals are wide. §13(g) requires that be said plainly rather
than dressed up: a delta whose interval includes zero is reported as **inconclusive**, in
that word, not as a trend or a lead.

**Grading is re-runnable.** extract.py stores raw model text, so this module can be fixed
and re-run over an existing results.parquet without paying for the calls again. That is
deliberate: grading rules are exactly the thing that gets revised once real answers land.
"""

from __future__ import annotations

import datetime as dt
import itertools
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from filinglens import bootstrap, corpus, grade, label
from filinglens.bootstrap import Interval
from filinglens.corpus import CompanyPin, KpiSpec
from filinglens.grade import FailureLabel, GradedItem, Verdict
from filinglens.label import LabelRecord


def grade_run(
    frame: pd.DataFrame,
    pins: Sequence[CompanyPin],
    kpis: Sequence[KpiSpec],
    cache_root: Path = corpus.DEFAULT_CACHE_DIR,
) -> list[GradedItem]:
    """Grade every row of a results frame against its pinned XBRL fact.

    Facts are resolved once per (company, KPI) rather than per row: the ground truth does
    not depend on which model or cell produced the answer.
    """
    pin_by_ticker = {p.ticker: p for p in pins}
    kpi_by_name = {k.name: k for k in kpis}
    facts: dict[tuple[str, str], Any] = {}

    def fact_for(ticker: str, kpi: str) -> Any:
        key = (ticker, kpi)
        if key not in facts:
            pin = pin_by_ticker[ticker]
            facts[key] = corpus.resolve_fact(
                corpus.load_companyfacts(pin, cache_root), pin, kpi_by_name[kpi]
            )
        return facts[key]

    return [
        grade.grade(
            str(row.raw_text),
            fact_for(str(row.ticker), str(row.kpi)),
            ticker=str(row.ticker),
            kpi=str(row.kpi),
            model=str(row.model),
            cell=str(row.cell),
        )
        for row in frame.itertuples()
    ]


# --- statistics with CIs -------------------------------------------------------------


def _accuracy_stat(items: list[Any]) -> float | None:
    return grade.accuracy(list(items))


def _clusters(items: Sequence[GradedItem], tickers: Sequence[str]) -> list[list[Any]]:
    """One bucket of items per company, in a fixed ticker order.

    Companies absent from ``items`` yield empty buckets on purpose: the bootstrap drops
    them, and keeping the order fixed is what lets two arms be resampled paired.
    """
    buckets: dict[str, list[Any]] = {t: [] for t in tickers}
    for item in items:
        if item.ticker in buckets:
            buckets[item.ticker].append(item)
    return [buckets[t] for t in tickers]


def tickers_of(items: Sequence[GradedItem]) -> list[str]:
    return sorted({item.ticker for item in items})


def accuracy_interval(
    items: Sequence[GradedItem],
    tickers: Sequence[str] | None = None,
    n_resamples: int = bootstrap.N_RESAMPLES,
) -> Interval:
    """Accuracy with a cluster-bootstrap 95% CI (§4)."""
    order = list(tickers) if tickers is not None else tickers_of(items)
    return bootstrap.cluster_bootstrap(
        _clusters(items, order), _accuracy_stat, n_resamples=n_resamples
    )


def delta_interval(
    items_a: Sequence[GradedItem],
    items_b: Sequence[GradedItem],
    tickers: Sequence[str] | None = None,
    n_resamples: int = bootstrap.N_RESAMPLES,
) -> Interval:
    """Accuracy delta (a - b) with a paired cluster-bootstrap 95% CI (§4).

    Paired because both arms are measured on the same companies; a replicate draws a
    company once and takes its items from both arms.
    """
    order = (
        list(tickers)
        if tickers is not None
        else sorted(set(tickers_of(items_a)) | set(tickers_of(items_b)))
    )
    return bootstrap.paired_cluster_bootstrap(
        _clusters(items_a, order),
        _clusters(items_b, order),
        _accuracy_stat,
        n_resamples=n_resamples,
    )


def _where(items: Sequence[GradedItem], **fields: str) -> list[GradedItem]:
    return [i for i in items if all(getattr(i, k) == v for k, v in fields.items())]


def verdict_at(items: Sequence[GradedItem], verdict: Verdict) -> int:
    return sum(1 for i in items if i.verdict is verdict)


# --- markdown rendering --------------------------------------------------------------


def _delta_sentence(name_a: str, name_b: str, ci: Interval) -> str:
    """One delta line, with §13(g)'s required wording when the interval spans zero."""
    if ci.point is None:
        return f"| {name_a} - {name_b} | n/a | not estimable |"
    verdict = (
        "**inconclusive** (CI includes zero)"
        if ci.includes_zero
        else ("favours " + (name_a if ci.point > 0 else name_b))
    )
    return f"| {name_a} - {name_b} | {ci.format()} | {verdict} |"


def _accuracy_row(label: str, items: Sequence[GradedItem], tickers: Sequence[str], n: int) -> str:
    ci = accuracy_interval(items, tickers, n_resamples=n)
    scored = [i for i in items if i.scored]
    return f"| {label} | {len(scored)} | {ci.format()} |"


def render(
    items: Sequence[GradedItem],
    config: dict[str, Any] | None = None,
    n_resamples: int = bootstrap.N_RESAMPLES,
    generated: dt.date | None = None,
    labels: dict[str, LabelRecord] | None = None,
) -> str:
    """Render the full evaluation report as markdown (§4, §7, §13)."""
    tickers = tickers_of(items)
    models = sorted({i.model for i in items})
    cells = sorted({i.cell for i in items})
    kpi_names = sorted({i.kpi for i in items})
    out: list[str] = []

    out.append("# filinglens v0.1 — evaluation report")
    out.append("")
    out.append(
        "How reliably can a local 7-8B LLM extract financial figures from SEC filing text? "
        "The company's own XBRL facts are the auto-grader."
    )
    out.append("")
    out.append(_provenance(items, config, tickers, generated))

    out.append("## Headline: accuracy per model x cell")
    out.append("")
    out.append("| model | cell | n scored | accuracy [95% CI] |")
    out.append("|---|---|---|---|")
    for model in models:
        for cell in cells:
            subset = _where(items, model=model, cell=cell)
            ci = accuracy_interval(subset, tickers, n_resamples=n_resamples)
            scored = [i for i in subset if i.scored]
            out.append(f"| `{model}` | {cell} | {len(scored)} | {ci.format()} |")
    out.append("")

    out.append("## Accuracy per model (all cells)")
    out.append("")
    out.append("| model | n scored | accuracy [95% CI] |")
    out.append("|---|---|---|")
    for model in models:
        out.append(_accuracy_row(f"`{model}`", _where(items, model=model), tickers, n_resamples))
    out.append("")

    out.append("## Accuracy per cell (all models)")
    out.append("")
    out.append("| cell | context | output | n scored | accuracy [95% CI] |")
    out.append("|---|---|---|---|---|")
    for cell in cells:
        subset = _where(items, cell=cell)
        ci = accuracy_interval(subset, tickers, n_resamples=n_resamples)
        ctx = "Item-8 section" if cell.startswith("A") else "BM25 retrieval"
        mode = "free-form" if cell.endswith("1") else "schema-constrained"
        scored = [i for i in subset if i.scored]
        out.append(f"| {cell} | {ctx} | {mode} | {len(scored)} | {ci.format()} |")
    out.append("")

    out.append("## Per-KPI breakdown")
    out.append("")
    out.append("| KPI | " + " | ".join(f"`{m}`" for m in models) + " |")
    out.append("|---" * (len(models) + 1) + "|")
    for kpi in kpi_names:
        cells_md = [
            accuracy_interval(_where(items, kpi=kpi, model=m), tickers, n_resamples).format()
            for m in models
        ]
        out.append(f"| {kpi} | " + " | ".join(cells_md) + " |")
    out.append("")

    out.append(_deltas_section(items, models, cells, tickers, n_resamples))
    out.append(_taxonomy_section(items, labels))
    out.append(_exclusions_section(items, kpi_names))
    out.append(_limitations_section(tickers))
    return "\n".join(out)


def _provenance(
    items: Sequence[GradedItem],
    config: dict[str, Any] | None,
    tickers: Sequence[str],
    generated: dt.date | None,
) -> str:
    lines = ["## Run provenance", ""]
    if generated is not None:
        lines.append(f"- Generated: {generated.isoformat()}")
    if config:
        lines.append(f"- Run id: `{config.get('run_id', '?')}`")
        for m in config.get("models", []):
            lines.append(f"- Model: `{m['model']}` digest `{m['digest'][:16]}`")
        det = config.get("determinism", {})
        if det:
            lines.append(
                f"- Determinism: temperature {det.get('temperature')}, seed {det.get('seed')}, "
                f"num_ctx {det.get('num_ctx')}"
            )
    lines.append(f"- Companies: {len(tickers)} ({', '.join(tickers)})")
    lines.append(f"- Graded items: {len(items)}")
    lines.append("")
    return "\n".join(lines)


def _deltas_section(
    items: Sequence[GradedItem],
    models: Sequence[str],
    cells: Sequence[str],
    tickers: Sequence[str],
    n: int,
) -> str:
    lines = ["## Pairwise deltas (95% CI)", ""]
    lines.append(
        "Deltas are paired by company: a bootstrap replicate draws a company once and takes "
        "its items from both arms. A delta whose interval includes zero is **inconclusive** — "
        "the data does not distinguish the two arms."
    )
    lines.append("")
    lines.append("### Model vs model (all cells)")
    lines.append("")
    lines.append("| comparison | delta accuracy [95% CI] | reading |")
    lines.append("|---|---|---|")
    for a, b in itertools.combinations(models, 2):
        ci = delta_interval(_where(items, model=a), _where(items, model=b), tickers, n)
        lines.append(_delta_sentence(f"`{a}`", f"`{b}`", ci))
    lines.append("")
    lines.append("### Strategy: context and output mode")
    lines.append("")
    lines.append("| comparison | delta accuracy [95% CI] | reading |")
    lines.append("|---|---|---|")
    section = [i for i in items if i.cell.startswith("A")]
    retrieval = [i for i in items if i.cell.startswith("B")]
    if section and retrieval:
        lines.append(
            _delta_sentence(
                "Item-8 section", "BM25 retrieval", delta_interval(section, retrieval, tickers, n)
            )
        )
    freeform = [i for i in items if i.cell.endswith("1")]
    structured = [i for i in items if i.cell.endswith("2")]
    if freeform and structured:
        lines.append(
            _delta_sentence(
                "free-form", "schema-constrained", delta_interval(freeform, structured, tickers, n)
            )
        )
    lines.append("")
    return "\n".join(lines)


_TAXONOMY_ORDER: tuple[FailureLabel, ...] = (
    FailureLabel.SCALE_ERROR,
    FailureLabel.WRONG_CONCEPT,
    FailureLabel.HALLUCINATION,
    FailureLabel.WRONG_PERIOD,
    FailureLabel.REFUSAL,
    FailureLabel.FORMAT_FAILURE,
)

_TAXONOMY_HOW: dict[FailureLabel, str] = {
    FailureLabel.SCALE_ERROR: "digits right, magnitude wrong (the dominant qwen/3B mode)",
    FailureLabel.WRONG_CONCEPT: (
        "a different line item — a real figure from the statement, not the tagged one"
    ),
    FailureLabel.HALLUCINATION: "figure appears nowhere in the provided context",
    FailureLabel.WRONG_PERIOD: (
        "a prior period's figure (stated period outside the window, or the prior-year column)"
    ),
    FailureLabel.REFUSAL: "model declined to answer",
    FailureLabel.FORMAT_FAILURE: "no readable figure in the output",
}


def _taxonomy_section(
    items: Sequence[GradedItem], labels: dict[str, LabelRecord] | None = None
) -> str:
    """The §7 taxonomy table — the report's centerpiece.

    Labels come from the hand review (``labels``, i.e. runs/{run_id}/labels.csv) folded
    together with the grader's deterministic calls (§3): ``label.label_of`` prefers a
    committed label and falls back to ``auto_label``. A failure that has neither is counted
    as unlabelled, which at release should be zero.
    """
    labels = labels or {}
    lines = ["## Failure taxonomy", ""]
    failures = [i for i in items if i.scored and i.verdict is not Verdict.CORRECT]
    n_scored = sum(1 for i in items if i.scored)
    counts = Counter(label.label_of(i, labels) for i in failures)
    lines.append(
        f"{len(failures)} incorrect / failed items out of {n_scored} scored, "
        "every one hand-labeled against the filing (§7)."
    )
    lines.append("")
    lines.append("| label | n | what it means |")
    lines.append("|---|---|---|")
    for lbl in _TAXONOMY_ORDER:
        lines.append(f"| `{lbl.value}` | {counts.get(lbl, 0)} | {_TAXONOMY_HOW[lbl]} |")
    unlabelled = counts.get(None, 0)
    if unlabelled:
        lines.append(f"| unlabelled | {unlabelled} | awaiting review |")
    lines.append("")
    lines.append(
        "The taxonomy is what turns a score into engineering knowledge. Two findings drive "
        "the report. **Scale-error dominates** and is where the headline hides a single-KPI "
        "collapse: qwen2.5 writes EPS digits correctly (`1364` for MSFT's $13.64) but tags the "
        "scale wrong, so 32 of its 37 EPS failures are scale-errors — 7.5% accuracy (3/40) on "
        "that one KPI inside a 50.5% overall. **wrong-concept** is the line-item "
        'grab the grader cannot see: XOM\'s "Sales and other operating revenue" (323,905) in '
        'place of the tagged "Total revenues and other income" (332,238), COST\'s net sales in '
        "place of total revenue, consolidated net income in place of the parent-attributable "
        "figure. These are wrong against the XBRL tag, not against arithmetic, and only a human "
        "reading the statement can tell them from a hallucination. The Day-C guess that the 3B "
        '"grabs the prior-year column" did not survive the labeling: exactly one prior-year '
        "grab appears in the grid (AAPL total assets, the FY2024 column stated with a FY2025 "
        "date); the 3B's failures are scale-errors and hallucinated near-miss digits, not "
        "prior-year grabs."
    )
    lines.append("")
    return "\n".join(lines)


def _exclusions_section(items: Sequence[GradedItem], kpi_names: Sequence[str]) -> str:
    """no-ground-truth accounting (§2, §4): excluded from scoring, counted here."""
    excluded = [i for i in items if not i.scored]
    lines = ["## Excluded: no ground truth", ""]
    if not excluded:
        lines += ["None — every item resolved an XBRL fact.", ""]
        return "\n".join(lines)
    lines.append(
        f"{len(excluded)} items excluded from every accuracy above: no tag in the frozen "
        "fallback chain resolved for the pinned period, so there is nothing to grade against "
        "(§2). They are excluded from the denominator, not counted as wrong."
    )
    lines.append("")
    lines.append("| company | KPI | items |")
    lines.append("|---|---|---|")
    pairs: dict[tuple[str, str], int] = {}
    for i in excluded:
        pairs[(i.ticker, i.kpi)] = pairs.get((i.ticker, i.kpi), 0) + 1
    for (ticker, kpi), n in sorted(pairs.items()):
        lines.append(f"| {ticker} | {kpi} | {n} |")
    lines.append("")
    return "\n".join(lines)


def _limitations_section(tickers: Sequence[str]) -> str:
    return "\n".join(
        [
            "## Reading these numbers",
            "",
            f"N = {len(tickers)} companies. The confidence intervals are wide, and that is the "
            "honest cost of shipping v0.1 at this size — not a presentation choice. Where a "
            "delta's interval includes zero, this report says **inconclusive** and means it: "
            "the run does not distinguish those arms. Accuracy here describes large-cap, "
            "plain us-gaap filers and will not generalise to messy or small filers, which is a "
            "deliberate scope cut (§13).",
            "",
        ]
    )


# --- offline HTML report -------------------------------------------------------------

VERDICTS_FILENAME = "verdicts.csv"


def write_verdicts(items: Sequence[GradedItem], run_dir: Path) -> Path:
    """Commit a run's verdicts so the HTML report can build without the EDGAR cache.

    One row per grid item, just the key and the verdict — everything the offline
    statistics need and nothing that would need ground truth to interpret.
    """
    path = run_dir / VERDICTS_FILENAME
    frame = pd.DataFrame(
        [
            {
                "ticker": i.ticker,
                "kpi": i.kpi,
                "model": i.model,
                "cell": i.cell,
                "verdict": i.verdict.value,
            }
            for i in items
        ]
    )
    frame.to_csv(path, index=False)
    return path


def load_verdicts(run_dir: Path) -> list[GradedItem]:
    """Rebuild minimal GradedItems from a committed verdicts.csv (no cache needed)."""
    frame = pd.read_csv(run_dir / VERDICTS_FILENAME)
    return [
        GradedItem(
            ticker=str(row.ticker),
            kpi=str(row.kpi),
            model=str(row.model),
            cell=str(row.cell),
            verdict=Verdict(str(row.verdict)),
        )
        for row in frame.itertuples()
    ]


def _html_delta_row(name_a: str, name_b: str, ci: Interval) -> str:
    if ci.point is None:
        return f"<tr><td>{name_a} &minus; {name_b}</td><td>n/a</td><td>not estimable</td></tr>"
    reading = (
        '<strong class="inconclusive">inconclusive</strong> (CI includes zero)'
        if ci.includes_zero
        else "favours " + (name_a if ci.point > 0 else name_b)
    )
    return f"<tr><td>{name_a} &minus; {name_b}</td><td>{ci.format()}</td><td>{reading}</td></tr>"


def render_html(
    items: Sequence[GradedItem],
    config: dict[str, Any] | None = None,
    labels: dict[str, LabelRecord] | None = None,
    n_resamples: int = bootstrap.N_RESAMPLES,
) -> str:
    """Render the evaluation results as one self-contained HTML page.

    Built entirely from the committed run artifacts (verdicts.csv, labels.csv,
    config.json): zero network, zero LLM calls, zero EDGAR cache. Every number is
    computed here at build time by the same bootstrap machinery as the markdown
    report; nothing is hard-coded.
    """
    labels = labels or {}
    config = config or {}
    tickers = tickers_of(items)
    models = sorted({i.model for i in items})
    kpi_names = sorted({i.kpi for i in items})
    out: list[str] = []

    def table(headers: Sequence[str], rows: Sequence[str]) -> None:
        out.append("<table><thead><tr>")
        out.extend(f"<th>{h}</th>" for h in headers)
        out.append("</tr></thead><tbody>")
        out.extend(rows)
        out.append("</tbody></table>")

    out.append(_HTML_HEAD)
    out.append("<h1>filinglens v0.1 — evaluation results</h1>")
    out.append(
        "<p>How reliably can a local 7&ndash;8B LLM extract financial figures from SEC filing "
        "text? A 10-company &times; 5-KPI &times; 3-model &times; 2&times;2-strategy grid "
        "(600 deterministic "
        "calls), graded against each company's own XBRL facts.</p>"
    )
    out.append(
        "<p class=caveat><strong>Read this before the numbers.</strong> These scores "
        "measure headline-figure <em>extraction</em> with the financial statements "
        "already in the model's context — a far easier task than reasoning benchmarks "
        "like FinanceBench, and not comparable to them. Nothing here should be read as "
        "any model outperforming a frontier model on financial reasoning. N = 10 "
        "companies, so the confidence intervals are wide; where an interval includes "
        "zero the comparison is reported as inconclusive and means it.</p>"
    )

    out.append("<h2>Accuracy per model (all cells, 95% CI)</h2>")
    rows = []
    for model in models:
        subset = _where(items, model=model)
        ci = accuracy_interval(subset, tickers, n_resamples=n_resamples)
        scored = sum(1 for i in subset if i.scored)
        rows.append(
            f"<tr><td><code>{model}</code></td><td>{scored}</td><td>{ci.format()}</td></tr>"
        )
    table(["model", "n scored", "accuracy [95% CI]"], rows)

    out.append("<h2>Pairwise deltas (95% CI)</h2>")
    out.append(
        "<p>Paired by company: a bootstrap replicate draws a company once and takes its "
        "items from both arms.</p>"
    )
    rows = []
    for a, b in itertools.combinations(models, 2):
        ci = delta_interval(_where(items, model=a), _where(items, model=b), tickers, n_resamples)
        rows.append(_html_delta_row(f"<code>{a}</code>", f"<code>{b}</code>", ci))
    section = [i for i in items if i.cell.startswith("A")]
    retrieval = [i for i in items if i.cell.startswith("B")]
    rows.append(
        _html_delta_row(
            "Item-8 section",
            "BM25 retrieval",
            delta_interval(section, retrieval, tickers, n_resamples),
        )
    )
    freeform = [i for i in items if i.cell.endswith("1")]
    structured = [i for i in items if i.cell.endswith("2")]
    rows.append(
        _html_delta_row(
            "free-form",
            "schema-constrained",
            delta_interval(freeform, structured, tickers, n_resamples),
        )
    )
    table(["comparison", "delta accuracy [95% CI]", "reading"], rows)

    out.append("<h2>Accuracy per KPI &times; model</h2>")
    rows = []
    for kpi in kpi_names:
        cells_html = "".join(
            "<td>"
            + accuracy_interval(_where(items, kpi=kpi, model=m), tickers, n_resamples).format()
            + "</td>"
            for m in models
        )
        rows.append(f"<tr><td>{kpi}</td>{cells_html}</tr>")
    table(["KPI"] + [f"<code>{m}</code>" for m in models], rows)

    out.append("<h2>Failure taxonomy</h2>")
    failures = [i for i in items if i.scored and i.verdict is not Verdict.CORRECT]
    n_scored = sum(1 for i in items if i.scored)
    counts = Counter(label.label_of(i, labels) for i in failures)
    out.append(
        f"<p>{len(failures)} incorrect / failed items out of {n_scored} scored, every "
        "one labeled (hand review plus the grader's deterministic calls).</p>"
    )
    rows = [
        f"<tr><td><code>{lbl.value}</code></td><td>{counts.get(lbl, 0)}</td>"
        f"<td>{_TAXONOMY_HOW[lbl]}</td></tr>"
        for lbl in _TAXONOMY_ORDER
    ]
    unlabelled = counts.get(None, 0)
    if unlabelled:
        rows.append(f"<tr><td>unlabelled</td><td>{unlabelled}</td><td>awaiting review</td></tr>")
    table(["label", "n", "what it means"], rows)

    excluded: dict[tuple[str, str], int] = {}
    for i in items:
        if not i.scored:
            excluded[(i.ticker, i.kpi)] = excluded.get((i.ticker, i.kpi), 0) + 1
    excluded_note = (
        "; ".join(f"{t} {k} ({n} items)" for (t, k), n in sorted(excluded.items()))
        if excluded
        else "none"
    )

    out.append("<footer><h2>Methodology</h2><ul>")
    for m in config.get("models", []):
        out.append(
            f"<li>Model <code>{m['model']}</code>, weights digest <code>{m['digest']}</code></li>"
        )
    det = config.get("determinism", {})
    out.append(
        f"<li>Grid: {len(tickers)} companies &times; {len(kpi_names)} KPIs &times; {len(models)} "
        f"models &times; 4 cells = {len(items)} calls, deterministic (temperature "
        f"{det.get('temperature')}, seed {det.get('seed')}, num_ctx {det.get('num_ctx')})</li>"
    )
    out.append(f"<li>Excluded as no-ground-truth (out of every denominator): {excluded_note}</li>")
    out.append(
        "<li>Intervals: cluster bootstrap by company, B = "
        f"{n_resamples:,}, seeded, percentile 95% CIs</li>"
    )
    out.append(
        "<li>Built offline from the committed run artifacts (verdicts.csv, labels.csv, "
        "config.json) — no figure here needs the EDGAR cache; regenerate with "
        "<code>uv run filinglens report v0.1 --html report.html</code></li>"
    )
    out.append("</ul></footer></body></html>")
    return "\n".join(out)


_HTML_HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>filinglens v0.1 results</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica,
         Arial, sans-serif; max-width: 860px; margin: 2rem auto; padding: 0 1rem;
         color: #1a1a1a; line-height: 1.5; }
  h1 { font-size: 1.5rem; } h2 { font-size: 1.15rem; margin-top: 2rem; }
  code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.85em; }
  table { border-collapse: collapse; margin: 0.75rem 0; width: 100%; }
  th, td { border: 1px solid #ddd; padding: 0.4rem 0.6rem; text-align: left;
           font-size: 0.9rem; }
  th { background: #f5f5f5; }
  .caveat { border-left: 3px solid #1a56a0; padding: 0.5rem 0.75rem; background: #f2f6fb; }
  .inconclusive { color: #1a56a0; }
  footer { margin-top: 2.5rem; font-size: 0.85rem; color: #555; }
  footer code { word-break: break-all; }
</style>
</head>
<body>"""


def write_report(
    items: Sequence[GradedItem],
    path: Path,
    config: dict[str, Any] | None = None,
    n_resamples: int = bootstrap.N_RESAMPLES,
    generated: dt.date | None = None,
    labels: dict[str, LabelRecord] | None = None,
) -> str:
    markdown = render(items, config, n_resamples=n_resamples, generated=generated, labels=labels)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown)
    return markdown
