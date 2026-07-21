"""The failure-label review loop (architecture.md §7).

The taxonomy is the report's centerpiece: it turns a score into engineering knowledge
("the 3B doesn't hallucinate, it grabs the prior-year column" — or, as it turned out, the
reverse). The grader assigns only the labels §3 dictates deterministically (`wrong-period`
for a stated period outside the window, `wrong-concept` for a non-USD currency, `refusal`,
`format-failure`). Every other incorrect item — a right number at the wrong scale, a
different line item grabbed from the same statement, a figure that appears nowhere in the
context — needs the filing in front of a human. This module is that pass.

Labels live in ``runs/{run_id}/labels.csv`` (committed; run-scoped per ADR-006, which
records the deviation from §7/§8's root-level ``runs/labels.csv``) and cover **every** incorrect
item, not just the hand-reviewed ones: a row per failure, carrying its final §7 label, its
source (``auto`` from the grader or ``hand`` from this loop), and a short evidence note so
the assignment can be audited without re-deriving it. report.py reads this file as the
source of truth for the taxonomy table; a failure with no row is counted as unlabelled.

The loop is resumable: it grades the run, skips items already in labels.csv, and appends
as it goes, so a review can be stopped and restarted without redoing work.
"""

from __future__ import annotations

import csv
import datetime as dt
from collections.abc import Iterable, Sequence
from pathlib import Path

from pydantic import BaseModel

from filinglens import corpus, extract, normalize
from filinglens.corpus import CompanyPin
from filinglens.extract import item_key
from filinglens.grade import FailureLabel, GradedItem, Verdict

# The frozen §7 taxonomy, in the order the review loop offers them. `scale-error`,
# `wrong-concept`, and `hallucination` are the three that need a human; the other three the
# grader already assigns, but the loop still accepts them so a hand review can override a
# deterministic call (e.g. a prior-year *value* stated with a right-looking date, which the
# date-based auto rule cannot see — it is `wrong-period` on the evidence, not the date).
TAXONOMY: tuple[FailureLabel, ...] = (
    FailureLabel.SCALE_ERROR,
    FailureLabel.WRONG_CONCEPT,
    FailureLabel.HALLUCINATION,
    FailureLabel.WRONG_PERIOD,
    FailureLabel.REFUSAL,
    FailureLabel.FORMAT_FAILURE,
)

LABELS_FILENAME = "labels.csv"
_FIELDNAMES = ("run_id", "model", "cell", "ticker", "kpi", "label", "source", "note")


class LabelRecord(BaseModel):
    """One incorrect item's final §7 label, as committed to labels.csv."""

    run_id: str
    model: str
    cell: str
    ticker: str
    kpi: str
    label: FailureLabel
    source: str  # "auto" (grader) or "hand" (this loop)
    note: str = ""

    @property
    def key(self) -> str:
        return item_key(self.model, self.cell, self.ticker, self.kpi)


def labels_path(run_dir: Path) -> Path:
    return run_dir / LABELS_FILENAME


def load_labels(path: Path) -> dict[str, LabelRecord]:
    """Existing labels keyed by item. Missing file → no labels yet."""
    if not path.exists():
        return {}
    records: dict[str, LabelRecord] = {}
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            record = LabelRecord.model_validate(row)
            records[record.key] = record
    return records


def save_labels(records: Iterable[LabelRecord], path: Path) -> None:
    """Write labels.csv in a fixed sort order, so re-runs produce identical files."""
    ordered = sorted(records, key=lambda r: (r.model, r.cell, r.ticker, r.kpi))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_FIELDNAMES)
        writer.writeheader()
        for record in ordered:
            writer.writerow(
                {
                    "run_id": record.run_id,
                    "model": record.model,
                    "cell": record.cell,
                    "ticker": record.ticker,
                    "kpi": record.kpi,
                    "label": record.label.value,
                    "source": record.source,
                    "note": record.note,
                }
            )


def incorrect_items(items: Sequence[GradedItem]) -> list[GradedItem]:
    """Scored items the model got wrong — everything the taxonomy must account for (§7)."""
    return [i for i in items if i.scored and i.verdict is not Verdict.CORRECT]


def pending_items(items: Sequence[GradedItem], labels: dict[str, LabelRecord]) -> list[GradedItem]:
    """Incorrect items with no label yet, in a stable review order."""
    ordered = sorted(incorrect_items(items), key=lambda i: (i.model, i.cell, i.ticker, i.kpi))
    return [i for i in ordered if item_key(i.model, i.cell, i.ticker, i.kpi) not in labels]


def auto_seed(items: Sequence[GradedItem], run_id: str) -> list[LabelRecord]:
    """Label records for the items the grader already decided (§3).

    These carry ``source="auto"`` so labels.csv is complete — every incorrect item has a
    row — without a human re-typing a deterministic verdict. The review loop then only asks
    about the items that genuinely need eyes.
    """
    seeded: list[LabelRecord] = []
    for i in incorrect_items(items):
        if i.auto_label is not None:
            seeded.append(
                LabelRecord(
                    run_id=run_id,
                    model=i.model,
                    cell=i.cell,
                    ticker=i.ticker,
                    kpi=i.kpi,
                    label=i.auto_label,
                    source="auto",
                    note=_auto_note(i),
                )
            )
    return seeded


def _auto_note(item: GradedItem) -> str:
    if item.auto_label is FailureLabel.WRONG_PERIOD:
        pp = item.predicted_period_end
        return f"stated period {pp} outside +/-7d of {item.truth_period_end}"
    if item.auto_label is FailureLabel.WRONG_CONCEPT:
        return f"currency {item.currency} is not USD"
    if item.auto_label is FailureLabel.REFUSAL:
        return "model declined to answer"
    if item.auto_label is FailureLabel.FORMAT_FAILURE:
        return "no readable figure in output"
    return ""


def label_of(item: GradedItem, labels: dict[str, LabelRecord]) -> FailureLabel | None:
    """The final §7 label for an item: a committed label wins, else the grader's auto call.

    report.py uses this so the taxonomy table reflects the hand review. A hand label can
    override an auto one (the loop is allowed to reclassify), which is why labels.csv is
    consulted before ``auto_label``.
    """
    record = labels.get(item_key(item.model, item.cell, item.ticker, item.kpi))
    if record is not None:
        return record.label
    return item.auto_label


def evidence(item: GradedItem, context: str, radius: int = 220) -> str:
    """A compact evidence block for one item: what the model said, the truth, and the
    lines of context around each figure — enough to assign a §7 label without leaving the
    terminal.
    """
    parsed = normalize.parse_output(item.raw_text)
    written = parsed.extraction.value if parsed.extraction else None
    lines = [
        f"{item.ticker}  {item.kpi}  {item.model}  cell {item.cell}",
        f"  model output : {item.raw_text.strip()[:300]}",
        f"  normalized   : {_fmt(item.predicted)}   truth: {_fmt(item.truth)}"
        + (f"   rel_error: {item.rel_error:.3%}" if item.rel_error is not None else ""),
        f"  period       : stated {item.predicted_period_end}   pinned {item.truth_period_end}",
    ]
    for label, value in (("truth", item.truth), ("written", written)):
        snippet = _context_snippet(context, value, radius)
        if snippet:
            lines.append(f"  {label} in context: ...{snippet}...")
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:g}"


def _printed_forms(value: float) -> list[str]:
    """The ways ``value`` might be printed in a filing: as issued and rounded to thousands,
    millions, or billions (the scales a 10-K prints), comma-grouped and plain, plus a
    two-decimal form for per-share figures. This is why an absolute-USD truth of 332.238B
    is found next to the text's "332,238"."""
    forms: list[str] = []
    n = abs(value)
    for divisor in (1.0, 1e3, 1e6, 1e9):
        scaled = n / divisor
        if scaled < 1:
            continue
        if abs(scaled - round(scaled)) < 1e-6:
            i = round(scaled)
            forms += [f"{i:,}", str(i)]
    forms.append(f"{n:.2f}")
    return forms


def _context_snippet(context: str, value: float | None, radius: int) -> str | None:
    """The span of context around the first appearance of ``value`` at any printed scale.

    Filings print monetary figures rounded to millions, so the search is on the figure's
    significant digits (comma-grouped or plain), not its absolute-USD form.
    """
    if value is None or not context:
        return None
    for form in _printed_forms(value):
        idx = context.find(form)
        if idx != -1:
            start = max(0, idx - radius)
            end = min(len(context), idx + len(form) + radius)
            return " ".join(context[start:end].split())
    return None


class _ContextCache:
    """Per-review context cache, so the section slice or BM25 index for a (filing, KPI,
    mode) is built at most once even though three models share it."""

    def __init__(self, pins: Sequence[CompanyPin], cache_root: Path) -> None:
        self._pins = {p.ticker: p for p in pins}
        self._cache_root = cache_root
        self._texts: dict[str, str] = {}
        self._ctx: dict[tuple[str, str, str], str] = {}

    def context(self, item: GradedItem) -> str:
        ticker = item.ticker
        if ticker not in self._texts:
            self._texts[ticker] = corpus.load_filing_text(self._pins[ticker], self._cache_root)
        cell = extract.CELLS[item.cell]
        key = (ticker, item.kpi, cell.context)
        if key not in self._ctx:
            self._ctx[key] = extract.build_context(self._texts[ticker], item.kpi, cell)
        return self._ctx[key]


def prompt_label(number: int, total: int) -> str:  # pragma: no cover - interactive glue
    """The menu shown per item in the interactive loop."""
    menu = "  ".join(f"[{n}] {lbl.value}" for n, lbl in enumerate(TAXONOMY, start=1))
    return f"[{number}/{total}] label ({menu}, [s]kip, [q]uit): "


def parse_choice(choice: str) -> FailureLabel | str | None:  # pragma: no cover - glue
    """Map a keypress to a label, a control word ('skip'/'quit'), or None (reprompt)."""
    choice = choice.strip().lower()
    if choice in {"q", "quit"}:
        return "quit"
    if choice in {"s", "skip", ""}:
        return "skip"
    if choice.isdigit() and 1 <= int(choice) <= len(TAXONOMY):
        return TAXONOMY[int(choice) - 1]
    for label in TAXONOMY:
        if choice == label.value:
            return label
    return None


def today() -> dt.date:  # pragma: no cover - trivial; seam for tests
    return dt.date.today()
