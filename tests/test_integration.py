"""End-to-end integration (architecture.md §10): fixture → run → grade → label → report.

One real test that walks the whole pipeline on a committed synthetic fixture — a fake
filing text and fake companyfacts JSON — with StubClient standing in for Ollama, so CI
exercises fetch-parse-grade-label-report without touching the network or a model (§8). It
is deliberately tiny (1 company x 2 KPIs x 1 model x 4 cells) so it runs in well under the
§10 two-minute budget.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd

from filinglens import extract, label, report
from filinglens.corpus import CompanyPin, KpiSpec
from filinglens.grade import FailureLabel, Verdict
from filinglens.models import StubClient

PIN = CompanyPin(
    ticker="AAA",
    cik=111,
    accession="0000000111-25-000001",
    fiscal_period_end=dt.date(2025, 6, 30),
    form="10-K",
    filed=dt.date(2025, 8, 1),
)
KPIS = [
    KpiSpec(name="revenue", unit="USD", tags=["Revenues"]),
    KpiSpec(name="total_assets", unit="USD", tags=["Assets"]),
]

FILING_TEXT = """\
Item 8. Financial Statements and Supplementary Data
CONSOLIDATED STATEMENTS OF OPERATIONS
Total revenue 281,724 245,122 211,915
Cost of revenue 120,000 110,000 100,000
Gross profit 161,724 135,122 111,915
Operating income 90,000 80,000 70,000
Net income $ 101,832 $ 88,136 $ 72,361
CONSOLIDATED BALANCE SHEETS
Total current assets 200,000 180,000
Total assets $ 619,003 $ 512,163
Total liabilities 300,000 250,000
Total equity 319,003 262,163

Item 9. Changes in and Disagreements with Accountants
None to report for the period covered by this annual report on Form 10-K.
"""

COMPANYFACTS = {
    "facts": {
        "us-gaap": {
            "Revenues": {
                "units": {
                    "USD": [
                        {
                            "start": "2024-07-01",
                            "end": "2025-06-30",
                            "val": 281_724_000_000,
                            "accn": PIN.accession,
                            "fy": 2025,
                            "fp": "FY",
                            "form": "10-K",
                        }
                    ]
                }
            },
            "Assets": {
                "units": {
                    "USD": [
                        {
                            "end": "2025-06-30",
                            "val": 619_003_000_000,
                            "accn": PIN.accession,
                            "fy": 2025,
                            "fp": "FY",
                            "form": "10-K",
                        }
                    ]
                }
            },
        }
    }
}


def _answer(prompt: str) -> str:
    """A model that reads revenue correctly but mis-scales total assets — one correct item
    and one scale-error per cell, so grading and labeling both have something to do."""
    period = '"fiscal_period_end": "2025-06-30"'
    if "total_assets" in prompt:
        # digits right, scale wrong: 619,003 tagged as-issued instead of millions.
        return '{"value": 619003, "scale": "units", "currency": "USD", ' + period + "}"
    return '{"value": 281724, "scale": "millions", "currency": "USD", ' + period + "}"


def _cache(root: Path) -> Path:
    d = root / str(PIN.cik) / PIN.accession
    d.mkdir(parents=True)
    (d / "text.txt").write_text(FILING_TEXT)
    (d / "companyfacts.json").write_text(json.dumps(COMPANYFACTS))
    return root


def test_fixture_run_grade_label_report_end_to_end(tmp_path: Path) -> None:
    cache = _cache(tmp_path / "cache")
    runs = tmp_path / "runs"

    # 1. run the grid (StubClient, no network/Ollama)
    frame = extract.run_grid(
        [PIN],
        KPIS,
        [StubClient(model="stub", responder=_answer)],
        run_id="e2e",
        runs_dir=runs,
        cache_root=cache,
        min_context_chars=200,
    )
    assert len(frame) == 1 * 2 * 1 * 4  # 8 items
    assert (runs / "e2e" / "results.parquet").exists()

    # 2. grade it against the fixture XBRL
    reloaded = pd.read_parquet(runs / "e2e" / "results.parquet")
    items = report.grade_run(reloaded, [PIN], KPIS, cache_root=cache)
    verdicts = {(i.kpi, i.verdict) for i in items}
    assert ("revenue", Verdict.CORRECT) in verdicts
    assert ("total_assets", Verdict.INCORRECT) in verdicts

    # 3. label: the 4 total_assets misses have no auto label, so they are the review queue
    run_dir = runs / "e2e"
    path = label.labels_path(run_dir)
    records = {r.key: r for r in label.auto_seed(items, "e2e")}
    pending = label.pending_items(items, records)
    assert len(pending) == 4  # the four total_assets cells
    for item in pending:
        rec = label.LabelRecord(
            run_id="e2e",
            model=item.model,
            cell=item.cell,
            ticker=item.ticker,
            kpi=item.kpi,
            label=FailureLabel.SCALE_ERROR,
            source="hand",
        )
        records[rec.key] = rec
    label.save_labels(records.values(), path)

    # 4. render the report with the committed labels folded into the taxonomy
    labels = label.load_labels(path)
    out = run_dir / "eval-report.md"
    md = report.write_report(items, out, generated=dt.date(2026, 7, 17), labels=labels)
    assert out.exists()
    assert "## Failure taxonomy" in md
    assert "| `scale-error` | 4 |" in md
    assert "unlabelled" not in md  # every failure accounted for
