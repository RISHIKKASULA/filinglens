import datetime as dt
from pathlib import Path

import pandas as pd

from filinglens import report
from filinglens.corpus import CompanyPin, KpiSpec
from filinglens.grade import FailureLabel, GradedItem, Verdict

# Small n_resamples throughout: these tests are about the report's logic and wording, not
# about the bootstrap's convergence (test_bootstrap.py owns that).
N = 200


def _item(
    ticker: str,
    verdict: Verdict,
    model: str = "m1",
    cell: str = "A1",
    kpi: str = "revenue",
    auto_label: FailureLabel | None = None,
) -> GradedItem:
    return GradedItem(
        ticker=ticker, kpi=kpi, model=model, cell=cell, verdict=verdict, auto_label=auto_label
    )


def _spread(verdicts: list[Verdict], **kw: str) -> list[GradedItem]:
    """One item per company, verdicts given in company order."""
    return [_item(f"C{i}", v, **kw) for i, v in enumerate(verdicts)]


# --- accuracy intervals -------------------------------------------------------------


def test_accuracy_interval_matches_the_observed_rate() -> None:
    items = _spread([Verdict.CORRECT] * 3 + [Verdict.INCORRECT])
    ci = report.accuracy_interval(items, n_resamples=N)
    assert ci.point == 0.75
    assert ci.low <= 0.75 <= ci.high


def test_accuracy_interval_excludes_no_ground_truth_from_the_denominator() -> None:
    items = _spread([Verdict.CORRECT, Verdict.INCORRECT, Verdict.NO_GROUND_TRUTH])
    ci = report.accuracy_interval(items, n_resamples=N)
    assert ci.point == 0.5  # 1 of 2 scored, not 1 of 3


def test_clusters_are_by_company() -> None:
    items = [
        _item("AAA", Verdict.CORRECT),
        _item("AAA", Verdict.CORRECT),
        _item("BBB", Verdict.INCORRECT),
    ]
    clusters = report._clusters(items, ["AAA", "BBB"])
    assert [len(c) for c in clusters] == [2, 1]


def test_clusters_keep_ticker_order_for_pairing() -> None:
    # Fixed order is what lets two arms resample the same company together.
    items = [_item("BBB", Verdict.CORRECT)]
    clusters = report._clusters(items, ["AAA", "BBB", "CCC"])
    assert [len(c) for c in clusters] == [0, 1, 0]


def test_tickers_of_is_sorted_and_unique() -> None:
    assert report.tickers_of(
        [_item("B", Verdict.CORRECT), _item("A", Verdict.CORRECT), _item("B", Verdict.INCORRECT)]
    ) == ["A", "B"]


# --- deltas and the §13(g) inconclusive rule ----------------------------------------


def test_delta_interval_is_paired_by_company() -> None:
    a = _spread([Verdict.CORRECT] * 4, model="m1")
    b = _spread([Verdict.INCORRECT] * 4, model="m2")
    ci = report.delta_interval(a, b, n_resamples=N)
    assert ci.point == 1.0
    assert not ci.includes_zero


def test_delta_sentence_says_inconclusive_in_that_word() -> None:
    # §13(g) requires the word, not a euphemism.
    from filinglens.bootstrap import Interval

    line = report._delta_sentence("A", "B", Interval(point=0.05, low=-0.2, high=0.3))
    assert "inconclusive" in line
    assert "CI includes zero" in line


def test_delta_sentence_names_the_winner_when_the_ci_excludes_zero() -> None:
    from filinglens.bootstrap import Interval

    assert "favours A" in report._delta_sentence("A", "B", Interval(point=0.4, low=0.1, high=0.7))
    assert "favours B" in report._delta_sentence(
        "A", "B", Interval(point=-0.4, low=-0.7, high=-0.1)
    )


def test_delta_sentence_handles_an_unestimable_delta() -> None:
    from filinglens.bootstrap import Interval

    line = report._delta_sentence("A", "B", Interval(point=None, low=None, high=None))
    assert "not estimable" in line


# --- the rendered report ------------------------------------------------------------


def _mixed_run() -> list[GradedItem]:
    items: list[GradedItem] = []
    for i in range(4):
        c = f"C{i}"
        items += [
            _item(c, Verdict.CORRECT, model="m1", cell="A1"),
            _item(c, Verdict.INCORRECT, model="m1", cell="B1"),
            _item(c, Verdict.CORRECT, model="m2", cell="A1", kpi="net_income"),
            _item(
                c,
                Verdict.REFUSAL,
                model="m2",
                cell="B1",
                kpi="net_income",
                auto_label=FailureLabel.REFUSAL,
            ),
        ]
    items.append(_item("C0", Verdict.NO_GROUND_TRUTH, model="m1", cell="A1", kpi="eps_diluted"))
    return items


def test_report_has_every_required_section() -> None:
    md = report.render(_mixed_run(), n_resamples=N)
    for heading in (
        "# filinglens v0.1",
        "## Run provenance",
        "## Headline: accuracy per model x cell",
        "## Accuracy per model",
        "## Accuracy per cell",
        "## Per-KPI breakdown",
        "## Pairwise deltas",
        "## Failure taxonomy",
        "## Excluded: no ground truth",
        "## Reading these numbers",
    ):
        assert heading in md, f"missing section: {heading}"


def test_every_accuracy_in_the_report_carries_a_ci() -> None:
    # §4: every accuracy and every delta carries a 95% CI. A bare percentage is a bug.
    md = report.render(_mixed_run(), n_resamples=N)
    for line in md.splitlines():
        if line.startswith("|") and "%" in line:
            assert "[" in line and "]" in line, f"accuracy without a CI: {line}"


def test_report_names_the_excluded_items() -> None:
    md = report.render(_mixed_run(), n_resamples=N)
    assert "1 items excluded" in md or "excluded from every accuracy" in md
    assert "eps_diluted" in md
    assert "not counted as wrong" in md


def test_report_states_n_and_the_wide_ci_caveat() -> None:
    md = report.render(_mixed_run(), n_resamples=N)
    assert "N = 4 companies" in md
    assert "wide" in md


def test_report_records_provenance_and_digests() -> None:
    config = {
        "run_id": "v0.1",
        "models": [{"model": "llama3.1:8b", "digest": "46e0c10c039e0191aaaa"}],
        "determinism": {"temperature": 0.0, "seed": 42, "num_ctx": 16384},
    }
    md = report.render(_mixed_run(), config, n_resamples=N, generated=dt.date(2026, 7, 16))
    assert "`v0.1`" in md
    assert "46e0c10c039e0191" in md
    assert "seed 42" in md
    assert "2026-07-16" in md


def test_report_without_config_still_renders() -> None:
    md = report.render(_mixed_run(), None, n_resamples=N)
    assert "## Run provenance" in md


def test_taxonomy_counts_auto_labels_and_defers_hand_labels() -> None:
    md = report.render(_mixed_run(), n_resamples=N)
    assert "`refusal` | 4 | auto" in md
    assert "hand-labelled (Day D)" in md
    assert "leaves the label to the review loop" in md


def test_report_with_no_exclusions_says_so() -> None:
    md = report.render(_spread([Verdict.CORRECT, Verdict.INCORRECT]), n_resamples=N)
    assert "None — every item resolved an XBRL fact." in md


def test_write_report_creates_the_file(tmp_path: Path) -> None:
    path = tmp_path / "docs" / "eval-report.md"
    md = report.write_report(_mixed_run(), path, n_resamples=N)
    assert path.read_text() == md
    assert md.startswith("# filinglens v0.1")


def test_report_is_deterministic() -> None:
    a = report.render(_mixed_run(), n_resamples=N)
    b = report.render(_mixed_run(), n_resamples=N)
    assert a == b  # seeded bootstrap => regenerable and diffable


# --- grading a run frame end to end (§10 integration shape) -------------------------

PIN = CompanyPin(
    ticker="AAA",
    cik=111,
    accession="0000000111-25-000001",
    fiscal_period_end=dt.date(2025, 6, 30),
    form="10-K",
    filed=dt.date(2025, 8, 1),
)
KPIS = [KpiSpec(name="revenue", unit="USD", tags=["Revenues"])]


def _companyfacts(val: float = 100_000_000.0) -> dict:
    return {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            {
                                "start": "2024-07-01",
                                "end": "2025-06-30",
                                "val": val,
                                "accn": PIN.accession,
                                "fy": 2025,
                                "fp": "FY",
                                "form": "10-K",
                            }
                        ]
                    }
                }
            }
        }
    }


def _cache(tmp_path: Path) -> Path:
    import json

    root = tmp_path / "cache"
    d = root / str(PIN.cik) / PIN.accession
    d.mkdir(parents=True)
    (d / "companyfacts.json").write_text(json.dumps(_companyfacts()))
    return root


def test_grade_run_grades_a_results_frame(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    frame = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "kpi": "revenue",
                "model": "m1",
                "cell": "A1",
                "raw_text": '{"value": 100, "scale": "millions", "currency": "USD",'
                ' "fiscal_period_end": "2025-06-30"}',
            },
            {
                "ticker": "AAA",
                "kpi": "revenue",
                "model": "m1",
                "cell": "B1",
                "raw_text": "I cannot find it.",
            },
        ]
    )
    items = report.grade_run(frame, [PIN], KPIS, cache_root=cache)
    assert [i.verdict for i in items] == [Verdict.CORRECT, Verdict.REFUSAL]
    assert items[0].predicted == 100_000_000.0


def test_grade_run_resolves_each_fact_once(tmp_path: Path) -> None:
    # Ground truth does not depend on model or cell; re-resolving per row would be waste.
    cache = _cache(tmp_path)
    rows = [
        {"ticker": "AAA", "kpi": "revenue", "model": f"m{i}", "cell": "A1", "raw_text": "{}"}
        for i in range(5)
    ]
    items = report.grade_run(pd.DataFrame(rows), [PIN], KPIS, cache_root=cache)
    assert len(items) == 5
    assert all(i.truth == 100_000_000.0 for i in items)


def test_verdict_at_counts() -> None:
    items = _spread([Verdict.CORRECT, Verdict.CORRECT, Verdict.REFUSAL])
    assert report.verdict_at(items, Verdict.CORRECT) == 2
    assert report.verdict_at(items, Verdict.REFUSAL) == 1


def test_report_compares_both_strategy_axes_when_all_four_cells_are_present() -> None:
    """The 2x2's two headline deltas: section vs retrieval, free-form vs schema."""
    items: list[GradedItem] = []
    for i in range(4):
        c = f"C{i}"
        # section arm right, retrieval arm wrong; free-form right, schema wrong.
        items += [
            _item(c, Verdict.CORRECT, cell="A1"),
            _item(c, Verdict.CORRECT, cell="A2"),
            _item(c, Verdict.INCORRECT, cell="B1"),
            _item(c, Verdict.INCORRECT, cell="B2"),
        ]
    md = report.render(items, n_resamples=N)
    assert "Item-8 section" in md
    assert "BM25 retrieval" in md
    assert "free-form - schema-constrained" in md
    # section beats retrieval by construction; the two output modes are identical here
    assert "favours Item-8 section" in md
    assert "inconclusive" in md


def test_output_mode_delta_is_reported_when_it_favours_schema() -> None:
    items: list[GradedItem] = []
    for i in range(4):
        c = f"C{i}"
        items += [
            _item(c, Verdict.INCORRECT, cell="A1"),
            _item(c, Verdict.CORRECT, cell="A2"),
        ]
    md = report.render(items, n_resamples=N)
    assert "favours schema-constrained" in md
