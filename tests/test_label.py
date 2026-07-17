"""Tests for the failure-label review loop (architecture.md §7).

These cover the pure, committable core: what counts as a pending item, how auto labels seed
labels.csv, how a hand label overrides an auto one, the CSV round-trip's stability, and the
evidence block a human reads to assign a label. The interactive input() loop in cli.py is
glue and is left to manual use.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from filinglens import label
from filinglens.grade import FailureLabel, GradedItem, Verdict


def _item(
    ticker: str = "AAA",
    kpi: str = "revenue",
    model: str = "m1",
    cell: str = "A1",
    verdict: Verdict = Verdict.INCORRECT,
    auto_label: FailureLabel | None = None,
    **kw: object,
) -> GradedItem:
    return GradedItem(
        ticker=ticker,
        kpi=kpi,
        model=model,
        cell=cell,
        verdict=verdict,
        auto_label=auto_label,
        **kw,  # type: ignore[arg-type]
    )


def _record(item: GradedItem, lbl: FailureLabel, source: str = "hand") -> label.LabelRecord:
    return label.LabelRecord(
        run_id="t",
        model=item.model,
        cell=item.cell,
        ticker=item.ticker,
        kpi=item.kpi,
        label=lbl,
        source=source,
    )


# --- what needs labeling ------------------------------------------------------------


def test_incorrect_items_excludes_correct_and_no_ground_truth() -> None:
    items = [
        _item(verdict=Verdict.CORRECT),
        _item(verdict=Verdict.INCORRECT, ticker="BBB"),
        _item(verdict=Verdict.NO_GROUND_TRUTH, ticker="CCC"),
        _item(verdict=Verdict.REFUSAL, ticker="DDD"),
    ]
    tickers = {i.ticker for i in label.incorrect_items(items)}
    assert tickers == {"BBB", "DDD"}  # INCORRECT and REFUSAL are scored failures


def test_pending_items_skips_already_labelled() -> None:
    a = _item(ticker="AAA")
    b = _item(ticker="BBB")
    labels = {a.model + "|A1|AAA|revenue": _record(a, FailureLabel.SCALE_ERROR)}
    pending = label.pending_items([a, b], labels)
    assert [i.ticker for i in pending] == ["BBB"]


def test_pending_items_is_ordered_stably() -> None:
    items = [
        _item(model="m2", cell="B1", ticker="ZZZ"),
        _item(model="m1", cell="A1", ticker="AAA"),
        _item(model="m1", cell="A1", ticker="BBB"),
    ]
    order = [(i.model, i.cell, i.ticker) for i in label.pending_items(items, {})]
    assert order == sorted(order)


# --- auto seeding and the auto/hand precedence --------------------------------------


def test_auto_seed_only_covers_grader_decided_items() -> None:
    items = [
        _item(auto_label=FailureLabel.WRONG_PERIOD, ticker="AAA"),
        _item(auto_label=None, ticker="BBB"),  # needs a human
        _item(verdict=Verdict.CORRECT, ticker="CCC"),
    ]
    seeded = label.auto_seed(items, "t")
    assert [r.ticker for r in seeded] == ["AAA"]
    assert seeded[0].source == "auto"
    assert seeded[0].label is FailureLabel.WRONG_PERIOD


def test_auto_seed_notes_explain_the_deterministic_call() -> None:
    wp = _item(
        auto_label=FailureLabel.WRONG_PERIOD,
        predicted_period_end=dt.date(2024, 6, 30),
        truth_period_end=dt.date(2025, 6, 30),
    )
    note = label.auto_seed([wp], "t")[0].note
    assert "2024-06-30" in note and "2025-06-30" in note


def test_auto_seed_notes_cover_currency_refusal_and_format() -> None:
    items = [
        _item(auto_label=FailureLabel.WRONG_CONCEPT, currency="EUR", ticker="AAA"),
        _item(auto_label=FailureLabel.REFUSAL, verdict=Verdict.REFUSAL, ticker="BBB"),
        _item(auto_label=FailureLabel.FORMAT_FAILURE, verdict=Verdict.FORMAT_FAILURE, ticker="CCC"),
    ]
    notes = {r.ticker: r.note for r in label.auto_seed(items, "t")}
    assert "EUR" in notes["AAA"]
    assert "declined" in notes["BBB"]
    assert "no readable figure" in notes["CCC"]


def test_label_of_prefers_a_committed_hand_label_over_auto() -> None:
    # A prior-year value stated with a right-looking date: the date-based auto rule can't see
    # it, so a hand label reclassifies it to wrong-period and must win.
    item = _item(auto_label=None)
    assert label.label_of(item, {}) is None
    labels = {"m1|A1|AAA|revenue": _record(item, FailureLabel.WRONG_PERIOD)}
    assert label.label_of(item, labels) is FailureLabel.WRONG_PERIOD


def test_label_of_falls_back_to_auto_label_when_unreviewed() -> None:
    item = _item(auto_label=FailureLabel.REFUSAL)
    assert label.label_of(item, {}) is FailureLabel.REFUSAL


# --- csv round-trip -----------------------------------------------------------------


def test_labels_csv_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    records = [
        _record(_item(ticker="AAA"), FailureLabel.SCALE_ERROR),
        _record(_item(ticker="BBB", kpi="eps_diluted"), FailureLabel.WRONG_CONCEPT, source="hand"),
    ]
    label.save_labels(records, path)
    loaded = label.load_labels(path)
    assert len(loaded) == 2
    assert loaded["m1|A1|AAA|revenue"].label is FailureLabel.SCALE_ERROR
    assert loaded["m1|A1|BBB|eps_diluted"].label is FailureLabel.WRONG_CONCEPT


def test_load_labels_missing_file_is_empty(tmp_path: Path) -> None:
    assert label.load_labels(tmp_path / "nope.csv") == {}


def test_save_labels_is_sorted_and_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    unordered = [
        _record(_item(model="m2", ticker="ZZZ"), FailureLabel.HALLUCINATION),
        _record(_item(model="m1", ticker="AAA"), FailureLabel.SCALE_ERROR),
    ]
    label.save_labels(unordered, path)
    first = path.read_text()
    label.save_labels(list(reversed(unordered)), path)
    assert path.read_text() == first  # order of input does not change the file
    body = first.splitlines()
    assert body[1].split(",")[3] == "AAA"  # m1/AAA sorts before m2/ZZZ


# --- the evidence block a human reads -----------------------------------------------


def test_evidence_shows_output_truth_and_context_snippets() -> None:
    item = _item(
        raw_text='{"value": 323905, "scale": "millions", "currency": "USD"}',
        predicted=323905e6,
        truth=332238e6,
        rel_error=0.025,
        predicted_period_end=dt.date(2025, 12, 31),
        truth_period_end=dt.date(2025, 12, 31),
    )
    context = (
        "Total revenues and other income 332,238 ... Sales and other operating revenue 323,905"
    )
    block = label.evidence(item, context)
    assert "323905" in block  # the model's raw output is echoed
    assert "323,905" in block  # written figure located in context at printed scale
    assert "332,238" in block  # truth located in context
    assert "2.500%" in block  # rel_error rendered


def test_evidence_handles_missing_figures_in_context() -> None:
    item = _item(raw_text='{"value": 999, "scale": "units", "currency": "USD"}', predicted=999.0)
    block = label.evidence(item, "no matching figures here")
    assert "AAA" in block  # still renders the header without crashing


def test_context_snippet_matches_at_printed_scale() -> None:
    # Filings print millions; the search is on significant digits, not absolute USD.
    assert label._context_snippet("...Assets 359,241 at year end...", 359_241e6, 40) is not None
    assert label._context_snippet("nothing here", 359_241e6, 40) is None
