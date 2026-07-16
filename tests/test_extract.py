import datetime as dt
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from filinglens import extract
from filinglens.corpus import CompanyPin, KpiSpec
from filinglens.extract import CELLS, Cell, ResultRow
from filinglens.models import ModelResponse, StubClient

PINS = [
    CompanyPin(
        ticker="AAA",
        cik=111,
        accession="0000000111-25-000001",
        fiscal_period_end=dt.date(2025, 6, 30),
        form="10-K",
        filed=dt.date(2025, 8, 1),
    ),
    CompanyPin(
        ticker="BBB",
        cik=222,
        accession="0000000222-25-000002",
        fiscal_period_end=dt.date(2025, 12, 31),
        form="10-K",
        filed=dt.date(2026, 2, 1),
    ),
]

KPIS = [
    KpiSpec(name="revenue", unit="USD", tags=["Revenues"]),
    KpiSpec(name="total_assets", unit="USD", tags=["Assets"]),
]

FILING_TEXT = """\
Item 7. Management's Discussion and Analysis
Revenue grew this year on strong demand across every segment we operate in.

Item 8. Financial Statements and Supplementary Data
CONSOLIDATED STATEMENTS OF OPERATIONS
Total revenue 281,724 245,122 211,915
Net income $ 101,832 $ 88,136 $ 72,361
CONSOLIDATED BALANCE SHEETS
Total assets $ 619,003 $ 512,163

Item 9. Changes in and Disagreements with Accountants
None to report for the period covered by this annual report.
"""


@pytest.fixture
def cache(tmp_path: Path) -> Path:
    """A cache tree holding filing text for both fixture companies."""
    root = tmp_path / "cache"
    for pin in PINS:
        d = root / str(pin.cik) / pin.accession
        d.mkdir(parents=True)
        (d / "text.txt").write_text(FILING_TEXT)
    return root


def _client(model: str = "stub", responder: Any = None) -> StubClient:
    return StubClient(model=model, responder=responder)


# --- the frozen 2x2 grid (§5) -------------------------------------------------------


def test_grid_is_the_frozen_four_cells() -> None:
    assert set(CELLS) == {"A1", "A2", "B1", "B2"}


def test_cells_span_both_context_modes_and_both_output_modes() -> None:
    assert (CELLS["A1"].context, CELLS["A1"].structured) == ("section", False)
    assert (CELLS["A2"].context, CELLS["A2"].structured) == ("section", True)
    assert (CELLS["B1"].context, CELLS["B1"].structured) == ("bm25", False)
    assert (CELLS["B2"].context, CELLS["B2"].structured) == ("bm25", True)


def test_output_mode_names() -> None:
    assert CELLS["A1"].output_mode == "freeform"
    assert CELLS["A2"].output_mode == "structured"


# --- context construction -----------------------------------------------------------


def test_section_context_is_the_item_8_slice() -> None:
    context = extract.build_context(FILING_TEXT, "revenue", CELLS["A1"])
    assert "Item 8." in context
    assert "Total revenue 281,724" in context
    assert "Item 7." not in context  # sliced away
    assert "Item 9." not in context  # ends at the next higher item


def test_bm25_context_retrieves_chunks() -> None:
    context = extract.build_context(FILING_TEXT, "revenue", CELLS["B1"])
    assert "Total revenue 281,724" in context


def test_bm25_context_differs_from_section_context() -> None:
    section = extract.build_context(FILING_TEXT, "revenue", CELLS["A1"])
    bm25 = extract.build_context(FILING_TEXT, "revenue", CELLS["B1"])
    assert section != bm25


def test_context_cache_builds_each_context_once(cache: Path) -> None:
    # Contexts never depend on the model, so the cache must not rebuild per client.
    contexts = extract._ContextCache(PINS, cache)
    first = contexts.context("AAA", "revenue", CELLS["B1"])
    second = contexts.context("AAA", "revenue", CELLS["B1"])
    assert first is second


# --- grid execution -----------------------------------------------------------------


def test_run_grid_covers_every_combination(cache: Path, tmp_path: Path) -> None:
    clients = [_client("m1"), _client("m2")]
    frame = extract.run_grid(
        PINS, KPIS, clients, run_id="t", runs_dir=tmp_path / "runs", cache_root=cache
    )
    # 2 companies x 2 KPIs x 2 models x 4 cells
    assert len(frame) == 2 * 2 * 2 * 4 == 32
    assert set(frame["model"]) == {"m1", "m2"}
    assert set(frame["cell"]) == {"A1", "A2", "B1", "B2"}
    assert set(frame["ticker"]) == {"AAA", "BBB"}
    assert set(frame["kpi"]) == {"revenue", "total_assets"}
    assert len(frame.drop_duplicates(["model", "cell", "ticker", "kpi"])) == 32


def test_run_grid_writes_parquet_and_raw_log(cache: Path, tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    extract.run_grid(PINS, KPIS, [_client()], run_id="t", runs_dir=runs, cache_root=cache)
    assert (runs / "t" / "results.parquet").exists()
    assert (runs / "t" / "raw.jsonl").exists()
    assert (runs / "t" / "config.json").exists()
    reloaded = pd.read_parquet(runs / "t" / "results.parquet")
    assert len(reloaded) == 16


def test_run_grid_records_the_model_answer(cache: Path, tmp_path: Path) -> None:
    client = _client(responder=lambda _: '{"value": 42, "scale": "millions"}')
    frame = extract.run_grid(
        PINS, KPIS, [client], run_id="t", runs_dir=tmp_path / "runs", cache_root=cache
    )
    assert all(frame["raw_text"] == '{"value": 42, "scale": "millions"}')
    assert all(frame["error"] == "")


def test_structured_cells_pass_the_schema_and_freeform_does_not(
    cache: Path, tmp_path: Path
) -> None:
    seen: list[dict[str, Any] | None] = []

    class Recorder(StubClient):
        def generate(self, prompt: str, schema: dict[str, Any] | None = None) -> ModelResponse:
            seen.append(schema)
            return super().generate(prompt, schema)

    extract.run_grid(
        PINS[:1],
        KPIS[:1],
        [Recorder(model="rec")],
        run_id="t",
        runs_dir=tmp_path / "runs",
        cache_root=cache,
    )
    assert sum(s is not None for s in seen) == 2  # A2, B2
    assert sum(s is None for s in seen) == 2  # A1, B1


def test_run_grid_honours_a_cell_subset(cache: Path, tmp_path: Path) -> None:
    frame = extract.run_grid(
        PINS,
        KPIS,
        [_client()],
        run_id="t",
        cells=[CELLS["A1"]],
        runs_dir=tmp_path / "runs",
        cache_root=cache,
    )
    assert set(frame["cell"]) == {"A1"}
    assert len(frame) == 4


def test_row_records_pinning_and_context_size(cache: Path, tmp_path: Path) -> None:
    frame = extract.run_grid(
        PINS, KPIS, [_client()], run_id="t", runs_dir=tmp_path / "runs", cache_root=cache
    )
    row = frame[(frame["ticker"] == "AAA") & (frame["cell"] == "A1")].iloc[0]
    assert row["accession"] == "0000000111-25-000001"
    assert row["cik"] == 111
    assert row["digest"] == "stub"
    assert row["run_id"] == "t"
    assert row["context_chars"] > 0
    assert row["prompt_chars"] > row["context_chars"]  # prompt wraps the context


# --- failure isolation --------------------------------------------------------------


def test_a_raising_call_is_recorded_and_does_not_kill_the_grid(cache: Path, tmp_path: Path) -> None:
    def responder(prompt: str) -> str:
        if "total_assets" in prompt:
            raise RuntimeError("ollama exploded")
        return '{"value": 1}'

    frame = extract.run_grid(
        PINS,
        KPIS,
        [_client(responder=responder)],
        run_id="t",
        runs_dir=tmp_path / "runs",
        cache_root=cache,
    )
    assert len(frame) == 16  # every item still present
    failed = frame[frame["kpi"] == "total_assets"]
    assert all(failed["error"].str.contains("RuntimeError: ollama exploded"))
    assert all(failed["raw_text"] == "")  # empty answer -> grades as format-failure
    survived = frame[frame["kpi"] == "revenue"]
    assert all(survived["error"] == "")


# --- resumability (§10: kill/restart -> identical results.parquet) ------------------


def test_resume_skips_completed_items(cache: Path, tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    calls: list[str] = []

    def counting(prompt: str) -> str:
        calls.append(prompt)
        return '{"value": 1}'

    extract.run_grid(
        PINS, KPIS, [_client(responder=counting)], run_id="t", runs_dir=runs, cache_root=cache
    )
    first_pass = len(calls)
    assert first_pass == 16

    # Re-running the same grid must call the model zero more times.
    extract.run_grid(
        PINS, KPIS, [_client(responder=counting)], run_id="t", runs_dir=runs, cache_root=cache
    )
    assert len(calls) == first_pass


def test_kill_midway_then_restart_reproduces_the_straight_through_parquet(
    cache: Path, tmp_path: Path
) -> None:
    """§10's property: a killed and resumed run is byte-identical to an unbroken one."""
    clean_runs = tmp_path / "clean"
    extract.run_grid(
        PINS,
        KPIS,
        [_client("m1"), _client("m2")],
        run_id="t",
        runs_dir=clean_runs,
        cache_root=cache,
    )
    clean = (clean_runs / "t" / "results.parquet").read_bytes()

    # Now simulate a kill: die partway through, then resume to completion.
    killed_runs = tmp_path / "killed"
    budget = {"n": 20}

    def dies_after_20(prompt: str) -> str:
        if budget["n"] <= 0:
            raise KeyboardInterrupt("killed mid-grid")
        budget["n"] -= 1
        return StubClient.DEFAULT_JSON

    with pytest.raises(KeyboardInterrupt):
        extract.run_grid(
            PINS,
            KPIS,
            [_client("m1", dies_after_20), _client("m2", dies_after_20)],
            run_id="t",
            runs_dir=killed_runs,
            cache_root=cache,
        )
    partial = extract.load_completed(killed_runs / "t" / "raw.jsonl")
    assert 0 < len(partial) < 32  # genuinely interrupted mid-grid

    extract.run_grid(
        PINS,
        KPIS,
        [_client("m1"), _client("m2")],
        run_id="t",
        runs_dir=killed_runs,
        cache_root=cache,
    )
    assert (killed_runs / "t" / "results.parquet").read_bytes() == clean


def test_load_completed_on_missing_log_is_empty(tmp_path: Path) -> None:
    assert extract.load_completed(tmp_path / "nope.jsonl") == {}


def test_load_completed_ignores_blank_lines(tmp_path: Path) -> None:
    row = ResultRow(
        run_id="t",
        ticker="AAA",
        cik=111,
        accession="a",
        kpi="revenue",
        model="m",
        digest="d",
        cell="A1",
        context_mode="section",
        output_mode="freeform",
        context_chars=1,
        prompt_chars=2,
        raw_text="x",
    )
    path = tmp_path / "raw.jsonl"
    path.write_text("\n" + row.model_dump_json() + "\n\n")
    assert list(extract.load_completed(path)) == ["m|A1|AAA|revenue"]


def test_load_completed_drops_a_half_written_trailing_line(tmp_path: Path) -> None:
    # A kill mid-write can leave a partial line; that item is simply re-run.
    row = ResultRow(
        run_id="t",
        ticker="AAA",
        cik=111,
        accession="a",
        kpi="revenue",
        model="m",
        digest="d",
        cell="A1",
        context_mode="section",
        output_mode="freeform",
        context_chars=1,
        prompt_chars=2,
        raw_text="x",
    )
    path = tmp_path / "raw.jsonl"
    path.write_text(row.model_dump_json() + "\n" + '{"run_id": "t", "ticker": "BB')
    completed = extract.load_completed(path)
    assert list(completed) == ["m|A1|AAA|revenue"]


def test_rows_are_written_in_a_fixed_order(cache: Path, tmp_path: Path) -> None:
    frame = extract.run_grid(
        PINS,
        KPIS,
        [_client("m2"), _client("m1")],
        run_id="t",
        runs_dir=tmp_path / "runs",
        cache_root=cache,
    )
    ordering = list(zip(frame["model"], frame["cell"], frame["ticker"], frame["kpi"], strict=True))
    assert ordering == sorted(ordering)  # sorted regardless of client order


# --- config / pinning (§8) ----------------------------------------------------------


def test_config_records_grid_and_model_digests(cache: Path, tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    extract.run_grid(
        PINS,
        KPIS,
        [_client("m1"), _client("m2")],
        run_id="t",
        runs_dir=runs,
        cache_root=cache,
    )
    config = json.loads((runs / "t" / "config.json").read_text())
    assert config["run_id"] == "t"
    assert config["models"] == [
        {"model": "m1", "digest": "stub"},
        {"model": "m2", "digest": "stub"},
    ]
    assert config["kpis"] == ["revenue", "total_assets"]
    assert [c["ticker"] for c in config["companies"]] == ["AAA", "BBB"]
    assert config["planned_calls"] == 32
    assert config["determinism"] == {"temperature": 0.0, "seed": 42, "num_ctx": 16384}


def test_config_records_the_frozen_v01_call_budget() -> None:
    # §6's run budget: 10 x 5 x 3 x 4 = 600.
    assert 10 * 5 * 3 * len(CELLS) == 600


def test_item_key_is_stable() -> None:
    assert extract.item_key("m", "A1", "AAA", "revenue") == "m|A1|AAA|revenue"


def test_result_row_key_matches_item_key() -> None:
    row = ResultRow(
        run_id="t",
        ticker="AAA",
        cik=1,
        accession="a",
        kpi="revenue",
        model="m",
        digest="d",
        cell="A1",
        context_mode="section",
        output_mode="freeform",
        context_chars=1,
        prompt_chars=2,
        raw_text="x",
    )
    assert row.key == extract.item_key("m", "A1", "AAA", "revenue")


def test_to_frame_is_empty_safe() -> None:
    assert len(extract.to_frame([])) == 0


def test_build_context_rejects_unknown_kpi_for_bm25() -> None:
    with pytest.raises(KeyError):
        extract.build_context(
            FILING_TEXT, "not_a_kpi", Cell(name="X", context="bm25", structured=False)
        )
