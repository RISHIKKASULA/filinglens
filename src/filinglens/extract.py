"""Run the extraction grid: company x KPI x model x cell (architecture.md §5, §6, §8).

The v0.1 grid is 10 companies x 5 KPIs x 3 models x 4 cells = 600 calls, run unattended.
Two properties matter more than speed here:

**Resumability.** A 1-2.5h unattended run must survive a kill. Every completed call is
appended to ``raw.jsonl`` and flushed immediately; a restart reads that log, skips the
keys already present, and carries on. results.parquet is written from the accumulated
rows in a fixed sort order, so a killed-and-restarted run and a straight-through run
produce identical files (§10).

**Pinning.** config.json records the full grid config and each model's resolved weights
digest, so a result set says exactly which weights produced it (§6, §8).

Models are the outer loop: §6 freezes one model loaded at a time, and iterating models
outermost means Ollama swaps weights 3 times per run rather than 600.

This module runs models and records what they said. It does not grade — grade.py owns
verdicts, and keeping the raw text unjudged on disk means the grader can be fixed and
re-run without paying for 600 calls again.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel

from filinglens import corpus, prompts, retrieve, sections
from filinglens.corpus import CompanyPin, KpiSpec
from filinglens.models import ModelClient

DEFAULT_RUNS_DIR = Path("runs")

ContextMode = Literal["section", "bm25"]


class Cell(BaseModel):
    """One cell of the frozen 2x2 strategy ablation (§5)."""

    name: str
    context: ContextMode
    structured: bool

    @property
    def output_mode(self) -> str:
        return "structured" if self.structured else "freeform"


# The frozen 2x2 grid (§5). Capped at four cells — no cell creep.
CELLS: dict[str, Cell] = {
    "A1": Cell(name="A1", context="section", structured=False),
    "A2": Cell(name="A2", context="section", structured=True),
    "B1": Cell(name="B1", context="bm25", structured=False),
    "B2": Cell(name="B2", context="bm25", structured=True),
}


class ResultRow(BaseModel):
    """One grid cell's raw result. Ungraded: verdicts are grade.py's job."""

    run_id: str
    ticker: str
    cik: int
    accession: str
    kpi: str
    model: str
    digest: str
    cell: str
    context_mode: str
    output_mode: str
    context_chars: int
    prompt_chars: int
    raw_text: str
    error: str = ""

    @property
    def key(self) -> str:
        return item_key(self.model, self.cell, self.ticker, self.kpi)


def item_key(model: str, cell: str, ticker: str, kpi: str) -> str:
    """Identity of one grid item, used to skip completed work on resume."""
    return f"{model}|{cell}|{ticker}|{kpi}"


def build_context(text: str, kpi: str, cell: Cell) -> str:
    """The context block for a cell: the Item 8 slice, or the BM25 top-k (§5)."""
    if cell.context == "section":
        return sections.statements_section(text).text
    chunks = retrieve.retrieve_for_kpi(text, kpi)
    return "\n\n".join(chunk.text for chunk in chunks)


class _ContextCache:
    """Per-run context cache.

    A context depends only on (filing, KPI, context mode) — never on the model — so each
    is built at most once and reused across the three models. Without this, every BM25
    cell would re-chunk and re-index the whole filing on each of 300 calls.
    """

    def __init__(self, pins: Sequence[CompanyPin], cache_root: Path) -> None:
        self._cache_root = cache_root
        self._pins = {pin.ticker: pin for pin in pins}
        self._texts: dict[str, str] = {}
        self._contexts: dict[tuple[str, str, str], str] = {}

    def filing_text(self, ticker: str) -> str:
        if ticker not in self._texts:
            self._texts[ticker] = corpus.load_filing_text(self._pins[ticker], self._cache_root)
        return self._texts[ticker]

    def context(self, ticker: str, kpi: str, cell: Cell) -> str:
        key = (ticker, kpi, cell.context)
        if key not in self._contexts:
            self._contexts[key] = build_context(self.filing_text(ticker), kpi, cell)
        return self._contexts[key]


def load_completed(raw_path: Path) -> dict[str, ResultRow]:
    """Rows already on disk from an earlier attempt, keyed by item.

    Malformed trailing lines are dropped rather than fatal: a kill mid-write can leave a
    half-line, and that item is simply re-run.
    """
    if not raw_path.exists():
        return {}
    completed: dict[str, ResultRow] = {}
    for line in raw_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = ResultRow.model_validate_json(line)
        except ValueError:
            continue
        completed[row.key] = row
    return completed


def _append(raw_path: Path, row: ResultRow) -> None:
    """Append one result and flush, so a kill loses at most the in-flight call."""
    with raw_path.open("a") as fh:
        fh.write(row.model_dump_json() + "\n")
        fh.flush()


def _sort_rows(rows: Iterable[ResultRow]) -> list[ResultRow]:
    """Fixed order, so resumed and straight-through runs write identical parquet."""
    return sorted(rows, key=lambda r: (r.model, r.cell, r.ticker, r.kpi))


def to_frame(rows: Iterable[ResultRow]) -> pd.DataFrame:
    return pd.DataFrame([row.model_dump() for row in _sort_rows(rows)])


def write_config(
    run_dir: Path,
    run_id: str,
    pins: Sequence[CompanyPin],
    kpis: Sequence[KpiSpec],
    clients: Sequence[ModelClient],
    cells: Sequence[Cell],
) -> dict[str, Any]:
    """Record the full grid config and model digests alongside the results (§8)."""
    config: dict[str, Any] = {
        "run_id": run_id,
        "companies": [{"ticker": p.ticker, "cik": p.cik, "accession": p.accession} for p in pins],
        "kpis": [k.name for k in kpis],
        "models": [{"model": c.model, "digest": c.digest} for c in clients],
        "cells": [c.model_dump() for c in cells],
        "determinism": {
            "temperature": 0.0,
            "seed": 42,
            "num_ctx": 16384,
        },
        "planned_calls": len(pins) * len(kpis) * len(clients) * len(cells),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    return config


def preflight_contexts(
    pins: Sequence[CompanyPin],
    cache_root: Path = corpus.DEFAULT_CACHE_DIR,
    min_context_chars: int = sections.MIN_STATEMENTS_CHARS,
) -> None:
    """Check every company's statements context before spending a call on it (ADR-002).

    A stub context does not announce itself in the results: the model answers "I can't
    find it", the grader records a failure, and the run looks like evidence about the
    model when it is evidence about the harness. Worse, the damage is one-directional and
    lands entirely on the section arm, which is half of what the ablation compares. So
    this runs first and raises on the whole grid rather than let one silent stub through.
    """
    broken: list[str] = []
    for pin in pins:
        text = corpus.load_filing_text(pin, cache_root)
        try:
            sections.validate_statements(text, label=pin.ticker, min_chars=min_context_chars)
        except sections.StatementsSliceError as exc:
            broken.append(str(exc))
    if broken:
        raise sections.StatementsSliceError(
            "refusing to run the grid; statements context missing for "
            f"{len(broken)} filing(s):\n  " + "\n  ".join(broken)
        )


def run_grid(
    pins: Sequence[CompanyPin],
    kpis: Sequence[KpiSpec],
    clients: Sequence[ModelClient],
    *,
    run_id: str,
    cells: Sequence[Cell] | None = None,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    cache_root: Path = corpus.DEFAULT_CACHE_DIR,
    progress: bool = False,
    min_context_chars: int = sections.MIN_STATEMENTS_CHARS,
) -> pd.DataFrame:
    """Run the grid, resuming any work already on disk, and write results.parquet.

    Every company's statements context is checked before the first call (ADR-002), so a
    filing whose Item 8 is a cross-reference stops the run instead of quietly producing
    fake model failures.

    Iterates models outermost (one model loaded at a time, §6). A call that raises is
    recorded with the exception text and an empty answer rather than aborting: one bad
    call must not cost the other 599, and an empty answer grades as a format-failure,
    which is the honest reading of a call that produced nothing.
    """
    grid_cells = list(CELLS.values()) if cells is None else list(cells)
    preflight_contexts(pins, cache_root, min_context_chars)
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_path = run_dir / "raw.jsonl"

    write_config(run_dir, run_id, pins, kpis, clients, grid_cells)

    completed = load_completed(raw_path)
    rows: dict[str, ResultRow] = dict(completed)
    contexts = _ContextCache(pins, cache_root)

    for client in clients:
        for cell in grid_cells:
            for pin in pins:
                for kpi in kpis:
                    key = item_key(client.model, cell.name, pin.ticker, kpi.name)
                    if key in rows:
                        continue
                    row = _run_item(client, cell, pin, kpi, contexts, run_id)
                    _append(raw_path, row)
                    rows[key] = row
                    if progress:  # pragma: no cover
                        status = "ERR" if row.error else "ok"
                        print(f"[{len(rows)}] {key} {status}", flush=True)

    frame = to_frame(rows.values())
    frame.to_parquet(run_dir / "results.parquet", index=False)
    return frame


def _run_item(
    client: ModelClient,
    cell: Cell,
    pin: CompanyPin,
    kpi: KpiSpec,
    contexts: _ContextCache,
    run_id: str,
) -> ResultRow:
    context = contexts.context(pin.ticker, kpi.name, cell)
    prompt = prompts.build_prompt(kpi.name, context, structured=cell.structured)
    schema = prompts.EXTRACTION_SCHEMA if cell.structured else None

    raw_text = ""
    error = ""
    digest = client.digest
    try:
        response = client.generate(prompt, schema=schema)
        raw_text = response.text
        digest = response.digest
    except Exception as exc:  # one failed call must not kill the other 599
        error = f"{type(exc).__name__}: {exc}"

    return ResultRow(
        run_id=run_id,
        ticker=pin.ticker,
        cik=pin.cik,
        accession=pin.accession,
        kpi=kpi.name,
        model=client.model,
        digest=digest,
        cell=cell.name,
        context_mode=cell.context,
        output_mode=cell.output_mode,
        context_chars=len(context),
        prompt_chars=len(prompt),
        raw_text=raw_text,
        error=error,
    )
