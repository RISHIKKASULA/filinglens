# STATE

Build order and current status for filinglens v0.1. The frozen design lives in
[docs/architecture.md](docs/architecture.md); decisions and deviations in
[docs/decisions.md](docs/decisions.md).

## Build order → commit arc (from the frozen spec, §11)

**Day A:** `feat: scaffold package, tooling, and CI skeleton` · `feat: add corpus
manifest with EDGAR fetch and accession pinning` · `feat: add ground-truth sanity check
command` → **stop at the gate: 10 sanity pairs reviewed manually, ADR-001 records pass/fail.**
⛔ Also the git checkpoint: repo name `filinglens`, public, full file list presented for
approval before first push.

**Day B:** `feat: add section slicing and BM25 retrieval` · `feat: add Ollama client
with stub for testing` · `feat: add prompts and extraction runner` (2-company smoke run).

**Day C:** `feat: add normalization and XBRL grading` · `feat: run v0.1 grid` (600
calls, unattended) · `feat: add cluster-bootstrap CIs and report generation`.

**Day D:** `feat: add failure-label review loop` + hand-labeling · `test: complete
coverage to gate` · `ci: finalize workflow` · `docs: write README with measured results` ·
`chore: release v0.1.0` — tag + release notes.

## Status

- [x] Day A: scaffold
- [x] Day A: corpus manifest + EDGAR fetch + pinning
- [x] Day A: sanity command
- [x] §0 sanity gate reviewed → ADR-001 PASS (10/10, 2026-07-14)
- [x] Day B: section slicing + BM25 retrieval
- [x] Day B: Ollama client + StubClient
- [x] Day B: prompts + extraction runner (resumable, digest-pinned)
- [x] Day C: normalization + XBRL grading (§3 tolerances)
- [x] Day C: v0.1 grid — 600 calls, 0 errors, all 3 models, digests pinned
- [x] Day C: cluster-bootstrap CIs + report generation → `docs/eval-report.md`
- [x] Day D: failure-label review loop (`filinglens label`) + all 230 incorrect items
  hand-labeled → `runs/v0.1/labels.csv` (§7 taxonomy complete)
- [x] Day D: coverage to gate (294 tests, 98.7% core, §10 fixture E2E) + CI green
- [x] Day D: §14 grid artifacts committed (parquet + config + labels), ADR-004/005
- [x] Day D: README with measured results + honest framing
- [x] Day D: release v0.1.0 (tag + CHANGELOG + notes)

## Failure taxonomy (Day D, hand-labeled, §7)

230 incorrect items, every one labeled against the filing (`runs/v0.1/labels.csv`):
scale-error 129 · hallucination 55 · wrong-concept 30 · wrong-period 16 (15 auto by date,
1 prior-year column). Two findings: scale-error dominates and hides qwen's EPS collapse
(32 of its 37 EPS failures are scale-tagging); wrong-concept is the line-item grab the
value-only grader cannot see (XOM 323,905 vs 332,238). The Day-C "3B grabs the prior-year
column" guess did not survive — exactly one prior-year grab in the grid.

## Measured v0.1 result (2026-07-16)

600 calls, 588 scored (12 excluded: CAT/net_income has no `NetIncomeLoss` in the pinned
accession — CAT tags `ProfitLoss`). Cluster bootstrap by company, B=10,000, seed 42.

| model | accuracy [95% CI] |
|---|---|
| `llama3.1:8b-instruct-q4_K_M` | 87.8% [81.5, 93.8] |
| `qwen2.5:7b-instruct-q4_K_M` | 50.5% [40.8, 60.5] |
| `llama3.2:3b-instruct-q4_K_M` | 44.4% [35.6, 53.6] |

Size cliff survives N=10 (8B minus 3B = 43.4% [31.5, 55.1]). 3B minus qwen is
**inconclusive** (-6.1% [-19.5, 7.7]). Section beats BM25 (15.6% [9.0, 22.3]); schema
beats free-form (4.8% [1.3, 8.5]). Sharpest cell: qwen on `eps_diluted` = 7.5% [0.0, 15.0].

## Deviations logged this build

- **ADR-002** — Item 8 is a cross-reference stub for NVDA and XOM (statements filed under
  Item 15 / in an appended Financial Section). Cell A was handing them 207- and 706-char
  contexts and grading the model as failing on an empty prompt. Statements are now located
  by their own heading; `preflight_contexts` refuses to start a grid on a stub context.
- **ADR-003** — `cents` is read at 0.01 on input. The `units` fallback was silently
  grading "273 cents" as $273. Flipped 2 of 600 rows. A *stated but unreadable* scale now
  fails loudly rather than becoming a guess.

## Day D does next

1. **`feat: add failure-label review loop`** + hand-label. 230 incorrect items; 15 carry an
   auto `wrong-period` label, **215 are unlabelled and need Rishik's eyes**. `filinglens
   label` does not exist yet — build it, then label into `runs/labels.csv` (committed, §7).
   Expect three clusters from the Day C spot-check: scale-error (digits right, label wrong
   — the dominant qwen/3B mode), wrong-concept line-item grabs (e.g. XOM `323,905` = Sales
   and other operating revenue, not Total revenues `332,238`), and the prior-year column.
   Budget for this: it is the report's centerpiece and it is manual.
2. **Settle §14's "full grid results committed (parquet + report)".** Only the report is
   committed; `results.parquet` (18 KB) and `raw.jsonl` (293 KB) are gitignored per
   instruction on 2026-07-16. Either commit the parquet or record an ADR — §14 is binding
   for the release.
3. **`test: complete coverage to gate`** — currently 278 tests, 100% on all 12 modules,
   gate is >=85%. Likely already satisfied; verify the §10 integration item (fixture ->
   run -> grade -> report end to end in CI under 2 min) is exercised as one real test.
4. **`ci: finalize workflow`** — CI must never touch network or Ollama (§8). Add the
   badge only after the first green run.
5. **`docs: write README with measured results`** — use the frozen §13 limitations and the
   honest framing paragraph verbatim. **Do not compare 87.8% to FinanceBench's 81%-wrong
   headline**: this measures headline-figure extraction with the statements already in
   context, a far easier task than FinanceBench's reasoning questions. Cite FinanceBench
   (arXiv 2311.11944) only as the verified prior-art anchor for the gap, never as a
   baseline. Also: §6's runtime estimate ("1-2.5 h unattended M4") measured ~49 s/call on
   a base M4 — roughly 5-7 h for 600 calls. Correct it or note it.
6. **`chore: release v0.1.0`** — tag + notes. Before tagging, run the §14 checks: boundary
   rule grep (zero employer/internship/private-work references in README, docs/, src/, and
   `git log`), `git log --format=full` sole-author check, no AI-tooling files in the repo.

Known gaps to state, not fix, in v0.1: `net_income`'s tag chain is one tag deep where
`revenue` has three (CAT fell through it); and the +/-7 day period rule counts a missing
`fiscal_period_end` as `wrong-period`, which is the strict reading of §3.
