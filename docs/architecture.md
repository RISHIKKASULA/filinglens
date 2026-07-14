# filinglens — Design Doc & Spec (FROZEN v1.0, 2026-07-08)

Status: frozen for implementation on Opus after July 12. Deviations require an ADR in
`docs/decisions.md`; simplest defensible choice wins. This file becomes `docs/architecture.md`
(commit-arc/scaffold sections move to STATE.md). Same governance as fraudscore: no `CLAUDE.md`,
no `.claude/` in the repo; sole author `Rishik Kasula <contact.rishikkasula@gmail.com>` verified
with `git log --format=full` on the first commit; the standing git checkpoint rule applies at
repo creation and first push.

**One-liner:** an XBRL-graded evaluation harness answering, with measured accuracy and
confidence intervals: *how reliably can a local 7–8B LLM extract financial figures from SEC
filing text?* The company's own XBRL facts are the auto-grader.

**Prior-art anchor (verified):** FinanceBench (arXiv 2311.11944) — GPT-4-Turbo + retrieval
answered incorrectly or refused 81% of financial questions. No public work found that grades
*local* models on extraction against XBRL at scale. That's the gap.

**Honest framing (frozen README language — stands entirely on its own):** SEC filings are
public — privacy is *not* a benefit for this data and the README never claims it. The point:
regulated teams that must run LLMs on-prem (their documents can't transit third-party APIs)
need to know whether small local models can be trusted on financial text — and you can't
benchmark that on confidential documents in public. Public filings are the license-clean
corpus for exactly that document class, and the one place ground truth is free: every company
files its own numbers as structured XBRL. This project measures the question directly.
Full stop. **Boundary rule (frozen): no reference to any employer, internship, or private
prior work anywhere in this repo — README, docs, code comments, or commit messages.**

---

## 0. DAY-1 GROUND-TRUTH SANITY GATE (blocking, manual, before any model work)

`filinglens sanity AAPL MSFT` — after fetch, for 2 companies × all 5 v0.1 KPIs, print side by
side: the XBRL fact (value, unit, fiscal period, accession) and the candidate line(s) from the
filing text containing that figure (string-searched at reported and millions-rounded scale).
**Rishik manually reviews all 10 pairs.** Gate: **≥ 9/10 clean matches** (right period, right
scale, figure findable in text) → proceed. Below that → **STOP; do not build the grader**;
re-scope (different KPIs, different companies, or tolerance redesign) and amend this spec by
ADR first. The gate result (pass/fail + notes) is committed to `docs/decisions.md` as ADR-001.

## 1. Corpus manifest (`corpus.yaml`, committed)

- **v0.1 (frozen):** 10 large-cap, plain us-gaap filers, financial-sector excluded (banks use
  different revenue concepts; deliberate scope cut, documented):
  `AAPL MSFT NVDA KO PG JNJ WMT XOM COST CAT` — most recent **10-K, annual period**, as filed.
- Manifest pins **CIK + accession number + fiscal-period end date** per company at first fetch;
  graders and prompts key off the pinned accession, never "latest," so runs reproduce.
- **Deepening (roadmap, not v0.1):** 40 companies incl. mid-caps; add one 10-Q per company to
  probe period confusion.
- Fetch via **edgartools** (plumbing for the solved parts: EDGAR access, companyfacts, text).
  SEC fair-access: declared User-Agent with real contact email, ≤10 req/s, cache-first.

## 2. KPIs & us-gaap tag map (`kpis.yaml`, committed)

v0.1 — 5 KPIs, each with a frozen ordered tag-fallback chain (first tag present for the pinned
period wins; if none resolve, the item is `no-ground-truth` and excluded from scoring, counted
in the report):

| KPI | us-gaap tag chain |
|---|---|
| revenue | `RevenueFromContractWithCustomerExcludingAssessedTax` → `Revenues` → `RevenueFromContractWithCustomerIncludingAssessedTax` |
| net_income | `NetIncomeLoss` |
| eps_diluted | `EarningsPerShareDiluted` |
| total_assets | `Assets` |
| operating_cash_flow | `NetCashProvidedByUsedInOperatingActivities` |

Deepening adds: gross_profit, total_liabilities, eps_basic, cash_and_equivalents, capex.
**Custom extension tags are out of scope permanently** (documented limitation, not a TODO).

## 3. Normalization & match-tolerance policy (frozen)

Model must output structured JSON: `{kpi, value, scale ∈ {units, thousands, millions, billions},
currency, fiscal_period_end}`. Normalization: `value × scale → absolute USD`; currency must be
USD (else `wrong-concept`); negative-parentheses convention handled at parse.

**Verdict = correct iff, after normalization:**
- monetary KPIs: relative error ≤ **0.5%** vs XBRL (covers text rounded to millions vs exact
  XBRL; e.g. Apple-scale revenue rounds well inside 0.5%)
- eps_diluted: absolute error ≤ **$0.005** (one half-cent)
- and `fiscal_period_end` within ±7 days of the pinned period end (else `wrong-period`
  regardless of value match — a right number for the wrong year is wrong).

## 4. Grading protocol

Per item (company × KPI × model × strategy-cell): raw output → parse → normalize → verdict ∈
`{correct, incorrect, format-failure, refusal}` (+ `no-ground-truth` exclusions).
**Headline metric:** accuracy per model × cell; per-KPI breakdown table.
**Uncertainty (frozen, fraudscore discipline reused):** items within a company are correlated,
so **cluster bootstrap by company** — resample the 10 companies with replacement, B = 10,000,
seed 42, percentile 95% CIs on every accuracy and every pairwise delta. v0.1 CIs will be wide;
the report says so plainly — that's the honest cost of shipping v0.1 at N=10.

## 5. The 2×2 strategy ablation (frozen grid, capped — no cell creep)

| | Free-form output (regex/JSON-ish parse) | Schema-constrained (Ollama structured output) |
|---|---|---|
| **Item-8 section context** (sliced statements section, trimmed to context budget) | cell A1 | cell A2 |
| **BM25 retrieval** (filing chunked ~800 tokens; `rank_bm25`, top-8) | cell B1 | cell B2 |

BM25 over embeddings: deterministic, dependency-light, and retrieval quality is not the object
of study. Section slicing: Item-heading regex on extracted text with whole-doc chunk fallback
(the retrieval arm absorbs messy documents). One prompt template per output mode, frozen in
`prompts.py` at build time; prompt-sensitivity is a documented non-goal for v0.1.

## 6. Models (frozen; all comfortable in 24 GB with headroom)

| Model (Ollama) | Size (q4) | Role |
|---|---|---|
| `llama3.1:8b-instruct-q4_K_M` | ~5 GB | primary |
| `qwen2.5:7b-instruct-q4_K_M` | ~4.7 GB | primary |
| `llama3.2:3b-instruct-q4_K_M` | ~2 GB | the size cliff |

Deepening: `qwen2.5:14b-instruct-q4_K_M` (~9 GB) as "big local"; optional clearly-marked API
reference column (needs a key; explicitly out of v0.1). Determinism: temperature 0, seed 42,
`num_ctx` 16384, one model loaded at a time; model digests recorded into results for pinning.
Run budget v0.1: 10 × 5 × 3 × 4 = **600 calls** ≈ 1–2.5 h unattended M4 time.

## 7. Failure taxonomy (frozen 6 labels; hand-labeled by Rishik)

`wrong-period` · `scale-error` · `wrong-concept` (different line item incl. non-GAAP grab) ·
`hallucination` (figure appears nowhere in provided context) · `refusal` · `format-failure`.
v0.1: label **every** incorrect item (N is small); labels live in `runs/labels.csv` (committed),
`filinglens label` gives a review loop. The taxonomy table is the report's centerpiece —
it converts a score into engineering knowledge ("the 3B doesn't hallucinate, it grabs the
prior-year column").

## 8. Harness architecture

```
src/filinglens/   corpus.py (manifest, fetch, cache, pinning)   sections.py (Item slicing)
                  retrieve.py (BM25)   models.py (OllamaClient + StubClient for tests)
                  prompts.py   extract.py (run grid, resumable)   normalize.py   grade.py
                  report.py (tables, CIs, markdown)   cli.py
CLI:  filinglens fetch | sanity | run --models --cells --kpis | grade | label | report
Cache: data/cache/{cik}/{accession}/{filing.html, text.txt, sections.json, companyfacts.json}
Runs:  runs/{run_id}/results.parquet + config.json (full grid config + model digests)
```
Committed: `corpus.yaml`, `kpis.yaml`, `runs/labels.csv`, generated `docs/eval-report.md`.
Gitignored: `data/cache/`, model artifacts. CI never touches network or Ollama: a committed
**synthetic mini-fixture** (fake filing text + fake companyfacts JSON, seeded) plus StubClient
exercises fetch-parse-grade-report end to end.

## 9. Quality tooling (frozen)

`pyproject.toml` (uv) · **ruff** (lint + format) · **mypy** on `src/` · **pytest + coverage,
gate ≥ 85%** on core logic (normalize, grade, report; excludes CLI glue) · **pre-commit**
(ruff, ruff-format, mypy, end-of-file/trailing-whitespace) · GitHub Actions CI: lint → type →
test on 3.12 with uv cache; badge only after first green run. Deps: edgartools, rank_bm25,
ollama, pydantic, pandas, pyarrow, pyyaml, matplotlib; dev: pytest, pytest-cov, ruff, mypy,
pre-commit. LICENSE MIT.

## 10. Test plan

- **normalize:** scale arithmetic (incl. billions), parentheses negatives, currency rejection,
  EPS tolerance boundary cases (exactly $0.005 off), period-window edges (±7 days).
- **grade:** hand-built toy items with known verdicts for all six failure labels; tag-fallback
  chain order; `no-ground-truth` exclusion accounting.
- **cluster bootstrap:** seeded determinism; CI contains point estimate; rigged 2-company case
  with hand-checkable percentiles; delta-CI sign sanity.
- **sections/retrieve:** Item-8 slicing on fixture; BM25 top-k determinism.
- **extract:** StubClient grid run is resumable (kill/restart mid-grid → identical results.parquet).
- **integration:** fixture → run → grade → report end to end in CI < 2 min.
- **contract with reality:** `sanity` output format snapshot-tested (it's the gate artifact).

## 11. Build order → commit arc

Moved to [STATE.md](../STATE.md) at repo scaffold, per the header note above.

## 12. v0.1 scope vs deepening roadmap

**v0.1 ships:** 10 companies × 5 KPIs × 3 models × 4 cells, annual 10-Ks, full grading +
taxonomy + CIs + findings README. Honest v0.1 framing: "small-N, wide CIs, real pipeline."
**Deepening (post-sprint, ordered):** 40 companies → tighter CIs; 10 KPIs; 10-Q period-confusion
probe; qwen-14B; API reference column; findings blog post; prompt-sensitivity study (currently
a non-goal); per-sector breakdown.

## 13. Limitations (frozen README section — all stated, none hidden)

(a) Public data — no privacy benefit claimed; see framing paragraph. (b) Large-cap clean
filers only — accuracy will NOT generalize to messy/small filers; selection is deliberate and
stated. (c) Custom XBRL extension tags excluded. (d) XBRL treated as ground truth; rare tagging
errors exist upstream. (e) One prompt template per cell — prompt sensitivity unmeasured in
v0.1. (f) BM25 retrieval is deliberately basic; retrieval quality is not the object of study.
(g) N=10 → wide confidence intervals; deltas whose CIs include zero are reported as
inconclusive, in those words.

## 14. Acceptance criteria for v0.1.0

- Sanity gate ADR-001 recorded as PASS (or project re-scoped before build — gate is binding)
- Full grid results committed (parquet + report); every accuracy and delta carries a 95% CI
- All incorrect items hand-labeled; taxonomy table in README
- CI green: ruff + mypy + pytest ≥ 85% core coverage; fixture E2E under 2 min
- README: verified FinanceBench citation only; honest framing paragraph; limitations complete
- `git log --format=full`: sole author, no co-author trailers; no AI-tooling files in repo
- Boundary rule verified: zero employer/internship/private-work references anywhere in the
  repo (grep pass over README, docs/, src/, and `git log` before release)
- Checkpoint rule satisfied at repo creation + first push
