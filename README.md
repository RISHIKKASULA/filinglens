# filinglens

[![ci](https://github.com/RISHIKKASULA/filinglens/actions/workflows/ci.yml/badge.svg)](https://github.com/RISHIKKASULA/filinglens/actions/workflows/ci.yml)

An XBRL-graded evaluation harness answering, with measured accuracy and confidence
intervals: *how reliably can a local 7–8B LLM extract financial figures from SEC filing
text?* The company's own XBRL facts are the auto-grader — every figure a model reports is
checked against the number the company itself filed as structured data.

SEC filings are public — privacy is *not* a benefit for this data and this README never
claims it. The point: regulated teams that must run LLMs on-prem (their documents can't
transit third-party APIs) need to know whether small local models can be trusted on
financial text — and you can't benchmark that on confidential documents in public. Public
filings are the license-clean corpus for exactly that document class, and the one place
ground truth is free: every company files its own numbers as structured XBRL. This project
measures the question directly.

## What was measured

The v0.1 grid: **10 large-cap, plain us-gaap filers × 5 KPIs × 3 local models × a 2×2
strategy ablation = 600 calls**, run unattended, deterministic (temperature 0, seed 42,
`num_ctx` 16384), each model's weights digest pinned into the results. 588 items scored; 12
excluded as `no-ground-truth` (CAT tags `ProfitLoss`, not `NetIncomeLoss`, so §2's one-tag
chain for `net_income` does not resolve). Uncertainty is a **cluster bootstrap by company**
(resampling the 10 companies, not the 600 items, because items within a filing are
correlated), B = 10,000, seed 42, percentile 95% CIs on every accuracy and every delta.

The full generated report is [docs/eval-report.md](docs/eval-report.md); the design is
frozen in [docs/architecture.md](docs/architecture.md) and deviations are ADRs in
[docs/decisions.md](docs/decisions.md).

## Read this before the numbers

**87.8% is not a FinanceBench-beating result, and this project does not claim it is.** The
number below measures *headline-figure extraction with the financial statements already in
the model's context* — the model is handed the statement and asked for a figure that is
printed in it. That is a far easier task than [FinanceBench](https://arxiv.org/abs/2311.11944)
(arXiv 2311.11944), where GPT-4-Turbo with retrieval answered incorrectly or refused **81%**
of *reasoning* questions over full filings. FinanceBench is cited here **only** as the
verified prior-art anchor for the gap it named — no public work grades *local* models on
extraction against XBRL at scale — and **never as a baseline these numbers beat**. A high
extraction score and a low reasoning score are not comparable, and nothing here should be
read as filinglens outperforming FinanceBench, GPT-4, or any frontier model. Different task,
different difficulty.

## The finding: a size cliff, and a single-KPI collapse hiding inside a middling score

Accuracy per model, all cells (95% CI):

| model | n scored | accuracy [95% CI] |
|---|---|---|
| `llama3.1:8b-instruct-q4_K_M` | 196 | **87.8% [81.5, 93.8]** |
| `qwen2.5:7b-instruct-q4_K_M` | 196 | **50.5% [40.8, 60.5]** |
| `llama3.2:3b-instruct-q4_K_M` | 196 | **44.4% [35.6, 53.6]** |

**The size cliff is real and survives N=10.** The 8B beats the 3B by 43.4% [31.5, 55.1] and
qwen2.5-7B by 37.2% [28.1, 46.8] — both intervals nowhere near zero. The 3B-vs-qwen gap
does *not* separate: −6.1% [−19.5, 7.7], reported as **inconclusive** in those words,
because the interval includes zero. At q4 on this task, the 8B is in a different class from
the two smaller models; the two smaller models are not distinguishable from each other.

**But the headline hides the more useful finding.** qwen2.5-7B's 50.5% is not uniform
weakness — it is one KPI falling off a cliff:

| KPI | `llama3.1:8b` | `qwen2.5:7b` |
|---|---|---|
| eps_diluted | **97.5% [92.5, 100.0]** | **7.5% [0.0, 15.0]** |

qwen writes the EPS *digits* correctly — `1364` for MSFT's $13.64, `746` for AAPL's $7.46 —
and then tags the **scale** catastrophically wrong (`"thousands"` or `"millions"` on a
per-share figure), so the answer normalizes to the wrong magnitude. 32 of its 37 EPS
failures are this exact scale-tagging error. A model that reads the number right and cannot
say what scale it is on is a *specific, fixable* failure — and it is invisible in a 50.5%
average. This is why the taxonomy, not the headline, is the deliverable.

**Strategy ablation.** Giving the model the Item-8 statements section beats BM25 retrieval
by 15.6% [9.0, 22.3]; schema-constrained (Ollama structured output) beats free-form by 4.8%
[1.3, 8.5]. Both favour the more-constrained arm, both with intervals clear of zero.

## The failure taxonomy (the centerpiece)

Every one of the 230 incorrect items was hand-labeled against the filing (§7), via the
`filinglens label` review loop; labels are committed to
[`runs/v0.1/labels.csv`](runs/v0.1/labels.csv).

| label | n | what it means |
|---|---|---|
| `scale-error` | 129 | digits right, magnitude wrong — the dominant qwen/3B mode |
| `hallucination` | 55 | figure appears nowhere in the provided context |
| `wrong-concept` | 30 | a different line item than the tagged one (incl. non-GAAP grabs) |
| `wrong-period` | 16 | a prior period's figure (15 by stated date, 1 prior-year column) |
| `refusal` | 0 | model declined to answer |
| `format-failure` | 0 | no readable figure in the output |

Two clusters carry the engineering knowledge. **Scale-error dominates** and is where the
qwen EPS collapse lives. **wrong-concept** is the grab a value-only grader cannot see: XOM's
"Sales and other operating revenue" (323,905) in place of the tagged "Total revenues and
other income" (332,238); COST's net sales in place of total revenue; consolidated net income
in place of the parent-attributable figure. These are wrong against the specific XBRL tag,
not against arithmetic — and telling them apart from a hallucination requires reading the
statement, which is what the hand-labeling pass is for. The Day-C guess that the 3B "grabs
the prior-year column" did **not** survive labeling: exactly one prior-year grab exists in
the grid (AAPL total assets, the FY2024 column reported under a FY2025 date); the 3B's real
failures are scale-errors and hallucinated near-miss digits.

## Limitations (all stated, none hidden)

- **N = 10 → wide confidence intervals.** This is the honest cost of shipping v0.1 at this
  size, not a presentation choice. Where a delta's interval includes zero, the report says
  **inconclusive** and means it. Per-cell and per-KPI cells (n ≈ 40–49) are wider still —
  read the intervals, not the point estimates.
- **Large-cap, clean, plain-us-gaap filers only**, financial sector excluded (banks use
  different revenue concepts). Accuracy here will **not** generalize to messy or small
  filers; the selection is deliberate and stated.
- **A single filing year** — most-recent annual 10-K per company. No period-confusion probe
  (10-Qs) in v0.1, so period robustness is barely exercised.
- **XBRL is treated as ground truth.** Rare upstream tagging errors exist, and the tag
  *choice* matters: some `wrong-concept` failures are the model picking the more natural
  line item than the tagged one (WMT's total revenues vs the tagged net sales), so a
  fraction of "wrong" is really a ground-truth-definition disagreement, not a model error.
- **Custom XBRL extension tags are out of scope** (permanent, not a TODO).
- **One prompt template per cell** — prompt sensitivity is unmeasured in v0.1.
- **BM25 retrieval is deliberately basic**; retrieval quality is not the object of study.

## Reproducing

The v0.1 results are committed (`runs/v0.1/results.parquet` + `config.json` + `labels.csv`),
so grading, the report, and CI reproduce from the repo with **no network and no Ollama**:

```bash
uv sync
uv run filinglens grade v0.1     # grade the committed grid against XBRL
uv run filinglens report v0.1    # regenerate docs/eval-report.md (taxonomy from labels.csv)
uv run pytest                    # 294 tests, ≥85% core coverage, fixture E2E
```

Re-running the grid itself needs Ollama and the three pinned models; on a base M4 it took
~49 s/call, roughly **5–7 h unattended** for 600 calls (the frozen §6 estimate of 1–2.5 h
was optimistic; see ADR-005).

## Roadmap (post-v0.1)

40 companies for tighter CIs · 10 KPIs · a 10-Q period-confusion probe · qwen2.5-14B as "big
local" · an optional clearly-marked API reference column · a prompt-sensitivity study
(currently a non-goal). None ship in v0.1.

## License

MIT
