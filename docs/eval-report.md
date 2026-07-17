# filinglens v0.1 — evaluation report

How reliably can a local 7-8B LLM extract financial figures from SEC filing text? The company's own XBRL facts are the auto-grader.

## Run provenance

- Generated: 2026-07-16
- Run id: `v0.1`
- Model: `llama3.1:8b-instruct-q4_K_M` digest `46e0c10c039e0191`
- Model: `qwen2.5:7b-instruct-q4_K_M` digest `845dbda0ea48ed74`
- Model: `llama3.2:3b-instruct-q4_K_M` digest `a80c4f17acd55265`
- Determinism: temperature 0.0, seed 42, num_ctx 16384
- Companies: 10 (AAPL, CAT, COST, JNJ, KO, MSFT, NVDA, PG, WMT, XOM)
- Graded items: 600

## Headline: accuracy per model x cell

| model | cell | n scored | accuracy [95% CI] |
|---|---|---|---|
| `llama3.1:8b-instruct-q4_K_M` | A1 | 49 | 93.9% [85.4, 100.0] |
| `llama3.1:8b-instruct-q4_K_M` | A2 | 49 | 89.8% [83.7, 95.9] |
| `llama3.1:8b-instruct-q4_K_M` | B1 | 49 | 85.7% [77.6, 93.8] |
| `llama3.1:8b-instruct-q4_K_M` | B2 | 49 | 81.6% [72.9, 90.0] |
| `llama3.2:3b-instruct-q4_K_M` | A1 | 49 | 49.0% [33.3, 63.3] |
| `llama3.2:3b-instruct-q4_K_M` | A2 | 49 | 69.4% [58.0, 81.6] |
| `llama3.2:3b-instruct-q4_K_M` | B1 | 49 | 28.6% [17.4, 40.0] |
| `llama3.2:3b-instruct-q4_K_M` | B2 | 49 | 30.6% [18.0, 42.9] |
| `qwen2.5:7b-instruct-q4_K_M` | A1 | 49 | 51.0% [38.0, 64.0] |
| `qwen2.5:7b-instruct-q4_K_M` | A2 | 49 | 59.2% [46.8, 71.4] |
| `qwen2.5:7b-instruct-q4_K_M` | B1 | 49 | 42.9% [34.0, 54.0] |
| `qwen2.5:7b-instruct-q4_K_M` | B2 | 49 | 49.0% [37.0, 61.2] |

## Accuracy per model (all cells)

| model | n scored | accuracy [95% CI] |
|---|---|---|
| `llama3.1:8b-instruct-q4_K_M` | 196 | 87.8% [81.5, 93.8] |
| `llama3.2:3b-instruct-q4_K_M` | 196 | 44.4% [35.6, 53.6] |
| `qwen2.5:7b-instruct-q4_K_M` | 196 | 50.5% [40.8, 60.5] |

## Accuracy per cell (all models)

| cell | context | output | n scored | accuracy [95% CI] |
|---|---|---|---|---|
| A1 | Item-8 section | free-form | 147 | 64.6% [57.3, 72.0] |
| A2 | Item-8 section | schema-constrained | 147 | 72.8% [66.7, 79.9] |
| B1 | BM25 retrieval | free-form | 147 | 52.4% [47.1, 57.8] |
| B2 | BM25 retrieval | schema-constrained | 147 | 53.7% [47.6, 61.2] |

## Per-KPI breakdown

| KPI | `llama3.1:8b-instruct-q4_K_M` | `llama3.2:3b-instruct-q4_K_M` | `qwen2.5:7b-instruct-q4_K_M` |
|---|---|---|---|
| eps_diluted | 97.5% [92.5, 100.0] | 45.0% [27.5, 62.5] | 7.5% [0.0, 15.0] |
| net_income | 88.9% [75.0, 100.0] | 44.4% [19.4, 70.0] | 75.0% [53.1, 94.4] |
| operating_cash_flow | 92.5% [82.5, 100.0] | 40.0% [22.5, 55.0] | 47.5% [25.0, 70.0] |
| revenue | 70.0% [47.5, 90.0] | 42.5% [25.0, 60.0] | 60.0% [42.5, 77.5] |
| total_assets | 90.0% [77.5, 100.0] | 50.0% [32.5, 70.0] | 65.0% [52.5, 77.5] |

## Pairwise deltas (95% CI)

Deltas are paired by company: a bootstrap replicate draws a company once and takes its items from both arms. A delta whose interval includes zero is **inconclusive** — the data does not distinguish the two arms.

### Model vs model (all cells)

| comparison | delta accuracy [95% CI] | reading |
|---|---|---|
| `llama3.1:8b-instruct-q4_K_M` - `llama3.2:3b-instruct-q4_K_M` | 43.4% [31.5, 55.1] | favours `llama3.1:8b-instruct-q4_K_M` |
| `llama3.1:8b-instruct-q4_K_M` - `qwen2.5:7b-instruct-q4_K_M` | 37.2% [28.1, 46.8] | favours `llama3.1:8b-instruct-q4_K_M` |
| `llama3.2:3b-instruct-q4_K_M` - `qwen2.5:7b-instruct-q4_K_M` | -6.1% [-19.5, 7.7] | **inconclusive** (CI includes zero) |

### Strategy: context and output mode

| comparison | delta accuracy [95% CI] | reading |
|---|---|---|
| Item-8 section - BM25 retrieval | 15.6% [9.0, 22.3] | favours Item-8 section |
| free-form - schema-constrained | -4.8% [-8.5, -1.3] | favours schema-constrained |

## Failure taxonomy

230 incorrect / failed items out of 588 scored.

| label | n | what the grader can see |
|---|---|---|
| `wrong-period` | 15 | stated period outside the +/-7 day window |
| `wrong-concept` | 0 | currency is not USD — **only** the currency flavour |
| `refusal` | 0 | model declined to answer |
| `format-failure` | 0 | no readable figure in the output |
| unlabelled — awaiting the Day D review loop | 215 | (see below) |

**Read the zeros carefully.** A zero here means the grader's *automatic* test did not fire, not that the failure mode is absent. `wrong-concept` counts only non-USD answers; a model that grabs the wrong line item — XOM's "Sales and other operating revenue" (323,905) in place of "Total revenues and other income" (332,238) — is a wrong-concept error that lands in the unlabelled bucket, because telling it apart from any other wrong number requires reading the filing. The same is true of `scale-error` and `hallucination`. Those labels are assigned by hand in the Day D review loop (§7), which is why the unlabelled row is large: it is the work, not a gap.

## Excluded: no ground truth

12 items excluded from every accuracy above: no tag in the frozen fallback chain resolved for the pinned period, so there is nothing to grade against (§2). They are excluded from the denominator, not counted as wrong.

| company | KPI | items |
|---|---|---|
| CAT | net_income | 12 |

## Reading these numbers

N = 10 companies. The confidence intervals are wide, and that is the honest cost of shipping v0.1 at this size — not a presentation choice. Where a delta's interval includes zero, this report says **inconclusive** and means it: the run does not distinguish those arms. Accuracy here describes large-cap, plain us-gaap filers and will not generalise to messy or small filers, which is a deliberate scope cut (§13).
