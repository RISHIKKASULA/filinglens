# Changelog

All notable changes to filinglens. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-07-17

First release. An XBRL-graded evaluation harness measuring how reliably local 7–8B LLMs
extract financial figures from SEC 10-K text, with confidence intervals and a hand-labeled
failure taxonomy. The company's own XBRL facts are the auto-grader.

### The measured result

- **10 large-cap plain-us-gaap filers × 5 KPIs × 3 local models × a 2×2 strategy ablation
  = 600 calls**, deterministic (temperature 0, seed 42, `num_ctx` 16384), model weight
  digests pinned. 588 scored, 12 excluded as `no-ground-truth`.
- Accuracy per model, all cells (cluster bootstrap by company, B = 10,000, seed 42, 95% CI):
  `llama3.1:8b` **87.8% [81.5, 93.8]**, `qwen2.5:7b` **50.5% [40.8, 60.5]**,
  `llama3.2:3b` **44.4% [35.6, 53.6]**.
- **Size cliff** survives N=10: 8B − 3B = 43.4% [31.5, 55.1]; 8B − qwen = 37.2% [28.1, 46.8].
  The 3B-vs-qwen gap is **inconclusive** (−6.1% [−19.5, 7.7]).
- **qwen EPS collapse:** 7.5% [0.0, 15.0] on `eps_diluted` vs the 8B's 97.5% — right digits,
  mis-tagged scale. A single-KPI failure hiding inside a 50.5% average.
- **Strategy:** Item-8 section beats BM25 by 15.6% [9.0, 22.3]; schema-constrained beats
  free-form by 4.8% [1.3, 8.5].

### Failure taxonomy (hand-labeled, §7)

All 230 incorrect items labeled against the filing into `runs/v0.1/labels.csv`:
scale-error 129 · hallucination 55 · wrong-concept 30 · wrong-period 16.

### Added

- `corpus` — EDGAR fetch, cache, and CIK/accession/fiscal-period pinning (`corpus.yaml`).
- `sanity` — the blocking day-1 ground-truth gate (ADR-001, PASS 10/10).
- `sections` — Item-8 statement slicing with incorporation-by-reference fallback (ADR-002).
- `retrieve` — deterministic BM25 retrieval.
- `models` — Ollama client plus a `StubClient` so CI never touches network or Ollama.
- `prompts` / `extract` — the resumable, digest-pinned extraction grid.
- `normalize` / `grade` — scale/currency/period normalization and XBRL grading at the frozen
  §3 tolerances (`cents` understood on input, ADR-003).
- `bootstrap` / `report` — cluster-bootstrap CIs and the generated evaluation report.
- `label` — the `filinglens label` failure-review loop and the committed taxonomy.
- CLI: `fetch · sanity · grade · label · report`.

### Notes

- The v0.1 grid results are committed (`runs/v0.1/results.parquet` + `config.json` +
  `labels.csv`), so verdicts, CIs, and the report regenerate deterministically with no
  Ollama and no model calls (ADR-004). Ground-truth facts re-fetch once from EDGAR against
  the pinned accessions; CI runs offline on fixtures.
- **87.8% is headline-figure extraction with the statements already in context.** It is
  **not** comparable to FinanceBench's reasoning-question results and does not beat
  FinanceBench or any frontier model. FinanceBench (arXiv 2311.11944) is cited only as the
  prior-art anchor for the gap.
- §6's "~1–2.5 h unattended" runtime estimate measured ~5–7 h for 600 calls on a base M4
  (ADR-005).

[0.1.0]: https://github.com/RISHIKKASULA/filinglens/releases/tag/v0.1.0
