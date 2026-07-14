# STATE

Build order and current status for filinglens v0.1. The frozen design lives in
[docs/architecture.md](docs/architecture.md); decisions in [docs/decisions.md](docs/decisions.md).

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
- [ ] Day B
- [ ] Day C
- [ ] Day D
