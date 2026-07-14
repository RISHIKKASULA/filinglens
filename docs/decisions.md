# Decisions (ADR log)

Deviations from the frozen spec ([architecture.md](architecture.md)) and gate results are
recorded here. Simplest defensible choice wins.

## ADR-001 — Ground-truth sanity gate (§0)

**Status: PENDING.** `filinglens sanity AAPL MSFT` output (2 companies × 5 KPIs = 10
XBRL-fact vs filing-text pairs) awaiting manual review. Gate: ≥ 9/10 clean matches → PASS
and proceed; below → STOP, re-scope by ADR before any grader work.
