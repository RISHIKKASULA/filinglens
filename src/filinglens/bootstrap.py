"""Cluster-bootstrap confidence intervals (architecture.md §4).

Items within a company are correlated — the same filing, the same layout, the same filer's
house style — so treating 50 items from 10 companies as 50 independent draws would
understate uncertainty badly. The frozen remedy is a **cluster bootstrap by company**:
resample the 10 companies with replacement, recompute the statistic on the resampled
item pool, and take percentile intervals over B = 10,000 replicates with seed 42.

Every accuracy and every pairwise delta in the report carries a 95% CI from this module.
At N=10 those intervals are wide, and §13(g) requires saying so plainly rather than
quietly reporting point estimates: a delta whose CI includes zero is reported as
inconclusive, in those words.

Determinism is a property, not an accident: the same inputs and seed always give the same
interval, so a report can be regenerated and compared.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from pydantic import BaseModel

# Frozen bootstrap settings (§4).
N_RESAMPLES = 10_000
SEED = 42
CI_LEVEL = 0.95


class Interval(BaseModel):
    """A point estimate with a percentile confidence interval.

    ``point`` is computed on the observed data, not from the bootstrap replicates: the
    replicates estimate the sampling distribution, they do not redefine the estimate.
    """

    point: float | None
    low: float | None
    high: float | None
    level: float = CI_LEVEL
    n_clusters: int = 0
    n_items: int = 0

    @property
    def includes_zero(self) -> bool:
        """Whether the interval spans zero — the §13(g) inconclusive test for a delta."""
        if self.low is None or self.high is None:
            return True
        return self.low <= 0.0 <= self.high

    def format(self, pct: bool = True, places: int = 1) -> str:
        """Human-readable "point [low, high]", or "n/a" when unestimable."""
        if self.point is None or self.low is None or self.high is None:
            return "n/a"
        scale = 100.0 if pct else 1.0
        suffix = "%" if pct else ""
        return (
            f"{self.point * scale:.{places}f}{suffix} "
            f"[{self.low * scale:.{places}f}, {self.high * scale:.{places}f}]"
        )


def _percentiles(replicates: list[float], level: float) -> tuple[float, float]:
    alpha = (1.0 - level) / 2.0
    lo = float(np.percentile(replicates, alpha * 100.0))
    hi = float(np.percentile(replicates, (1.0 - alpha) * 100.0))
    return lo, hi


def cluster_bootstrap(
    clusters: Sequence[Sequence[object]],
    statistic: Callable[[list[object]], float | None],
    *,
    n_resamples: int = N_RESAMPLES,
    seed: int = SEED,
    level: float = CI_LEVEL,
) -> Interval:
    """Percentile CI for ``statistic``, resampling whole ``clusters`` with replacement.

    ``clusters`` is one sequence of items per company. Each replicate draws len(clusters)
    companies with replacement, concatenates their items, and evaluates the statistic.
    Replicates where the statistic is undefined (e.g. an all-excluded resample) are
    dropped rather than counted as zero, which would drag the interval toward a value the
    data never supports.
    """
    pools = [list(c) for c in clusters if len(c) > 0]
    n_items = sum(len(p) for p in pools)
    if not pools:
        return Interval(point=None, low=None, high=None, level=level)

    observed = statistic([item for pool in pools for item in pool])
    if observed is None:
        return Interval(
            point=None, low=None, high=None, level=level, n_clusters=len(pools), n_items=n_items
        )

    rng = np.random.default_rng(seed)
    k = len(pools)
    replicates: list[float] = []
    for _ in range(n_resamples):
        picks = rng.integers(0, k, size=k)
        resampled = [item for i in picks for item in pools[i]]
        value = statistic(resampled)
        if value is not None:
            replicates.append(value)

    if not replicates:  # pragma: no cover — needs a statistic defined on the whole but no part
        return Interval(
            point=observed, low=None, high=None, level=level, n_clusters=k, n_items=n_items
        )

    low, high = _percentiles(replicates, level)
    return Interval(point=observed, low=low, high=high, level=level, n_clusters=k, n_items=n_items)


def paired_cluster_bootstrap(
    clusters_a: Sequence[Sequence[object]],
    clusters_b: Sequence[Sequence[object]],
    statistic: Callable[[list[object]], float | None],
    *,
    n_resamples: int = N_RESAMPLES,
    seed: int = SEED,
    level: float = CI_LEVEL,
) -> Interval:
    """CI for the delta ``statistic(a) - statistic(b)``, resampling companies **paired**.

    The two arms are measured on the same companies, so a replicate must draw a company
    once and take that company's items from *both* arms. Resampling the arms independently
    would discard the pairing and inflate the delta's interval — the comparison is
    within-company by construction, and the CI should reflect that.

    ``clusters_a[i]`` and ``clusters_b[i]`` must describe the same company.
    """
    if len(clusters_a) != len(clusters_b):
        raise ValueError("paired bootstrap needs one cluster per company in both arms")

    pairs = [
        (list(a), list(b))
        for a, b in zip(clusters_a, clusters_b, strict=True)
        if len(a) > 0 or len(b) > 0
    ]
    n_items = sum(len(a) + len(b) for a, b in pairs)
    if not pairs:
        return Interval(point=None, low=None, high=None, level=level)

    def delta(sample: list[tuple[list[object], list[object]]]) -> float | None:
        a_items = [item for a, _ in sample for item in a]
        b_items = [item for _, b in sample for item in b]
        stat_a = statistic(a_items)
        stat_b = statistic(b_items)
        if stat_a is None or stat_b is None:
            return None
        return stat_a - stat_b

    observed = delta(pairs)
    if observed is None:
        return Interval(
            point=None, low=None, high=None, level=level, n_clusters=len(pairs), n_items=n_items
        )

    rng = np.random.default_rng(seed)
    k = len(pairs)
    replicates: list[float] = []
    for _ in range(n_resamples):
        picks = rng.integers(0, k, size=k)
        value = delta([pairs[i] for i in picks])
        if value is not None:
            replicates.append(value)

    if not replicates:  # pragma: no cover
        return Interval(
            point=observed, low=None, high=None, level=level, n_clusters=k, n_items=n_items
        )

    low, high = _percentiles(replicates, level)
    return Interval(point=observed, low=low, high=high, level=level, n_clusters=k, n_items=n_items)
