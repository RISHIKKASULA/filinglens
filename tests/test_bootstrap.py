from typing import Any

import pytest

from filinglens import bootstrap
from filinglens.bootstrap import Interval


# The statistic under test throughout: share of 1s in a pool of 0/1 items — accuracy's
# shape, without dragging grade.py's types into a test about resampling.
def _rate(items: list[Any]) -> float | None:
    if not items:
        return None
    return sum(items) / len(items)


def _company(correct: int, wrong: int) -> list[int]:
    return [1] * correct + [0] * wrong


# --- seeded determinism (§10) -------------------------------------------------------


def test_same_seed_gives_identical_intervals() -> None:
    clusters = [_company(3, 2), _company(5, 0), _company(0, 5), _company(2, 3)]
    a = bootstrap.cluster_bootstrap(clusters, _rate, n_resamples=500, seed=42)
    b = bootstrap.cluster_bootstrap(clusters, _rate, n_resamples=500, seed=42)
    assert (a.low, a.point, a.high) == (b.low, b.point, b.high)


def test_different_seeds_give_different_intervals() -> None:
    clusters = [_company(3, 2), _company(5, 0), _company(0, 5), _company(2, 3)]
    a = bootstrap.cluster_bootstrap(clusters, _rate, n_resamples=500, seed=42)
    b = bootstrap.cluster_bootstrap(clusters, _rate, n_resamples=500, seed=7)
    assert (a.low, a.high) != (b.low, b.high)  # sampling noise is real...
    assert a.point == b.point  # ...but the point estimate is not resampled


def test_frozen_defaults_are_the_spec_values() -> None:
    assert bootstrap.N_RESAMPLES == 10_000
    assert bootstrap.SEED == 42
    assert bootstrap.CI_LEVEL == 0.95


# --- the interval contains the point estimate (§10) ---------------------------------


def test_ci_contains_the_point_estimate() -> None:
    clusters = [_company(4, 1), _company(3, 2), _company(5, 0), _company(1, 4)]
    ci = bootstrap.cluster_bootstrap(clusters, _rate, n_resamples=2000, seed=42)
    assert ci.point is not None and ci.low is not None and ci.high is not None
    assert ci.low <= ci.point <= ci.high


def test_point_estimate_is_the_observed_value_not_a_replicate_mean() -> None:
    clusters = [_company(1, 0), _company(0, 1), _company(1, 0), _company(1, 0)]
    ci = bootstrap.cluster_bootstrap(clusters, _rate, n_resamples=1000, seed=42)
    assert ci.point == 0.75  # 3 of 4 items correct, computed on the data as observed


def test_ci_is_wider_with_fewer_clusters() -> None:
    # The §13(g) reality: N=10 gives wide intervals. More clusters, tighter interval.
    few = [_company(3, 2), _company(2, 3)]
    many = [_company(3, 2), _company(2, 3)] * 10
    ci_few = bootstrap.cluster_bootstrap(few, _rate, n_resamples=2000, seed=42)
    ci_many = bootstrap.cluster_bootstrap(many, _rate, n_resamples=2000, seed=42)
    assert ci_few.point == ci_many.point
    assert (ci_few.high - ci_few.low) > (ci_many.high - ci_many.low)


# --- the rigged 2-company case with hand-checkable percentiles (§10) ----------------


def test_rigged_two_company_case_has_hand_checkable_support() -> None:
    """Two companies, one all-correct and one all-wrong.

    Resampling 2 clusters with replacement gives exactly three equally-likely-in-shape
    outcomes: {A,A} -> 1.0, {B,B} -> 0.0, and {A,B} or {B,A} -> 0.5. So every replicate
    must land on one of {0.0, 0.5, 1.0}, the observed point is 0.5, and a 95% percentile
    interval over that support can only be [0.0, 1.0] — the widest possible. This is the
    case where the bootstrap's output is checkable by hand rather than by trust.
    """
    clusters = [_company(5, 0), _company(0, 5)]
    ci = bootstrap.cluster_bootstrap(clusters, _rate, n_resamples=2000, seed=42)
    assert ci.point == 0.5
    assert ci.low == 0.0
    assert ci.high == 1.0
    assert ci.n_clusters == 2
    assert ci.n_items == 10


def test_identical_clusters_give_a_degenerate_interval() -> None:
    # Every resample is the same pool, so there is no sampling variation to find.
    clusters = [_company(1, 1), _company(1, 1), _company(1, 1)]
    ci = bootstrap.cluster_bootstrap(clusters, _rate, n_resamples=500, seed=42)
    assert ci.point == ci.low == ci.high == 0.5


def test_all_correct_pins_the_interval_at_one() -> None:
    clusters = [_company(5, 0), _company(3, 0), _company(4, 0)]
    ci = bootstrap.cluster_bootstrap(clusters, _rate, n_resamples=500, seed=42)
    assert (ci.point, ci.low, ci.high) == (1.0, 1.0, 1.0)


# --- degenerate inputs --------------------------------------------------------------


def test_no_clusters_is_unestimable() -> None:
    ci = bootstrap.cluster_bootstrap([], _rate, n_resamples=100, seed=42)
    assert (ci.point, ci.low, ci.high) == (None, None, None)
    assert ci.format() == "n/a"


def test_empty_clusters_are_dropped() -> None:
    ci = bootstrap.cluster_bootstrap([[], _company(2, 0), []], _rate, n_resamples=500, seed=42)
    assert ci.n_clusters == 1
    assert ci.point == 1.0


def test_statistic_undefined_on_the_whole_pool_is_unestimable() -> None:
    ci = bootstrap.cluster_bootstrap([[1, 2]], lambda _: None, n_resamples=100, seed=42)
    assert ci.point is None
    assert ci.n_clusters == 1


def test_single_cluster_has_no_between_cluster_variation() -> None:
    ci = bootstrap.cluster_bootstrap([_company(3, 1)], _rate, n_resamples=500, seed=42)
    assert ci.point == ci.low == ci.high == 0.75


# --- delta CIs and the inconclusive test (§13(g)) -----------------------------------


def test_delta_ci_sign_sanity_when_one_arm_clearly_wins() -> None:
    # Arm A is right everywhere, arm B wrong everywhere, on the same companies.
    a = [_company(5, 0) for _ in range(6)]
    b = [_company(0, 5) for _ in range(6)]
    ci = bootstrap.paired_cluster_bootstrap(a, b, _rate, n_resamples=1000, seed=42)
    assert ci.point == 1.0
    assert ci.low > 0  # the delta's CI excludes zero: a real difference
    assert not ci.includes_zero


def test_delta_ci_is_negative_when_the_second_arm_wins() -> None:
    a = [_company(0, 5) for _ in range(6)]
    b = [_company(5, 0) for _ in range(6)]
    ci = bootstrap.paired_cluster_bootstrap(a, b, _rate, n_resamples=1000, seed=42)
    assert ci.point == -1.0
    assert ci.high < 0


def test_identical_arms_give_a_zero_delta_reported_as_inconclusive() -> None:
    arm = [_company(3, 2) for _ in range(5)]
    ci = bootstrap.paired_cluster_bootstrap(arm, list(arm), _rate, n_resamples=1000, seed=42)
    assert ci.point == 0.0
    assert ci.includes_zero  # §13(g): reported as inconclusive, in those words


def test_delta_pairing_requires_matching_cluster_counts() -> None:
    with pytest.raises(ValueError, match="one cluster per company"):
        bootstrap.paired_cluster_bootstrap(
            [_company(1, 0)], [_company(1, 0), _company(0, 1)], _rate
        )


def test_paired_resampling_draws_each_company_once_for_both_arms() -> None:
    """The pairing is the point: a company must contribute to both arms or neither.

    Rigged so that pairing matters. Company 1: A right, B right. Company 2: A wrong, B
    wrong. Within every company the two arms agree exactly, so the true delta is 0 with
    no variation — and paired resampling must report exactly that. Resampling the arms
    independently would let company 1's A meet company 2's B and manufacture a spread.
    """
    a = [_company(5, 0), _company(0, 5)]
    b = [_company(5, 0), _company(0, 5)]
    ci = bootstrap.paired_cluster_bootstrap(a, b, _rate, n_resamples=2000, seed=42)
    assert ci.point == 0.0
    assert ci.low == 0.0
    assert ci.high == 0.0  # zero spread: the arms move together, by construction


def test_delta_is_unestimable_when_an_arm_is_empty() -> None:
    ci = bootstrap.paired_cluster_bootstrap([[]], [[]], _rate, n_resamples=100, seed=42)
    assert ci.point is None


def test_delta_unestimable_when_a_whole_arm_has_no_items() -> None:
    ci = bootstrap.paired_cluster_bootstrap([_company(2, 0)], [[]], _rate, n_resamples=100, seed=42)
    assert ci.point is None  # statistic undefined on arm B


# --- Interval formatting / reporting helpers ----------------------------------------


def test_format_as_percent() -> None:
    assert Interval(point=0.5, low=0.25, high=0.75).format() == "50.0% [25.0, 75.0]"


def test_format_raw() -> None:
    assert Interval(point=0.5, low=0.25, high=0.75).format(pct=False, places=2) == (
        "0.50 [0.25, 0.75]"
    )


def test_format_unestimable() -> None:
    assert Interval(point=None, low=None, high=None).format() == "n/a"


def test_includes_zero() -> None:
    assert Interval(point=0.1, low=-0.05, high=0.25).includes_zero
    assert not Interval(point=0.5, low=0.25, high=0.75).includes_zero
    assert not Interval(point=-0.5, low=-0.75, high=-0.25).includes_zero
    assert Interval(point=None, low=None, high=None).includes_zero  # unknown is not a claim


def test_interval_records_its_level() -> None:
    ci = bootstrap.cluster_bootstrap(
        [_company(3, 2), _company(2, 3)], _rate, n_resamples=500, seed=42, level=0.90
    )
    assert ci.level == 0.90


def test_narrower_level_gives_a_narrower_interval() -> None:
    clusters = [_company(4, 1), _company(3, 2), _company(5, 0), _company(1, 4)]
    ci95 = bootstrap.cluster_bootstrap(clusters, _rate, n_resamples=2000, seed=42, level=0.95)
    ci50 = bootstrap.cluster_bootstrap(clusters, _rate, n_resamples=2000, seed=42, level=0.50)
    assert (ci50.high - ci50.low) <= (ci95.high - ci95.low)
