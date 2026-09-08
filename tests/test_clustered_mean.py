"""Unit tests for the continuous cluster-robust mean used by the Q-model comparison."""

from __future__ import annotations

import math
import random

import pytest

from vgc.evaluation import clustered_mean


def test_empty_input_is_reported_as_zero_rather_than_crashing():
    result = clustered_mean([])
    assert result.clusters == 0 and result.items == 0 and result.mean == 0.0


def test_single_cluster_falls_back_to_the_naive_error():
    result = clustered_mean([("only", [1.0, 2.0, 3.0])])
    assert result.clusters == 1
    assert result.mean == pytest.approx(2.0)
    assert result.clustered_se == pytest.approx(result.naive_se)
    assert result.se_floor == 0.0


def test_identical_clusters_show_no_between_cluster_spread():
    clusters = [(f"c{index}", [1.0, -1.0, 0.5, -0.5]) for index in range(10)]
    result = clustered_mean(clusters)
    assert result.mean == pytest.approx(0.0)
    assert result.tau == pytest.approx(0.0)
    assert result.se_floor == pytest.approx(0.0)
    # With no cluster effect the robust SE should not inflate much over the naive one.
    assert result.design_effect < 1.5


def test_a_real_cluster_effect_inflates_the_error_and_creates_a_floor():
    # Each cluster is internally constant but clusters differ -- pure between-cluster
    # variance, which more items per cluster can never reduce.
    clusters = [(f"c{index}", [index / 10.0] * 20) for index in range(10)]
    result = clustered_mean(clusters)
    assert result.tau > 0.2
    assert result.within_sd == pytest.approx(0.0, abs=1e-9)
    assert result.clustered_se > result.naive_se
    assert result.design_effect > 5.0
    assert result.se_floor == pytest.approx(result.tau / math.sqrt(10), rel=1e-6)


def test_interval_brackets_the_mean_and_widens_with_z():
    clusters = [(f"c{index}", [index / 10.0] * 5) for index in range(8)]
    result = clustered_mean(clusters)
    low, high = result.interval()
    assert low < result.mean < high
    wider_low, wider_high = result.interval(z=2.58)
    assert wider_low < low and wider_high > high


def test_detectable_effects_follow_from_the_two_error_sources():
    clusters = [(f"c{index}", [index / 10.0] * 5) for index in range(8)]
    result = clustered_mean(clusters)
    assert result.minimum_detectable_effect == pytest.approx(1.96 * result.clustered_se)
    assert result.floor_detectable_effect == pytest.approx(1.96 * result.se_floor)
    # The floor is what more items per cluster cannot buy, so it is the smaller bound.
    assert result.floor_detectable_effect <= result.minimum_detectable_effect + 1e-12


def test_clusters_needed_grows_as_the_target_effect_shrinks():
    clusters = [(f"c{index}", [index / 10.0] * 5) for index in range(8)]
    result = clustered_mean(clusters)
    assert result.clusters_needed(0.0, 4) == 0
    coarse = result.clusters_needed(0.10, 4)
    fine = result.clusters_needed(0.03, 4)
    assert fine > coarse > 0
    # More items per cluster helps only through the within-cluster term.
    assert result.clusters_needed(0.03, 40) <= fine


def test_unequal_cluster_sizes_are_handled_and_weighting_is_by_item():
    result = clustered_mean([("big", [1.0] * 90), ("small", [0.0] * 10)])
    assert result.items == 100
    assert result.mean == pytest.approx(0.9)


def test_robust_error_beats_the_naive_one_on_clustered_random_data():
    # 30 clusters whose true means vary; the naive SE ignores that and reads too small.
    rng = random.Random(5)
    clusters = []
    for index in range(30):
        centre = rng.gauss(0.0, 0.3)
        clusters.append((f"c{index}", [centre + rng.gauss(0.0, 0.1) for _ in range(20)]))
    result = clustered_mean(clusters)
    assert result.clusters == 30 and result.items == 600
    assert result.tau > 0.15
    assert result.clustered_se > 2.0 * result.naive_se
