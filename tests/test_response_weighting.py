"""Unit tests for the opponent-response weighting sweep's math."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from vgc.rl.response_weighting import (
    RootTable,
    aggregate_values,
    build_root_tables,
    equal_weight,
    evaluate_rule,
    evaluate_rule_over_truth_grid,
    group_by_stratum,
    headroom_summary,
    null_distribution,
    null_resample,
    regret_surface,
    search_proportional_weight,
    sensitivity_summary,
    softmax_weight_fn,
    sweep_fixed_weights,
    top_response_weight,
    weight_grid,
)


@dataclass(frozen=True)
class FakeSample:
    root_id: str
    own_action: str
    opponent_action: str
    mean_outcome: float
    search_score: float
    opponent_search_score: float
    team_id: str = "pool/team_a.packed"
    opponent_team_id: str = "pool/team_b.packed"
    rollout_count: int = 4
    root_metadata: dict[str, Any] = field(default_factory=lambda: {"turn_bucket": "early"})


def make_table(outcomes, *, root_id="root", bucket="early", response_scores=(20.0, 10.0)):
    outcomes = np.asarray(outcomes, dtype=np.float64)
    return RootTable(
        root_id=root_id,
        turn_bucket=bucket,
        team_id="pool/team_a.packed",
        opponent_team_id="pool/team_b.packed",
        own_actions=tuple(f"a{index}" for index in range(outcomes.shape[0])),
        responses=("b0", "b1"),
        outcomes=outcomes,
        rollout_counts=np.full(outcomes.shape, 4, dtype=np.int64),
        search_scores=np.arange(outcomes.shape[0], 0, -1, dtype=np.float64),
        response_scores=np.asarray(response_scores, dtype=np.float64),
    )


def test_weight_grid_keeps_the_uniform_baseline_on_the_grid():
    grid = weight_grid(100)
    assert grid[0] == 0.0 and grid[-1] == 1.0
    assert 0.5 in set(grid.tolist())
    with pytest.raises(ValueError):
        weight_grid(101)


def test_aggregate_values_puts_the_weight_on_the_first_response():
    outcomes = np.asarray([[1.0, -1.0], [0.0, 0.5]])
    values = aggregate_values(outcomes, np.asarray([0.0, 0.5, 1.0]))
    assert values.shape == (2, 3)
    np.testing.assert_allclose(values[:, 0], [-1.0, 0.5])  # all mass on b1
    np.testing.assert_allclose(values[:, 1], [0.0, 0.25])
    np.testing.assert_allclose(values[:, 2], [1.0, 0.0])  # all mass on b0


def test_build_root_tables_orders_actions_and_responses_by_search_score():
    samples = [
        FakeSample("r", "weak", "reply_lo", 0.0, 10.0, 5.0),
        FakeSample("r", "weak", "reply_hi", 0.5, 10.0, 9.0),
        FakeSample("r", "strong", "reply_lo", 1.0, 90.0, 5.0),
        FakeSample("r", "strong", "reply_hi", -1.0, 90.0, 9.0),
    ]
    tables, skipped = build_root_tables(samples)
    assert skipped == {}
    table = tables[0]
    assert table.own_actions == ("strong", "weak")
    assert table.responses == ("reply_hi", "reply_lo")
    np.testing.assert_allclose(table.outcomes, [[-1.0, 1.0], [0.5, 0.0]])
    assert table.turn_bucket == "early"


def test_build_root_tables_skips_roots_without_exactly_two_responses():
    samples = [
        FakeSample("r", "a", "only_reply", 1.0, 10.0, 5.0),
        FakeSample("r", "b", "only_reply", 0.0, 9.0, 5.0),
    ]
    tables, skipped = build_root_tables(samples)
    assert tables == []
    assert skipped == {"responses_1": 1}


def test_build_root_tables_skips_ragged_roots():
    samples = [
        FakeSample("r", "a", "x", 1.0, 10.0, 9.0),
        FakeSample("r", "a", "y", 1.0, 10.0, 5.0),
        FakeSample("r", "b", "x", 0.0, 9.0, 9.0),
    ]
    tables, skipped = build_root_tables(samples)
    assert tables == []
    assert skipped == {"ragged_matrix": 1}


def test_regret_surface_is_zero_on_the_diagonal():
    table = make_table([[1.0, -1.0], [-1.0, 1.0]])
    grid = weight_grid(10)
    surface = regret_surface(table, grid)
    np.testing.assert_allclose(np.diag(surface), 0.0, atol=1e-12)
    assert (surface >= -1e-12).all()


def test_a_response_independent_root_has_no_sensitivity_and_no_headroom():
    # Both responses give identical payoffs, so the weight cannot matter at all.
    table = make_table([[0.5, 0.5], [-0.5, -0.5]])
    grid = weight_grid(20)
    sensitivity = sensitivity_summary([table], grid)
    headroom = headroom_summary([table], grid)
    assert sensitivity["stable_fraction"] == 1.0
    assert sensitivity["flip_fraction"] == 0.0
    assert headroom["uniform_regret_mean_over_p"] == pytest.approx(0.0)
    assert headroom["uniform_regret_worst_over_p"] == pytest.approx(0.0)


def test_a_crossing_root_flips_and_exposes_headroom():
    # a0 wins when the opponent plays b0, a1 wins when they play b1; they cross at 0.5.
    table = make_table([[1.0, -1.0], [-1.0, 1.0]])
    grid = weight_grid(20)
    sensitivity = sensitivity_summary([table], grid)
    headroom = headroom_summary([table], grid)
    assert sensitivity["flip_fraction"] == 1.0
    assert sensitivity["mean_distinct_best_actions"] == 2.0
    assert sensitivity["crossover_weights"]["count"] == 1
    assert sensitivity["crossover_weights"]["median"] == pytest.approx(0.55, abs=0.06)
    # Committing to 0.5 and being wrong costs up to the full payoff swing.
    assert headroom["uniform_regret_worst_over_p"] == pytest.approx(2.0, abs=0.11)
    assert headroom["uniform_regret_mean_over_p"] > 0.4


def test_uniform_is_optimal_under_a_uniform_prior_by_linearity():
    # An action's value is linear in p, so its average over a uniform prior on p equals
    # its value at p=0.5 -- no other fixed weight can beat 0.5 on that criterion, ever.
    rng = np.random.default_rng(7)
    grid = weight_grid(20)
    for index in range(15):
        table = make_table(rng.uniform(-1.0, 1.0, size=(3, 2)), root_id=f"r{index}")
        headroom = headroom_summary([table], grid)
        assert headroom["linearity_check_uniform_achieves_best_mean_over_p"]
        assert headroom["uniform_regret_mean_over_p_minus_best_fixed"] <= 1e-9


def test_minimax_weight_can_beat_uniform_where_the_prior_average_cannot():
    # Uniform picks a0, whose downside at p=1 is the worst on the board; nudging the
    # deployed weight to 0.55 selects the flatter a1 and caps the worst case instead.
    table = make_table([[-0.75, 0.75], [0.5, -0.5], [0.75, -1.0]])
    grid = weight_grid(20)
    headroom = headroom_summary([table], grid)
    assert headroom["linearity_check_uniform_achieves_best_mean_over_p"]
    assert headroom["minimax_fixed_weight"] == pytest.approx(0.55)
    assert headroom["uniform_worst_regret_minus_minimax"] == pytest.approx(0.25)


def test_minimax_weight_is_never_worse_than_uniform_on_the_worst_case():
    rng = np.random.default_rng(11)
    grid = weight_grid(20)
    for index in range(15):
        table = make_table(rng.uniform(-1.0, 1.0, size=(3, 2)), root_id=f"r{index}")
        headroom = headroom_summary([table], grid)
        assert headroom["uniform_worst_regret_minus_minimax"] >= -1e-12


def test_evaluate_rule_scores_a_perfect_predictor_as_zero_regret_at_matching_truth():
    table = make_table([[1.0, -1.0], [-1.0, 1.0]])
    predictions = {table.root_id: table.outcomes}
    result = evaluate_rule(
        [table], predictions, rule="equal", weight_fn=equal_weight, truth_weight=0.5
    )
    assert result["mean_regret"] == pytest.approx(0.0)
    assert result["top_action_accuracy"] == 1.0


def test_evaluate_rule_penalises_a_mismatched_deployment_weight():
    table = make_table([[1.0, -1.0], [-1.0, 1.0]])
    predictions = {table.root_id: table.outcomes}
    # Deploy "opponent always plays b0" while the truth is "always b1".
    result = evaluate_rule(
        [table], predictions, rule="top", weight_fn=top_response_weight, truth_weight=0.0
    )
    assert result["mean_regret"] == pytest.approx(2.0)
    assert result["top_action_accuracy"] == 0.0


def test_worst_case_rule_ignores_the_weight_and_maximises_the_minimum():
    # a0 is a coin flip; a1 is a guaranteed small win.  Minimax must take a1.
    table = make_table([[1.0, -0.5], [0.2, 0.2]])
    predictions = {table.root_id: table.outcomes}
    result = evaluate_rule([table], predictions, rule="worst_case", truth_weight=0.5)
    # Minimax takes the flat a1 (worst -0.5 vs 0.2) even though a0 wins on average,
    # so it gives up 0.25 - 0.2 against the uniform truth.
    assert result["mean_regret"] == pytest.approx(0.05)
    assert result["deployed_weight_spread"] == pytest.approx(0.0)


def test_evaluate_rule_rejects_a_prediction_shape_mismatch():
    table = make_table([[1.0, -1.0], [-1.0, 1.0]])
    with pytest.raises(ValueError):
        evaluate_rule([table], {table.root_id: np.zeros((3, 2))}, rule="equal")


def test_rule_over_truth_grid_averages_the_unknown_true_distribution():
    table = make_table([[1.0, -1.0], [-1.0, 1.0]])
    predictions = {table.root_id: table.outcomes}
    result = evaluate_rule_over_truth_grid(
        [table], predictions, weight_grid(10), rule="equal", weight_fn=equal_weight
    )
    assert result["regret_at_uniform_truth"] == pytest.approx(0.0)
    # Judging only at 0.5 hides the cost of being wrong about the true weight.
    assert result["mean_regret_over_truth"] > 0.0
    assert result["worst_regret_over_truth"] >= result["mean_regret_over_truth"]
    assert len(result["by_truth_weight"]) == 11


def test_softmax_and_proportional_rules_track_the_opponent_search_gap():
    wide = make_table([[1.0, -1.0], [-1.0, 1.0]], response_scores=(100.0, 0.0))
    narrow = make_table([[1.0, -1.0], [-1.0, 1.0]], response_scores=(50.0, 50.0))
    sharp = softmax_weight_fn(1.0)
    smooth = softmax_weight_fn(250.0)
    assert sharp(wide) > 0.99
    assert smooth(wide) == pytest.approx(0.5, abs=0.15)
    assert sharp(narrow) == pytest.approx(0.5)
    assert search_proportional_weight(wide) == pytest.approx(1.0)
    assert search_proportional_weight(narrow) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        softmax_weight_fn(0.0)


def test_search_proportional_weight_handles_negative_scores():
    table = make_table([[1.0, -1.0], [-1.0, 1.0]], response_scores=(-10.0, -30.0))
    assert 0.5 < search_proportional_weight(table) <= 1.0


def test_sweep_fixed_weights_covers_the_whole_grid():
    table = make_table([[1.0, -1.0], [-1.0, 1.0]])
    rows = sweep_fixed_weights([table], {table.root_id: table.outcomes}, weight_grid(10))
    assert len(rows) == 11
    assert rows[0]["mean_deployed_weight"] == pytest.approx(0.0)
    assert rows[-1]["mean_deployed_weight"] == pytest.approx(1.0)


def test_null_resample_removes_the_response_effect_in_expectation():
    # Deterministic payoffs that depend strongly on the response, pooled to a constant.
    table = make_table([[1.0, -1.0], [-1.0, 1.0]])
    rng = np.random.default_rng(0)
    pooled = np.stack([null_resample([table], rng)[0].outcomes for _ in range(400)])
    # Pooling averages each action to 0, so both cells are fair coins with mean ~0.
    assert abs(float(pooled.mean())) < 0.05
    # And the two responses are exchangeable under the null.
    assert abs(float(pooled[:, :, 0].mean() - pooled[:, :, 1].mean())) < 0.07


def test_null_resample_preserves_shape_and_rollout_granularity():
    table = make_table([[0.5, 0.0], [-0.5, 0.25]])
    drawn = null_resample([table], np.random.default_rng(1))[0]
    assert drawn.outcomes.shape == table.outcomes.shape
    assert drawn.root_id == table.root_id
    # Four continuations per cell means every estimate is a multiple of 0.5.
    np.testing.assert_allclose(drawn.outcomes * 2.0, np.round(drawn.outcomes * 2.0))


def test_null_distribution_reports_a_nonzero_noise_floor():
    tables = [make_table([[0.0, 0.0], [0.0, 0.0]], root_id=f"r{index}") for index in range(20)]
    null = null_distribution(tables, weight_grid(10), draws=25, seed=3)
    # Even with zero true response effect, 4 rollouts per cell manufacture flips.
    assert null["flip_fraction"]["mean"] > 0.0
    assert null["uniform_regret_mean_over_p"]["mean"] > 0.0
    with pytest.raises(ValueError):
        null_distribution(tables, weight_grid(10), draws=0, seed=3)


def test_group_by_stratum_partitions_roots():
    tables = [
        make_table([[1.0, 0.0], [0.0, 1.0]], root_id="a", bucket="early"),
        make_table([[1.0, 0.0], [0.0, 1.0]], root_id="b", bucket="late"),
        make_table([[1.0, 0.0], [0.0, 1.0]], root_id="c", bucket="early"),
    ]
    grouped = group_by_stratum(tables)
    assert sorted(grouped) == ["early", "late"]
    assert len(grouped["early"]) == 2


def test_summaries_reject_empty_input():
    with pytest.raises(ValueError):
        sensitivity_summary([], weight_grid(10))
    with pytest.raises(ValueError):
        headroom_summary([], weight_grid(10))
