from __future__ import annotations

from types import SimpleNamespace

import pytest

from offline.evaluate_counterfactual_value import (
    _root_metadata,
    paired_clustered_comparison,
    ranking_metrics,
    summarize_records,
)


def test_root_metadata_labels_turn_actions_and_public_speed_control() -> None:
    battle = SimpleNamespace(
        turn=5,
        weather={"SUNNYDAY": 3},
        fields={"TRICK_ROOM": 2},
        side_conditions={"TAILWIND": 1},
        opponent_side_conditions={},
    )

    metadata = _root_metadata(battle, legal_action_count=42, search_gap=3.5)

    assert metadata["turn_bucket"] == "late"
    assert metadata["legal_action_count"] == 42
    assert metadata["search_top_gap"] == 3.5
    assert metadata["weather"] == ["sunnyday"]
    assert metadata["trick_room"] is True
    assert metadata["speed_control_active"] is True


def test_ranking_metrics_are_within_group_and_measure_regret() -> None:
    groups = [
        [
            {"score": 0.1, "mean_outcome": 1.0},
            {"score": 0.9, "mean_outcome": 0.0},
            {"score": 0.2, "mean_outcome": -1.0},
        ]
    ]
    metrics = ranking_metrics(groups, "score")
    assert metrics["pairwise_accuracy"] == pytest.approx(1 / 3)
    assert metrics["top_action_accuracy"] == 0.0
    assert metrics["mean_regret"] == 1.0


def test_summary_keeps_root_and_opponent_conditioned_questions_separate() -> None:
    records = []
    for opponent, outcomes in (("x", (1.0, -1.0)), ("y", (-1.0, 1.0))):
        for own, outcome, value, search in zip(
            ("a", "b"), outcomes, (0.8, 0.2), (2.0, 1.0), strict=True
        ):
            records.append(
                {
                    "root_id": "r1",
                    "own_action": own,
                    "opponent_action": opponent,
                    "search_score": search,
                    "values": {"model": value},
                    "outcome": outcome,
                }
            )
    summary = summarize_records(records, ["model"])
    assert summary["roots"] == 1
    assert summary["branch_rollouts"] == 4
    assert summary["models"]["model"]["action_aggregated"]["pairwise_comparisons"] == 0
    conditioned = summary["models"]["model"]["opponent_response_conditioned"]
    assert conditioned["pairwise_comparisons"] == 2
    assert conditioned["pairwise_accuracy"] == 0.5


def test_clustered_comparison_resamples_whole_matchups() -> None:
    records = []
    for root_id, team, outcomes in (
        ("r1", "team-a", (1.0, -1.0)),
        ("r2", "team-a", (1.0, -1.0)),
        ("r3", "team-b", (-1.0, 1.0)),
        ("r4", "team-b", (-1.0, 1.0)),
    ):
        for action, outcome, search, value in zip(
            ("a", "b"), outcomes, (0.0, 1.0), (1.0, 0.0), strict=True
        ):
            records.append(
                {
                    "root_id": root_id,
                    "team": team,
                    "opponent_team": f"opp-{team}",
                    "own_action": action,
                    "opponent_action": "x",
                    "search_score": search,
                    "values": {"model": value},
                    "outcome": outcome,
                }
            )
    comparison = paired_clustered_comparison(
        records,
        ["model"],
        "model",
        bootstrap_samples=100,
        seed=3,
    )
    assert comparison["matchup_clusters"] == 2
    assert comparison["roots"] == 4
    assert len(comparison["improvements"]["top_action_accuracy_95"]) == 2
