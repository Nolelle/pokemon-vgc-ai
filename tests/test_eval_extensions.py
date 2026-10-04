from __future__ import annotations

import pytest

from offline.evaluate_own_spread_pool import aggregate, build_report, build_units
from vgc.evaluation import futility_stop, merge_cluster_results, plan_rounds


def test_plan_rounds_keeps_every_round_seat_balanced() -> None:
    assert plan_rounds(8, None, 4) == [8]
    assert plan_rounds(12, 8, 4) == [8, 4]
    with pytest.raises(ValueError):
        plan_rounds(10, None, 4)
    with pytest.raises(ValueError):
        plan_rounds(8, 6, 4)


def test_futility_stop_only_fires_when_target_is_out_of_reach() -> None:
    losing = [(2, 10)] * 12  # 20% in every cluster: upper bound far below 53%
    assert futility_stop(losing, 0.03)
    winning = [(8, 10)] * 12
    assert not futility_stop(winning, 0.03)  # never a "success" stop
    assert not futility_stop(losing[:5], 0.03)  # too few clusters to trust


def test_merge_collapses_pairs_into_one_cluster_per_opponent_team() -> None:
    def row(team: str, ours: str, wins: int) -> dict:
        return {
            "team_file": team,
            "unit_id": f"{team}|{ours}",
            "games": 4,
            "p1_wins": wins,
            "p2_wins": 4 - wins,
            "draws": 0,
            "mean_turns": 5.0,
            "p1_seat_games": 2,
            "p1_seat_wins": min(wins, 2),
            "p2_seat_games": 2,
            "p2_seat_wins": max(wins - 2, 0),
        }

    pairs = [row("a", "x", 3), row("a", "y", 1), row("b", "x", 2)]
    clusters = merge_cluster_results(pairs, "team_file")

    assert [(c["team_file"], c["games"], c["p1_wins"]) for c in clusters] == [
        ("a", 8, 4),
        ("b", 4, 2),
    ]
    overall = aggregate([{**c, "archetype": "t"} for c in clusters], "overall")
    assert overall["teams"] == 2
    assert overall["by_seat"]["p1"]["games"] == 6


def test_min_gain_requires_the_point_estimate() -> None:
    rows = [
        {
            "team_file": f"t{i}",
            "archetype": "a",
            "games": 40,
            "p1_wins": 22,
            "p2_wins": 18,
            "draws": 0,
            "mean_turns": 6.0,
        }
        for i in range(30)
    ]
    assert build_report(rows)["overall_passed"]  # 55%, tight CI
    assert not build_report(rows, min_gain=0.08)["overall_passed"]


def test_build_units_crosses_opponents_with_our_teams() -> None:
    entries = [{"index": 0, "file": "o0", "archetype": "a", "team": "T0"}]
    ours = [{"index": 0, "name": "x", "team": "X"}, {"index": 1, "name": "y", "team": "Y"}]
    units = build_units(entries, ours)
    assert [u["unit_id"] for u in units] == ["o0|x", "o0|y"]
    assert build_units(entries, None) == entries
