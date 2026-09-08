from __future__ import annotations

import pytest

from offline.evaluate_neural_search import (
    _strength_summary,
    paired_clustered_interval,
    parse_args,
)


def test_paired_interval_is_exact_zero_when_policies_match_every_pair() -> None:
    assert paired_clustered_interval({"a": [0.0, 0.0], "b": [0.0, 0.0]}) == (
        0.0,
        0.0,
    )


def test_strength_summary_uses_paired_team_differences() -> None:
    summary = _strength_summary(
        [
            {
                "team_id": "a",
                "opponent": "vgc",
                "shadow_win": 1.0,
                "hybrid_win": 1.0,
            },
            {
                "team_id": "a",
                "opponent": "vgc",
                "shadow_win": 1.0,
                "hybrid_win": 0.0,
            },
            {
                "team_id": "b",
                "opponent": "heuristic",
                "shadow_win": 0.0,
                "hybrid_win": 1.0,
            },
        ]
    )

    assert summary["shadow_win_rate"] == pytest.approx(2 / 3)
    assert summary["hybrid_win_rate"] == pytest.approx(2 / 3)
    assert summary["difference"] == pytest.approx(0.0)
    assert summary["teams"] == 2


def test_cli_only_allows_declared_hybrid_sequence() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--search-k", "6"])
    assert parse_args(["--team-part", "train"]).team_part == "train"
