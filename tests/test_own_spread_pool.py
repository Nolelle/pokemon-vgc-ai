from __future__ import annotations

import json
from pathlib import Path

import pytest

from offline.evaluate_own_spread_pool import aggregate, balanced_chunks, build_report, load_pool


def _result(archetype: str, wins: int, games: int, team_file: str) -> dict:
    return {
        "archetype": archetype,
        "team_file": team_file,
        "games": games,
        "p1_wins": wins,
        "p2_wins": games - wins,
        "draws": 0,
        "mean_turns": 6.0,
    }


def test_load_pool_resolves_team_files_from_manifest(tmp_path: Path) -> None:
    team_path = tmp_path / "sun" / "team.packed.txt"
    team_path.parent.mkdir()
    team_path.write_text("packed-team")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([{"file": "sun/team.packed.txt", "archetype": "sun"}]))

    assert load_pool(manifest) == [
        {
            "index": 0,
            "file": "sun/team.packed.txt",
            "archetype": "sun",
            "team": "packed-team",
        }
    ]


def test_load_pool_rejects_a_missing_team_file(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([{"file": "missing.txt", "archetype": "sun"}]))

    with pytest.raises(FileNotFoundError):
        load_pool(manifest)


def test_balanced_chunks_are_deterministic_and_cover_every_entry() -> None:
    entries = [{"index": index} for index in range(7)]

    chunks = balanced_chunks(entries, 3)

    assert [[entry["index"] for entry in chunk] for chunk in chunks] == [
        [0, 3, 6],
        [1, 4],
        [2, 5],
    ]


def test_aggregate_recomputes_the_interval_from_all_games() -> None:
    result = aggregate(
        [_result("sun", 7, 10, "a"), _result("sun", 5, 10, "b")],
        "sun",
    )

    assert result["teams"] == 2
    assert result["games"] == 20
    assert result["wins"] == 12
    assert result["win_rate"] == 0.6


def test_gate_requires_overall_pass_and_no_clearly_losing_archetype() -> None:
    strong = [_result("sun", 70, 100, "sun-a"), _result("rain", 70, 100, "rain-a")]
    assert build_report(strong)["passed"] is True

    hidden_collapse = [
        _result("sun", 95, 100, "sun-a"),
        _result("rain", 35, 100, "rain-a"),
    ]
    report = build_report(hidden_collapse)
    assert report["overall_passed"] is True
    assert report["clearly_losing_archetypes"] == ["rain"]
    assert report["passed"] is False
