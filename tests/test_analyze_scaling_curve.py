from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from offline.analyze_scaling_curve import (
    bootstrap_delta_interval,
    classify,
    fit_log_slope,
    load_rows,
    nearest_level,
    paired_delta_by_team,
    resolve_anchor_levels,
)

SEEDS = ("seedA", "seedB", "seedC")
ANCHORS = (25_000, 100_000)


def _write_records(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_load_rows_combines_multiple_seed_files(tmp_path: Path) -> None:
    first = tmp_path / "seed-a.jsonl"
    second = tmp_path / "seed-b.jsonl"
    _write_records(first, [{"seed_label": "seed-a"}])
    _write_records(second, [{"seed_label": "seed-b"}])

    assert load_rows([first, second]) == [
        {"seed_label": "seed-a"},
        {"seed_label": "seed-b"},
    ]


def _records_for(
    path: Path,
    *,
    opponent: str,
    games: int,
    wins: list[float],
    teams: list[str] | None = None,
) -> None:
    assert len(wins) == games
    sides = ["p1" if index % 2 == 0 else "p2" for index in range(games)]
    team_labels = teams or [f"team-{index % 4}" for index in range(games)]
    _write_records(
        path,
        [
            {
                "game_index": index,
                "team": team_labels[index],
                "opponent": opponent,
                "side": sides[index],
                "win": wins[index],
            }
            for index in range(games)
        ],
    )


def test_nearest_level_resolves_within_tolerance_only() -> None:
    levels = [10_000, 24_576, 49_152]
    assert nearest_level(levels, 25_000, 8_192) == 24_576
    assert nearest_level(levels, 90_000, 8_192) is None


def test_fit_log_slope_recovers_a_known_trend() -> None:
    # rate = 0.2 + 0.1 * log10(games): exactly +0.1 per decade.
    points = [(games, 0.2 + 0.1 * math.log10(games)) for games in (1_000, 10_000, 100_000)]
    slope, se = fit_log_slope(points)
    assert slope == pytest.approx(0.1)
    assert se == pytest.approx(0.0, abs=1e-9)


def test_bootstrap_delta_interval_separates_signal_from_null(tmp_path) -> None:
    # Strong: every paired game flips lose->win, so every team's diffs are all +1 and
    # any team resample reproduces the same mean -- the interval must exclude 0.
    # Null: early and late wins alternate out of phase, so every paired diff is
    # exactly -1/+1 alternating -- pooled point 0 and an interval that contains 0.
    early_lose: dict[int, tuple[str, str, float]] = {}
    early_alt: dict[int, tuple[str, str, float]] = {}
    strong: dict[int, tuple[str, str, float]] = {}
    null: dict[int, tuple[str, str, float]] = {}
    for index in range(40):
        team = f"team-{index % 8}"
        early_lose[index] = (team, "p1", 0.0)
        early_alt[index] = (team, "p1", float(index % 2))
        strong[index] = (team, "p1", 1.0)
        null[index] = (team, "p1", float((index + 1) % 2))

    point, low, _high = bootstrap_delta_interval(paired_delta_by_team(early_lose, strong))
    assert point == pytest.approx(1.0)
    assert low > 0.99

    point_null, low_null, high_null = bootstrap_delta_interval(
        paired_delta_by_team(early_alt, null)
    )
    assert abs(point_null) < 0.05
    assert low_null <= 0.0 <= high_null


def test_paired_delta_rejects_schedule_mismatch() -> None:
    early = {0: ("team-a", "p1", 0.0), 1: ("team-b", "p2", 0.0)}
    late = {0: ("team-a", "p1", 1.0), 1: ("team-ZZ", "p2", 1.0)}
    with pytest.raises(SystemExit, match="schedule mismatch"):
        paired_delta_by_team(early, late)


def _build_case_rows(
    tmp_path: Path,
    *,
    case: str,
) -> list[dict]:
    """Rows+record files engineered so exactly the requested case's rule fires."""

    rows: list[dict] = []
    for seed_index, label in enumerate(SEEDS):
        for split in ("train", "holdout"):
            if case == "A":
                holdout_vgc = {25_000: 0.30, 100_000: 0.36}
                train_overall = {25_000: 0.60, 100_000: 0.64}
            elif case == "B":
                holdout_vgc = {25_000: 0.30, 100_000: 0.30}
                train_overall = {25_000: 0.60, 100_000: 0.61}
            else:  # C
                holdout_vgc = {25_000: 0.30, 100_000: 0.31}
                train_overall = {25_000: 0.60, 100_000: 0.70}
            for games in ANCHORS:
                record_name = f"{label}-{split}-{games}.jsonl"
                records_path = tmp_path / record_name
                if split == "holdout" and case == "A":
                    # All-lose at 25k, all-win at 100k vs `vgc`: paired delta +1.
                    _records_for(
                        records_path,
                        opponent="vgc",
                        games=20,
                        wins=[0.0] * 20 if games == 25_000 else [1.0] * 20,
                    )
                elif split == "holdout" and case == "B":
                    # Identical outcomes: delta exactly 0, not significant.
                    _records_for(
                        records_path,
                        opponent="vgc",
                        games=20,
                        wins=[float(index % 3 == 0) for index in range(20)],
                    )
                elif split == "holdout":
                    _records_for(
                        records_path,
                        opponent="vgc",
                        games=20,
                        wins=[0.0] * 20 if games == 25_000 else [0.0] * 19 + [1.0],
                    )
                by_opponent = {
                    "random": 0.95,
                    "maxpower": 0.85,
                    "heuristic": 0.70,
                    "vgc_myopic": 0.50,
                    "vgc_shallow": 0.39,
                    "vgc": holdout_vgc[games] if split == "holdout" else 0.35,
                }
                overall = holdout_vgc[games] if split == "holdout" else train_overall[games]
                rows.append(
                    {
                        "schema": "vgc-scaling-milestone-eval-v1",
                        "seed_label": label,
                        "checkpoint": f"/tmp/{label}/games_{games:06d}.pt",
                        "games_seen": games,
                        "split": split,
                        "game_records": str(records_path),
                        "result": {"by_opponent": by_opponent, "win_rate": overall},
                    }
                )
    return rows


@pytest.mark.parametrize(
    "case,expected_verdict",
    [
        ("A", "CASE_A_KEEP_SCALING"),
        ("B", "CASE_B_PLATEAU"),
    ],
)
def test_classify_matches_pre_registered_cases(tmp_path, case: str, expected_verdict: str) -> None:
    rows = _build_case_rows(tmp_path, case=case)
    findings = classify(
        rows,
        from_games=25_000,
        to_games=100_000,
        required_seeds=2,
        case_a_points=0.03,
        case_b_tolerance=0.02,
        tolerance_games=8_192,
    )
    assert findings["verdict"] == expected_verdict


def test_classify_detects_generalization_wall(tmp_path) -> None:
    rows = _build_case_rows(tmp_path, case="C")
    findings = classify(
        rows,
        from_games=25_000,
        to_games=100_000,
        required_seeds=2,
        case_a_points=0.03,
        case_b_tolerance=0.02,
        tolerance_games=8_192,
    )
    assert findings["verdict"] == "CASE_C_GENERALIZATION_WALL"
    assert findings["case_c"]["train_gain"] >= 0.05
    assert findings["case_c"]["holdout_gain"] <= 0.01


def test_resolve_anchor_levels_requires_both_ends(tmp_path) -> None:
    rows = [{"games_seen": level} for level in (8_192, 16_384, 32_768)]
    start, end = resolve_anchor_levels(rows, 12_000, 100_000, tolerance=100_000)
    assert (start, end) == (8_192, 32_768)
    with pytest.raises(SystemExit, match="cannot resolve anchors"):
        resolve_anchor_levels(rows, 25_000, 100_000, tolerance=8_192)
