from __future__ import annotations

import json
import random
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


def _pool(
    spec: dict[str, tuple[float, int]],
    seed: int,
    games: int = 18,
    team_sd: float = 0.10,
) -> list[dict]:
    """Sample a realistic pool: `spec` maps archetype -> (true win rate, team count).

    Teams within an archetype must differ, and by roughly as much as they do in the real
    pool (`team_sd` 0.10, measured on the 58-team archetype pool). Fixtures where every
    team posts the SAME record have zero between-team variance, which silently makes the
    cluster-robust interval collapse to a point and the guardrail untestable -- so these
    are drawn from a seeded binomial with a team effect rather than written by hand.
    """

    rng = random.Random(seed)
    results = []
    for archetype, (rate, count) in spec.items():
        for index in range(count):
            team_rate = min(0.98, max(0.02, rng.gauss(rate, team_sd)))
            wins = sum(1 for _ in range(games) if rng.random() < team_rate)
            results.append(_result(archetype, wins, games, f"{archetype}-{index}"))
    return results


def test_gate_requires_overall_pass_and_no_clearly_losing_archetype() -> None:
    strong = _pool({"sun": (0.68, 8), "rain": (0.66, 8)}, seed=1)
    assert build_report(strong)["passed"] is True

    hidden_collapse = _pool(
        {"sun": (0.88, 9), "moon": (0.86, 9), "star": (0.87, 9), "rain": (0.28, 9)},
        seed=1,
    )
    report = build_report(hidden_collapse)
    assert report["overall_passed"] is True
    assert report["clearly_losing_archetypes"] == ["rain"]
    assert report["passed"] is False


def test_overall_pass_uses_the_cluster_robust_bound_not_the_naive_one() -> None:
    """Teams that violently disagree must not certify an overall win.

    The naive Wilson bound sees 360 games at 0.58 and clears 0.50 comfortably. Clustering
    by team sees two populations that disagree far more than sampling noise allows, and
    correctly cannot rule out that the pooled number is an accident of which teams
    happened to be in the pool.
    """

    split = _pool({"sun": (0.85, 10), "rain": (0.32, 10)}, seed=2)
    report = build_report(split)
    overall = report["overall"]

    assert overall["wilson"][0] > 0.5
    assert overall["clustered"][0] < 0.5
    assert overall["design_effect"] > 2.0
    assert report["overall_passed"] is False


def test_guardrail_holm_correction_absorbs_a_single_noisy_archetype() -> None:
    """One mildly-below-even archetype among six must not trip the guardrail.

    This is the exact failure the uncorrected gate had: six independent 95% subgroup
    checks flag something roughly one run in seven even when nothing is truly broken.
    `uncorrected_would_flag` records what the old rule would have done, so the test pins
    the correction's effect rather than just its outcome.
    """

    spec: dict[str, tuple[float, int]] = {
        name: (0.53, 9) for name in ("sun", "rain", "sand", "trick", "tail")
    }
    spec["soft"] = (0.36, 9)
    report = build_report(_pool(spec, seed=2))

    # p ~ 0.013: comfortably "significant" on its own, and still not enough once you
    # account for having taken six shots at finding something.
    assert report["guardrail"]["pvalues"]["soft"] < 0.05
    assert report["guardrail"]["uncorrected_would_flag"] == ["soft"]
    assert report["guardrail"]["clearly_losing"] == []
    assert report["clearly_losing_archetypes"] == []


def test_guardrail_still_catches_a_consistently_losing_archetype() -> None:
    spec: dict[str, tuple[float, int]] = {
        name: (0.60, 9) for name in ("sun", "rain", "sand", "trick", "tail")
    }
    spec["broken"] = (0.22, 9)
    report = build_report(_pool(spec, seed=3))

    assert report["guardrail"]["clearly_losing"] == ["broken"]
    assert report["passed"] is False


def test_power_reporting_states_what_the_pool_cannot_certify() -> None:
    report = build_report(_pool({"sun": (0.55, 12), "rain": (0.48, 12)}, seed=4))
    power = report["power"]

    # A pool whose teams genuinely differ has an irreducible floor that more games per
    # team cannot lower -- only more teams can.
    assert power["team_effect_sd"] > 0.0
    assert 0.0 < power["floor_detectable_effect"] < power["minimum_detectable_effect"]


def test_null_test_gives_both_arms_the_same_config() -> None:
    from offline.evaluate_own_spread_pool import build_arms

    null_a, null_b = build_arms(null_test=True).values()
    assert null_a == null_b

    candidate, incumbent = build_arms(null_test=False).values()
    assert candidate != incumbent
