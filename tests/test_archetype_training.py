"""Unit tests for archetype-pool mode (`--archetype-pool`) in `selfplay/train_ppo.py`.

Pure unit tests -- no local Showdown server, no real archetype pool on disk (a tiny
hand-built manifest/team-file fixture is used instead). See CLAUDE.md's "archetype
pool" work: `tools/build_archetype_pool.py` writes the real
`data/selfplay/archetype_pool/manifest.json` this loader targets in production.
"""

from __future__ import annotations

import json
import random

import pytest

torch = pytest.importorskip("torch")

from selfplay.train_ppo import (  # noqa: E402
    OpponentTeamChoice,
    PoolTeam,
    build_pool_worker_assignments,
    build_same_archetype_pool_worker_assignments,
    load_archetype_pool,
    load_diverse_opponent_teams,
    sample_learner_and_opponent,
    split_holdout_by_archetype,
    split_holdout_teams,
)


def _write_manifest(tmp_path, records: list[dict]) -> "object":
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(records))
    return manifest_path


def test_load_archetype_pool_round_trips_a_tiny_manifest(tmp_path) -> None:
    (tmp_path / "sun").mkdir()
    (tmp_path / "sun" / "team_00.packed.txt").write_text("sun-anchor-packed\n")
    (tmp_path / "sun" / "team_01.packed.txt").write_text("sun-variant-packed\n")
    (tmp_path / "rain.packed.txt").write_text("rain-anchor-packed\n")
    manifest_path = _write_manifest(
        tmp_path,
        [
            {
                "file": "sun/team_00.packed.txt",
                "archetype": "charizard_sun_offense",
                "source": "anchor",
                "species": ["charizard", "garchomp"],
            },
            {
                "file": "sun/team_01.packed.txt",
                "archetype": "charizard_sun_offense",
                "source": "variant",
                "species": ["charizard", "garchomp", "sylveon"],
            },
            {
                "file": "rain.packed.txt",
                "archetype": "rain_offense",
                "source": "anchor",
                "species": ["pelipper", "archaludon"],
            },
        ],
    )

    pool_teams = load_archetype_pool(manifest_path)

    assert [team.packed for team in pool_teams] == [
        "sun-anchor-packed",
        "sun-variant-packed",
        "rain-anchor-packed",
    ]
    assert [team.archetype for team in pool_teams] == [
        "charizard_sun_offense",
        "charizard_sun_offense",
        "rain_offense",
    ]
    assert [team.source for team in pool_teams] == ["anchor", "variant", "anchor"]
    # `.stem` only strips the LAST suffix (matches `load_diverse_opponent_teams`'s own
    # `path.stem` convention, e.g. "dev.packed.txt" -> "dev.packed").
    assert [team.label for team in pool_teams] == [
        "team_00.packed",
        "team_01.packed",
        "rain.packed",
    ]


def test_load_archetype_pool_rejects_missing_manifest(tmp_path) -> None:
    with pytest.raises(SystemExit):
        load_archetype_pool(tmp_path / "does_not_exist.json")


def test_load_archetype_pool_rejects_missing_team_file(tmp_path) -> None:
    manifest_path = _write_manifest(
        tmp_path,
        [{"file": "missing.packed.txt", "archetype": "sand_offense", "source": "anchor"}],
    )

    with pytest.raises(SystemExit):
        load_archetype_pool(manifest_path)


def test_load_archetype_pool_rejects_empty_team_file(tmp_path) -> None:
    (tmp_path / "empty.packed.txt").write_text("   \n")
    manifest_path = _write_manifest(
        tmp_path,
        [{"file": "empty.packed.txt", "archetype": "sand_offense", "source": "anchor"}],
    )

    with pytest.raises(SystemExit):
        load_archetype_pool(manifest_path)


def _pool_teams(archetype_sizes: dict[str, int]) -> list[PoolTeam]:
    teams: list[PoolTeam] = []
    for archetype, count in archetype_sizes.items():
        for index in range(count):
            teams.append(
                PoolTeam(
                    label=f"team_{index:02d}",
                    packed=f"{archetype}-packed-{index}",
                    archetype=archetype,
                    source="anchor" if index == 0 else "variant",
                )
            )
    return teams


def test_split_holdout_by_archetype_respects_per_archetype_count_and_is_deterministic() -> None:
    teams = _pool_teams({"sun": 6, "rain": 5})

    train_teams, holdout_teams = split_holdout_by_archetype(
        teams, holdout_per_archetype=2, seed=11
    )
    repeated_train, repeated_holdout = split_holdout_by_archetype(
        teams, holdout_per_archetype=2, seed=11
    )

    assert train_teams == repeated_train
    assert holdout_teams == repeated_holdout

    from collections import Counter

    holdout_counts = Counter(team.archetype for team in holdout_teams)
    train_counts = Counter(team.archetype for team in train_teams)
    assert holdout_counts == {"sun": 2, "rain": 2}
    assert train_counts == {"sun": 4, "rain": 3}


def test_split_holdout_by_archetype_disjoint_and_union_equals_input() -> None:
    teams = _pool_teams({"sun": 8, "sand": 4})

    train_teams, holdout_teams = split_holdout_by_archetype(
        teams, holdout_per_archetype=3, seed=2
    )

    assert set(train_teams).isdisjoint(holdout_teams)
    assert sorted(train_teams + holdout_teams, key=lambda t: t.packed) == sorted(
        teams, key=lambda t: t.packed
    )


def test_split_holdout_by_archetype_clamps_at_least_one_train_per_archetype() -> None:
    # An archetype with 3 teams and holdout_per_archetype=5 must clamp to 2 held-out /
    # 1 train, never 0 train.
    teams = _pool_teams({"tiny": 3})

    train_teams, holdout_teams = split_holdout_by_archetype(
        teams, holdout_per_archetype=5, seed=0
    )

    assert len(train_teams) == 1
    assert len(holdout_teams) == 2


def test_split_holdout_by_archetype_single_team_archetype_yields_zero_holdout() -> None:
    teams = _pool_teams({"solo": 1, "sun": 6})

    train_teams, holdout_teams = split_holdout_by_archetype(
        teams, holdout_per_archetype=2, seed=0
    )

    solo_train = [t for t in train_teams if t.archetype == "solo"]
    solo_holdout = [t for t in holdout_teams if t.archetype == "solo"]
    assert len(solo_train) == 1
    assert len(solo_holdout) == 0


def test_split_holdout_by_archetype_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        split_holdout_by_archetype([], holdout_per_archetype=1, seed=0)
    with pytest.raises(ValueError):
        split_holdout_by_archetype(_pool_teams({"sun": 3}), holdout_per_archetype=-1, seed=0)


def test_sample_learner_and_opponent_mirror_fraction_one_always_mirrors() -> None:
    teams = _pool_teams({"sun": 4, "rain": 4})
    rng = random.Random(42)

    for _ in range(50):
        learner, opponent = sample_learner_and_opponent(teams, rng=rng, mirror_fraction=1.0)
        assert opponent.group == "mirror"
        assert opponent.packed == learner.packed


def test_sample_learner_and_opponent_mirror_fraction_zero_never_mirrors() -> None:
    teams = _pool_teams({"sun": 4, "rain": 4})
    rng = random.Random(42)

    for _ in range(50):
        _learner, opponent = sample_learner_and_opponent(teams, rng=rng, mirror_fraction=0.0)
        assert opponent.group == "diverse"


def test_sample_learner_and_opponent_is_deterministic_for_a_fixed_seed() -> None:
    teams = _pool_teams({"sun": 4, "rain": 4})

    first = [
        sample_learner_and_opponent(teams, rng=random.Random(7), mirror_fraction=0.5)
        for _ in range(1)
    ]
    # Re-run with a fresh RNG of the same seed -- must reproduce identical draws.
    rng_a = random.Random(7)
    rng_b = random.Random(7)
    draws_a = [sample_learner_and_opponent(teams, rng=rng_a, mirror_fraction=0.5) for _ in range(10)]
    draws_b = [sample_learner_and_opponent(teams, rng=rng_b, mirror_fraction=0.5) for _ in range(10)]

    assert draws_a == draws_b
    assert first  # sanity: the throwaway call above didn't raise


def test_sample_learner_and_opponent_only_draws_from_provided_list() -> None:
    teams = _pool_teams({"sun": 3})
    packed_values = {team.packed for team in teams}
    rng = random.Random(3)

    for _ in range(30):
        learner, opponent = sample_learner_and_opponent(teams, rng=rng, mirror_fraction=0.5)
        assert learner.packed in packed_values
        assert opponent.packed in packed_values


def test_sample_learner_and_opponent_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        sample_learner_and_opponent([], rng=random.Random(0), mirror_fraction=0.5)
    with pytest.raises(ValueError):
        sample_learner_and_opponent(
            _pool_teams({"sun": 2}), rng=random.Random(0), mirror_fraction=1.5
        )


def test_build_pool_worker_assignments_is_deterministic_and_sized() -> None:
    teams = _pool_teams({"sun": 4, "rain": 4})

    assignments = build_pool_worker_assignments(6, teams, mirror_fraction=0.3, seed=5)
    repeated = build_pool_worker_assignments(6, teams, mirror_fraction=0.3, seed=5)

    assert len(assignments) == 6
    assert assignments == repeated


def test_build_same_archetype_pool_worker_assignments_pairs_within_archetype() -> None:
    teams = _pool_teams({"sun": 4, "rain": 4, "sand": 4})

    assignments = build_same_archetype_pool_worker_assignments(
        6, teams, mirror_fraction=0.0, seed=1
    )

    for learner, opponent in assignments:
        # opponent.label carries "<archetype>:<team label>" -- confirm it matches the
        # learner's own archetype (the whole point of same-archetype pairing).
        opponent_archetype = opponent.label.split(":", 1)[0]
        assert opponent_archetype == learner.archetype


def test_default_mode_unaffected_split_holdout_teams_and_schedule_still_work() -> None:
    """Guard: importing/using the pre-existing flat-holdout API is completely
    unaffected by the archetype-pool additions above -- default (no --archetype-pool)
    behavior must stay byte-for-byte identical.
    """
    teams = [
        OpponentTeamChoice(label=f"team-{i}", packed=f"packed-{i}", group="diverse")
        for i in range(6)
    ]

    train_teams, holdout_teams = split_holdout_teams(teams, holdout_fraction=0.0, seed=1)
    assert train_teams == teams
    assert holdout_teams == []

    train_teams, holdout_teams = split_holdout_teams(teams, holdout_fraction=0.3, seed=5)
    assert set(train_teams).isdisjoint(holdout_teams)


def test_default_mode_unaffected_load_diverse_opponent_teams_still_works(tmp_path) -> None:
    dev = tmp_path / "dev.packed.txt"
    pool = tmp_path / "pool"
    pool.mkdir()
    dev.write_text("dev-team\n")
    (pool / "team_00.packed.txt").write_text("team-zero\n")

    choices = load_diverse_opponent_teams(pool, dev)

    assert [choice.label for choice in choices] == ["dev.packed", "team_00.packed"]
