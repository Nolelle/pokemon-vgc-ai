"""Unit tests for `vgc.selfplay_pool` -- pure assembly/sampling/packing logic, using
small hand-built `set_priors`/`spreads`-shaped dicts (no real corpus files, no
`validate-team` CLI -- that real-legality check lives in
`tools/build_selfplay_pool.py` and is exercised manually/at pool-build time, not here).
"""

from __future__ import annotations

from vgc.selfplay_pool import (
    GeneratedSet,
    build_priority_species,
    resolve_team_items,
    sample_teams,
    team_to_packed,
)

# --- build_priority_species: ranking, legality filtering, fallbacks ------------------


def _set_priors(entries: dict[str, dict]) -> dict:
    return {"species": entries}


def test_build_priority_species_ranks_by_appearances_and_respects_top_n() -> None:
    priors = _set_priors(
        {
            "garchomp": {
                "appearances": 100,
                "moves": {"earthquake": 90, "dragonclaw": 80, "protect": 70, "rockslide": 60},
                "items": {"lifeorb": 50},
                "abilities": {"roughskin": 100},
            },
            "klefki": {
                "appearances": 50,
                "moves": {"protect": 40, "spikes": 30, "lightscreen": 20, "foulplay": 10},
                "items": {"leftovers": 20},
                "abilities": {"prankster": 50},
            },
            "sylveon": {
                "appearances": 10,
                "moves": {"hypervoice": 10, "protect": 5},
                "items": {"choicescarf": 5},
                "abilities": {"pixilate": 10},
            },
        }
    )
    pool = build_priority_species(priors, {}, top_n=2)
    assert [gen_set.species_id for gen_set in pool] == ["garchomp", "klefki"]


def test_build_priority_species_picks_highest_frequency_legal_ability() -> None:
    # Garchomp's real abilities are Sand Veil (slot 0) and Rough Skin (hidden) -- a
    # bogus corpus-noise ability should be skipped in favor of the next legal one.
    priors = _set_priors(
        {
            "garchomp": {
                "appearances": 20,
                "moves": {"earthquake": 10, "dragonclaw": 9, "protect": 8, "rockslide": 7},
                "items": {},
                "abilities": {"totallynotarealability": 99, "roughskin": 10},
            }
        }
    )
    pool = build_priority_species(priors, {}, top_n=1)
    assert pool[0].ability == "roughskin"


def test_build_priority_species_ability_falls_back_to_species_default_when_no_valid_prior() -> None:
    priors = _set_priors(
        {
            "garchomp": {
                "appearances": 20,
                "moves": {"earthquake": 10, "dragonclaw": 9, "protect": 8, "rockslide": 7},
                "items": {},
                "abilities": {},
            }
        }
    )
    pool = build_priority_species(priors, {}, top_n=1)
    assert pool[0].ability in ("sandveil", "roughskin")


def test_build_priority_species_filters_illegal_and_unlearnable_moves() -> None:
    priors = _set_priors(
        {
            "garchomp": {
                "appearances": 20,
                "moves": {
                    "earthquake": 10,  # legal, learnable
                    "dragonclaw": 9,  # legal, learnable
                    "not_a_real_move_xyz": 100,  # bogus -- should never be picked
                    "moonblast": 1,  # not in Garchomp's learnset -- should be skipped
                },
                "items": {},
                "abilities": {"roughskin": 1},
            }
        }
    )
    pool = build_priority_species(priors, {}, top_n=1)
    assert "not_a_real_move_xyz" not in pool[0].moves
    assert "moonblast" not in pool[0].moves
    assert "earthquake" in pool[0].moves
    assert "dragonclaw" in pool[0].moves


def test_build_priority_species_pads_thin_moveset_from_learnset() -> None:
    priors = _set_priors(
        {
            "garchomp": {
                "appearances": 20,
                "moves": {"earthquake": 10},  # only 1 real prior move
                "items": {},
                "abilities": {"roughskin": 1},
            }
        }
    )
    pool = build_priority_species(priors, {}, top_n=1)
    assert len(pool[0].moves) == 4
    assert "earthquake" in pool[0].moves


def test_build_priority_species_skips_unrecognized_species_id() -> None:
    priors = _set_priors(
        {
            "not_a_real_species_xyz": {
                "appearances": 999,
                "moves": {"tackle": 1},
                "items": {},
                "abilities": {},
            }
        }
    )
    pool = build_priority_species(priors, {}, top_n=5)
    assert pool == []


def test_build_priority_species_uses_spreads_when_available() -> None:
    priors = _set_priors(
        {
            "garchomp": {
                "appearances": 20,
                "moves": {"earthquake": 10, "dragonclaw": 9, "protect": 8, "rockslide": 7},
                "items": {},
                "abilities": {"roughskin": 1},
            }
        }
    )
    spreads = {
        "garchomp": [
            {"sp": {"hp": 4, "atk": 32, "spe": 30}, "nature": "adamant", "weight": 0.9},
            {"sp": {"hp": 32, "atk": 2, "spe": 32}, "nature": "jolly", "weight": 0.1},
        ]
    }
    pool = build_priority_species(priors, spreads, top_n=1)
    assert pool[0].nature == "adamant"
    assert pool[0].sp["atk"] == 32


# --- resolve_team_items: Item Clause (max 1 of each item per team) ------------------


def _gen_set(species_id: str, item_candidates: tuple[str, ...]) -> GeneratedSet:
    return GeneratedSet(
        species_id=species_id,
        item_candidates=item_candidates,
        ability="someability",
        moves=("protect", "tackle"),
        nature="jolly",
        sp={"hp": 0, "atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0},
    )


def test_resolve_team_items_falls_back_to_next_candidate_on_collision() -> None:
    team = [
        _gen_set("garchomp", ("choicescarf", "lifeorb")),
        _gen_set("swampert", ("choicescarf", "leftovers")),
    ]
    resolved = resolve_team_items(team)
    assert resolved[0] == "choicescarf"
    assert resolved[1] == "leftovers"  # bumped off its top choice by the collision


def test_resolve_team_items_goes_itemless_when_every_candidate_taken() -> None:
    team = [
        _gen_set("a", ("choicescarf",)),
        _gen_set("b", ("choicescarf",)),
    ]
    resolved = resolve_team_items(team)
    assert resolved[0] == "choicescarf"
    assert resolved[1] is None


def test_resolve_team_items_no_collision_keeps_everyone_top_choice() -> None:
    team = [_gen_set("a", ("leftovers",)), _gen_set("b", ("choicescarf",))]
    assert resolve_team_items(team) == ["leftovers", "choicescarf"]


# --- sample_teams: distinct-within-team, deterministic per seed ---------------------


def _dummy_pool(n: int) -> list[GeneratedSet]:
    return [_gen_set(f"species{i}", (f"item{i}",)) for i in range(n)]


def test_sample_teams_each_team_has_no_duplicate_species() -> None:
    pool = _dummy_pool(10)
    teams = sample_teams(pool, n_teams=5, seed=0)
    assert len(teams) == 5
    for team in teams:
        assert len(team) == 6
        assert len({gen_set.species_id for gen_set in team}) == 6


def test_sample_teams_deterministic_for_same_seed() -> None:
    pool = _dummy_pool(10)
    teams_a = sample_teams(pool, n_teams=5, seed=42)
    teams_b = sample_teams(pool, n_teams=5, seed=42)
    assert [[g.species_id for g in team] for team in teams_a] == [
        [g.species_id for g in team] for team in teams_b
    ]


def test_sample_teams_returns_empty_when_pool_too_small() -> None:
    pool = _dummy_pool(3)  # fewer than TEAM_SIZE=6
    assert sample_teams(pool, n_teams=5, seed=0) == []


# --- team_to_packed: layout matches teams/*.packed.txt's exact shape ----------------


def test_team_to_packed_layout_matches_real_team_files() -> None:
    team = [
        GeneratedSet(
            species_id="garchomp",
            item_candidates=("lifeorb",),
            ability="roughskin",
            moves=("earthquake", "dragonclaw", "protect", "rockslide"),
            nature="jolly",
            sp={"hp": 2, "atk": 32, "def": 0, "spa": 0, "spd": 0, "spe": 32},
        )
    ]
    packed = team_to_packed(team)
    assert packed == (
        "garchomp||lifeorb|roughskin|earthquake,dragonclaw,protect,rockslide|jolly|2,32,,,,32||||50|"
    )


def test_team_to_packed_joins_multiple_slots_with_bracket() -> None:
    team = [
        GeneratedSet(
            species_id="garchomp",
            item_candidates=("lifeorb",),
            ability="roughskin",
            moves=("earthquake",),
            nature="jolly",
            sp={"hp": 0, "atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0},
        ),
        GeneratedSet(
            species_id="klefki",
            item_candidates=("leftovers",),
            ability="prankster",
            moves=("protect",),
            nature="bold",
            sp={"hp": 0, "atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0},
        ),
    ]
    packed = team_to_packed(team)
    assert packed.count("]") == 1
    species_slots = packed.split("]")
    assert species_slots[0].startswith("garchomp||lifeorb|roughskin|")
    assert species_slots[1].startswith("klefki||leftovers|prankster|")
