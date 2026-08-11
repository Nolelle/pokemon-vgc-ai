"""Filling in our own Stat Points/nature (`vgc.own_team`)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from vgc.own_team import apply_own_spreads, index_from_packed, spread_index

TEAM = (Path(__file__).resolve().parents[1] / "teams" / "meta1.packed.txt").read_text().strip()


def _mon(evs=None, nature=None, stats=None) -> SimpleNamespace:
    return SimpleNamespace(evs=evs, nature=nature, _evs=evs, _ivs=None, _nature=nature,
                           stats=stats or {"hp": 163, "spa": 177})


def test_fills_stat_points_and_nature_from_the_packed_team() -> None:
    battle = SimpleNamespace(team={"p1: Charizard": _mon(), "p1: Garchomp": _mon()})
    assert apply_own_spreads(battle, index_from_packed(TEAM)) == 2
    assert battle.team["p1: Charizard"]._evs == [10, 0, 0, 32, 0, 24]
    assert battle.team["p1: Charizard"]._nature == "modest"
    assert battle.team["p1: Garchomp"]._nature == "jolly"


def test_leaves_the_simulators_own_stats_untouched() -> None:
    # poke-env's apply_teambuilder_team would recompute these with the VANILLA gen-9 EV
    # formula; the champions mod is linear in Stat Points, so the request's numbers are
    # the correct ones and must survive.
    battle = SimpleNamespace(team={"p1: Charizard": _mon(stats={"hp": 163, "spa": 177})})
    apply_own_spreads(battle, index_from_packed(TEAM))
    assert battle.team["p1: Charizard"].stats == {"hp": 163, "spa": 177}


def test_is_a_no_op_when_open_team_sheets_already_supplied_the_spread() -> None:
    known = _mon(evs=[1, 2, 3, 4, 5, 6], nature="timid")
    battle = SimpleNamespace(team={"p1: Charizard": known})
    assert apply_own_spreads(battle, index_from_packed(TEAM)) == 0
    assert known._evs == [1, 2, 3, 4, 5, 6]


def test_defaults_to_the_battles_own_teambuilder_team() -> None:
    battle = SimpleNamespace(
        team={"p1: Charizard": _mon()},
        teambuilder_team=list(index_from_packed(TEAM).values()),
    )
    assert apply_own_spreads(battle) == 1


def test_never_raises_on_missing_or_malformed_input() -> None:
    assert apply_own_spreads(SimpleNamespace()) == 0
    assert apply_own_spreads(SimpleNamespace(team=None), {}) == 0
    assert apply_own_spreads(SimpleNamespace(team="not-a-dict"), index_from_packed(TEAM)) == 0
    assert apply_own_spreads(None) == 0
    # A Pokemon on the roster with no matching teambuilder entry is skipped, not fatal.
    battle = SimpleNamespace(team={"p1: Missingno": _mon()})
    assert apply_own_spreads(battle, index_from_packed(TEAM)) == 0


def test_spread_index_keys_on_the_name_the_battle_ident_carries() -> None:
    index = spread_index([SimpleNamespace(nickname="Chomp", species="Garchomp")])
    assert "Chomp" in index
    species_only = spread_index([SimpleNamespace(nickname=None, species="Garchomp")])
    assert "Garchomp" in species_only
