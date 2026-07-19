"""Unit tests for `vgc.team_preview` -- lightweight stand-ins for poke-env's `Pokemon`/
`AbstractBattle` (same style as tests/test_sets.py's `_FakePokemon`) so these don't need
a real battle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vgc.damage import PokemonState
from vgc.models import PolicyConfig
from vgc.team_preview import _preview_speed, build_team_order


@dataclass
class _FakeMon:
    species: str
    item: str | None = None
    ability: str | None = None
    boosts: dict = field(default_factory=dict)
    status: Any = None
    current_hp_fraction: float = 1.0
    current_hp: int | None = None
    evs: list[int] | None = None
    nature: str | None = None
    moves: dict = field(default_factory=dict)
    _selected_in_teampreview: bool = False


class _FakeBattle:
    def __init__(self, our_team, opp_team):
        self.team = {f"p1: mon{i}": mon for i, mon in enumerate(our_team)}
        self.teampreview_opponent_team = opp_team


def _mon(species, moves, **kwargs):
    return _FakeMon(species=species, moves=dict.fromkeys(moves), **kwargs)


def _our_team():
    return [
        _mon("charizard", ["heatwave", "protect"], item="charizarditey", ability="blaze"),
        _mon("farigiraf", ["psychic", "trickroom", "helpinghand"], ability="armortail"),
        _mon("venusaur", ["leafstorm", "sleeppowder", "protect"], ability="chlorophyll"),
        _mon("garchomp", ["earthquake", "dragonclaw", "protect"], item="lifeorb", ability="roughskin"),
        _mon("incineroar", ["flareblitz", "fakeout", "partingshot"], ability="intimidate"),
        _mon("sylveon", ["hypervoice", "protect"], ability="pixilate"),
    ]


def _opp_team():
    return [
        _mon("klefki", ["playrough", "protect"], ability="prankster"),
        _mon("garchomp", ["earthquake", "protect"], ability="roughskin"),
        _mon("incineroar", ["flareblitz", "fakeout"], ability="intimidate"),
        _mon("sylveon", ["hypervoice", "protect"], ability="pixilate"),
        _mon("venusaur", ["leafstorm", "sleeppowder"], ability="chlorophyll"),
        _mon("charizard", ["heatwave", "protect"], ability="blaze"),
    ]


def test_returns_a_valid_team_order_string() -> None:
    battle = _FakeBattle(_our_team(), _opp_team())
    order = build_team_order(battle, PolicyConfig())

    assert order.startswith("/team ")
    digits = order.removeprefix("/team ")
    assert len(digits) == 4
    picks = [int(d) for d in digits]
    assert all(1 <= pick <= 6 for pick in picks)
    assert len(set(picks)) == 4  # no repeats


def test_leads_are_a_subset_of_the_picked_four() -> None:
    battle = _FakeBattle(_our_team(), _opp_team())
    order = build_team_order(battle, PolicyConfig())
    digits = [int(d) for d in order.removeprefix("/team ")]
    leads, bench = digits[:2], digits[2:]
    assert set(leads).isdisjoint(bench)
    assert len(leads) == 2
    assert len(bench) == 2


def test_marks_picked_pokemon_as_selected_in_teampreview() -> None:
    our_team = _our_team()
    battle = _FakeBattle(our_team, _opp_team())
    order = build_team_order(battle, PolicyConfig())
    digits = [int(d) for d in order.removeprefix("/team ")]

    for i, mon in enumerate(our_team, start=1):
        assert mon._selected_in_teampreview == (i in digits)


def test_falls_back_gracefully_with_no_opponent_preview() -> None:
    battle = _FakeBattle(_our_team(), [])
    order = build_team_order(battle, PolicyConfig())
    assert order == "/team 1234"


def test_sun_mega_is_recognized_as_a_lead_mode() -> None:
    battle = _FakeBattle(_our_team(), _our_team())

    order = build_team_order(battle, PolicyConfig())

    leads = {int(digit) for digit in order.removeprefix("/team ")[:2]}
    assert 1 in leads  # Charizard-Y's Mega stats and Drought are no longer ignored.


def test_chlorophyll_speed_is_doubled_in_preview_sun() -> None:
    venusaur = PokemonState(species_id="venusaur", ability="chlorophyll")

    assert _preview_speed(venusaur, "sun") == 2 * _preview_speed(venusaur, None)


def test_preview_carries_engine_closer_lead_backline_and_mega_plan_into_battle() -> None:
    battle = _FakeBattle(_our_team(), _opp_team())
    order = build_team_order(battle, PolicyConfig())
    plan = battle._vgc_preview_plan

    picked = {to_id for to_id in plan.picked_species}
    assert plan.our_closer_species in picked
    assert plan.default_mega_species == "charizard"
    assert len(plan.lead_functions) >= 2
    assert len(plan.picked_species) == 4
    assert len(plan.lead_species) == 2
    assert plan.lead_covers_engine
    assert plan.balanced_structure["two_attackers"]
    assert plan.balanced_structure["closer"]
    assert plan.speed_modes
    assert order.startswith("/team ")
