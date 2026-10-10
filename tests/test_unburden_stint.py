"""Unburden only counts an item lost during the CURRENT stint: Showdown ends it on switch-out and
returning without the item does not bring it back."""

from __future__ import annotations

from types import SimpleNamespace

from vgc.battle_memory import BattleMemory
from vgc.evaluator import _our_pokemon_state
from vgc.sets import opponent_state


def _feed(memory: BattleMemory, *lines: str) -> None:
    memory.observe_protocol([line.split("|") for line in lines])


def _memory() -> BattleMemory:
    return BattleMemory(battle_tag="t", our_role="p1")


def test_memory_tracks_item_loss_per_stint_for_both_sides() -> None:
    memory = _memory()
    _feed(
        memory,
        "|switch|p2a: Hitmonlee|Hitmonlee, L50, M|100/100",
        "|switch|p1a: Sneasler|Sneasler, L50, F|100/100",
    )
    assert memory.item_lost_this_stint("p2", "hitmonlee") is False
    _feed(
        memory,
        "|-enditem|p2a: Hitmonlee|Sitrus Berry|[from] move: Knock Off|[of] p1a: Sneasler",
        "|-enditem|p1a: Sneasler|Psychic Seed",
    )
    assert memory.item_lost_this_stint("p2", "hitmonlee") is True
    assert memory.item_lost_this_stint("p1", "sneasler") is True
    # Hitmonlee switches out and later returns: a new stint, no Unburden.
    _feed(memory, "|switch|p2a: Venusaur|Venusaur, L50, M|100/100")
    assert memory.item_lost_this_stint("p2", "hitmonlee") is True  # not yet re-entered
    _feed(memory, "|switch|p2a: Hitmonlee|Hitmonlee, L50, M|29/100")
    assert memory.item_lost_this_stint("p2", "hitmonlee") is False
    assert memory.item_lost_this_stint("p1", "sneasler") is True  # untouched
    assert memory.item_lost_this_stint(None, "sneasler") is None


def _mon(item, ability="unburden"):
    return SimpleNamespace(
        species="sneasler", ability=ability, item=item, evs=None, nature=None, boosts={},
        status=None, current_hp=100, current_hp_fraction=1.0, fainted=False,
    )


def test_builders_require_the_loss_to_be_in_this_stint() -> None:
    assert _our_pokemon_state(_mon(""), True, True, True).item_lost is True
    assert _our_pokemon_state(_mon(""), True, True, False).item_lost is False  # earlier stint
    assert _our_pokemon_state(_mon(""), True, True, None).item_lost is True  # no memory: guess
    assert _our_pokemon_state(_mon("leftovers"), True, True, True).item_lost is False
    assert opponent_state(_mon(None), unburden=True, stint_lost=False).item_lost is False
    assert opponent_state(_mon(None), unburden=True, stint_lost=True).item_lost is True
    assert opponent_state(_mon("unknown_item"), unburden=True, stint_lost=True).item_lost is False
