"""Remaining-duration bookkeeping that every exact branch inherits.

Before `vgc.condition_clock`, the public mirror rebuilt weather with the wrong remaining
duration 56/72 times and terrain 108/134 times on 16 direct games (2026-10-05): poke-env
restamps weather every upkeep, and switch-in setters were charged a turn Showdown never
charged them.
"""

from __future__ import annotations

from types import SimpleNamespace

from vgc.condition_clock import elapsed_ticks, observe_condition_line


def _feed(battle, *lines: str) -> None:
    for line in lines:
        observe_condition_line(battle, line.split("|"))


def test_weather_upkeep_announcements_do_not_restart_the_clock() -> None:
    battle = SimpleNamespace()
    _feed(battle, "|-weather|SunnyDay|[from] ability: Drought|[of] p1a: Torkoal")
    for _ in range(3):
        _feed(battle, "|-weather|SunnyDay|[upkeep]", "|upkeep")
    assert elapsed_ticks(battle, "weather", "sunnyday") == 3
    _feed(battle, "|-weather|none")
    assert elapsed_ticks(battle, "weather", "sunnyday") is None


def test_switch_in_setter_after_residual_is_not_charged_that_turn() -> None:
    battle = SimpleNamespace()
    # Move-set Trick Room: the same turn's residual charges it one tick.
    _feed(battle, "|-fieldstart|move: Trick Room|[of] p1a: Farigiraf", "|upkeep")
    # Psychic Surge on a replacement switch-in, after that residual.
    _feed(battle, "|-fieldstart|move: Psychic Terrain|[from] ability: Psychic Surge")
    assert elapsed_ticks(battle, "field", "trickroom") == 1
    assert elapsed_ticks(battle, "field", "psychicterrain") == 0
    # A new terrain replaces the old one.
    _feed(battle, "|upkeep", "|-fieldstart|move: Grassy Terrain|[from] ability: Grassy Surge")
    assert elapsed_ticks(battle, "field", "psychicterrain") is None
    assert elapsed_ticks(battle, "field", "grassyterrain") == 0


def test_side_conditions_are_tracked_per_side_and_extensions_net_out() -> None:
    battle = SimpleNamespace()
    _feed(battle, "|-sidestart|p2: Bob|move: Tailwind", "|upkeep")
    assert elapsed_ticks(battle, "side", "tailwind", "p2") == 1
    assert elapsed_ticks(battle, "side", "tailwind", "p1") is None
    _feed(battle, "|-weather|RainDance|[from] ability: Drizzle")
    for _ in range(6):  # outlived 5 turns: a Damp Rock extension (8 turns total)
        _feed(battle, "|upkeep")
    # Net of the 3-turn extension: base 5 - 3 = 2 turns left, as with 8 - 6.
    assert elapsed_ticks(battle, "weather", "raindance") == 3
