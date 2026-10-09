"""Fast-search weather handling: Mega weather abilities beyond Drought/Drizzle, and the
weather-dependent accuracy of Thunder/Hurricane/Blizzard. Hand-built contexts (reuses the
helpers in tests/test_search.py)."""

from __future__ import annotations

import pytest

from test_search import (
    _NO_OPP,
    _build_ctx,
    _fake_order,
    _fake_single,
    _klefki,
    _mon,
)
from vgc.damage import PokemonState
from vgc.models import PolicyConfig
from vgc.search import resolve_exchange


def _ctx(attacker: PokemonState, defender: PokemonState, weather: str | None = None):
    return _build_ctx(
        our_states=[attacker, None],
        opp_states=[defender, None],
        our_pokemon=[_mon(), None],
        opp_pokemon=[_mon(), None],
        weather=weather,
    )


def _loss(move_id: str, weather: str | None, config: PolicyConfig, mega: bool = False) -> float:
    attacker = PokemonState("garchomp", sp_spread={"spa": 32}, nature="modest")
    ctx = _ctx(attacker, _klefki(), weather)
    order = _fake_order(_fake_single(move_id, move_target=1, mega=mega), None)
    return resolve_exchange(order, _NO_OPP, ctx, config).opp_hp_lost_pct


# --- weather-dependent accuracy ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("move_id", "weather", "expected_accuracy"),
    [
        ("hurricane", None, 0.7),
        ("hurricane", "rain", 1.0),
        ("hurricane", "sun", 0.5),
        ("thunder", "rain", 1.0),
        ("thunder", "sun", 0.5),
        ("blizzard", "snow", 1.0),
        ("blizzard", None, 0.7),
    ],
)
def test_weather_dependent_accuracy_scales_expected_damage(
    move_id: str, weather: str | None, expected_accuracy: float
) -> None:
    on = _loss(move_id, weather, PolicyConfig())
    off = _loss(move_id, weather, PolicyConfig(weather_accuracy_modifiers=False))
    assert off > 0.0
    assert on == pytest.approx(off * expected_accuracy)


def test_other_moves_ignore_weather_accuracy() -> None:
    assert _loss("dragonpulse", "rain", PolicyConfig()) == _loss(
        "dragonpulse", "rain", PolicyConfig(weather_accuracy_modifiers=False)
    )


# --- Mega weather abilities in the fast search ------------------------------------------------


@pytest.mark.parametrize(
    ("base", "stone", "weather"),
    [
        ("tyranitar", "tyranitarite", "sand"),
        ("abomasnow", "abomasite", "snow"),
        ("froslass", "froslassite", "snow"),
        ("charizard", "charizarditey", "sun"),
    ],
)
def test_our_mega_sets_its_weather_for_the_exchange(base: str, stone: str, weather: str) -> None:
    attacker = PokemonState(base, item=stone)
    ctx = _ctx(attacker, _klefki())
    order = _fake_order(_fake_single("bodyslam", move_target=1, mega=True), None)
    assert resolve_exchange(order, _NO_OPP, ctx, PolicyConfig()).weather == weather


def test_legacy_table_misses_sand_and_snow_megas() -> None:
    legacy = PolicyConfig(weather_abilities_complete=False)
    for base, stone in (("tyranitar", "tyranitarite"), ("abomasnow", "abomasite")):
        ctx = _ctx(PokemonState(base, item=stone), _klefki())
        order = _fake_order(_fake_single("bodyslam", move_target=1, mega=True), None)
        assert resolve_exchange(order, _NO_OPP, ctx, legacy).weather is None
    sun = _ctx(PokemonState("charizard", item="charizarditey"), _klefki())
    order = _fake_order(_fake_single("bodyslam", move_target=1, mega=True), None)
    assert resolve_exchange(order, _NO_OPP, sun, legacy).weather == "sun"
