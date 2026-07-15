"""Unit tests for `vgc.sets` -- opponent state assembly, with a lightweight stand-in for
poke-env's `Pokemon` (see `vgc.sets.ObservedPokemon`'s Protocol) so these don't need a
real battle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from vgc.sets import load_usage_spreads, opponent_state
from vgc.stats import default_opponent_nature, default_opponent_spread


@dataclass
class _FakePokemon:
    species: str
    item: str | None = None
    ability: str | None = None
    boosts: dict[str, int] = field(default_factory=dict)
    status: Any = None
    current_hp_fraction: float = 1.0


def test_falls_back_to_default_spread_when_no_usage_data() -> None:
    fake = _FakePokemon(species="garchomp", item="focussash", ability="roughskin")
    state = opponent_state(fake, usage={})
    assert state.sp_spread == default_opponent_spread("garchomp")
    assert state.nature == default_opponent_nature("garchomp")
    assert state.item == "focussash"
    assert state.ability == "roughskin"


def test_unknown_item_sentinel_normalized_to_none() -> None:
    fake = _FakePokemon(species="klefki", item="unknown_item")
    state = opponent_state(fake, usage={})
    assert state.item is None


def test_usage_data_overrides_default_when_present() -> None:
    usage = {"garchomp": [{"sp": {"hp": 4, "spe": 32}, "nature": "jolly", "weight": 10}]}
    fake = _FakePokemon(species="garchomp")
    state = opponent_state(fake, usage=usage)
    assert state.sp_spread == {"hp": 4, "spe": 32}
    assert state.nature == "jolly"


def test_usage_data_picks_highest_weight_entry() -> None:
    usage = {
        "garchomp": [
            {"sp": {"atk": 32}, "nature": "adamant", "weight": 1},
            {"sp": {"spe": 32}, "nature": "jolly", "weight": 99},
        ]
    }
    fake = _FakePokemon(species="garchomp")
    state = opponent_state(fake, usage=usage)
    assert state.sp_spread == {"spe": 32}
    assert state.nature == "jolly"


def test_curated_nature_override_wins_over_usage_guess() -> None:
    usage = {"garchomp": [{"sp": {"atk": 32}, "nature": "adamant", "weight": 1}]}
    fake = _FakePokemon(species="garchomp")

    state = opponent_state(fake, usage=usage, nature_override="jolly")

    assert state.sp_spread == {"atk": 32}
    assert state.nature == "jolly"


def test_boosts_filtered_to_damage_relevant_stats() -> None:
    fake = _FakePokemon(species="garchomp", boosts={"atk": 2, "accuracy": 1, "evasion": -1, "def": 0})
    state = opponent_state(fake, usage={})
    assert state.boosts == {"atk": 2}  # accuracy/evasion/zero-value dropped


def test_status_normalized_from_status_like_object() -> None:
    fake = _FakePokemon(species="garchomp", status=SimpleNamespace(name="BRN"))
    state = opponent_state(fake, usage={})
    assert state.status == "brn"


def test_fainted_status_dropped() -> None:
    fake = _FakePokemon(species="garchomp", status=SimpleNamespace(name="FNT"))
    state = opponent_state(fake, usage={})
    assert state.status is None


def test_current_hp_rescaled_from_fraction_not_raw_percent_scale() -> None:
    # poke-env reports opponent HP on a 0-100-ish scale, not real Champions HP -- make
    # sure opponent_state uses current_hp_fraction against OUR computed max_hp, not
    # some raw number from the fake.
    fake = _FakePokemon(species="garchomp", current_hp_fraction=0.5)
    state = opponent_state(fake, usage={})
    assert state.current_hp == round(state.max_hp() * 0.5)


def test_load_usage_spreads_missing_file_returns_empty_dict() -> None:
    load_usage_spreads.cache_clear()
    assert load_usage_spreads("/nonexistent/path/spreads.json") == {}
