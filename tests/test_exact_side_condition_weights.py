"""Tailwind / screen weights in the exact search's scorecard (2026-10-18 replay fit)."""

from dataclasses import replace
from types import SimpleNamespace

from vgc.mechanics_state import EffectSnapshot
from vgc.models import PolicyConfig
from vgc.rl.exact_search import _side_position


def _side(*conditions: str):
    return SimpleNamespace(
        pokemon=(), side_conditions=tuple(EffectSnapshot(id=c) for c in conditions)
    )


def test_default_prices_every_signed_condition_at_the_effect_weight():
    config = PolicyConfig()
    w = config.exact_search_effect_weight
    assert _side_position(_side("tailwind"), config) == w
    assert _side_position(_side("reflect", "lightscreen"), config) == 2 * w
    assert _side_position(_side("stealthrock"), config) == -w


def test_tailwind_and_screen_weights_override_only_their_own_conditions():
    config = replace(
        PolicyConfig(), exact_search_tailwind_weight=60.0, exact_search_screen_weight=35.0
    )
    assert _side_position(_side("tailwind"), config) == 60.0
    assert _side_position(_side("reflect", "auroraveil"), config) == 70.0
    assert _side_position(_side("stealthrock"), config) == -config.exact_search_effect_weight
