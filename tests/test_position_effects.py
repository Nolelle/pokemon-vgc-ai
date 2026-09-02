"""Rung 3a: the exact-search value function must know which way an effect points.

`_side_position` scored volatiles and side conditions with `len(...)`, so being Leech
Seeded and setting up a Substitute were worth the same +`exact_search_effect_weight`.
These tests pin the sign, the neutral default, and the legacy control path.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from vgc.mechanics_state import _LAYERED_SIDE_CONDITIONS
from vgc.models import PolicyConfig
from vgc.position_effects import (
    HELPS_HOLDER,
    HURTS_HOLDER,
    NEUTRAL,
    _derived,
    effect_polarity,
    side_condition_polarity,
)
from vgc.rl.exact_search import _side_position


def _mon(effects=(), boosts=(), status=None, hp=100):
    return SimpleNamespace(
        fainted=False, current_hp=hp, max_hp=100, status=status, boosts=boosts, effects=effects
    )


def _side(mons, side_conditions=()):
    return SimpleNamespace(pokemon=mons, side_conditions=side_conditions)


# --- the sign itself --------------------------------------------------------------


@pytest.mark.parametrize(
    ("effect", "expected"),
    [
        ("leechseed", HURTS_HOLDER),
        ("taunt", HURTS_HOLDER),
        ("encore", HURTS_HOLDER),
        ("confusion", HURTS_HOLDER),
        ("attract", HURTS_HOLDER),
        ("partiallytrapped", HURTS_HOLDER),
        ("yawn", HURTS_HOLDER),
        ("saltcure", HURTS_HOLDER),
        ("substitute", HELPS_HOLDER),
        ("protect", HELPS_HOLDER),
        ("aquaring", HELPS_HOLDER),
        ("focusenergy", HELPS_HOLDER),
        ("ingrain", HELPS_HOLDER),
        ("helpinghand", HELPS_HOLDER),
    ],
)
def test_move_applied_volatiles_are_signed_by_who_the_move_targets(effect, expected):
    assert effect_polarity(effect) == expected


@pytest.mark.parametrize(
    ("effect", "expected"),
    [
        ("stealthrock", HURTS_HOLDER),
        ("spikes", HURTS_HOLDER),
        ("toxicspikes", HURTS_HOLDER),
        ("stickyweb", HURTS_HOLDER),
        ("tailwind", HELPS_HOLDER),
        ("reflect", HELPS_HOLDER),
        ("lightscreen", HELPS_HOLDER),
        ("auroraveil", HELPS_HOLDER),
        ("safeguard", HELPS_HOLDER),
    ],
)
def test_side_conditions_are_signed_by_which_side_receives_them(effect, expected):
    assert side_condition_polarity(effect) == expected


def test_derivation_reproduces_the_repos_own_hazard_list():
    """Independent cross-check on the derivation rule.

    `_LAYERED_SIDE_CONDITIONS` was written by hand for a different purpose (layer
    counting). Deriving "harmful to the side that has it" purely from move targets must
    land on exactly that set -- if it does not, the target rule is wrong.
    """

    _volatiles, sides = _derived()
    derived_hazards = {key for key, sign in sides.items() if sign == HURTS_HOLDER}
    assert derived_hazards == set(_LAYERED_SIDE_CONDITIONS)


def test_unsigned_effects_score_nothing_rather_than_a_bonus():
    # Ability-activation markers are the bulk of poke-env's Effect vocabulary and are not
    # persistent position advantages. The old len() paid full weight for each one.
    for marker in ("aftermath", "dancer", "ironbarbs", "quickdraw", "synchronize"):
        assert effect_polarity(marker) == NEUTRAL
    assert effect_polarity("a_volatile_that_does_not_exist") == NEUTRAL
    assert side_condition_polarity("a_side_condition_that_does_not_exist") == NEUTRAL


def test_bookkeeping_volatiles_are_not_mistaken_for_harm():
    # Sparkling Aria targets foes, so the target rule alone calls its mark harmful; the
    # mark only tells the engine whose burn to cure.
    assert effect_polarity("sparklingaria") == NEUTRAL
    # Two-sided self-applied volatiles score zero rather than a guess either way.
    assert effect_polarity("lockedmove") == NEUTRAL
    assert effect_polarity("uproar") == NEUTRAL


def test_effects_no_legal_move_applies_are_still_signed():
    # Perish Song targets `all`, so no target-derived sign exists; it is still a countdown
    # to a guaranteed faint. Trapping and Slow Start come from abilities, not moves.
    for effect in ("perish0", "perish1", "perish2", "perish3", "trapped", "slowstart"):
        assert effect_polarity(effect) == HURTS_HOLDER
    for effect in ("protosynthesisatk", "quarkdrivespe"):
        assert effect_polarity(effect) == HELPS_HOLDER


def test_illegal_champions_moves_do_not_contribute_signs():
    # Octolock/Telekinesis/Embargo/Nightmare/Tar Shot are isNonstandard "Past" in this
    # mod, so their volatiles can never appear and must not be derived into the map.
    volatiles, sides = _derived()
    for effect in ("octolock", "telekinesis", "embargo", "nightmare", "tarshot"):
        assert effect not in volatiles
    for effect in ("mist", "luckychant", "craftyshield", "matblock"):
        assert effect not in sides


# --- what the value function does with it -----------------------------------------


def test_being_crippled_and_setting_up_no_longer_score_the_same():
    config = PolicyConfig()
    clean = _side_position(_side([_mon(), _mon()]), config)
    seeded = _side_position(_side([_mon(effects=("leechseed",)), _mon()]), config)
    substitute = _side_position(_side([_mon(effects=("substitute",)), _mon()]), config)

    assert seeded < clean < substitute
    assert clean - seeded == pytest.approx(config.exact_search_effect_weight)
    assert substitute - clean == pytest.approx(config.exact_search_effect_weight)


def test_hazards_on_our_own_side_cost_us_and_tailwind_pays_us():
    config = PolicyConfig()
    clean = _side_position(_side([_mon(), _mon()]), config)
    rocks = _side_position(_side([_mon(), _mon()], side_conditions=("stealthrock",)), config)
    tailwind = _side_position(_side([_mon(), _mon()], side_conditions=("tailwind",)), config)

    assert rocks < clean < tailwind


def test_unsigned_effects_leave_the_position_score_untouched():
    config = PolicyConfig()
    clean = _side_position(_side([_mon(), _mon()]), config)
    noisy = _side_position(_side([_mon(effects=("dancer", "ironbarbs")), _mon()]), config)
    assert noisy == pytest.approx(clean)


def test_legacy_control_reproduces_the_sign_blind_behaviour_exactly():
    # False must be the pre-Rung-3a function byte-for-byte so a same-session A/B measures
    # only the sign change.
    legacy = replace(PolicyConfig(), exact_search_signed_effects=False)
    clean = _side_position(_side([_mon(), _mon()]), legacy)
    for effects in (("leechseed",), ("substitute",)):
        assert _side_position(_side([_mon(effects=effects), _mon()]), legacy) - clean == (
            pytest.approx(legacy.exact_search_effect_weight)
        )
    for condition in ("stealthrock", "tailwind"):
        assert _side_position(
            _side([_mon(), _mon()], side_conditions=(condition,)), legacy
        ) - clean == pytest.approx(legacy.exact_search_effect_weight)
