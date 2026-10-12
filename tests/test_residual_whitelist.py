"""The repeat-Protect residual-KO exception is conservative by construction.

Anything residual-relevant that `vgc.evaluator.residual_hp` does not model exactly must give
"no KO" (a false KO keeps a 1/3 Protect gamble). Each test is a reviewer repro.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from poke_env.battle.effect import Effect

import vgc.evaluator as ev
from tests.test_repeat_protect_rule import _ctx
from vgc.sets import PokemonState


def _foe(
    species="vaporeon",
    *,
    status="psn",
    fraction=0.05,
    item="choicescarf",
    ability="waterabsorb",
    effects=None,
    moves=None,
    weather=None,
):
    ctx = _ctx(counter=1)
    ctx.weather = weather
    state = PokemonState(species, status=status, item=item, ability=ability)
    mon = SimpleNamespace(
        fainted=False,
        moves=moves or {},
        ability=ability,
        item=item,
        effects=effects or {},
        status_counter=0,
        current_hp_fraction=fraction,
        species=species,
    )
    ctx.opp_states[0], ctx.opp_pokemon[0] = state, mon
    return state, mon, ctx


def _kos(**kw) -> bool:
    state, mon, ctx = _foe(**kw)
    return ev._residual_kos(state, mon, ctx)


def test_sure_ko_still_predicted_with_known_item_and_ability() -> None:
    assert _kos(status="psn", fraction=0.05)
    assert not _kos(status="psn", fraction=0.25)


def test_hydration_in_rain_cures_poison_before_it_ticks() -> None:
    assert _kos(status="psn", fraction=0.05, weather="rain")  # Water Absorb: no cure
    assert not _kos(status="psn", fraction=0.05, weather="rain", ability="hydration")
    state, mon, ctx = _foe(status="psn", fraction=0.05, weather="rain")
    mon.ability = None  # ability still hidden: Vaporeon could have Hydration
    state.ability = None
    assert not ev._residual_kos(state, mon, ctx)


@pytest.mark.parametrize("effect", ["AQUA_RING", "INGRAIN", "LEECH_SEED", "WRAP", "BIND"])
def test_unmodelled_residual_effects_block_the_ko(effect: str) -> None:
    # Aqua Ring / Ingrain heal; a bind may expire before its tick; a Leech Seed needs a
    # living recipient.
    assert not _kos(status="psn", fraction=0.05, effects={getattr(Effect, effect): 1})


def test_revealed_wish_blocks_the_ko() -> None:
    assert not _kos(status="psn", fraction=0.05, moves={"wish": object()})


def test_burned_vaporeon_at_six_percent_survives_at_one_hp() -> None:
    # 13/205 also displays 6% and a burn tick of 205 // 16 = 12 leaves 1 HP.
    assert not _kos(status="brn", fraction=0.06)
    state, mon, _ = _foe(status="brn", fraction=0.06)
    assert (13, 205) in ev._foe_hp_hypotheses(mon, state)


def test_hidden_item_could_be_leftovers() -> None:
    assert not _kos(status="psn", fraction=0.10, item=None)
    assert _kos(status="psn", fraction=0.01, item=None)


def test_hidden_berry_could_heal_a_foe_above_half() -> None:
    assert not _kos(status="tox", fraction=0.60, item="sitrusberry")
