"""Critical behaviour of `PolicyConfig.search_model_field_setters` (vgc.field_setters).

Hand-built boards (no server): the evaluator's setter score, the "already up" and
"replacing our own weather" rules, flag-off identity, and the fast search's exchange applying
the new weather to the actions after it.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
from poke_env.battle.move import Move

from tests.mechanics_state_helpers import make_battle_state, make_pokemon_state
from tests.test_search import _build_ctx, _fake_order, _fake_single, _mon
from vgc.damage import PokemonState
from vgc.evaluator import _Context, _ThreatInfo, _score_single, effective_speed
from vgc.mechanics_state import EffectSnapshot, MoveSnapshot
from vgc.models import PolicyConfig
from vgc.search import OppResponse, _OppSlotAction, resolve_exchange

TURN = 6
ON = PolicyConfig(search_model_field_setters=True, exact_search_field_fit_weight=1.0)
OFF = PolicyConfig(exact_search_field_fit_weight=1.0)


def _snap(species, moves, *, ours=True, active=True, ability="torrent"):
    mon = make_pokemon_state(species, ability_id=ability, active=active)
    return replace(
        mon,
        stats=tuple((stat, 100) for stat in ("hp", "atk", "def", "spa", "spd", "spe")),
        moves=tuple(MoveSnapshot(m, 10, 10, False, None, False) for m in moves),
        base_move_ids=tuple(moves),
        item_known=ours,
        ability_known=ours,
    )


def _board(*, weather=(), fields=()):
    state = make_battle_state(
        our_pokemon=[
            _snap("blastoise", ("waterspout", "protect")),
            _snap("tinkaton", ("gigaimpact", "protect"), ability="moldbreaker"),
        ],
        opponent_pokemon=[
            _snap("garchomp", ("earthquake", "protect"), ours=False, ability="roughskin"),
            _snap("incineroar", ("flareblitz", "protect"), ours=False, ability="intimidate"),
        ],
        turn=TURN,
    )
    return replace(state, weather=tuple(weather), fields=tuple(fields))


def _effect(effect_id, start):
    return EffectSnapshot(id=effect_id, turns=start, raw_value=start, counter_kind="start_turn")


def _ctx(mechanics_state, *, weather=None, terrain=None) -> _Context:
    blastoise = PokemonState("blastoise", ability="torrent", nature="modest", sp_spread={"spa": 32})
    tinkaton = PokemonState("tinkaton", nature="adamant", sp_spread={"atk": 32})
    garchomp = PokemonState("garchomp", nature="jolly", sp_spread={"atk": 32, "spe": 32})
    incineroar = PokemonState("incineroar", nature="careful", sp_spread={"hp": 32})
    ours = [blastoise, tinkaton]
    theirs = [garchomp, incineroar]
    return _Context(
        battle=SimpleNamespace(side_conditions=[], opponent_side_conditions=[]),
        trick_room=False,
        weather=weather,
        terrain=terrain,
        our_side_screens=frozenset(),
        opp_side_screens=frozenset(),
        our_pokemon=[
            SimpleNamespace(first_turn=False, fainted=False, species="blastoise", ability=None),
            SimpleNamespace(first_turn=False, fainted=False, species="tinkaton", ability=None),
        ],
        opp_pokemon=[
            SimpleNamespace(ability=None, fainted=False, species="garchomp"),
            SimpleNamespace(ability=None, fainted=False, species="incineroar"),
        ],
        our_states=ours,
        opp_states=theirs,
        our_speed=[effective_speed(s) for s in ours],
        opp_speed=[effective_speed(s) for s in theirs],
        threat_on_us=[_ThreatInfo(), _ThreatInfo()],
        opp_threat_score=[0.0, 0.0],
        opp_protect_prob=[0.0, 0.0],
        opp_switch_prob=[0.0, 0.0],
        priors={},
        gameplan=None,
        mechanics_state=mechanics_state,
    )


def _score(move_id, ctx, config):
    single = SimpleNamespace(order=Move(move_id, gen=9), mega=False, move_target=0)
    return _score_single(single, 0, ctx, config)


def test_rain_dance_beats_a_weak_attack_when_only_we_benefit_from_rain() -> None:
    ctx = _ctx(_board())
    rain = _score("raindance", ctx, ON)
    weak = _score("watergun", ctx, ON)
    assert rain["field_condition"] == "weather:rain"
    assert rain["score"] > weak["score"] > 0.0
    assert rain["score"] > 20.0


def test_flag_off_leaves_setter_at_zero_and_scores_identical() -> None:
    ctx = _ctx(_board())
    assert _score("raindance", ctx, OFF)["score"] == 0.0
    # An ordinary attack scores exactly the same with the flag on or off.
    assert _score("watergun", ctx, ON)["score"] == _score("watergun", ctx, OFF)["score"]


def test_setting_the_weather_that_is_already_up_is_not_credited() -> None:
    # Rain set last turn has 4 turns left (covers turns 5..9); Rain Dance would just fail.
    rain_up = _board(weather=(_effect("raindance", TURN - 1),))
    ctx = _ctx(rain_up, weather="rain")
    assert _score("raindance", ctx, ON)["score"] == 0.0


def test_replacing_our_own_helpful_weather_is_negative() -> None:
    rain_up = _board(weather=(_effect("raindance", TURN - 1),))
    ctx = _ctx(rain_up, weather="rain")
    assert _score("sunnyday", ctx, ON)["score"] < 0.0


def test_exchange_applies_new_weather_to_later_actions_and_forecast() -> None:
    # Blastoise (faster) opens with Rain Dance; the slower partner's Hydro Pump then lands
    # in the rain it just made.
    blastoise = PokemonState("blastoise", nature="modest", sp_spread={"spe": 32}, ability="torrent")
    partner = PokemonState("tinkaton", sp_spread={"spa": 32})
    garchomp = PokemonState("garchomp", nature="jolly", sp_spread={"atk": 32, "spe": 32})
    ctx = _build_ctx(
        our_states=[blastoise, partner],
        opp_states=[garchomp, None],
        our_pokemon=[
            _mon(moves={"raindance": None}, species="blastoise"),
            _mon(moves={"hydropump": None}, species="tinkaton"),
        ],
        opp_pokemon=[_mon(moves={"earthquake": None}, species="garchomp"), None],
    )
    quiet = OppResponse(slot0=_OppSlotAction(kind="none"), slot1=_OppSlotAction(kind="none"))
    order = _fake_order(_fake_single("raindance"), _fake_single("hydropump", move_target=1))

    off = resolve_exchange(order, quiet, ctx, PolicyConfig())
    on = resolve_exchange(order, quiet, ctx, PolicyConfig(search_model_field_setters=True))
    assert off.weather is None
    assert on.weather == "rain"
    assert ctx.weather is None  # the shared context is never mutated
    assert on.opp_hp_lost_pct > off.opp_hp_lost_pct  # Hydro Pump got the rain boost

    expiring = resolve_exchange(
        order,
        quiet,
        ctx,
        PolicyConfig(search_model_field_setters=True, search_condition_expiry=True),
    )
    assert expiring.weather_last == 5  # set on projected turn 1, base duration 5


def test_opponent_setter_is_a_response_only_when_known() -> None:
    from vgc.search import _opp_slot_candidates

    config = PolicyConfig(search_model_field_setters=True)
    garchomp = PokemonState("garchomp", sp_spread={"atk": 32})
    ctx = _build_ctx(
        our_states=[PokemonState("tinkaton"), None],
        opp_states=[garchomp, None],
        our_pokemon=[_mon(species="tinkaton"), None],
        opp_pokemon=[_mon(moves={"sandstorm": None, "earthquake": None}, species="garchomp"), None],
    )
    kinds = {(a.kind, a.move_id) for a in _opp_slot_candidates(0, ctx, config)}
    assert ("utility", "sandstorm") in kinds
    legacy = {(a.kind, a.move_id) for a in _opp_slot_candidates(0, ctx, PolicyConfig())}
    assert ("utility", "sandstorm") not in legacy
    unknown = _build_ctx(
        our_states=[PokemonState("tinkaton"), None],
        opp_states=[garchomp, None],
        our_pokemon=[_mon(species="tinkaton"), None],
        opp_pokemon=[_mon(moves={"earthquake": None}, species="garchomp"), None],
    )
    assert ("utility", "sandstorm") not in {
        (a.kind, a.move_id) for a in _opp_slot_candidates(0, unknown, config)
    }


@pytest.mark.parametrize("ability,expected", [("drought", "sun"), ("sandstream", "sand")])
def test_switch_in_setter_ability_changes_exchange_weather(ability, expected) -> None:
    from vgc.search import _pre_move_field_events

    switching = PokemonState("torkoal", ability=ability)
    pokemon = SimpleNamespace(species="torkoal")
    order = _fake_order(SimpleNamespace(order=Pokemon_stub(pokemon), mega=False, move_target=0), None)
    events = _pre_move_field_events(order, [switching, None], _no_response(), [None, None])
    assert [cond for _s, _r, cond in events] == [("weather", expected)]


def _no_response():
    return OppResponse(slot0=_OppSlotAction(kind="none"), slot1=_OppSlotAction(kind="none"))


def Pokemon_stub(_pokemon):  # noqa: N802 - a real poke-env Pokemon instance is required
    from poke_env.battle.pokemon import Pokemon

    return Pokemon(gen=9, species="torkoal")
