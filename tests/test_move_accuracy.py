"""Move accuracy (hit chance) in the quick scorer and the fast search, and first-turn-only
moves (Fake Out / First Impression). Hand-built contexts, no server."""

from __future__ import annotations

import pytest

from test_evaluator import _attack_ctx, _klefki, _single_for
from test_search import (
    _NO_OPP,
    _build_ctx,
    _fake_order,
    _fake_single,
    _mon,
)
from vgc.accuracy import hit_probability
from vgc.damage import PokemonState
from vgc.data import load_moves
from vgc.evaluator import _score_attack_order
from vgc.models import PolicyConfig
from vgc.priority_rules import usable_move_ids
from vgc.search import _opp_slot_candidates, resolve_exchange

MOVES = load_moves()
OFF = PolicyConfig(model_move_accuracy=False, weather_accuracy_modifiers=False)


def _mon_state(species="garchomp", **kwargs) -> PokemonState:
    return PokemonState(species, **kwargs)


# --- hit_probability --------------------------------------------------------------------------


def test_hit_probability_base_and_sure_hit_moves() -> None:
    a, d = _mon_state(), _klefki()
    assert hit_probability(MOVES["focusblast"], a, d) == pytest.approx(0.7)
    assert hit_probability(MOVES["headsmash"], a, d) == pytest.approx(0.8)
    assert hit_probability(MOVES["earthquake"], a, d) == 1.0
    assert hit_probability(MOVES["aerialace"], a, d) == 1.0  # accuracy: true


def test_hurricane_and_thunder_depend_on_weather() -> None:
    a, d = _mon_state(), _klefki()
    assert hit_probability(MOVES["hurricane"], a, d, "rain") == 1.0
    assert hit_probability(MOVES["hurricane"], a, d, None) == pytest.approx(0.7)
    assert hit_probability(MOVES["hurricane"], a, d, "sun") == pytest.approx(0.5)
    assert hit_probability(MOVES["thunder"], a, d, "sun") == pytest.approx(0.5)
    assert hit_probability(MOVES["blizzard"], a, d, "snow") == 1.0


def test_evasion_and_accuracy_stages() -> None:
    a = _mon_state()
    evasive = _klefki(boosts={"evasion": 1})
    # Showdown: trunc(100 * 3 / (3 + 1)) = 75.
    assert hit_probability(MOVES["earthquake"], a, evasive) == pytest.approx(0.75)
    assert hit_probability(MOVES["earthquake"], a, _klefki(boosts={"evasion": 2})) == pytest.approx(0.6)
    # +1 accuracy cancels +1 evasion; +1 accuracy on a 70% move: trunc(70 * 4 / 3) = 93.
    boosted = _mon_state(boosts={"accuracy": 1})
    assert hit_probability(MOVES["earthquake"], boosted, evasive) == 1.0
    assert hit_probability(MOVES["focusblast"], boosted, _klefki()) == pytest.approx(0.93)
    # Moves and abilities that ignore evasion.
    assert hit_probability(MOVES["chipaway"], a, evasive) == 1.0
    assert hit_probability(MOVES["earthquake"], _mon_state(ability="keeneye"), evasive) == 1.0
    assert hit_probability(MOVES["earthquake"], boosted, _klefki(ability="unaware")) == 1.0
    assert hit_probability(
        MOVES["earthquake"], _mon_state(ability="unaware"), _klefki(boosts={"evasion": 6})
    ) == 1.0


def test_accuracy_abilities_and_items() -> None:
    d = _klefki()
    assert hit_probability(MOVES["focusblast"], _mon_state(ability="compoundeyes"), d) == pytest.approx(0.91)
    assert hit_probability(MOVES["earthquake"], _mon_state(ability="hustle"), d) == pytest.approx(0.8)
    assert hit_probability(MOVES["flashcannon"], _mon_state(ability="hustle"), d) == 1.0  # special
    assert hit_probability(MOVES["focusblast"], _mon_state(ability="noguard"), d) == 1.0
    assert hit_probability(MOVES["focusblast"], _mon_state(), _klefki(ability="noguard")) == 1.0
    assert hit_probability(MOVES["rockslide"], _mon_state(item="widelens"), d) == pytest.approx(0.99)
    assert hit_probability(MOVES["earthquake"], _mon_state(), _klefki(item="brightpowder")) == pytest.approx(0.9)
    assert hit_probability(MOVES["earthquake"], _mon_state(), _klefki(ability="sandveil"), "sand") == pytest.approx(0.8)
    assert hit_probability(MOVES["earthquake"], _mon_state(), _klefki(ability="sandveil"), None) == 1.0


# --- evaluator --------------------------------------------------------------------------------


def _attack(move_id: str, defender: PokemonState, config: PolicyConfig, weather=None):
    attacker = PokemonState("raichu", sp_spread={"spa": 32}, nature="modest")
    ctx = _attack_ctx(ally_state=None, opp_state=defender, attacker_state=attacker, weather=weather)
    move, single = _single_for(move_id, move_target=1)
    return _score_attack_order(move, MOVES[move_id], single, 0, ctx, config)


def test_thunder_outranks_thunderbolt_only_when_accuracy_is_ignored() -> None:
    foe = _klefki()
    legacy = PolicyConfig(model_move_accuracy=False)
    assert _attack("thunder", foe, legacy)[0] > _attack("thunderbolt", foe, legacy)[0]
    # 110 BP at 70% is worth less than 90 BP that always lands.
    modelled = PolicyConfig()
    assert _attack("thunder", foe, modelled)[0] < _attack("thunderbolt", foe, modelled)[0]


def test_focus_blast_damage_score_is_scaled_by_its_hit_chance() -> None:
    foe = PokemonState("garchomp", sp_spread={"hp": 32, "spd": 2})
    attacker = PokemonState("lucario", sp_spread={"spa": 32}, nature="modest")
    ctx = _attack_ctx(ally_state=None, opp_state=foe, attacker_state=attacker)
    move, single = _single_for("focusblast", move_target=1)
    on = _score_attack_order(move, MOVES["focusblast"], single, 0, ctx, PolicyConfig())
    off = _score_attack_order(move, MOVES["focusblast"], single, 0, ctx, OFF)
    assert on[1] == pytest.approx(off[1] * 0.7)  # raw damage score
    assert on[2]["hit_probability_by_target"] == {0: pytest.approx(0.7)}


def test_guaranteed_ko_with_a_miss_chance_is_credited_in_expectation() -> None:
    foe = PokemonState("garchomp", current_hp=1)
    attacker = PokemonState("golem", sp_spread={"atk": 32}, nature="adamant")
    ctx = _attack_ctx(ally_state=None, opp_state=foe, attacker_state=attacker)
    move, single = _single_for("headsmash", move_target=1)
    on = _score_attack_order(move, MOVES["headsmash"], single, 0, ctx, PolicyConfig())
    off = _score_attack_order(move, MOVES["headsmash"], single, 0, ctx, OFF)
    assert 0 in on[2]["guaranteed_ko_slots"] and 0 in off[2]["guaranteed_ko_slots"]
    assert on[0] == pytest.approx(off[0] * 0.8, rel=0.02)  # ~80% of a guaranteed KO's value


def test_spread_move_rolls_accuracy_per_target() -> None:
    attacker = PokemonState("tyranitar", sp_spread={"atk": 32}, nature="adamant")
    evasive = _klefki(boosts={"evasion": 1})
    ctx = _attack_ctx(ally_state=None, opp_state=_klefki(), attacker_state=attacker)
    ctx = ctx.__class__(**{**ctx.__dict__, "opp_states": [_klefki(), evasive],
                           "opp_pokemon": [ctx.opp_pokemon[0], ctx.opp_pokemon[0]],
                           "opp_speed": [100.0, 100.0]})
    move, single = _single_for("rockslide")
    info = _score_attack_order(move, MOVES["rockslide"], single, 0, ctx, PolicyConfig())[2]
    assert info["hit_probability_by_target"][0] == pytest.approx(0.9)
    assert info["hit_probability_by_target"][1] == pytest.approx(0.67)  # trunc(90 * 3 / 4) = 67


# --- search -----------------------------------------------------------------------------------


def _search_ctx(our, opp, weather=None):
    return _build_ctx(
        our_states=our,
        opp_states=opp,
        our_pokemon=[_mon() if s is not None else None for s in our],
        opp_pokemon=[_mon() if s is not None else None for s in opp],
        weather=weather,
    )


def test_search_head_smash_ko_is_an_eighty_percent_ko() -> None:
    attacker = PokemonState("golem", sp_spread={"atk": 32}, nature="adamant")
    foe = PokemonState("garchomp", current_hp=1)
    ctx = _search_ctx([attacker, None], [foe, None])
    order = _fake_order(_fake_single("headsmash", move_target=1), None)
    on = resolve_exchange(order, _NO_OPP, ctx, PolicyConfig())
    off = resolve_exchange(order, _NO_OPP, ctx, OFF)
    assert off.opp_faints == 1.0
    assert on.opp_faints == pytest.approx(0.8)  # neither impossible nor certain


def test_search_hurricane_in_rain_vs_sun() -> None:
    attacker = PokemonState("garchomp", sp_spread={"spa": 32}, nature="modest")

    def loss(weather):
        ctx = _search_ctx([attacker, None], [_klefki(), None], weather)
        order = _fake_order(_fake_single("hurricane", move_target=1), None)
        return resolve_exchange(order, _NO_OPP, ctx, PolicyConfig()).opp_hp_lost_pct

    assert loss("rain") == pytest.approx(2 * loss("sun"))
    assert loss(None) == pytest.approx(1.4 * loss("sun"))


def test_search_evasion_stage_lowers_expected_damage() -> None:
    attacker = PokemonState("garchomp", sp_spread={"atk": 32}, nature="adamant")
    order = _fake_order(_fake_single("earthquake", move_target=1), None)

    def loss(foe, config):
        ctx = _search_ctx([attacker, None], [foe, None])
        return resolve_exchange(order, _NO_OPP, ctx, config).opp_hp_lost_pct

    plain = loss(_klefki(), PolicyConfig())
    evasive = loss(_klefki(boosts={"evasion": 1}), PolicyConfig())
    assert evasive == pytest.approx(plain * 0.75)
    assert loss(_klefki(boosts={"evasion": 1}), OFF) == pytest.approx(plain)


# --- first-turn-only moves --------------------------------------------------------------------


def test_usable_move_ids_drops_first_turn_moves_after_the_first_turn() -> None:
    moves = ["fakeout", "firstimpression", "closecombat"]
    spent = _mon()
    spent.first_turn = False
    fresh = _mon()
    fresh.first_turn = True
    assert usable_move_ids(moves, spent, True) == ["closecombat"]
    assert usable_move_ids(moves, fresh, True) == moves
    assert usable_move_ids(moves, spent, False) == moves
    assert usable_move_ids(moves, _mon(), True) == moves  # unknown -> keep


def _opp_candidates(first_turn: bool, config: PolicyConfig):
    opp = _mon(moves={"fakeout": None, "earthquake": None})
    opp.first_turn = first_turn
    ctx = _build_ctx(
        our_states=[PokemonState("garchomp"), None],
        opp_states=[PokemonState("incineroar", sp_spread={"atk": 32}, nature="adamant"), None],
        our_pokemon=[_mon(), None],
        opp_pokemon=[opp, None],
    )
    return {c.move_id for c in _opp_slot_candidates(0, ctx, config) if c.kind == "move"}


def test_search_only_offers_the_opponent_fake_out_on_its_first_turn() -> None:
    cfg = PolicyConfig(search_opp_moves_per_slot=4)
    assert "fakeout" in _opp_candidates(True, cfg)
    assert "fakeout" not in _opp_candidates(False, cfg)
    legacy = PolicyConfig(search_opp_moves_per_slot=4, first_turn_moves_restricted=False)
    assert "fakeout" in _opp_candidates(False, legacy)
