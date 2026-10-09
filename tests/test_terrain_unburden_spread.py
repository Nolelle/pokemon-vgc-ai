"""Psychic Terrain priority block, Unburden Speed, and spread-move target recount in the
quick scorer and the fast search. Hand-built contexts, no server."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from test_evaluator import _attack_ctx, _garchomp, _klefki, _single_for
from test_search import (
    _NO_OPP,
    _build_ctx,
    _fake_order,
    _fake_single,
    _mon,
)
from vgc.damage import FieldState, PokemonState
from vgc.data import load_moves
from vgc.evaluator import (
    _best_attacking_move,
    _our_pokemon_state,
    _score_attack_order,
    effective_speed,
)
from vgc.models import PolicyConfig
from vgc.priority_rules import effective_priority, psychic_terrain_blocks
from vgc.search import (
    OppResponse,
    _consume_terrain_seeds,
    _OppSlotAction,
    resolve_exchange,
)
from vgc.sets import item_was_lost, opponent_state

MOVES = load_moves()


# --- effective priority / the block rule ----------------------------------------------------


def test_effective_priority_counts_priority_abilities() -> None:
    assert effective_priority(MOVES["fakeout"], None, True) == 3
    assert effective_priority(MOVES["thunderwave"], "prankster", True) == 1
    assert effective_priority(MOVES["earthquake"], "prankster", True) == 0
    assert effective_priority(MOVES["bravebird"], "galewings", True) == 1
    assert effective_priority(MOVES["bravebird"], "galewings", False) == 0
    assert effective_priority(MOVES["drainpunch"], "triage", True) == 3


def test_psychic_terrain_blocks_only_priority_into_grounded_targets() -> None:
    grounded = _klefki()
    airborne = PokemonState("talonflame")
    garchomp = _garchomp()
    assert psychic_terrain_blocks(MOVES["fakeout"], garchomp, grounded, "psychic") is True
    assert psychic_terrain_blocks(MOVES["fakeout"], garchomp, grounded, None) is False
    assert psychic_terrain_blocks(MOVES["fakeout"], garchomp, grounded, "grassy") is False
    assert psychic_terrain_blocks(MOVES["fakeout"], garchomp, airborne, "psychic") is False
    assert psychic_terrain_blocks(MOVES["earthquake"], garchomp, grounded, "psychic") is False
    # Prankster status move counts; the same move without the ability does not.
    prankster = PokemonState("klefki", ability="prankster")
    assert psychic_terrain_blocks(MOVES["thunderwave"], prankster, garchomp, "psychic") is True
    assert psychic_terrain_blocks(MOVES["thunderwave"], garchomp, garchomp, "psychic") is False


# --- evaluator --------------------------------------------------------------------------------


def _fake_out_scores(foe: PokemonState, config: PolicyConfig, terrain: str | None):
    ctx = replace(
        _attack_ctx(
            ally_state=None,
            opp_state=foe,
            attacker_state=PokemonState("incineroar", sp_spread={"atk": 32}, nature="adamant"),
        ),
        terrain=terrain,
    )
    move, single = _single_for("fakeout", move_target=1)
    return _score_attack_order(move, MOVES["fakeout"], single, 0, ctx, config)


def test_evaluator_credits_no_fake_out_damage_into_grounded_psychic_terrain() -> None:
    on, off = PolicyConfig(), PolicyConfig(psychic_terrain_blocks_priority=False)
    blocked_score, blocked_raw, blocked_info = _fake_out_scores(_klefki(), on, "psychic")
    assert blocked_raw == 0.0
    assert blocked_info["flinch_targets"] == []
    legacy_score, legacy_raw, _ = _fake_out_scores(_klefki(), off, "psychic")
    assert legacy_raw > 0.0 and legacy_score > blocked_score
    # No terrain, or an airborne target: unchanged.
    assert _fake_out_scores(_klefki(), on, None)[1] == legacy_raw
    assert _fake_out_scores(PokemonState("talonflame"), on, "psychic")[1] > 0.0


def test_threat_estimate_ignores_priority_moves_psychic_terrain_blocks() -> None:
    field = FieldState(terrain="psychic", is_doubles=True)
    attacker = PokemonState("incineroar", sp_spread={"atk": 32}, nature="adamant")
    defender = _klefki()
    plain = _best_attacking_move(attacker, ["fakeout"], defender, field)
    blocked = _best_attacking_move(attacker, ["fakeout"], defender, field, True)
    assert plain[1] == "fakeout" and plain[0] > 0
    assert blocked == (0.0, None, 0)


# --- search ---------------------------------------------------------------------------------


def _search_ctx(our, opp, terrain=None):
    ctx = _build_ctx(
        our_states=our,
        opp_states=opp,
        our_pokemon=[_mon() if s is not None else None for s in our],
        opp_pokemon=[_mon() if s is not None else None for s in opp],
    )
    return replace(ctx, terrain=terrain)


def test_search_fake_out_does_nothing_into_grounded_psychic_terrain() -> None:
    attacker = PokemonState("incineroar", sp_spread={"atk": 32}, nature="adamant")
    order = _fake_order(_fake_single("fakeout", move_target=1), None)

    def lost(foe, config, terrain="psychic"):
        ctx = _search_ctx([attacker, None], [foe, None], terrain)
        return resolve_exchange(order, _NO_OPP, ctx, config).opp_hp_lost_pct

    assert lost(_klefki(), PolicyConfig()) == 0.0
    assert lost(_klefki(), PolicyConfig(psychic_terrain_blocks_priority=False)) > 0.0
    assert lost(_klefki(), PolicyConfig(), terrain=None) > 0.0
    assert lost(PokemonState("talonflame"), PolicyConfig()) > 0.0


def test_search_blocks_the_opponents_priority_move_into_our_grounded_mon() -> None:
    ours = _klefki()
    foe = PokemonState("incineroar", sp_spread={"atk": 32}, nature="adamant")
    response = OppResponse(
        slot0=_OppSlotAction(kind="move", move_id="fakeout", target_our_slot=0),
        slot1=_OppSlotAction(kind="none"),
    )
    order = _fake_order(_fake_single("protect"), None)

    def taken(config):
        ctx = _search_ctx([ours, None], [foe, None], "psychic")
        return resolve_exchange(_fake_order(None, None), response, ctx, config).our_hp_lost_pct

    assert order is not None
    assert taken(PolicyConfig()) == 0.0
    assert taken(PolicyConfig(psychic_terrain_blocks_priority=False)) > 0.0


# --- Unburden ---------------------------------------------------------------------------------


def test_unburden_doubles_speed_only_once_the_item_is_gone() -> None:
    kwargs = dict(sp_spread={"spe": 32}, nature="jolly", ability="unburden")
    holding = PokemonState("sneasler", item="psychicseed", **kwargs)
    lost = PokemonState("sneasler", item_lost=True, **kwargs)
    other = PokemonState("sneasler", item_lost=True, sp_spread={"spe": 32}, nature="jolly", ability="poisontouch")
    assert effective_speed(lost) == 2 * effective_speed(holding)
    assert effective_speed(other) == effective_speed(holding)


def test_state_builders_mark_lost_items_only_with_the_knob() -> None:
    assert item_was_lost(None) and item_was_lost("")
    assert not item_was_lost("unknown_item") and not item_was_lost("leftovers")

    def mon(item):
        return SimpleNamespace(
            species="sneasler", ability="unburden", item=item, evs=None, nature=None,
            boosts={}, status=None, current_hp=100, current_hp_fraction=1.0, fainted=False,
        )

    assert _our_pokemon_state(mon(""), True, True).item_lost is True
    assert _our_pokemon_state(mon(""), True, False).item_lost is False
    assert _our_pokemon_state(mon("leftovers"), True, True).item_lost is False
    assert opponent_state(mon(None), unburden=True).item_lost is True
    assert opponent_state(mon("unknown_item"), unburden=True).item_lost is False


def test_terrain_seed_is_consumed_for_unburden_holders_only() -> None:
    sneasler = PokemonState("sneasler", item="psychicseed", ability="unburden")
    other = PokemonState("garchomp", item="psychicseed", ability="roughskin")
    _consume_terrain_seeds([sneasler, other, None], "psychic")
    assert sneasler.item is None and sneasler.item_lost is True
    assert other.item == "psychicseed" and other.item_lost is False
    wrong_terrain = PokemonState("sneasler", item="psychicseed", ability="unburden")
    _consume_terrain_seeds([wrong_terrain], "grassy")
    assert wrong_terrain.item_lost is False


def test_search_move_order_sees_unburden_after_the_seed_is_eaten() -> None:
    # Sneasler is slower than the Garchomp until Psychic Terrain eats its seed (x2 Speed).
    sneasler = PokemonState(
        "sneasler", sp_spread={"atk": 32, "spe": 4}, nature="adamant", item="psychicseed",
        ability="unburden",
    )
    foe = PokemonState("garchomp", sp_spread={"hp": 2, "atk": 32, "spe": 32}, nature="jolly", current_hp=1)
    ctx = _search_ctx([sneasler, None], [foe, None], "psychic")
    order = _fake_order(_fake_single("closecombat", move_target=1), None)
    response = OppResponse(
        slot0=_OppSlotAction(kind="move", move_id="earthquake", target_our_slot=0),
        slot1=_OppSlotAction(kind="none"),
    )

    on = resolve_exchange(order, response, ctx, PolicyConfig())
    assert on.opp_faints == 1 and on.our_hp_lost_pct == 0.0  # Sneasler moved first
    off = resolve_exchange(order, response, ctx, PolicyConfig(model_unburden=False))
    assert off.our_hp_lost_pct > 0.0  # legacy: the Garchomp's Earthquake landed first


# --- spread recount -----------------------------------------------------------------------------


def test_spread_move_loses_the_penalty_once_its_partner_target_has_fainted() -> None:
    fast = _garchomp(sp_spread={"atk": 32, "spe": 32}, nature="jolly")
    slow = PokemonState("tyranitar", sp_spread={"atk": 32}, nature="adamant")
    weak_foe = _klefki(current_hp=1)
    tough_foe = PokemonState("blissey", sp_spread={"hp": 32, "def": 32}, nature="bold")
    ctx = _search_ctx([fast, slow], [weak_foe, tough_foe])
    order = _fake_order(
        _fake_single("earthquake", move_target=1),  # slot 0 KOs the 1-HP foe first
        _fake_single("rockslide"),  # then slot 1's spread move has one living target
    )

    def lost(config):
        return resolve_exchange(order, _NO_OPP, ctx, config).opp_hp_lost_pct

    on = lost(PolicyConfig())
    off = lost(PolicyConfig(spread_recount_targets=False))
    assert on > off


def test_spread_move_with_two_living_targets_is_unchanged() -> None:
    slow = PokemonState("tyranitar", sp_spread={"atk": 32}, nature="adamant")
    foes = [_klefki(), PokemonState("blissey", sp_spread={"hp": 32}, nature="bold")]
    ctx = _search_ctx([slow, None], foes)
    order = _fake_order(_fake_single("rockslide"), None)
    assert resolve_exchange(order, _NO_OPP, ctx, PolicyConfig()).opp_hp_lost_pct == pytest.approx(
        resolve_exchange(
            order, _NO_OPP, ctx, PolicyConfig(spread_recount_targets=False)
        ).opp_hp_lost_pct
    )


# --- priority STATUS moves under Psychic Terrain ---------------------------------------------


def _prankster_taunt_score(foe: PokemonState, config: PolicyConfig, terrain, ability="prankster"):
    from vgc.evaluator import _score_status_move

    attacker = PokemonState("whimsicott", ability=ability)
    ctx = replace(
        _attack_ctx(ally_state=None, opp_state=foe, attacker_state=attacker),
        terrain=terrain,
    )
    ctx.opp_pokemon[0].moves = {"protect": None, "swordsdance": None, "earthquake": None}
    move, single = _single_for("taunt", move_target=1)
    return _score_status_move("taunt", MOVES["taunt"], single, 0, ctx, config)


def test_prankster_taunt_into_grounded_foe_is_blocked_by_psychic_terrain() -> None:
    on, off = PolicyConfig(), PolicyConfig(psychic_terrain_blocks_priority=False)
    blocked = _prankster_taunt_score(_garchomp(), on, "psychic")
    assert blocked[0] == 0.0 and blocked[1]["reason"] == "psychic_terrain_blocks_priority"
    assert _prankster_taunt_score(_garchomp(), off, "psychic")[0] > 0.0
    assert _prankster_taunt_score(_garchomp(), on, None)[0] > 0.0
    # Not Prankster (priority 0), or an airborne foe: Taunt still lands.
    assert _prankster_taunt_score(_garchomp(), on, "psychic", ability="infiltrator")[0] > 0.0
    assert _prankster_taunt_score(PokemonState("talonflame"), on, "psychic")[0] > 0.0


def test_psychic_terrain_ignores_status_moves_that_do_not_target_a_foe() -> None:
    prankster = PokemonState("whimsicott", ability="prankster")
    foe = _garchomp()
    for move_id in ("tailwind", "reflect", "protect", "stealthrock"):
        assert psychic_terrain_blocks(MOVES[move_id], prankster, foe, "psychic") is False
    assert psychic_terrain_blocks(MOVES["taunt"], prankster, foe, "psychic") is True


def test_search_prankster_taunt_is_blocked_by_psychic_terrain() -> None:
    prankster = PokemonState("whimsicott", ability="prankster")
    order = _fake_order(_fake_single("taunt", move_target=1), None)

    def utility(foe, config, terrain="psychic"):
        ctx = _search_ctx([prankster, None], [foe, None], terrain)
        ctx.opp_pokemon[0].moves = {"protect": None, "earthquake": None, "swordsdance": None}
        return resolve_exchange(order, _NO_OPP, ctx, config).our_utility_value

    assert utility(_garchomp(), PolicyConfig(psychic_terrain_blocks_priority=False)) > 0.0
    assert utility(_garchomp(), PolicyConfig()) == 0.0
    assert utility(_garchomp(), PolicyConfig(), terrain=None) > 0.0
    assert utility(PokemonState("talonflame"), PolicyConfig()) > 0.0
