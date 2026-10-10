"""The fast search's projected turns (`forecast_position`) respect move locks and accuracy.

`PolicyConfig.forecast_respects_locks` / `forecast_move_accuracy` (default True; False is the
legacy control). Hand-built contexts, no server. The first test replays the ladder position
that exposed the bug (game 2695880700 turn 10) when its saved bundle is present locally.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from poke_env.battle.effect import Effect

import vgc.search as search_module
from tests.test_search import _build_ctx, _fake_order, _fake_single, _garchomp, _klefki, _mon
from vgc.damage import PokemonState
from vgc.models import PolicyConfig
from vgc.search import ExchangeResult, OppResponse, _OppSlotAction, resolve_exchange

NO_OP = OppResponse(slot0=_OppSlotAction(kind="none"), slot1=_OppSlotAction(kind="none"))
LEGACY = PolicyConfig(forecast_respects_locks=False, forecast_move_accuracy=False)
LOCKS_ONLY = PolicyConfig(forecast_move_accuracy=False)
ACCURACY_ONLY = PolicyConfig(forecast_respects_locks=False)


def _holder(item: str | None, moves: list[str], species: str = "garchomp") -> SimpleNamespace:
    mon = _mon(moves={move: None for move in moves}, species=species)
    mon.item = item
    return mon


def _board(our_moves: list[str], item: str | None = "choicescarf"):
    """Garchomp (ours, slot 0) against a lone Klefki that has no moves."""

    return _build_ctx(
        our_states=[_garchomp(), None],
        opp_states=[_klefki(), None],
        our_pokemon=[_holder(item, our_moves), None],
        opp_pokemon=[_mon(species="klefki"), None],
    )


def _our_attacks(ctx, exchange, config):
    return search_module._best_joint_forecast_attacks(
        "our", exchange.our_states, exchange.opp_states, exchange, ctx, config
    )


# --- locks ------------------------------------------------------------------------------------


def test_choice_holder_locked_into_protect_makes_no_projected_attack() -> None:
    ctx = _board(["earthquake", "protect"])
    exchange = resolve_exchange(
        _fake_order(_fake_single("protect"), None), NO_OP, ctx, LOCKS_ONLY
    )
    assert exchange.move_locks == {("our", 0): frozenset({"protect"})}
    assert _our_attacks(ctx, exchange, LOCKS_ONLY) == []

    legacy = resolve_exchange(_fake_order(_fake_single("protect"), None), NO_OP, ctx, LEGACY)
    assert legacy.move_locks == {}
    assert [a.move_id for a in _our_attacks(ctx, legacy, LEGACY)] == ["earthquake"]


def test_a_lock_created_by_an_attack_repeats_that_attack() -> None:
    ctx = _board(["earthquake", "bodyslam"])  # Earthquake is super effective on Klefki
    order = _fake_order(_fake_single("bodyslam", 1), None)
    locked = resolve_exchange(order, NO_OP, ctx, LOCKS_ONLY)
    assert [a.move_id for a in _our_attacks(ctx, locked, LOCKS_ONLY)] == ["bodyslam"]
    legacy = resolve_exchange(order, NO_OP, ctx, LEGACY)
    assert [a.move_id for a in _our_attacks(ctx, legacy, LEGACY)] == ["earthquake"]


def test_trick_does_not_lock_and_a_non_choice_holder_is_unrestricted() -> None:
    ctx = _board(["earthquake", "trick"])
    exchange = resolve_exchange(_fake_order(_fake_single("trick", 1), None), NO_OP, ctx, LOCKS_ONLY)
    assert exchange.move_locks == {}
    ctx = _board(["earthquake", "bodyslam"], item="leftovers")
    exchange = resolve_exchange(
        _fake_order(_fake_single("bodyslam", 1), None), NO_OP, ctx, LOCKS_ONLY
    )
    assert exchange.move_locks == {}


def test_a_slot_that_switches_out_carries_no_lock() -> None:
    ctx = _board(["earthquake", "protect"])
    switch = SimpleNamespace(order=SimpleNamespace(species="klefki"), mega=False, move_target=0)
    locks, no_repeat = search_module._projected_move_locks(
        _fake_order(switch, None), NO_OP, ctx
    )
    assert locks == {} and no_repeat == {}


def test_request_restrictions_carry_forward_for_a_non_choice_pokemon() -> None:
    ctx = _board(["earthquake", "bodyslam", "protect"], item="leftovers")

    def move(move_id: str, disabled: bool) -> dict:
        return {"id": move_id, "disabled": disabled}

    # Encore: only the encored move stays enabled.
    ctx.battle.last_request = {
        "active": [{"moves": [move("earthquake", True), move("bodyslam", False), move("protect", True)]}]
    }
    exchange = resolve_exchange(
        _fake_order(_fake_single("bodyslam", 1), None), NO_OP, ctx, LOCKS_ONLY
    )
    assert exchange.move_locks == {("our", 0): frozenset({"bodyslam"})}
    assert [a.move_id for a in _our_attacks(ctx, exchange, LOCKS_ONLY)] == ["bodyslam"]

    # A one-move request is a multi-turn lock / recharge turn: it binds this turn only.
    ctx.battle.last_request = {"active": [{"moves": [move("earthquake", False)]}]}
    exchange = resolve_exchange(
        _fake_order(_fake_single("bodyslam", 1), None), NO_OP, ctx, LOCKS_ONLY
    )
    assert exchange.move_locks == {}


def test_torment_bars_repeating_the_previous_turns_move() -> None:
    ctx = _board(["earthquake", "bodyslam"], item="leftovers")
    ctx.our_pokemon[0].effects = {Effect.TORMENT: 1}
    exchange = resolve_exchange(
        _fake_order(_fake_single("earthquake", 1), None), NO_OP, ctx, LOCKS_ONLY
    )
    assert exchange.no_repeat_moves == {("our", 0): "earthquake"}
    assert [a.move_id for a in _our_attacks(ctx, exchange, LOCKS_ONLY)] == ["bodyslam"]


def _opp_ctx(item: str | None, last_move: str | None, effects: dict | None = None):
    ctx = _build_ctx(
        our_states=[_garchomp(), None],
        opp_states=[_klefki(), None],
        our_pokemon=[_holder(None, ["earthquake"]), None],
        opp_pokemon=[_holder(item, ["psychic", "protect"], species="klefki"), None],
    )
    opp = ctx.opp_pokemon[0]
    opp.last_move = SimpleNamespace(id=last_move) if last_move else None
    opp.effects = effects or {}
    return ctx


def test_foe_locks_only_when_its_choice_item_is_publicly_known() -> None:
    response = OppResponse(
        slot0=_OppSlotAction(kind="protect", move_id="protect"), slot1=_OppSlotAction(kind="none")
    )
    order = _fake_order(_fake_single("earthquake", 1), None)
    known = _opp_ctx("choicescarf", "protect")
    locks, _ = search_module._projected_move_locks(order, response, known)
    assert locks == {("opp", 0): frozenset({"protect"})}
    exchange = resolve_exchange(order, response, known, LOCKS_ONLY)
    attacks = search_module._best_joint_forecast_attacks(
        "opp", exchange.opp_states, exchange.our_states, exchange, known, LOCKS_ONLY
    )
    assert attacks == []

    hidden = _opp_ctx("unknown_item", "protect")
    assert search_module._projected_move_locks(order, response, hidden)[0] == {}

    # Already locked into Protect: the lock is the move it used, not this response's candidate.
    attack = OppResponse(
        slot0=_OppSlotAction(kind="move", move_id="psychic", target_our_slot=0),
        slot1=_OppSlotAction(kind="none"),
    )
    assert search_module._projected_move_locks(order, attack, known)[0] == {
        ("opp", 0): frozenset({"protect"})
    }
    # First move since switching in: this turn's candidate creates the lock.
    fresh = _opp_ctx("choicescarf", None)
    assert search_module._projected_move_locks(order, attack, fresh)[0] == {
        ("opp", 0): frozenset({"psychic"})
    }
    # Encore on the foe pins its last move without any item.
    encored = _opp_ctx(None, "psychic", {Effect.ENCORE: 1})
    assert search_module._projected_move_locks(order, attack, encored)[0] == {
        ("opp", 0): frozenset({"psychic"})
    }


@pytest.mark.skipif(
    not Path(
        "runs/ladder/state-replays/battle-gen9championsvgc2026regmc-2695880700.json"
    ).exists(),
    reason="local ladder state-replay bundle not present",
)
def test_ladder_game_17_turn_10_scarf_indeedee_is_not_forecast_to_attack() -> None:
    from vgc.battle_state_replay import replay_battle_at_cutoff
    from vgc.evaluator import build_context

    bundle = json.loads(
        Path("runs/ladder/state-replays/battle-gen9championsvgc2026regmc-2695880700.json").read_text()
    )
    battle = asyncio.run(replay_battle_at_cutoff(bundle, 12))
    assert battle.turn == 10
    assert [m.id for m in battle.available_moves[0]] == ["protect"]  # the saved request
    config = PolicyConfig()
    ctx = build_context(battle, config)
    protect = next(
        entry
        for entry in search_module.score_joint_orders(battle, config)
        if entry.order.first_order.order.id == "protect"
        and entry.order.second_order.order.id == "rockslide"
    )
    response = search_module._enumerate_opp_responses(ctx, config)[0]
    exchange = resolve_exchange(protect.order, response, ctx, config)
    assert exchange.move_locks[("our", 0)] == frozenset({"protect"})
    attacks = search_module._best_joint_forecast_attacks(
        "our", exchange.our_states, exchange.opp_states, exchange, ctx, config
    )
    assert 0 not in {attack.slot for attack in attacks}
    legacy_exchange = resolve_exchange(protect.order, response, ctx, LEGACY)
    legacy = search_module._best_joint_forecast_attacks(
        "our", legacy_exchange.our_states, legacy_exchange.opp_states, legacy_exchange, ctx, LEGACY
    )
    assert 0 in {attack.slot for attack in legacy}  # the bug: Expanding Force was projected


# --- accuracy ---------------------------------------------------------------------------------


def _ko_board():
    foe = _klefki(current_hp=1)  # any hit is lethal
    return _build_ctx(
        our_states=[_garchomp(), None],
        opp_states=[foe, None],
        our_pokemon=[_holder(None, ["focusblast"]), None],
        opp_pokemon=[_mon(species="klefki"), None],
    )


def test_projected_ko_is_weighted_by_the_hit_chance() -> None:
    ctx = _ko_board()
    exchange = ExchangeResult(our_states=ctx.our_states, opp_states=ctx.opp_states)
    one_turn = {"rolling_horizon_turns": 1}
    legacy = search_module.forecast_position(
        exchange, ctx, PolicyConfig(forecast_move_accuracy=False, **one_turn)
    )
    modelled = search_module.forecast_position(exchange, ctx, PolicyConfig(**one_turn))
    assert legacy.opp_faints == 1
    assert modelled.opp_faints == pytest.approx(0.7)  # Focus Blast is 70% accurate
    assert modelled.opp_hp_lost_pct < legacy.opp_hp_lost_pct


def test_forecast_continues_from_the_exchanges_alive_probability() -> None:
    # After an exchange the foe is alive with probability 0.5 and its `current_hp` is the
    # expected HP (0.5 x 2). A sure-hit lethal projected attack can remove only the 0.5 left.
    foe = _klefki(current_hp=1)
    ctx = _build_ctx(
        our_states=[_garchomp(), None],
        opp_states=[foe, None],
        our_pokemon=[_holder(None, ["earthquake"]), None],
        opp_pokemon=[_mon(species="klefki"), None],
    )
    exchange = ExchangeResult(
        our_states=ctx.our_states,
        opp_states=ctx.opp_states,
        move_accuracy=True,
        opp_alive=[0.5, 1.0],
    )
    config = PolicyConfig(rolling_horizon_turns=1)
    modelled = search_module.forecast_position(exchange, ctx, config)
    legacy = search_module.forecast_position(
        exchange, ctx, PolicyConfig(forecast_move_accuracy=False, rolling_horizon_turns=1)
    )
    assert modelled.opp_faints == pytest.approx(0.5)
    assert legacy.opp_faints == 1


def test_accuracy_picks_the_reliable_move_when_expected_damage_is_close() -> None:
    raichu = PokemonState("raichu", sp_spread={"spa": 32}, nature="modest")
    ctx = _build_ctx(
        our_states=[raichu, None],
        opp_states=[_klefki(), None],
        our_pokemon=[_holder(None, ["thunder", "thunderbolt"], species="raichu"), None],
        opp_pokemon=[_mon(species="klefki"), None],
    )
    exchange = ExchangeResult(our_states=ctx.our_states, opp_states=ctx.opp_states)
    # 110 BP at 70% is worth less than 90 BP that always lands.
    assert [a.move_id for a in _our_attacks(ctx, exchange, LEGACY)] == ["thunder"]
    assert [a.move_id for a in _our_attacks(ctx, exchange, ACCURACY_ONLY)] == ["thunderbolt"]
