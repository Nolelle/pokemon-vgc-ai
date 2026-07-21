"""Unit tests for `vgc.search`'s Phase 2c shallow 2-ply search, using directly
hand-built `_Context`/`PokemonState`s -- no live battle/server needed. Mirrors
tests/test_evaluator.py's "hand-computed expected values over fake contexts" style.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from poke_env.battle.double_battle import DoubleBattle
from poke_env.battle.move import Move
from poke_env.battle.pokemon import Pokemon

import vgc.agent as agent_module
import vgc.search as search_module
from vgc.agent import VgcPlayer
from vgc.damage import PokemonState
from vgc.evaluator import ScoredOrder, _Context, _ThreatInfo
from vgc.models import PolicyConfig
from vgc.search import (
    ExchangeResult,
    OppResponse,
    _OppSlotAction,
    _aggregate_exchange_values,
    _enumerate_opp_responses,
    _opp_slot_candidates,
    _response_weights,
    _value_head_delta,
    resolve_exchange,
)

# --- hand-built PokemonState/context helpers (mirrors test_evaluator.py) -------------


def _garchomp(**kwargs) -> PokemonState:
    defaults = dict(sp_spread={"hp": 2, "atk": 32, "spe": 32}, nature="jolly")
    defaults.update(kwargs)
    return PokemonState("garchomp", **defaults)


def _klefki(**kwargs) -> PokemonState:
    defaults = dict(sp_spread={"hp": 32, "def": 32, "spd": 2}, nature="bold")
    defaults.update(kwargs)
    return PokemonState("klefki", **defaults)


def _mon(
    fainted: bool = False,
    moves: dict | None = None,
    protect_counter: int = 0,
    species: str = "missingno",
) -> SimpleNamespace:
    return SimpleNamespace(
        fainted=fainted,
        moves=moves or {},
        ability=None,
        protect_counter=protect_counter,
        species=species,
    )


def _build_ctx(
    *,
    our_states: list,
    opp_states: list,
    our_pokemon: list,
    opp_pokemon: list,
    trick_room: bool = False,
    weather: str | None = None,
    our_side_screens: frozenset = frozenset(),
    opp_side_screens: frozenset = frozenset(),
    priors: dict | None = None,
    gameplan=None,
) -> _Context:
    return _Context(
        battle=SimpleNamespace(side_conditions=[], opponent_side_conditions=[]),
        trick_room=trick_room,
        weather=weather,
        terrain=None,
        our_side_screens=our_side_screens,
        opp_side_screens=opp_side_screens,
        our_pokemon=our_pokemon,
        opp_pokemon=opp_pokemon,
        our_states=our_states,
        opp_states=opp_states,
        our_speed=[0.0, 0.0],
        opp_speed=[0.0, 0.0],
        threat_on_us=[_ThreatInfo(), _ThreatInfo()],
        opp_threat_score=[0.0, 0.0],
        opp_protect_prob=[0.0, 0.0],
        opp_switch_prob=[0.0, 0.0],
        priors=priors if priors is not None else {},
        gameplan=gameplan,
    )


def _fake_single(move_id: str, move_target: int = 0, mega: bool = False) -> SimpleNamespace:
    move = Move(move_id, gen=9)
    return SimpleNamespace(order=move, mega=mega, move_target=move_target)


def _fake_order(first, second) -> SimpleNamespace:
    return SimpleNamespace(first_order=first, second_order=second)


# --- resolve_exchange: turn-order, Protect, switches, ally spread damage -------------


def test_faster_attacker_ko_prevents_slower_defender_from_acting() -> None:
    attacker = _garchomp()  # fast (32 SP jolly)
    defender = _klefki(current_hp=1)  # any hit guarantee-KOs; slow (2 SP spd, no spe)
    ctx = _build_ctx(
        our_states=[attacker, None],
        opp_states=[defender, None],
        our_pokemon=[_mon(), None],
        opp_pokemon=[_mon(moves={"psychic": None}), None],
    )
    our_order = _fake_order(_fake_single("earthquake"), None)
    opp_response = OppResponse(
        slot0=_OppSlotAction(kind="move", move_id="psychic", target_our_slot=0),
        slot1=_OppSlotAction(kind="none"),
    )

    result = resolve_exchange(our_order, opp_response, ctx, PolicyConfig())

    # Garchomp is faster and KOs Klefki before Klefki's queued psychic ever fires.
    assert result.opp_faints == 1
    assert result.our_hp_lost_pct == 0.0


def test_trick_room_inverts_turn_order_so_slower_attacker_lands_its_hit_first() -> None:
    attacker = _garchomp()
    defender = _klefki(current_hp=1)
    ctx = _build_ctx(
        our_states=[attacker, None],
        opp_states=[defender, None],
        our_pokemon=[_mon(), None],
        opp_pokemon=[_mon(moves={"psychic": None}), None],
        trick_room=True,
    )
    our_order = _fake_order(_fake_single("earthquake"), None)
    opp_response = OppResponse(
        slot0=_OppSlotAction(kind="move", move_id="psychic", target_our_slot=0),
        slot1=_OppSlotAction(kind="none"),
    )

    result = resolve_exchange(our_order, opp_response, ctx, PolicyConfig())

    # Under Trick Room the SLOWER Klefki now acts first: its psychic lands on Garchomp
    # before Garchomp's earthquake (still guaranteed-KOs the 1 HP Klefki afterward).
    assert result.our_hp_lost_pct > 0.0
    assert result.opp_faints == 1


def test_protect_blocks_a_single_target_hit() -> None:
    attacker = _garchomp()
    defender = _klefki(current_hp=1)
    ctx = _build_ctx(
        our_states=[attacker, None],
        opp_states=[defender, None],
        our_pokemon=[_mon(), None],
        opp_pokemon=[_mon(moves={"protect": None}), None],
    )
    our_order = _fake_order(_fake_single("bodyslam", move_target=1), None)
    opp_response = OppResponse(
        slot0=_OppSlotAction(kind="protect", move_id="protect"),
        slot1=_OppSlotAction(kind="none"),
    )

    result = resolve_exchange(our_order, opp_response, ctx, PolicyConfig())

    assert result.opp_faints == 0
    assert result.opp_hp_lost_pct == 0.0


def test_protect_blocks_spread_hit_on_protector_but_not_the_other_slot() -> None:
    attacker = _garchomp()
    protector = _klefki(current_hp=1)  # would faint if hit -- must survive under Protect
    other = _klefki()  # full HP -- should still take real spread damage
    ctx = _build_ctx(
        our_states=[attacker, None],
        opp_states=[protector, other],
        our_pokemon=[_mon(), None],
        opp_pokemon=[_mon(moves={"protect": None}), _mon(moves={})],
    )
    our_order = _fake_order(_fake_single("earthquake"), None)
    opp_response = OppResponse(
        slot0=_OppSlotAction(kind="protect", move_id="protect"),
        slot1=_OppSlotAction(kind="none"),
    )

    result = resolve_exchange(our_order, opp_response, ctx, PolicyConfig())

    assert result.opp_faints == 0  # protector survived; "other" wasn't at lethal HP
    assert result.opp_hp_lost_pct > 0.0  # but "other" still took its share of the spread


def test_repeated_our_protect_uses_expected_failure_cost_in_exchange() -> None:
    defender = _garchomp(current_hp=1)
    attacker = _klefki()
    ctx = _build_ctx(
        our_states=[defender, None],
        opp_states=[attacker, None],
        our_pokemon=[_mon(moves={"protect": None}, protect_counter=1), None],
        opp_pokemon=[_mon(moves={"bodyslam": None}), None],
    )
    our_order = _fake_order(_fake_single("protect"), None)
    opp_response = OppResponse(
        slot0=_OppSlotAction(kind="move", move_id="bodyslam", target_our_slot=0),
        slot1=_OppSlotAction(kind="none"),
    )

    result = resolve_exchange(our_order, opp_response, ctx, PolicyConfig())

    success_prob = PolicyConfig().protect_success_decay
    failure_prob = 1.0 - success_prob
    assert result.our_faints == pytest.approx(failure_prob)
    assert result.our_hp_lost_pct == pytest.approx(
        100.0 / defender.max_hp() * failure_prob
    )
    assert result.our_states[0].hp_or_max() == pytest.approx(success_prob)
    # The context remains immutable even though the returned state is the expected
    # success/failure blend consumed by the value head.
    assert ctx.our_states[0].current_hp == 1


def test_repeated_our_protect_legacy_control_still_assumes_success() -> None:
    defender = _garchomp(current_hp=1)
    ctx = _build_ctx(
        our_states=[defender, None],
        opp_states=[_klefki(), None],
        our_pokemon=[_mon(moves={"protect": None}, protect_counter=1), None],
        opp_pokemon=[_mon(moves={"bodyslam": None}), None],
    )
    our_order = _fake_order(_fake_single("protect"), None)
    opp_response = OppResponse(
        slot0=_OppSlotAction(kind="move", move_id="bodyslam", target_our_slot=0),
        slot1=_OppSlotAction(kind="none"),
    )

    result = resolve_exchange(
        our_order,
        opp_response,
        ctx,
        PolicyConfig(search_respect_our_protect_odds=False),
    )

    assert result.our_faints == 0.0
    assert result.our_hp_lost_pct == 0.0
    assert result.our_states[0].hp_or_max() == 1


def test_our_switch_replaces_slot_and_takes_no_action() -> None:
    opp_attacker = _klefki(current_hp=1)  # would guarantee-KO the outgoing mon's move
    switch_target = Pokemon(gen=9, species="klefki")
    switch_target._current_hp = 100  # comfortably alive, not full/overridden max

    ctx = _build_ctx(
        our_states=[_garchomp(), None],
        opp_states=[opp_attacker, None],
        our_pokemon=[_mon(), None],
        opp_pokemon=[_mon(moves={"bodyslam": None}), None],
    )
    switch_single = SimpleNamespace(order=switch_target, mega=False, move_target=0)
    our_order = _fake_order(switch_single, None)
    opp_response = OppResponse(
        slot0=_OppSlotAction(kind="move", move_id="bodyslam", target_our_slot=0),
        slot1=_OppSlotAction(kind="none"),
    )

    result = resolve_exchange(our_order, opp_response, ctx, PolicyConfig())

    # The outgoing mon's own move never fires (it switched instead) -- the opposing
    # attacker (1 HP) is untouched.
    assert result.opp_faints == 0
    assert result.opp_hp_lost_pct == 0.0
    # But the INCOMING mon is now the active target and eats the opponent's bodyslam.
    assert result.our_hp_lost_pct > 0.0


def test_opponent_defensive_switch_replaces_target_before_our_attack() -> None:
    outgoing = _garchomp(current_hp=1)
    incoming = _klefki()
    ctx = _build_ctx(
        our_states=[_garchomp(), None],
        opp_states=[outgoing, None],
        our_pokemon=[_mon(), None],
        opp_pokemon=[_mon(species="garchomp"), None],
    )
    our_order = _fake_order(_fake_single("bodyslam", move_target=1), None)
    response = OppResponse(
        slot0=_OppSlotAction(
            kind="switch",
            switch_state=incoming,
            switch_species="klefki",
        ),
        slot1=_OppSlotAction(kind="none"),
    )

    result = resolve_exchange(our_order, response, ctx, PolicyConfig())

    assert result.opp_states[0].species_id == "klefki"
    assert result.opp_faints == 0


def test_opponent_redirection_changes_our_single_target() -> None:
    redirector = _klefki()
    fragile_partner = _klefki(current_hp=1)
    ctx = _build_ctx(
        our_states=[_garchomp(), None],
        opp_states=[redirector, fragile_partner],
        our_pokemon=[_mon(), None],
        opp_pokemon=[_mon(moves={"ragepowder": None}), _mon()],
    )
    our_order = _fake_order(_fake_single("bodyslam", move_target=2), None)
    response = OppResponse(
        slot0=_OppSlotAction(
            kind="utility",
            move_id="ragepowder",
            utility_value=20.0,
        ),
        slot1=_OppSlotAction(kind="none"),
    )

    result = resolve_exchange(our_order, response, ctx, PolicyConfig())

    assert result.opp_states[1].hp_or_max() == 1
    assert result.opp_utility_value == 20.0


def test_our_ally_takes_spread_damage_counted_into_our_hp_lost() -> None:
    attacker = _garchomp()
    ally = _garchomp()
    opp = _klefki()
    ctx = _build_ctx(
        our_states=[attacker, ally],
        opp_states=[opp, None],
        our_pokemon=[_mon(), _mon()],
        opp_pokemon=[_mon(moves={}), None],
    )
    our_order = _fake_order(_fake_single("earthquake"), None)
    opp_response = OppResponse(slot0=_OppSlotAction(kind="none"), slot1=_OppSlotAction(kind="none"))

    result = resolve_exchange(our_order, opp_response, ctx, PolicyConfig())

    assert result.our_hp_lost_pct > 0.0


# --- response enumeration: per-slot cap, joint cap, always-include Protect -----------


def test_opp_slot_candidates_respects_per_slot_move_cap() -> None:
    config = PolicyConfig(search_opp_moves_per_slot=2)
    ctx = _build_ctx(
        our_states=[_garchomp(), None],
        opp_states=[_klefki(), None],
        our_pokemon=[_mon(), None],
        opp_pokemon=[
            _mon(
                moves={"bodyslam": None, "shadowball": None, "flashcannon": None, "icebeam": None}
            ),
            None,
        ],
    )
    candidates = _opp_slot_candidates(0, ctx, config)
    move_candidates = [c for c in candidates if c.kind == "move"]
    assert len(move_candidates) == 2


def test_opp_slot_candidates_always_includes_protect_when_available_and_unused() -> None:
    config = PolicyConfig()
    ctx = _build_ctx(
        our_states=[_garchomp(), None],
        opp_states=[_klefki(), None],
        our_pokemon=[_mon(), None],
        opp_pokemon=[_mon(moves={"bodyslam": None, "protect": None}, protect_counter=0), None],
    )
    candidates = _opp_slot_candidates(0, ctx, config)
    assert any(c.kind == "protect" for c in candidates)


def test_opp_slot_candidates_excludes_protect_when_already_used_last_turn() -> None:
    config = PolicyConfig()
    ctx = _build_ctx(
        our_states=[_garchomp(), None],
        opp_states=[_klefki(), None],
        our_pokemon=[_mon(), None],
        opp_pokemon=[_mon(moves={"bodyslam": None, "protect": None}, protect_counter=1), None],
    )
    candidates = _opp_slot_candidates(0, ctx, config)
    assert not any(c.kind == "protect" for c in candidates)


def test_opp_slot_candidates_include_setup_and_control_utility() -> None:
    config = PolicyConfig(search_opp_utility_per_slot=3)
    ctx = _build_ctx(
        our_states=[_garchomp(), None],
        opp_states=[_klefki(), None],
        our_pokemon=[_mon(), None],
        opp_pokemon=[
            _mon(moves={"taunt": None, "tailwind": None, "swordsdance": None}),
            None,
        ],
    )
    candidates = _opp_slot_candidates(0, ctx, config)
    utility_moves = {candidate.move_id for candidate in candidates if candidate.kind == "utility"}
    assert {"taunt", "tailwind", "swordsdance"} <= utility_moves


def test_opp_slot_candidates_no_protect_when_move_unknown() -> None:
    config = PolicyConfig()
    ctx = _build_ctx(
        our_states=[_garchomp(), None],
        opp_states=[_klefki(), None],
        our_pokemon=[_mon(), None],
        opp_pokemon=[_mon(moves={"bodyslam": None}, protect_counter=0), None],
    )
    candidates = _opp_slot_candidates(0, ctx, config)
    assert not any(c.kind == "protect" for c in candidates)


def test_enumerate_opp_responses_respects_joint_cap() -> None:
    config = PolicyConfig(search_opp_moves_per_slot=4, search_opp_candidates=3)
    moves = {"bodyslam": None, "shadowball": None, "flashcannon": None, "icebeam": None}
    ctx = _build_ctx(
        our_states=[_garchomp(), _garchomp()],
        opp_states=[_klefki(), _klefki()],
        our_pokemon=[_mon(), _mon()],
        opp_pokemon=[_mon(moves=moves), _mon(moves=moves)],
    )
    responses = _enumerate_opp_responses(ctx, config)
    assert 0 < len(responses) <= config.search_opp_candidates


# --- _response_weights: enumeration-score softmax + opp_protect_prob discounting ----


def _empty_ctx() -> _Context:
    return _build_ctx(
        our_states=[None, None],
        opp_states=[None, None],
        our_pokemon=[None, None],
        opp_pokemon=[None, None],
    )


def _no_op_response(enum_score: float) -> OppResponse:
    return OppResponse(
        slot0=_OppSlotAction(kind="none"),
        slot1=_OppSlotAction(kind="none"),
        enum_score=enum_score,
    )


def test_response_weights_higher_enum_score_gets_strictly_higher_weight() -> None:
    ctx = _empty_ctx()
    responses = [_no_op_response(10.0), _no_op_response(50.0)]

    weights = _response_weights(responses, ctx, PolicyConfig(search_response_temperature=30.0))

    assert weights[1] > weights[0]
    assert sum(weights) == pytest.approx(1.0)


def test_response_weights_approach_uniform_as_temperature_grows() -> None:
    ctx = _empty_ctx()
    responses = [_no_op_response(10.0), _no_op_response(90.0)]

    weights = _response_weights(
        responses, ctx, PolicyConfig(search_response_temperature=1_000_000.0)
    )

    assert weights[0] == pytest.approx(0.5, abs=1e-3)
    assert weights[1] == pytest.approx(0.5, abs=1e-3)


def test_response_weights_scaled_by_opp_protect_prob() -> None:
    ctx = _empty_ctx()
    ctx.opp_protect_prob[0] = 0.2
    # Same enum_score for both -- the only difference is that the second response
    # contains a Protect on opponent slot 0, whose probability of really happening
    # (per the evaluator's own estimate) is only 0.2.
    move_response = OppResponse(
        slot0=_OppSlotAction(kind="move", move_id="bodyslam"),
        slot1=_OppSlotAction(kind="none"),
        enum_score=50.0,
    )
    protect_response = OppResponse(
        slot0=_OppSlotAction(kind="protect", move_id="protect"),
        slot1=_OppSlotAction(kind="none"),
        enum_score=50.0,
    )

    weights = _response_weights([move_response, protect_response], ctx, PolicyConfig())

    assert weights[1] < weights[0]
    assert weights[1] / weights[0] == pytest.approx(0.2, rel=1e-6)


def test_response_weights_falls_back_to_uniform_when_all_zero() -> None:
    ctx = _empty_ctx()
    ctx.opp_protect_prob[0] = 0.0
    # Both responses Protect on a slot with 0.0 real Protect probability -- every raw
    # weight collapses to 0, which must fall back to uniform rather than divide by zero.
    responses = [
        OppResponse(
            slot0=_OppSlotAction(kind="protect", move_id="protect"),
            slot1=_OppSlotAction(kind="none"),
            enum_score=10.0,
        ),
        OppResponse(
            slot0=_OppSlotAction(kind="protect", move_id="protect"),
            slot1=_OppSlotAction(kind="none"),
            enum_score=50.0,
        ),
    ]

    weights = _response_weights(responses, ctx, PolicyConfig())

    assert weights == pytest.approx([0.5, 0.5])


# --- aggregation: worst-case vs response-likelihood-weighted expectation blend ------


def test_aggregate_exchange_values_blends_worst_case_and_expectation() -> None:
    config = PolicyConfig(search_worst_case_weight=0.7, search_response_temperature=1_000_000.0)
    ctx = _empty_ctx()
    responses = [_no_op_response(0.0), _no_op_response(0.0), _no_op_response(0.0)]
    values = [10.0, -20.0, 30.0]

    result = _aggregate_exchange_values(values, responses, ctx, config)

    # A huge temperature makes _response_weights essentially uniform, so the
    # expectation term reduces to the plain mean -- a useful special case to pin down.
    expected = 0.7 * min(values) + 0.3 * (sum(values) / len(values))
    assert result == pytest.approx(expected, rel=1e-3)


def test_aggregate_exchange_values_pure_worst_case_at_weight_one() -> None:
    config = PolicyConfig(search_worst_case_weight=1.0)
    ctx = _empty_ctx()
    responses = [_no_op_response(0.0), _no_op_response(0.0), _no_op_response(0.0)]
    values = [10.0, -20.0, 30.0]
    assert _aggregate_exchange_values(values, responses, ctx, config) == pytest.approx(min(values))


def test_aggregate_exchange_values_pure_expectation_at_weight_zero() -> None:
    config = PolicyConfig(search_worst_case_weight=0.0)
    ctx = _empty_ctx()
    # Equal enum_scores -> equal weights -> the expectation term is just the mean here.
    responses = [_no_op_response(5.0), _no_op_response(5.0), _no_op_response(5.0)]
    values = [10.0, -20.0, 30.0]
    assert _aggregate_exchange_values(values, responses, ctx, config) == pytest.approx(
        sum(values) / len(values)
    )


def test_aggregate_exchange_values_weights_toward_higher_scoring_responses() -> None:
    # Low temperature (sharp softmax) concentrates almost all weight on the
    # highest-enum_score response -- the expectation term should land close to ITS
    # value, not the plain average of all three.
    config = PolicyConfig(search_worst_case_weight=0.0, search_response_temperature=1.0)
    ctx = _empty_ctx()
    responses = [_no_op_response(0.0), _no_op_response(0.0), _no_op_response(100.0)]
    values = [10.0, -20.0, 30.0]
    assert _aggregate_exchange_values(values, responses, ctx, config) == pytest.approx(
        30.0, abs=0.5
    )


def test_aggregate_exchange_values_empty_list_is_zero() -> None:
    assert _aggregate_exchange_values([], [], _empty_ctx(), PolicyConfig()) == 0.0


# --- search_joint_orders: final blend / position_weight=0 regression guard ----------


def _tagged_fake_order(tag: str) -> SimpleNamespace:
    single = SimpleNamespace(
        order=tag, mega=False, z_move=False, dynamax=False, terastallize=False, move_target=0
    )
    return SimpleNamespace(first_order=single, second_order=single)


def test_search_position_weight_zero_matches_myopic_ranking_exactly(monkeypatch) -> None:
    order_a = _tagged_fake_order("a")
    order_b = _tagged_fake_order("b")
    order_c = _tagged_fake_order("c")
    myopic = [
        ScoredOrder(order=order_b, score=50.0, breakdown={}),
        ScoredOrder(order=order_a, score=30.0, breakdown={}),
        ScoredOrder(order=order_c, score=10.0, breakdown={}),
    ]

    monkeypatch.setattr(search_module, "score_joint_orders", lambda battle, config: list(myopic))
    monkeypatch.setattr(search_module, "build_context", lambda battle, config: object())
    fake_response = OppResponse(
        slot0=_OppSlotAction(kind="none"), slot1=_OppSlotAction(kind="none")
    )
    monkeypatch.setattr(
        search_module, "_enumerate_opp_responses", lambda ctx, config: [fake_response]
    )
    # Exchange values are deliberately scrambled relative to the myopic order (order_b
    # looks WORST in the exchange simulation despite scoring highest myopically) -- with
    # search_position_weight=0 this must have zero effect on the final ranking.
    monkeypatch.setattr(
        search_module,
        "resolve_exchange",
        lambda our_order, response, ctx, config: ExchangeResult(
            our_hp_lost_pct=100.0 if our_order is order_b else 0.0
        ),
    )

    config = PolicyConfig(
        search_position_weight=0.0, search_myopic_weight=1.0, search_our_candidates=10
    )
    scored = search_module.search_joint_orders(object(), config)

    assert [entry.order for entry in scored] == [order_b, order_a, order_c]


def test_search_position_weight_zero_holds_for_searched_block_with_tail_below(monkeypatch) -> None:
    orders = {tag: _tagged_fake_order(tag) for tag in "abcde"}
    myopic = [
        ScoredOrder(order=orders["b"], score=50.0, breakdown={}),  # searched
        ScoredOrder(order=orders["a"], score=30.0, breakdown={}),  # searched
        ScoredOrder(order=orders["c"], score=10.0, breakdown={}),  # searched
        ScoredOrder(order=orders["d"], score=9999.0, breakdown={}),  # tail (huge myopic score)
        ScoredOrder(order=orders["e"], score=8888.0, breakdown={}),  # tail
    ]

    monkeypatch.setattr(search_module, "score_joint_orders", lambda battle, config: list(myopic))
    monkeypatch.setattr(search_module, "build_context", lambda battle, config: object())
    fake_response = OppResponse(
        slot0=_OppSlotAction(kind="none"), slot1=_OppSlotAction(kind="none")
    )
    monkeypatch.setattr(
        search_module, "_enumerate_opp_responses", lambda ctx, config: [fake_response]
    )
    monkeypatch.setattr(
        search_module,
        "resolve_exchange",
        lambda our_order, response, ctx, config: ExchangeResult(
            our_hp_lost_pct=100.0 if our_order is orders["b"] else 0.0
        ),
    )

    config = PolicyConfig(
        search_position_weight=0.0, search_myopic_weight=1.0, search_our_candidates=3
    )
    scored = search_module.search_joint_orders(object(), config)

    # The searched block (b, a, c) keeps its myopic order despite scrambled exchange
    # values (position_weight=0), and BOTH tail entries -- despite their much larger raw
    # myopic scores -- rank strictly below every searched entry, preserving their own
    # myopic order (d before e) beneath the searched block (Fix 1).
    assert [entry.order for entry in scored] == [
        orders["b"],
        orders["a"],
        orders["c"],
        orders["d"],
        orders["e"],
    ]
    assert all(entry.breakdown["searched"] is True for entry in scored[:3])
    assert all(entry.breakdown["searched"] is False for entry in scored[3:])


def test_unsearched_tail_never_outranks_any_searched_order(monkeypatch) -> None:
    """Fix 1 regression guard: a tail order with a much larger bare myopic score must
    never outrank a searched order, even when that searched order's simulated exchange
    value is strongly negative (mirrors the live trace: an unsearched order at myopic
    116 beat the searched myopic-best at myopic 219 before this fix).
    """
    order_searched = _tagged_fake_order("searched-best")
    order_tail = _tagged_fake_order("tail-huge-myopic")
    myopic = [
        ScoredOrder(order=order_searched, score=219.0, breakdown={}),
        ScoredOrder(order=order_tail, score=9999.0, breakdown={}),
    ]

    monkeypatch.setattr(search_module, "score_joint_orders", lambda battle, config: list(myopic))
    monkeypatch.setattr(search_module, "build_context", lambda battle, config: object())
    fake_response = OppResponse(
        slot0=_OppSlotAction(kind="none"), slot1=_OppSlotAction(kind="none")
    )
    monkeypatch.setattr(
        search_module, "_enumerate_opp_responses", lambda ctx, config: [fake_response]
    )
    monkeypatch.setattr(
        search_module,
        "resolve_exchange",
        lambda our_order, response, ctx, config: ExchangeResult(
            our_hp_lost_pct=500.0, our_faints=3
        ),
    )

    config = PolicyConfig(search_our_candidates=1)  # only the first myopic entry is searched
    scored = search_module.search_joint_orders(object(), config)

    assert scored[0].order is order_searched
    assert scored[0].breakdown["searched"] is True
    assert scored[1].order is order_tail
    assert scored[1].breakdown["searched"] is False
    assert scored[0].score > scored[1].score


# --- agent wiring: use_two_ply_search gate --------------------------------------------


def _bare_player(config: PolicyConfig) -> VgcPlayer:
    player = VgcPlayer.__new__(VgcPlayer)
    player.config = config
    return player


def test_use_two_ply_search_false_never_consults_search_joint_orders(monkeypatch) -> None:
    battle = DoubleBattle(
        "battle-gen9championsvgc2026regmb-1", "user", logging.getLogger(__name__), 9
    )
    sentinel_order = object()
    search_calls: list[object] = []

    monkeypatch.setattr(
        agent_module, "search_joint_orders", lambda b, c: (search_calls.append(b), [])[1]
    )
    monkeypatch.setattr(
        agent_module,
        "score_joint_orders",
        lambda b, c: [ScoredOrder(order=sentinel_order, score=1.0, breakdown={})],
    )

    config = PolicyConfig(use_two_ply_search=False, use_heuristic_evaluator=True)
    result = VgcPlayer.decide(_bare_player(config), battle)

    assert search_calls == []
    assert result is sentinel_order


def test_use_two_ply_search_true_uses_shallow_search_result(monkeypatch) -> None:
    battle = DoubleBattle(
        "battle-gen9championsvgc2026regmb-2", "user", logging.getLogger(__name__), 9
    )
    sentinel_order = object()
    myopic_calls: list[object] = []

    monkeypatch.setattr(
        agent_module,
        "search_joint_orders",
        lambda b, c: [ScoredOrder(order=sentinel_order, score=1.0, breakdown={})],
    )
    monkeypatch.setattr(
        agent_module, "score_joint_orders", lambda b, c: (myopic_calls.append(b), [])[1]
    )

    config = PolicyConfig(use_two_ply_search=True)
    result = VgcPlayer.decide(_bare_player(config), battle)

    assert myopic_calls == []
    assert result is sentinel_order


# --- resolve_exchange: ExchangeResult carries post-exchange states/weather -----------


def test_resolve_exchange_result_carries_post_exchange_states_and_weather() -> None:
    attacker = _garchomp()
    defender = _klefki(current_hp=1)
    ctx = _build_ctx(
        our_states=[attacker, None],
        opp_states=[defender, None],
        our_pokemon=[_mon(), None],
        opp_pokemon=[_mon(moves={"psychic": None}), None],
        weather="rain",
    )
    our_order = _fake_order(_fake_single("earthquake"), None)
    opp_response = OppResponse(
        slot0=_OppSlotAction(kind="move", move_id="psychic", target_our_slot=0),
        slot1=_OppSlotAction(kind="none"),
    )

    result = resolve_exchange(our_order, opp_response, ctx, PolicyConfig())

    assert result.weather == "rain"
    assert len(result.our_states) == 2
    assert len(result.opp_states) == 2
    # Klefki (1 HP) was guarantee-KO'd by earthquake -- its post-exchange state should
    # reflect 0 (or near-0) HP, not the pre-exchange snapshot's.
    assert result.opp_states[0].hp_or_max() <= 1
    # ctx's OWN states must never be mutated (module docstring's "never mutate ctx"
    # contract) -- still the original object, still at its original 1 HP, even though
    # result.opp_states[0] (a COPY) now reflects the KO.
    assert ctx.opp_states[0] is defender
    assert defender.current_hp == 1
    assert result.opp_states[0] is not defender


# --- _value_head_delta: only nonzero when the full signal chain is available --------


def test_value_head_delta_zero_when_v_after_is_none() -> None:
    delta = _value_head_delta(None, 0.5, PolicyConfig())
    assert delta == 0.0


def test_value_head_delta_zero_when_v_before_is_none() -> None:
    delta = _value_head_delta(0.7, None, PolicyConfig())
    assert delta == 0.0


def test_value_head_delta_zero_when_both_are_none() -> None:
    delta = _value_head_delta(None, None, PolicyConfig())
    assert delta == 0.0


def test_value_head_delta_matches_the_documented_formula_when_available() -> None:
    config = PolicyConfig(value_head_weight=2.0)
    v_before = 0.3
    v_after = 0.55

    delta = _value_head_delta(v_after, v_before, config)

    expected = config.value_head_weight * 100.0 * (v_after - v_before)
    assert delta == pytest.approx(expected)
    assert delta == pytest.approx(50.0)  # 2.0 * 100 * (0.55 - 0.3)
    # Sanity: the weight actually scales the term (not silently ignored).
    delta_double_weight = _value_head_delta(v_after, v_before, PolicyConfig(value_head_weight=4.0))
    assert delta_double_weight == pytest.approx(delta * 2.0)


def test_position_values_batch_matches_position_value_per_state() -> None:
    """`position_values_batch` (used by search_joint_orders for latency -- one batched
    forward pass instead of one call per (candidate, response) pair) must produce
    numerically identical results to calling position_value once per state.
    """
    pytest.importorskip("torch")
    from vgc.bc.encoding import (
        ABILITY_VOCAB,
        ENCODER_LAYOUT_VERSION,
        ITEM_VOCAB,
        MOVE_VOCAB,
        SPECIES_VOCAB,
        TARGET_VOCAB,
    )
    from vgc.bc.model import BcPolicyNet
    from vgc.bc.policy import BcPolicy, exchange_state_record, position_value, position_values_batch

    import torch as torch_module

    torch_module.manual_seed(0)
    model = BcPolicyNet(heads=("value",))
    model.eval()
    policy = BcPolicy(
        model=model,
        species_vocab=list(SPECIES_VOCAB),
        move_vocab=list(MOVE_VOCAB),
        item_vocab=list(ITEM_VOCAB),
        ability_vocab=list(ABILITY_VOCAB),
        target_vocab=list(TARGET_VOCAB),
        encoder_layout_version=ENCODER_LAYOUT_VERSION,
        heads=model.heads,
    )
    ctx = _build_ctx(
        our_states=[_garchomp(), None],
        opp_states=[_klefki(), None],
        our_pokemon=[_mon(), None],
        opp_pokemon=[_mon(), None],
    )
    exchange_a = ExchangeResult(our_states=[_garchomp(), None], opp_states=[_klefki(), None])
    exchange_b = ExchangeResult(
        our_states=[_garchomp(current_hp=1), None], opp_states=[_klefki(), None]
    )
    records = [
        exchange_state_record(exchange_a.our_states, exchange_a.opp_states, ctx),
        exchange_state_record(exchange_b.our_states, exchange_b.opp_states, ctx),
    ]

    batched = position_values_batch(policy, records)
    individual = [position_value(policy, record) for record in records]

    assert len(batched) == 2
    for b, i in zip(batched, individual, strict=True):
        assert b == pytest.approx(i, abs=1e-5)


def test_position_values_batch_none_when_no_value_head() -> None:
    pytest.importorskip("torch")
    from vgc.bc.encoding import (
        ABILITY_VOCAB,
        ENCODER_LAYOUT_VERSION,
        ITEM_VOCAB,
        MOVE_VOCAB,
        SPECIES_VOCAB,
        TARGET_VOCAB,
    )
    from vgc.bc.model import BcPolicyNet
    from vgc.bc.policy import BcPolicy, exchange_state_record, position_values_batch

    model = BcPolicyNet(heads=("move", "target"))
    policy = BcPolicy(
        model=model,
        species_vocab=list(SPECIES_VOCAB),
        move_vocab=list(MOVE_VOCAB),
        item_vocab=list(ITEM_VOCAB),
        ability_vocab=list(ABILITY_VOCAB),
        target_vocab=list(TARGET_VOCAB),
        encoder_layout_version=ENCODER_LAYOUT_VERSION,
        heads=model.heads,
    )
    ctx = _build_ctx(
        our_states=[_garchomp(), None],
        opp_states=[_klefki(), None],
        our_pokemon=[_mon(), None],
        opp_pokemon=[_mon(), None],
    )
    exchange = ExchangeResult(our_states=[_garchomp(), None], opp_states=[_klefki(), None])
    records = [exchange_state_record(exchange.our_states, exchange.opp_states, ctx)]

    assert position_values_batch(policy, records) == [None]
    assert position_values_batch(None, records) == [None]
    assert position_values_batch(policy, []) == []
