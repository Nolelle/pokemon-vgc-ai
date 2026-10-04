"""`PolicyConfig.search_apply_setup_boosts`: the fast search simulates stat-stage changes.

Hand-built contexts (same style as tests/test_search.py). Off is the legacy behaviour:
no stage ever changes and the flat utility proxy is credited as before.
"""

from __future__ import annotations

import pytest

import vgc.search as search_module
from tests.test_search import _build_ctx, _fake_order, _fake_single, _garchomp, _klefki, _mon
from vgc.data import load_moves
from vgc.evaluator import ScoredOrder
from vgc.models import PolicyConfig
from vgc.search import OppResponse, _OppSlotAction, resolve_exchange
from vgc.setup_boosts import SETUP_BOOSTS, apply_stages

ON = PolicyConfig(search_apply_setup_boosts=True)
NO_OP = OppResponse(slot0=_OppSlotAction(kind="none"), slot1=_OppSlotAction(kind="none"))


def _swords_dance_ctx():
    return _build_ctx(
        our_states=[_garchomp(), None],
        opp_states=[_klefki(), None],
        our_pokemon=[
            _mon(moves={"swordsdance": None, "bodyslam": None}, species="garchomp"),
            None,
        ],
        opp_pokemon=[_mon(species="klefki"), None],
    )


def test_boost_table_matches_champions_move_data() -> None:
    moves = load_moves()
    for move_id, plan in SETUP_BOOSTS.items():
        data = moves[move_id]
        assert data["isNonstandard"] is None, move_id  # legal in Reg M-C
        assert data["category"] == "Status", move_id
        if plan.ally_stages and not plan.self_stages:
            assert data["target"] == "adjacentAlly", move_id
        elif plan.ally_stages:
            assert data["target"] == "allies", move_id  # Howl: user and partner
        else:
            assert data["target"] == "self", move_id


def test_swords_dance_raises_attack_and_forecast_damage_when_on() -> None:
    ctx = _swords_dance_ctx()
    order = _fake_order(_fake_single("swordsdance"), None)

    off = resolve_exchange(order, NO_OP, ctx, PolicyConfig())
    on = resolve_exchange(order, NO_OP, ctx, ON)

    assert off.our_states[0].boosts == {}
    assert off.our_utility_value == pytest.approx(25.0 * 0.9)  # legacy flat proxy
    assert on.our_states[0].boosts == {"atk": 2}
    assert on.our_utility_value == 0.0  # payoff is simulated, not also credited flat
    assert ctx.our_states[0].boosts == {}  # the shared context is never mutated

    # The same two-turn forecast now prices the payoff through damage.
    forecast_off = search_module.forecast_position(off, ctx, PolicyConfig())
    forecast_on = search_module.forecast_position(on, ctx, ON)
    assert forecast_on.opp_hp_lost_pct > forecast_off.opp_hp_lost_pct

    kept = resolve_exchange(order, NO_OP, ctx, PolicyConfig(
        search_apply_setup_boosts=True, setup_boost_flat_utility_scale=1.0
    ))
    assert kept.our_utility_value == pytest.approx(25.0 * 0.9)


def test_stage_changes_clamp_and_respect_abilities_and_costs() -> None:
    boosts = {"atk": 5}
    apply_stages(boosts, {"atk": 2})
    assert boosts == {"atk": 6}
    contrary = {}
    apply_stages(contrary, {"atk": 2}, ability="contrary")
    assert contrary == {"atk": -2}

    ctx = _build_ctx(
        our_states=[_garchomp(), None],
        opp_states=[_klefki(), None],
        our_pokemon=[_mon(moves={"shellsmash": None, "bellydrum": None}, species="garchomp"), None],
        opp_pokemon=[_mon(species="klefki"), None],
    )
    smash = resolve_exchange(_fake_order(_fake_single("shellsmash"), None), NO_OP, ctx, ON)
    assert smash.our_states[0].boosts == {
        "atk": 2, "spa": 2, "spe": 2, "def": -1, "spd": -1
    }
    drum = resolve_exchange(_fake_order(_fake_single("bellydrum"), None), NO_OP, ctx, ON)
    assert drum.our_states[0].boosts == {"atk": 6}
    assert drum.our_hp_lost_pct == pytest.approx(50.0, abs=1.0)

    weak = _garchomp(current_hp=_garchomp().max_hp() // 2)  # at half HP Belly Drum fails
    ctx.our_states[0] = weak
    failed = resolve_exchange(_fake_order(_fake_single("bellydrum"), None), NO_OP, ctx, ON)
    assert failed.our_states[0].boosts == {} and failed.our_hp_lost_pct == 0.0


def test_coaching_boosts_the_partner_when_on_and_is_unmodelled_when_off() -> None:
    ctx = _build_ctx(
        our_states=[_klefki(), _garchomp()],
        opp_states=[_klefki(), None],
        our_pokemon=[_mon(moves={"coaching": None}, species="klefki"), _mon(species="garchomp")],
        opp_pokemon=[_mon(species="klefki"), None],
    )
    order = _fake_order(_fake_single("coaching", move_target=-2), None)

    off = resolve_exchange(order, NO_OP, ctx, PolicyConfig())
    on = resolve_exchange(order, NO_OP, ctx, ON)

    assert all(state is None or state.boosts == {} for state in off.our_states)
    assert on.our_states[1].boosts == {"atk": 1, "def": 1}
    assert on.our_states[0].boosts == {}
    assert on.our_utility_value == 0.0

    # A fainted partner cannot be coached.
    ctx.our_states[1] = _garchomp(current_hp=0)
    gone = resolve_exchange(order, NO_OP, ctx, ON)
    assert gone.our_states[1].boosts == {}


def test_coaching_line_is_shortlisted_as_setup_only_when_on() -> None:
    attack = ScoredOrder(order=_fake_order(_fake_single("bodyslam", 1), _fake_single("bodyslam", 1)),
                         score=100.0, breakdown={})
    other = ScoredOrder(order=_fake_order(_fake_single("protect"), _fake_single("bodyslam", 1)),
                        score=90.0, breakdown={})
    coach = ScoredOrder(order=_fake_order(_fake_single("coaching", -2), _fake_single("bodyslam", 1)),
                        score=10.0, breakdown={})
    assert "setup" not in search_module._order_tags(coach.order)
    assert "setup" in search_module._order_tags(coach.order, True)

    for knob, expected in ((False, False), (True, True)):
        searched, _rest = search_module._select_search_candidates(
            [attack, other, coach],
            PolicyConfig(search_our_candidates=2, search_apply_setup_boosts=knob),
        )
        assert (coach in searched) is expected
