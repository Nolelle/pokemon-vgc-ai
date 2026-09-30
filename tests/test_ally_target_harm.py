"""Regression: foe-directed status moves aimed at our OWN partner are a cost, never a gain.

Ladder game battle-gen9championsvgc2026regmc-2678505187, turn 5: the 2-ply search
picked `sleeppowder@-2 / flareblitz@2` and our Venusaur put our own Incineroar to sleep.
`resolve_exchange` credited the action-denial utility to our side for ANY sleep hit,
including one on our ally (+18.75, the whole margin over `sludgebomb@2`). The myopic
evaluator also scored Parting Shot at full value whatever it targeted, which put it on
our own partner (or into an empty ally slot) across ~40 recorded ladder decisions.

Ally targets stay LEGAL (docs/action_generation_contract.md: every target Showdown
offers is enumerated); these tests pin the scoring, not the enumeration.
"""

from __future__ import annotations

import pytest

from tests.test_search import _build_ctx, _fake_order, _fake_single, _mon
from vgc.damage import PokemonState
from vgc.evaluator import _score_status_move
from vgc.data import load_moves
from vgc.models import PolicyConfig
from vgc.principles import harms_ally_target
from vgc.search import OppResponse, _OppSlotAction, _exchange_value, resolve_exchange

NO_OP = OppResponse(slot0=_OppSlotAction(kind="none"), slot1=_OppSlotAction(kind="none"))


def _venusaur() -> PokemonState:
    return PokemonState("venusaur", sp_spread={"hp": 32, "spa": 32}, nature="modest")


def _incineroar(**kwargs) -> PokemonState:
    return PokemonState("incineroar", sp_spread={"hp": 32, "atk": 32}, nature="adamant", **kwargs)


def _foe() -> PokemonState:
    # Not Grass (Rillaboom, the real game's foe, is powder-immune).
    return PokemonState("garchomp", sp_spread={"hp": 32, "atk": 32}, nature="adamant")


def _board(ally_fainted: bool = False, ally_ability: str | None = None):
    ally = None if ally_fainted else _incineroar(ability=ally_ability)
    return _build_ctx(
        our_states=[_venusaur(), ally],
        opp_states=[None, _foe()],
        our_pokemon=[
            _mon(moves={"sleeppowder": None, "sludgebomb": None}, species="venusaur"),
            _mon(fainted=ally_fainted, moves={"flareblitz": None}, species="incineroar"),
        ],
        opp_pokemon=[None, _mon(species="garchomp")],
    )


def test_sleep_powder_on_own_ally_is_a_cost_in_the_search_exchange() -> None:
    ctx = _board()
    config = PolicyConfig()
    on_ally = resolve_exchange(
        _fake_order(_fake_single("sleeppowder", -2), _fake_single("flareblitz", 2)),
        NO_OP,
        ctx,
        config,
    )
    on_foe = resolve_exchange(
        _fake_order(_fake_single("sleeppowder", 2), _fake_single("flareblitz", 2)),
        NO_OP,
        ctx,
        config,
    )

    # The sleep lands on our side, and it is scored as our loss, not our gain.
    assert on_ally.our_states[1].status == "slp"
    assert on_ally.our_utility_value < 0.0
    assert on_foe.our_utility_value > 0.0
    assert _exchange_value(on_ally, config) < _exchange_value(on_foe, config)


@pytest.mark.parametrize("move_id", ["partingshot", "taunt", "thunderwave", "willowisp"])
def test_foe_directed_utility_on_own_ally_is_a_cost_in_the_search_exchange(move_id) -> None:
    ctx = _board()
    result = resolve_exchange(
        _fake_order(_fake_single(move_id, -2), None), NO_OP, ctx, PolicyConfig()
    )
    assert result.our_utility_value < 0.0


def test_utility_aimed_at_a_fainted_ally_slot_earns_nothing_in_the_search() -> None:
    # Showdown does not retarget a move aimed at a fainted ally ([notarget]).
    ctx = _board(ally_fainted=True)
    result = resolve_exchange(
        _fake_order(_fake_single("partingshot", -2), None), NO_OP, ctx, PolicyConfig()
    )
    assert result.our_utility_value == 0.0


def test_myopic_scores_harmful_status_on_ally_below_zero() -> None:
    ctx = _board()
    config = PolicyConfig()
    for move_id in ("sleeppowder", "partingshot", "taunt", "willowisp"):
        score, info = _score_status_move(
            move_id, load_moves()[move_id], _fake_single(move_id, -2), 0, ctx, config
        )
        assert score == -config.ally_harmful_status_penalty, move_id
        assert info["reason"] == "harmful_status_on_ally"


def test_myopic_parting_shot_into_a_fainted_ally_slot_scores_zero() -> None:
    ctx = _board(ally_fainted=True)
    score, info = _score_status_move(
        "partingshot", load_moves()["partingshot"], _fake_single("partingshot", -2), 0, ctx,
        PolicyConfig(),
    )
    assert score == 0.0
    assert info["reason"] == "fainted_ally_target"


def test_ally_ability_that_benefits_is_the_deliberate_exception() -> None:
    assert not harms_ally_target("willowisp", "guts")
    assert not harms_ally_target("thunderwave", "motordrive")
    assert harms_ally_target("willowisp", "intimidate")
    # No ability makes sleep a benefit; ally-support moves are never "harmful".
    assert harms_ally_target("sleeppowder", "guts")
    assert not harms_ally_target("helpinghand")
    assert not harms_ally_target("pollenpuff")

    ctx = _board(ally_ability="guts")
    score, info = _score_status_move(
        "willowisp", load_moves()["willowisp"], _fake_single("willowisp", -2), 0, ctx,
        PolicyConfig(),
    )
    assert score == 0.0
    assert info["reason"] == "ally_target_unmodeled"


def test_search_does_not_pick_sleep_on_own_ally_when_every_exchange_is_flat() -> None:
    """The game-5 shape: no modeled opponent reply, so exchanges are otherwise equal and
    the only thing separating Sleep Powder on the ally from an attack was the bogus
    credit. The attack must now rank above it."""
    ctx = _board()
    config = PolicyConfig()
    def value(move_id: str, target: int) -> float:
        order = _fake_order(_fake_single(move_id, target), None)
        return _exchange_value(resolve_exchange(order, NO_OP, ctx, config), config)

    values = {"sleep_ally": value("sleeppowder", -2), "sludge_foe": value("sludgebomb", 2)}
    assert values["sludge_foe"] > values["sleep_ally"]


def test_opposing_follow_me_pulls_an_ally_aimed_sleep_onto_the_redirector() -> None:
    # Showdown redirects a single-target move to the foe's Follow Me user even when it
    # was aimed at our own partner, so this is a gain for us, not a cost.
    ctx = _build_ctx(
        our_states=[_venusaur(), _incineroar()],
        opp_states=[PokemonState("clefable", sp_spread={"hp": 32}, nature="bold"), _foe()],
        our_pokemon=[_mon(species="venusaur"), _mon(species="incineroar")],
        opp_pokemon=[_mon(species="clefable"), _mon(species="garchomp")],
    )
    follow_me = OppResponse(
        slot0=_OppSlotAction(kind="utility", move_id="followme", utility_value=10.0),
        slot1=_OppSlotAction(kind="none"),
    )
    result = resolve_exchange(
        _fake_order(_fake_single("spore", -2), None), follow_me, ctx, PolicyConfig()
    )
    assert result.opp_states[0].status == "slp"
    assert result.our_states[1].status is None
    assert result.our_utility_value > 0.0


def test_no_cost_charged_when_the_ally_faints_before_the_move_executes() -> None:
    ctx = _build_ctx(
        our_states=[_venusaur(), _incineroar(current_hp=1)],
        opp_states=[None, _foe()],
        our_pokemon=[_mon(species="venusaur"), _mon(species="incineroar")],
        opp_pokemon=[None, _mon(species="garchomp")],
    )
    ko_ally_first = OppResponse(
        slot0=_OppSlotAction(kind="none"),
        slot1=_OppSlotAction(kind="move", move_id="dragonclaw", target_our_slot=1),
    )
    result = resolve_exchange(
        _fake_order(_fake_single("partingshot", -2), None), ko_ally_first, ctx, PolicyConfig()
    )
    assert result.our_faints == 1
    assert result.our_utility_value == 0.0  # Parting Shot fails; neither credit nor cost


def test_contrary_ally_makes_stat_drops_a_deliberate_exception() -> None:
    assert not harms_ally_target("partingshot", "contrary")
    assert not harms_ally_target("scaryface", "contrary")
    assert harms_ally_target("partingshot", "defiant")  # Defiant ignores an ally's drop
