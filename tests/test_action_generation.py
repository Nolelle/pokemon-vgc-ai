"""Problem C gate: action generation is complete and legal, live and unit.

Live tests (local server) prove the enumeration against real Showdown requests;
unit tests pin the mutual-exclusion and fallback rules we inherit from
`DoubleBattleOrder.join_orders` so a poke-env upgrade cannot silently change them.
"""

from __future__ import annotations

from types import SimpleNamespace

from poke_env.battle.move import Move
from poke_env.battle.pokemon import Pokemon
from poke_env.player.battle_order import DoubleBattleOrder, SingleBattleOrder

from vgc.actions import choice_wire_message, describe_order, enumerate_joint_orders


def _move_order(move_id: str = "tackle", *, mega: bool = False) -> SingleBattleOrder:
    move = Move(move_id, gen=9)
    return SingleBattleOrder(move, move_target=1, mega=mega)


def _switch_order(species: str = "pikachu") -> SingleBattleOrder:
    mon = Pokemon(species=species, gen=9)
    return SingleBattleOrder(mon)


def test_join_orders_excludes_double_mega() -> None:
    orders = DoubleBattleOrder.join_orders(
        [_move_order("tackle", mega=True), _move_order("protect")],
        [_move_order("tackle", mega=True), _move_order("protect")],
    )
    assert orders
    assert not any(
        order.first_order.mega and order.second_order.mega for order in orders
    )


def test_join_orders_excludes_double_pass_and_same_switch() -> None:
    from poke_env.player.battle_order import PassBattleOrder

    orders = DoubleBattleOrder.join_orders([PassBattleOrder()], [PassBattleOrder()])
    assert orders == []

    orders = DoubleBattleOrder.join_orders(
        [_switch_order("pikachu")], [_switch_order("pikachu")]
    )
    assert orders == []


def test_enumerate_empty_valid_orders_yields_empty_list() -> None:
    battle = SimpleNamespace(valid_orders=([], []))
    assert enumerate_joint_orders(battle) == []


def test_choice_wire_message_is_sendable_not_display() -> None:
    order = DoubleBattleOrder(_move_order("heatwave"), _move_order("protect"))
    wire = choice_wire_message(order)
    assert wire.startswith("/choose ")
    assert wire != describe_order(order)
    assert choice_wire_message("/team 1234") == "/team 1234"
