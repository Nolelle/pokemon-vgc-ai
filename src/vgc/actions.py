"""Helpers for enumerating and describing legal doubles actions.

poke-env's `DoubleBattle.valid_orders` returns one `list[SingleBattleOrder]` per active
slot (mega/tera/dynamax variants included whenever they're legal for that mon this turn).
`DoubleBattleOrder.join_orders` takes the two per-slot lists and returns every legal
*joint* order, already filtered for showdown's mutual-exclusion rules (e.g. both slots
can't mega evolve the same turn, both slots can't be PassBattleOrder, etc.).
"""

from __future__ import annotations

from poke_env.battle.double_battle import DoubleBattle
from poke_env.battle.move import Move
from poke_env.battle.pokemon import Pokemon
from poke_env.player.battle_order import DoubleBattleOrder, SingleBattleOrder


def enumerate_joint_orders(battle: DoubleBattle) -> list[DoubleBattleOrder]:
    """Return every legal joint (both-slot) order for the current doubles turn.

    Returns an empty list if `battle.valid_orders` yields no legal combination (e.g. both
    slots fainted with nothing to switch in) -- callers should fall back to
    `Player.choose_random_move`/`choose_default_move` in that case, same as poke-env's own
    `choose_random_doubles_move` does.
    """
    first_orders, second_orders = battle.valid_orders
    return DoubleBattleOrder.join_orders(first_orders, second_orders)


def _describe_single(order: SingleBattleOrder) -> str:
    target = order.order
    if isinstance(target, Move):
        label = target.id
        if order.mega:
            label += "-mega"
        if order.z_move:
            label += "-zmove"
        if order.dynamax:
            label += "-dynamax"
        if order.terastallize:
            label += "-tera"
        if order.move_target:
            label += f"@{order.move_target}"
        return label
    if isinstance(target, Pokemon):
        return f"switch->{target.species}"
    return str(target) or "pass"


def describe_order(order: DoubleBattleOrder) -> str:
    """Human-readable summary of a joint order, e.g. 'earthquake@1 / protect'."""
    return f"{_describe_single(order.first_order)} / {_describe_single(order.second_order)}"
