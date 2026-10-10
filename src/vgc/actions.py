"""Helpers for enumerating and describing legal doubles actions.

poke-env's `DoubleBattle.valid_orders` returns one `list[SingleBattleOrder]` per active
slot (mega/tera/dynamax variants included whenever they're legal for that mon this turn).
`DoubleBattleOrder.join_orders` takes the two per-slot lists and returns every legal
*joint* order, already filtered for showdown's mutual-exclusion rules (e.g. both slots
can't mega evolve the same turn, both slots can't be PassBattleOrder, etc.).
"""

from __future__ import annotations

import re

from poke_env.battle.double_battle import DoubleBattle
from poke_env.battle.move import Move
from poke_env.battle.pokemon import Pokemon
from poke_env.player.battle_order import BattleOrder, DoubleBattleOrder, SingleBattleOrder


def choice_wire_message(order: BattleOrder | str) -> str:
    """Return the exact string sent to Showdown for one choice.

    This is the round-trip form: `DoubleBattleOrder.message` for in-battle
    orders (`/choose move ...`), the `/team ...` string unchanged for preview.
    Display labels from `describe_order` are for humans and must never be sent.
    Plain strings pass through so preview choices and fallbacks work unchanged.
    """
    if isinstance(order, str):
        return order
    message = order.message
    return message if isinstance(message, str) else str(message)


def enumerate_joint_orders(battle: DoubleBattle) -> list[DoubleBattleOrder]:
    """Return every legal joint (both-slot) order for the current doubles turn.

    Returns an empty list if `battle.valid_orders` yields no legal combination (e.g. both
    slots fainted with nothing to switch in) -- callers should fall back to
    `Player.choose_random_move`/`choose_default_move` in that case, same as poke-env's own
    `choose_random_doubles_move` does.
    """
    first_orders, second_orders = battle.valid_orders
    return DoubleBattleOrder.join_orders(first_orders, second_orders)


def locked_move_ids(battle) -> dict[int, str]:
    """Active slot -> id of the move it is locked into, from the current request."""
    request = getattr(battle, "last_request", None) or {}
    if request.get("forceSwitch") or request.get("teamPreview") or request.get("wait"):
        return {}
    locked: dict[int, str] = {}
    for slot, active in enumerate(request.get("active") or []):
        moves = (active or {}).get("moves") or []
        if len(moves) == 1 and "target" not in moves[0] and moves[0].get("id") != "recharge":
            locked[slot] = moves[0].get("id")
    return locked


def index_locked_choice(battle, message: str) -> str:
    """Rewrite a choice so a locked move (Outrage, Petal Dance, ...) is sent as `move 1`.

    While a Pokemon is locked, Showdown's request lists only that move and omits its
    `target`. Choosing it by name (`move outrage`, poke-env's form) makes Showdown fall
    back to target type `normal` (`move.target || 'normal'` in `sim/side.ts`) and reject
    it with "needs a target"; by index it is accepted.

    This must run where a choice is SENT, against the request of the battle it is sent
    to (`DirectBattle.step_payload`, `VgcPlayer._handle_battle_request`), never when
    orders are enumerated: an order can be enumerated against a different battle object
    than the one it is sent to (the search's mirror), and an enumeration-time `move 1`
    reached a real battle whose slot was not locked, where index 1 was a disabled
    Thunderbolt. Sending-time also covers poke-env's own pickers (random/default
    fallbacks, baseline opponents).
    """
    locked = locked_move_ids(battle)
    if not locked or not isinstance(message, str):
        return message
    prefix = "/choose " if message.startswith("/choose ") else ""
    parts = message.removeprefix(prefix).split(", ")
    for slot, locked_id in locked.items():
        if slot >= len(parts):
            continue
        tokens = parts[slot].split()
        if len(tokens) >= 2 and tokens[0] == "move" and tokens[1] == locked_id:
            parts[slot] = "move 1" + (" mega" if "mega" in tokens[2:] else "")
    return prefix + ", ".join(parts)


def _showdown_id(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def index_switch_choice(battle, message: str) -> str:
    """Rewrite `switch <name>` to `switch <position>` from the current request.

    Showdown resolves a switch by name OR current species (`chooseSwitch` in
    `sim/side.ts`), scanning the team in order. If an active Pokemon has transformed into
    a teammate's species (our Ditto copying our own Golurk), `switch golurk` matches the
    ACTIVE Ditto first and is rejected ("You can't switch to an active Pokemon"). A
    switch can only ever target a benched Pokemon, so pick the non-active request entry
    whose ident names it and send its 1-based position. Same send-time rule as
    `index_locked_choice`.
    """
    if not isinstance(message, str) or "switch " not in message:
        return message
    request = getattr(battle, "last_request", None) or {}
    team = (request.get("side") or {}).get("pokemon") or []
    if not team:
        return message
    prefix = "/choose " if message.startswith("/choose ") else ""
    parts = message.removeprefix(prefix).split(", ")
    for slot, part in enumerate(parts):
        tokens = part.split(maxsplit=1)
        if len(tokens) != 2 or tokens[0] != "switch" or tokens[1].isdigit():
            continue
        wanted = _showdown_id(tokens[1])
        for position, entry in enumerate(team, start=1):
            name = str(entry.get("ident", "")).split(": ", 1)[-1]
            species = str(entry.get("details", "")).split(",", 1)[0]
            if entry.get("active") or entry.get("condition", "").endswith(" fnt"):
                continue
            if wanted in (_showdown_id(name), _showdown_id(species)):
                parts[slot] = f"switch {position}"
                break
    return prefix + ", ".join(parts)


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
