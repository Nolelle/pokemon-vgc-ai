"""Cheap, public-information checks for joint orders that waste a Pokemon's turn.

Three blunders both judges (the fast myopic evaluator and the exact Showdown search)
used to walk into, because neither models the *interaction* that voids the action:

1. ``priority_blocked``: a priority-above-zero move into a target the engine blocks it
   against. Armor Tail / Queenly Majesty / Dazzling on ANY active Pokemon of the target's
   side (the ability protects its holder and the holder's ally, per Showdown's
   ``onFoeTryMove``), or Psychic Terrain on a grounded target. Mold Breaker-style
   attackers ignore the abilities, not the terrain.
2. ``helping_hand_wasted``: Helping Hand when the partner is absent/fainted, switches,
   uses a status move (Protect included), or passes.
3. ``choice_lock_status``: a status move by a Choice-item holder that is not already
   locked. It locks the holder into that status move (Protect, Trick Room, ...). Trick /
   Switcheroo are exempt (they remove the item); an already-locked or single-move slot
   is exempt (the request already restricts it).

Hidden abilities (target side only): a revealed ability is believed. Otherwise the
ability is treated as blocking when EVERY possible ability of the species blocks, or when
the M-C usage-share of blocking abilities for that species is >= ``USAGE_BLOCK_SHARE``.
Nothing here reads private opponent information.

``wasted_action_reasons`` is pure; the penalty and the on/off switch live on
``PolicyConfig.penalize_wasted_actions`` / ``wasted_action_penalty``.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from poke_env.battle.field import Field
from poke_env.battle.move import Move
from poke_env.battle.pokemon import Pokemon

from vgc.config import DATA_DIR
from vgc.data import load_moves, load_species

PRIORITY_BLOCK_ABILITIES = frozenset({"armortail", "queenlymajesty", "dazzling"})
IGNORES_ABILITIES = frozenset({"moldbreaker", "teravolt", "turboblaze"})
CHOICE_ITEMS = frozenset({"choicescarf", "choiceband", "choicespecs"})
# Choice holders may legitimately spend the lock on these (they hand the item away).
CHOICE_LOCK_EXEMPT_MOVES = frozenset({"trick", "switcheroo"})
USAGE_BLOCK_SHARE = 0.5  # usage share of blocking abilities that counts as "has it"
_CHAOS_FILE = "gen9championsvgc2026regmc-1760.json"
_FOE_SINGLE_TARGETS = frozenset({"normal", "any", "adjacentFoe", "randomNormal"})
_FOE_SPREAD_TARGETS = frozenset({"allAdjacentFoes"})


def _id(value: object) -> str:
    return "".join(ch for ch in str(value or "").lower() if ch.isalnum())


@lru_cache(maxsize=1)
def _usage_abilities() -> dict[str, dict[str, float]]:
    """species id -> {ability id: weight}, from the M-C chaos file ({} if absent)."""

    path = DATA_DIR.parent / "usage" / _CHAOS_FILE
    try:
        with path.open() as file:
            data = json.load(file).get("data", {})
    except (OSError, ValueError):
        return {}
    return {_id(name): dict(row.get("Abilities") or {}) for name, row in data.items()}


def _possible_abilities(mon: Pokemon) -> list[str]:
    species = load_species().get(_id(getattr(mon, "species", None))) or {}
    return [_id(a) for a in (species.get("abilities") or {}).values()]


def blocks_priority_probability(mon: Pokemon | None) -> float:
    """1.0 if ``mon`` is believed to hold a priority-blocking ability, else 0.0."""

    if mon is None or getattr(mon, "fainted", False):
        return 0.0
    revealed = _id(getattr(mon, "ability", None))
    if revealed and revealed != "unknown_ability":
        return 1.0 if revealed in PRIORITY_BLOCK_ABILITIES else 0.0
    possible = _possible_abilities(mon)
    if not possible:
        return 0.0
    blocking = [a for a in possible if a in PRIORITY_BLOCK_ABILITIES]
    if not blocking:
        return 0.0
    if len(blocking) == len(possible):
        return 1.0
    usage = _usage_abilities().get(_id(getattr(mon, "species", None)))
    if usage:
        total = sum(usage.values())
        if total > 0 and sum(usage.get(a, 0.0) for a in blocking) / total >= USAGE_BLOCK_SHARE:
            return 1.0
    return 0.0


def _grounded(mon: Pokemon) -> bool:
    types = {_id(t.name if hasattr(t, "name") else t) for t in (mon.types or [])}
    if "flying" in types:
        return False
    if _id(getattr(mon, "ability", None)) == "levitate":
        return False
    if _id(getattr(mon, "item", None)) == "airballoon":
        return False
    return True


def _psychic_terrain(battle: Any) -> bool:
    return Field.PSYCHIC_TERRAIN in (getattr(battle, "fields", None) or {})


def _effective_priority(move_data: dict, battle: Any) -> int:
    priority = int(move_data.get("priority") or 0)
    if move_data.get("id") == "grassyglide" and Field.GRASSY_TERRAIN in (
        getattr(battle, "fields", None) or {}
    ):
        priority += 1
    return priority


def _singles(order: Any) -> list[tuple[int, Any]]:
    pairs = []
    for slot, attr in enumerate(("first_order", "second_order")):
        single = getattr(order, attr, None)
        if single is not None:
            pairs.append((slot, single))
    return pairs


def _priority_blocked(battle: Any, slot: int, single: Any) -> str | None:
    move = single.order
    data = load_moves().get(_id(move.id))
    if not data or _effective_priority(data, battle) <= 0:
        return None
    kind = data.get("target")
    opp = list(getattr(battle, "opponent_active_pokemon", None) or [])
    targets: list[Pokemon] = []
    if kind in _FOE_SPREAD_TARGETS:
        targets = [m for m in opp if m is not None and not m.fainted]
    elif kind in _FOE_SINGLE_TARGETS and single.move_target in (1, 2):
        mon = opp[single.move_target - 1] if len(opp) >= single.move_target else None
        if mon is not None and not mon.fainted:
            targets = [mon]
    if not targets:
        return None
    attacker = (getattr(battle, "active_pokemon", None) or [None, None])[slot]
    ignores = _id(getattr(attacker, "ability", None)) in IGNORES_ABILITIES
    terrain = _psychic_terrain(battle)
    holders = [] if ignores else [m for m in opp if blocks_priority_probability(m) >= 1.0]
    for target in targets:
        if terrain and _grounded(target):
            return f"priority_blocked:{_id(move.id)}:psychic_terrain:{_id(target.species)}"
        if holders:
            return (
                f"priority_blocked:{_id(move.id)}:{_id(holders[0].ability) or 'blocking_ability'}"
                f":{_id(holders[0].species)}"
            )
    return None


def _helping_hand_wasted(battle: Any, slot: int, partner_single: Any) -> str | None:
    active = getattr(battle, "active_pokemon", None) or [None, None]
    partner = active[1 - slot] if len(active) == 2 else None
    if partner is None or partner.fainted or partner_single is None:
        return "helping_hand_wasted:partner_absent"
    target = partner_single.order
    if isinstance(target, Pokemon):
        return "helping_hand_wasted:partner_switches"
    if not isinstance(target, Move):
        return "helping_hand_wasted:partner_passes"
    data = load_moves().get(_id(target.id)) or {}
    if data.get("category") == "Status":
        return f"helping_hand_wasted:partner_status:{_id(target.id)}"
    return None


def _choice_lock_status(battle: Any, slot: int, single: Any) -> str | None:
    data = load_moves().get(_id(single.order.id)) or {}
    if data.get("category") != "Status" or _id(single.order.id) in CHOICE_LOCK_EXEMPT_MOVES:
        return None
    active = getattr(battle, "active_pokemon", None) or [None, None]
    mon = active[slot] if len(active) > slot else None
    if mon is None or _id(getattr(mon, "item", None)) not in CHOICE_ITEMS:
        return None
    from vgc.actions import locked_move_ids  # local: actions imports poke_env heavily

    if slot in locked_move_ids(battle):
        return None
    available = (getattr(battle, "available_moves", None) or [[], []])[slot]
    if len(available) <= 1:
        return None
    return f"choice_lock_status:{_id(single.order.id)}:{_id(mon.item)}"


def wasted_action_reasons(battle: Any, order: Any) -> list[str]:
    """Reasons ``order`` (a DoubleBattleOrder) wastes an action; [] if none are detected."""

    reasons: list[str] = []
    singles = _singles(order)
    by_slot = dict(singles)
    for slot, single in singles:
        if not isinstance(single.order, Move):
            continue
        move_id = _id(single.order.id)
        reason = _priority_blocked(battle, slot, single)
        if reason:
            reasons.append(reason)
        if move_id == "helpinghand":
            reason = _helping_hand_wasted(battle, slot, by_slot.get(1 - slot))
            if reason:
                reasons.append(reason)
        reason = _choice_lock_status(battle, slot, single)
        if reason:
            reasons.append(reason)
    return reasons
