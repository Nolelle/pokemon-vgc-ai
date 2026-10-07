"""Cheap, public-information checks for joint orders that waste or hobble a turn.

Blunders both judges (the fast myopic evaluator and the exact Showdown search) used to
walk into, because neither models the *interaction* that voids the action:

1. ``priority_blocked``: a priority-above-zero move into a target the engine blocks it
   against. Armor Tail / Queenly Majesty / Dazzling on ANY active foe of the attacker
   (the ability protects its holder and the holder's ally, per Showdown's
   ``onFoeTryMove``), or Psychic Terrain on a grounded target. Mold Breaker-style
   attackers and Neutralizing Gas suppress the abilities, not the terrain.
2. ``helping_hand_wasted``: Helping Hand when the partner is absent/fainted, switches,
   uses a status move (Protect included), or passes.
3. ``choice_lock_status``: a status move by an unlocked Choice-item holder locks it into
   that move. Trick / Switcheroo are exempt; an already-locked or single-move slot is
   exempt (the request already restricts it).
4. ``trick_to_ally``: Trick / Switcheroo onto our own ally that holds no item or has
   Unburden active (it loses its 2x Speed, and gets Choice-locked).

Hidden abilities (target side only): a revealed ability is certain. Otherwise the
probability of a blocker is 1 if EVERY possible ability of the species blocks, else the
M-C usage share of blocking abilities for that species (0 if the chaos file has none);
the priority-block cost scales with it. Nothing here reads private opponent information.

Also modeled, from public state only: our own switch-ins in the SAME order change the
terrain before moves (Surge abilities), Prankster / Gale Wings add priority, Grassy
Glide needs a grounded user, and Magnet Rise / Telekinesis / Gravity / Smack Down / Iron
Ball change who is grounded. Skipped (not exposed cheaply): Prankster failing into Dark
types, a foe's own switch-in setting terrain, two different Surge switch-ins (speed
order: treated as unknown terrain).

``wasted_action_reasons`` is pure; ``wasted_action_cost`` turns the findings into points
using ``PolicyConfig.wasted_action_penalty`` / ``choice_lock_status_penalty`` /
``trick_to_ally_penalty``. The costs are strategic (hundreds of points at most), well
below a game result, so a confirmed win is never outranked by a flagged order.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from poke_env.battle.effect import Effect
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
MIN_REPORTED_PROBABILITY = 0.05  # below this a hidden blocker is not worth reporting
_CHAOS_FILE = "gen9championsvgc2026regmc-1760.json"
_FOE_SINGLE_TARGETS = frozenset({"normal", "any", "adjacentFoe", "randomNormal"})
_FOE_SPREAD_TARGETS = frozenset({"allAdjacentFoes"})
_FIELD_TERRAIN = {
    Field.PSYCHIC_TERRAIN: "psychic",
    Field.GRASSY_TERRAIN: "grassy",
    Field.ELECTRIC_TERRAIN: "electric",
    Field.MISTY_TERRAIN: "misty",
}
_SURGE_ABILITIES = {
    "psychicsurge": "psychic",
    "grassysurge": "grassy",
    "electricsurge": "electric",
    "hadronengine": "electric",
    "mistysurge": "misty",
}


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
    """Probability that ``mon`` holds a priority-blocking ability (revealed = 0 or 1)."""

    if mon is None or getattr(mon, "fainted", False):
        return 0.0
    revealed = _id(getattr(mon, "ability", None))
    if revealed and revealed != "unknown_ability":
        return 1.0 if revealed in PRIORITY_BLOCK_ABILITIES else 0.0
    possible = _possible_abilities(mon)
    blocking = [a for a in possible if a in PRIORITY_BLOCK_ABILITIES]
    if not blocking:
        return 0.0
    if len(blocking) == len(possible):
        return 1.0
    usage = _usage_abilities().get(_id(getattr(mon, "species", None)))
    if usage:
        total = sum(usage.values())
        if total > 0:
            return min(1.0, sum(usage.get(a, 0.0) for a in blocking) / total)
    return 0.0


def _fields(battle: Any) -> Any:
    return getattr(battle, "fields", None) or {}


def _grounded(mon: Pokemon, battle: Any) -> bool:
    effects = getattr(mon, "effects", None) or {}
    if Field.GRAVITY in _fields(battle) or Effect.SMACK_DOWN in effects:
        return True
    if _id(getattr(mon, "item", None)) == "ironball":
        return True
    if Effect.MAGNET_RISE in effects or Effect.TELEKINESIS in effects:
        return False
    types = {_id(t.name if hasattr(t, "name") else t) for t in (mon.types or [])}
    if "flying" in types:
        return False
    if _id(getattr(mon, "ability", None)) == "levitate":
        return False
    if _id(getattr(mon, "item", None)) == "airballoon":
        return False
    return True


def _singles(order: Any) -> list[tuple[int, Any]]:
    pairs = []
    for slot, attr in enumerate(("first_order", "second_order")):
        single = getattr(order, attr, None)
        if single is not None:
            pairs.append((slot, single))
    return pairs


def terrain_after_switch_ins(battle: Any, order: Any) -> str | None:
    """Terrain when moves resolve: switches happen first, so our Surge switch-ins count.

    Two different Surge switch-ins in one order resolve by speed, which we do not model:
    the result is "unknown" (None) so no terrain-dependent rule fires on a guess.
    """

    base = next((name for f, name in _FIELD_TERRAIN.items() if f in _fields(battle)), None)
    surges = {
        _SURGE_ABILITIES[_id(getattr(single.order, "ability", None))]
        for _slot, single in _singles(order)
        if isinstance(single.order, Pokemon)
        and _id(getattr(single.order, "ability", None)) in _SURGE_ABILITIES
    }
    if len(surges) == 1:
        return next(iter(surges))
    if len(surges) > 1:
        return None
    return base


def _effective_priority(
    move_data: dict, attacker: Pokemon | None, battle: Any, terrain: str | None
) -> int:
    priority = int(move_data.get("priority") or 0)
    ability = _id(getattr(attacker, "ability", None))
    if (
        move_data.get("id") == "grassyglide"
        and terrain == "grassy"
        and attacker is not None
        and _grounded(attacker, battle)
    ):
        priority += 1
    if ability == "prankster" and move_data.get("category") == "Status":
        priority += 1
    if (
        ability == "galewings"
        and _id(move_data.get("type")) == "flying"
        and attacker is not None
        and (getattr(attacker, "current_hp_fraction", 0.0) or 0.0) >= 1.0
    ):
        priority += 1
    return priority


def _neutralizing_gas(battle: Any) -> bool:
    mons = list(getattr(battle, "active_pokemon", None) or []) + list(
        getattr(battle, "opponent_active_pokemon", None) or []
    )
    return any(
        m is not None and not m.fainted and _id(getattr(m, "ability", None)) == "neutralizinggas"
        for m in mons
    )


def _priority_blocked(
    battle: Any, slot: int, single: Any, terrain: str | None
) -> tuple[str, float] | None:
    """(reason, probability the move is voided), or None."""

    move = single.order
    data = load_moves().get(_id(move.id))
    attacker = (getattr(battle, "active_pokemon", None) or [None, None])[slot]
    if not data or _effective_priority(data, attacker, battle, terrain) <= 0:
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
    for target in targets:
        if terrain == "psychic" and _grounded(target, battle):
            return f"priority_blocked:{_id(move.id)}:psychic_terrain:{_id(target.species)}", 1.0
    if _id(getattr(attacker, "ability", None)) in IGNORES_ABILITIES or _neutralizing_gas(battle):
        return None
    p_none = 1.0
    for holder in opp:
        p_none *= 1.0 - blocks_priority_probability(holder)
    probability = 1.0 - p_none
    if probability < MIN_REPORTED_PROBABILITY:
        return None
    return f"priority_blocked:{_id(move.id)}:blocking_ability:p={probability:.2f}", probability


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


def _trick_to_ally(battle: Any, slot: int, single: Any) -> str | None:
    """Trick / Switcheroo onto OUR OWN ally whose item slot makes the swap bad.

    The ally receives a Choice item (it becomes locked), and an Unburden ally that has
    already used its item loses its 2x Speed the moment it holds one again (and
    Acrobatics loses its doubled power). An ally with no item at all is the same case.
    """

    if _id(single.order.id) not in CHOICE_LOCK_EXEMPT_MOVES or single.move_target not in (-1, -2):
        return None
    active = getattr(battle, "active_pokemon", None) or [None, None]
    ally = active[1 - slot] if len(active) == 2 else None
    if ally is None or ally.fainted:
        return None
    item = _id(getattr(ally, "item", None))
    held = bool(item) and item not in ("unknownitem", "noitem")
    if held:
        return None
    if _id(getattr(ally, "ability", None)) == "unburden":
        return f"trick_to_ally:{_id(single.order.id)}:unburden_active:{_id(ally.species)}"
    return f"trick_to_ally:{_id(single.order.id)}:ally_has_no_item:{_id(ally.species)}"


def wasted_action_findings(battle: Any, order: Any) -> list[tuple[str, str, float]]:
    """(reason, kind, probability) per flagged action; ``kind`` picks the config weight."""

    findings: list[tuple[str, str, float]] = []
    singles = _singles(order)
    by_slot = dict(singles)
    terrain = terrain_after_switch_ins(battle, order)
    for slot, single in singles:
        if not isinstance(single.order, Move):
            continue
        blocked = _priority_blocked(battle, slot, single, terrain)
        if blocked:
            findings.append((blocked[0], "wasted", blocked[1]))
        if _id(single.order.id) == "helpinghand":
            reason = _helping_hand_wasted(battle, slot, by_slot.get(1 - slot))
            if reason:
                findings.append((reason, "wasted", 1.0))
        reason = _choice_lock_status(battle, slot, single)
        if reason:
            findings.append((reason, "choice", 1.0))
        reason = _trick_to_ally(battle, slot, single)
        if reason:
            findings.append((reason, "trick_ally", 1.0))
    return findings


def wasted_action_reasons(battle: Any, order: Any) -> list[str]:
    """Reasons ``order`` (a DoubleBattleOrder) wastes an action; [] if none are detected."""

    return [reason for reason, _kind, _p in wasted_action_findings(battle, order)]


def wasted_action_cost(battle: Any, order: Any, config: Any) -> tuple[float, list[str]]:
    """(points to subtract, reasons) for ``order`` under the config's strategic weights."""

    weights = {
        "wasted": config.wasted_action_penalty,
        "choice": config.choice_lock_status_penalty,
        "trick_ally": config.trick_to_ally_penalty,
    }
    findings = wasted_action_findings(battle, order)
    cost = sum(weights[kind] * probability for _reason, kind, probability in findings)
    return cost, [reason for reason, _kind, _p in findings]
