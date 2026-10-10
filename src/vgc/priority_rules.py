"""Effective move priority and the Psychic Terrain priority block (data/conditions.ts, data/moves.ts).

Psychic Terrain's ``onTryHit`` stops a move whose EFFECTIVE priority is above 0.1 from
hitting a grounded target that is not the user's ally (and any move targeting ``self``).
"Effective" means after priority-changing abilities, so Prankster (status moves, +1), Gale
Wings (Flying moves at full HP, +1) and Triage (healing moves, +3) count. Fake Out, Sucker
Punch, Extreme Speed, Aqua Jet etc. have their priority in the move data already.
"""

from __future__ import annotations

from vgc.damage import PokemonState, _is_grounded, to_id

# Psychic Terrain's onTryHit runs once per targeted Pokemon. Moves aimed at the user, its side,
# a whole side or the field (``self``, ``allySide``, ``foeSide``, ``all``, ``allies``,
# ``adjacentAlly*``) never ask a foe to be hit, so a priority boost does not get them blocked.
_TARGETS_WITH_PER_FOE_TRYHIT = frozenset(
    {"normal", "any", "adjacentFoe", "allAdjacentFoes", "allAdjacent", "randomNormal"}
)
_GALE_WINGS_BONUS = 1
_PRANKSTER_BONUS = 1
_TRIAGE_BONUS = 3


def effective_priority(move_data: dict, ability: str | None, at_full_hp: bool) -> int:
    """Move priority after the user's priority-modifying ability."""

    priority = int(move_data.get("priority", 0))
    ability_id = to_id(ability)
    if ability_id == "prankster" and move_data.get("category") == "Status":
        priority += _PRANKSTER_BONUS
    elif ability_id == "galewings" and move_data.get("type") == "Flying" and at_full_hp:
        priority += _GALE_WINGS_BONUS
    elif ability_id == "triage" and (move_data.get("flags") or {}).get("heal"):
        priority += _TRIAGE_BONUS
    return priority


# Moves that fail once the user has made a move action since switching in (data/moves.ts
# onTry; the Champions mod turns them into onDisableMove). Mat Block is `Past` in this mod.
FIRST_TURN_ONLY_MOVES = frozenset({"fakeout", "firstimpression"})


def usable_move_ids(move_ids, mon, restrict: bool) -> list[str]:
    """``move_ids`` minus the first-turn-only moves unless ``mon`` is on its first turn out.

    ``mon`` is a poke-env ``Pokemon`` (``first_turn``: exactly one turn boundary since it came
    in); objects without the attribute are treated as being on their first turn so stubs and
    unknown Pokemon keep their moves.
    """

    moves = list(move_ids)
    if not restrict or mon is None or getattr(mon, "first_turn", True):
        return moves
    return [m for m in moves if to_id(m) not in FIRST_TURN_ONLY_MOVES]


def psychic_terrain_blocks(
    move_data: dict, attacker: PokemonState, defender: PokemonState, terrain: str | None
) -> bool:
    """True when Psychic Terrain stops ``attacker``'s move from hitting FOE ``defender``.

    The caller decides "foe": the user's own allies are never protected. Self-targeting moves
    are exempt by the caller never asking.
    """

    if terrain != "psychic" or move_data.get("target") not in _TARGETS_WITH_PER_FOE_TRYHIT:
        return False
    at_full_hp = attacker.hp_or_max() >= attacker.max_hp()
    if effective_priority(move_data, attacker.ability, at_full_hp) <= 0:
        return False
    return _is_grounded(defender)
