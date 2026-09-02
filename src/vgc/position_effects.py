"""Whether a volatile or side condition HELPS or HURTS whoever is carrying it.

`vgc.rl.exact_search._position_value` used to score both with `len(...)`, which is
sign-blind: our own Leech Seed, our own Stealth Rock, a Substitute we set up, and a
Tailwind we set up all scored the same +`exact_search_effect_weight`. Being crippled and
setting up were worth identical points, and because the value function is
`our_side - opponent_side` the error ran backwards in both directions at once -- the
search read walking into a Leech Seed as good for us, and read landing a Taunt on the
opponent as bad for us.

The sign is DERIVED from the committed Champions move data rather than hand-listed, so a
format change reaches it through `tools/export_champions_data.py` instead of silently
going stale:

    a volatile/side condition applied by a FOE-targeting move  -> hurts its holder
    a volatile/side condition applied by a SELF/ALLY-targeting move -> helps its holder

That rule reproduces `vgc.mechanics_state._LAYERED_SIDE_CONDITIONS` (spikes, toxicspikes,
stealthrock, stickyweb) exactly, from move targets alone, which is the cross-check that it
is sound -- see `tests/test_position_effects.py`.

Two things the rule cannot see, handled explicitly below and nowhere else:

- Volatiles no legal move applies (perish counters, trapping, ability-driven buffs).
- Volatiles the rule mis-signs because they are engine bookkeeping rather than an effect
  on the holder (`sparklingaria` marks targets so their burn is cured -- the move targets
  a foe, but carrying the mark is not a bad thing).

Anything this module cannot confidently sign is NEUTRAL (0), not +1. That is deliberate
and is itself a correctness improvement: the effect vocabulary is dominated by one-shot
ability-activation markers (`aftermath`, `dancer`, `ironbarbs`, `quickdraw`, ...) that are
not persistent position advantages at all, and the old `len()` paid full weight for every
one of them.

Signs only. No magnitudes and no tunable weights live here -- `exact_search_effect_weight`
keeps its frozen value and this module only decides which way it points.
"""

from __future__ import annotations

from functools import lru_cache

from vgc.damage import to_id
from vgc.data import load_moves

# Move `target` values that aim at an opponent. A volatile these apply lands on someone
# the user wants to be worse off, so carrying it is bad for the carrier.
_FOE_TARGETS = frozenset(
    {
        "normal",
        "adjacentFoe",
        "allAdjacentFoes",
        "any",
        "randomNormal",
        "allAdjacent",
        "foeSide",
        "scripted",
    }
)
# Move `target` values that aim at the user's own side. `all` (Perish Song, Court Change)
# is deliberately absent: it hits everyone, so it signs nothing.
_ALLY_TARGETS = frozenset(
    {
        "self",
        "adjacentAlly",
        "adjacentAllyOrSelf",
        "allySide",
        "allies",
        "allyTeam",
    }
)

HELPS_HOLDER = 1
NEUTRAL = 0
HURTS_HOLDER = -1

# Volatiles no legal Champions move applies via `volatileStatus`, so the derivation cannot
# see them. Each is signed individually, and only where the direction is not arguable.
_SUPPLEMENT: dict[str, int] = {
    # Perish Song counts down to a guaranteed faint on whoever carries it. The move's
    # target is `all`, so direction has to be stated here rather than derived.
    "perish0": HURTS_HOLDER,
    "perish1": HURTS_HOLDER,
    "perish2": HURTS_HOLDER,
    "perish3": HURTS_HOLDER,
    # Set by trapping abilities (Arena Trap, Shadow Tag, Magnet Pull) rather than a move.
    "trapped": HURTS_HOLDER,
    # Slow Start halves Attack and Speed for five turns. Ability-driven, unambiguous.
    "slowstart": HURTS_HOLDER,
    # Paradox-ability boosts. Ability-driven, and common enough in this format to matter:
    # poke-env reports both the bare marker and the per-stat variant.
    "protosynthesis": HELPS_HOLDER,
    "protosynthesisatk": HELPS_HOLDER,
    "protosynthesisdef": HELPS_HOLDER,
    "protosynthesisspa": HELPS_HOLDER,
    "protosynthesisspd": HELPS_HOLDER,
    "protosynthesisspe": HELPS_HOLDER,
    "quarkdrive": HELPS_HOLDER,
    "quarkdriveatk": HELPS_HOLDER,
    "quarkdrivedef": HELPS_HOLDER,
    "quarkdrivespa": HELPS_HOLDER,
    "quarkdrivespd": HELPS_HOLDER,
    "quarkdrivespe": HELPS_HOLDER,
}

# Derivation results that are wrong because the volatile is engine bookkeeping, not an
# effect on the holder. Keep this list short and justify every entry.
_OVERRIDES: dict[str, int] = {
    # Sparkling Aria marks the Pokemon it hit so their burn is cured after the move
    # resolves. The move targets foes, so the rule calls it harmful; carrying the mark is
    # not harmful, and by the time anyone reads the snapshot it means nothing.
    "sparklingaria": NEUTRAL,
    # Applied to the USER by Outrage/Petal Dance/Thrash (attacking, but locked in and
    # confused afterwards) and Uproar/Roost (a trade, not a gain). Genuinely two-sided,
    # so the honest score is zero rather than a guess in either direction.
    "lockedmove": NEUTRAL,
    "uproar": NEUTRAL,
    "roost": NEUTRAL,
}


def _sign_from_target(target: str) -> int:
    if target in _FOE_TARGETS:
        return HURTS_HOLDER
    if target in _ALLY_TARGETS:
        return HELPS_HOLDER
    return NEUTRAL


def _record(table: dict[str, set[int]], effect: str | None, sign: int) -> None:
    effect_id = to_id(effect)
    if not effect_id or sign == NEUTRAL:
        return
    table.setdefault(effect_id, set()).add(sign)


@lru_cache(maxsize=1)
def _derived() -> tuple[dict[str, int], dict[str, int]]:
    """Sign every volatile and side condition a LEGAL Champions move can apply.

    Illegal (`isNonstandard`) moves are skipped, which is most of what a vanilla gen9
    derivation would pick up -- Octolock, Telekinesis, Embargo, Nightmare, Tar Shot,
    Obstruct, Burning Bulwark, Silk Trap, Mist, Lucky Chant, Crafty Shield and Mat Block
    are all `"Past"` in this mod and can never appear.
    """

    volatiles: dict[str, set[int]] = {}
    sides: dict[str, set[int]] = {}
    for move in load_moves().values():
        if move.get("isNonstandard"):
            continue
        sign = _sign_from_target(str(move.get("target") or ""))
        _record(volatiles, move.get("volatileStatus"), sign)
        for effect in move.get("secondaryVolatileStatuses") or ():
            _record(volatiles, effect, sign)
        _record(sides, move.get("sideCondition"), sign)
        # A slot condition (Wish, Healing Wish) is left for an incoming ally, so it is
        # always the user's own side regardless of how the move's target reads.
        _record(sides, move.get("slotCondition"), HELPS_HOLDER)
        # `self`-volatiles land on the user whichever way the move points, so the move's
        # target says nothing about them. Hyper Beam targets a foe but `mustrecharge`
        # lands on us. These are signed in _SUPPLEMENT/_OVERRIDES or left neutral.
        for effect in [
            move.get("selfVolatileStatus"),
            *(move.get("secondarySelfVolatileStatuses") or ()),
        ]:
            _record(volatiles, effect, NEUTRAL)

    def collapse(table: dict[str, set[int]]) -> dict[str, int]:
        # A volatile applied by both foe-targeting and ally-targeting legal moves has no
        # single direction, so it scores zero rather than whichever move was seen last.
        return {key: next(iter(signs)) for key, signs in table.items() if len(signs) == 1}

    return collapse(volatiles), collapse(sides)


def effect_polarity(effect: object) -> int:
    """+1 if this volatile helps whoever carries it, -1 if it hurts them, 0 if unsigned."""

    effect_id = to_id(getattr(effect, "id", None) or effect)
    if not effect_id:
        return NEUTRAL
    if effect_id in _OVERRIDES:
        return _OVERRIDES[effect_id]
    if effect_id in _SUPPLEMENT:
        return _SUPPLEMENT[effect_id]
    return _derived()[0].get(effect_id, NEUTRAL)


def side_condition_polarity(effect: object) -> int:
    """+1 if this side condition helps the side carrying it, -1 if it hurts them."""

    effect_id = to_id(getattr(effect, "id", None) or effect)
    if not effect_id:
        return NEUTRAL
    if effect_id in _OVERRIDES:
        return _OVERRIDES[effect_id]
    return _derived()[1].get(effect_id, NEUTRAL)
