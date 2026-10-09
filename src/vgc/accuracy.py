"""Hit probability of a damaging move (sim/battle-actions.ts ``hitStepAccuracy``).

Showdown's order, mirrored here:

1. ``ModifyAccuracy`` event: the base accuracy, scaled by one chained 4096ths modifier from the
   target's abilities/items (Sand Veil / Snow Cloak in their weather x0.8, Bright Powder x0.9)
   and the source's (Compound Eyes x1.3, Hustle on physical moves x0.8, Wide Lens x1.1, Zoom Lens
   x1.2 when the target has already moved).
2. Accuracy / evasion stages, as one net stage clamped to [-6, 6]: ``acc * (3 + s) / 3`` for
   ``s > 0`` and ``acc * 3 / (3 - s)`` for ``s < 0``. Chip Away / Darkest Lariat / Sacred Sword /
   Nihil Light and Keen Eye ignore the target's evasion; Unaware zeroes the opposing side's
   relevant stage.
3. ``Accuracy`` event: No Guard on either side makes the move always hit.

``accuracy: true`` moves (and rain-Thunder/Hurricane, snow-Blizzard) skip steps 1-2. The
Champions mod does not change any of these rules; its per-move accuracy overrides are already in
``data/champions/moves.json``. Not modelled: Tangled Feet (confusion is not in ``PokemonState``),
Victory Star, Gravity, Micle Berry, and OHKO moves.
"""

from __future__ import annotations

from vgc.damage import PokemonState, to_id
from vgc.weather_abilities import weather_adjusted_accuracy

_IGNORE_EVASION_MOVES = frozenset({"chipaway", "darkestlariat", "sacredsword", "nihillight"})

# 4096ths numerators of the chained modifiers (data/abilities.ts, data/items.ts).
_COMPOUND_EYES = 5325
_HUSTLE_PHYSICAL = 3277
_WIDE_LENS = 4505
_ZOOM_LENS = 4915
_SAND_VEIL = _SNOW_CLOAK = 3277
_BRIGHT_POWDER = 3686


def _chain(modifier: int, numerator: int) -> int:
    """``battle.chainModify([numerator, 4096])`` on a modifier held in 4096ths."""

    return (modifier * numerator + 2048) >> 12


def hit_probability(
    move_data: dict,
    attacker: PokemonState,
    defender: PokemonState,
    weather: str | None = None,
    *,
    target_moves_first: bool | None = None,
) -> float:
    """Probability ``attacker``'s move connects with ``defender`` (1.0 for sure-hit moves).

    ``target_moves_first`` feeds Zoom Lens (x1.2 only when the target has already moved before
    the user); ``None`` means unknown and the item is ignored.
    """

    move_id = to_id(move_data.get("id"))
    accuracy = move_data.get("accuracy", 100)
    if accuracy is True:
        return 1.0
    base = float(accuracy)
    weather_accuracy = weather_adjusted_accuracy(move_id, weather)
    if weather_accuracy is not None:
        if weather_accuracy >= 1.0:
            return 1.0  # the move's accuracy became `true` (rain Thunder, snow Blizzard)
        base = weather_accuracy * 100.0
    if attacker.ability == "noguard" or defender.ability == "noguard":
        return 1.0

    modifier = 4096
    if attacker.ability == "compoundeyes":
        modifier = _chain(modifier, _COMPOUND_EYES)
    if attacker.ability == "hustle" and move_data.get("category") == "Physical":
        modifier = _chain(modifier, _HUSTLE_PHYSICAL)
    if attacker.item == "widelens":
        modifier = _chain(modifier, _WIDE_LENS)
    if attacker.item == "zoomlens" and target_moves_first:
        modifier = _chain(modifier, _ZOOM_LENS)
    if defender.ability == "sandveil" and weather == "sand":
        modifier = _chain(modifier, _SAND_VEIL)
    if defender.ability == "snowcloak" and weather == "snow":
        modifier = _chain(modifier, _SNOW_CLOAK)
    if defender.item == "brightpowder":
        modifier = _chain(modifier, _BRIGHT_POWDER)
    accuracy_value = (int(base * modifier) + 2047) // 4096 if modifier != 4096 else int(base)

    boost = 0
    if defender.ability != "unaware":
        boost = max(-6, min(6, attacker.boost_stage("accuracy")))
    ignores_evasion = move_id in _IGNORE_EVASION_MOVES or attacker.ability == "keeneye"
    if not ignores_evasion and attacker.ability != "unaware":
        boost = max(-6, min(6, boost - defender.boost_stage("evasion")))
    if boost > 0:
        accuracy_value = accuracy_value * (3 + boost) // 3
    elif boost < 0:
        accuracy_value = accuracy_value * 3 // (3 - boost)
    return max(0.0, min(1.0, accuracy_value / 100.0))
