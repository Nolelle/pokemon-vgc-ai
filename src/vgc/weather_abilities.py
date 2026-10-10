"""Canonical weather facts for the quick scorer (`vgc.evaluator`) and the fast search.

Three small, pure things live here so they cannot drift between the two callers:

* ``SETTER_ABILITY_WEATHER`` -- ability id -> weather name (``"sun"|"rain"|"sand"|"snow"``) for
  every weather-setting ability. Derived from `vgc.field_setters.ABILITY_CONDITIONS` (the most
  complete table in the repo) rather than hand-listed, so Mega Tyranitar (Sand Stream) and Mega
  Abomasnow / Mega Froslass (Snow Warning) are not invisible the way they were when the
  evaluator only knew Drought and Drizzle.
* ``BENEFIT_ABILITIES`` / ``team_weather_scores`` -- how many of a team's Pokemon gain from each
  weather, used to SIGN a weather change (does a Mega's weather replace one our team wants?).
* ``weather_adjusted_accuracy`` -- the weather-dependent accuracy of the few legal moves that
  have one (Thunder, Hurricane, Blizzard), read against ``data/champions/moves.json``.

Weather accuracy facts (data/moves.ts ``onModifyMove``, not overridden by the champions mod):
Thunder and Hurricane never miss in rain/Primordial Sea and have 50% accuracy in
sun/Desolate Land; Blizzard never misses in snow (hail/snowscape). Bleakwind/Wildbolt/Sandsear
Storm have the rain rule too but are ``isNonstandard: "Past"`` in this mod, so they are not
listed.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from vgc.damage import to_id
from vgc.data import load_moves, load_species
from vgc.field_setters import ABILITY_CONDITIONS
from vgc.sets import mega_species_id

SETTER_ABILITY_WEATHER: dict[str, str] = {
    ability: value for ability, (kind, value) in ABILITY_CONDITIONS.items() if kind == "weather"
}

# Abilities that gain from a weather being up (setters included below via
# SETTER_ABILITY_WEATHER). Speed doublers, residual heal/chip and accuracy/evasion boosters.
BENEFIT_ABILITIES: dict[str, frozenset[str]] = {
    "sun": frozenset({"chlorophyll", "solarpower", "flowergift"}),
    "rain": frozenset({"swiftswim", "raindish", "dryskin"}),
    "sand": frozenset({"sandrush", "sandforce", "sandveil"}),
    "snow": frozenset({"slushrush", "icebody", "snowcloak"}),
}

# move id -> (weather where it always hits, {weather: accuracy percent} overrides).
_WEATHER_ACCURACY: dict[str, tuple[str | None, dict[str, float]]] = {
    "thunder": ("rain", {"sun": 50.0}),
    "hurricane": ("rain", {"sun": 50.0}),
    "blizzard": ("snow", {}),
}


def weather_adjusted_accuracy(move_id: str | None, weather: str | None) -> float | None:
    """Hit probability of a weather-dependent move under ``weather``, else ``None``.

    ``None`` means "this move's accuracy does not depend on weather" -- callers keep whatever
    they did before. Moves with no listed accuracy (always hit) are also ``None``.
    """

    entry = _WEATHER_ACCURACY.get(to_id(move_id))
    if entry is None:
        return None
    always_hits_in, overrides = entry
    if weather is not None and weather == always_hits_in:
        return 1.0
    if weather in overrides:
        return overrides[weather] / 100.0
    accuracy = (load_moves().get(to_id(move_id)) or {}).get("accuracy", 100)
    return 1.0 if accuracy is True else min(1.0, max(0.0, float(accuracy) / 100.0))


def mon_ability_ids(species_id: str | None, ability: str | None, item: str | None) -> set[str]:
    """Ability ids a Pokemon can field in this battle: its own, plus its Mega forme's when it
    holds that stone (Charizard holding Charizardite Y is a Drought setter in waiting)."""

    ids: set[str] = set()
    if ability:
        ids.add(to_id(ability))
    mega_id = mega_species_id(to_id(species_id), to_id(item)) if species_id else None
    if mega_id is not None:
        mega_ability = to_id(((load_species().get(mega_id) or {}).get("abilities") or {}).get("0"))
        if mega_ability:
            ids.add(mega_ability)
    return ids


def team_weather_scores(mons: Iterable[Any]) -> dict[str, int]:
    """Per weather, how many Pokemon in ``mons`` gain from it (a setter or a benefit ability).

    ``mons`` need ``species``/``ability``/``item``/``fainted`` attributes (poke-env
    ``Pokemon``). Each Pokemon counts once per weather. Fainted ones are skipped.
    """

    scores: dict[str, int] = {}
    for mon in mons:
        if mon is None or getattr(mon, "fainted", False):
            continue
        helped: set[str] = set()
        for ability_id in mon_ability_ids(
            getattr(mon, "species", None), getattr(mon, "ability", None), getattr(mon, "item", None)
        ):
            setter_weather = SETTER_ABILITY_WEATHER.get(ability_id)
            if setter_weather is not None:
                helped.add(setter_weather)
            for weather, abilities in BENEFIT_ABILITIES.items():
                if ability_id in abilities:
                    helped.add(weather)
        for weather in helped:
            scores[weather] = scores.get(weather, 0) + 1
    return scores
