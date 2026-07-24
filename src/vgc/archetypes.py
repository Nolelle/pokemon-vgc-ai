"""Archetype-level structure over the curated meta teams (`vgc.meta.load_meta_teams`).

This is the domain layer for the archetype-tagged self-play pool
(`tools/build_archetype_pool.py`): a plain, reviewable mapping from each of the 10 real
named teams in `data/meta/popular_teams_H8v7TEZcbXo.json` to one of a handful of
archetype labels, plus each archetype's DEFINING CORE -- the small set of species that
make the strategy work and must be present in every team classified under that label.
Everything outside the core is a "flex" slot the pool builder is free to vary.

The assignment below is a domain call, not something inferred by fragile runtime
heuristics -- it is derived from each anchor team's own `name`/`plan`/`common_leads`
text and species list (see each `ArchetypeDef.description`), and is meant to be read and
sanity-checked by a human, not just executed. Some anchors that look distinct by name
share an archetype once you look at what's actually doing the work (e.g. AeroZard, Big
Six, and Charizard Y Toxapex are all "Mega Charizard-Y sun plus Garchomp" underneath);
others (Sand Staraptor, Gardevoir Maushold, Triple Setup Balance) are distinctive enough
to stand alone. This keeps the total at 6 archetypes -- inside the "roughly 5-8" target
-- rather than forcing all 10 anchors into singletons.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from vgc.damage import to_id
from vgc.data import load_species
from vgc.meta import load_meta_teams

# Ability ids (to_id-normalized) that set a persistent field weather immediately upon
# switch-in, mapped to the weather label they set. Includes both the "modern" quartet
# (Drizzle/Drought/Sand Stream/Snow Warning) AND the Primal-only pair (Primordial Sea/
# Desolate Land), since both are present as legal abilities somewhere in the Champions-
# mod roster (`data/champions/species.json`). Deliberately excludes conditional/
# triggered weather abilities that do NOT set weather on switch-in -- e.g. Sand Spit
# (Rock; starts a sandstorm only when the Pokemon is HIT by an attack) isn't a
# persistent archetype identity marker the way Drought/Sand Stream are, so it's not
# treated as an "active weather source" here.
WEATHER_ABILITY_MAP: dict[str, str] = {
    "drizzle": "rain",
    "primordialsea": "rain",
    "drought": "sun",
    "desolateland": "sun",
    "sandstream": "sand",
    "snowwarning": "snow",
}


def weather_mega_stone_map() -> dict[str, str]:
    """Mega Stone item id -> weather label, for every Mega forme in `data/champions/
    species.json` whose (fixed) ability is one of `WEATHER_ABILITY_MAP`'s setters.

    Computed from the exported champions-mod data rather than hardcoded per species --
    this is how e.g. `charizardmegay`'s `Charizardite Y` -> "sun" mapping is discovered,
    with no Charizard-specific code anywhere. The general problem this solves: a flex
    Pokemon's OWN base-forme ability is very often not a weather ability (e.g. base
    Charizard is Blaze/Solar Power), but the mod auto-swaps its ability to the Mega
    forme's the instant it holds that Mega's required item and battles -- so a team
    that merely HOLDS a weather Mega Stone silently sets that weather regardless of
    what ability string the set declares. `tools/build_archetype_pool.py` uses this to
    keep flex slots from smuggling in an off-archetype weather source this way.
    """
    species = load_species()
    stones: dict[str, str] = {}
    for entry in species.values():
        if not entry.get("isMega"):
            continue
        ability_ids = {to_id(name) for name in (entry.get("abilities") or {}).values()}
        weather_hit = sorted(ability_ids & set(WEATHER_ABILITY_MAP))
        if not weather_hit:
            continue
        required_item = entry.get("requiredItem")
        item_id = to_id(required_item) if required_item else None
        if item_id:
            # A single Mega forme has one fixed ability in practice, so `weather_hit`
            # is one element; `sorted(...)[0]` is just a deterministic guard in case a
            # future roster change ever gave one forme two ability slots that both
            # happen to be weather setters.
            stones[item_id] = WEATHER_ABILITY_MAP[weather_hit[0]]
    return stones


def active_weathers(ability_item_pairs: Iterable[tuple[str | None, str | None]]) -> set[str]:
    """The set of distinct active weather labels a team produces, given each of its six
    slots' `(ability_id, item_id)` pair (both already `to_id`-normalized, or `None`).

    Checks both leak mechanisms a flex slot can introduce: an ability directly in
    `WEATHER_ABILITY_MAP`, and a held item that's a weather Mega Stone per
    `weather_mega_stone_map()` (which fires on Mega Evolution regardless of the set's
    OWN declared ability -- see that function's docstring). Used by `tools/
    build_archetype_pool.py`'s flex-exclusion filter and post-generation guard, and by
    `tests/test_archetypes.py` to check an archetype's anchors actually produce its
    declared `weather`.
    """
    stones = weather_mega_stone_map()
    weathers: set[str] = set()
    for ability_id, item_id in ability_item_pairs:
        if ability_id and ability_id in WEATHER_ABILITY_MAP:
            weathers.add(WEATHER_ABILITY_MAP[ability_id])
        if item_id and item_id in stones:
            weathers.add(stones[item_id])
    return weathers


@dataclass(frozen=True)
class ArchetypeDef:
    """One archetype: a label, the anchor team id(s) it was derived from, the defining
    core species every team of this archetype must contain, and its weather identity
    (if any). `description` documents WHY these anchors were grouped and WHY these
    species are the core -- read it before changing the assignment below.
    """

    label: str
    anchor_ids: tuple[str, ...]
    core_species: tuple[str, ...]
    # "sun" | "rain" | "sand" | "snow" | None -- the single persistent weather this
    # archetype's own core is built around, or None if it has no weather identity.
    # `tools/build_archetype_pool.py` uses this to keep flex slots from introducing a
    # DIFFERENT active weather source (see `WEATHER_ABILITY_MAP`/
    # `weather_mega_stone_map` above) -- e.g. a rain team's flex slot must never
    # smuggle in a Drought/Charizardite-Y sun setter.
    weather: str | None
    description: str


# The archetype -> anchor(s) -> core-species assignment. Every `anchor_ids` entry must
# be a real `id` from `data/meta/popular_teams_H8v7TEZcbXo.json`'s `teams` list, and
# every `core_species` entry must be one of that anchor's `species_id`s -- both are
# asserted by `tests/test_archetypes.py`, which is the actual guardrail against this
# list drifting out of sync with the source JSON.
ARCHETYPES: tuple[ArchetypeDef, ...] = (
    ArchetypeDef(
        label="charizard_sun_offense",
        anchor_ids=("aerozard", "charizard_y_toxapex", "big_six"),
        core_species=("charizard", "garchomp"),
        weather="sun",
        description=(
            "Mega Charizard Y sun (Drought / Heat Wave / Weather Ball) is the shared win "
            "condition across all three anchors, always paired with Garchomp -- the "
            "aerozard anchor's own plan text calls 'Charizard plus Garchomp' out by name "
            "as 'another immediate offense lead'. The remaining 4 slots are where the "
            "anchors diverge: AeroZard adds a Trick Room mode (Farigiraf) and Tailwind "
            "(Aerodactyl-Mega), Charizard Y Toxapex adds Toxapex trapping/Venusaur sun "
            "support, and Big Six adds Tailwind (Whimsicott) plus a second Mega (Floette)."
        ),
    ),
    ArchetypeDef(
        label="rain_offense",
        anchor_ids=("double_weather", "rain_screens"),
        core_species=("pelipper", "archaludon", "basculegion"),
        weather="rain",
        description=(
            "Drizzle Pelipper sets rain for a durable Stamina Archaludon attacker and an "
            "Adaptability Basculegion cleaner in both anchors. double_weather layers a "
            "second (sun) weather mode with Charizard/Venusaur on top of this core; "
            "rain_screens instead protects the same rain core with Grimmsnarl dual "
            "screens and a Swift Swim Mega Swampert. Grouped as 'rain', not 'double "
            "weather', because the sun half of double_weather isn't shared with the "
            "other rain anchor."
        ),
    ),
    ArchetypeDef(
        label="anti_meta_lycanroc_tech",
        anchor_ids=("mega_froslass_lycanroc", "mega_blaziken_anti_meta"),
        core_species=("lycanrocdusk", "kingambit", "basculegion"),
        # No weather identity: the core (Lycanroc-Dusk/Kingambit/Basculegion) sets none,
        # and the two anchors don't even agree on a weather Mega (Froslass's snow vs.
        # Blaziken/Venusaur's none) -- so ANY weather setter in a flex slot is off-
        # archetype for this one, not just a different weather than "the" one.
        weather=None,
        description=(
            "Both anchors state an explicit anti-Charizard-Y-sun purpose ('Anti-"
            "Charizard and anti-Big-Six composition' / 'Lycanroc and Ninetales "
            "strengthen the Charizard and Garchomp matchups'), and both build that "
            "answer from the same three species: Tough Claws Lycanroc-Dusk (priority "
            "Rock/Fighting pressure), Kingambit (Defiant setup attacker), and "
            "Basculegion (late-game cleaner). mega_froslass_lycanroc's own Mega is "
            "Froslass (snow/Aurora Veil) or Scovillain; mega_blaziken_anti_meta's is "
            "Blaziken or Venusaur -- the Mega choice is flex here, not core, since the "
            "two anchors don't agree on it."
        ),
    ),
    ArchetypeDef(
        label="triple_setup_balance",
        anchor_ids=("triple_setup_balance",),
        core_species=("blastoise", "delphox", "kingambit"),
        weather=None,
        description=(
            "Named directly from the anchor's own plan text: 'set up Blastoise, "
            "Delphox, or Kingambit'. Shell Smash Mega Blastoise, Nasty Plot Mega "
            "Delphox, and Swords Dance Kingambit are three interchangeable setup win "
            "conditions behind Fake Out (Sneasler/Incineroar) and Sinistcha redirection."
        ),
    ),
    ArchetypeDef(
        label="sand_offense",
        anchor_ids=("sand_staraptor",),
        core_species=("tyranitar", "excadrill"),
        weather="sand",
        description=(
            "Named directly from the anchor's own plan text: 'Sand offense built around "
            "Tyranitar and Excadrill'. Sand Stream Mega Tyranitar sets the weather for a "
            "Sand Rush Excadrill sweeper; Staraptor/Milotic/Sinistcha/Gholdengo are "
            "support/flex, not what the plan names as the core idea."
        ),
    ),
    ArchetypeDef(
        label="gardevoir_maushold",
        anchor_ids=("gardevoir_maushold",),
        core_species=("gardevoir", "maushold"),
        weather=None,
        description=(
            "The anchor's own plan text: 'Maushold redirects for Gardevoir or "
            "Kingambit ... Talonflame supplies Tailwind, while Scarf Basculegion cleans "
            "late' -- Friend Guard Maushold's Follow Me redirection exists to protect "
            "Mega Gardevoir's spread attacks (Kingambit is a secondary beneficiary, not "
            "load-bearing enough on its own to be core here, unlike in "
            "triple_setup_balance where it's explicitly one of the named setup wins)."
        ),
    ),
)

ARCHETYPE_BY_LABEL: dict[str, ArchetypeDef] = {archetype.label: archetype for archetype in ARCHETYPES}

# anchor team id -> the one archetype label it was assigned to. Built from ARCHETYPES
# rather than hand-duplicated so the two structures can never drift apart.
ANCHOR_ID_TO_LABEL: dict[str, str] = {
    anchor_id: archetype.label for archetype in ARCHETYPES for anchor_id in archetype.anchor_ids
}


def anchors_for(label: str) -> tuple[dict[str, Any], ...]:
    """The real anchor team dict(s) (from `vgc.meta.load_meta_teams`) assigned to
    `label`, in `ARCHETYPES` order. Raises `KeyError` for an unknown label.
    """
    archetype = ARCHETYPE_BY_LABEL[label]
    teams_by_id = {team["id"]: team for team in load_meta_teams()}
    return tuple(teams_by_id[anchor_id] for anchor_id in archetype.anchor_ids)


def _set_key(entry: dict[str, Any]) -> tuple[Any, ...]:
    """A hashable identity for a curated set entry, used to deduplicate near-identical
    core sets across an archetype's multiple anchors (see `core_set_options`).
    """
    return (
        entry.get("battle_form_id"),
        entry["item_id"],
        entry["ability_id"],
        entry["nature_id"],
        tuple(entry["move_ids"]),
    )


def core_set_options(archetype: ArchetypeDef) -> dict[str, list[dict[str, Any]]]:
    """Per core species, every DISTINCT curated set (item/ability/nature/moves) used for
    it across `archetype`'s anchor(s), in anchor order. A multi-anchor archetype's
    anchors often don't state identical sets for a shared core species (e.g. the three
    charizard_sun_offense anchors give Charizard three slightly different movesets) --
    this collects all of them as equally "known-good" options rather than arbitrarily
    picking just the first anchor's version, so `tools/build_archetype_pool.py` can
    sample across that real variety when building within-archetype variants.
    """
    options: dict[str, list[dict[str, Any]]] = {species_id: [] for species_id in archetype.core_species}
    seen: dict[str, set[tuple[Any, ...]]] = {species_id: set() for species_id in archetype.core_species}
    for anchor in anchors_for(archetype.label):
        for entry in anchor["sets"]:
            species_id = entry["species_id"]
            if species_id not in options:
                continue
            key = _set_key(entry)
            if key in seen[species_id]:
                continue
            seen[species_id].add(key)
            options[species_id].append(entry)
    return options


def classify_team(species_ids: Iterable[Any]) -> str | None:
    """The archetype label whose `core_species` are all present in `species_ids`, or
    `None` if no archetype's core is fully contained.

    Deterministic and simple by design (per-core containment, first match in
    `ARCHETYPES` order) rather than a scored/fuzzy classifier -- the cores were chosen
    specifically so they don't overlap as subsets of each other (see
    `tests/test_archetypes.py`), so match order shouldn't matter in practice.
    `species_ids` entries are `to_id`-normalized before comparison, so callers may pass
    raw display names or already-normalized ids interchangeably.
    """
    normalized = {to_id(species_id) for species_id in species_ids if to_id(species_id)}
    for archetype in ARCHETYPES:
        if set(archetype.core_species).issubset(normalized):
            return archetype.label
    return None
