"""Measured weather/terrain value of one of OUR Pokemon sets (the "plan value").

Weather, terrain, Tailwind and Trick Room are worth what they ENABLE for a team's plan:
rain lets Archaludon fire Electro Shot (a charge move) in one turn at +1 Sp. Atk, sun does
that for Solar Beam, rain/snow make Thunder/Hurricane/Blizzard never miss, Psychic Terrain
makes Expanding Force hit both foes at 1.5x, Electric Terrain doubles Rising Voltage, Terrain
Pulse and Weather Ball change type and power. None of that is in `data/champions/moves.json`;
it lives in Showdown callbacks. `vgc.field_control`'s old fit term (base power x the generic
weather/terrain modifiers) therefore values rain-for-Archaludon at about zero.

This module replaces that guess, for OUR side, with damage MEASURED by the real engine:

* `tools/plan_value_probe.mjs` plays one set against reference foes, for each of the 25
  weather x terrain conditions, two consecutive turns repeating each damaging move, and
  returns the mean damage as % of the reference foe's max HP (its docstring says how the
  condition is held and how luck is removed).
* `tools/build_plan_value_cache.py` runs it over every set in a list of packed teams and
  stores, per set, the best move's per-turn damage in each (condition, profile) plus the gain
  over the bare field. The result is the committed `data/usage/plan_value_cache.json`.
* `register_own_team(packed_team)` loads those entries for the team a player was given, and
  `lookup(species_id, moves)` is what `vgc.field_control` calls for each of our Pokemon.

Profiles (what the foe side looks like):

    single    one grounded neutral foe
    double    two grounded neutral foes (spread moves and Expanding Force hit both; the engine
              applies the spread reduction; % is summed over both)
    airborne  one Levitate foe, which captures the grounded-only terrain effects

Gains are in %-of-max-HP per turn of the reference foe, the same currency as
`vgc.field_control`'s F_t. They are a property of OUR set against a reference foe -- not of
the actual opponent -- which is the right grain for "how much is this condition worth to us".

Cache key: sha1 of (canonical set, probe version, Showdown pinned commit), so a change to the
probe or to the engine invalidates entries instead of silently serving old numbers.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from vgc.config import DATA_DIR
from vgc.data import load_mechanics_catalog, load_moves, load_species
from vgc.team_scope import resolve_table, team_key

# Bump when the probe's measurement changes in any way that moves numbers.
PROBE_VERSION = 1

WEATHERS = ("none", "sunnyday", "raindance", "sandstorm", "snowscape")
TERRAINS = ("none", "electricterrain", "grassyterrain", "psychicterrain", "mistyterrain")
CONDITIONS: tuple[tuple[str, str], ...] = tuple((w, t) for w in WEATHERS for t in TERRAINS)
PROFILES = ("single", "double", "airborne")
PROFILE_INDEX = {name: index for index, name in enumerate(PROFILES)}

# `vgc.field_control` names its conditions sun/rain/sand/snow and electric/grassy/...
_FIELD_WEATHER = {None: "none", "sun": "sunnyday", "rain": "raindance", "sand": "sandstorm",
                  "snow": "snowscape"}
_FIELD_TERRAIN = {None: "none", "electric": "electricterrain", "grassy": "grassyterrain",
                  "psychic": "psychicterrain", "misty": "mistyterrain"}
_CONDITION_INDEX = {condition: index for index, condition in enumerate(CONDITIONS)}

CACHE_PATH = DATA_DIR.parent / "usage" / "plan_value_cache.json"
PROBE_SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "plan_value_probe.mjs"
DEFAULT_SEEDS = (1, 2, 3, 4)


def condition_index(weather: str | None, terrain: str | None) -> int:
    """Index into the 25-condition arrays for `vgc.field_control`'s weather/terrain names."""
    return _CONDITION_INDEX[(_FIELD_WEATHER[weather], _FIELD_TERRAIN[terrain])]


def to_id(value: str | None) -> str:
    return "".join(ch for ch in (value or "").lower() if ch.isalnum())


# --- packed sets -------------------------------------------------------------------------


@dataclass(frozen=True)
class PackedSet:
    """The parts of one packed set that decide what it does, in canonical form."""

    species_id: str
    item_id: str
    ability_id: str
    nature: str
    evs: tuple[int, ...]  # Stat Points: hp, atk, def, spa, spd, spe
    level: int
    moves: tuple[str, ...]  # sorted ids; order never changes behavior
    packed: str  # the original single-set text, what the probe is given

    def canonical(self) -> str:
        return "|".join(
            [
                self.species_id,
                self.item_id,
                self.ability_id,
                self.nature,
                ",".join(str(point) for point in self.evs),
                str(self.level),
                ",".join(self.moves),
            ]
        )


def parse_packed_set(text: str) -> PackedSet:
    fields = text.strip().split("|")
    fields += [""] * (12 - len(fields))
    evs = [int(part) if part.strip() else 0 for part in (fields[6].split(",") + [""] * 6)[:6]]
    level = int(fields[10]) if fields[10].strip().isdigit() else 50
    return PackedSet(
        species_id=to_id(fields[1] or fields[0]),
        item_id=to_id(fields[2]),
        ability_id=to_id(fields[3]),
        nature=to_id(fields[5]) or "serious",
        evs=tuple(evs),
        level=level,
        moves=tuple(sorted(to_id(move) for move in fields[4].split(",") if move.strip())),
        packed=text.strip(),
    )


def parse_packed_team(packed_team: str) -> list[PackedSet]:
    return [parse_packed_set(part) for part in packed_team.strip().split("]") if part.strip()]


@lru_cache(maxsize=1)
def showdown_commit() -> str:
    """The Showdown commit the mechanics catalog pins (what the cache must have been built on)."""
    return str(load_mechanics_catalog()["generated_from"]["showdown_commit"])


def set_key(pset: PackedSet) -> str:
    material = json.dumps([pset.canonical(), PROBE_VERSION, showdown_commit()])
    return hashlib.sha1(material.encode()).hexdigest()[:16]


@lru_cache(maxsize=None)
def mega_species_id(species_id: str, item_id: str) -> str | None:
    """The Mega form this species becomes holding this stone, per the exported data."""
    if not item_id:
        return None
    species = load_species()
    base = species.get(species_id) or {}
    base_id = to_id(base.get("baseSpecies") or species_id)
    fallback = None
    for candidate_id, data in species.items():
        if not data.get("isMega"):
            continue
        stones = {to_id(data.get("requiredItem"))} | {
            to_id(stone) for stone in (data.get("requiredItems") or [])
        }
        if item_id not in stones:
            continue
        changes_from = to_id(data.get("changesFrom") or "")
        if changes_from == species_id:
            return candidate_id
        if to_id(data.get("baseSpecies")) == base_id and fallback is None:
            fallback = candidate_id
    return fallback


def damaging_moves(pset: PackedSet) -> list[str]:
    moves = load_moves()
    return [
        move_id
        for move_id in pset.moves
        if (moves.get(move_id) or {}).get("category") not in (None, "Status")
    ]


# --- cache -------------------------------------------------------------------------------


def _empty_cache() -> dict[str, Any]:
    return {
        "schema": 1,
        "probe_version": PROBE_VERSION,
        "showdown_commit": showdown_commit(),
        "conditions": [list(condition) for condition in CONDITIONS],
        "profiles": list(PROFILES),
        "entries": {},
    }


def read_cache(path: Path | str = CACHE_PATH) -> dict[str, Any]:
    """The cache file, or an empty one when it is missing or was built by another probe/engine."""
    path = Path(path)
    if not path.exists():
        return _empty_cache()
    data = json.loads(path.read_text())
    if data.get("probe_version") != PROBE_VERSION or data.get("showdown_commit") != showdown_commit():
        return _empty_cache()
    return data


def write_cache(cache: dict[str, Any], path: Path | str = CACHE_PATH) -> None:
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, separators=(",", ":"), sort_keys=True))
    os.replace(tmp, path)


def entry_from_probe(pset: PackedSet, tested: list[str], results: dict[str, Any]) -> dict[str, Any]:
    """Collapse the probe's per-move two-turn totals into the stored per-set entry."""
    per_turn = [[0.0] * len(PROFILES) for _ in CONDITIONS]
    best = [[-1] * len(PROFILES) for _ in CONDITIONS]
    for move_index, move_id in enumerate(tested):
        table = results.get(move_id)
        if not table:
            continue
        for c, row in enumerate(table):
            for p, total in enumerate(row):
                if total is None:
                    continue
                value = total / 2.0
                if value > per_turn[c][p]:
                    per_turn[c][p] = value
                    best[c][p] = move_index
    base = per_turn[0]
    gain = [[value - base[p] for p, value in enumerate(row)] for row in per_turn]
    return {
        "species": pset.species_id,
        "mega": mega_species_id(pset.species_id, pset.item_id),
        "tested": tested,
        "per_turn": [[round(v, 2) for v in row] for row in per_turn],
        "gain": [[round(v, 2) for v in row] for row in gain],
        "best": best,
    }


class ProbeWorker:
    """One long-lived `tools/plan_value_probe.mjs` process (JSON lines, like `SimWorker`)."""

    def __init__(self, showdown_repo: str | Path | None = None) -> None:
        from vgc.config import SHOWDOWN_REPO
        from vgc.rl.env import SimWorker

        self._worker = SimWorker(showdown_repo or SHOWDOWN_REPO, script=PROBE_SCRIPT)

    def measure(self, pset: PackedSet, seeds: Iterable[int] = DEFAULT_SEEDS) -> dict[str, Any]:
        tested = damaging_moves(pset)
        results: dict[str, Any] = {}
        errors: list[str] = []
        if tested:
            response = self._worker.request(
                {
                    "packed": pset.packed,
                    "moves": tested,
                    "conditions": [list(condition) for condition in CONDITIONS],
                    "seeds": list(seeds),
                }
            )
            results = response.get("results") or {}
            errors = response.get("errors") or []
        entry = entry_from_probe(pset, tested, results)
        if errors:
            entry["errors"] = errors[:3]
        return entry

    def close(self) -> None:
        self._worker.close()

    def __enter__(self) -> "ProbeWorker":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# --- lookup for vgc.field_control ----------------------------------------------------------


@dataclass(frozen=True)
class PlanEntry:
    species_id: str
    gain: tuple[tuple[float, ...], ...]  # [condition][profile], %HP per turn vs the bare field

    def gain_at(self, weather: str | None, terrain: str | None, profile: int) -> float:
        return self.gain[condition_index(weather, terrain)][profile]


# Per team (vgc.team_scope): team key -> ((species, moves) -> entry, species -> entries).
_TEAMS: dict[str, tuple[dict[tuple[str, frozenset[str]], PlanEntry], dict[str, list[PlanEntry]]]] = {}

# Diagnostic: our Pokemon the registry had no cache entry for (so field_control fell back to
# its estimate). Keyed by species id; read it after a run to see whether a cache covers a pool.
MISSING: Counter[str] = Counter()
REGISTERED = Counter()  # {"found": n, "missing": n}
# Diagnostic: times `vgc.field_control` built one of our Pokemon with the measured plan switched
# on but found no registered entry (a set was never registered or not in the cache).
FALLBACKS: Counter[str] = Counter()


def _entry_from_cache(raw: dict[str, Any], species_id: str) -> PlanEntry:
    return PlanEntry(species_id, tuple(tuple(float(v) for v in row) for row in raw["gain"]))


def register_own_team(packed_team: str | None, cache: dict[str, Any] | None = None) -> int:
    """Make this team's measured entries visible to `lookup`; returns how many were found.

    Idempotent and cheap after the first call. A set with no cache entry is counted in
    `MISSING` and simply absent (`field_control` falls back to its estimate for it).
    """
    if not packed_team:
        return 0
    cache = cache if cache is not None else _cached_default()
    entries = cache.get("entries", {})
    found = 0
    _registry, _by_species = _TEAMS.setdefault(team_key(packed_team), ({}, {}))
    for pset in parse_packed_team(packed_team):
        raw = entries.get(set_key(pset))
        if raw is None:
            MISSING[pset.species_id] += 1
            REGISTERED["missing"] += 1
            continue
        found += 1
        REGISTERED["found"] += 1
        names = {pset.species_id}
        mega = raw.get("mega")
        if mega:
            names.add(mega)
        for species_id in names:
            entry = _entry_from_cache(raw, species_id)
            _registry[(species_id, frozenset(pset.moves))] = entry
            bucket = _by_species.setdefault(species_id, [])
            if entry not in bucket:
                bucket.append(entry)
    return found


@lru_cache(maxsize=1)
def _cached_default() -> dict[str, Any]:
    return read_cache(CACHE_PATH)


def clear_registry() -> None:
    _TEAMS.clear()
    MISSING.clear()
    REGISTERED.clear()
    FALLBACKS.clear()
    _cached_default.cache_clear()


def lookup(species_id: str, moves: Iterable[str]) -> PlanEntry | None:
    """The measured entry for one of our in-battle Pokemon, or None when it was not cached.

    Matches on species and move set (a Mega keeps its base form's moves); falls back to the
    species alone when only one set of that species was registered, since an in-battle move
    list can drift (a copied/transformed move) while the Species Clause keeps the match
    unambiguous within a team.
    """
    tables = resolve_table(_TEAMS)
    if tables is None:
        return None  # no team bound and several registered: never guess another team's set
    registry, by_species = tables
    found = registry.get((species_id, frozenset(moves)))
    if found is not None:
        return found
    candidates = by_species.get(species_id) or ()
    return candidates[0] if len(candidates) == 1 else None
