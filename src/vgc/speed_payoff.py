"""Measured Tailwind / Trick Room payoff of one Pokemon set ("speed payoff").

`vgc.field_control` used to value speed control with a generic term: the fraction of pairings
in which our Pokemon move first. That ignores what moving first is WORTH for a specific set:
a fast attacker under Tailwind gets its KO off before the foe acts, a slow bulky attacker under
Trick Room finally does, and a fast attacker under Trick Room just got slower. This module
measures that with the real Showdown engine (`tools/speed_payoff_probe.mjs`), the way
`vgc.plan_value` measures weather/terrain.

What is measured (see the probe's header for the full protocol)
---------------------------------------------------------------
One subject set against each of a PANEL of real opposing sets (the most-used sets in the M-C
usage data, at their real top spread/item/ability/moves, real HP -- KO timing is the point).
Each side uses its single strongest damaging move into the other, both turns, in every
scenario (the same moves, so only the move order changes). Over two turns, with luck removed
(mean roll, no crits, accuracy-weighted):

    net(s) = %HP the subject deals - %HP the subject takes          (s = a scenario)

    tw_ours      = (net(Tailwind on our side)  - net(bare)) / 2     per turn
    tw_against   = (net(Tailwind on the foe)   - net(bare)) / 2     per turn  (usually < 0)
    tr           = (net(Trick Room)            - net(bare)) / 2     per turn

averaged over the panel by usage weight. Units are %-of-max-HP per turn per Pokemon -- the same
currency as `vgc.field_control`'s weather/terrain fit term (F_t), so the same weight applies.

Cache: `data/usage/speed_payoff_cache.json`, built by `tools/build_speed_payoff_cache.py` for the
owner's teams, the M-C real-team pool, and the most-used usage sets (so the OPPONENT's Pokemon,
known only by species, can be read from `usage`). Key = sha1 of (canonical set, probe version,
Showdown pin, panel hash): a change to the probe, the engine or the panel invalidates entries.

Known limits: one-versus-one duels (no ally, no Protect, no switching, no priority/support
moves chosen), a panel rather than the actual opposing team, hidden opponent sets read from
the species' most common set.
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
from vgc.data import load_moves
from vgc.plan_value import (
    PackedSet,
    mega_species_id,
    parse_packed_set,
    parse_packed_team,
    showdown_commit,
    to_id,
)

# Bump when the probe's measurement changes in any way that moves numbers.
PROBE_VERSION = 1

TURNS = 2  # horizon of one duel; payoffs are divided by this
SCENARIOS = ("base", "ourtw", "foetw", "tr")
PANEL_SIZE = 12  # most-used usage sets that serve as reference foes
USAGE_SETS = 70  # most-used species stored as OPPONENT-side entries

CACHE_PATH = DATA_DIR.parent / "usage" / "speed_payoff_cache.json"
CHAOS_PATH = DATA_DIR.parent / "usage" / "gen9championsvgc2026regmc-1760.json"
PROBE_SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "speed_payoff_probe.mjs"
DEFAULT_SEEDS = (1,)


# --- usage sets (the reference panel and the opponent-side subjects) ---------------------------


@dataclass(frozen=True)
class UsageSet:
    name: str  # Showdown display name as in the chaos file, e.g. "Salamence-Mega"
    species_id: str
    usage: float
    packed: str


def _top(counter: dict[str, float], n: int) -> list[str]:
    return [key for key, _ in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:n]]


def usage_set(name: str, data: dict[str, Any]) -> UsageSet | None:
    """The most common set of one chaos-file species as a packed set, or None if unusable."""
    moves_known = load_moves()
    abilities = _top(data.get("Abilities") or {}, 1)
    spreads = _top(data.get("Spreads") or {}, 1)
    if not abilities or not spreads:
        return None
    items = _top({k: v for k, v in (data.get("Items") or {}).items() if k != "nothing"}, 1)
    moves = [m for m in _top(data.get("Moves") or {}, 12) if m in moves_known][:4]
    if not moves:
        return None
    nature, points = spreads[0].split(":")
    evs = ",".join(points.split("/"))
    packed = (
        f"{name}||{items[0] if items else ''}|{abilities[0]}|{','.join(moves)}|"
        f"{nature.capitalize()}|{evs}||||50|"
    )
    return UsageSet(name, to_id(name), float(data.get("usage") or 0.0), packed)


def usage_sets(path: Path | str = CHAOS_PATH, count: int = USAGE_SETS) -> list[UsageSet]:
    """The `count` most-used species' top sets, most used first (deterministic)."""
    data = json.loads(Path(path).read_text())["data"]
    ranked = sorted(data.items(), key=lambda kv: (-float(kv[1].get("usage") or 0.0), kv[0]))
    result: list[UsageSet] = []
    for name, entry in ranked:
        built = usage_set(name, entry)
        if built is not None:
            result.append(built)
        if len(result) >= count:
            break
    return result


def panel_hash(panel: Iterable[str]) -> str:
    return hashlib.sha1("\n".join(panel).encode()).hexdigest()[:12]


def set_key(pset: PackedSet, panel_id: str) -> str:
    material = json.dumps([pset.canonical(), PROBE_VERSION, showdown_commit(), panel_id])
    return hashlib.sha1(material.encode()).hexdigest()[:16]


# --- cache --------------------------------------------------------------------------------------


def _empty_cache() -> dict[str, Any]:
    return {
        "schema": 1,
        "probe_version": PROBE_VERSION,
        "showdown_commit": showdown_commit(),
        "turns": TURNS,
        "panel": [],
        "panel_hash": "",
        "usage": {},
        "entries": {},
    }


def read_cache(path: Path | str = CACHE_PATH) -> dict[str, Any]:
    """The cache file, or an empty one when it is missing or built by another probe/engine."""
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


def entry_from_probe(
    pset: PackedSet, panel: list[dict[str, Any]], results: list[dict[str, Any]]
) -> dict[str, Any]:
    """Collapse the probe's per-foe duels into the stored per-set entry (per-turn %HP)."""
    per_foe: dict[str, list[float | None]] = {"tw": [], "tw_against": [], "tr": [], "base": []}
    ours_move: list[str] = []
    foe_move: list[str] = []
    speeds: list[list[int]] = []

    def net(row: dict[str, Any], scenario: str) -> float | None:
        cell = row.get(scenario)
        return None if cell is None else float(cell[0]) - float(cell[1])

    for row in results:
        ours_move.append(row.get("ourMove", ""))
        foe_move.append(row.get("foeMove", ""))
        speeds.append([int(row.get("ourSpeed") or 0), int(row.get("foeSpeed") or 0)])
        base = net(row, "base")
        per_foe["base"].append(None if base is None else round(base / TURNS, 2))
        for field, scenario in (("tw", "ourtw"), ("tw_against", "foetw"), ("tr", "tr")):
            value = net(row, scenario)
            per_foe[field].append(
                None if value is None or base is None else round((value - base) / TURNS, 2)
            )

    weights = [float(foe["weight"]) for foe in panel]

    def mean(values: list[float | None]) -> float:
        pairs = [(v, w) for v, w in zip(values, weights) if v is not None]
        total = sum(w for _, w in pairs)
        return round(sum(v * w for v, w in pairs) / total, 3) if total else 0.0

    return {
        "species": pset.species_id,
        "mega": mega_species_id(pset.species_id, pset.item_id),
        "tw": mean(per_foe["tw"]),
        "tw_against": mean(per_foe["tw_against"]),
        "tr": mean(per_foe["tr"]),
        "base": mean(per_foe["base"]),
        "per_foe": per_foe,
        "ours_move": ours_move,
        "foe_move": foe_move,
        "speeds": speeds,
    }


class ProbeWorker:
    """One long-lived `tools/speed_payoff_probe.mjs` process (JSON lines, like `SimWorker`)."""

    def __init__(self, showdown_repo: str | Path | None = None) -> None:
        from vgc.config import SHOWDOWN_REPO
        from vgc.rl.env import SimWorker

        self._worker = SimWorker(showdown_repo or SHOWDOWN_REPO, script=PROBE_SCRIPT)

    def measure(
        self, pset: PackedSet, panel: list[dict[str, Any]], seeds: Iterable[int] = DEFAULT_SEEDS
    ) -> dict[str, Any]:
        response = self._worker.request(
            {"subject": pset.packed, "foes": [foe["packed"] for foe in panel], "seeds": list(seeds)}
        )
        entry = entry_from_probe(pset, panel, response.get("results") or [])
        errors = response.get("errors") or []
        if errors:
            entry["errors"] = errors[:3]
        return entry

    def close(self) -> None:
        self._worker.close()

    def __enter__(self) -> "ProbeWorker":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# --- lookup for vgc.field_control ----------------------------------------------------------------


@dataclass(frozen=True)
class SpeedEntry:
    species_id: str
    tw: float  # %HP/turn this set gains when ITS side has Tailwind
    tw_against: float  # %HP/turn it gains when the FOE has Tailwind (normally negative)
    tr: float  # %HP/turn it gains under Trick Room (negative for a fast set)


_OWN: dict[tuple[str, frozenset[str]], SpeedEntry] = {}
_OWN_BY_SPECIES: dict[str, list[SpeedEntry]] = {}
_USAGE: dict[str, SpeedEntry] = {}

# Diagnostics: our Pokemon / opponent species with no entry (field_control then falls back to
# the generic speed term for that condition).
MISSING: Counter[str] = Counter()
FALLBACKS: Counter[str] = Counter()


def _entry(raw: dict[str, Any], species_id: str) -> SpeedEntry:
    return SpeedEntry(species_id, float(raw["tw"]), float(raw["tw_against"]), float(raw["tr"]))


def register_own_team(packed_team: str | None, cache: dict[str, Any] | None = None) -> int:
    """Make this team's measured entries visible to `lookup_own`; returns how many were found."""
    if not packed_team:
        return 0
    cache = cache if cache is not None else _cached_default()
    entries = cache.get("entries", {})
    panel_id = cache.get("panel_hash", "")
    found = 0
    for pset in parse_packed_team(packed_team):
        raw = entries.get(set_key(pset, panel_id))
        if raw is None:
            MISSING[pset.species_id] += 1
            continue
        found += 1
        names = {pset.species_id}
        if raw.get("mega"):
            names.add(raw["mega"])
        for species_id in names:
            entry = _entry(raw, species_id)
            _OWN[(species_id, frozenset(pset.moves))] = entry
            bucket = _OWN_BY_SPECIES.setdefault(species_id, [])
            if entry not in bucket:
                bucket.append(entry)
    return found


def register_usage(cache: dict[str, Any] | None = None) -> int:
    """Load the opponent-side entries (species id -> most common set's payoff)."""
    cache = cache if cache is not None else _cached_default()
    entries = cache.get("entries", {})
    loaded = 0
    for species_id, key in (cache.get("usage") or {}).items():
        raw = entries.get(key)
        if raw is not None:
            _USAGE[species_id] = _entry(raw, species_id)
            loaded += 1
    return loaded


@lru_cache(maxsize=1)
def _cached_default() -> dict[str, Any]:
    return read_cache(CACHE_PATH)


def clear_registry() -> None:
    _OWN.clear()
    _OWN_BY_SPECIES.clear()
    _USAGE.clear()
    MISSING.clear()
    FALLBACKS.clear()
    _cached_default.cache_clear()


def lookup_own(species_id: str, moves: Iterable[str]) -> SpeedEntry | None:
    """The measured entry for one of OUR in-battle Pokemon, or None when it was not cached."""
    found = _OWN.get((species_id, frozenset(moves)))
    if found is not None:
        return found
    candidates = _OWN_BY_SPECIES.get(species_id) or ()
    return candidates[0] if len(candidates) == 1 else None


def lookup_opponent(species_id: str) -> SpeedEntry | None:
    """The entry for an opponent species (its most common set), or None when not cached."""
    if not _USAGE:
        register_usage()
    return _USAGE.get(species_id)


__all__ = [
    "SpeedEntry",
    "UsageSet",
    "entry_from_probe",
    "lookup_opponent",
    "lookup_own",
    "panel_hash",
    "parse_packed_set",
    "read_cache",
    "register_own_team",
    "register_usage",
    "set_key",
    "usage_sets",
]
