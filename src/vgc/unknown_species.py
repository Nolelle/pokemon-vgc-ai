"""Capture sightings of opponent species we have no (or too little) history for.

Fog-safe by construction: a capture record contains only what a real player
could see in that battle -- the previewed species, which of them appeared,
moves/items/abilities actually revealed, and teammates. It never contains
Stat Points or nature (never revealed for an opponent), an unrevealed move,
or anything from a simulator root. Those fields are exactly what a capture
*can* teach us (move/item/ability frequencies); spreads can only come from
public usage stats, so records flag ``spread_learnable: False`` to stop a
future builder pretending otherwise.

Live play appends JSON Lines (one record per species per battle, updated as
more is revealed) when ``PolicyConfig.capture_unknown_species`` is on.
``tools/build_priors_from_captures.py`` folds those sightings into a
set-priors-shaped supplement that can be merged with the replay corpus.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from vgc.damage import to_id
from vgc.models import PolicyConfig
from vgc.sets import USAGE_DIR, load_set_priors, load_usage_spreads, set_priors_for, usage_spreads_for

CAPTURE_SCHEMA = "unknown_species_capture_v1"
DEFAULT_CAPTURE_FILE = "unknown_species.jsonl"


def capture_path_for(config: PolicyConfig) -> Path:
    return USAGE_DIR / config.unknown_species_log_file


def species_history_status(
    species_id: str,
    *,
    usage: dict[str, list[dict[str, Any]]] | None = None,
    set_priors: dict[str, Any] | None = None,
    min_appearances: int = 5,
) -> str:
    """``"known"``, ``"thin"`` (some history, below the trust bar), or ``"unknown"``."""

    species_id = to_id(species_id)
    usage = load_usage_spreads() if usage is None else usage
    set_priors = load_set_priors() if set_priors is None else set_priors
    has_spread = bool(usage.get(species_id))
    entry = (set_priors.get("species") or {}).get(species_id) if set_priors else None
    appearances = int((entry or {}).get("appearances", 0) or 0)
    if not has_spread and appearances == 0:
        return "unknown"
    if appearances < min_appearances:
        return "thin"
    return "known"


def build_capture_records(
    battle,
    memory,
    config: PolicyConfig | None = None,
    *,
    usage: dict[str, list[dict[str, Any]]] | None = None,
    set_priors: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """One fog-safe capture record per no-history opponent species in ``battle``."""

    from vgc.opponent_belief import build_opponent_beliefs

    config = config or PolicyConfig()
    usage = usage_spreads_for(config) if usage is None else usage
    set_priors = set_priors_for(config) if set_priors is None else set_priors
    beliefs = build_opponent_beliefs(battle, memory, config, usage=usage, set_priors=set_priors)
    preview = [belief.species_id for belief in beliefs]
    records: list[dict[str, Any]] = []
    for belief in beliefs:
        status = species_history_status(
            belief.species_id,
            usage=usage,
            set_priors=set_priors,
            min_appearances=config.set_prior_min_games,
        )
        if status == "known":
            continue
        records.append(
            {
                "schema": CAPTURE_SCHEMA,
                "species": belief.species_id,
                "history_status": status,
                "format_id": config.format_id,
                "battle_tag": str(getattr(battle, "battle_tag", "") or ""),
                "turn": int(getattr(battle, "turn", 0) or 0),
                "appeared": bool(belief.appeared),
                "active": bool(belief.active),
                "revealed_moves": sorted(belief.revealed_moves),
                "known_item": belief.known_item,
                "known_ability": belief.known_ability,
                "teammates": [s for s in preview if s != belief.species_id],
                "speed_observations": sum(
                    1 for o in memory.speed_observations if o.opponent_species == belief.species_id
                ),
                "damage_observations": sum(
                    1 for o in memory.damage_observations if o.opponent_species == belief.species_id
                ),
                # An opponent's Stat Points/nature are never revealed in battle,
                # so a sighting can teach moves/item/ability only.
                "spread_learnable": False,
            }
        )
    return records


def capture_signature(record: dict[str, Any]) -> tuple:
    return (
        record.get("species"),
        tuple(record.get("revealed_moves") or ()),
        record.get("known_item"),
        record.get("known_ability"),
        bool(record.get("appeared")),
    )


def append_capture_records(path: Path | str, records: Iterable[dict[str, Any]]) -> int:
    rows = list(records)
    if not rows:
        return 0
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        for record in rows:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return len(rows)


def load_capture_records(path: Path | str) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and record.get("species"):
            records.append(record)
    return records


def aggregate_captures(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Fold capture records into the ``set_priors.json`` species shape.

    Counting rules mirror ``tools/build_set_priors``: a species counts once
    per battle (the latest record for that battle wins, since captures are
    appended as more is revealed), and each revealed move/item/ability counts
    once per battle.
    """

    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        key = (str(record.get("battle_tag") or ""), to_id(record.get("species")))
        previous = latest.get(key)
        if previous is None or int(record.get("turn", 0)) >= int(previous.get("turn", 0)):
            latest[key] = record

    appearances: Counter[str] = Counter()
    preview_sightings: Counter[str] = Counter()
    moves: dict[str, Counter] = defaultdict(Counter)
    items: dict[str, Counter] = defaultdict(Counter)
    abilities: dict[str, Counter] = defaultdict(Counter)
    configurations: dict[str, Counter] = defaultdict(Counter)
    for record in latest.values():
        species_id = to_id(record.get("species"))
        if not species_id:
            continue
        if not record.get("appeared"):
            # Previewed but never brought: a sighting, not an appearance.
            # Counting it would inflate the frequency denominator.
            preview_sightings[species_id] += 1
            continue
        appearances[species_id] += 1
        revealed = tuple(sorted({to_id(m) for m in record.get("revealed_moves") or [] if to_id(m)}))
        for move_id in revealed:
            moves[species_id][move_id] += 1
        item = to_id(record.get("known_item")) or ""
        ability = to_id(record.get("known_ability")) or ""
        if item:
            items[species_id][item] += 1
        if ability:
            abilities[species_id][ability] += 1
        if revealed or item or ability:
            configurations[species_id][(revealed, item, ability)] += 1

    species_out: dict[str, dict[str, Any]] = {}
    for species_id, count in sorted(appearances.items()):
        species_out[species_id] = {
            "appearances": count,
            "preview_sightings": int(preview_sightings.get(species_id, 0)),
            "moves": dict(sorted(moves.get(species_id, {}).items())),
            "items": dict(sorted(items.get(species_id, {}).items())),
            "abilities": dict(sorted(abilities.get(species_id, {}).items())),
            "configurations": [
                {"moves": list(key[0]), "item": key[1] or None, "ability": key[2] or None, "count": n}
                for key, n in sorted(
                    configurations.get(species_id, {}).items(), key=lambda e: (-e[1], e[0])
                )[:20]
            ],
        }
    return {
        "meta": {
            "source": CAPTURE_SCHEMA,
            "captured_battles": len(latest),
            "species": len(species_out),
            "capture_keys": sorted(f"{tag}|{sid}" for tag, sid in latest),
        },
        "species": species_out,
    }


def merge_priors(base: dict[str, Any], supplement: dict[str, Any]) -> dict[str, Any]:
    """Sum a capture supplement into a set-priors dict (counts are additive).

    Idempotent for the same captures: if every capture key in the supplement
    was already imported into ``base``, ``base`` is returned unchanged. A
    partial overlap raises, because aggregate counts cannot be partially
    subtracted -- rebuild from the original base instead.
    """

    supp_keys = set((supplement.get("meta") or {}).get("capture_keys") or [])
    base_keys = set((base.get("meta") or {}).get("capture_keys") or [])
    if supp_keys and supp_keys <= base_keys:
        return base
    if supp_keys & base_keys:
        raise ValueError(
            "captures already partially imported into this base; "
            "rebuild from the original (pre-merge) priors file"
        )

    merged_species: dict[str, Any] = {}
    base_species = base.get("species") or {}
    supp_species = supplement.get("species") or {}
    for species_id in sorted(set(base_species) | set(supp_species)):
        base_entry = base_species.get(species_id) or {}
        supp_entry = supp_species.get(species_id) or {}
        entry: dict[str, Any] = {
            "appearances": int(base_entry.get("appearances", 0)) + int(supp_entry.get("appearances", 0))
        }
        for key in ("moves", "items", "abilities"):
            counts: Counter[str] = Counter()
            counts.update({str(k): int(v) for k, v in (base_entry.get(key) or {}).items()})
            counts.update({str(k): int(v) for k, v in (supp_entry.get(key) or {}).items()})
            entry[key] = dict(sorted(counts.items()))
        configs: Counter = Counter()
        for source in (base_entry, supp_entry):
            for row in source.get("configurations") or []:
                key = (
                    tuple(sorted(row.get("moves") or [])),
                    row.get("item") or "",
                    row.get("ability") or "",
                )
                configs[key] += int(row.get("count", 0))
        entry["configurations"] = [
            {"moves": list(key[0]), "item": key[1] or None, "ability": key[2] or None, "count": n}
            for key, n in sorted(configs.items(), key=lambda e: (-e[1], e[0]))[:20]
        ]
        merged_species[species_id] = entry
    meta = dict(base.get("meta") or {})
    meta["capture_supplement_battles"] = int((supplement.get("meta") or {}).get("captured_battles", 0))
    meta["capture_keys"] = sorted(base_keys | supp_keys)
    return {"meta": meta, "species": merged_species}
