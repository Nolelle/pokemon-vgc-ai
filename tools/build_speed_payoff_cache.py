#!/usr/bin/env python
"""Build `data/usage/speed_payoff_cache.json`: measured Tailwind / Trick Room payoff per set.

For every distinct set in the named teams (and the most-used M-C usage sets, which serve as
OPPONENT-side entries and as the reference panel), run the real Showdown engine via
`tools/speed_payoff_probe.mjs` against the panel and store the per-turn %HP payoff of Tailwind
(ours / against us) and Trick Room. See `vgc/speed_payoff.py` for the design.

Incremental: a set already cached (same canonical set, probe version, Showdown pin, panel) is
skipped. Changing the panel changes every key, so everything is rebuilt.

Usage:
    .venv/bin/python tools/build_speed_payoff_cache.py \
        --teams 'teams/owner/*.packed.txt' \
        --manifest data/selfplay/mc_sheet_pool_v2/manifest.json
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

from vgc import speed_payoff as sp
from vgc.plan_value import PackedSet, parse_packed_set, parse_packed_team


def collect_sets(team_globs: list[str], manifests: list[str], panel_id: str) -> dict[str, PackedSet]:
    files: list[Path] = []
    for pattern in team_globs:
        files.extend(Path(path) for path in sorted(glob.glob(pattern)))
    for manifest in manifests:
        manifest_path = Path(manifest)
        for entry in json.loads(manifest_path.read_text()):
            files.append(manifest_path.parent / entry["file"])
    sets: dict[str, PackedSet] = {}
    for path in files:
        for pset in parse_packed_team(path.read_text()):
            sets.setdefault(sp.set_key(pset, panel_id), pset)
    return sets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teams", action="append", default=[], help="glob of packed team files")
    parser.add_argument("--manifest", action="append", default=[], help="team pool manifest.json")
    parser.add_argument("--cache", default=str(sp.CACHE_PATH))
    parser.add_argument("--chaos", default=str(sp.CHAOS_PATH))
    parser.add_argument("--panel-size", type=int, default=sp.PANEL_SIZE)
    parser.add_argument("--usage-sets", type=int, default=sp.USAGE_SETS)
    parser.add_argument("--limit", type=int, default=0, help="measure at most this many new sets")
    parser.add_argument("--save-every", type=int, default=100)
    args = parser.parse_args()

    started = time.time()
    usage = sp.usage_sets(args.chaos, args.usage_sets)
    panel_sets = usage[: args.panel_size]
    total_usage = sum(item.usage for item in panel_sets) or 1.0
    panel = [
        {"species": item.name, "packed": item.packed, "weight": item.usage / total_usage}
        for item in panel_sets
    ]
    panel_id = sp.panel_hash(foe["packed"] for foe in panel)

    cache = sp.read_cache(args.cache)
    if cache.get("panel_hash") != panel_id:
        cache = sp._empty_cache()  # a new panel invalidates every entry
    cache["panel"] = panel
    cache["panel_hash"] = panel_id

    sets = collect_sets(args.teams, args.manifest, panel_id)
    usage_keys: dict[str, str] = {}
    for item in usage:
        pset = parse_packed_set(item.packed)
        key = sp.set_key(pset, panel_id)
        sets.setdefault(key, pset)
        usage_keys[item.species_id] = key
    # An opponent shown as the base form (not yet Mega Evolved) reads the Mega's set.
    cache["usage"] = _with_base_forms(usage_keys)

    todo = [(key, pset) for key, pset in sets.items() if key not in cache["entries"]]
    if args.limit:
        todo = todo[: args.limit]
    print(
        f"panel: {', '.join(foe['species'] for foe in panel)}\n"
        f"{len(sets)} distinct sets, {len(sets) - len(todo)} cached, {len(todo)} to measure"
    )

    errors = 0
    with sp.ProbeWorker() as worker:
        for done, (key, pset) in enumerate(todo, 1):
            entry = worker.measure(pset, panel)
            if "errors" in entry:
                # Never cache a measurement with failed duels; the next build retries it.
                errors += 1
                print(f"  ERROR {pset.species_id} {entry['errors'][0]}")
                continue
            cache["entries"][key] = entry
            if done % args.save_every == 0:
                sp.write_cache(cache, args.cache)
                elapsed = time.time() - started
                remaining = elapsed / done * (len(todo) - done)
                print(f"  {done}/{len(todo)} sets, {elapsed:.0f}s elapsed, ~{remaining:.0f}s left")
    sp.write_cache(cache, args.cache)
    size = Path(args.cache).stat().st_size
    print(
        f"done: {len(todo)} measured ({errors} with probe errors, NOT cached), "
        f"{len(cache['entries'])} total "
        f"entries, {size / 1e6:.2f} MB, wall time {time.time() - started:.0f}s"
    )


def _with_base_forms(usage_keys: dict[str, str]) -> dict[str, str]:
    """Add base-form ids that map to a Mega's entry when the base itself has no usage set."""
    from vgc.data import load_species

    species = load_species()
    result = dict(usage_keys)
    for species_id, key in usage_keys.items():
        data = species.get(species_id) or {}
        if not data.get("isMega"):
            continue
        base = "".join(ch for ch in str(data.get("changesFrom") or data.get("baseSpecies") or "").lower() if ch.isalnum())
        if base and base not in result:
            result[base] = key
    return result


if __name__ == "__main__":
    main()
