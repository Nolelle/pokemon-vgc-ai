#!/usr/bin/env python
"""Build `data/usage/plan_value_cache.json`: measured weather/terrain value per team set.

For every distinct Pokemon set in the given teams, run the real Showdown engine (via
`tools/plan_value_probe.mjs`) over all 25 weather x terrain conditions and store the best
move's per-turn damage and its gain over the bare field. See `vgc/plan_value.py` for the
design and `tools/plan_value_probe.mjs` for how conditions are held and luck removed.

Incremental: a set already in the cache (same canonical set, probe version, Showdown pin)
is skipped, so rerunning after adding teams only measures the new sets. A set whose probe
reported any error is NOT stored (a failed cell would read as zero), so it is retried next run.

Usage:
    .venv/bin/python tools/build_plan_value_cache.py \
        --teams 'teams/owner/*.packed.txt' \
        --manifest data/selfplay/mc_sheet_pool_v2/manifest.json
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

from vgc import plan_value as pv


def collect_sets(team_globs: list[str], manifests: list[str]) -> dict[str, pv.PackedSet]:
    """Distinct sets (by cache key) across every named team file."""
    files: list[Path] = []
    for pattern in team_globs:
        files.extend(Path(path) for path in sorted(glob.glob(pattern)))
    for manifest in manifests:
        manifest_path = Path(manifest)
        for entry in json.loads(manifest_path.read_text()):
            files.append(manifest_path.parent / entry["file"])
    sets: dict[str, pv.PackedSet] = {}
    for path in files:
        for pset in pv.parse_packed_team(path.read_text()):
            sets.setdefault(pv.set_key(pset), pset)
    return sets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teams", action="append", default=[], help="glob of packed team files")
    parser.add_argument("--manifest", action="append", default=[], help="team pool manifest.json")
    parser.add_argument("--cache", default=str(pv.CACHE_PATH))
    parser.add_argument("--seeds", type=int, default=len(pv.DEFAULT_SEEDS))
    parser.add_argument("--limit", type=int, default=0, help="measure at most this many new sets")
    parser.add_argument("--save-every", type=int, default=100)
    args = parser.parse_args()

    started = time.time()
    sets = collect_sets(args.teams, args.manifest)
    cache = pv.read_cache(args.cache)
    todo = [(key, pset) for key, pset in sets.items() if key not in cache["entries"]]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(sets)} distinct sets, {len(sets) - len(todo)} cached, {len(todo)} to measure")

    seeds = tuple(range(1, args.seeds + 1))
    errors = 0
    with pv.ProbeWorker() as worker:
        for done, (key, pset) in enumerate(todo, 1):
            entry = worker.measure(pset, seeds)
            if "errors" in entry:
                # Never cache a measurement with failed cells (they read as zero): the next
                # build retries this set, and the player falls back to its estimate meanwhile.
                errors += 1
                print(f"  ERROR {pset.species_id} {entry['errors'][0]}")
                continue
            cache["entries"][key] = entry
            if done % args.save_every == 0:
                pv.write_cache(cache, args.cache)
                elapsed = time.time() - started
                remaining = elapsed / done * (len(todo) - done)
                print(f"  {done}/{len(todo)} sets, {elapsed:.0f}s elapsed, ~{remaining:.0f}s left")
    pv.write_cache(cache, args.cache)
    size = Path(args.cache).stat().st_size
    print(
        f"done: {len(todo)} measured ({errors} with probe errors, NOT cached), "
        f"{len(cache['entries'])} total "
        f"entries, {size / 1e6:.2f} MB, wall time {time.time() - started:.0f}s"
    )


if __name__ == "__main__":
    main()
