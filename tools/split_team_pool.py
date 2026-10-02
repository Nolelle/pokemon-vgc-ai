#!/usr/bin/env python
"""Split a team-pool manifest into team-disjoint train and holdout manifests.

Teams are grouped by their set of six species before splitting, so near-copies (same six
Pokemon, different moves or items) always land on the same side and cannot leak a
training team into the holdout. Groups are assigned per archetype tag, so each side
keeps roughly the pool's style mix. Deterministic for a given `--seed`.

Writes `<manifest dir>/train_manifest.json` and `holdout_manifest.json` (same entry
format as the input, so `--team-manifest` and `evaluate_own_spread_pool.py --manifest`
accept either) and `split.json` recording the seed, fraction and file lists.

Usage:
    .venv/bin/python tools/split_team_pool.py \
        --manifest data/selfplay/mc_sheet_pool_v2/manifest.json --holdout-fraction 0.3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


def split_manifest(
    entries: list[dict], holdout_fraction: float, seed: int
) -> tuple[list[dict], list[dict]]:
    groups: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for entry in entries:
        groups[tuple(sorted(entry["species"]))].append(entry)
    by_archetype: dict[str, list[tuple[str, ...]]] = defaultdict(list)
    for key, members in groups.items():
        # A group's archetype is its most common member tag (ties by name).
        tags = sorted(member["archetype"] for member in members)
        by_archetype[max(set(tags), key=lambda tag: (tags.count(tag), tag))].append(key)
    rng = random.Random(seed)
    holdout_keys: set[tuple[str, ...]] = set()
    for archetype in sorted(by_archetype):
        keys = sorted(by_archetype[archetype])
        rng.shuffle(keys)
        target = round(len(keys) * holdout_fraction)
        holdout_keys.update(keys[:target])
    train = [entry for entry in entries if tuple(sorted(entry["species"])) not in holdout_keys]
    holdout = [entry for entry in entries if tuple(sorted(entry["species"])) in holdout_keys]
    return train, holdout


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--holdout-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=20261001)
    args = parser.parse_args()

    entries = json.loads(args.manifest.read_text())
    contents = [(args.manifest.parent / entry["file"]).read_text().strip() for entry in entries]
    if len(set(contents)) != len(contents):
        raise SystemExit("manifest has byte-identical teams; dedupe before splitting")
    train, holdout = split_manifest(entries, args.holdout_fraction, args.seed)

    out_dir = args.manifest.parent
    (out_dir / "train_manifest.json").write_text(json.dumps(train, indent=1) + "\n")
    (out_dir / "holdout_manifest.json").write_text(json.dumps(holdout, indent=1) + "\n")
    record = {
        "source_manifest": args.manifest.name,
        "source_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "holdout_fraction": args.holdout_fraction,
        "seed": args.seed,
        "grouping": "sorted six-species set; groups never straddle the split",
        "train_files": sorted(entry["file"] for entry in train),
        "holdout_files": sorted(entry["file"] for entry in holdout),
    }
    (out_dir / "split.json").write_text(json.dumps(record, indent=1) + "\n")
    for name, part in (("train", train), ("holdout", holdout)):
        tags: dict[str, int] = defaultdict(int)
        for entry in part:
            tags[entry["archetype"]] += 1
        print(f"{name}: {len(part)} teams {dict(sorted(tags.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
