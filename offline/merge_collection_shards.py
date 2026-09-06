#!/usr/bin/env python
"""Merge same-pool collection shards, checking every invariant except shard identity.

`offline/merge_demonstrations.py` refuses inputs whose metadata differs in ANY
invariant field, including `team_source` (the manifest path) and `seed`. That
is correct for arbitrary datasets, but collection shards of one pool run share
everything except their partition manifest, seed, and game count by
construction. This script verifies all of THOSE fields match exactly across
inputs, merges, and records the shard list -- so a heterogeneous mix still
fails loudly instead of training silently.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vgc.rl.demonstrations import (  # noqa: E402
    load_demonstration_dataset,
    save_demonstrations,
)

# Fields that must match byte-for-byte across shards. Deliberately excludes
# `team_source`, `seed`, and `requested_games`, which differ per partition.
INVARIANT_FIELDS = (
    "repository_commit",
    "repository_dirty",
    "showdown_commit",
    "showdown_dirty",
    "format_id",
    "opponents",
    "policy_config",
    "information_contract",
    "teacher_source",
)


def load_shard_datasets(paths: Sequence[Path]) -> tuple[list, list[dict[str, object]]]:
    """Load shard files, enforcing the same-pool invariants. Shared by merge and
    multi-file training so both paths accept exactly the same inputs.

    Battle ids are namespaced by shard file stem: every shard numbers its games
    from zero (`imitation-train-000000` exists in each one), so without this the
    audit's duplicate-decision check false-positives across shards.
    """

    from dataclasses import replace

    merged: list = []
    source_metadata: list[dict[str, object]] = []
    for path in paths:
        chunk, metadata = load_demonstration_dataset(path)
        print(f"{path}: {len(chunk)} samples", flush=True)
        # Namespace by parent directory (shard id): every shard file is named
        # demonstrations.pt and numbers its games from zero, so the stem alone
        # does not disambiguate.
        namespace = path.parent.name
        merged.extend(
            replace(sample, battle_id=f"{namespace}:{sample.battle_id}")
            for sample in chunk
        )
        source_metadata.append(metadata)
    for field in INVARIANT_FIELDS:
        values = {repr(metadata[field]) for metadata in source_metadata}
        if len(values) != 1:
            raise SystemExit(f"shard datasets disagree on {field}")
    return merged, source_metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("inputs", type=Path, nargs="+")
    args = parser.parse_args()

    merged, source_metadata = load_shard_datasets(args.inputs)
    battles = len({sample.battle_id for sample in merged})
    teams = len({sample.team_id for sample in merged if sample.team_id})
    metadata = {
        **source_metadata[0],
        "created_at_utc": datetime.now(UTC).isoformat(),
        "collector": "offline/merge_collection_shards.py",
        "requested_games": sum(int(entry["requested_games"]) for entry in source_metadata),
        "source_datasets": [str(path.resolve()) for path in args.inputs],
    }
    save_demonstrations(args.out, merged, metadata=metadata)
    print(f"merged {len(merged)} samples / {battles} battles / {teams} teams -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
