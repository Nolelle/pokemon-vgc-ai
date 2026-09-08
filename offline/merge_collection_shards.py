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
    save_demonstrations,
)

def load_shard_datasets(paths: Sequence[Path]) -> tuple[list, list[dict[str, object]]]:
    """Use the same copy-safe source identities as training and the auditor."""
    from vgc.rl.demonstrations import load_datasets

    try:
        return load_datasets(paths)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


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
