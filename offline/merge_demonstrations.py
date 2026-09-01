#!/usr/bin/env python
"""Merge demonstration files into one dataset, verifying schema compatibility.

Sharded collections write one file per worker; training wants a single file. Every
input must carry the current DEMONSTRATION_FORMAT_VERSION (load_demonstrations already
enforces that), so a stale shard fails loudly here instead of silently shrinking the
training set later.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("inputs", type=Path, nargs="+")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if len(args.inputs) < 2:
        raise SystemExit("merging needs at least two input files")

    from vgc.rl.demonstrations import load_demonstration_dataset, save_demonstrations

    merged: list = []
    source_metadata: list[dict[str, object]] = []
    for path in args.inputs:
        chunk, metadata = load_demonstration_dataset(path)
        print(f"{path}: {len(chunk)} samples")
        merged.extend(chunk)
        source_metadata.append(metadata)
    battles = len({sample.battle_id for sample in merged})
    teams = len({sample.team_id for sample in merged if sample.team_id})
    invariant_fields = (
        "repository_commit",
        "repository_dirty",
        "showdown_commit",
        "showdown_dirty",
        "format_id",
        "team_source",
        "opponents",
        "policy_config",
        "information_contract",
        "teacher_source",
    )
    for field in invariant_fields:
        values = {repr(metadata[field]) for metadata in source_metadata}
        if len(values) != 1:
            raise SystemExit(f"input datasets disagree on {field}")
    metadata = {
        **source_metadata[0],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "collector": "offline/merge_demonstrations.py",
        "requested_games": sum(int(entry["requested_games"]) for entry in source_metadata),
        "source_datasets": [str(path.resolve()) for path in args.inputs],
    }
    save_demonstrations(args.out, merged, metadata=metadata)
    print(f"merged {len(merged)} samples / {battles} battles / {teams} teams -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
