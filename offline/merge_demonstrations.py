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

    from vgc.rl.demonstrations import load_demonstrations, save_demonstrations

    merged: list = []
    for path in args.inputs:
        chunk = load_demonstrations(path)
        print(f"{path}: {len(chunk)} samples")
        merged.extend(chunk)
    battles = len({sample.battle_id for sample in merged})
    teams = len({sample.team_id for sample in merged if sample.team_id})
    save_demonstrations(args.out, merged)
    print(f"merged {len(merged)} samples / {battles} battles / {teams} teams -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
