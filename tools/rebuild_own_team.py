#!/usr/bin/env python
"""Rebuild the acting player's own-team item/ability/moves in a schema-8 decisions
file (`vgc.bc.own_team_rebuild`'s Layers 1-3). Writes a NEW output file -- never
overwrites the source.

    .venv/bin/python tools/rebuild_own_team.py \
        --data data/bc/decisions_regmc_s8_20261104.jsonl.gz \
        --manifest runs/pretraining_selection_20261102_v3/manifest.jsonl \
        --raw-dir data/replays/gen9championsvgc2026regmc \
        --facts-cache runs/own_team_rebuild_20261104/game_facts.jsonl.gz \
        --out runs/own_team_rebuild_20261104/decisions_regmc_s8_rebuilt.jsonl.gz
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vgc.bc.own_team_rebuild import (  # noqa: E402
    RebuildContext,
    build_corpus_facts,
    load_manifest,
    rebuild_record,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, required=True, help="schema-8 decisions JSONL(.gz)")
    parser.add_argument("--manifest", type=Path, required=True, help="selection manifest JSONL")
    parser.add_argument("--raw-dir", type=Path, required=True, help="raw <id>.json replay dir")
    parser.add_argument("--facts-cache", type=Path, default=None,
                        help="cached analyze_log facts (reused across runs)")
    parser.add_argument("--out", type=Path, required=True, help="output JSONL(.gz)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.out == args.data:
        print("--out must differ from --data (never overwrite)", file=sys.stderr)
        return 2
    manifest = load_manifest(args.manifest)
    facts = build_corpus_facts(manifest, args.raw_dir, args.facts_cache)
    ctx = RebuildContext(manifest, facts)
    stats = Counter()
    opener_in = gzip.open if args.data.suffix == ".gz" else open
    opener_out = gzip.open if args.out.suffix == ".gz" else open
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with opener_in(args.data, "rt") as src, opener_out(args.out, "wt") as dst:
        for line in src:
            record = json.loads(line)
            if record.get("schema") != 8:
                raise ValueError("rebuild expects schema-8 records (log_line + canonical species)")
            game_facts = facts.get(record["replay_id"])
            if game_facts is not None and record.get("decision_kind") in ("turn", "forced_switch"):
                rebuild_record(record, game_facts, ctx)
                for info in record.get("own_info", {}).get("active", []):
                    if info:
                        stats[f"item:{info.get('item_source')}"] += 1
                        stats[f"ability:{info.get('ability_source')}"] += 1
                stats["fully_confirmed" if record.get("own_fully_confirmed") else "not_fully"] += 1
            dst.write(json.dumps(record, sort_keys=True) + "\n")
            stats["records"] += 1
    print(json.dumps(dict(stats.most_common()), indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
