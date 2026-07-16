#!/usr/bin/env python
"""Build `data/usage/set_priors.json` -- per-species move/item/ability frequencies
aggregated from the downloaded replay corpus (`tools/download_replays.py`'s output) --
consumed by `vgc.sets.opponent_move_ids` to fill UNREVEALED opponent moves.

Reuses `vgc.replay_parse.parse_replay` (the same turn-by-turn protocol walker
`tools/parse_replays.py` uses) rather than re-implementing any protocol parsing --
this tool only adds the aggregation step on top of that module's already-normalized
per-decision records (species/move ids are already `vgc.damage.to_id`-normalized and
megas already resolved back to their base species by `vgc.replay_parse._resolve_species`
before they ever reach a record, so this tool doesn't need to think about mega forms at
all).

Per (replay, player), a species is counted as "appeared" at most once (from any
`active`/`bench` entry seen across that player's decision records), and each move it's
seen to USE is counted at most once per game (a Pokemon repeating Protect five times in
one game shouldn't inflate Protect's prior 5x relative to a Pokemon that used four
different moves once each) -- items/abilities take the LAST non-null value revealed for
that species in that game (rare mid-game changes like Trick don't get to multi-count).

Output schema (deterministic -- keys sorted):

    {"meta": {"corpus_size": 2939, "generated_from_replays": 2900, "min_rating": 1000,
              "min_rating_found": 1002},
     "species": {"garchomp": {"appearances": 812,
                               "moves": {"protect": 640, "earthquake": 705, ...},
                               "items": {"lifeorb": 210, ...},
                               "abilities": {"roughskin": 800, ...}},
                 ...}}

Usage:
    .venv/bin/python tools/build_set_priors.py
    .venv/bin/python tools/build_set_priors.py --min-rating 1200
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vgc.config import DATA_DIR, FORMAT_ID  # noqa: E402
from vgc.replay_parse import parse_replay  # noqa: E402

DEFAULT_REPLAYS_DIR = REPO_ROOT / "data" / "replays" / FORMAT_ID
DEFAULT_OUT_PATH = DATA_DIR.parent / "usage" / "set_priors.json"
# Ladder norms matter more than elite-only play here (we FACE the whole ladder, not
# just its top), so the default keeps essentially every rated game rather than
# restricting to a high-rating slice the way a "best play" usage stat might.
DEFAULT_MIN_RATING = 1000


def _aggregate_replay(
    records: list[dict],
    move_counts: dict[str, Counter],
    item_counts: dict[str, Counter],
    ability_counts: dict[str, Counter],
    appearance_counts: Counter,
) -> None:
    """Mutates the four aggregator dicts in place from one replay's decision records
    (`vgc.replay_parse.ParsedReplay.records`).
    """
    for player in ("p1", "p2"):
        appeared: set[str] = set()
        moves_seen: dict[str, set[str]] = defaultdict(set)
        last_item: dict[str, str] = {}
        last_ability: dict[str, str] = {}

        for record in records:
            if record["player"] != player:
                continue
            state = record.get("state") or {}
            our = state.get("our") or {}
            active = our.get("active") or []
            for slot_index, mon in enumerate(active):
                if not mon:
                    continue
                species_id = mon["species"]
                appeared.add(species_id)
                if mon.get("item"):
                    last_item[species_id] = mon["item"]
                if mon.get("ability"):
                    last_ability[species_id] = mon["ability"]
                if record["decision_kind"] == "turn":
                    action = (record.get("action") or {}).get(f"slot{slot_index}")
                    if action and action.get("kind") == "move" and action.get("move_id"):
                        moves_seen[species_id].add(action["move_id"])
            for bench_mon in our.get("bench") or []:
                appeared.add(bench_mon["species"])

        for species_id in appeared:
            appearance_counts[species_id] += 1
        for species_id, move_ids in moves_seen.items():
            for move_id in move_ids:
                move_counts[species_id][move_id] += 1
        for species_id, item in last_item.items():
            item_counts[species_id][item] += 1
        for species_id, ability in last_ability.items():
            ability_counts[species_id][ability] += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--replays-dir",
        type=Path,
        default=DEFAULT_REPLAYS_DIR,
        help="directory of downloaded <id>.json replays (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_PATH,
        help="output path for set_priors.json (default: %(default)s)",
    )
    parser.add_argument(
        "--min-rating",
        type=int,
        default=DEFAULT_MIN_RATING,
        help="skip replays rated below this, and null-rated replays (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.replays_dir.exists():
        print(f"{args.replays_dir} does not exist", file=sys.stderr)
        return 1

    files = sorted(args.replays_dir.glob("*.json"))

    move_counts: dict[str, Counter] = defaultdict(Counter)
    item_counts: dict[str, Counter] = defaultdict(Counter)
    ability_counts: dict[str, Counter] = defaultdict(Counter)
    appearance_counts: Counter = Counter()

    generated_from = 0
    ratings_seen: list[int] = []

    for path in files:
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        rating = payload.get("rating")
        if rating is None or rating < args.min_rating:
            continue
        log = payload.get("log")
        if not isinstance(log, str) or not log:
            continue

        result = parse_replay(payload.get("id", path.stem), rating, log)
        if not result.ok:
            continue

        generated_from += 1
        ratings_seen.append(rating)
        _aggregate_replay(
            result.records, move_counts, item_counts, ability_counts, appearance_counts
        )

    species_out: dict[str, dict[str, object]] = {}
    for species_id, appearances in appearance_counts.items():
        species_out[species_id] = {
            "appearances": appearances,
            "moves": dict(move_counts.get(species_id, {})),
            "items": dict(item_counts.get(species_id, {})),
            "abilities": dict(ability_counts.get(species_id, {})),
        }

    output = {
        "meta": {
            "corpus_size": len(files),
            "generated_from_replays": generated_from,
            "min_rating": args.min_rating,
            "min_rating_found": min(ratings_seen) if ratings_seen else None,
        },
        "species": species_out,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True))

    print(f"wrote {len(species_out)} species to {args.out}")
    print(
        f"corpus_size={len(files)} generated_from_replays={generated_from} "
        f"min_rating={args.min_rating} min_rating_found={output['meta']['min_rating_found']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
