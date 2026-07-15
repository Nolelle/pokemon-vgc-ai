#!/usr/bin/env python
"""Build data/usage/spreads.json (per-species top Stat Point spreads) from a raw Smogon
chaos usage-stats export.

`vgc.sets.opponent_state` uses data/usage/spreads.json to guess an opponent's hidden
Stat Point spread/nature (Open Team Sheets reveal species/item/ability/moves but never
Stat Points -- see vgc/sets.py's module docstring) instead of falling back to
`vgc.stats.default_opponent_spread`/`default_opponent_nature`'s generic heuristic.

Input: a standard Smogon chaos-stats JSON (`{"data": {species: {"Spreads": {"Nature:
hp/atk/def/spa/spd/spe": weight, ...}, ...}, ...}}`) -- see
https://www.smogon.com/stats/ for the general shape; this project's copy is downloaded
manually into data/usage/ (there's no in-repo scraper) since it's regenerated
infrequently and doesn't need to track the live ladder in real time.

Output schema (consumed by `vgc.sets.load_usage_spreads`), per id-normalized species,
the top `--top-n` spreads by usage weight:

    {"garchomp": [{"sp": {"hp": 2, "atk": 32, "def": 0, "spa": 0, "spd": 0, "spe": 32},
                    "nature": "jolly", "weight": 0.42}, ...], ...}

`weight` here is normalized to that spread's *share* of the species' total observed
spread weight (0-1, sums to <=1 across all of a species' spreads, not just the kept top
N) -- e.g. 0.42 means "42% of recorded Garchomp sightings ran this exact spread". This
is more meaningful than the chaos file's raw (unnormalized, battle-count-weighted)
numbers, and `vgc.sets._usage_spread_for_species` only needs relative ordering (it picks
max-weight) so any monotonic rescaling would have worked -- this one is chosen because
it's also directly readable as a probability if a future caller wants to sample instead
of argmax.

Usage:
    .venv/bin/python tools/build_usage_spreads.py
    .venv/bin/python tools/build_usage_spreads.py --input data/usage/other.json --top-n 5
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vgc.config import DATA_DIR  # noqa: E402
from vgc.damage import to_id  # noqa: E402
from vgc.stats import STAT_IDS  # noqa: E402

DEFAULT_INPUT = REPO_ROOT / "data" / "usage" / "gen9championsvgc2026regmb-1760.json"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "usage" / "spreads.json"
DEFAULT_TOP_N = 3

# "Nature:hp/atk/def/spa/spd/spe" -- the chaos file's per-spread key format. Stat order
# matches the standard EV-order convention Smogon chaos stats use across every format
# (HP/Atk/Def/SpA/SpD/Spe), which is also STAT_IDS' order here.
_SPREAD_KEY_RE = re.compile(r"^([A-Za-z]+):(\d+)/(\d+)/(\d+)/(\d+)/(\d+)/(\d+)$")


def parse_spread_key(key: str) -> tuple[str, dict[str, int]] | None:
    """Parse one "Nature:h/a/d/sa/sd/s" chaos-file key into (nature_id, sp_spread).

    Returns None for a key that doesn't match the expected shape (defensive -- chaos
    files occasionally carry odd/legacy entries; skip rather than crash the whole build).
    """
    match = _SPREAD_KEY_RE.match(key)
    if match is None:
        return None
    nature_name = match.group(1)
    values = [int(v) for v in match.groups()[1:]]
    sp = dict(zip(STAT_IDS, values, strict=True))
    return to_id(nature_name), sp


def build_spreads(raw: dict, top_n: int) -> dict[str, list[dict[str, object]]]:
    result: dict[str, list[dict[str, object]]] = {}
    for species_name, entry in raw.get("data", {}).items():
        raw_spreads = entry.get("Spreads") or {}
        parsed: list[tuple[str, dict[str, int], float]] = []
        for key, weight in raw_spreads.items():
            if weight <= 0:
                continue
            parsed_key = parse_spread_key(key)
            if parsed_key is None:
                continue
            nature, sp = parsed_key
            parsed.append((nature, sp, float(weight)))
        if not parsed:
            continue

        total_weight = sum(weight for _, _, weight in parsed)
        if total_weight <= 0:
            continue

        parsed.sort(key=lambda item: item[2], reverse=True)
        top = parsed[:top_n]

        species_id = to_id(species_name)
        result[species_id] = [
            {"sp": sp, "nature": nature, "weight": weight / total_weight}
            for nature, sp, weight in top
        ]
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="raw Smogon chaos-stats JSON")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N, help="spreads to keep per species")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        print(f"{args.input} not found", file=sys.stderr)
        return 1

    with args.input.open() as file:
        raw = json.load(file)

    spreads = build_spreads(raw, args.top_n)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(spreads, indent=2, sort_keys=True))

    print(f"wrote {len(spreads)} species to {args.output}")
    # Spot-check: report how many species from data/champions/species.json (the format's
    # actual legal pool) got a usage spread, since the chaos file is drawn from the live
    # ladder and may not cover every legal species evenly.
    if DATA_DIR.joinpath("species.json").exists():
        from vgc.data import load_species

        species_ids = set(load_species().keys())
        covered = species_ids & set(spreads)
        print(f"covers {len(covered)}/{len(species_ids)} known species ids")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
