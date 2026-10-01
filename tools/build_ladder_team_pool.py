#!/usr/bin/env python
"""Build an evaluation pool of REAL ladder teams from the `|showteam|` lines in the
downloaded replay corpus (`tools/download_replays.py`'s output).

A `|showteam|` line appears when both players accept Open Team Sheets. It gives the
exact species, item, ability, moves and nature, but NOT the Stat Point spread (sheets
hide it). The spread is filled from `data/usage/spreads.json`: the most popular tracked
spread whose nature matches the sheet, else the most popular one, else
`vgc.stats.default_opponent_spread`. Item/ability/nature/moves are never changed.

One team per player (their most recently uploaded sheet): a player's sheets are usually
near-copies, and the multi-team gate clusters by team file, so keeping several would
overstate how many independent teams the pool has.

Every team is checked with the real Showdown `validate-team` CLI. A failing team gets
one retry with `default_opponent_spread` for every member, then is dropped and reported.

`archetype` in the manifest is a coarse, deterministic tag (weather setter, else Trick
Room, else Tailwind, else `other`) used only for the gate's per-subgroup report.

Output: `<out>/team_NNN.packed.txt` plus `<out>/manifest.json` (same contract as
`tools/build_archetype_pool.py`). Gitignored like the other pools.

Usage:
    .venv/bin/python tools/build_ladder_team_pool.py
    .venv/bin/python tools/build_ladder_team_pool.py --min-rating 1300 \
        --out data/selfplay/mc_sheet_pool
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vgc.config import FORMAT_ID, SHOWDOWN_REPO  # noqa: E402
from vgc.damage import to_id  # noqa: E402
from vgc.node import find_node, node_environment  # noqa: E402
from vgc.sets import load_usage_spreads  # noqa: E402
from vgc.stats import default_opponent_spread  # noqa: E402

DEFAULT_REPLAYS_DIR = REPO_ROOT / "data" / "replays" / FORMAT_ID
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "selfplay" / "mc_sheet_pool"
DEFAULT_MIN_RATING = 1200
STAT_ORDER = ("hp", "atk", "def", "spa", "spd", "spe")
# Packed-format field positions (NICKNAME|SPECIES|ITEM|ABILITY|MOVES|NATURE|EVS|...).
_SPECIES, _NATURE, _EVS = 1, 5, 6

_WEATHER_ABILITIES = {
    "drizzle": "rain",
    "drought": "sun",
    "sandstream": "sand",
    "snowwarning": "snow",
}
_WEATHER_MEGA_STONES = {"charizarditey": "sun", "tyranitarite": "sand", "abomasite": "snow"}


def extract_sheets(replays_dir: Path, min_rating: int) -> dict[str, dict[str, Any]]:
    """Latest `|showteam|` team per player id, from replays rated >= `min_rating`."""
    latest: dict[str, dict[str, Any]] = {}
    for path in sorted(replays_dir.glob("*-*.json")):
        replay = json.loads(path.read_text())
        rating = replay.get("rating")
        if rating is None or rating < min_rating:
            continue
        names: dict[str, str] = {}
        for line in replay.get("log", "").split("\n"):
            if line.startswith("|player|"):
                parts = line.split("|")
                if len(parts) > 3 and parts[3]:
                    names[parts[2]] = parts[3]
            elif line.startswith("|showteam|"):
                _, _, side, team = line.split("|", 3)
                player = to_id(names.get(side, ""))
                if not player:
                    continue
                uploaded = replay.get("uploadtime", 0)
                if player not in latest or uploaded > latest[player]["uploadtime"]:
                    latest[player] = {
                        "team": team,
                        "uploadtime": uploaded,
                        "replay_id": replay.get("id"),
                        "rating": rating,
                    }
    return latest


def _species_id(fields: list[str]) -> str:
    return to_id(fields[_SPECIES] or fields[0])


def _pick_spread(species_id: str, nature: str, spreads: dict[str, list[dict[str, Any]]]):
    entries = spreads.get(species_id) or []
    matching = [entry for entry in entries if to_id(entry.get("nature")) == nature]
    pool = matching or entries
    if pool:
        return dict(max(pool, key=lambda entry: entry.get("weight", 0))["sp"])
    return default_opponent_spread(species_id)


def _format_evs(sp: dict[str, int]) -> str:
    return ",".join(str(sp.get(stat, 0) or "") for stat in STAT_ORDER)


def fill_spreads(team: str, spreads: dict[str, list[dict[str, Any]]], *, default: bool) -> str:
    """`team` with every member's empty Stat Point field filled (see module docstring)."""
    members = []
    for member in team.split("]"):
        fields = member.split("|")
        species_id = _species_id(fields)
        sp = (
            default_opponent_spread(species_id)
            if default
            else _pick_spread(species_id, to_id(fields[_NATURE]), spreads)
        )
        fields[_EVS] = _format_evs(sp)
        members.append("|".join(fields))
    return "]".join(members)


def archetype_tag(team: str) -> str:
    """Coarse deterministic tag: weather, else trickroom, else tailwind, else other."""
    members = [member.split("|") for member in team.split("]")]
    for fields in members:
        weather = _WEATHER_ABILITIES.get(to_id(fields[3])) or _WEATHER_MEGA_STONES.get(
            to_id(fields[2])
        )
        if weather:
            return weather
    moves = {to_id(move) for fields in members for move in fields[4].split(",")}
    if "trickroom" in moves:
        return "trickroom"
    if "tailwind" in moves:
        return "tailwind"
    return "other"


def validate_team(packed_team: str, node: str) -> str | None:
    """Real `validate-team` CLI; `None` if legal, else its output (never raises)."""
    try:
        result = subprocess.run(
            [node, "pokemon-showdown", "validate-team", FORMAT_ID],
            cwd=str(SHOWDOWN_REPO),
            input=packed_team,
            capture_output=True,
            text=True,
            env=node_environment(node),
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"validate-team invocation failed: {exc}"
    output = (result.stdout + result.stderr).strip()
    return output or None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--replays-dir", type=Path, default=DEFAULT_REPLAYS_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-rating", type=int, default=DEFAULT_MIN_RATING)
    args = parser.parse_args()

    node = find_node()
    spreads = load_usage_spreads()
    sheets = extract_sheets(args.replays_dir, args.min_rating)
    args.out.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    dropped: list[tuple[str, str]] = []
    retried = 0
    for player in sorted(sheets):
        sheet = sheets[player]
        packed = fill_spreads(sheet["team"], spreads, default=False)
        error = validate_team(packed, node)
        if error:
            packed = fill_spreads(sheet["team"], spreads, default=True)
            error = validate_team(packed, node)
            retried += 1
        if error:
            dropped.append((player, error.splitlines()[0] if error else ""))
            continue
        file_name = f"team_{len(manifest):03d}.packed.txt"
        (args.out / file_name).write_text(packed + "\n")
        manifest.append(
            {
                "file": file_name,
                "archetype": archetype_tag(packed),
                "source": "showteam",
                "replay_id": sheet["replay_id"],
                "rating": sheet["rating"],
                "species": [_species_id(member.split("|")) for member in packed.split("]")],
            }
        )

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    tags: dict[str, int] = {}
    for entry in manifest:
        tags[entry["archetype"]] = tags.get(entry["archetype"], 0) + 1
    print(f"players with a sheet: {len(sheets)}")
    print(f"teams written:        {len(manifest)} ({args.out / 'manifest.json'})")
    print(f"default-spread retry: {retried}")
    print(f"dropped (illegal):    {len(dropped)}")
    for player, reason in dropped:
        print(f"  {player}: {reason}")
    print(f"archetype tags:       {dict(sorted(tags.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
