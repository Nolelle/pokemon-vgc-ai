"""Run local matches between two registered baseline players against the local
Showdown server (`node pokemon-showdown start --no-security`, see CLAUDE.md).

    .venv/bin/python offline/run_matches.py --p1 random --p2 maxpower --n 50 \
        --team teams/dev.packed.txt
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from vgc.baselines import make_player  # noqa: E402
from vgc.config import FORMAT_ID, RUNS_DIR, TEAMS_DIR  # noqa: E402

DEFAULT_OUTPUT_DIR = RUNS_DIR / "eval"


def wilson_interval(wins: int, games: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval -- same formula as pokemon-tcg-ai's offline/run_matches.py."""
    if games <= 0:
        return 0.0, 0.0
    p = wins / games
    z2 = z * z
    denom = 1.0 + z2 / games
    center = (p + z2 / (2.0 * games)) / denom
    margin = (z / denom) * math.sqrt((p * (1.0 - p) / games) + (z2 / (4.0 * games * games)))
    return max(0.0, center - margin), min(1.0, center + margin)


async def run_matches(
    p1_name: str, p2_name: str, n: int, team: str, battle_format: str = FORMAT_ID
) -> dict[str, object]:
    p1 = make_player(p1_name, team, battle_format)
    p2 = make_player(p2_name, team, battle_format)
    try:
        await p1.battle_against(p2, n_battles=n)
    finally:
        # Always close both websocket connections, even if battling raised, so the
        # asyncio event loop and the local server's connection table don't leak.
        await p1.ps_client.stop_listening()
        await p2.ps_client.stop_listening()

    games = p1.n_finished_battles
    p1_wins = p1.n_won_battles
    p2_wins = p2.n_won_battles
    draws = games - p1_wins - p2_wins
    p1_low, p1_high = wilson_interval(p1_wins, games)
    p2_low, p2_high = wilson_interval(p2_wins, games)

    return {
        "p1": p1_name,
        "p2": p2_name,
        "format": battle_format,
        "games": games,
        "p1_wins": p1_wins,
        "p2_wins": p2_wins,
        "draws": draws,
        "p1_win_rate": p1_wins / games if games else 0.0,
        "p2_win_rate": p2_wins / games if games else 0.0,
        "p1_wilson": [p1_low, p1_high],
        "p2_wilson": [p2_low, p2_high],
        "timestamp": datetime.now(UTC).isoformat(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p1", required=True, help="baseline name, e.g. random")
    parser.add_argument("--p2", required=True, help="baseline name, e.g. maxpower")
    parser.add_argument("--n", type=int, default=50, help="number of battles")
    parser.add_argument(
        "--team",
        type=Path,
        default=TEAMS_DIR / "dev.packed.txt",
        help="path to a packed or paste-format Showdown team file",
    )
    parser.add_argument("--format", default=FORMAT_ID)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    team = args.team.read_text().strip()

    result = asyncio.run(run_matches(args.p1, args.p2, args.n, team, args.format))

    print(
        f"{result['p1']} vs {result['p2']}: "
        f"{result['p1_wins']}-{result['p2_wins']}-{result['draws']} of {result['games']} games"
    )
    print(
        f"  {result['p1']} win rate: {result['p1_win_rate']:.3f} "
        f"(95% CI [{result['p1_wilson'][0]:.3f}, {result['p1_wilson'][1]:.3f}])"
    )
    print(
        f"  {result['p2']} win rate: {result['p2_win_rate']:.3f} "
        f"(95% CI [{result['p2_wilson'][0]:.3f}, {result['p2_wilson'][1]:.3f}])"
    )

    output_dir = args.output.parent if args.output else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output or (
        output_dir / f"{result['p1']}_vs_{result['p2']}_{int(datetime.now(UTC).timestamp())}.json"
    )
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
