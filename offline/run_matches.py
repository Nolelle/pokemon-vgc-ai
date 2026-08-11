"""Run local matches between two registered baseline players against the local
Showdown server (`node pokemon-showdown start --no-security`, see CLAUDE.md).

    .venv/bin/python offline/run_matches.py --p1 random --p2 maxpower --n 50 \
        --team teams/dev.packed.txt
"""

from __future__ import annotations

import argparse
import asyncio
import json
import secrets
import sys
from datetime import UTC, datetime
from pathlib import Path

from poke_env.ps_client.account_configuration import AccountConfiguration

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from vgc.baselines import make_player  # noqa: E402
from vgc.config import FORMAT_ID, RUNS_DIR, TEAMS_DIR  # noqa: E402
from vgc.evaluation import wilson_interval  # noqa: E402

DEFAULT_OUTPUT_DIR = RUNS_DIR / "eval"


# Re-exported from vgc.evaluation, which is where it lives now so that vgc.rl.match can
# share it without src/vgc/ importing this scripts package. Kept importable from here:
# offline.run_gates and any existing analysis still do `from offline.run_matches import
# wilson_interval`.
__all__ = ["run_matches", "wilson_interval"]


async def run_matches(
    p1_name: str,
    p2_name: str,
    n: int,
    team: str,
    battle_format: str = FORMAT_ID,
    *,
    accept_open_team_sheet: bool = True,
) -> dict[str, object]:
    # Keep both sides on the same OTS setting. Besides making comparisons fair, this
    # avoids accidentally benchmarking an accept/reject protocol race instead of the
    # two policies. VgcPlayer itself still handles a real opponent rejecting on ladder.
    token = secrets.token_hex(3)
    common_kwargs = {"accept_open_team_sheet": accept_open_team_sheet}
    p1 = make_player(
        p1_name,
        team,
        battle_format,
        account_configuration=AccountConfiguration(f"{p1_name[:8]}-{token}a", None),
        **common_kwargs,
    )
    p2 = make_player(
        p2_name,
        team,
        battle_format,
        account_configuration=AccountConfiguration(f"{p2_name[:8]}-{token}b", None),
        **common_kwargs,
    )
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

    result: dict[str, object] = {
        "p1": p1_name,
        "p2": p2_name,
        "format": battle_format,
        "open_team_sheets": "accept" if accept_open_team_sheet else "reject",
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
    for label, player in (("p1", p1), ("p2", p2)):
        traces = getattr(player, "decision_trace_history", None)
        if traces:
            result[f"{label}_diagnostics"] = {
                "battles": {
                    tag: {"won": battle.won, "turns": battle.turn}
                    for tag, battle in player.battles.items()
                },
                "decisions": traces,
            }
    return result


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
    parser.add_argument(
        "--open-team-sheets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="make both players accept OTS (default); use --no-open-team-sheets for both to reject",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--direct",
        action="store_true",
        help=(
            "run on the direct BattleStream environment (vgc.rl.match) instead of a "
            "local Showdown server: no server, no websocket, no accounts, and the "
            "n>=500 evaluation budget becomes practical. Same decision code either way "
            "-- see vgc/rl/agents.py. --open-team-sheets does not apply (the direct env "
            "reveals nothing beyond the simulator's own fog)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="simulator seed stream for --direct, making the series reproducible",
    )
    return parser.parse_args()


def run_direct(args: argparse.Namespace, team: str) -> dict[str, object]:
    """The --direct path: same result shape, no server."""

    from functools import partial

    from vgc.rl.agents import make_direct_agent
    from vgc.rl.env import SimWorker
    from vgc.rl.match import run_series, summarize

    # Distinct dict keys are required (run_series is keyed by agent name), so a mirror
    # like --p1 vgc --p2 vgc needs disambiguating before it reaches the runner.
    p1_name, p2_name = args.p1, args.p2
    if p1_name == p2_name:
        p1_name, p2_name = f"{p1_name}-a", f"{p2_name}-b"
    factories = {
        p1_name: partial(make_direct_agent, args.p1, team, battle_format=args.format),
        p2_name: partial(make_direct_agent, args.p2, team, battle_format=args.format),
    }
    with SimWorker() as worker:
        outcomes = run_series(
            worker,
            factories,
            {p1_name: team, p2_name: team},
            args.n,
            battle_format=args.format,
            seed=args.seed,
        )
    return summarize(outcomes, p1_name, p2_name, battle_format=args.format)


def main() -> int:
    args = parse_args()
    team = args.team.read_text().strip()

    if args.direct:
        result = run_direct(args, team)
    else:
        result = asyncio.run(
            run_matches(
                args.p1,
                args.p2,
                args.n,
                team,
                args.format,
                accept_open_team_sheet=args.open_team_sheets,
            )
        )

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
