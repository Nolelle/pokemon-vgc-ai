"""Acceptance gate: candidate vs incumbent baseline, Wilson-CI lower bound must clear a
threshold before a change is allowed to ship. Ported from pokemon-tcg-ai's
offline/run_gates.py Wilson-CI pattern.

Either point this at an existing results JSON (from offline/run_matches.py) or let it
run the matches itself:

    .venv/bin/python offline/run_gates.py --candidate vgc --incumbent random --n 100 \
        --threshold 0.55 --team teams/dev.packed.txt

    .venv/bin/python offline/run_gates.py --results runs/eval/vgc_vs_random_123.json \
        --threshold 0.55
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from offline.run_matches import run_matches, wilson_interval  # noqa: E402
from vgc.config import FORMAT_ID, RUNS_DIR, TEAMS_DIR  # noqa: E402

DEFAULT_OUTPUT = RUNS_DIR / "eval" / "acceptance_gates.json"


def gate_from_result(result: dict[str, object], threshold: float) -> dict[str, object]:
    wins = int(result["p1_wins"])  # type: ignore[arg-type]
    games = int(result["games"])  # type: ignore[arg-type]
    rate = wins / games if games else 0.0
    low, high = wilson_interval(wins, games)
    passed = low > threshold
    gate = {
        "name": f"{result['p1']}_vs_{result['p2']}",
        "passed": passed,
        "games": games,
        "wins": wins,
        "win_rate": rate,
        "wilson": [low, high],
        "threshold": f"wilson_low > {threshold}",
    }
    if "open_team_sheets" in result:
        gate["open_team_sheets"] = result["open_team_sheets"]
    return gate


def run_gate(
    candidate: str,
    incumbent: str,
    n: int,
    team: str,
    threshold: float,
    battle_format: str = FORMAT_ID,
    *,
    accept_open_team_sheet: bool = False,
) -> dict[str, object]:
    result = asyncio.run(
        run_matches(
            candidate,
            incumbent,
            n,
            team,
            battle_format,
            accept_open_team_sheet=accept_open_team_sheet,
        )
    )
    return gate_from_result(result, threshold)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", help="baseline name to gate, e.g. vgc")
    parser.add_argument("--incumbent", help="baseline name to gate against, e.g. random")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--team", type=Path, default=TEAMS_DIR / "dev.packed.txt")
    parser.add_argument("--format", default=FORMAT_ID)
    parser.add_argument(
        "--open-team-sheets",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="make both players accept OTS; rejected by default to match public ladder play",
    )
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument(
        "--results", type=Path, default=None, help="reuse an existing run_matches.py result JSON"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.results is not None:
        result = json.loads(args.results.read_text())
        gate = gate_from_result(result, args.threshold)
    else:
        if not args.candidate or not args.incumbent:
            print("--candidate and --incumbent are required unless --results is given", file=sys.stderr)
            return 2
        team = args.team.read_text().strip()
        gate = run_gate(
            args.candidate,
            args.incumbent,
            args.n,
            team,
            args.threshold,
            args.format,
            accept_open_team_sheet=args.open_team_sheets,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(gate, indent=2, sort_keys=True))
    print(json.dumps(gate, indent=2))
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
