#!/usr/bin/env python
"""Measure shipped-search decision latency on live local games.

Plays --n games of the shipped default (`vgc` baseline: 2-ply search + rolling
horizon) against a configurable opponent on the local server, timing every
`decide()`/`decide_teampreview()` call. Prints the decision-time distribution
(p50/p99/max) plus fallbacks and unfinished games.

Server must already be running (see CLAUDE.md). Example:

    .venv/bin/python offline/measure_search_latency.py --n 10 \
        --team teams/meta1.packed.txt --out runs/eval/search_latency.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from vgc.agent import VgcPlayer  # noqa: E402
from vgc.baselines import make_player  # noqa: E402
from vgc.config import FORMAT_ID, TEAMS_DIR  # noqa: E402
from vgc.models import PolicyConfig  # noqa: E402


class TimedPlayer(VgcPlayer):
    """VgcPlayer that records wall time of every strategy call."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.decision_seconds: list[float] = []
        self.preview_seconds: list[float] = []

    def decide(self, battle):
        started = time.perf_counter()
        try:
            return super().decide(battle)
        finally:
            self.decision_seconds.append(time.perf_counter() - started)

    def decide_teampreview(self, battle) -> str:
        started = time.perf_counter()
        try:
            return super().decide_teampreview(battle)
        finally:
            self.preview_seconds.append(time.perf_counter() - started)


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    rank = min(len(sorted_values) - 1, int(round(pct * (len(sorted_values) - 1))))
    return sorted_values[rank]


async def _play(n: int, team: str, opponent: str) -> dict[str, object]:
    config = PolicyConfig(format_id=FORMAT_ID)
    assert config.use_two_ply_search and config.use_rolling_horizon
    p1 = TimedPlayer(config=config, team=team, battle_format=FORMAT_ID)
    p2 = make_player(opponent, team, FORMAT_ID)
    try:
        await asyncio.wait_for(p1.battle_against(p2, n_battles=n), timeout=60 * n)
    finally:
        await p1.ps_client.stop_listening()
        await p2.ps_client.stop_listening()
    decisions = sorted(p1.decision_seconds)
    previews = sorted(p1.preview_seconds)
    return {
        "games_requested": n,
        "games_finished": p1.n_finished_battles,
        "wins": p1.n_won_battles,
        "fallbacks": p1.fallback_count,
        "decisions": len(decisions),
        "decision_s_mean": statistics.fmean(decisions) if decisions else 0.0,
        "decision_s_p50": _percentile(decisions, 0.50),
        "decision_s_p99": _percentile(decisions, 0.99),
        "decision_s_max": max(decisions) if decisions else 0.0,
        "preview_s_max": max(previews) if previews else 0.0,
        "preview_s_p99": _percentile(previews, 0.99),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--team", default=str(TEAMS_DIR / "meta1.packed.txt"))
    parser.add_argument("--opponent", default="heuristic")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    team = Path(args.team).read_text().strip()
    report = asyncio.run(_play(args.n, team, args.opponent))
    report["opponent"] = args.opponent
    report["team_file"] = args.team
    print(json.dumps(report, indent=2))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n")
    ok = report["games_finished"] == args.n and report["fallbacks"] == 0
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
