#!/usr/bin/env python
"""Measure exact-judge decision latency vs continuation depth, policy and search width.

Plays direct battles (no server needed) between two `ExactSearchPlayer`s on pool teams for
a grid of `(continuation turns N, continuation policy, width)` cells and reports per-
decision wall time (p50/p95/p99/max), decisions/game, thinking seconds/game and exact
fallbacks. Compare p99 to the live clock guard (12 s normal / 20 s endgame per decision).

    PYTHONPATH=$PWD/src .venv/bin/python offline/measure_exact_depth.py --games 2 \
        --turns-list 0,1,2,3 --policies myopic,search --widths narrow \
        --output runs/eval/exact_depth.json

Width "narrow" = search_our_candidates=4, search_opp_candidates=4,
exact_search_future_samples=1; "production" = PolicyConfig defaults. N=0 has one policy
(it is unused), so it is run once as "myopic".
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vgc.config import FORMAT_ID  # noqa: E402
from vgc.models import PolicyConfig  # noqa: E402
from vgc.rl.agents import make_direct_agent  # noqa: E402
from vgc.rl.env import SimWorker  # noqa: E402
from vgc.rl.match import run_series  # noqa: E402

MANIFEST = REPO_ROOT / "data" / "selfplay" / "archetype_pool_150" / "manifest.json"
WIDTHS: dict[str, dict[str, Any]] = {
    "narrow": {
        "search_our_candidates": 4,
        "search_opp_candidates": 4,
        "exact_search_future_samples": 1,
    },
    "production": {},
}


def percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    rank = min(len(sorted_values) - 1, int(round(pct * (len(sorted_values) - 1))))
    return sorted_values[rank]


class DecisionCapReached(BaseException):
    """Raised inside a timed decide() to end a slow cell early (partial sample)."""


def _timed_agent(
    times: list[float], built: list[Any], team: str, config: PolicyConfig, cap: int | None = None
):
    agent = make_direct_agent("vgc_exact", team, config=config)
    player = agent.player
    inner = player.decide

    def decide(battle):
        if cap is not None and len(times) >= cap:
            raise DecisionCapReached
        started = time.perf_counter()
        try:
            return inner(battle)
        finally:
            times.append(time.perf_counter() - started)

    player.decide = decide
    built.append(agent)
    return agent


def run_cell(cell: dict[str, Any]) -> dict[str, Any]:
    """Play `games` battles for one grid cell (runs inside a worker process)."""

    config = PolicyConfig(
        format_id=FORMAT_ID,
        exact_search_continuation_turns=cell["turns"],
        exact_search_continuation_policy=cell["policy"],
        **WIDTHS[cell["width"]],
    )
    pool = json.loads(Path(cell["manifest"]).read_text())
    times: list[float] = []
    built: list[Any] = []
    outcomes = []
    wall = time.perf_counter()
    cap = cell.get("decision_cap")
    with SimWorker() as worker:
        try:
            for game in range(cell["games"]):
                entry = pool[(cell["team_offset"] + game) % len(pool)]
                team = (Path(cell["manifest"]).parent / entry["file"]).read_text().strip()
                outcomes.extend(
                    run_series(
                        worker,
                        {
                            "a": partial(_timed_agent, times, built, team, config, cap),
                            "b": partial(_timed_agent, times, built, team, config, cap),
                        },
                        {"a": team, "b": team},
                        1,
                        seed=cell["seed"] + game,
                    )
                )
        except DecisionCapReached:
            pass  # partial sample: the cap is for cells whose full games take too long
        finally:
            # An aborted game never reaches the battle-finished callback that closes each
            # player's mirror worker; close them so leftover processes do not slow the
            # next cell's timings.
            for agent in built:
                close = getattr(agent.player, "close_public_mirror", None)
                if close is not None:
                    close()
    ordered = sorted(times)
    fallbacks = sum(getattr(a.player, "exact_fallbacks", 0) for a in built)
    exact = sum(getattr(a.player, "exact_decisions", 0) for a in built)
    games = max(1, len(outcomes))
    if cap is not None and not outcomes:  # capped before a game finished
        games = max(1.0, len(ordered) / 20.0)  # ~20 decisions per game, labelled below
    return {
        "turns": cell["turns"],
        "policy": cell["policy"],
        "width": cell["width"],
        "games": len(outcomes),
        "decision_cap": cap,
        "decisions": len(ordered),
        "decisions_per_game": len(ordered) / games,
        "p50_s": percentile(ordered, 0.50),
        "p95_s": percentile(ordered, 0.95),
        "p99_s": percentile(ordered, 0.99),
        "max_s": ordered[-1] if ordered else 0.0,
        "thinking_s_per_game": sum(ordered) / games,
        "exact_decisions": exact,
        "exact_fallbacks": fallbacks,
        "fallback_examples": [
            r for a in built for r in getattr(a.player, "exact_fallback_reasons", ())
        ][:3],
        "wall_s": time.perf_counter() - wall,
    }


def build_cells(args: argparse.Namespace) -> list[dict[str, Any]]:
    cells = []
    for width, turns, policy in itertools.product(args.widths, args.turns_list, args.policies):
        if turns == 0 and policy != args.policies[0]:
            continue  # policy is irrelevant without continuation
        if width == "production" and args.production_max_turns is not None:
            if turns > args.production_max_turns:
                continue
        cells.append(
            {
                "turns": turns,
                "policy": "myopic" if turns == 0 else policy,
                "width": width,
                "games": args.games,
                "seed": args.seed,
                "team_offset": args.team_offset,
                "manifest": str(args.manifest),
                "decision_cap": args.decision_cap,
            }
        )
    return cells


def print_table(rows: list[dict[str, Any]]) -> None:
    head = (
        f"{'width':<10}{'N':>2} {'policy':<7}{'dec/g':>7}{'p50':>7}{'p95':>7}{'p99':>7}"
        f"{'max':>7}{'think/g':>9}{'fallbk':>7}"
    )
    print(head)
    for r in rows:
        print(
            f"{r['width']:<10}{r['turns']:>2} {r['policy']:<7}{r['decisions_per_game']:>7.1f}"
            f"{r['p50_s']:>7.2f}{r['p95_s']:>7.2f}{r['p99_s']:>7.2f}{r['max_s']:>7.2f}"
            f"{r['thinking_s_per_game']:>9.1f}{r['exact_fallbacks']:>4}/{r['exact_decisions']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--games", type=int, default=2, help="games per cell")
    parser.add_argument("--turns-list", default="0,1,2,3")
    parser.add_argument("--policies", default="myopic,search")
    parser.add_argument("--widths", default="narrow,production")
    parser.add_argument(
        "--production-max-turns",
        type=int,
        default=None,
        help="skip production-width cells with more continuation turns than this",
    )
    parser.add_argument(
        "--decision-cap",
        type=int,
        default=None,
        help="stop each cell after this many timed decisions (partial sample for slow cells)",
    )
    parser.add_argument("--workers", type=int, default=1, help="parallel processes")
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--team-offset", type=int, default=0)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument(
        "--output", type=Path, default=REPO_ROOT / "runs" / "eval" / "exact_depth.json"
    )
    args = parser.parse_args()
    args.turns_list = [int(v) for v in args.turns_list.split(",")]
    args.policies = args.policies.split(",")
    args.widths = args.widths.split(",")
    cells = build_cells(args)
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(run_cell, cells))
    else:
        rows = [run_cell(cell) for cell in cells]
    print_table(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"args": vars(args), "cells": rows}, indent=2, default=str))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
