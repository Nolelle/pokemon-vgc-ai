#!/usr/bin/env python
"""Compare one-point opponent guesses with the Part B multi-belief search path."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vgc.actions import describe_order  # noqa: E402
from vgc.battle_memory import BattleMemory  # noqa: E402
from vgc.models import PolicyConfig  # noqa: E402
from vgc.rl.env import DEFAULT_SHOWDOWN_REPO, DirectBattle, SimWorker  # noqa: E402
from vgc.rl.exact_search import combine_belief_rankings, search_joint_orders_exact  # noqa: E402
from vgc.rl.live_mirror import LiveExactMirror, MirrorHypothesis  # noqa: E402


def reduced_width_config(**overrides) -> PolicyConfig:
    values = dict(
        search_our_candidates=2,
        search_opp_candidates=2,
        exact_search_future_samples=1,
        exact_search_state_hypotheses=1,
        use_rolling_horizon=False,
        use_value_head=False,
    )
    values.update(overrides)
    return replace(PolicyConfig(), **values)


def _rank(
    view,
    own_team: str,
    config: PolicyConfig,
    *,
    check_sensitivity: bool = False,
) -> tuple[list, dict, float]:
    mirror = LiveExactMirror(own_team, config)
    root = None
    cache: dict[int, list] = {}
    started = time.perf_counter()
    try:
        beliefs = mirror.hypotheses(view, BattleMemory("belief-audit"))

        def rank_fn(hypothesis: MirrorHypothesis) -> list:
            nonlocal root
            cached = cache.get(id(hypothesis))
            if cached is not None:
                return cached
            root = (
                mirror.rebase(root, view, hypothesis)
                if root is not None
                else mirror.build(view, hypothesis)
            )
            ranked = search_joint_orders_exact(root, "p1", config)
            cache[id(hypothesis)] = ranked
            return ranked

        rankings = [(belief.weight, rank_fn(belief)) for belief in beliefs]
        combined = combine_belief_rankings(rankings)
        if check_sensitivity:
            mirror.record_branch_sensitivity(rank_fn, beliefs)
        audit = dict(mirror.last_hypothesis_audit)
        checked = int(audit.get("excluded_branches_checked") or 0)
        flipping = int(audit.get("flipping_branches") or 0)
        if check_sensitivity:
            audit["flip_rate"] = (flipping / checked) if checked else 0.0
        return combined, audit, time.perf_counter() - started
    finally:
        if root is not None:
            root.close()
        mirror.close()


def run_belief_audit(own_team: str, opponent_team: str) -> dict:
    diagnostic = reduced_width_config()
    point = reduced_width_config(
        exact_search_spread_hypotheses=1,
        exact_search_set_hypotheses=1,
        exact_search_bring_hypotheses=1,
        exact_search_total_hypotheses=1,
    )
    with SimWorker(DEFAULT_SHOWDOWN_REPO) as worker:
        root = DirectBattle.start(
            worker, "battle-state-belief-audit", own_team, opponent_team, seed=[7, 11, 13, 17]
        )
        try:
            root.step({"p1": "team 1234", "p2": "team 1234"})
            view = root.battles["p1"]
            point_rank, point_audit, point_seconds = _rank(view, own_team, point)
            multi_rank, multi_audit, multi_seconds = _rank(
                view, own_team, diagnostic, check_sensitivity=True
            )
        finally:
            root.close()
    return {
        "point": {
            "top_order": describe_order(point_rank[0].order),
            "seconds": point_seconds,
            "audit": point_audit,
        },
        "multi": {
            "top_order": describe_order(multi_rank[0].order),
            "seconds": multi_seconds,
            "audit": multi_audit,
        },
        "top_order_changed": describe_order(point_rank[0].order)
        != describe_order(multi_rank[0].order),
        "slowdown_factor": multi_seconds / point_seconds if point_seconds else None,
        "flip_rate": multi_audit.get("flip_rate"),
        "flipping_mass": multi_audit.get("flipping_mass"),
        "flipping_branches": multi_audit.get("flipping_branches"),
        "excluded_branches_checked": multi_audit.get("excluded_branches_checked"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--our-team", type=Path, default=REPO_ROOT / "teams/meta1.packed.txt")
    parser.add_argument(
        "--opponent-team", type=Path, default=REPO_ROOT / "teams/regmb_rain_balance.packed.txt"
    )
    parser.add_argument(
        "--output", type=Path, default=REPO_ROOT / "runs/eval/battle_state_belief_audit.json"
    )
    args = parser.parse_args()
    own_team = args.our_team.read_text().strip()
    opponent_team = args.opponent_team.read_text().strip()
    result = run_belief_audit(own_team, opponent_team)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    checked = result.get("excluded_branches_checked") or 0
    flipping = result.get("flipping_branches") or 0
    mass = result.get("flipping_mass")
    rate = result.get("flip_rate")
    print(
        f"branch sensitivity: excluded={checked} flipping={flipping} "
        f"flip_rate={rate} flipping_mass={mass}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
