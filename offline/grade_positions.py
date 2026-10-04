#!/usr/bin/env python
"""Grade saved positions with the exact Showdown engine.

For each position written by ``offline/record_positions.py`` this rebuilds the public
battle from the saved player-view bundle (the same replay technique as
``offline/review_lost_decisions.py``), builds a ``LiveExactMirror`` root under the
opponent-belief model live play uses, and scores with
``search_joint_orders_exact(..., candidate_selector=...)``:

* (a) the engine's chosen order, and
* (b) a list of extra candidate orders ("proposals") for that position.

Only those orders are searched (the selector forces exactly that set), every order shares
one random-future key (common random numbers), and the per-order value is averaged over
the belief branches.  Per position it reports each order's exact value, the best order,
and whether any proposal beat the engine's pick and by how much.  The summary
reports three clearly labelled shares, each clustered by OPPONENT TEAM
(``vgc.evaluation.clustered_mean``), per team-disjoint split: the OVERALL share of all graded
positions where a proposal beats the engine by more than a margin (positions with no usable
proposal count as no improvement), the proposal COVERAGE (share with at least one legal,
distinct proposal), and the CONDITIONAL share among covered positions.  "Decided" positions
(engine's own pick ends in a wipe) are classified from the engine's pick only.

Proposals
---------
``--proposals engine-rank-2..4``   grade the engine's own 2nd..4th choices (no LLM; this
                                   exercises the grader and measures how often the exact
                                   engine disagrees with the shipped search ranking)
``--proposals file.jsonl``         rows ``{"position_id": ..., "orders": [...]}`` where an
                                   order is a canonical ``describe_order`` string (e.g.
                                   ``"rockslide / earthquake"``) or a wire string
                                   (``"/choose move rockslide, move earthquake"``).
                                   Orders illegal in the position are reported, not scored.

Value
-----
``--metric exchange_value`` (default) is the pure exact one-turn position change from the
Showdown branches (about 1 point per 1% HP; a wiped side is +-10,000, so read the capped
mean or the shares, not the raw mean).  ``--metric score`` is the engine's final blend
(myopic term + exact term).  Both are always written per order.

Width
-----
Full width costs 2-28 s per decision.  Defaults here are diagnostic (``--future-samples 2``,
``--opp-candidates 4``, one belief branch); ``--production-width`` uses PolicyConfig's
defaults.  Per-position seconds are reported.  A one-turn exact search with a hand-weighted
value is a filter before full games, not a win-rate measurement.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vgc.actions import choice_wire_message, describe_order  # noqa: E402
from vgc.evaluation import clustered_mean  # noqa: E402
from vgc.evaluator import score_joint_orders  # noqa: E402
from vgc.models import PolicyConfig  # noqa: E402
from vgc.positions import load_bundle, rebuild_position  # noqa: E402
from vgc.rl.exact_search import search_joint_orders_exact  # noqa: E402
from vgc.rl.live_mirror import LiveExactMirror  # noqa: E402

METRICS = ("exchange_value", "score")
DECIDED_VALUE = 5000.0
CAP = 300.0  # for the capped mean: terminal-position values are +-10,000, not HP points


def parse_rank_spec(spec: str) -> tuple[int, int] | None:
    match = re.fullmatch(r"engine-rank-(\d+)\.\.(\d+)", spec)
    if not match:
        return None
    low, high = int(match.group(1)), int(match.group(2))
    if low < 2 or high < low:
        raise SystemExit("engine-rank-A..B needs 2 <= A <= B")
    return low, high


def load_proposals(path: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            out.setdefault(str(row["position_id"]), []).extend(str(o) for o in row["orders"])
    return out


def grade_config(args: argparse.Namespace) -> PolicyConfig:
    if args.production_width:
        return PolicyConfig()
    return replace(
        PolicyConfig(),
        search_opp_candidates=args.opp_candidates,
        exact_search_future_samples=args.future_samples,
        exact_search_state_hypotheses=args.state_hypotheses,
        exact_search_spread_hypotheses=args.spread_hypotheses,
        exact_search_set_hypotheses=args.set_hypotheses,
        exact_search_bring_hypotheses=args.bring_hypotheses,
        exact_search_total_hypotheses=args.total_hypotheses,
        use_rolling_horizon=False,
        use_value_head=False,
    )


def grade_position(
    position: dict[str, Any],
    run_dir: Path,
    proposals: list[str],
    config: PolicyConfig,
    metric: str,
    margin: float,
) -> dict[str, Any]:
    """Score the engine's pick and the proposals for one position."""

    started = time.perf_counter()
    bundle = load_bundle(position, run_dir)
    battle, memory = rebuild_position(bundle, int(position["decision_index"]))
    rebuild_seconds = time.perf_counter() - started

    legal = score_joint_orders(battle, config)
    by_key: dict[str, Any] = {}
    for entry in legal:
        by_key[describe_order(entry.order)] = entry
        by_key[choice_wire_message(entry.order)] = entry

    engine_entry = by_key.get(position["engine_chosen"]) or by_key.get(
        position["engine_chosen_wire"]
    )
    if engine_entry is None:
        raise ValueError("engine's recorded order is not legal in the rebuilt position")
    engine_key = describe_order(engine_entry.order)

    roles: dict[str, str] = {engine_key: "engine"}
    illegal: list[str] = []
    for raw in proposals:
        entry = by_key.get(raw)
        if entry is None:
            illegal.append(raw)
            continue
        key = describe_order(entry.order)
        roles.setdefault(key, "proposal")
    forced = [by_key[key] for key in roles]
    wanted = {entry.order.message for entry in forced}

    def selector(ranked, cfg):
        searched = [entry for entry in ranked if entry.order.message in wanted]
        searched.sort(key=lambda entry: [e.order.message for e in forced].index(entry.order.message))
        rest = [entry for entry in ranked if entry.order.message not in wanted]
        return searched, rest

    search_cfg = replace(config, search_our_candidates=len(forced))
    mirror = LiveExactMirror(bundle["own_packed_team"], search_cfg)
    t_search = time.perf_counter()
    try:
        beliefs = mirror.hypotheses(battle, memory)
        weighted: list[tuple[float, list]] = []
        root = None
        try:
            for belief in beliefs:
                root = mirror.rebase(root, battle, belief) if root is not None else mirror.build(
                    battle, belief
                )
                decision_battle = getattr(root, "_decision_battles", {}).get("p1", root.battles["p1"])
                decision_battle._vgc_battle_memory = memory
                ranking = search_joint_orders_exact(
                    root,
                    "p1",
                    search_cfg,
                    candidate_selector=selector,
                    randomness_key=position["position_id"],
                )
                weighted.append((belief.weight, ranking))
        finally:
            if root is not None:
                root.close()
    finally:
        mirror.close()
    search_seconds = time.perf_counter() - t_search

    total = sum(weight for weight, _ in weighted)
    values: dict[str, dict[str, float]] = {}
    for weight, ranking in weighted:
        for entry in ranking:
            if not entry.breakdown.get("searched"):
                continue
            key = describe_order(entry.order)
            slot = values.setdefault(key, {"exchange_value": 0.0, "score": 0.0})
            slot["exchange_value"] += weight * float(entry.breakdown["exchange_value"]) / total
            slot["score"] += weight * float(entry.score) / total

    orders = []
    for key, role in roles.items():
        if key not in values:
            continue
        rank = next(
            (
                i + 1
                for i, row in enumerate(position["engine_top_k"])
                if row["order"] == key
            ),
            None,
        )
        orders.append(
            {
                "order": key,
                "role": role,
                "engine_rank": rank,
                "value": values[key][metric],
                "exchange_value": values[key]["exchange_value"],
                "score": values[key]["score"],
            }
        )
    engine_row = next(row for row in orders if row["role"] == "engine")
    proposal_rows = [row for row in orders if row["role"] == "proposal"]
    best_proposal = max(proposal_rows, key=lambda row: row["value"], default=None)
    best = max(orders, key=lambda row: row["value"])
    gain = best_proposal["value"] - engine_row["value"] if best_proposal else None
    return {
        "position_id": position["position_id"],
        "split": position["split"],
        "opp_team_id": position["opp_team_id"],
        "our_team": position["our_team"],
        "turn": position["turn"],
        "n_legal": position["n_legal"],
        "engine_order": engine_key,
        "orders": orders,
        "best_order": best["order"],
        "engine_value": engine_row["value"],
        "best_proposal_order": best_proposal["order"] if best_proposal else None,
        "best_proposal_value": best_proposal["value"] if best_proposal else None,
        "gain": gain,
        "proposal_beats_engine": bool(gain is not None and gain > margin),
        # A wiped side scores +-10,000: the position was already decided one way or the
        # other, so it says little about move choice. Classified from the ENGINE's own pick
        # only (a function of the starting position and the engine's order, never of any
        # proposal), so a proposal that finds a winning wipe cannot exclude its own success
        # from the contested-only summary.
        "decided": abs(engine_row["exchange_value"]) >= DECIDED_VALUE,
        "illegal_proposals": illegal,
        "belief_branches": len(beliefs),
        "rebuild_seconds": round(rebuild_seconds, 2),
        "search_seconds": round(search_seconds, 2),
        "total_seconds": round(time.perf_counter() - started, 2),
    }


def _job(job: tuple) -> dict[str, Any]:
    position, run_dir, proposals, config, metric, margin = job
    try:
        return grade_position(position, Path(run_dir), proposals, config, metric, margin)
    except Exception as exc:  # noqa: BLE001 - one bad position must not end the run
        return {
            "position_id": position["position_id"],
            "split": position["split"],
            "opp_team_id": position["opp_team_id"],
            "error": f"{type(exc).__name__}: {exc}",
            "trace": traceback.format_exc(limit=4),
        }


def _share(rows: list[dict[str, Any]], predicate) -> dict[str, Any]:
    clusters: dict[str, list[float]] = {}
    for row in rows:
        clusters.setdefault(row["opp_team_id"], []).append(1.0 if predicate(row) else 0.0)
    cm = clustered_mean(list(clusters.items()))
    half = 1.96 * cm.clustered_se
    return {
        "count": int(sum(sum(v) for v in clusters.values())),
        "of": sum(len(v) for v in clusters.values()),
        "share": cm.mean,
        "cluster_robust_ci95": [max(0.0, cm.mean - half), min(1.0, cm.mean + half)],
        "opponent_teams": cm.clusters,
        "team_effect_sd_tau": cm.tau,
    }


def summarize(results: list[dict[str, Any]], margin: float) -> dict[str, Any]:
    graded = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]

    def block(rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Three clearly separated shares, each clustered by opponent team.

        ``rows`` is EVERY graded position in the group.  A position with no usable
        proposal (none given, only the engine's own, or only illegal ones) has
        ``gain is None``: it stays in the overall denominator as "no improvement", so one
        improvement in 100 positions can never read as 100%.
        """

        if not rows:
            return {"positions": 0}
        covered = [r for r in rows if r["gain"] is not None]
        out: dict[str, Any] = {
            "positions": len(rows),
            "opponent_teams": len({r["opp_team_id"] for r in rows}),
            f"overall_share_proposal_beats_engine_by_gt_{margin:g}": _share(
                rows, lambda r: r["gain"] is not None and r["gain"] > margin
            ),
            "proposal_coverage_share": _share(rows, lambda r: r["gain"] is not None),
        }
        if not covered:
            out["covered_positions"] = 0
            return out
        gains = [r["gain"] for r in covered]
        clusters: dict[str, list[float]] = {}
        for row in covered:
            clusters.setdefault(row["opp_team_id"], []).append(min(max(row["gain"], 0.0), CAP))
        cm = clustered_mean(list(clusters.items()))
        half = 1.96 * cm.clustered_se
        out.update(
            {
                "covered_positions": len(covered),
                f"conditional_share_among_covered_gt_{margin:g}": _share(
                    covered, lambda r: r["gain"] > margin
                ),
                "conditional_among_covered_gt_0": _share(covered, lambda r: r["gain"] > 0),
                "conditional_among_covered_gt_25": _share(covered, lambda r: r["gain"] > 25),
                "median_gain_among_covered": statistics.median(gains),
                "mean_positive_gain_capped_300_among_covered": cm.mean,
                "mean_positive_gain_cluster_ci95_among_covered": [
                    max(0.0, cm.mean - half),
                    cm.mean + half,
                ],
            }
        )
        return out

    seconds = [r["total_seconds"] for r in graded]
    return {
        "positions_graded": len(graded),
        "positions_errored": len(errors),
        "error_reasons": sorted({r["error"][:100] for r in errors}),
        "all": block(graded),
        "contested_only": block([r for r in graded if not r["decided"]]),
        "decided_positions": sum(1 for r in graded if r["decided"]),
        "by_split": {
            split: block([r for r in graded if r["split"] == split])
            for split in sorted({r["split"] for r in graded})
        },
        "seconds_per_position": {
            "mean": statistics.fmean(seconds) if seconds else None,
            "median": statistics.median(seconds) if seconds else None,
            "max": max(seconds) if seconds else None,
            "mean_search": statistics.fmean(r["search_seconds"] for r in graded) if graded else None,
            "mean_rebuild": statistics.fmean(r["rebuild_seconds"] for r in graded) if graded else None,
        },
        "illegal_proposals": sum(len(r["illegal_proposals"]) for r in graded),
        "caveat": (
            "One-turn exact-Showdown value with hand-weighted position terms, hidden opponent "
            "sets/spreads as beliefs, a few random-future samples: a filter before full games, "
            "not a win-rate measurement. Gains of a few points are noise; read shares at a margin "
            "and the cluster CIs. Clustering is by opponent team, so power is set by the number "
            "of distinct opponent teams, not positions."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("positions_dir", type=Path, help="runs/positions/<run>")
    ap.add_argument("--proposals", default="engine-rank-2..4", help="engine-rank-A..B or a JSONL file")
    ap.add_argument("--split", choices=("all", "tune", "test"), default="all")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--metric", choices=METRICS, default="exchange_value")
    ap.add_argument("--margin", type=float, default=10.0, help="gain needed to count as 'beats'")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--production-width", action="store_true")
    ap.add_argument("--future-samples", type=int, default=2)
    ap.add_argument("--opp-candidates", type=int, default=4)
    ap.add_argument("--state-hypotheses", type=int, default=1)
    ap.add_argument("--spread-hypotheses", type=int, default=1)
    ap.add_argument("--set-hypotheses", type=int, default=1)
    ap.add_argument("--bring-hypotheses", type=int, default=1)
    ap.add_argument("--total-hypotheses", type=int, default=1)
    args = ap.parse_args()

    run_dir = args.positions_dir
    positions = [
        json.loads(line)
        for line in (run_dir / "positions.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if args.split != "all":
        positions = [p for p in positions if p["split"] == args.split]
    if args.limit:
        positions = positions[: args.limit]
    rank_spec = parse_rank_spec(args.proposals)
    file_proposals = None if rank_spec else load_proposals(Path(args.proposals))
    config = grade_config(args)

    jobs = []
    for position in positions:
        if rank_spec:
            low, high = rank_spec
            proposals = [row["order"] for row in position["engine_top_k"][low - 1 : high]]
        else:
            proposals = file_proposals.get(position["position_id"], [])
        jobs.append((position, str(run_dir), proposals, config, args.metric, args.margin))

    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            iterator = pool.map(_job, jobs)
            for result in iterator:
                results.append(result)
                _print_row(result, len(results), len(jobs))
    else:
        for job in jobs:
            result = _job(job)
            results.append(result)
            _print_row(result, len(results), len(jobs))
    elapsed = time.perf_counter() - started

    summary = summarize(results, args.margin)
    summary["wall_seconds"] = round(elapsed, 1)
    output = {
        "summary": summary,
        "metric": args.metric,
        "margin": args.margin,
        "proposals": args.proposals,
        "width": "production" if args.production_width else {
            "future_samples": config.exact_search_future_samples,
            "opp_candidates": config.search_opp_candidates,
            "total_hypotheses": config.exact_search_total_hypotheses,
        },
        "positions": results,
    }
    label = args.proposals if rank_spec else Path(args.proposals).stem
    out_path = args.output or run_dir / f"grades_{re.sub(r'[^A-Za-z0-9_-]+', '-', label)}.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_path}")
    return 0


def _print_row(result: dict[str, Any], index: int, total: int) -> None:
    if "error" in result:
        print(f"[{index}/{total}] {result['position_id']}: ERROR {result['error']}", flush=True)
        return
    gain = result["gain"]
    print(
        f"[{index}/{total}] {result['position_id']}: engine {result['engine_value']:.1f}, "
        f"best proposal {'-' if gain is None else f'{result['best_proposal_value']:.1f} (gain {gain:+.1f})'}, "
        f"{result['total_seconds']:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
