"""Derived gate: does the belief-mixture shortlist keep the wide-search winner?

Rung 2a/2b made the exact search's K=10 shortlist selectable from a belief-mixture
ranking over hidden opponent Stat Point spreads (`PolicyConfig.shortlist_belief_hypotheses`,
default 1 = point estimate, 3 = the candidate). The pre-registered claim is that the
winner of a WIDE exact search survives into that K=10 shortlist at least as often under
the mixture as under the point estimate.

Games are driven like `offline/measure_opponent_information_leak.py` (myopic picks, cheap
config). At each checkpoint the public `LiveExactMirror` root is the information boundary
of real play. `BattleMemory` is fed from each step's protocol lines -- including the
opening burst -- and attached to the decision battle so the posterior Rung 1 built is
what the mixture sees, not the corpus prior.

Do not retune after seeing results. Verdict is non-inferiority of the paired hit
difference at margin 0.02, team-clustered by own team file.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

from vgc.actions import describe_order, enumerate_joint_orders
from vgc.battle_memory import BattleMemory
from vgc.belief_scoring import belief_ordered_candidates, joint_spread_hypotheses
from vgc.config import REPO_ROOT, RUNS_DIR
from vgc.evaluation import clustered_interval, clustered_mean, variance_components
from vgc.evaluator import score_joint_orders
from vgc.gate_evidence import git_head
from vgc.models import PolicyConfig
from vgc.rl.env import DEFAULT_SHOWDOWN_REPO, DirectBattle, SimWorker, choice_string
from vgc.rl.exact_search import search_joint_orders_exact
from vgc.rl.live_mirror import LiveExactMirror
from vgc.search import _select_search_candidates

DEFAULT_MANIFEST = REPO_ROOT / "data" / "selfplay" / "archetype_pool_150" / "manifest.json"
DEFAULT_OUTPUT = RUNS_DIR / "eval" / "belief_shortlist_recall.json"

SHORTLIST_K = 10
BELIEF_HYPOTHESES = 3
NONINFERIORITY_MARGIN = 0.02
MIN_CLUSTERS = 40
MIN_DECISIONS = 100

# Deliberately cheap: this drives the game between checkpoints and is the budget for
# every wide-search / shortlist comparison. Identical for every decision.
DIAGNOSTIC_CONFIG = PolicyConfig(
    search_our_candidates=4,
    search_opp_candidates=4,
    exact_search_future_samples=1,
    exact_search_state_hypotheses=1,
    exact_search_spread_hypotheses=1,
    use_rolling_horizon=False,
    use_value_head=False,
)


def _load_team_files(manifest_path: Path, count: int, seed: int) -> list[Path]:
    """One team per distinct archetype where possible -- cluster-by-team, not by game."""

    manifest = json.loads(manifest_path.read_text())
    by_archetype: dict[str, list[str]] = {}
    for entry in manifest:
        by_archetype.setdefault(entry["archetype"], []).append(entry["file"])
    rng = random.Random(seed)
    archetypes = sorted(by_archetype)
    rng.shuffle(archetypes)
    chosen: list[Path] = []
    pool_dir = manifest_path.parent
    index = 0
    while len(chosen) < count:
        archetype = archetypes[index % len(archetypes)]
        files = by_archetype[archetype]
        chosen.append(pool_dir / rng.choice(files))
        index += 1
    return chosen


def _pick_myopic(battle, config: PolicyConfig, rng: random.Random):
    """Cheap move selection to drive the game between checkpoints -- no exact search."""

    scored = score_joint_orders(battle, config)
    if scored:
        return scored[0].order
    orders = enumerate_joint_orders(battle)
    return rng.choice(orders) if orders else None


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _feed_memory(memory: BattleMemory, battle, lines: list[str]) -> None:
    """Protocol lines first (DirectAgent.observe), then a board snapshot for evidence.

    Speed and damage observations need a `TurnMemory` snapshot of our effective Speed /
    HP taken at the request, before the turn's moves arrive. `observe_protocol` alone
    cannot populate that; `observe_battle` is what `VgcPlayer.decide` does at the same
    moment. Without it the posterior never leaves the prior.
    """

    if lines:
        memory.observe_protocol([line.split("|") for line in lines])
    memory.observe_battle(battle)
    setattr(battle, "_vgc_battle_memory", memory)


def _hypothesis_weights(hypotheses: list) -> tuple[float, ...]:
    return tuple(round(float(weight), 10) for weight, _assignment in hypotheses)


def _rank_of(winner: str, scored: list) -> int | None:
    for index, entry in enumerate(scored, start=1):
        if describe_order(entry.order) == winner:
            return index
    return None


def _shortlist_set(scored: list, config: PolicyConfig) -> set[str]:
    selected, _tail = _select_search_candidates(scored, config)
    return {describe_order(entry.order) for entry in selected}


def _evaluate_checkpoint(
    root: DirectBattle,
    memory: BattleMemory,
    own_team: str,
    own_team_file: str,
    opp_team_file: str,
    turn: int,
    config: PolicyConfig,
) -> tuple[dict[str, Any] | None, str | None]:
    """Return `(record, skip_reason)`. Exactly one is set."""

    if "p1" not in root.sides_to_move():
        return None, "p1_not_to_move"

    memory.observe_battle(root.battles["p1"])
    setattr(root.battles["p1"], "_vgc_battle_memory", memory)

    mirror = LiveExactMirror(own_team, config)
    try:
        mirror_root = mirror.build(root.battles["p1"])
        try:
            decision_battle = getattr(mirror_root, "_decision_battles", {}).get(
                "p1", mirror_root.battles["p1"]
            )
            decision_battle._vgc_battle_memory = memory

            myopic = score_joint_orders(decision_battle, config)
            n = len(myopic)
            if n <= SHORTLIST_K:
                return None, "n_le_10"

            wide_cfg = replace(
                config,
                search_our_candidates=n,
                shortlist_belief_hypotheses=1,
            )
            t0 = time.perf_counter()
            wide = search_joint_orders_exact(mirror_root, "p1", wide_cfg)
            wide_seconds = time.perf_counter() - t0
            if not wide:
                return None, "empty_wide_search"
            winner = describe_order(wide[0].order)

            shortlist_cfg = replace(config, search_our_candidates=SHORTLIST_K)
            incumbent_shortlist = _shortlist_set(myopic, shortlist_cfg)
            myopic_order_before = [describe_order(entry.order) for entry in myopic]

            belief_cfg = replace(config, shortlist_belief_hypotheses=BELIEF_HYPOTHESES)
            belief_ranked = belief_ordered_candidates(
                decision_battle, list(myopic), belief_cfg, memory=memory
            )
            myopic_order_after = [describe_order(entry.order) for entry in myopic]
            if myopic_order_before != myopic_order_after:
                raise AssertionError(
                    "belief_ordered_candidates mutated the myopic list order"
                )

            candidate_shortlist = _shortlist_set(belief_ranked, shortlist_cfg)
            mixture = myopic[0].breakdown.get("belief_mixture") or {}
            posterior = joint_spread_hypotheses(
                decision_battle, memory, config, limit=BELIEF_HYPOTHESES
            )
            prior = joint_spread_hypotheses(
                decision_battle, None, config, limit=BELIEF_HYPOTHESES
            )
            incumbent_hit = winner in incumbent_shortlist
            candidate_hit = winner in candidate_shortlist
            return {
                "own_team": own_team_file,
                "opp_team": opp_team_file,
                "turn": turn,
                "n": n,
                "winner": winner,
                "incumbent_hit": incumbent_hit,
                "candidate_hit": candidate_hit,
                "incumbent_rank": _rank_of(winner, myopic),
                "candidate_rank": _rank_of(winner, belief_ranked),
                "hypotheses_used": mixture.get("hypotheses"),
                "posterior_differs_from_prior": _hypothesis_weights(posterior)
                != _hypothesis_weights(prior),
                "speed_observations": len(memory.speed_observations),
                "damage_observations": len(memory.damage_observations),
                "sets_differ": incumbent_shortlist != candidate_shortlist,
                "wide_search_seconds": wide_seconds,
            }, None
        finally:
            mirror_root.close()
    finally:
        mirror.close()


def run(
    *,
    manifest_path: Path,
    pairs: int,
    checkpoints: tuple[int, ...],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    team_files = _load_team_files(manifest_path, pairs * 2, seed)
    records: list[dict[str, Any]] = []
    skips: dict[str, int] = defaultdict(int)
    config = DIAGNOSTIC_CONFIG
    with SimWorker(DEFAULT_SHOWDOWN_REPO) as worker:
        for pair_index in range(pairs):
            own_path = team_files[2 * pair_index]
            opp_path = team_files[2 * pair_index + 1]
            own_team = own_path.read_text().strip()
            opp_team = opp_path.read_text().strip()
            own_team_file = str(own_path)
            opp_team_file = str(opp_path)
            label = f"{own_path.parent.name}_vs_{opp_path.parent.name}"
            rng = random.Random(seed + pair_index)

            root = DirectBattle.start(
                worker,
                f"belief-shortlist-{pair_index}",
                own_team,
                opp_team,
                seed=[rng.randint(1, 2**31 - 1) for _ in range(4)],
            )
            memory = BattleMemory(root.battle_id, our_role="p1")
            try:
                _feed_memory(memory, root.battles["p1"], root.last_lines.get("p1") or [])
                preview = root.step({"p1": "team 1234", "p2": "team 1234"})
                _feed_memory(memory, root.battles["p1"], preview.lines.get("p1") or [])
                next_checkpoint = 0
                for turn in range(1, max(checkpoints) + 1):
                    if root.ended:
                        break
                    if (
                        next_checkpoint < len(checkpoints)
                        and turn == checkpoints[next_checkpoint]
                    ):
                        try:
                            record, skip_reason = _evaluate_checkpoint(
                                root,
                                memory,
                                own_team,
                                own_team_file,
                                opp_team_file,
                                turn,
                                config,
                            )
                        except AssertionError:
                            raise
                        except Exception as exc:  # noqa: BLE001 -- a rare worker-side
                            # race (a request answered between snapshot and patch) should
                            # skip this one checkpoint, not abort the run over it.
                            print(f"[{label} turn {turn}] SKIPPED ({exc})", flush=True)
                            skips[f"exception:{type(exc).__name__}"] += 1
                            record, skip_reason = None, None
                        if skip_reason is not None:
                            skips[skip_reason] += 1
                            print(f"[{label} turn {turn}] SKIPPED ({skip_reason})", flush=True)
                        elif record is not None:
                            records.append(record)
                            print(
                                f"[{label} turn {turn}] "
                                f"inc={'HIT' if record['incumbent_hit'] else 'MISS'} "
                                f"cand={'HIT' if record['candidate_hit'] else 'MISS'} "
                                f"n={record['n']} "
                                f"wide={record['wide_search_seconds']:.2f}s "
                                f"post={'diff' if record['posterior_differs_from_prior'] else 'same'}",
                                flush=True,
                            )
                        next_checkpoint += 1
                    choices = {}
                    for side in root.sides_to_move():
                        order = _pick_myopic(root.battles[side], config, rng)
                        if order is None:
                            continue
                        choices[side] = choice_string(order)
                    if not choices:
                        break
                    stepped = root.step(choices)
                    _feed_memory(
                        memory, root.battles["p1"], stepped.lines.get("p1") or []
                    )
            finally:
                root.close()
    return records, dict(skips)


def summarize(
    records: list[dict[str, Any]],
    *,
    skips: dict[str, int] | None = None,
    config: dict[str, Any] | None = None,
    git_commit: str | None = None,
    margin: float = NONINFERIORITY_MARGIN,
    min_clusters: int = MIN_CLUSTERS,
    min_decisions: int = MIN_DECISIONS,
) -> dict[str, Any]:
    """Clustered recall, paired difference, and pre-registered non-inferiority verdict."""

    skips = dict(skips or {})
    grouped_inc: dict[str, list[int]] = defaultdict(list)
    grouped_cand: dict[str, list[int]] = defaultdict(list)
    grouped_diff: dict[str, list[float]] = defaultdict(list)
    for record in records:
        cluster = str(record["own_team"])
        inc = int(bool(record["incumbent_hit"]))
        cand = int(bool(record["candidate_hit"]))
        grouped_inc[cluster].append(inc)
        grouped_cand[cluster].append(cand)
        grouped_diff[cluster].append(float(cand - inc))

    clusters = sorted(grouped_diff)
    n_clusters = len(clusters)
    n_decisions = len(records)
    inc_counts = [(sum(grouped_inc[key]), len(grouped_inc[key])) for key in clusters]
    cand_counts = [(sum(grouped_cand[key]), len(grouped_cand[key])) for key in clusters]
    inc_low, inc_high = clustered_interval(inc_counts)
    cand_low, cand_high = clustered_interval(cand_counts)
    recall_inc = (
        sum(hit for hit, _n in inc_counts) / n_decisions if n_decisions else 0.0
    )
    recall_cand = (
        sum(hit for hit, _n in cand_counts) / n_decisions if n_decisions else 0.0
    )
    paired = clustered_mean([(key, grouped_diff[key]) for key in clusters])
    diff_low, diff_high = paired.interval()
    inc_components = variance_components(inc_counts)
    cand_components = variance_components(cand_counts)
    differ = [record for record in records if record.get("sets_differ")]
    differ_inc = sum(1 for record in differ if record["incumbent_hit"])
    differ_cand = sum(1 for record in differ if record["candidate_hit"])
    posterior_differs = sum(
        1 for record in records if record.get("posterior_differs_from_prior")
    )
    with_evidence = [
        record
        for record in records
        if int(record.get("speed_observations") or 0)
        + int(record.get("damage_observations") or 0)
        > 0
    ]
    posterior_differs_with_evidence = sum(
        1 for record in with_evidence if record.get("posterior_differs_from_prior")
    )
    wide_times = [float(record["wide_search_seconds"]) for record in records]
    # Binary variance_components does not apply to d in {-1, 0, +1}; clustered_mean's
    # tau / se_floor use the same ANOVA estimator. Report both names so the gate log
    # matches the pre-registered wording.
    if n_clusters < min_clusters or n_decisions < min_decisions:
        verdict = "INDETERMINATE"
    elif diff_low >= -margin:
        verdict = "PASS"
    else:
        verdict = "FAIL"

    return {
        "git_commit": git_commit,
        "config": config or {},
        "skips": skips,
        "decisions": n_decisions,
        "clusters": n_clusters,
        "recall_incumbent": recall_inc,
        "recall_incumbent_interval": [inc_low, inc_high],
        "recall_candidate": recall_cand,
        "recall_candidate_interval": [cand_low, cand_high],
        "difference_mean": paired.mean,
        "difference_se": paired.clustered_se,
        "difference_interval": [diff_low, diff_high],
        "difference_tau": paired.tau,
        "difference_power_floor": paired.se_floor,
        "variance_components_tau": paired.tau,
        "variance_components_se_floor": paired.se_floor,
        "recall_incumbent_tau": inc_components.tau,
        "recall_candidate_tau": cand_components.tau,
        "sets_differ": len(differ),
        "sets_differ_incumbent_hits": differ_inc,
        "sets_differ_candidate_hits": differ_cand,
        "posterior_differs_from_prior": posterior_differs,
        "decisions_with_speed_or_damage_evidence": len(with_evidence),
        "posterior_differs_with_evidence": posterior_differs_with_evidence,
        "wide_search_seconds_median": _median(wide_times),
        "margin": margin,
        "min_clusters": min_clusters,
        "min_decisions": min_decisions,
        "verdict": verdict,
        "records": records,
    }


def _safe_git_head() -> str | None:
    try:
        return git_head()
    except (OSError, subprocess.CalledProcessError):
        return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--pairs", type=int, default=60, help="number of team-pair games")
    parser.add_argument(
        "--checkpoints",
        type=int,
        nargs="+",
        default=[2, 4, 6],
        help="turn numbers to score shortlist recall at, per game",
    )
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    records, skips = run(
        manifest_path=args.manifest,
        pairs=args.pairs,
        checkpoints=tuple(sorted(args.checkpoints)),
        seed=args.seed,
    )
    summary = summarize(
        records,
        skips=skips,
        git_commit=_safe_git_head(),
        config={
            "manifest": str(args.manifest),
            "pairs": args.pairs,
            "checkpoints": list(sorted(args.checkpoints)),
            "seed": args.seed,
            "shortlist_k": SHORTLIST_K,
            "belief_hypotheses": BELIEF_HYPOTHESES,
            "search_opp_candidates": DIAGNOSTIC_CONFIG.search_opp_candidates,
            "exact_search_future_samples": DIAGNOSTIC_CONFIG.exact_search_future_samples,
            "exact_search_state_hypotheses": DIAGNOSTIC_CONFIG.exact_search_state_hypotheses,
            "exact_search_spread_hypotheses": DIAGNOSTIC_CONFIG.exact_search_spread_hypotheses,
            "use_rolling_horizon": DIAGNOSTIC_CONFIG.use_rolling_horizon,
            "use_value_head": DIAGNOSTIC_CONFIG.use_value_head,
            "margin": NONINFERIORITY_MARGIN,
            "min_clusters": MIN_CLUSTERS,
            "min_decisions": MIN_DECISIONS,
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")

    print("\n=== belief shortlist recall: point-estimate vs mixture K=10 ===")
    print(f"decisions: {summary['decisions']}")
    print(f"clusters: {summary['clusters']}")
    print(f"skips: {summary['skips']}")
    print(
        f"recall incumbent: {summary['recall_incumbent']:.3f} "
        f"CI {summary['recall_incumbent_interval']}"
    )
    print(
        f"recall candidate: {summary['recall_candidate']:.3f} "
        f"CI {summary['recall_candidate_interval']}"
    )
    print(
        f"difference mean: {summary['difference_mean']:.4f} "
        f"SE {summary['difference_se']:.4f} "
        f"CI {summary['difference_interval']}"
    )
    print(
        f"difference tau: {summary['difference_tau']:.4f} "
        f"power floor sqrt(tau^2/K): {summary['difference_power_floor']:.4f}"
    )
    print(
        f"sets differ: {summary['sets_differ']} "
        f"(inc hits {summary['sets_differ_incumbent_hits']}, "
        f"cand hits {summary['sets_differ_candidate_hits']})"
    )
    print(
        f"posterior differs from prior: {summary['posterior_differs_from_prior']}/"
        f"{summary['decisions']} "
        f"(with speed/damage evidence: {summary['posterior_differs_with_evidence']}/"
        f"{summary['decisions_with_speed_or_damage_evidence']})"
    )
    median = summary["wide_search_seconds_median"]
    if median is not None:
        print(f"wide-search median: {median:.2f}s")
    print(f"written to {args.output}")
    print(f"verdict: {summary['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
