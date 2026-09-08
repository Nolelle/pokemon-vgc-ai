#!/usr/bin/env python
"""Shadow-test neural search ranking, then run paired default-off hybrid gates."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from selfplay.train_full_pipeline import (  # noqa: E402
    DEFAULT_MANIFEST,
    load_team_pool,
    split_teams_by_archetype,
)
from vgc.baselines import BASELINES  # noqa: E402
from vgc.config import FORMAT_ID, RUNS_DIR  # noqa: E402
from vgc.models import PolicyConfig  # noqa: E402
from vgc.rl.agents import DirectAgent, make_direct_agent  # noqa: E402
from vgc.rl.env import DEFAULT_SHOWDOWN_REPO, SimWorker  # noqa: E402
from vgc.rl.match import play_battle  # noqa: E402
from vgc.rl.opponents import load_snapshot  # noqa: E402
from vgc.rl.search_guidance import (  # noqa: E402
    NeuralSearchPlayer,
    checkpoint_sha256,
    summarize_decision_records,
    write_decision_records,
    write_guidance_samples,
)

DEFAULT_CHECKPOINT = RUNS_DIR / "full_pipeline" / "rl_promotion_10k" / "best.pt"
DEFAULT_OUTPUT = RUNS_DIR / "eval" / "neural_search_shadow.json"
DEFAULT_RECORDS = RUNS_DIR / "eval" / "neural_search_shadow.jsonl"
DEFAULT_DATASET = RUNS_DIR / "eval" / "neural_search_shadow.pt"
DEFAULT_OPPONENTS = "heuristic,vgc_myopic,vgc_shallow,vgc"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--mode", choices=("shadow", "hybrid"), default="shadow")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--team-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument(
        "--team-part",
        choices=("train", "holdout"),
        default="holdout",
        help="which deterministic side of the team split supplies learner teams",
    )
    parser.add_argument("--opponents", default=DEFAULT_OPPONENTS)
    parser.add_argument("--search-k", type=int, choices=(5, 8, 10), default=10)
    parser.add_argument(
        "--prior-gate",
        type=Path,
        default=None,
        help="required passed Hybrid-10 gate for K=8, or Hybrid-8 gate for K=5",
    )
    parser.add_argument("--safety-slots", type=int, default=4)
    parser.add_argument("--noninferiority-margin", type=float, default=0.02)
    parser.add_argument("--recall-threshold", type=float, default=0.98)
    parser.add_argument("--minimum-games", type=int, default=100)
    parser.add_argument("--minimum-teams", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def paired_clustered_interval(
    deltas_by_team: dict[str, list[float]], z: float = 1.96
) -> tuple[float, float]:
    """Confidence range for a paired hybrid-minus-shadow win-rate difference."""

    usable = [values for values in deltas_by_team.values() if values]
    all_values = [value for values in usable for value in values]
    if not all_values:
        return 0.0, 0.0
    mean = sum(all_values) / len(all_values)
    if len(usable) < 2:
        variance = sum((value - mean) ** 2 for value in all_values)
        variance /= max(1, len(all_values) - 1)
        standard_error = math.sqrt(variance / len(all_values))
    else:
        cluster_residuals = [sum(value - mean for value in values) for values in usable]
        k = len(usable)
        standard_error = math.sqrt(
            k
            / ((k - 1) * len(all_values) ** 2)
            * sum(residual**2 for residual in cluster_residuals)
        )
    return max(-1.0, mean - z * standard_error), min(1.0, mean + z * standard_error)


def _win_value(result: float) -> float:
    if result > 0:
        return 1.0
    if result < 0:
        return 0.0
    return 0.5


def _make_guided_agent(
    *,
    model,
    checkpoint: Path,
    checkpoint_hash: str,
    mode: str,
    team,
    opponent_team,
    config: PolicyConfig,
    device: str,
    safety_slots: int,
) -> DirectAgent:
    return DirectAgent(
        NeuralSearchPlayer(
            model=model,
            checkpoint_path=checkpoint,
            checkpoint_hash=checkpoint_hash,
            mode=mode,
            device=device,
            team_id=team.label,
            opponent_team_id=opponent_team.label,
            safety_slots=safety_slots,
            config=config,
            team=team.packed,
            battle_format=FORMAT_ID,
            start_listening=False,
        ),
        name=mode,
    )


def _play_one(
    worker: SimWorker,
    *,
    agent: DirectAgent,
    opponent_name: str,
    team,
    opponent_team,
    learner_side: str,
    battle_id: str,
    battle_seed: list[int],
    policy_seed: int,
):
    random.seed(policy_seed)
    opponent = make_direct_agent(opponent_name, opponent_team.packed)
    other_side = "p2" if learner_side == "p1" else "p1"
    outcome = play_battle(
        worker,
        battle_id,
        {learner_side: agent, other_side: opponent},
        {learner_side: team.packed, other_side: opponent_team.packed},
        seed=battle_seed,
    )
    return (
        outcome,
        list(agent.player.decision_records),
        list(agent.player.training_samples),
        agent.player.fallback_count,
    )


def _strength_summary(pairs: list[dict[str, object]]) -> dict[str, object]:
    by_team: dict[str, list[float]] = defaultdict(list)
    by_opponent: dict[str, list[float]] = defaultdict(list)
    for pair in pairs:
        delta = float(pair["hybrid_win"]) - float(pair["shadow_win"])
        by_team[str(pair["team_id"])].append(delta)
        by_opponent[str(pair["opponent"])].append(delta)
    games = len(pairs)
    shadow_rate = sum(float(pair["shadow_win"]) for pair in pairs) / games if games else 0.0
    hybrid_rate = sum(float(pair["hybrid_win"]) for pair in pairs) / games if games else 0.0
    return {
        "paired_games": games,
        "teams": len(by_team),
        "shadow_win_rate": shadow_rate,
        "hybrid_win_rate": hybrid_rate,
        "difference": hybrid_rate - shadow_rate,
        "team_clustered_difference_interval": paired_clustered_interval(by_team),
        "difference_by_opponent": {
            name: sum(values) / len(values) for name, values in sorted(by_opponent.items())
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if min(args.games, args.search_k, args.minimum_games, args.minimum_teams) <= 0:
        raise SystemExit(
            "--games, --search-k, --minimum-games, and --minimum-teams must be positive"
        )
    if args.safety_slots < 0:
        raise SystemExit("--safety-slots must be nonnegative")
    if not 0.0 <= args.noninferiority_margin <= 1.0 or not 0.0 <= args.recall_threshold <= 1.0:
        raise SystemExit("--noninferiority-margin and --recall-threshold must be between 0 and 1")
    opponents = [name.strip() for name in args.opponents.split(",") if name.strip()]
    unknown = sorted(set(opponents) - set(BASELINES))
    if unknown:
        raise SystemExit(f"unknown opponents: {unknown}; available: {sorted(BASELINES)}")
    if not opponents:
        raise SystemExit("--opponents must not be empty")

    model = load_snapshot(args.checkpoint, device=args.device)
    model_hash = checkpoint_sha256(args.checkpoint)
    prior_stage: dict[str, object] | None = None
    if args.mode == "hybrid" and args.search_k < 10:
        if args.prior_gate is None:
            raise SystemExit("Hybrid-8/5 requires --prior-gate; do not skip the fixed sequence")
        prior_report = json.loads(args.prior_gate.read_text())
        expected_k = 10 if args.search_k == 8 else 8
        prior_stage = {
            "path": str(args.prior_gate.resolve()),
            "expected_search_k": expected_k,
            "actual_search_k": int(prior_report.get("search_k", 0)),
            "passed": bool(prior_report.get("passed")),
            "checkpoint_matches": prior_report.get("checkpoint_sha256") == model_hash,
        }
        if not (
            prior_stage["actual_search_k"] == expected_k
            and prior_stage["passed"]
            and prior_stage["checkpoint_matches"]
        ):
            raise SystemExit(
                "--prior-gate must be a passed result for the immediately preceding K "
                "using the exact same checkpoint"
            )
    train_teams, holdout_teams = split_teams_by_archetype(
        load_team_pool(args.team_manifest),
        holdout_fraction=args.holdout_fraction,
        seed=args.seed,
    )
    selected_teams = train_teams if args.team_part == "train" else holdout_teams
    schedule = [
        (team, opponent_team, opponent, learner_side)
        for team_index, team in enumerate(selected_teams)
        for opponent_team in (
            team,
            selected_teams[(team_index + 1) % len(selected_teams)],
        )
        for opponent in opponents
        for learner_side in ("p1", "p2")
    ]
    random.Random(args.seed + 17).shuffle(schedule)
    simulator_rng = random.Random(args.seed + 29)
    default_config = PolicyConfig(log_decisions=False)
    hybrid_config = replace(default_config, search_our_candidates=args.search_k)

    shadow_records: list[dict[str, object]] = []
    hybrid_records: list[dict[str, object]] = []
    shadow_samples: list = []
    hybrid_samples: list = []
    pairs: list[dict[str, object]] = []
    shadow_fallbacks = 0
    hybrid_fallbacks = 0
    with SimWorker(DEFAULT_SHOWDOWN_REPO) as worker:
        for game_index in range(args.games):
            team, opponent_team, opponent_name, learner_side = schedule[
                game_index % len(schedule)
            ]
            battle_seed = [simulator_rng.randrange(1, 2**31) for _ in range(4)]
            policy_seed = args.seed + game_index * 1_000_003
            shadow_agent = _make_guided_agent(
                model=model,
                checkpoint=args.checkpoint,
                checkpoint_hash=model_hash,
                mode="shadow",
                team=team,
                opponent_team=opponent_team,
                config=default_config,
                device=args.device,
                safety_slots=args.safety_slots,
            )
            shadow_outcome, records, samples, fallbacks = _play_one(
                worker,
                agent=shadow_agent,
                opponent_name=opponent_name,
                team=team,
                opponent_team=opponent_team,
                learner_side=learner_side,
                battle_id=f"neural-shadow-{game_index:06d}",
                battle_seed=list(battle_seed),
                policy_seed=policy_seed,
            )
            shadow_records.extend(records)
            shadow_samples.extend(samples)
            shadow_fallbacks += fallbacks

            if args.mode == "hybrid":
                hybrid_agent = _make_guided_agent(
                    model=model,
                    checkpoint=args.checkpoint,
                    checkpoint_hash=model_hash,
                    mode="hybrid",
                    team=team,
                    opponent_team=opponent_team,
                    config=hybrid_config,
                    device=args.device,
                    safety_slots=args.safety_slots,
                )
                hybrid_outcome, records, samples, fallbacks = _play_one(
                    worker,
                    agent=hybrid_agent,
                    opponent_name=opponent_name,
                    team=team,
                    opponent_team=opponent_team,
                    learner_side=learner_side,
                    battle_id=f"neural-hybrid-{game_index:06d}",
                    battle_seed=list(battle_seed),
                    policy_seed=policy_seed,
                )
                hybrid_records.extend(records)
                hybrid_samples.extend(samples)
                hybrid_fallbacks += fallbacks
                pairs.append(
                    {
                        "team_id": team.label,
                        "opponent_team_id": opponent_team.label,
                        "opponent": opponent_name,
                        "learner_side": learner_side,
                        "shadow_win": _win_value(shadow_outcome.result_for("shadow")),
                        "hybrid_win": _win_value(hybrid_outcome.result_for("hybrid")),
                    }
                )

    write_decision_records(args.records, shadow_records)
    write_guidance_samples(args.dataset, shadow_samples)
    shadow_summary = summarize_decision_records(shadow_records)
    guided_metric = dict(shadow_summary["retention"]).get(
        f"guided_at_{args.search_k}", {}
    )
    guided_interval = list(dict(guided_metric).get("team_clustered_interval", [0.0, 0.0]))
    shadow_checks = {
        "minimum_teams": int(shadow_summary["teams"]) >= args.minimum_teams,
        "guided_recall_lower_bound": float(guided_interval[0]) >= args.recall_threshold,
        "zero_fallbacks": shadow_fallbacks == 0,
    }
    report: dict[str, object] = {
        "schema": "vgc-neural-search-evaluation-v1",
        "mode": args.mode,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": model_hash,
        "games": args.games,
        "holdout_teams": len(holdout_teams),
        "team_part": args.team_part,
        "selected_teams": len(selected_teams),
        "opponents": opponents,
        "search_k": args.search_k,
        "safety_slots": args.safety_slots,
        "prior_stage": prior_stage,
        "shadow": shadow_summary,
        "shadow_checks": shadow_checks,
        "shadow_ready_for_hybrid": all(shadow_checks.values()),
        "fallbacks": {"shadow": shadow_fallbacks, "hybrid": hybrid_fallbacks},
        "records": {"shadow": str(args.records.resolve())},
        "datasets": {"shadow": str(args.dataset.resolve())},
    }
    if args.mode == "hybrid":
        hybrid_records_path = args.records.with_name(
            f"{args.records.stem}_hybrid{args.records.suffix}"
        )
        write_decision_records(hybrid_records_path, hybrid_records)
        hybrid_dataset_path = args.dataset.with_name(
            f"{args.dataset.stem}_hybrid{args.dataset.suffix}"
        )
        write_guidance_samples(hybrid_dataset_path, hybrid_samples)
        strength = _strength_summary(pairs)
        shadow_work = dict(report["shadow"]["search_work"])
        hybrid_summary = summarize_decision_records(hybrid_records)
        hybrid_work = dict(hybrid_summary["search_work"])
        shadow_exchanges = float(shadow_work["exchange_count_mean"])
        hybrid_exchanges = float(hybrid_work["exchange_count_mean"])
        compute_reduction = (
            1.0 - hybrid_exchanges / shadow_exchanges if shadow_exchanges > 0.0 else 0.0
        )
        noninferior = (
            float(strength["team_clustered_difference_interval"][0])
            > -args.noninferiority_margin
        )
        compute_gate = args.search_k >= default_config.search_our_candidates or compute_reduction >= 0.15
        report.update(
            {
                "hybrid": hybrid_summary,
                "paired_strength": strength,
                "compute_reduction": compute_reduction,
                "records": {
                    "shadow": str(args.records.resolve()),
                    "hybrid": str(hybrid_records_path.resolve()),
                },
                "datasets": {
                    "shadow": str(args.dataset.resolve()),
                    "hybrid": str(hybrid_dataset_path.resolve()),
                },
                "checks": {
                    "prior_stage_passed": (
                        all(shadow_checks.values()) if args.search_k == 10 else True
                    ),
                    "minimum_paired_games": int(strength["paired_games"])
                    >= args.minimum_games,
                    "minimum_teams": int(strength["teams"]) >= args.minimum_teams,
                    "paired_noninferiority": noninferior,
                    "meaningful_compute_reduction_when_pruned": compute_gate,
                    "zero_fallbacks": shadow_fallbacks + hybrid_fallbacks == 0,
                },
            }
        )
        report["passed"] = all(report["checks"].values())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if args.mode == "shadow" or bool(report.get("passed")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
