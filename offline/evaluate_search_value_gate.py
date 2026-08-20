#!/usr/bin/env python
"""Paired default-off gate for existing full search plus a small RL value weight."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from offline.evaluate_neural_search import _strength_summary, _win_value  # noqa: E402
from offline.evaluate_search_value import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_MANIFEST,
)
from selfplay.train_full_pipeline import (  # noqa: E402
    load_team_pool,
    split_teams_by_archetype,
)
from vgc.agent import VgcPlayer  # noqa: E402
from vgc.baselines import BASELINES  # noqa: E402
from vgc.config import FORMAT_ID, RUNS_DIR  # noqa: E402
from vgc.models import PolicyConfig  # noqa: E402
from vgc.rl.agents import DirectAgent, make_direct_agent  # noqa: E402
from vgc.rl.env import DEFAULT_SHOWDOWN_REPO, SimWorker  # noqa: E402
from vgc.rl.match import play_battle  # noqa: E402
from vgc.rl.opponents import load_snapshot  # noqa: E402
from vgc.rl.search_guidance import checkpoint_sha256  # noqa: E402
from vgc.rl.search_value import (  # noqa: E402
    NeuralValueSearchPlayer,
    write_value_records,
)

DEFAULT_PRIOR_SHADOW = RUNS_DIR / "eval" / "search_value_shadow.json"
DEFAULT_OUTPUT = RUNS_DIR / "eval" / "search_value_gate_alpha005.json"
DEFAULT_RECORDS = RUNS_DIR / "eval" / "search_value_gate_alpha005.jsonl"
DEFAULT_OPPONENTS = "heuristic,vgc_myopic,vgc_shallow,vgc"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--prior-shadow", type=Path, default=DEFAULT_PRIOR_SHADOW)
    parser.add_argument("--value-weight", type=float, default=0.05)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--team-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--opponents", default=DEFAULT_OPPONENTS)
    parser.add_argument("--noninferiority-margin", type=float, default=0.02)
    parser.add_argument("--minimum-games", type=int, default=200)
    parser.add_argument("--minimum-teams", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--null-test", action="store_true")
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def _baseline_agent(team: str, config: PolicyConfig, *, name: str) -> DirectAgent:
    return DirectAgent(
        VgcPlayer(
            config=config,
            team=team,
            battle_format=FORMAT_ID,
            start_listening=False,
        ),
        name=name,
    )


def _candidate_agent(
    *,
    model,
    checkpoint: Path,
    checkpoint_hash: str,
    value_weight: float,
    team,
    opponent_team,
    config: PolicyConfig,
    device: str,
) -> DirectAgent:
    return DirectAgent(
        NeuralValueSearchPlayer(
            model=model,
            checkpoint_path=checkpoint,
            value_weight=value_weight,
            checkpoint_hash=checkpoint_hash,
            device=device,
            team_id=team.label,
            opponent_team_id=opponent_team.label,
            config=config,
            team=team.packed,
            battle_format=FORMAT_ID,
            start_listening=False,
        ),
        name="value_search",
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if min(args.games, args.minimum_games, args.minimum_teams) <= 0:
        raise SystemExit("--games, --minimum-games, and --minimum-teams must be positive")
    if args.value_weight <= 0.0:
        raise SystemExit("--value-weight must be positive")
    if not 0.0 <= args.noninferiority_margin <= 1.0:
        raise SystemExit("--noninferiority-margin must be between 0 and 1")
    opponents = [name.strip() for name in args.opponents.split(",") if name.strip()]
    unknown = sorted(set(opponents) - set(BASELINES))
    if unknown or not opponents:
        raise SystemExit(f"invalid opponents {unknown}; available: {sorted(BASELINES)}")

    checkpoint_hash = checkpoint_sha256(args.checkpoint)
    prior = json.loads(args.prior_shadow.read_text())
    prior_incremental = dict(prior.get("summary", {}).get("incremental_predictive_value", {}))
    prior_checks = {
        "schema": prior.get("schema") == "vgc-search-value-evaluation-v1",
        "shadow_complete": bool(prior.get("shadow_audit_complete")),
        "checkpoint_matches": prior.get("checkpoint_sha256") == checkpoint_hash,
        "incremental_information": bool(prior_incremental.get("adds_predictive_information")),
        "weight_was_shadowed": args.value_weight in prior.get("value_weights", []),
    }
    if not args.null_test and not all(prior_checks.values()):
        raise SystemExit(f"prior shadow audit does not authorize this paired gate: {prior_checks}")

    model = None if args.null_test else load_snapshot(args.checkpoint, device=args.device)
    _train_teams, holdout_teams = split_teams_by_archetype(
        load_team_pool(args.team_manifest),
        holdout_fraction=args.holdout_fraction,
        seed=args.seed,
    )
    schedule = [
        (team, opponent_team, opponent, learner_side)
        for team_index, team in enumerate(holdout_teams)
        for opponent_team in (
            team,
            holdout_teams[(team_index + 1) % len(holdout_teams)],
        )
        for opponent in opponents
        for learner_side in ("p1", "p2")
    ]
    random.Random(args.seed + 17).shuffle(schedule)
    simulator_rng = random.Random(args.seed + 29)
    config = replace(
        PolicyConfig(log_decisions=False),
        use_value_head=False,
        use_bc_policy=False,
    )
    pairs: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    baseline_fallbacks = 0
    candidate_fallbacks = 0
    with SimWorker(DEFAULT_SHOWDOWN_REPO) as worker:
        for game_index in range(args.games):
            team, opponent_team, opponent_name, learner_side = schedule[
                game_index % len(schedule)
            ]
            battle_seed = [simulator_rng.randrange(1, 2**31) for _ in range(4)]
            policy_seed = args.seed + game_index * 1_000_003
            other_side = "p2" if learner_side == "p1" else "p1"

            random.seed(policy_seed)
            baseline = _baseline_agent(team.packed, config, name="baseline")
            baseline_opponent = make_direct_agent(opponent_name, opponent_team.packed)
            baseline_outcome = play_battle(
                worker,
                f"value-baseline-{game_index:06d}",
                {learner_side: baseline, other_side: baseline_opponent},
                {learner_side: team.packed, other_side: opponent_team.packed},
                seed=list(battle_seed),
            )
            baseline_fallbacks += baseline.player.fallback_count

            random.seed(policy_seed)
            candidate = (
                _baseline_agent(team.packed, config, name="value_search")
                if args.null_test
                else _candidate_agent(
                    model=model,
                    checkpoint=args.checkpoint,
                    checkpoint_hash=checkpoint_hash,
                    value_weight=args.value_weight,
                    team=team,
                    opponent_team=opponent_team,
                    config=config,
                    device=args.device,
                )
            )
            candidate_opponent = make_direct_agent(opponent_name, opponent_team.packed)
            candidate_outcome = play_battle(
                worker,
                f"value-candidate-{game_index:06d}",
                {learner_side: candidate, other_side: candidate_opponent},
                {learner_side: team.packed, other_side: opponent_team.packed},
                seed=list(battle_seed),
            )
            candidate_fallbacks += candidate.player.fallback_count
            if not args.null_test:
                records.extend(candidate.player.decision_records)
            pairs.append(
                {
                    "team_id": team.label,
                    "opponent_team_id": opponent_team.label,
                    "opponent": opponent_name,
                    "learner_side": learner_side,
                    "shadow_win": _win_value(baseline_outcome.result_for("baseline")),
                    "hybrid_win": _win_value(candidate_outcome.result_for("value_search")),
                }
            )

    strength = _strength_summary(pairs)
    interval = list(strength["team_clustered_difference_interval"])
    total_fallbacks = baseline_fallbacks + candidate_fallbacks
    checks = {
        "minimum_games": int(strength["paired_games"]) >= args.minimum_games,
        "minimum_teams": int(strength["teams"]) >= args.minimum_teams,
        "paired_noninferiority": float(interval[0]) > -args.noninferiority_margin,
        "zero_fallbacks": total_fallbacks == 0,
    }
    if args.null_test:
        checks["exact_null"] = float(strength["difference"]) == 0.0
    write_value_records(args.records, records)
    report = {
        "schema": "vgc-search-value-paired-gate-v1",
        "null_test": args.null_test,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "prior_shadow": str(args.prior_shadow.resolve()),
        "prior_checks": prior_checks,
        "value_weight": args.value_weight,
        "opponents": opponents,
        "paired_strength": strength,
        "fallbacks": {"baseline": baseline_fallbacks, "candidate": candidate_fallbacks},
        "checks": checks,
        "passed": all(checks.values()),
        "strength_proven": float(interval[0]) > 0.0,
        "promotion_authority": False,
        "default_remains_off": True,
        "records": str(args.records.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if bool(report["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())

