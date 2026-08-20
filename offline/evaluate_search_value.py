#!/usr/bin/env python
"""Audit whether the frozen RL value adds information beyond existing full search."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

try:
    import torch
except ImportError as exc:  # pragma: no cover - command requires train extra
    raise ImportError("run `uv sync --extra train` before this audit") from exc

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
from vgc.rl.search_guidance import checkpoint_sha256  # noqa: E402
from vgc.rl.search_value import (  # noqa: E402
    DEFAULT_VALUE_WEIGHTS,
    NeuralValueShadowPlayer,
    summarize_value_records,
    write_value_records,
)

DEFAULT_CHECKPOINT = RUNS_DIR / "full_pipeline" / "rl_promotion_10k" / "best.pt"
DEFAULT_OUTPUT = RUNS_DIR / "eval" / "search_value_shadow.json"
DEFAULT_RECORDS = RUNS_DIR / "eval" / "search_value_shadow.jsonl"
DEFAULT_OPPONENTS = "heuristic,vgc_myopic,vgc_shallow,vgc"


def _parse_weights(raw: str) -> tuple[float, ...]:
    try:
        values = tuple(sorted({float(value.strip()) for value in raw.split(",") if value.strip()}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value weights must be comma-separated numbers") from exc
    if not values or min(values) <= 0.0:
        raise argparse.ArgumentTypeError("value weights must be positive")
    return values


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--team-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--opponents", default=DEFAULT_OPPONENTS)
    parser.add_argument(
        "--value-weights",
        type=_parse_weights,
        default=DEFAULT_VALUE_WEIGHTS,
        help="shadow-only alpha values, comma separated",
    )
    parser.add_argument("--minimum-games", type=int, default=100)
    parser.add_argument("--minimum-teams", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def _checkpoint_reward_contract(path: Path) -> dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    ppo = dict(checkpoint.get("ppo_config") or {})
    contract = {
        "terminal_rewards": [-1.0, 0.0, 1.0],
        "gamma": float(ppo.get("gamma", float("nan"))),
        "reward_shaping_coef": float(ppo.get("reward_shaping_coef", float("nan"))),
    }
    contract["supports_win_probability_conversion"] = (
        contract["gamma"] == 1.0 and contract["reward_shaping_coef"] == 0.0
    )
    return contract


def _make_agent(
    *,
    model,
    checkpoint: Path,
    checkpoint_hash: str,
    team,
    opponent_team,
    config: PolicyConfig,
    device: str,
    value_weights: tuple[float, ...],
) -> DirectAgent:
    return DirectAgent(
        NeuralValueShadowPlayer(
            model=model,
            checkpoint_path=checkpoint,
            checkpoint_hash=checkpoint_hash,
            device=device,
            team_id=team.label,
            opponent_team_id=opponent_team.label,
            value_weights=value_weights,
            config=config,
            team=team.packed,
            battle_format=FORMAT_ID,
            start_listening=False,
        ),
        name="value_shadow",
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if min(args.games, args.minimum_games, args.minimum_teams) <= 0:
        raise SystemExit("--games, --minimum-games, and --minimum-teams must be positive")
    if not 0.0 < args.holdout_fraction < 1.0:
        raise SystemExit("--holdout-fraction must be between 0 and 1")
    opponents = [name.strip() for name in args.opponents.split(",") if name.strip()]
    unknown = sorted(set(opponents) - set(BASELINES))
    if unknown:
        raise SystemExit(f"unknown opponents: {unknown}; available: {sorted(BASELINES)}")
    if not opponents:
        raise SystemExit("--opponents must not be empty")

    reward_contract = _checkpoint_reward_contract(args.checkpoint)
    if not reward_contract["supports_win_probability_conversion"]:
        raise SystemExit(
            "this audit currently requires terminal +/-1 rewards, gamma=1, and no reward shaping"
        )
    model = load_snapshot(args.checkpoint, device=args.device)
    model_hash = checkpoint_sha256(args.checkpoint)
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
    records: list[dict[str, object]] = []
    fallbacks = 0
    with SimWorker(DEFAULT_SHOWDOWN_REPO) as worker:
        for game_index in range(args.games):
            team, opponent_team, opponent_name, learner_side = schedule[
                game_index % len(schedule)
            ]
            battle_seed = [simulator_rng.randrange(1, 2**31) for _ in range(4)]
            policy_seed = args.seed + game_index * 1_000_003
            random.seed(policy_seed)
            agent = _make_agent(
                model=model,
                checkpoint=args.checkpoint,
                checkpoint_hash=model_hash,
                team=team,
                opponent_team=opponent_team,
                config=config,
                device=args.device,
                value_weights=args.value_weights,
            )
            opponent = make_direct_agent(opponent_name, opponent_team.packed)
            other_side = "p2" if learner_side == "p1" else "p1"
            play_battle(
                worker,
                f"value-shadow-{game_index:06d}",
                {learner_side: agent, other_side: opponent},
                {learner_side: team.packed, other_side: opponent_team.packed},
                seed=battle_seed,
            )
            for record in agent.player.decision_records:
                record["opponent"] = opponent_name
                record["learner_side"] = learner_side
            records.extend(agent.player.decision_records)
            fallbacks += agent.player.fallback_count

    write_value_records(args.records, records)
    summary = summarize_value_records(records, args.value_weights)
    incremental = dict(summary["incremental_predictive_value"])
    checks = {
        "minimum_games": args.games >= args.minimum_games,
        "minimum_teams": int(summary["teams"]) >= args.minimum_teams,
        "zero_fallbacks": fallbacks == 0,
        "incremental_metric_available": bool(incremental.get("available")),
    }
    report = {
        "schema": "vgc-search-value-evaluation-v1",
        "mode": "shadow_only",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": model_hash,
        "checkpoint_reward_contract": reward_contract,
        "games": args.games,
        "holdout_teams": len(holdout_teams),
        "opponents": opponents,
        "value_weights": list(args.value_weights),
        "fallbacks": fallbacks,
        "records": str(args.records.resolve()),
        "summary": summary,
        "checks": checks,
        "shadow_audit_complete": all(checks.values()),
        "promotion_authority": False,
        "next_gate": "paired unchanged-search versus search-plus-value battles",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

