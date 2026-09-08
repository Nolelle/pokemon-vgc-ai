#!/usr/bin/env python
"""Conservative held-out-team promotion gate for a learned joint-action policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

try:
    import torch
except ImportError as exc:  # pragma: no cover - train extra is optional
    raise SystemExit(
        "evaluate_learned_policy requires the train extra -- run `uv sync --extra train`."
    ) from exc

from selfplay.train_full_pipeline import (  # noqa: E402
    DEFAULT_MANIFEST,
    evaluate_policy,
    load_team_pool,
    split_teams_by_archetype,
)
from vgc.baselines import BASELINES  # noqa: E402
from vgc.config import RUNS_DIR  # noqa: E402
from vgc.rl.env import DEFAULT_SHOWDOWN_REPO, SimWorker  # noqa: E402
from vgc.rl.opponents import load_snapshot  # noqa: E402

DEFAULT_CHECKPOINT = RUNS_DIR / "full_pipeline" / "rl" / "best.pt"
DEFAULT_OUTPUT = RUNS_DIR / "eval" / "learned_policy_gate.json"
DEFAULT_OPPONENTS = "random,maxpower,heuristic,vgc_myopic,vgc_shallow,vgc"


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        while chunk := checkpoint_file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def materialize_evaluated_checkpoint(
    *,
    source: Path,
    output: Path,
    payload: dict[str, object],
    train_result: dict[str, object],
    holdout_result: dict[str, object],
) -> dict[str, object]:
    """Preserve a preselected checkpoint with the gate evaluation attached.

    Scaling-study endpoints are selected by game count rather than by an in-training
    score.  This creates a new, explicitly named artifact instead of overwriting the
    original endpoint, while keeping the model and optimizer bytes represented by the
    payload unchanged.
    """

    if source.resolve() == output.resolve():
        raise ValueError("evaluated checkpoint output must not overwrite its source")
    evaluated_payload = dict(payload)
    evaluated_payload["evaluation"] = {
        "schema": "vgc-promotion-gate-linked-evaluation-v1",
        "source_checkpoint": str(source.resolve()),
        "source_checkpoint_sha256": checkpoint_sha256(source),
        "train": train_result,
        "holdout": holdout_result,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(evaluated_payload, output)
    return evaluated_payload


def build_gate_report(
    *,
    checkpoint: Path,
    checkpoint_payload: dict[str, object],
    train_result: dict[str, object],
    holdout_result: dict[str, object],
    incumbent: str,
    minimum_games: int,
    minimum_rl_games: int,
    overall_threshold: float,
    incumbent_threshold: float,
    minimum_opponent_rate: float,
    maximum_generalization_gap: float,
) -> dict[str, object]:
    incumbent_result = dict(holdout_result.get("opponent_results", {})).get(incumbent)
    heldout_rate = float(holdout_result["win_rate"])
    train_rate = float(train_result["win_rate"])
    generalization_gap = max(0.0, train_rate - heldout_rate)
    heldout_low = float(holdout_result["clustered_interval"][0])
    incumbent_low = (
        float(incumbent_result["wilson_interval"][0])
        if isinstance(incumbent_result, dict)
        else 0.0
    )
    checks = {
        "complete_context_checkpoint": bool(
            checkpoint_payload.get("use_information_features")
            and checkpoint_payload.get("use_tactical_features")
            and checkpoint_payload.get("use_mechanics_features")
        ),
        "evaluation_linked_checkpoint": isinstance(
            checkpoint_payload.get("evaluation"), dict
        ),
        "minimum_rl_experience": int(checkpoint_payload.get("games_seen", 0))
        >= minimum_rl_games,
        "minimum_heldout_games": int(holdout_result["games"]) >= minimum_games,
        "heldout_clustered_lower_bound": heldout_low > overall_threshold,
        "beats_shipped_policy_lower_bound": incumbent_low > incumbent_threshold,
        "no_collapsed_opponent": float(holdout_result["min_opponent_win_rate"])
        >= minimum_opponent_rate,
        "generalization_gap": generalization_gap <= maximum_generalization_gap,
        "zero_fallbacks": int(train_result["fallback_count"])
        + int(holdout_result["fallback_count"])
        == 0,
    }
    return {
        "schema": "vgc-learned-policy-gate-v1",
        "passed": all(checks.values()),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256(checkpoint),
        "checkpoint_iteration": int(checkpoint_payload.get("iteration", 0)),
        "checkpoint_games_seen": int(checkpoint_payload.get("games_seen", 0)),
        "incumbent": incumbent,
        "thresholds": {
            "minimum_games": minimum_games,
            "minimum_rl_games": minimum_rl_games,
            "overall_clustered_lower_bound_above": overall_threshold,
            "incumbent_wilson_lower_bound_above": incumbent_threshold,
            "minimum_opponent_win_rate": minimum_opponent_rate,
            "maximum_generalization_gap": maximum_generalization_gap,
        },
        "generalization_gap": generalization_gap,
        "checks": checks,
        "train_teams": train_result,
        "heldout_teams": holdout_result,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--team-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--games-per-split", type=int, default=1000)
    parser.add_argument("--minimum-games", type=int, default=1000)
    parser.add_argument("--minimum-rl-games", type=int, default=10_000)
    parser.add_argument("--opponents", default=DEFAULT_OPPONENTS)
    parser.add_argument("--incumbent", default="vgc")
    parser.add_argument("--overall-threshold", type=float, default=0.50)
    parser.add_argument("--incumbent-threshold", type=float, default=0.50)
    parser.add_argument("--minimum-opponent-rate", type=float, default=0.45)
    parser.add_argument("--maximum-generalization-gap", type=float, default=0.05)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--evaluated-checkpoint-output",
        type=Path,
        default=None,
        help="write a copy of a preselected checkpoint with this gate evaluation attached",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.games_per_split <= 0 or args.minimum_games <= 0:
        raise SystemExit("--games-per-split and --minimum-games must be positive")
    if args.minimum_rl_games < 0:
        raise SystemExit("--minimum-rl-games must be nonnegative")
    bounded_thresholds = {
        "--overall-threshold": args.overall_threshold,
        "--incumbent-threshold": args.incumbent_threshold,
        "--minimum-opponent-rate": args.minimum_opponent_rate,
        "--maximum-generalization-gap": args.maximum_generalization_gap,
    }
    invalid = [name for name, value in bounded_thresholds.items() if not 0.0 <= value <= 1.0]
    if invalid:
        raise SystemExit(f"gate thresholds must be between 0 and 1: {invalid}")
    opponents = [name.strip() for name in args.opponents.split(",") if name.strip()]
    unknown = sorted(set(opponents) - set(BASELINES))
    if unknown:
        raise SystemExit(f"unknown opponents: {unknown}; available: {sorted(BASELINES)}")
    if args.incumbent not in opponents:
        raise SystemExit("--incumbent must be included in --opponents")
    checkpoint_payload = torch.load(
        args.checkpoint, map_location=args.device, weights_only=False
    )
    model = load_snapshot(args.checkpoint, device=args.device)
    all_teams = load_team_pool(args.team_manifest)
    train_teams, holdout_teams = split_teams_by_archetype(
        all_teams, holdout_fraction=args.holdout_fraction, seed=args.seed
    )
    with SimWorker(DEFAULT_SHOWDOWN_REPO) as worker:
        train_result = evaluate_policy(
            worker,
            model=model,
            teams=train_teams,
            opponents=opponents,
            games=args.games_per_split,
            seed=args.seed + 10_000_000,
            device=args.device,
        )
        holdout_result = evaluate_policy(
            worker,
            model=model,
            teams=holdout_teams,
            opponents=opponents,
            games=args.games_per_split,
            seed=args.seed + 20_000_000,
            device=args.device,
        )
    report_checkpoint = args.checkpoint
    if args.evaluated_checkpoint_output is not None:
        checkpoint_payload = materialize_evaluated_checkpoint(
            source=args.checkpoint,
            output=args.evaluated_checkpoint_output,
            payload=checkpoint_payload,
            train_result=train_result,
            holdout_result=holdout_result,
        )
        report_checkpoint = args.evaluated_checkpoint_output
    report = build_gate_report(
        checkpoint=report_checkpoint,
        checkpoint_payload=checkpoint_payload,
        train_result=train_result,
        holdout_result=holdout_result,
        incumbent=args.incumbent,
        minimum_games=args.minimum_games,
        minimum_rl_games=args.minimum_rl_games,
        overall_threshold=args.overall_threshold,
        incumbent_threshold=args.incumbent_threshold,
        minimum_opponent_rate=args.minimum_opponent_rate,
        maximum_generalization_gap=args.maximum_generalization_gap,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
