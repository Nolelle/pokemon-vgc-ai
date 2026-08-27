#!/usr/bin/env python
"""Build and train the complete-context strategic starting model.

Human-replay behavior cloning supplies the shared state trunk.  Fresh simulator games
then let the existing search policy label one complete legal doubles action at each
decision while all six own sets and fog-safe opponent beliefs are available.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

try:
    import torch
except ImportError as exc:  # pragma: no cover - train extra is optional
    raise SystemExit(
        "train_imitation requires the train extra -- run `uv sync --extra train`."
    ) from exc

from vgc.config import FORMAT_ID, REPO_ROOT
from vgc.models import PolicyConfig
from vgc.rl.agents import DirectAgent, make_direct_agent
from vgc.rl.demonstrations import (
    annotate_samples,
    load_demonstrations,
    save_demonstrations,
    split_samples_grouped,
)
from vgc.rl.distill import (
    DistillationConfig,
    TeacherRecordingPlayer,
    distill_policy,
    evaluate_agreement,
)
from vgc.rl.env import DEFAULT_SHOWDOWN_REPO, SimWorker
from vgc.rl.match import play_battle, run_series, summarize
from vgc.rl.model import CandidatePolicyValueNet
from vgc.rl.opponents import RL_ARCHITECTURE_VERSION
from vgc.rl.player import PpoVgcPlayer
from vgc.rl.ppo import PpoConfig

DEFAULT_TEAM = REPO_ROOT / "teams" / "phase2_mirror.packed.txt"
DEFAULT_BC_CHECKPOINT = REPO_ROOT / "data" / "models" / "bc_policy_v4.pt"
DEFAULT_OUT_DIR = REPO_ROOT / "runs" / "imitation" / "complete_context"
DEFAULT_OPPONENTS = "random,maxpower,heuristic,vgc_myopic,vgc_shallow,vgc"


@dataclass(frozen=True)
class TeamEntry:
    label: str
    packed: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--team", type=Path, default=DEFAULT_TEAM)
    parser.add_argument("--team-manifest", type=Path, default=None)
    parser.add_argument("--opponents", default=DEFAULT_OPPONENTS)
    parser.add_argument("--dataset", type=Path, default=None, help="reuse a saved .pt dataset")
    parser.add_argument("--bc-checkpoint", type=Path, default=DEFAULT_BC_CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument(
        "--split-by",
        choices=("battle", "team"),
        default="battle",
        help="hold out whole battles or whole learner teams",
    )
    parser.add_argument("--eval-games", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--select-metric",
        choices=("loss", "recall_at_k"),
        default="recall_at_k",
        help="validation metric that picks the restored best epoch (default: Recall@10, "
        "the shortlist gate; 'loss' restores the historical cross-entropy selection)",
    )
    parser.add_argument(
        "--hard-example-weight",
        type=float,
        default=0.0,
        help="extra loss weight for samples whose teacher action currently ranks past "
        "--hard-example-rank (0 disables)",
    )
    parser.add_argument("--hard-example-rank", type=int, default=10)
    parser.add_argument(
        "--balance-action-count-bins",
        action="store_true",
        help="equalize expected draws across the legal-action-count bins so rare "
        "large-branching turns get proportional gradient updates",
    )
    parser.add_argument(
        "--head-width",
        type=int,
        default=None,
        help="capacity probe: widen the action-scoring path's hidden width "
        "(default keeps ACTION_HIDDEN_DIM and checkpoint compatibility)",
    )
    parser.add_argument(
        "--soft-targets",
        action="store_true",
        help="train against softmax(teacher search scores / T) instead of the argmax "
        "(schema-v2.x datasets only; rows without stored scores fall back to hard CE)",
    )
    parser.add_argument(
        "--soft-target-temperature",
        type=float,
        default=16.0,
        help="softmax temperature in evaluator points; higher flattens the target",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args(argv)


def load_teams(team_path: Path, manifest_path: Path | None) -> list[TeamEntry]:
    if manifest_path is None:
        return [TeamEntry(label=team_path.stem, packed=team_path.read_text().strip())]
    records = json.loads(manifest_path.read_text())
    teams = [
        TeamEntry(
            label=f"{record.get('archetype', 'unknown')}/{Path(record['file']).stem}",
            packed=(manifest_path.parent / record["file"]).read_text().strip(),
        )
        for record in records
    ]
    if not teams:
        raise ValueError(f"team manifest is empty: {manifest_path}")
    return teams


def collect_demonstrations(
    worker: SimWorker,
    *,
    teams: list[TeamEntry],
    opponents: list[str],
    games: int,
    seed: int,
) -> list:
    rng = random.Random(seed)
    samples: list = []
    for game_index in range(games):
        random.seed(seed + game_index * 1_000_003)
        learner_team = rng.choice(teams)
        opponent_team = rng.choice(teams)
        opponent_name = opponents[game_index % len(opponents)]
        teacher_player = TeacherRecordingPlayer(
            config=PolicyConfig(),
            team=learner_team.packed,
            battle_format=FORMAT_ID,
            start_listening=False,
        )
        teacher = DirectAgent(teacher_player, name="teacher")
        opponent = make_direct_agent(
            opponent_name,
            opponent_team.packed,
        )
        teacher_side = "p1" if game_index % 2 == 0 else "p2"
        other_side = "p2" if teacher_side == "p1" else "p1"
        agents = {teacher_side: teacher, other_side: opponent}
        teams_by_side = {
            teacher_side: learner_team.packed,
            other_side: opponent_team.packed,
        }
        battle_id = f"imitation-{game_index:06d}"
        play_battle(
            worker,
            battle_id,
            agents,
            teams_by_side,
            seed=[rng.randrange(1, 2**31) for _ in range(4)],
        )
        game_samples = [
            sample for sample in teacher_player.distillation_samples if sample.battle_id == battle_id
        ]
        samples.extend(
            annotate_samples(
                game_samples,
                team_id=learner_team.label,
                opponent_team_id=opponent_team.label,
            )
        )
    return samples


def save_model_checkpoint(
    path: Path,
    model: CandidatePolicyValueNet,
    optimizer,
    *,
    metrics: dict[str, object],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "architecture": RL_ARCHITECTURE_VERSION,
            "use_meta_features": model.use_meta_features,
            "use_information_features": model.use_information_features,
            "use_tactical_features": model.use_tactical_features,
            "head_dropout": model.head_dropout_p,
            "value_output_transform": model.value_output_transform,
            "head_width": getattr(model, "head_width", None),
            "stage": "complete_context_imitation",
            "metrics": metrics,
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        },
        path,
    )


def evaluate_games(
    worker: SimWorker,
    *,
    model: CandidatePolicyValueNet,
    teams: list[TeamEntry],
    opponents: list[str],
    games: int,
    seed: int,
    device: str,
) -> dict[str, object]:
    if games <= 0:
        return {}
    results: dict[str, object] = {}
    per_opponent = max(2, games // len(opponents))
    for opponent_index, opponent_name in enumerate(opponents):
        random.seed(seed + opponent_index * 1_000_003)
        team = teams[(seed + opponent_index) % len(teams)]
        outcomes = run_series(
            worker,
            {
                "imitation": lambda team=team: DirectAgent(
                    PpoVgcPlayer(
                        model=model,
                        ppo_config=PpoConfig(teacher_anchor_weight=0.0),
                        deterministic=True,
                        device=device,
                        team=team.packed,
                        battle_format=FORMAT_ID,
                        start_listening=False,
                    ),
                    name="imitation",
                ),
                "opponent": lambda team=team, opponent_name=opponent_name: make_direct_agent(
                    opponent_name, team.packed
                ),
            },
            {"imitation": team.packed, "opponent": team.packed},
            per_opponent,
            seed=seed + opponent_index * 10_000,
        )
        results[opponent_name] = summarize(outcomes, "imitation", "opponent")
    return results


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.games <= 0 and args.dataset is None:
        raise SystemExit("--games must be positive when --dataset is not supplied")
    if args.epochs <= 0 or args.batch_size <= 0:
        raise SystemExit("--epochs and --batch-size must be positive")
    opponents = [name.strip() for name in args.opponents.split(",") if name.strip()]
    if not opponents:
        raise SystemExit("--opponents must list at least one baseline")
    teams = load_teams(args.team, args.team_manifest)
    if args.split_by == "team" and len(teams) < 2:
        raise SystemExit("--split-by team requires at least two teams")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = args.out_dir / "demonstrations.pt"

    with SimWorker(DEFAULT_SHOWDOWN_REPO) as worker:
        if args.dataset is not None:
            samples = load_demonstrations(args.dataset)
        else:
            samples = collect_demonstrations(
                worker,
                teams=teams,
                opponents=opponents,
                games=args.games,
                seed=args.seed,
            )
            save_demonstrations(dataset_path, samples)
        train_samples, validation_samples = split_samples_grouped(
            samples,
            val_fraction=args.val_fraction,
            seed=args.seed,
            group_by=args.split_by,
        )

        torch.manual_seed(args.seed)
        model = CandidatePolicyValueNet(
            use_meta_features=True,
            use_information_features=True,
            use_tactical_features=True,
            head_dropout=0.1,
            head_width=args.head_width,
        ).to(args.device)
        warm_start = {"loaded": 0, "available": 0}
        if args.bc_checkpoint.is_file():
            warm_start = model.warm_start_state_encoder(args.bc_checkpoint)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        before = evaluate_agreement(
            model, validation_samples, batch_size=args.batch_size, device=args.device
        )
        training = distill_policy(
            model,
            optimizer,
            train_samples,
            DistillationConfig(
                epochs=args.epochs,
                batch_size=args.batch_size,
                seed=args.seed,
                early_stopping_patience=max(2, args.epochs // 4),
                checkpoint_metric=args.select_metric,
                checkpoint_recall_k=10,
                hard_example_weight=args.hard_example_weight,
                hard_example_rank=args.hard_example_rank,
                balance_action_count_bins=args.balance_action_count_bins,
                soft_targets=args.soft_targets,
                soft_target_temperature=args.soft_target_temperature,
            ),
            device=args.device,
            val_samples=validation_samples,
        )
        after = evaluate_agreement(
            model, validation_samples, batch_size=args.batch_size, device=args.device
        )
        game_evaluation = evaluate_games(
            worker,
            model=model,
            teams=teams,
            opponents=opponents,
            games=args.eval_games,
            seed=args.seed + 1_000_000,
            device=args.device,
        )

    improved = (
        after["teacher_probability"] > before["teacher_probability"]
        or after["loss"] < before["loss"]
    )
    metrics: dict[str, object] = {
        "sample_count": len(samples),
        "battle_count": len({sample.battle_id for sample in samples}),
        "team_count": len({sample.team_id for sample in samples}),
        "train_samples": len(train_samples),
        "validation_samples": len(validation_samples),
        "warm_start": warm_start,
        "before": before,
        "training": training,
        "after": after,
        "teacher_probability_improved": improved,
        "game_evaluation": game_evaluation,
    }
    metrics_path = args.out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    save_model_checkpoint(
        args.out_dir / "best.pt", model, optimizer, metrics=metrics, args=args
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"demonstrations: {dataset_path if args.dataset is None else args.dataset}")
    print(f"checkpoint: {args.out_dir / 'best.pt'}")
    print(f"metrics: {metrics_path}")
    if not improved:
        raise SystemExit("validation teacher probability did not improve")


if __name__ == "__main__":
    from vgc.mechanics_gate import enforce_mechanics_gate_for_cli

    enforce_mechanics_gate_for_cli("model training")
    main()
