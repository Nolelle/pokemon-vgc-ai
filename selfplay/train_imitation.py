#!/usr/bin/env python
"""Build and train the complete-context strategic starting model.

Human-replay behavior cloning supplies the shared state trunk.  Fresh simulator games
then let the existing search policy label one complete legal doubles action at each
decision while all six own sets and fog-safe opponent beliefs are available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

try:
    import torch
except ImportError as exc:  # pragma: no cover - train extra is optional
    raise SystemExit(
        "train_imitation requires the train extra -- run `uv sync --extra train`."
    ) from exc

from vgc.config import FORMAT_ID, REPO_ROOT

sys.path.insert(0, str(REPO_ROOT))  # sibling `offline/` tooling (shard loading)
from vgc.models import PolicyConfig
from vgc.rl.agents import DirectAgent, make_direct_agent
from vgc.rl.demonstrations import (
    INFORMATION_CONTRACT_VERSION,
    annotate_samples,
    load_demonstration_dataset,
    save_demonstrations,
    save_split_manifest,
    split_samples_grouped,
)
from vgc.rl.distill import (
    DistillationConfig,
    PUBLIC_TEACHER_SOURCE_ID,
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
    sha256: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument(
        "--local-smoke",
        action="store_true",
        help="use a tiny exact-search budget for validating data collection, not training",
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="save and split demonstrations, then stop before fitting a model",
    )
    parser.add_argument("--team", type=Path, default=DEFAULT_TEAM)
    parser.add_argument("--team-manifest", type=Path, default=None)
    parser.add_argument("--opponents", default=DEFAULT_OPPONENTS)
    parser.add_argument(
        "--dataset",
        type=Path,
        nargs="+",
        default=None,
        help="reuse saved .pt datasets (several paths train on their concatenation, "
        "which must pass the same-pool invariant check as merge_collection_shards)",
    )
    parser.add_argument("--bc-checkpoint", type=Path, default=DEFAULT_BC_CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument(
        "--split-by",
        choices=("battle", "team"),
        default="team",
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
        "--max-skip-rate",
        type=float,
        default=0.05,
        help="fail the run when skipped teacher decisions (exact-search failures "
        "with a myopic-evaluator fallback played instead) exceed this fraction of "
        "attempted decisions. Skips are always reported by cause; 0 restores the "
        "original fail-on-any-skip behavior.",
    )
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
        packed = team_path.read_text().strip()
        return [
            TeamEntry(
                label=team_path.stem,
                packed=packed,
                sha256=hashlib.sha256(packed.encode()).hexdigest(),
            )
        ]
    records = json.loads(manifest_path.read_text())
    teams = [
        TeamEntry(
            label=f"{record.get('archetype', 'unknown')}/{Path(record['file']).stem}",
            packed=(packed := (manifest_path.parent / record["file"]).read_text().strip()),
            sha256=hashlib.sha256(packed.encode()).hexdigest(),
        )
        for record in records
    ]
    if not teams:
        raise ValueError(f"team manifest is empty: {manifest_path}")
    return teams


def split_team_entries(
    teams: list[TeamEntry], *, val_fraction: float, seed: int
) -> tuple[list[TeamEntry], list[TeamEntry]]:
    """Split content-unique teams before games are paired, keeping both sides disjoint."""

    if len({team.sha256 for team in teams}) != len(teams):
        raise ValueError("team source contains duplicate packed-team contents")
    if len(teams) < 2:
        raise ValueError("team-level validation requires at least two content-unique teams")
    grouped: dict[str, list[TeamEntry]] = {}
    for team in teams:
        archetype = team.label.split("/", 1)[0]
        grouped.setdefault(archetype, []).append(team)
    train: list[TeamEntry] = []
    validation: list[TeamEntry] = []
    for index, archetype in enumerate(sorted(grouped)):
        choices = sorted(grouped[archetype], key=lambda team: team.label)
        random.Random(seed + index).shuffle(choices)
        if len(choices) == 1:
            train.extend(choices)
            continue
        count = max(1, min(len(choices) - 1, round(len(choices) * val_fraction)))
        validation.extend(choices[:count])
        train.extend(choices[count:])
    if not train or not validation:
        raise ValueError("team-level validation produced an empty partition")
    return train, validation


def check_collection_skip_rate(
    *, attempted: int, skipped: int, max_skip_rate: float
) -> float:
    """Fail closed when too many teacher decisions went unlabeled."""

    skip_rate = skipped / attempted if attempted else 0.0
    if skip_rate > max_skip_rate:
        raise RuntimeError(
            f"teacher collection skip rate {skip_rate:.3%} exceeds --max-skip-rate "
            f"{max_skip_rate:.3%} over {attempted} decisions"
        )
    return skip_rate


def collect_demonstrations(
    worker: SimWorker,
    *,
    teams: list[TeamEntry],
    opponents: list[str],
    games: int,
    seed: int,
    teacher_config: PolicyConfig,
    battle_prefix: str = "imitation",
    max_skip_rate: float = 0.0,
) -> list:
    rng = random.Random(seed)
    samples: list = []
    attempted = 0
    skipped = 0
    myopic_fallbacks = 0
    random_fallbacks = 0
    skip_causes: dict[str, int] = {}
    for game_index in range(games):
        random.seed(seed + game_index * 1_000_003)
        learner_team = rng.choice(teams)
        opponent_team = rng.choice(teams)
        opponent_name = opponents[game_index % len(opponents)]
        teacher_player = TeacherRecordingPlayer(
            config=teacher_config,
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
        battle_id = f"{battle_prefix}-{game_index:06d}"
        try:
            play_battle(
                worker,
                battle_id,
                agents,
                teams_by_side,
                seed=[rng.randrange(1, 2**31) for _ in range(4)],
            )
        finally:
            teacher_player.close_public_mirror()
        attempted += teacher_player.attempted_decisions()
        skipped += teacher_player.skipped_decisions()
        myopic_fallbacks += teacher_player.skipped_fallback_to_myopic
        random_fallbacks += teacher_player.skipped_fallback_to_random
        for failure in teacher_player.recording_failures:
            cause = failure.split("exact search raised", 1)[-1].strip()[:300] if "exact search raised" in failure else failure.rsplit(":", 1)[-1].strip()[:300]
            skip_causes[cause] = skip_causes.get(cause, 0) + 1
        if teacher_player.fallback_count:
            raise RuntimeError(
                f"teacher collection failed closed in {battle_id}: "
                f"{teacher_player.fallback_count} exception fallbacks"
            )
        game_samples = [
            sample for sample in teacher_player.distillation_samples if sample.battle_id == battle_id
        ]
        samples.extend(
            annotate_samples(
                game_samples,
                team_id=learner_team.label,
                opponent_team_id=opponent_team.label,
                team_sha256=learner_team.sha256,
                opponent_team_sha256=opponent_team.sha256,
            )
        )
    skip_rate = check_collection_skip_rate(
        attempted=attempted, skipped=skipped, max_skip_rate=max_skip_rate
    )
    print(
        f"collection skips: {skipped}/{attempted} decisions "
        f"({skip_rate:.3%}; {myopic_fallbacks} continued on myopic evaluator, "
        f"{random_fallbacks} on random)",
        flush=True,
    )
    for cause, count in sorted(skip_causes.items(), key=lambda row: -row[1])[:10]:
        print(f"  skip cause x{count}: {cause}", flush=True)
    return samples


def _git_commit(path: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_dirty(path: Path) -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def dataset_metadata(
    args: argparse.Namespace,
    *,
    teacher_config: PolicyConfig,
    opponents: list[str],
) -> dict[str, object]:
    team_source = args.team_manifest or args.team
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit": _git_commit(REPO_ROOT),
        "repository_dirty": _git_dirty(REPO_ROOT),
        "showdown_commit": _git_commit(DEFAULT_SHOWDOWN_REPO),
        "showdown_dirty": _git_dirty(DEFAULT_SHOWDOWN_REPO),
        "format_id": FORMAT_ID,
        "collector": "selfplay/train_imitation.py",
        "requested_games": args.games,
        "seed": args.seed,
        "team_source": str(Path(team_source).resolve()),
        "opponents": opponents,
        "policy_config": asdict(teacher_config),
        "information_contract": INFORMATION_CONTRACT_VERSION,
        "teacher_source": PUBLIC_TEACHER_SOURCE_ID,
        "collector_fail_closed": True,
        "local_smoke": bool(args.local_smoke),
    }


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
            "use_mechanics_features": model.use_mechanics_features,
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


def load_training_dataset(paths: Sequence[Path]) -> tuple[list, dict[str, object]]:
    """Load one dataset file or concatenate several shard files.

    Multi-file inputs pass the same same-pool invariant check as
    `offline/merge_collection_shards.py`, and battle ids are namespaced per
    shard file so the audit cannot mistake identically-numbered games for
    duplicated decisions.
    """

    if len(paths) == 1:
        return load_demonstration_dataset(paths[0])
    from offline.merge_collection_shards import load_shard_datasets

    samples, shard_metadata = load_shard_datasets(paths)
    metadata = {
        **shard_metadata[0],
        "source_datasets": [str(path.resolve()) for path in paths],
    }
    return samples, metadata


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
    if args.split_by == "team" and len(teams) < 2 and args.dataset is None:
        raise SystemExit("--split-by team requires at least two teams")
    if args.split_by == "team" and args.games < 2 and args.dataset is None:
        raise SystemExit("--split-by team collection requires at least two games")
    teacher_config = replace(
        PolicyConfig(),
        accept_open_team_sheet=False,
        **(
            {
                "search_our_candidates": 2,
                "search_opp_candidates": 2,
                "exact_search_future_samples": 1,
                "exact_search_state_hypotheses": 1,
                "exact_search_spread_hypotheses": 1,
                "use_rolling_horizon": False,
                "use_value_head": False,
            }
            if args.local_smoke
            else {}
        ),
    )
    if args.dataset is None and not args.local_smoke:
        dirty_sources = [
            name
            for name, path in (
                ("pokemon-vgc-ai", REPO_ROOT),
                ("pokemon-showdown", DEFAULT_SHOWDOWN_REPO),
            )
            if _git_dirty(path)
        ]
        if dirty_sources:
            raise SystemExit(
                "large data collection requires clean source checkouts; commit or "
                f"remove local changes in {', '.join(dirty_sources)}"
            )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = args.out_dir / "demonstrations.pt"
    split_manifest_path = args.out_dir / "split_manifest.json"

    with SimWorker(DEFAULT_SHOWDOWN_REPO) as worker:
        if args.dataset is not None:
            samples, metadata = load_training_dataset(args.dataset)
            train_samples, validation_samples = split_samples_grouped(
                samples,
                val_fraction=args.val_fraction,
                seed=args.seed,
                group_by=args.split_by,
            )
            audited_dataset_path = args.dataset[0] if len(args.dataset) == 1 else None
        else:
            metadata = dataset_metadata(
                args,
                teacher_config=teacher_config,
                opponents=opponents,
            )
            if args.split_by == "team":
                train_teams, validation_teams = split_team_entries(
                    teams,
                    val_fraction=args.val_fraction,
                    seed=args.seed,
                )
                validation_games = max(
                    1,
                    min(args.games - 1, round(args.games * args.val_fraction)),
                )
                training_games = args.games - validation_games
                train_samples = collect_demonstrations(
                    worker,
                    teams=train_teams,
                    opponents=opponents,
                    games=training_games,
                    seed=args.seed,
                    teacher_config=teacher_config,
                    battle_prefix="imitation-train",
                    max_skip_rate=args.max_skip_rate,
                )
                validation_samples = collect_demonstrations(
                    worker,
                    teams=validation_teams,
                    opponents=opponents,
                    games=validation_games,
                    seed=args.seed + 1_000_000,
                    teacher_config=teacher_config,
                    battle_prefix="imitation-validation",
                    max_skip_rate=args.max_skip_rate,
                )
                samples = [*train_samples, *validation_samples]
            else:
                samples = collect_demonstrations(
                    worker,
                    teams=teams,
                    opponents=opponents,
                    games=args.games,
                    seed=args.seed,
                    teacher_config=teacher_config,
                    max_skip_rate=args.max_skip_rate,
                )
                train_samples, validation_samples = split_samples_grouped(
                    samples,
                    val_fraction=args.val_fraction,
                    seed=args.seed,
                    group_by=args.split_by,
                )
            save_demonstrations(dataset_path, samples, metadata=metadata)
            audited_dataset_path = dataset_path
        if args.dataset is not None and len(args.dataset) == 1:
            save_split_manifest(
                split_manifest_path,
                dataset_path=args.dataset[0],
                train=train_samples,
                validation=validation_samples,
                group_by=args.split_by,
                seed=args.seed,
                val_fraction=args.val_fraction,
            )
        elif args.dataset is not None:
            save_split_manifest(
                split_manifest_path,
                dataset_paths=args.dataset,
                train=train_samples,
                validation=validation_samples,
                group_by=args.split_by,
                seed=args.seed,
                val_fraction=args.val_fraction,
            )
        else:
            save_split_manifest(
                split_manifest_path,
                dataset_path=audited_dataset_path,
                train=train_samples,
                validation=validation_samples,
                group_by=args.split_by,
                seed=args.seed,
                val_fraction=args.val_fraction,
            )

        if args.collect_only:
            metrics: dict[str, object] = {
                "mode": "collect_only",
                "sample_count": len(samples),
                "battle_count": len({sample.battle_id for sample in samples}),
                "team_count": len({sample.team_sha256 for sample in samples}),
                "opponent_team_count": len(
                    {sample.opponent_team_sha256 for sample in samples}
                ),
                "train_samples": len(train_samples),
                "validation_samples": len(validation_samples),
                "dataset_metadata": metadata,
                "dataset": (
                    str(audited_dataset_path)
                    if audited_dataset_path is not None
                    else sorted(str(path.resolve()) for path in (args.dataset or ()))
                ),
                "split_manifest": str(split_manifest_path),
            }
            metrics_path = args.out_dir / "collection_summary.json"
            metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
            print(json.dumps(metrics, indent=2, sort_keys=True))
            print(f"collection summary: {metrics_path}")
            return

        torch.manual_seed(args.seed)
        model = CandidatePolicyValueNet(
            use_meta_features=True,
            use_information_features=True,
            use_mechanics_features=True,
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
        "dataset_metadata": metadata,
        "split_manifest": str(split_manifest_path),
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
    print(f"split manifest: {split_manifest_path}")
    print(f"checkpoint: {args.out_dir / 'best.pt'}")
    print(f"metrics: {metrics_path}")
    if not improved:
        raise SystemExit("validation teacher probability did not improve")


if __name__ == "__main__":
    from vgc.mechanics_gate import enforce_mechanics_gate_for_cli
    from vgc.battle_state_gate import enforce_battle_state_gate_for_cli
    from vgc.action_gate import enforce_action_gate_for_cli

    enforce_mechanics_gate_for_cli("model training")
    enforce_battle_state_gate_for_cli("model training")
    enforce_action_gate_for_cli("model training")
    main()
