#!/usr/bin/env python
"""Evaluate saved RL milestones on the frozen scaling-study protocol.

The scaling experiment freezes three things across every replicate: the unseen-team
split (`--split-seed`), the battle schedule inside each evaluation (`--eval-seed`), and
the opponent set. This script applies exactly that protocol to any list of milestone
checkpoints (e.g. `runs/full_pipeline/rl_scale_seedX/milestones/games_*.pt`), writing
one aggregate JSONL row per (checkpoint, split) plus a per-game JSONL whose rows can be
joined on `game_index` across checkpoints evaluated under the same seed -- turning
"two independent win rates" into a much tighter PAIRED comparison.

Learned checkpoints can also appear as OPPONENTS via repeated
`--checkpoint-opponent name=path` (for example the original 10k policy), which is how
the curve answers "does the 100k policy actually beat the 10k policy head-to-head".

Rows already present in the output file (same checkpoint bytes + split) are skipped, so
the script can be re-run incrementally as new milestones appear.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

try:
    import torch
except ImportError as exc:  # pragma: no cover - train extra is optional
    raise SystemExit(
        "evaluate_scaling_milestones requires the train extra -- run `uv sync --extra train`."
    ) from exc

from offline.evaluate_learned_policy import checkpoint_sha256  # noqa: E402
from selfplay.train_full_pipeline import (  # noqa: E402
    DEFAULT_FROZEN_SEED,
    DEFAULT_MANIFEST,
    evaluate_policy,
    load_team_pool,
    split_teams_by_archetype,
)
from vgc.baselines import BASELINES  # noqa: E402
from vgc.config import RUNS_DIR  # noqa: E402
from vgc.rl.env import DEFAULT_SHOWDOWN_REPO, SimWorker  # noqa: E402
from vgc.rl.opponents import load_snapshot  # noqa: E402

DEFAULT_OPPONENTS = "random,maxpower,heuristic,vgc_myopic,vgc_shallow,vgc"
SPLIT_SEED_OFFSETS = {"train": 10_000_000, "holdout": 20_000_000}


def parse_checkpoint_opponents(raw: list[str]) -> dict[str, Path]:
    """Parse repeated `name=path` flags into an ordered name -> checkpoint map."""

    parsed: dict[str, Path] = {}
    for entry in raw:
        name, separator, path_text = entry.partition("=")
        if not separator or not name or not path_text:
            raise SystemExit(f"--checkpoint-opponent entry {entry!r} must be name=path")
        if name in parsed:
            raise SystemExit(f"--checkpoint-opponent name {name!r} given twice")
        path = Path(path_text)
        if not path.exists():
            raise SystemExit(f"--checkpoint-opponent path does not exist: {path}")
        if name in BASELINES:
            raise SystemExit(f"--checkpoint-opponent name {name!r} collides with a fixed baseline")
        parsed[name] = path
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        nargs="+",
        required=True,
        help="milestone/best/latest checkpoints to score (shell globs are fine)",
    )
    parser.add_argument(
        "--seed-label",
        default="",
        help="replicate label stored on every row (e.g. seed20260901)",
    )
    parser.add_argument("--team-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_FROZEN_SEED)
    parser.add_argument("--eval-seed", type=int, default=DEFAULT_FROZEN_SEED)
    parser.add_argument("--opponents", default=DEFAULT_OPPONENTS)
    parser.add_argument(
        "--checkpoint-opponent",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="add learned checkpoint PATH as an extra opponent under NAME",
    )
    parser.add_argument("--games-per-split", type=int, default=500)
    parser.add_argument(
        "--splits",
        default="holdout",
        help="comma list from {train,holdout} (default holdout only)",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=RUNS_DIR / "full_pipeline" / "rl_scale" / "milestone_eval.jsonl",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.games_per_split <= 0:
        raise SystemExit("--games-per-split must be positive")
    splits = [part.strip() for part in args.splits.split(",") if part.strip()]
    unknown_splits = sorted(set(splits) - set(SPLIT_SEED_OFFSETS))
    if unknown_splits:
        raise SystemExit(f"unknown splits: {unknown_splits}")
    opponents = [part.strip() for part in args.opponents.split(",") if part.strip()]
    unknown = sorted(set(opponents) - set(BASELINES))
    if unknown:
        raise SystemExit(f"unknown opponents: {unknown}; available: {sorted(BASELINES)}")
    extra_opponents = parse_checkpoint_opponents(args.checkpoint_opponent)

    checkpoints: list[Path] = []
    seen_paths: set[Path] = set()
    for path in args.checkpoint:
        resolved = path.resolve()
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        if not resolved.exists():
            raise SystemExit(f"checkpoint does not exist: {path}")
        checkpoints.append(resolved)

    # Resume-friendly skip list keyed by checkpoint fingerprint + split.
    done: set[tuple[str, str]] = set()
    if args.output.exists():
        for line in args.output.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add((row.get("checkpoint_sha256"), row.get("split")))

    all_teams = load_team_pool(args.team_manifest)
    train_teams, holdout_teams = split_teams_by_archetype(
        all_teams, holdout_fraction=args.holdout_fraction, seed=args.split_seed
    )
    teams_by_split = {"train": train_teams, "holdout": holdout_teams}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with SimWorker(DEFAULT_SHOWDOWN_REPO) as worker:
        for checkpoint in checkpoints:
            digest = checkpoint_sha256(checkpoint)
            payload = torch.load(checkpoint, map_location=args.device, weights_only=False)
            games_seen = int(payload.get("games_seen", -1))
            iteration = int(payload.get("iteration", -1))
            for split in splits:
                if (digest, split) in done:
                    print(f"skip {checkpoint.name} [{split}] (already evaluated)")
                    continue
                records_path = (
                    args.output.parent / f"{args.output.stem}.{checkpoint.stem}.{split}.games.jsonl"
                )
                result = evaluate_policy(
                    worker,
                    model=load_snapshot(checkpoint, device=args.device),
                    teams=teams_by_split[split],
                    opponents=opponents,
                    games=args.games_per_split,
                    seed=args.eval_seed + SPLIT_SEED_OFFSETS[split],
                    device=args.device,
                    extra_opponents=dict(extra_opponents),
                    record_games_path=records_path,
                )
                row = {
                    "schema": "vgc-scaling-milestone-eval-v1",
                    "seed_label": args.seed_label,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": digest,
                    "checkpoint_stem": checkpoint.stem,
                    "games_seen": games_seen,
                    "iteration": iteration,
                    "split": split,
                    "opponents": [*opponents, *extra_opponents],
                    "games_per_split": args.games_per_split,
                    "eval_seed": args.eval_seed,
                    "split_seed": args.split_seed,
                    "game_records": str(records_path),
                    "result": result,
                }
                with args.output.open("a") as file:
                    file.write(json.dumps(row, sort_keys=True) + "\n")
                written += 1
                rates = {
                    name: round(float(rate), 3) for name, rate in result["by_opponent"].items()
                }
                print(
                    f"{checkpoint.name} [{split}] games={games_seen}: "
                    f"overall={result['win_rate']:.3f} {rates}",
                    flush=True,
                )
    print(f"wrote {written} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
