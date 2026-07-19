#!/usr/bin/env python
"""CLI for `vgc.bc.train` -- trains the Phase 3 behavior-cloning PIPELINE-VALIDATION
policy over `data/bc/decisions.jsonl` (`tools/parse_replays.py`'s output).

SCOPE: this validates a pipeline (encoder -> dataset -> training -> accuracy), it does
NOT produce a playable agent -- see `vgc.bc.train`'s module docstring. Requires the
`train` extra: `uv sync --extra train`.

Usage:
    .venv/bin/python tools/train_bc.py
    .venv/bin/python tools/train_bc.py --epochs 30 --min-rating 1150 \
        --extra-checkpoint data/models/bc_policy_v2.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vgc.bc.train import TrainConfig, train  # noqa: E402
from vgc.config import DATA_DIR  # noqa: E402

DEFAULT_DATA_PATH = DATA_DIR.parent / "bc" / "decisions.jsonl"
DEFAULT_OUT_DIR = REPO_ROOT / "runs" / "bc"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="decisions.jsonl path")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--min-rating",
        type=int,
        default=1150,
        help="skip records from replays rated below this, and null-rated replays",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="auto = mps if available, else cpu",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT_DIR, help="checkpoint output directory"
    )
    parser.add_argument(
        "--target-loss-weight",
        type=float,
        default=0.5,
        help="weight on the target-head loss: total = move_loss + weight * target_loss + ...",
    )
    parser.add_argument(
        "--value-loss-weight",
        type=float,
        default=1.0,
        help="weight on the value-head loss: total = move_loss + ... + weight * value_loss",
    )
    parser.add_argument(
        "--heads",
        default="move,target,value",
        help="comma-separated subset of move,target,value to train (default: all three)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="early-stop after this many epochs without a new best val move top-1",
    )
    parser.add_argument(
        "--extra-checkpoint",
        type=Path,
        default=None,
        help="also save the best checkpoint to this path (e.g. data/models/bc_policy_v3.pt)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.data.exists():
        print(f"{args.data} does not exist -- run tools/parse_replays.py first", file=sys.stderr)
        return 1

    heads = tuple(h.strip() for h in args.heads.split(",") if h.strip())
    config = TrainConfig(
        data=str(args.data),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        min_rating=args.min_rating,
        device=args.device,
        out_dir=str(args.out),
        target_loss_weight=args.target_loss_weight,
        value_loss_weight=args.value_loss_weight,
        patience=args.patience,
        extra_checkpoint_path=str(args.extra_checkpoint) if args.extra_checkpoint else None,
        heads=heads,
    )
    result = train(config)

    print("\ntraining summary:")
    print(
        f"  train samples:          {result['train_samples']} (skipped {result['train_skipped']})"
    )
    print(f"  val samples:            {result['val_samples']} (skipped {result['val_skipped']})")
    print(f"  move baseline:          {result['move_baseline']:.4f}")
    print(f"  target baseline:        {result['target_baseline']:.4f}")
    print(
        f"  value label balance:    {result['value_positive']}/{result['value_total']} True "
        f"({result['value_label_rate']:.1%})"
    )
    print(f"  value baseline:         {result['value_baseline']:.4f}")
    print(f"  best val move top-1:    {result['best_val_move_top1']:.4f}")
    print(f"  best val move top-3:    {result['best_val_move_top3']:.4f}")
    print(f"  best val target top-1:  {result['best_val_target_top1']:.4f}")
    print(f"  best val value acc:     {result['best_val_value_accuracy']:.4f}")
    print(f"  best val value AUC:     {result['best_val_value_auc']:.4f}")
    for name, acc in result["best_val_value_accuracy_by_turn_bucket"].items():
        print(f"    value acc (turn {name}):  {acc:.4f}")
    print(
        f"  best epoch:             {result['best_epoch']} (ran {result['epochs_run']}/{args.epochs})"
    )
    print(f"  device:                 {result['device']}")
    print(f"  elapsed:                {result['elapsed_seconds']:.1f}s")
    print(f"  checkpoint:             {result['checkpoint_path']}")
    if result["extra_checkpoint_path"]:
        print(f"  extra checkpoint:       {result['extra_checkpoint_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
