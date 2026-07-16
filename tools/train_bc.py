#!/usr/bin/env python
"""CLI for `vgc.bc.train` -- trains the Phase 3 behavior-cloning PIPELINE-VALIDATION
policy over `data/bc/decisions.jsonl` (`tools/parse_replays.py`'s output).

SCOPE: this validates a pipeline (encoder -> dataset -> training -> accuracy), it does
NOT produce a playable agent -- see `vgc.bc.train`'s module docstring. Requires the
`train` extra: `uv sync --extra train`.

Usage:
    .venv/bin/python tools/train_bc.py
    .venv/bin/python tools/train_bc.py --epochs 3 --min-rating 1150
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
    parser.add_argument("--epochs", type=int, default=5)
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.data.exists():
        print(f"{args.data} does not exist -- run tools/parse_replays.py first", file=sys.stderr)
        return 1

    config = TrainConfig(
        data=str(args.data),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        min_rating=args.min_rating,
        device=args.device,
        out_dir=str(args.out),
    )
    result = train(config)

    print("\ntraining summary:")
    print(f"  train samples:     {result['train_samples']} (skipped {result['train_skipped']})")
    print(f"  val samples:       {result['val_samples']} (skipped {result['val_skipped']})")
    print(f"  majority baseline: {result['majority_baseline']:.4f}")
    print(f"  best val top-1:    {result['best_val_top1']:.4f}")
    print(f"  best val top-3:    {result['best_val_top3']:.4f}")
    print(f"  device:            {result['device']}")
    print(f"  elapsed:           {result['elapsed_seconds']:.1f}s")
    print(f"  checkpoint:        {result['checkpoint_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
