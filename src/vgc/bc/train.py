"""Behavior-cloning training loop over `BcTurnDataset`/`BcPolicyNet` (v2: two heads --
move id AND target).

SCOPE (see `vgc.bc` package docstring, repeated here because it's the single most
important fact about this module): this validates the BC PIPELINE end to end (encoder
-> dataset -> training -> accuracy). It does NOT produce a playable agent --
`runs/bc/bc_policy.pt` predicts a human-plausible next MOVE ID and its TARGET CLASS from
a state snapshot, not a legal, targeting-resolved `DoubleBattleOrder`. Do not wire this
checkpoint into `vgc.agent.VgcPlayer.decide()` and call it a bot.

Requires the `train` extra (torch) -- see `vgc.bc.dataset`'s module docstring for why
this import is guarded the same way. `tools/train_bc.py` is the thin CLI over `train()`.

## Loss

`loss = move_loss + target_loss_weight * target_loss` (default weight 0.5, a
`TrainConfig`/CLI knob). `move_loss` is plain cross-entropy over every sample.
`target_loss` is cross-entropy over ONLY samples with `has_target == 1`
(`vgc.bc.dataset`'s mask -- a blocked `|cant|`-attempted move's true target is
genuinely unknown, so it must not be trained against a fabricated label) -- if a batch
has zero such samples, `target_loss` contributes 0 for that batch rather than raising a
division error.

## Early stopping and LR schedule

`--epochs` defaults to 30 (up from v1's 5, since v2's richer features need more passes
to converge) with early stopping: training stops once `patience` (default 5) epochs
pass without a new best val move top-1. LR schedule is COSINE annealing
(`torch.optim.lr_scheduler.CosineAnnealingLR`, `T_max = epochs`) -- chosen over a step
decay because it needs no extra "decay at which epoch" knob to tune and anneals
smoothly to ~0 by the final epoch regardless of how many epochs early stopping actually
lets it run.

Every epoch prints train loss, val move top-1/top-3, val target top-1, AND the
majority-class baseline for BOTH heads (move: always guessing the single most common
move id; target: always guessing the single most common target class among samples that
HAVE a real target) -- printed every epoch so "the model learned something" is never
just "the model learned to always guess the most common class" in disguise.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

try:
    import torch
    from torch import nn
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.data import DataLoader
except ImportError as exc:  # pragma: no cover - exercised via test_bc.py's importorskip
    raise ImportError(
        "vgc.bc.train requires the 'train' extra (torch) -- run `uv sync --extra train`."
    ) from exc

from vgc.bc.dataset import BcTurnDataset
from vgc.bc.encoding import (
    ABILITY_VOCAB,
    ENCODER_LAYOUT_VERSION,
    ITEM_VOCAB,
    MOVE_VOCAB,
    SPECIES_VOCAB,
    TARGET_VOCAB,
)
from vgc.bc.model import BcPolicyNet

DEFAULT_CHECKPOINT_NAME = "bc_policy.pt"


@dataclass
class TrainConfig:
    data: str
    epochs: int = 30
    batch_size: int = 512
    lr: float = 1e-3
    min_rating: int = 1150
    device: str = "auto"
    out_dir: str = "runs/bc"
    target_loss_weight: float = 0.5
    patience: int = 5
    extra_checkpoint_path: str | None = None


def resolve_device(requested: str) -> str:
    """`requested` verbatim unless it's `"auto"`, in which case: MPS (Apple Silicon) if
    available, else plain CPU. Never silently picks CUDA -- an explicit `--device cuda`
    is required for that, since this repo's development machine is a Mac.
    """
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def majority_class_accuracy(counts: Counter[int], total: int) -> float:
    """Baseline: always predicting the single most common class in `counts`/`total`."""
    if not counts or total == 0:
        return 0.0
    return counts.most_common(1)[0][1] / total


def _val_majority_baselines(loader: DataLoader) -> tuple[float, float]:
    """`(move_baseline, target_baseline)` -- the target baseline is computed only over
    samples with a real target (`has_target == 1`), matching how target accuracy/loss
    are computed everywhere else in this module.
    """
    move_counts: Counter[int] = Counter()
    target_counts: Counter[int] = Counter()
    move_total = 0
    target_total = 0
    for _index_array, _scalars, move_idx, target_idx, has_target in loader:
        move_counts.update(move_idx.tolist())
        move_total += move_idx.shape[0]
        mask = has_target.bool()
        if mask.any():
            target_counts.update(target_idx[mask].tolist())
            target_total += int(mask.sum().item())
    return (
        majority_class_accuracy(move_counts, move_total),
        majority_class_accuracy(target_counts, target_total),
    )


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> tuple[float, float, float]:
    """`(move_top1, move_top3, target_top1)` over `loader`. `target_top1` is computed
    only over samples with `has_target == 1`.
    """
    model.eval()
    move_correct_top1 = 0
    move_correct_top3 = 0
    move_total = 0
    target_correct = 0
    target_total = 0
    with torch.no_grad():
        for index_array, scalars, move_idx, target_idx, has_target in loader:
            index_array = index_array.to(device)
            scalars = scalars.to(device)
            move_idx = move_idx.to(device)
            target_idx = target_idx.to(device)
            has_target = has_target.to(device)

            move_logits, target_logits = model(index_array, scalars)

            move_top3 = move_logits.topk(min(3, move_logits.shape[-1]), dim=-1).indices
            move_correct_top1 += (move_top3[:, 0] == move_idx).sum().item()
            move_correct_top3 += (move_top3 == move_idx.unsqueeze(1)).any(dim=1).sum().item()
            move_total += move_idx.shape[0]

            mask = has_target.bool()
            if mask.any():
                target_pred = target_logits[mask].argmax(dim=-1)
                target_correct += (target_pred == target_idx[mask]).sum().item()
                target_total += int(mask.sum().item())

    move_top1 = move_correct_top1 / move_total if move_total else 0.0
    move_top3 = move_correct_top3 / move_total if move_total else 0.0
    target_top1 = target_correct / target_total if target_total else 0.0
    return move_top1, move_top3, target_top1


def train(config: TrainConfig) -> dict[str, object]:
    """Runs the full BC training loop (with early stopping) and returns a summary dict
    (also everything `tools/train_bc.py` prints) -- see module docstring for scope.
    """
    device = resolve_device(config.device)
    train_ds = BcTurnDataset(config.data, min_rating=config.min_rating, split="train")
    val_ds = BcTurnDataset(config.data, min_rating=config.min_rating, split="val")

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    model = BcPolicyNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    move_criterion = nn.CrossEntropyLoss()
    target_criterion = nn.CrossEntropyLoss(reduction="none")

    move_baseline, target_baseline = _val_majority_baselines(val_loader)

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / DEFAULT_CHECKPOINT_NAME

    best_val_move_top1 = -1.0
    best_val_move_top3 = 0.0
    best_val_target_top1 = 0.0
    best_epoch = 0
    epochs_since_best = 0
    history: list[dict[str, float]] = []
    start = time.time()
    epochs_run = 0

    def _checkpoint_payload() -> dict[str, object]:
        return {
            "state_dict": model.state_dict(),
            "species_vocab": SPECIES_VOCAB,
            "move_vocab": MOVE_VOCAB,
            "item_vocab": ITEM_VOCAB,
            "ability_vocab": ABILITY_VOCAB,
            "target_vocab": TARGET_VOCAB,
            "encoder_layout_version": ENCODER_LAYOUT_VERSION,
        }

    for epoch in range(1, config.epochs + 1):
        epochs_run = epoch
        model.train()
        total_loss = 0.0
        n_batches = 0
        for index_array, scalars, move_idx, target_idx, has_target in train_loader:
            index_array = index_array.to(device)
            scalars = scalars.to(device)
            move_idx = move_idx.to(device)
            target_idx = target_idx.to(device)
            has_target = has_target.to(device)

            optimizer.zero_grad()
            move_logits, target_logits = model(index_array, scalars)
            move_loss = move_criterion(move_logits, move_idx)

            per_sample_target_loss = target_criterion(target_logits, target_idx)
            mask_sum = has_target.sum()
            if mask_sum > 0:
                target_loss = (per_sample_target_loss * has_target).sum() / mask_sum
            else:
                target_loss = torch.zeros((), device=device)

            loss = move_loss + config.target_loss_weight * target_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
        scheduler.step()

        train_loss = total_loss / max(1, n_batches)
        val_move_top1, val_move_top3, val_target_top1 = evaluate(model, val_loader, device)
        print(
            f"epoch {epoch}/{config.epochs} train_loss={train_loss:.4f} "
            f"val_move_top1={val_move_top1:.4f} val_move_top3={val_move_top3:.4f} "
            f"val_target_top1={val_target_top1:.4f} "
            f"move_baseline={move_baseline:.4f} target_baseline={target_baseline:.4f} "
            f"lr={scheduler.get_last_lr()[0]:.6f}"
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_move_top1": val_move_top1,
                "val_move_top3": val_move_top3,
                "val_target_top1": val_target_top1,
            }
        )

        if val_move_top1 > best_val_move_top1:
            best_val_move_top1 = val_move_top1
            best_val_move_top3 = val_move_top3
            best_val_target_top1 = val_target_top1
            best_epoch = epoch
            epochs_since_best = 0
            torch.save(_checkpoint_payload(), checkpoint_path)
            if config.extra_checkpoint_path:
                extra_path = Path(config.extra_checkpoint_path)
                extra_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(_checkpoint_payload(), extra_path)
        else:
            epochs_since_best += 1
            if epochs_since_best >= config.patience:
                print(
                    f"early stopping: no val_move_top1 improvement in {config.patience} "
                    f"epochs (best was epoch {best_epoch})"
                )
                break

    elapsed_seconds = time.time() - start
    return {
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "train_skipped": train_ds.skipped,
        "val_skipped": val_ds.skipped,
        "move_baseline": move_baseline,
        "target_baseline": target_baseline,
        "best_val_move_top1": best_val_move_top1,
        "best_val_move_top3": best_val_move_top3,
        "best_val_target_top1": best_val_target_top1,
        "best_epoch": best_epoch,
        "epochs_run": epochs_run,
        "history": history,
        "elapsed_seconds": elapsed_seconds,
        "device": device,
        "checkpoint_path": str(checkpoint_path),
        "extra_checkpoint_path": config.extra_checkpoint_path,
    }
