"""Behavior-cloning training loop over `BcTurnDataset`/`BcPolicyNet`.

SCOPE (see `vgc.bc` package docstring, and repeated here because it's the single most
important fact about this module): this validates the BC PIPELINE end to end (encoder
-> dataset -> training -> accuracy). It does NOT produce a playable agent --
`runs/bc/bc_policy.pt` predicts a human-plausible next MOVE ID from a state snapshot,
not a legal, targeting-resolved `DoubleBattleOrder`. Do not wire this checkpoint into
`vgc.agent.VgcPlayer.decide()` and call it a bot.

Requires the `train` extra (torch) -- see `vgc.bc.dataset`'s module docstring for why
this import is guarded the same way. `tools/train_bc.py` is the thin CLI over `train()`.

Every epoch prints train loss, val top-1/top-3 accuracy, AND the majority-class
baseline accuracy (always predicting the single most common action in the val split) --
printed every epoch (not just once) so it's never lost in scrollback when judging
whether the model actually learned anything beyond "guess the most common move".
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
except ImportError as exc:  # pragma: no cover - exercised via test_bc.py's importorskip
    raise ImportError(
        "vgc.bc.train requires the 'train' extra (torch) -- run `uv sync --extra train`."
    ) from exc

from vgc.bc.dataset import BcTurnDataset
from vgc.bc.encoding import ENCODER_LAYOUT_VERSION, MOVE_VOCAB, SPECIES_VOCAB
from vgc.bc.model import BcPolicyNet

DEFAULT_CHECKPOINT_NAME = "bc_policy.pt"


@dataclass
class TrainConfig:
    data: str
    epochs: int = 5
    batch_size: int = 512
    lr: float = 1e-3
    min_rating: int = 1150
    device: str = "auto"
    out_dir: str = "runs/bc"


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


def majority_class_accuracy(loader: DataLoader) -> tuple[float, int | None]:
    """`(accuracy, majority_action_idx)` of always predicting the single most common
    action index seen in `loader` -- the honest baseline every epoch's real accuracy
    gets printed alongside, so "the model learned something" isn't just "the model
    learned to always guess Protect" in disguise.
    """
    counts: Counter[int] = Counter()
    total = 0
    for _species_idx, _scalars, actions in loader:
        counts.update(actions.tolist())
        total += actions.shape[0]
    if not counts or total == 0:
        return 0.0, None
    majority_idx, majority_count = counts.most_common(1)[0]
    return majority_count / total, majority_idx


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> tuple[float, float]:
    """`(top1_accuracy, top3_accuracy)` over `loader`."""
    model.eval()
    correct_top1 = 0
    correct_top3 = 0
    total = 0
    with torch.no_grad():
        for species_idx, scalars, actions in loader:
            species_idx = species_idx.to(device)
            scalars = scalars.to(device)
            actions = actions.to(device)
            logits = model(species_idx, scalars)
            top3 = logits.topk(min(3, logits.shape[-1]), dim=-1).indices
            correct_top1 += (top3[:, 0] == actions).sum().item()
            correct_top3 += (top3 == actions.unsqueeze(1)).any(dim=1).sum().item()
            total += actions.shape[0]
    if total == 0:
        return 0.0, 0.0
    return correct_top1 / total, correct_top3 / total


def train(config: TrainConfig) -> dict[str, object]:
    """Runs the full BC training loop and returns a summary dict (also everything
    `tools/train_bc.py` prints) -- see module docstring for the pipeline's scope.
    """
    device = resolve_device(config.device)
    train_ds = BcTurnDataset(config.data, min_rating=config.min_rating, split="train")
    val_ds = BcTurnDataset(config.data, min_rating=config.min_rating, split="val")

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    model = BcPolicyNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()

    majority_baseline, _majority_idx = majority_class_accuracy(val_loader)

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / DEFAULT_CHECKPOINT_NAME

    best_val_top1 = -1.0
    best_val_top3 = 0.0
    history: list[dict[str, float]] = []
    start = time.time()

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for species_idx, scalars, actions in train_loader:
            species_idx = species_idx.to(device)
            scalars = scalars.to(device)
            actions = actions.to(device)

            optimizer.zero_grad()
            logits = model(species_idx, scalars)
            loss = criterion(logits, actions)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        train_loss = total_loss / max(1, n_batches)
        val_top1, val_top3 = evaluate(model, val_loader, device)
        print(
            f"epoch {epoch}/{config.epochs} train_loss={train_loss:.4f} "
            f"val_top1={val_top1:.4f} val_top3={val_top3:.4f} "
            f"majority_baseline={majority_baseline:.4f}"
        )
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_top1": val_top1, "val_top3": val_top3}
        )

        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            best_val_top3 = val_top3
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "species_vocab": SPECIES_VOCAB,
                    "move_vocab": MOVE_VOCAB,
                    "encoder_layout_version": ENCODER_LAYOUT_VERSION,
                },
                checkpoint_path,
            )

    elapsed_seconds = time.time() - start
    return {
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "train_skipped": train_ds.skipped,
        "val_skipped": val_ds.skipped,
        "majority_baseline": majority_baseline,
        "best_val_top1": best_val_top1,
        "best_val_top3": best_val_top3,
        "history": history,
        "elapsed_seconds": elapsed_seconds,
        "device": device,
        "checkpoint_path": str(checkpoint_path),
    }
