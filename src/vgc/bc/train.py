"""Behavior-cloning training loop over `BcTurnDataset`/`BcPolicyNet` (v3: THREE heads --
move id, target, AND outcome value).

SCOPE (see `vgc.bc` package docstring, repeated here because it's the single most
important fact about this module): this validates the BC PIPELINE end to end (encoder
-> dataset -> training -> accuracy). It does NOT produce a playable agent --
`runs/bc/bc_policy.pt` predicts a human-plausible next MOVE ID and its TARGET CLASS from
a state snapshot, not a legal, targeting-resolved `DoubleBattleOrder`. Do not wire this
checkpoint into `vgc.agent.VgcPlayer.decide()` and call it a bot. The VALUE head is
different in kind from the other two: it's trained on GAME OUTCOMES (`vgc.replay_parse`
schema 3's `"won"` label), not on imitating a human's chosen action -- there's no
imitation ceiling on "who wins" the way there is on "what would an average-rated ladder
player click here" (BC's move/target heads can never exceed the skill of the players in
the corpus; the value head's ceiling is only how well the position predicts the outcome).

Requires the `train` extra (torch) -- see `vgc.bc.dataset`'s module docstring for why
this import is guarded the same way. `tools/train_bc.py` is the thin CLI over `train()`.

## Loss

`loss = move_loss + target_loss_weight * target_loss + value_loss_weight * value_loss`
(weights default 0.5/1.0, both `TrainConfig`/CLI knobs). `move_loss` is plain
cross-entropy over every sample. `target_loss` is cross-entropy over ONLY samples with
`has_target == 1` (`vgc.bc.dataset`'s mask -- a blocked `|cant|`-attempted move's true
target is genuinely unknown, so it must not be trained against a fabricated label).
`value_loss` is binary cross-entropy (`BCEWithLogitsLoss`, i.e. computed from the RAW
logit, not a post-sigmoid probability -- numerically stable, matches
`nn.CrossEntropyLoss`'s own "give me logits" contract for the other two heads) over
samples with `has_value == 1` -- masked the same way, for the (should be rare on a
schema-3 dataset) case of a pre-schema-3 record with no `"won"` label. If a batch has
zero unmasked samples for a given head, that head's loss contributes 0 for that batch
rather than raising a division error -- same pattern for target and value.

Multi-task training (all three heads sharing one trunk, trained jointly) rather than a
separate value-only model: at this corpus's data scale (order 10^4-10^5 samples), a
shared trunk lets the move/target heads' much larger supervision signal (every sample
has a real move label; only every sample-per-GAME has one outcome label, so the value
head's effective sample diversity per unique state is lower) regularize the trunk's
state representation, which should help the value head generalize better than training
it alone on the same trunk-sized network would.

## Mixing in self-play data (`TrainConfig.extra_data`/`selfplay_weight`)

`extra_data` (default `None`, no behavior change) points at a second JSONL file --
typically `data/selfplay/records.jsonl` (`vgc.bc.selfplay.RecordingVgcPlayer`'s output,
see `selfplay/run_selfplay.py`) -- loaded via a SECOND `BcTurnDataset` with
`allow_null_rating=True` (self-play games have no ladder Elo at all, see that flag's
docstring) and `val_fraction=0.0`/`split="train"` so 100% of it lands in TRAIN, never
val: this module always validates on the ORIGINAL corpus's val split ONLY, regardless of
`extra_data` -- mixing self-play into the reported accuracy numbers would silently
answer a different, less meaningful question ("does the net fit self-play-flavored
positions") than the one that actually matters here ("did adding self-play data improve
the net's judgment on REAL human ladder positions").

The two datasets are concatenated (`torch.utils.data.ConcatDataset`) and drawn from via
a `WeightedRandomSampler` instead of `shuffle=True` -- every corpus sample gets weight
1.0, every self-play sample gets `selfplay_weight` (default 1.0, i.e. sampled at the same
rate as a corpus sample; raise it to over-sample self-play, lower it to down-weight a
much larger but possibly lower-quality self-play set relative to the corpus). The
sampler still draws `len(combined)` samples per epoch (with replacement), so a
default-weighted mix behaves like plain concatenated shuffling, not a different epoch
size. `extra_data=None` (the default) leaves everything byte-for-byte identical to the
pre-self-play training path -- no `ConcatDataset`/sampler machinery is even constructed.

## Value-head metrics (every epoch)

Beyond move/target top-1/top-3 and their majority-class baselines (unchanged from v2):
`val_value_accuracy` (threshold 0.5 on `sigmoid(value_logit)`), `val_value_auc`
(Mann-Whitney U / rank-sum AUC -- see `compute_auc`, no sklearn dependency), and the
value label's own majority-class baseline (`max(label_rate, 1 - label_rate)` -- "always
guess the more common class", the value-head equivalent of `move_baseline`/
`target_baseline`; label balance is close to 50/50 by construction -- see
`vgc.replay_parse`'s schema 3 docstring -- so this baseline is close to 0.5, meaning
`val_value_accuracy` needs to clear ~0.5 by a real margin to mean anything). Also
reported: `val_value_accuracy_by_turn_bucket` for turn ranges 1-3/4-6/7+ -- the sanity
check that the value head is learning actual POSITIONS rather than a constant-ish
prediction: early-game positions (turn 1-3) are close to a coin flip even for a perfect
model (game hasn't developed enough to be very predictable), so accuracy there should be
noticeably lower than turn 7+, where the position has usually mostly resolved. A value
head that DOESN'T show this gradient (flat accuracy across buckets, or worse in
late-game) is a red flag that it learned something spurious rather than positional
judgment.
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
    from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler
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
from vgc.bc.model import DEFAULT_HEADS, BcPolicyNet

DEFAULT_CHECKPOINT_NAME = "bc_policy.pt"
# Turn-bucket boundaries for val_value_accuracy_by_turn_bucket -- see module docstring.
TURN_BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("1-3", 1, 3),
    ("4-6", 4, 6),
    ("7+", 7, 10_000),
)


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
    value_loss_weight: float = 1.0
    patience: int = 5
    extra_checkpoint_path: str | None = None
    heads: tuple[str, ...] = DEFAULT_HEADS
    # A second JSONL path (e.g. data/selfplay/records.jsonl) MIXED INTO TRAINING ONLY --
    # not related to extra_checkpoint_path above (which just saves an additional copy of
    # the checkpoint). None (default) leaves training byte-for-byte unchanged from before
    # this knob existed. See module docstring's "Mixing in self-play data" section.
    extra_data: str | None = None
    # Relative sampling weight for each `extra_data` sample vs each main-corpus sample
    # (1.0 each by default -- see module docstring).
    selfplay_weight: float = 1.0


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


def compute_auc(scores: list[float], labels: list[int]) -> float:
    """AUC via the Mann-Whitney U statistic (rank-sum formulation) -- no sklearn
    dependency. `scores`/`labels` must be the same length; `labels` are 0/1. Ties in
    `scores` get the average rank across the tied block (standard treatment -- without
    it, an all-identical-score model would get a spuriously perfect or terrible AUC
    depending on sort stability). Returns 0.5 (chance) if `labels` is empty or all one
    class -- AUC is undefined in that case, and 0.5 is the least-misleading stand-in
    (matches what a random classifier would score, rather than 0.0 or 1.0 implying a
    real measurement).
    """
    n = len(scores)
    if n == 0:
        return 0.5
    pairs = sorted(zip(scores, labels), key=lambda pair: pair[0])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-indexed average rank across the tied block
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    n_pos = sum(1 for _, label in pairs if label == 1)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    rank_sum_pos = sum(rank for rank, (_, label) in zip(ranks, pairs) if label == 1)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _val_majority_baselines(loader: DataLoader) -> tuple[float, float, float]:
    """`(move_baseline, target_baseline, value_baseline)` -- target/value baselines are
    computed only over samples with a real label (`has_target`/`has_value == 1`),
    matching how their accuracy/loss are computed everywhere else in this module.
    """
    move_counts: Counter[int] = Counter()
    target_counts: Counter[int] = Counter()
    move_total = 0
    target_total = 0
    value_positive = 0
    value_total = 0
    for batch in loader:
        _index_array, _scalars, move_idx, target_idx, has_target, value_label, has_value, _turn = (
            batch
        )
        move_counts.update(move_idx.tolist())
        move_total += move_idx.shape[0]
        target_mask = has_target.bool()
        if target_mask.any():
            target_counts.update(target_idx[target_mask].tolist())
            target_total += int(target_mask.sum().item())
        value_mask = has_value.bool()
        if value_mask.any():
            value_positive += int(value_label[value_mask].sum().item())
            value_total += int(value_mask.sum().item())
    value_baseline = (
        max(value_positive, value_total - value_positive) / value_total if value_total else 0.0
    )
    return (
        majority_class_accuracy(move_counts, move_total),
        majority_class_accuracy(target_counts, target_total),
        value_baseline,
    )


def _value_label_balance(loader: DataLoader) -> tuple[int, int]:
    """`(positive_count, total_count)` over samples with `has_value == 1` -- the actual
    label balance to report alongside the baseline (see module docstring)."""
    positive = 0
    total = 0
    for batch in loader:
        has_value, value_label = batch[6], batch[5]
        mask = has_value.bool()
        if mask.any():
            positive += int(value_label[mask].sum().item())
            total += int(mask.sum().item())
    return positive, total


def evaluate(
    model: nn.Module, loader: DataLoader, device: str
) -> dict[str, float | dict[str, float]]:
    """Move/target/value metrics over `loader`. `target_top1` and the value metrics are
    computed only over samples with `has_target`/`has_value == 1` respectively.
    """
    model.eval()
    move_correct_top1 = 0
    move_correct_top3 = 0
    move_total = 0
    target_correct = 0
    target_total = 0
    value_scores: list[float] = []
    value_labels: list[int] = []
    value_turns: list[int] = []
    with torch.no_grad():
        for batch in loader:
            index_array, scalars, move_idx, target_idx, has_target, value_label, has_value, turn = (
                batch
            )
            index_array = index_array.to(device)
            scalars = scalars.to(device)
            move_idx = move_idx.to(device)
            target_idx = target_idx.to(device)
            has_target = has_target.to(device)

            move_logits, target_logits, value_logit = model(index_array, scalars)

            if move_logits is not None:
                move_top3 = move_logits.topk(min(3, move_logits.shape[-1]), dim=-1).indices
                move_correct_top1 += (move_top3[:, 0] == move_idx).sum().item()
                move_correct_top3 += (move_top3 == move_idx.unsqueeze(1)).any(dim=1).sum().item()
                move_total += move_idx.shape[0]

            if target_logits is not None:
                mask = has_target.bool()
                if mask.any():
                    target_pred = target_logits[mask].argmax(dim=-1)
                    target_correct += (target_pred == target_idx[mask]).sum().item()
                    target_total += int(mask.sum().item())

            if value_logit is not None:
                value_mask = has_value.bool()
                if value_mask.any():
                    probs = torch.sigmoid(value_logit[value_mask]).cpu().tolist()
                    labels = value_label[value_mask].cpu().tolist()
                    turns = turn[value_mask].cpu().tolist()
                    value_scores.extend(probs)
                    value_labels.extend(int(label) for label in labels)
                    value_turns.extend(int(t) for t in turns)

    move_top1 = move_correct_top1 / move_total if move_total else 0.0
    move_top3 = move_correct_top3 / move_total if move_total else 0.0
    target_top1 = target_correct / target_total if target_total else 0.0

    value_accuracy = 0.0
    if value_scores:
        value_correct = sum(
            1 for score, label in zip(value_scores, value_labels) if (score >= 0.5) == bool(label)
        )
        value_accuracy = value_correct / len(value_scores)
    value_auc = compute_auc(value_scores, value_labels)

    by_bucket: dict[str, float] = {}
    for name, low, high in TURN_BUCKETS:
        bucket_scores = []
        bucket_labels = []
        for score, label, t in zip(value_scores, value_labels, value_turns):
            if low <= t <= high:
                bucket_scores.append(score)
                bucket_labels.append(label)
        if bucket_scores:
            correct = sum(
                1
                for score, label in zip(bucket_scores, bucket_labels)
                if (score >= 0.5) == bool(label)
            )
            by_bucket[name] = correct / len(bucket_scores)
        else:
            by_bucket[name] = 0.0

    return {
        "move_top1": move_top1,
        "move_top3": move_top3,
        "target_top1": target_top1,
        "value_accuracy": value_accuracy,
        "value_auc": value_auc,
        "value_accuracy_by_turn_bucket": by_bucket,
    }


def train(config: TrainConfig) -> dict[str, object]:
    """Runs the full BC training loop (with early stopping) and returns a summary dict
    (also everything `tools/train_bc.py` prints) -- see module docstring for scope.
    """
    device = resolve_device(config.device)
    train_ds = BcTurnDataset(config.data, min_rating=config.min_rating, split="train")
    # Validation ALWAYS comes only from the main corpus's val split -- see module
    # docstring's "Mixing in self-play data" section for why extra_data never touches
    # this, regardless of whether it's set below.
    val_ds = BcTurnDataset(config.data, min_rating=config.min_rating, split="val")

    extra_ds: BcTurnDataset | None = None
    if config.extra_data:
        # min_rating is irrelevant here (allow_null_rating=True bypasses the rating
        # filter entirely for null-rated records -- see that flag's docstring), and
        # val_fraction=0.0 with split="train" means 100% of this data lands in TRAIN,
        # never val, regardless of vgc.bc.dataset.split_for_replay's hash.
        extra_ds = BcTurnDataset(
            config.extra_data,
            min_rating=config.min_rating,
            split="train",
            val_fraction=0.0,
            allow_null_rating=True,
        )

    if extra_ds is not None and len(extra_ds) > 0:
        combined_train_ds = ConcatDataset([train_ds, extra_ds])
        sample_weights = [1.0] * len(train_ds) + [config.selfplay_weight] * len(extra_ds)
        sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(combined_train_ds), replacement=True
        )
        train_loader = DataLoader(combined_train_ds, batch_size=config.batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    heads = tuple(config.heads)
    model = BcPolicyNet(heads=heads).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    move_criterion = nn.CrossEntropyLoss()
    target_criterion = nn.CrossEntropyLoss(reduction="none")
    value_criterion = nn.BCEWithLogitsLoss(reduction="none")

    move_baseline, target_baseline, value_baseline = _val_majority_baselines(val_loader)
    value_positive, value_total = _value_label_balance(val_loader)
    value_label_rate = value_positive / value_total if value_total else 0.0

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / DEFAULT_CHECKPOINT_NAME

    best_val_move_top1 = -1.0
    best_metrics: dict[str, float | dict[str, float]] = {}
    best_epoch = 0
    epochs_since_best = 0
    history: list[dict[str, object]] = []
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
            "heads": heads,
        }

    for epoch in range(1, config.epochs + 1):
        epochs_run = epoch
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            (
                index_array,
                scalars,
                move_idx,
                target_idx,
                has_target,
                value_label,
                has_value,
                _turn,
            ) = batch
            index_array = index_array.to(device)
            scalars = scalars.to(device)
            move_idx = move_idx.to(device)
            target_idx = target_idx.to(device)
            has_target = has_target.to(device)
            value_label = value_label.to(device)
            has_value = has_value.to(device)

            optimizer.zero_grad()
            move_logits, target_logits, value_logit = model(index_array, scalars)

            loss = torch.zeros((), device=device)
            if move_logits is not None:
                loss = loss + move_criterion(move_logits, move_idx)

            if target_logits is not None:
                per_sample_target_loss = target_criterion(target_logits, target_idx)
                mask_sum = has_target.sum()
                if mask_sum > 0:
                    target_loss = (per_sample_target_loss * has_target).sum() / mask_sum
                    loss = loss + config.target_loss_weight * target_loss

            if value_logit is not None:
                per_sample_value_loss = value_criterion(value_logit, value_label)
                mask_sum = has_value.sum()
                if mask_sum > 0:
                    value_loss = (per_sample_value_loss * has_value).sum() / mask_sum
                    loss = loss + config.value_loss_weight * value_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
        scheduler.step()

        train_loss = total_loss / max(1, n_batches)
        metrics = evaluate(model, val_loader, device)
        bucket_str = " ".join(
            f"{name}={metrics['value_accuracy_by_turn_bucket'][name]:.3f}"
            for name, _low, _high in TURN_BUCKETS
        )
        print(
            f"epoch {epoch}/{config.epochs} train_loss={train_loss:.4f} "
            f"val_move_top1={metrics['move_top1']:.4f} val_move_top3={metrics['move_top3']:.4f} "
            f"val_target_top1={metrics['target_top1']:.4f} "
            f"val_value_acc={metrics['value_accuracy']:.4f} val_value_auc={metrics['value_auc']:.4f} "
            f"move_baseline={move_baseline:.4f} target_baseline={target_baseline:.4f} "
            f"value_baseline={value_baseline:.4f} value_by_turn=[{bucket_str}] "
            f"lr={scheduler.get_last_lr()[0]:.6f}"
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_move_top1": metrics["move_top1"],
                "val_move_top3": metrics["move_top3"],
                "val_target_top1": metrics["target_top1"],
                "val_value_accuracy": metrics["value_accuracy"],
                "val_value_auc": metrics["value_auc"],
                "val_value_accuracy_by_turn_bucket": metrics["value_accuracy_by_turn_bucket"],
            }
        )

        if metrics["move_top1"] > best_val_move_top1:
            best_val_move_top1 = metrics["move_top1"]
            best_metrics = metrics
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
    extra_train_samples = len(extra_ds) if extra_ds is not None else 0
    return {
        # Corpus-only train count (unchanged meaning from before extra_data existed) --
        # "train_samples_total" below is what actually fed the DataLoader when
        # extra_data was mixed in.
        "train_samples": len(train_ds),
        "extra_train_samples": extra_train_samples,
        "train_samples_total": len(train_ds) + extra_train_samples,
        "val_samples": len(val_ds),
        "train_skipped": train_ds.skipped,
        "val_skipped": val_ds.skipped,
        "train_no_value_count": train_ds.no_value_count,
        "val_no_value_count": val_ds.no_value_count,
        "move_baseline": move_baseline,
        "target_baseline": target_baseline,
        "value_baseline": value_baseline,
        "value_label_rate": value_label_rate,
        "value_positive": value_positive,
        "value_total": value_total,
        "best_val_move_top1": best_metrics.get("move_top1", best_val_move_top1),
        "best_val_move_top3": best_metrics.get("move_top3", 0.0),
        "best_val_target_top1": best_metrics.get("target_top1", 0.0),
        "best_val_value_accuracy": best_metrics.get("value_accuracy", 0.0),
        "best_val_value_auc": best_metrics.get("value_auc", 0.5),
        "best_val_value_accuracy_by_turn_bucket": best_metrics.get(
            "value_accuracy_by_turn_bucket", {}
        ),
        "best_epoch": best_epoch,
        "epochs_run": epochs_run,
        "history": history,
        "elapsed_seconds": elapsed_seconds,
        "device": device,
        "checkpoint_path": str(checkpoint_path),
        "extra_checkpoint_path": config.extra_checkpoint_path,
        "heads": heads,
    }
