"""`BcTurnDataset`: one training sample per (turn-kind decision record, active slot)
from `data/bc/decisions.jsonl` (`vgc.replay_parse`'s schema-2+ output).

SCOPE reminder (see `vgc.bc` package docstring): this is pipeline-validation
infrastructure, not a playable agent's data loader.

Requires the `train` extra (torch) -- `vgc.bc.encoding` (this module's only intra-
package dependency) does not, so `import vgc.bc.encoding` keeps working without it;
importing THIS module without torch installed raises a clear `ImportError` instead of
whatever confusing error torch's own import machinery would otherwise produce.

## Sampling and the two-head (move, target) contract

Only `decision_kind == "turn"` records are used. Each qualifying record contributes UP
TO 2 samples (one per active slot). A sample is DROPPED ENTIRELY (counted in
`BcTurnDataset.skipped`) only when `encode_action` (the move head's label) returns
`None` -- an unrecognized move id (see that function's docstring). `encode_target` (the
target head's label) has its OWN, more lenient None case (e.g. a blocked
`|cant|`-attempted move's true target is genuinely unknown): when it returns `None`,
the sample is KEPT (the move head still has a valid label) but tagged with
`has_target=False`, so `vgc.bc.train`'s target loss term can mask that sample out
instead of training the target head on a fabricated label.

Each sample is an 8-tuple: `(index_array, scalar_array, move_idx, target_idx,
has_target, value_label, has_value, turn)` --
  - `index_array`: `(INDEX_DIM,)` int64 (`vgc.bc.encoding.flatten_state`'s index half).
  - `scalar_array`: `(STATE_SCALAR_DIM + SLOT_FEATURE_DIM,)` float32 (the scalar half
    plus a 2-dim one-hot for which of our 2 active slots is deciding).
  - `move_idx`: scalar int64, the move head's label (`MOVE_VOCAB` index).
  - `target_idx`: scalar int64, the target head's label (`TARGET_VOCAB` index) when
    `has_target` is true, else an arbitrary placeholder (0) that MUST be masked out by
    the caller rather than trained against.
  - `has_target`: scalar float32, 1.0/0.0.
  - `value_label`: scalar float32, `vgc.bc.encoding.encode_value`'s 1.0/0.0 ("did THIS
    record's player win the game") when `has_value` is true, else an arbitrary
    placeholder (0.0) that MUST be masked out -- same has_X-masking contract as target
    (a `None` from `encode_value` only happens on a pre-schema-3 record, so on any
    dataset built after this feature it's always 1.0, but the mask keeps mixed-schema
    data safe rather than silently training a fabricated label).
  - `turn`: scalar float32, the record's raw turn number (`record["turn"]`, NOT the
    `encode_state`-normalized `min(1.0, turn/20)` scalar that's already part of
    `scalar_array`) -- carried separately so `vgc.bc.train` can report value accuracy
    bucketed by turn (early positions are much harder to call than late ones; see that
    module's docstring for why this is the sanity check that the value head is learning
    positions rather than noise).

## Rating filter and train/val split

`min_rating` drops records from replays rated below it (or null-rated). The train/val
split is deterministic and keyed by `replay_id` (via a stable hash, NOT by shuffling
individual records), so every record from one replay always lands in the same split --
no replay ever straddles train/val.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

try:
    import numpy as np
    import torch
    from torch.utils.data import Dataset
except ImportError as exc:  # pragma: no cover - exercised via test_bc.py's importorskip
    raise ImportError(
        "vgc.bc.dataset requires the 'train' extra (torch) -- run `uv sync --extra train`."
    ) from exc

from vgc.bc.encoding import (
    SLOT_FEATURE_DIM,
    encode_action,
    encode_state,
    encode_target,
    encode_value,
    flatten_state,
)

DEFAULT_VAL_FRACTION = 0.1


def split_for_replay(replay_id: str, val_fraction: float = DEFAULT_VAL_FRACTION) -> str:
    """`"train"` or `"val"` for `replay_id`, deterministic via a stable hash (NOT
    Python's salted `hash()`, which varies per-process) -- the same replay always maps
    to the same split across runs/processes, which is what makes "no replay straddles
    splits" possible in the first place.
    """
    digest = hashlib.sha256(replay_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF  # deterministic float in [0, 1)
    return "val" if bucket < val_fraction else "train"


class BcTurnDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str | Path,
        min_rating: int = 1150,
        split: str = "train",
        val_fraction: float = DEFAULT_VAL_FRACTION,
    ) -> None:
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        self.jsonl_path = Path(jsonl_path)
        self.min_rating = min_rating
        self.split = split
        self.val_fraction = val_fraction

        # (state dict, slot, move_idx, target_idx_or_None, value_label_or_None, turn) --
        # state/value_label/turn are shared by both of a record's slot samples (no need
        # to duplicate the encoding), re-flattened lazily in __getitem__ (cheap: a
        # handful of small numpy concatenations/call).
        self._samples: list[tuple[dict, int, int, int | None, float | None, int]] = []
        self.skipped = 0
        self.no_target_count = 0
        self.no_value_count = 0
        self.replays_included: set[str] = set()

        with self.jsonl_path.open() as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("decision_kind") != "turn":
                    continue
                rating = record.get("rating")
                if rating is None or rating < self.min_rating:
                    continue
                replay_id = record.get("replay_id")
                if not replay_id or split_for_replay(replay_id, val_fraction) != split:
                    continue

                state = encode_state(record)
                value_label = encode_value(record)
                turn = int(record.get("turn") or 0)
                added_any = False
                for slot in (0, 1):
                    move_idx = encode_action(record, slot)
                    if move_idx is None:
                        self.skipped += 1
                        continue
                    target_idx = encode_target(record, slot)
                    if target_idx is None:
                        self.no_target_count += 1
                    if value_label is None:
                        self.no_value_count += 1
                    self._samples.append((state, slot, move_idx, target_idx, value_label, turn))
                    added_any = True
                if added_any:
                    self.replays_included.add(replay_id)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(
        self, index: int
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        state, slot, move_idx, target_idx, value_label, turn = self._samples[index]
        index_array, scalar_array = flatten_state(state)
        slot_onehot = np.zeros(SLOT_FEATURE_DIM, dtype=np.float32)
        slot_onehot[slot] = 1.0
        scalars_full = np.concatenate([scalar_array, slot_onehot])
        has_target = target_idx is not None
        has_value = value_label is not None
        return (
            torch.from_numpy(index_array.copy()),
            torch.from_numpy(scalars_full),
            torch.tensor(move_idx, dtype=torch.long),
            torch.tensor(target_idx if has_target else 0, dtype=torch.long),
            torch.tensor(1.0 if has_target else 0.0, dtype=torch.float32),
            torch.tensor(value_label if has_value else 0.0, dtype=torch.float32),
            torch.tensor(1.0 if has_value else 0.0, dtype=torch.float32),
            torch.tensor(float(turn), dtype=torch.float32),
        )
