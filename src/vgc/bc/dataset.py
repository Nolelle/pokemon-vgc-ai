"""`BcTurnDataset`: one training sample per (turn-kind decision record, active slot)
from `data/bc/decisions.jsonl` (`vgc.replay_parse`'s output).

SCOPE reminder (see `vgc.bc` package docstring): this is pipeline-validation
infrastructure, not a playable agent's data loader.

Requires the `train` extra (torch) -- `vgc.bc.encoding` (this module's only intra-
package dependency) does not, so `import vgc.bc.encoding` keeps working without it;
importing THIS module without torch installed raises a clear `ImportError` instead of
whatever confusing error torch's own import machinery would otherwise produce.

## Sampling

Only `decision_kind == "turn"` records are used (forced_switch/teampreview are a
different, not-yet-modeled decision shape in this v1 scope -- see
`vgc.bc.encoding.encode_action`'s docstring). Each qualifying record contributes UP TO 2
samples (one per active slot, `slot in (0, 1)`) -- a sample is skipped (and counted in
`BcTurnDataset.skipped`) when `encode_action` returns `None` for that slot (an
unrecognized move id; see that function's docstring for why that's `None` and not a
generic "unknown" class).

## Rating filter and train/val split

`min_rating` drops records from replays rated below it (or null-rated) -- same
semantics as `tools/download_replays.py`/`tools/build_set_priors.py`'s own rating
filters. The train/val split is deterministic and keyed by `replay_id` (via a stable
hash, NOT by shuffling individual records), so every record from one replay always
lands in the same split -- no replay ever straddles train/val, which would leak
within-game information (e.g. the same board state pattern) across the split boundary.
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

from vgc.bc.encoding import SLOT_FEATURE_DIM, encode_action, encode_state, flatten_state

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

        # (state dict, slot, action_idx) -- state is shared by both of a record's slot
        # samples (no need to duplicate the encoding), re-flattened lazily in
        # __getitem__ (cheap: a handful of small numpy concatenations per call).
        self._samples: list[tuple[dict, int, int]] = []
        self.skipped = 0
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
                added_any = False
                for slot in (0, 1):
                    action_idx = encode_action(record, slot)
                    if action_idx is None:
                        self.skipped += 1
                        continue
                    self._samples.append((state, slot, action_idx))
                    added_any = True
                if added_any:
                    self.replays_included.add(replay_id)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state, slot, action_idx = self._samples[index]
        scalars = flatten_state(state)
        slot_onehot = np.zeros(SLOT_FEATURE_DIM, dtype=np.float32)
        slot_onehot[slot] = 1.0
        scalars_full = np.concatenate([scalars, slot_onehot])
        return (
            torch.from_numpy(state["species_idx"].copy()),
            torch.from_numpy(scalars_full),
            torch.tensor(action_idx, dtype=torch.long),
        )
