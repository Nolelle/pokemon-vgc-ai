"""`BcPolicyNet`: a plain `torch.nn` species-embedding + MLP behavior-cloning policy
network (no Lightning/other training-framework dependency).

SCOPE reminder (see `vgc.bc` package docstring): this validates that a BC pipeline can
learn a human-plausible next-move-id distribution -- it is NOT a playable agent's
policy network (no legal-targeting/switching model, no battle-order construction).

Requires the `train` extra (torch) -- see `vgc.bc.dataset`'s module docstring for why
this import is guarded the same way.

Input (see `vgc.bc.encoding`'s module docstring for the exact source layout):
  - `species_idx`: `(batch, 4)` int64 -- `SPECIES_VOCAB` index per active slot
    (`[our0, our1, opp0, opp1]`), embedded via `nn.Embedding` and flattened.
  - `scalars`: `(batch, STATE_SCALAR_DIM + SLOT_FEATURE_DIM)` float32 -- `vgc.bc.
    encoding.flatten_state`'s output concatenated with `vgc.bc.dataset`'s 2-dim
    slot-index one-hot.
Output: `(batch, len(MOVE_VOCAB))` logits over the move vocabulary.
"""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - exercised via test_bc.py's importorskip
    raise ImportError(
        "vgc.bc.model requires the 'train' extra (torch) -- run `uv sync --extra train`."
    ) from exc

from vgc.bc.encoding import MOVE_VOCAB, SLOT_FEATURE_DIM, SPECIES_VOCAB, STATE_SCALAR_DIM

SPECIES_EMBED_DIM = 32
NUM_ACTIVE_SLOTS = 4  # [our0, our1, opp0, opp1] -- see vgc.bc.encoding's layout table
HIDDEN_DIM = 256
DROPOUT_P = 0.1
DEFAULT_SCALAR_DIM = STATE_SCALAR_DIM + SLOT_FEATURE_DIM


class BcPolicyNet(nn.Module):
    def __init__(
        self,
        species_vocab_size: int = len(SPECIES_VOCAB),
        move_vocab_size: int = len(MOVE_VOCAB),
        scalar_dim: int = DEFAULT_SCALAR_DIM,
        embed_dim: int = SPECIES_EMBED_DIM,
        hidden_dim: int = HIDDEN_DIM,
        dropout: float = DROPOUT_P,
    ) -> None:
        super().__init__()
        self.species_embedding = nn.Embedding(species_vocab_size, embed_dim)
        input_dim = NUM_ACTIVE_SLOTS * embed_dim + scalar_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, move_vocab_size),
        )

    def forward(self, species_idx: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        """`species_idx`: `(batch, 4)` int64. `scalars`: `(batch, scalar_dim)` float32.
        Returns `(batch, move_vocab_size)` logits.
        """
        embedded = self.species_embedding(species_idx)  # (batch, 4, embed_dim)
        embedded_flat = embedded.reshape(embedded.shape[0], -1)  # (batch, 4*embed_dim)
        features = torch.cat([embedded_flat, scalars], dim=-1)
        return self.net(features)
