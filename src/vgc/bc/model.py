"""`BcPolicyNet`: a plain `torch.nn` species/item/ability/move-embedding + MLP
behavior-cloning policy network with up to THREE heads sharing one trunk (no
Lightning/other training-framework dependency).

SCOPE reminder (see `vgc.bc` package docstring): this validates that a BC pipeline can
learn a human-plausible next-move-id (and target) distribution -- it is NOT a playable
agent's policy network (no legal-targeting/switching model, no battle-order
construction).

Requires the `train` extra (torch) -- see `vgc.bc.dataset`'s module docstring for why
this import is guarded the same way.

## Input (see `vgc.bc.encoding`'s module docstring for the exact source layout)

- `index_array`: `(batch, INDEX_DIM)` int64 -- sliced (at `vgc.bc.encoding`'s documented
  `INDEX_*_SLICE` offsets) into 6 embedding-lookup groups:
  - active-slot species (4), bench species (8), AND (v4, NEW) each side's up-to-6
    previewed species (12: our0..5, opp0..5) share ONE embedding table
    (`species_embedding`) -- the same species means the same thing whether it's
    currently active, on the bench, or only known from Team Preview, so there's no
    reason to learn separate representations for it. The previewed-species group is
    MEAN-POOLED per SIDE (2 vectors: our-preview-mean, opp-preview-mean), `"<pad>"`
    entries masked out of the mean the same way revealed moves already are below --
    this is the network's only view of a side's full 6-mon roster (including the 2
    that were previewed but never brought, and any brought mon that hasn't appeared
    yet), letting it infer archetype context (e.g. "these two previewed species
    together usually mean a rain team") purely from the shared embedding, with no
    hand-engineered archetype features.
  - active-slot item (4) -> `item_embedding`.
  - active-slot ability (4) -> `ability_embedding`.
  - active-slot revealed moves (4 slots x 4 moves = 16) -> `move_embedding`, then
    MEAN-POOLED down to one vector per active slot (4 vectors), with `"<pad>"` entries
    masked OUT of the mean (a slot with 0 revealed moves -- all 4 padding -- gets an
    all-zero vector, not garbage from averaging in the pad embedding).
- `scalars`: `(batch, STATE_SCALAR_DIM + SLOT_FEATURE_DIM)` float32 -- `vgc.bc.
  encoding.flatten_state`'s scalar half (v4, NEW: includes each side's
  `alive_known_count`/`alive_known_mean_hp`/`preview_unseen_count` resource-state
  summary, see that module's docstring) concatenated with `vgc.bc.dataset`'s 2-dim
  slot-index one-hot.

## Output

`forward` always returns a 3-tuple `(move_logits, target_logits, value_logit)`; a head
that isn't in `heads` (constructor arg, default `("move", "target", "value")`) returns
`None` in its slot instead of a tensor, and its `nn.Linear` is never constructed at all
(so a value-only checkpoint carries no move/target parameters, and vice versa) -- ONE
model class covers move+target (the original v2 policy), move+target+value (Phase 3's
outcome-value net trained jointly), or value-only, entirely via which strings are in
`heads`. Shapes when present: `move_logits` `(batch, len(MOVE_VOCAB))`, `target_logits`
`(batch, len(TARGET_VOCAB))`, `value_logit` `(batch,)` (a single raw logit per sample --
callers apply `sigmoid` themselves for `P(this record's player wins)`, matching how
`move_logits`/`target_logits` are raw too and `vgc.bc.train`/`vgc.bc.policy` do their own
softmax/log-softmax). See `vgc.bc.train` for how the (up to three) losses are combined --
move loss always applies when the move head exists; target/value loss are masked
per-sample by `vgc.bc.dataset`'s `has_target`/`has_value` flags.

Every checkpoint saves its own `"heads"` tuple (`vgc.bc.train._checkpoint_payload`)
alongside the vocab lists, so `vgc.bc.policy.load_bc_policy` rebuilds the exact same head
configuration a checkpoint was trained with -- a checkpoint saved before this feature
(no `"heads"` key) is treated as `("move", "target")`, matching what it actually is.

## `legacy_v2_layout` (compatibility with pre-v4 checkpoints)

A checkpoint saved under `vgc.bc.encoding`'s pre-v4 layout (`"bc-encoding-v2"` --
e.g. `bc_policy_v3.pt`/`bc_policy_v3sp.pt`) has NO preview-species embedding lookup or
its mean-pooling in its trained weights at all -- the trunk's `nn.Linear` input width was
computed without that term. `legacy_v2_layout=True` (default `False`, no behavior change
for current/v4 checkpoints) makes `__init__` skip the preview-species contribution to
`input_dim` entirely and makes `forward` read `index_array` at the OLDER, narrower
`INDEX_*_SLICE_V2` offsets (`vgc.bc.encoding`) instead of the current ones -- the two
layouts differ ONLY in whether the preview-species group exists at all, so this is the
one flag needed to reproduce the exact pre-v4 architecture byte-for-byte.
`vgc.bc.policy.load_bc_policy` sets this from the checkpoint's own saved
`encoder_layout_version`, never from the running module's current version.
"""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - exercised via test_bc.py's importorskip
    raise ImportError(
        "vgc.bc.model requires the 'train' extra (torch) -- run `uv sync --extra train`."
    ) from exc

from vgc.bc.encoding import (
    ABILITY_VOCAB,
    INDEX_ABILITY_SLICE,
    INDEX_ABILITY_SLICE_V2,
    INDEX_ITEM_SLICE,
    INDEX_ITEM_SLICE_V2,
    INDEX_MOVES_SLICE,
    INDEX_MOVES_SLICE_V2,
    INDEX_SPECIES_ACTIVE_SLICE,
    INDEX_SPECIES_ACTIVE_SLICE_V2,
    INDEX_SPECIES_BENCH_SLICE,
    INDEX_SPECIES_BENCH_SLICE_V2,
    INDEX_SPECIES_PREVIEW_SLICE,
    ITEM_VOCAB,
    MOVE_TO_IDX,
    MOVE_VOCAB,
    SLOT_FEATURE_DIM,
    SPECIES_TO_IDX,
    SPECIES_VOCAB,
    STATE_SCALAR_DIM,
    TARGET_VOCAB,
)

SPECIES_EMBED_DIM = 32
ITEM_EMBED_DIM = 12
ABILITY_EMBED_DIM = 12
MOVE_EMBED_DIM = 16
NUM_ACTIVE_SLOTS = 4  # [our0, our1, opp0, opp1]
NUM_BENCH_SLOTS = 8  # [our_bench0..3, opp_bench0..3]
NUM_PREVIEW_SIDES = 2  # [our, opp] -- each side's up-to-6 previewed species mean-pooled
NUM_PREVIEW_SLOTS_PER_SIDE = 6
MOVES_PER_SLOT = 4
HIDDEN_DIM = 256
DROPOUT_P = 0.1
DEFAULT_SCALAR_DIM = STATE_SCALAR_DIM + SLOT_FEATURE_DIM
_MOVE_PAD_IDX = MOVE_TO_IDX["<pad>"]
_SPECIES_PAD_IDX = SPECIES_TO_IDX["<pad>"]

# Every head this architecture knows how to build -- `heads` args elsewhere are always a
# subset of this tuple, in this order (mirrors forward()'s return order).
ALL_HEADS: tuple[str, ...] = ("move", "target", "value")
DEFAULT_HEADS: tuple[str, ...] = ALL_HEADS


class BcPolicyNet(nn.Module):
    def __init__(
        self,
        species_vocab_size: int = len(SPECIES_VOCAB),
        item_vocab_size: int = len(ITEM_VOCAB),
        ability_vocab_size: int = len(ABILITY_VOCAB),
        move_vocab_size: int = len(MOVE_VOCAB),
        target_vocab_size: int = len(TARGET_VOCAB),
        scalar_dim: int = DEFAULT_SCALAR_DIM,
        species_embed_dim: int = SPECIES_EMBED_DIM,
        item_embed_dim: int = ITEM_EMBED_DIM,
        ability_embed_dim: int = ABILITY_EMBED_DIM,
        move_embed_dim: int = MOVE_EMBED_DIM,
        hidden_dim: int = HIDDEN_DIM,
        dropout: float = DROPOUT_P,
        heads: tuple[str, ...] = DEFAULT_HEADS,
        legacy_v2_layout: bool = False,
    ) -> None:
        super().__init__()
        self.heads: tuple[str, ...] = tuple(heads)
        # See module docstring's "legacy_v2_layout" section: a pre-v4 checkpoint's
        # trunk never had a preview-species term, and its index_array is the narrower
        # `bc-encoding-v2` layout -- `scalar_dim` itself is NOT auto-derived from this
        # flag (same as every other dim here, the caller -- vgc.bc.policy.load_bc_policy
        # -- must pass the value matching whichever layout this checkpoint actually is).
        self.legacy_v2_layout = legacy_v2_layout
        # Shared across active, bench, AND (v4-only) previewed species -- see module
        # docstring.
        self.species_embedding = nn.Embedding(species_vocab_size, species_embed_dim)
        self.item_embedding = nn.Embedding(item_vocab_size, item_embed_dim)
        self.ability_embedding = nn.Embedding(ability_vocab_size, ability_embed_dim)
        self.move_embedding = nn.Embedding(move_vocab_size, move_embed_dim)

        input_dim = (
            (NUM_ACTIVE_SLOTS + NUM_BENCH_SLOTS) * species_embed_dim
            + (0 if legacy_v2_layout else NUM_PREVIEW_SIDES * species_embed_dim)
            + NUM_ACTIVE_SLOTS * item_embed_dim
            + NUM_ACTIVE_SLOTS * ability_embed_dim
            + NUM_ACTIVE_SLOTS * move_embed_dim  # mean-pooled per slot, then concatenated
            + scalar_dim
        )
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # A head not in `self.heads` is `None` -- never constructed, never trained,
        # never appears in state_dict() -- rather than built-but-unused, so a value-only
        # checkpoint genuinely carries no move/target parameters (and vice versa).
        self.move_head = nn.Linear(hidden_dim, move_vocab_size) if "move" in self.heads else None
        self.target_head = (
            nn.Linear(hidden_dim, target_vocab_size) if "target" in self.heads else None
        )
        # Single logit -- sigmoid(value_logit) = P(this record's player wins). No hidden
        # layer of its own (unlike move/target, "who's winning" is a much lower-capacity
        # question than "which of ~400 moves" -- a linear readout off the shared trunk is
        # plenty, and keeps the head cheap to add without changing the trunk's own size).
        self.value_head = nn.Linear(hidden_dim, 1) if "value" in self.heads else None

    def encode_hidden(self, index_array: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        """Return the shared state representation before any prediction head.

        Keeping this public lets candidate-aware policies reuse a BC-pretrained state
        encoder without pretending the old independent move heads are a joint-action
        policy. Existing BC callers still go through :meth:`forward` unchanged.
        """
        batch_size = index_array.shape[0]

        if self.legacy_v2_layout:
            active_slice = INDEX_SPECIES_ACTIVE_SLICE_V2
            bench_slice = INDEX_SPECIES_BENCH_SLICE_V2
            item_slice = INDEX_ITEM_SLICE_V2
            ability_slice = INDEX_ABILITY_SLICE_V2
            moves_slice = INDEX_MOVES_SLICE_V2
        else:
            active_slice = INDEX_SPECIES_ACTIVE_SLICE
            bench_slice = INDEX_SPECIES_BENCH_SLICE
            item_slice = INDEX_ITEM_SLICE
            ability_slice = INDEX_ABILITY_SLICE
            moves_slice = INDEX_MOVES_SLICE

        species_active = index_array[:, active_slice]
        species_bench = index_array[:, bench_slice]
        item_idx = index_array[:, item_slice]
        ability_idx = index_array[:, ability_slice]
        move_idx = index_array[:, moves_slice].reshape(batch_size, NUM_ACTIVE_SLOTS, MOVES_PER_SLOT)

        species_active_emb = self.species_embedding(species_active).reshape(batch_size, -1)
        species_bench_emb = self.species_embedding(species_bench).reshape(batch_size, -1)
        item_emb = self.item_embedding(item_idx).reshape(batch_size, -1)
        ability_emb = self.ability_embedding(ability_idx).reshape(batch_size, -1)

        feature_parts = [species_active_emb, species_bench_emb]
        if not self.legacy_v2_layout:
            # v4 only: mean-pooled per SIDE (our-preview-mean, opp-preview-mean),
            # "<pad>" masked out of the mean -- mirrors the revealed-moves pooling
            # below, just pooled over a side's 6 preview slots instead of a slot's 4
            # revealed moves. A legacy_v2_layout checkpoint's trunk has no parameters
            # for this term at all, so it must not be computed/concatenated here.
            species_preview = index_array[:, INDEX_SPECIES_PREVIEW_SLICE].reshape(
                batch_size, NUM_PREVIEW_SIDES, NUM_PREVIEW_SLOTS_PER_SIDE
            )
            preview_emb = self.species_embedding(species_preview)  # (batch,2,6,species_embed_dim)
            preview_mask = (species_preview != _SPECIES_PAD_IDX).unsqueeze(-1).to(preview_emb.dtype)
            preview_emb_sum = (preview_emb * preview_mask).sum(dim=2)  # (batch,2,species_embed_dim)
            preview_counts = preview_mask.sum(dim=2).clamp(min=1.0)  # (batch, 2, 1)
            preview_emb_mean = preview_emb_sum / preview_counts  # all-pad side -> zeros
            feature_parts.append(preview_emb_mean.reshape(batch_size, -1))

        feature_parts.append(item_emb)
        feature_parts.append(ability_emb)

        move_emb = self.move_embedding(move_idx)  # (batch, 4, 4, move_embed_dim)
        move_mask = (move_idx != _MOVE_PAD_IDX).unsqueeze(-1).to(move_emb.dtype)  # (batch,4,4,1)
        move_emb_sum = (move_emb * move_mask).sum(dim=2)  # (batch, 4, move_embed_dim)
        move_counts = move_mask.sum(dim=2).clamp(min=1.0)  # (batch, 4, 1)
        move_emb_mean = move_emb_sum / move_counts  # all-pad slot -> 0/1 == 0 (zeros)
        feature_parts.append(move_emb_mean.reshape(batch_size, -1))
        feature_parts.append(scalars)

        features = torch.cat(feature_parts, dim=-1)
        return self.trunk(features)

    def forward(
        self, index_array: torch.Tensor, scalars: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """`index_array`: `(batch, INDEX_DIM)` int64. `scalars`: `(batch, scalar_dim)`
        float32. Returns `(move_logits, target_logits, value_logit)` -- `None` in any
        slot whose head isn't in `self.heads` (see module docstring).
        """
        hidden = self.encode_hidden(index_array, scalars)
        move_logits = self.move_head(hidden) if self.move_head is not None else None
        target_logits = self.target_head(hidden) if self.target_head is not None else None
        value_logit = self.value_head(hidden).squeeze(-1) if self.value_head is not None else None
        return move_logits, target_logits, value_logit
