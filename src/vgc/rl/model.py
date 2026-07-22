"""Candidate-aware policy/value network over complete legal doubles orders."""

from __future__ import annotations

from pathlib import Path

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - train extra is optional
    raise ImportError(
        "vgc.rl.model requires the 'train' extra (torch) -- run `uv sync --extra train`."
    ) from exc

from vgc.bc.encoding import TARGET_VOCAB
from vgc.bc.model import BcPolicyNet, HIDDEN_DIM

TARGET_EMBED_DIM = 8
ACTION_HIDDEN_DIM = 128
NUM_ORDER_SLOTS = 2
NUM_ACTION_FLAGS = 4


class CandidatePolicyValueNet(nn.Module):
    """Score a padded set of legal joint actions and value the current position."""

    def __init__(self, *, dropout: float = 0.0) -> None:
        super().__init__()
        # heads=() makes this purely the shared state representation. Its scalar width
        # remains the BC default, including the two zero slot-marker values supplied by
        # rl.encoding.encode_live_state, so current BC trunks can warm-start it.
        self.state_encoder = BcPolicyNet(heads=(), dropout=dropout)
        self.target_embedding = nn.Embedding(len(TARGET_VOCAB), TARGET_EMBED_DIM)
        per_slot_dim = (
            self.state_encoder.move_embedding.embedding_dim
            + TARGET_EMBED_DIM
            + self.state_encoder.species_embedding.embedding_dim
            + NUM_ACTION_FLAGS
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(NUM_ORDER_SLOTS * per_slot_dim, ACTION_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(ACTION_HIDDEN_DIM, ACTION_HIDDEN_DIM),
            nn.ReLU(),
        )
        self.state_projection = nn.Linear(HIDDEN_DIM, ACTION_HIDDEN_DIM)
        self.policy_head = nn.Sequential(
            nn.Linear(ACTION_HIDDEN_DIM * 3, ACTION_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(ACTION_HIDDEN_DIM, 1),
        )
        self.value_head = nn.Linear(HIDDEN_DIM, 1)

    def forward(
        self,
        state_indices: torch.Tensor,
        state_scalars: torch.Tensor,
        move_indices: torch.Tensor,
        target_indices: torch.Tensor,
        switch_species_indices: torch.Tensor,
        action_flags: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if candidate_mask.ndim != 2 or not torch.all(candidate_mask.any(dim=1)):
            raise ValueError("each batch row must contain at least one legal candidate")

        state_hidden = self.state_encoder.encode_hidden(state_indices, state_scalars)
        move_emb = self.state_encoder.move_embedding(move_indices)
        target_emb = self.target_embedding(target_indices)
        species_emb = self.state_encoder.species_embedding(switch_species_indices)
        action_input = torch.cat((move_emb, target_emb, species_emb, action_flags), dim=-1)
        batch, candidates = action_input.shape[:2]
        action_hidden = self.action_encoder(action_input.reshape(batch, candidates, -1))

        state_action = self.state_projection(state_hidden).unsqueeze(1).expand(-1, candidates, -1)
        policy_input = torch.cat(
            (state_action, action_hidden, state_action * action_hidden), dim=-1
        )
        logits = self.policy_head(policy_input).squeeze(-1)
        logits = logits.masked_fill(~candidate_mask.bool(), torch.finfo(logits.dtype).min)
        values = self.value_head(state_hidden).squeeze(-1)
        return logits, values

    def warm_start_state_encoder(self, checkpoint_path: str | Path) -> dict[str, int]:
        """Load compatible BC embedding/trunk weights, ignoring old prediction heads."""

        checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
        source = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        target = self.state_encoder.state_dict()
        compatible = {
            key: value for key, value in source.items() if key in target and target[key].shape == value.shape
        }
        self.state_encoder.load_state_dict(compatible, strict=False)
        return {"loaded": len(compatible), "available": len(target)}
