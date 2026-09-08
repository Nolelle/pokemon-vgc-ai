"""Candidate-aware policy/value network over complete legal doubles orders."""

from __future__ import annotations

from pathlib import Path

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except ImportError as exc:  # pragma: no cover - train extra is optional
    raise ImportError(
        "vgc.rl.model requires the 'train' extra (torch) -- run `uv sync --extra train`."
    ) from exc

from vgc.bc.encoding import TARGET_VOCAB
from vgc.bc.model import BcPolicyNet, HIDDEN_DIM
from vgc.rl.encoding import (
    HISTORY_SCALAR_DIM,
    INFORMATION_MOVES_PER_MON,
    INFORMATION_TEAM_SLOTS,
    META_SCALAR_DIM,
    OPP_SCALARS_PER_MON,
    OWN_SCALARS_PER_MON,
    TACTICAL_FEATURE_DIM,
)
from vgc.rl.mechanics_encoding import MECHANICS_TOKEN_VOCAB_SIZE

TARGET_EMBED_DIM = 8
ACTION_HIDDEN_DIM = 128
HISTORY_HIDDEN_DIM = 64
NUM_ORDER_SLOTS = 2
NUM_ACTION_FLAGS = 4
# Meta-features branch (opponent/our archetype + set-prior reveal scalars -- see
# vgc.rl.encoding.encode_meta_context). Only built when use_meta_features=True, so the
# default architecture is byte-for-byte unchanged.
META_HIDDEN_DIM = 32
INFORMATION_MON_HIDDEN_DIM = 64
INFORMATION_HIDDEN_DIM = 128
TACTICAL_HIDDEN_DIM = 64
MECHANICS_BYTE_EMBED_DIM = 16
MECHANICS_CONV_DIM = 32
MECHANICS_HIDDEN_DIM = 64


class CandidatePolicyValueNet(nn.Module):
    """Score a padded set of legal joint actions and value the current position."""

    def __init__(
        self,
        *,
        dropout: float = 0.0,
        use_meta_features: bool = False,
        use_information_features: bool = False,
        use_tactical_features: bool = False,
        use_mechanics_features: bool = False,
        head_dropout: float = 0.0,
        value_output_transform: str = "identity",
        head_width: int | None = None,
    ) -> None:
        super().__init__()
        if value_output_transform not in ("identity", "tanh"):
            raise ValueError("value_output_transform must be 'identity' or 'tanh'")
        self.use_meta_features = use_meta_features
        self.use_information_features = use_information_features
        self.use_tactical_features = use_tactical_features
        self.use_mechanics_features = use_mechanics_features
        self.value_output_transform = value_output_transform
        # Capacity lever for the action-scoring path only. None keeps the historical
        # ACTION_HIDDEN_DIM so existing checkpoints stay byte-compatible; a larger value
        # widens action_encoder/state_projection/policy_head, which are freshly
        # initialized on every distillation run (only the state trunk warm-starts), so
        # this never breaks warm-starting. PPO snapshots of a wide model are only
        # loadable into the same width (load_snapshot restores it).
        self.head_width = int(head_width) if head_width else ACTION_HIDDEN_DIM
        # Applied functionally in forward() (see there) rather than as nn.Dropout
        # modules inserted into the existing nn.Sequential stacks -- inserting modules
        # would shift child indices (action_encoder.0/.2, ...) and break every existing
        # checkpoint's state_dict keys. head_dropout=0.0 (the default) makes
        # F.dropout(..., p=0.0) an exact no-op, so this is byte-for-byte unchanged from
        # before this argument existed.
        self.head_dropout_p = float(head_dropout)
        # heads=() makes this purely the shared state representation. Its scalar width
        # remains the BC default, including the two zero slot-marker values supplied by
        # rl.encoding.encode_live_state, so current BC trunks can warm-start it.
        self.state_encoder = BcPolicyNet(heads=(), dropout=dropout)
        self.target_embedding = nn.Embedding(len(TARGET_VOCAB), TARGET_EMBED_DIM)
        self.history_encoder = nn.Sequential(
            nn.Linear(HISTORY_SCALAR_DIM, HISTORY_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HISTORY_HIDDEN_DIM, HISTORY_HIDDEN_DIM),
            nn.ReLU(),
        )
        context_input_dim = HIDDEN_DIM + HISTORY_HIDDEN_DIM
        if use_meta_features:
            # A new parallel branch beside history_encoder -- never touches the
            # warm-started state_encoder trunk (see warm_start_state_encoder).
            self.meta_encoder = nn.Sequential(
                nn.Linear(META_SCALAR_DIM, META_HIDDEN_DIM),
                nn.ReLU(),
                nn.Linear(META_HIDDEN_DIM, META_HIDDEN_DIM),
                nn.ReLU(),
            )
            context_input_dim += META_HIDDEN_DIM
        if use_information_features:
            species_dim = self.state_encoder.species_embedding.embedding_dim
            item_dim = self.state_encoder.item_embedding.embedding_dim
            ability_dim = self.state_encoder.ability_embedding.embedding_dim
            move_dim = self.state_encoder.move_embedding.embedding_dim
            categorical_dim = species_dim + item_dim + ability_dim + move_dim
            self.own_information_mon_encoder = nn.Sequential(
                nn.Linear(categorical_dim + OWN_SCALARS_PER_MON, INFORMATION_MON_HIDDEN_DIM),
                nn.ReLU(),
            )
            self.opp_information_mon_encoder = nn.Sequential(
                nn.Linear(categorical_dim + OPP_SCALARS_PER_MON, INFORMATION_MON_HIDDEN_DIM),
                nn.ReLU(),
            )
            self.information_encoder = nn.Sequential(
                nn.Linear(
                    INFORMATION_TEAM_SLOTS * INFORMATION_MON_HIDDEN_DIM * 2,
                    INFORMATION_HIDDEN_DIM,
                ),
                nn.ReLU(),
            )
            context_input_dim += INFORMATION_HIDDEN_DIM
        if use_mechanics_features:
            # Complete, reversible public-state bytes run beside the historical summary
            # branches. A strided convolution reads local JSON key/value patterns; both
            # mean and max pooling retain signals regardless of snapshot length.
            self.mechanics_embedding = nn.Embedding(
                MECHANICS_TOKEN_VOCAB_SIZE,
                MECHANICS_BYTE_EMBED_DIM,
                padding_idx=0,
            )
            self.mechanics_conv = nn.Conv1d(
                MECHANICS_BYTE_EMBED_DIM,
                MECHANICS_CONV_DIM,
                kernel_size=7,
                stride=4,
                padding=3,
            )
            self.mechanics_encoder = nn.Sequential(
                nn.Linear(MECHANICS_CONV_DIM * 2, MECHANICS_HIDDEN_DIM),
                nn.ReLU(),
            )
            context_input_dim += MECHANICS_HIDDEN_DIM
        self.context_encoder = nn.Sequential(
            nn.Linear(context_input_dim, HIDDEN_DIM),
            nn.ReLU(),
        )
        per_slot_dim = (
            self.state_encoder.move_embedding.embedding_dim
            + TARGET_EMBED_DIM
            + self.state_encoder.species_embedding.embedding_dim
            + NUM_ACTION_FLAGS
        )
        action_input_dim = NUM_ORDER_SLOTS * per_slot_dim
        if use_tactical_features:
            self.tactical_encoder = nn.Sequential(
                nn.Linear(TACTICAL_FEATURE_DIM, TACTICAL_HIDDEN_DIM),
                nn.ReLU(),
                nn.Linear(TACTICAL_HIDDEN_DIM, TACTICAL_HIDDEN_DIM),
                nn.ReLU(),
            )
            action_input_dim += TACTICAL_HIDDEN_DIM
        self.action_encoder = nn.Sequential(
            nn.Linear(action_input_dim, self.head_width),
            nn.ReLU(),
            nn.Linear(self.head_width, self.head_width),
            nn.ReLU(),
        )
        self.state_projection = nn.Linear(HIDDEN_DIM, self.head_width)
        self.policy_head = nn.Sequential(
            nn.Linear(self.head_width * 3, self.head_width),
            nn.ReLU(),
            nn.Linear(self.head_width, 1),
        )
        self.value_head = nn.Linear(HIDDEN_DIM, 1)

    def forward(
        self,
        state_indices: torch.Tensor,
        state_scalars: torch.Tensor,
        history_scalars: torch.Tensor,
        move_indices: torch.Tensor,
        target_indices: torch.Tensor,
        switch_species_indices: torch.Tensor,
        action_flags: torch.Tensor,
        candidate_mask: torch.Tensor,
        meta_scalars: torch.Tensor | None = None,
        information_indices: torch.Tensor | None = None,
        information_scalars: torch.Tensor | None = None,
        tactical_features: torch.Tensor | None = None,
        mechanics_tokens: torch.Tensor | None = None,
        mechanics_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if candidate_mask.ndim != 2 or not torch.all(candidate_mask.any(dim=1)):
            raise ValueError("each batch row must contain at least one legal candidate")

        context_hidden = self.encode_context(
            state_indices,
            state_scalars,
            history_scalars,
            meta_scalars=meta_scalars,
            information_indices=information_indices,
            information_scalars=information_scalars,
            mechanics_tokens=mechanics_tokens,
            mechanics_mask=mechanics_mask,
        )
        # head_dropout applies only to the randomly-initialised action-scoring path --
        # the part of the network that overfits fastest on a small teacher-distillation
        # sample set. `action_context` is a dropped-out COPY of context_hidden used only
        # to build state_action below; `context_hidden` itself is left untouched so that
        # value_head's input (below) is never affected -- value learning shouldn't be
        # destabilized by a regularizer aimed at the policy's action-scoring path.
        action_context = F.dropout(context_hidden, p=self.head_dropout_p, training=self.training)
        action_hidden = self.encode_actions(
            move_indices,
            target_indices,
            switch_species_indices,
            action_flags,
            tactical_features=tactical_features,
        )

        candidates = action_hidden.shape[1]
        state_action = self.state_projection(action_context).unsqueeze(1).expand(-1, candidates, -1)
        policy_input = torch.cat(
            (state_action, action_hidden, state_action * action_hidden), dim=-1
        )
        logits = self.policy_head(policy_input).squeeze(-1)
        logits = logits.masked_fill(~candidate_mask.bool(), torch.finfo(logits.dtype).min)
        values = self.value_head(context_hidden).squeeze(-1)
        if self.value_output_transform == "tanh":
            # Terminal-return calibration checkpoints use +/-1 targets. Tanh makes
            # that contract explicit and prevents a search leaf from receiving an
            # impossible value such as +1.6 merely because the linear head extrapolated.
            values = torch.tanh(values)
        return logits, values

    def encode_context(
        self,
        state_indices: torch.Tensor,
        state_scalars: torch.Tensor,
        history_scalars: torch.Tensor,
        *,
        meta_scalars: torch.Tensor | None = None,
        information_indices: torch.Tensor | None = None,
        information_scalars: torch.Tensor | None = None,
        mechanics_tokens: torch.Tensor | None = None,
        mechanics_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode one fog-safe battle observation for any candidate-aware head."""

        state_hidden = self.state_encoder.encode_hidden(state_indices, state_scalars)
        history_hidden = self.history_encoder(history_scalars)
        context_parts = [state_hidden, history_hidden]
        if self.use_meta_features:
            if meta_scalars is None:
                raise ValueError("use_meta_features=True requires meta_scalars")
            meta_hidden = self.meta_encoder(meta_scalars)
            context_parts.append(meta_hidden)
        if self.use_information_features:
            if information_indices is None or information_scalars is None:
                raise ValueError(
                    "use_information_features=True requires information indices and scalars"
                )
            context_parts.append(
                self._encode_information(information_indices, information_scalars)
            )
        if self.use_mechanics_features:
            if mechanics_tokens is None or mechanics_mask is None:
                raise ValueError(
                    "use_mechanics_features=True requires mechanics tokens and mask"
                )
            context_parts.append(
                self._encode_mechanics(mechanics_tokens, mechanics_mask)
            )
        return self.context_encoder(torch.cat(context_parts, dim=-1))

    def _encode_mechanics(
        self, tokens: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Read the complete public snapshot without truncating padded batches."""

        if tokens.ndim != 2 or mask.shape != tokens.shape:
            raise ValueError("mechanics tokens and mask must have matching (batch, bytes) shape")
        if not torch.all(mask.bool().any(dim=1)):
            raise ValueError("every mechanics-enabled row must contain a complete snapshot")
        embedded = self.mechanics_embedding(tokens).transpose(1, 2)
        convolved = F.relu(self.mechanics_conv(embedded))
        pooled_mask = F.max_pool1d(
            mask.to(convolved.dtype).unsqueeze(1),
            kernel_size=7,
            stride=4,
            padding=3,
        ).bool()
        masked = convolved.masked_fill(~pooled_mask, 0.0)
        counts = pooled_mask.sum(dim=2).clamp_min(1).to(convolved.dtype)
        mean = masked.sum(dim=2) / counts
        maximum = convolved.masked_fill(~pooled_mask, torch.finfo(convolved.dtype).min).amax(dim=2)
        return self.mechanics_encoder(torch.cat((mean, maximum), dim=1))

    def encode_actions(
        self,
        move_indices: torch.Tensor,
        target_indices: torch.Tensor,
        switch_species_indices: torch.Tensor,
        action_flags: torch.Tensor,
        *,
        tactical_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode joint actions while preserving the existing policy representation."""

        move_emb = self.state_encoder.move_embedding(move_indices)
        target_emb = self.target_embedding(target_indices)
        species_emb = self.state_encoder.species_embedding(switch_species_indices)
        action_input = torch.cat((move_emb, target_emb, species_emb, action_flags), dim=-1)
        batch, candidates = action_input.shape[:2]
        action_input = action_input.reshape(batch, candidates, -1)
        if self.use_tactical_features:
            if tactical_features is None:
                raise ValueError("use_tactical_features=True requires tactical_features")
            action_input = torch.cat(
                (action_input, self.tactical_encoder(tactical_features)), dim=-1
            )
        action_hidden = self.action_encoder(action_input)
        return F.dropout(action_hidden, p=self.head_dropout_p, training=self.training)

    def _encode_information(
        self, indices: torch.Tensor, scalars: torch.Tensor
    ) -> torch.Tensor:
        """Embed the fixed six-by-six information contract into one context vector."""

        slots = INFORMATION_TEAM_SLOTS
        moves_width = slots * INFORMATION_MOVES_PER_MON
        own_species = indices[:, 0:slots]
        own_items = indices[:, slots : 2 * slots]
        own_abilities = indices[:, 2 * slots : 3 * slots]
        own_moves = indices[:, 3 * slots : 3 * slots + moves_width]
        opp_start = 3 * slots + moves_width
        opp_species = indices[:, opp_start : opp_start + slots]
        opp_items = indices[:, opp_start + slots : opp_start + 2 * slots]
        opp_abilities = indices[:, opp_start + 2 * slots : opp_start + 3 * slots]
        opp_moves = indices[
            :, opp_start + 3 * slots : opp_start + 3 * slots + moves_width
        ]

        def categorical(species, items, abilities, moves):
            move_embeddings = self.state_encoder.move_embedding(
                moves.reshape(-1, slots, INFORMATION_MOVES_PER_MON)
            ).mean(dim=2)
            return torch.cat(
                (
                    self.state_encoder.species_embedding(species),
                    self.state_encoder.item_embedding(items),
                    self.state_encoder.ability_embedding(abilities),
                    move_embeddings,
                ),
                dim=-1,
            )

        own_scalar_width = slots * OWN_SCALARS_PER_MON
        own_scalars = scalars[:, :own_scalar_width].reshape(
            -1, slots, OWN_SCALARS_PER_MON
        )
        opp_scalars = scalars[:, own_scalar_width:].reshape(
            -1, slots, OPP_SCALARS_PER_MON
        )
        own_hidden = self.own_information_mon_encoder(
            torch.cat(
                (categorical(own_species, own_items, own_abilities, own_moves), own_scalars),
                dim=-1,
            )
        )
        opp_hidden = self.opp_information_mon_encoder(
            torch.cat(
                (categorical(opp_species, opp_items, opp_abilities, opp_moves), opp_scalars),
                dim=-1,
            )
        )
        return self.information_encoder(
            torch.cat((own_hidden.flatten(1), opp_hidden.flatten(1)), dim=-1)
        )

    def warm_start_state_encoder(self, checkpoint_path: str | Path) -> dict[str, int]:
        """Load compatible BC embedding/trunk weights, ignoring old prediction heads."""

        checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
        source = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        target = self.state_encoder.state_dict()
        compatible = {
            key: value
            for key, value in source.items()
            if key in target and target[key].shape == value.shape
        }
        self.state_encoder.load_state_dict(compatible, strict=False)
        return {"loaded": len(compatible), "available": len(target)}
