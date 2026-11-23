"""Replay-pretraining models: TurnPretrainNet + PreviewPretrainNet.

`TurnPretrainNet.state_encoder` is `BcPolicyNet(heads=(), dropout=0.1)` — the
identical class/config as `CandidatePolicyValueNet.state_encoder`, so the trained
trunk warm-starts the RL candidate scorer unchanged. Heads are slot-conditioned,
option-masked scorers over the rebuilt input's own `revealed_moves` / bench
species. `PreviewPretrainNet` is a separate species-only model; its inputs are
both sides' six preview species canonicalized and SORTED by species id, because
the human preview order carries lead hints the live bot's team-file order does
not.
"""

from __future__ import annotations

import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from vgc.bc.encoding import MOVE_TO_IDX, SPECIES_TO_IDX, SPECIES_VOCAB
from vgc.bc.model import BcPolicyNet, HIDDEN_DIM
from vgc.data import canonical_species_id

SWITCH_IDX = MOVE_TO_IDX.get("<switch>")
assert SWITCH_IDX is not None, "MOVE_VOCAB is missing the <switch> option"
HEAD_WIDTH = 128
COMBOS_4OF6 = list(itertools.combinations(range(6), 4))  # 15 ordered index sets
LEAD_PAIRS = [(a, b) for a in range(6) for b in range(6) if a != b]  # 30 ordered


def _scorer(in_dim: int, width: int = HEAD_WIDTH) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, width), nn.ReLU(), nn.Linear(width, 1))


class TurnPretrainNet(nn.Module):
    """Trunk + option-masked action/destination/value heads for turn records."""

    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        self.state_encoder = BcPolicyNet(heads=(), dropout=dropout)
        move_dim = self.state_encoder.move_embedding.embedding_dim
        species_dim = self.state_encoder.species_embedding.embedding_dim
        self.action_head = _scorer(HIDDEN_DIM + 2 + move_dim)
        self.destination_head = _scorer(HIDDEN_DIM + 2 + species_dim)
        self.value_head = nn.Linear(HIDDEN_DIM, 1)

    def hidden(self, index_array: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        return self.state_encoder.encode_hidden(index_array, scalars)

    def action_logits(
        self, hidden: torch.Tensor, slot: torch.Tensor,
        option_move_idx: torch.Tensor, option_mask: torch.Tensor,
    ) -> torch.Tensor:
        """`(B,K)` logits over each active's `revealed_moves` + `<switch>`."""
        emb = self.state_encoder.move_embedding(option_move_idx)  # (B,K,move_dim)
        feat = torch.cat(
            (
                hidden.unsqueeze(1).expand(-1, option_move_idx.shape[1], -1),
                F.one_hot(slot, 2).float().unsqueeze(1).expand(-1, option_move_idx.shape[1], -1),
                emb,
            ),
            dim=-1,
        )
        logits = self.action_head(feat).squeeze(-1)
        return logits.masked_fill(~option_mask, float("-inf"))

    def destination_logits(
        self, hidden: torch.Tensor, slot: torch.Tensor,
        option_species_idx: torch.Tensor, option_mask: torch.Tensor,
    ) -> torch.Tensor:
        """`(B,K)` logits over the record's own bench species."""
        emb = self.state_encoder.species_embedding(option_species_idx)
        feat = torch.cat(
            (
                hidden.unsqueeze(1).expand(-1, option_species_idx.shape[1], -1),
                F.one_hot(slot, 2).float().unsqueeze(1).expand(-1, option_species_idx.shape[1], -1),
                emb,
            ),
            dim=-1,
        )
        logits = self.destination_head(feat).squeeze(-1)
        return logits.masked_fill(~option_mask, float("-inf"))

    def value_logit(self, hidden: torch.Tensor) -> torch.Tensor:
        """Raw win-probability logit -- use BCE-with-logits in training and apply
        sigmoid only for metrics."""
        return self.value_head(hidden).squeeze(-1)


PREVIEW_EMBED_DIM = 32
PREVIEW_CTX_DIM = 64
PREVIEW_FEAT_DIM = 64


class PreviewPretrainNet(nn.Module):
    """Species-only preview model (this version): brought combo + lead pair."""

    def __init__(self) -> None:
        super().__init__()
        self.species_embedding = nn.Embedding(len(SPECIES_VOCAB), PREVIEW_EMBED_DIM)
        self.opp_ctx_mlp = nn.Sequential(
            nn.Linear(PREVIEW_EMBED_DIM, PREVIEW_CTX_DIM), nn.ReLU()
        )
        self.own_feat_mlp = nn.Sequential(
            nn.Linear(PREVIEW_EMBED_DIM * 2 + PREVIEW_CTX_DIM, PREVIEW_FEAT_DIM), nn.ReLU()
        )
        self.brought_head = _scorer(PREVIEW_FEAT_DIM + PREVIEW_CTX_DIM)
        self.leads_head = _scorer(PREVIEW_FEAT_DIM * 2 + PREVIEW_CTX_DIM)

    def forward(
        self, own6: torch.Tensor, opp6: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """own6/opp6 `(B,6)` species indices -> `(B,15)` brought + `(B,30)` leads."""
        emb = self.species_embedding
        opp_ctx = self.opp_ctx_mlp(emb(opp6).mean(dim=1))  # (B,ctx)
        own_emb = emb(own6)  # (B,6,e)
        own_mean = own_emb.mean(dim=1)  # (B,e)
        feats = self.own_feat_mlp(
            torch.cat(
                (
                    own_emb,
                    own_mean.unsqueeze(1).expand(-1, 6, -1),
                    opp_ctx.unsqueeze(1).expand(-1, 6, -1),
                ),
                dim=-1,
            )
        )  # (B,6,F)
        batch = own6.shape[0]
        combo_idx = torch.tensor(COMBOS_4OF6, device=feats.device)  # (15,4)
        combo_feats = feats[:, combo_idx].mean(dim=2)  # (B,15,F)
        brought = self.brought_head(
            torch.cat(
                (combo_feats, opp_ctx.unsqueeze(1).expand(-1, 15, -1)), dim=-1
            )
        ).squeeze(-1)
        pair_idx = torch.tensor(LEAD_PAIRS, device=feats.device)  # (30,2)
        pair_feats = feats[:, pair_idx].reshape(batch, 30, -1)  # (B,30,2F)
        leads = self.leads_head(
            torch.cat(
                (pair_feats, opp_ctx.unsqueeze(1).expand(-1, 30, -1)), dim=-1
            )
        ).squeeze(-1)
        return brought, leads

    @staticmethod
    def species_indices(species_list: list[str]) -> list[int]:
        unk = SPECIES_TO_IDX["<unk>"]
        return [SPECIES_TO_IDX.get(canonical_species_id(sp), unk) for sp in species_list]

    @torch.no_grad()
    def choose(self, own6: list[str], opp6: list[str]) -> tuple[list[str], list[str]]:
        """Joint argmax: (brought four species, ordered lead pair) by species id.
        log_softmax each head first -- raw logits live on different scales."""
        own_idx = sorted(self.species_indices(own6))
        opp_idx = sorted(self.species_indices(opp6))
        brought, leads = self(
            torch.tensor([own_idx]), torch.tensor([opp_idx])
        )
        log_b = F.log_softmax(brought, dim=1)
        log_l = F.log_softmax(leads, dim=1)
        best = None
        for c_i, combo in enumerate(COMBOS_4OF6):
            pairs_inside = [
                (p_i, pair) for p_i, pair in enumerate(LEAD_PAIRS)
                if pair[0] in combo and pair[1] in combo
            ]
            p_i, pair = max(pairs_inside, key=lambda t: log_l[0, t[0]].item())
            score = log_b[0, c_i].item() + log_l[0, p_i].item()
            if best is None or score > best[0]:
                best = (score, combo, pair)
        _score, combo, pair = best
        return (
            [SPECIES_VOCAB[own_idx[i]] for i in combo],
            [SPECIES_VOCAB[own_idx[pair[0]]], SPECIES_VOCAB[own_idx[pair[1]]]],
        )
