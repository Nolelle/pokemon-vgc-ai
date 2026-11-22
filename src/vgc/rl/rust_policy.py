"""Autoregressive factored transformer policy over pikapro's native Rust `rich` observation.

Reads the flat 1,028-float `observation="rich"` tensor directly from:
- `pikapro.rl.VecEnv(..., observation="rich", masks="factored")` (self-play training)
- `pikapro.rl.ChannelSession(..., observation="rich", masks="factored")` (live Showdown play)
- `pikapro.Battle.encode_rich(side)` / `encode_state_rich(side)` (Rust lookahead search)

All three paths share the exact same Rust `Rich` encoder and `ActionSpace`, giving
zero train/live observation drift and full Apple Silicon MPS / GPU vectorization.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pikapro
import pikapro.rl as rl
import torch
import torch.nn.functional as F
from torch import nn

RICH_MON_WIDTH = 80
RICH_MON_IDS = 8
RICH_MON_SCALARS = RICH_MON_WIDTH - RICH_MON_IDS  # 72
RICH_SIDE_WIDTH = 20
RICH_FIELD_WIDTH = 28
RICH_GLOBAL_DIM = 2 * RICH_SIDE_WIDTH + RICH_FIELD_WIDTH  # 68
RICH_OBS_SIZE = 12 * RICH_MON_WIDTH + RICH_GLOBAL_DIM  # 1028
RUST_POLICY_VERSION = 1


def champions_dex_sizes(fmt: str = "championsvgc") -> dict[str, int]:
    return {
        "species": len(pikapro.names("species", fmt)),
        "moves": len(pikapro.names("moves", fmt)),
        "abilities": len(pikapro.names("abilities", fmt)),
        "items": len(pikapro.names("items", fmt)),
    }


class RustFactoredPolicy(nn.Module):
    """Encoder-only transformer with autoregressive doubles policy + calibrated value head."""

    def __init__(
        self,
        width: int = 192,
        layers: int = 6,
        *,
        format_id: str = "championsvgc",
        dex_sizes: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.layers = int(layers)
        self.format_id = str(format_id)
        self.dex_sizes = dict(dex_sizes) if dex_sizes is not None else champions_dex_sizes(format_id)

        space = pikapro.ActionSpace(format_id)
        self.slot_size = int(space.slot_size)
        self.preview_size = int(space.preview_size)
        table = rl.action_table(format_id)
        # (360, 4) team indices 0..5 for each preview order.
        preview_orders = np.asarray(table["preview_orders"], dtype=np.int64)
        self.register_buffer("preview_orders", torch.from_numpy(preview_orders), persistent=False)

        self.species_emb = nn.Embedding(self.dex_sizes["species"] + 2, width, padding_idx=0)
        self.item_emb = nn.Embedding(self.dex_sizes["items"] + 2, width, padding_idx=0)
        self.ability_emb = nn.Embedding(self.dex_sizes["abilities"] + 2, width, padding_idx=0)
        self.move_emb = nn.Embedding(self.dex_sizes["moves"] + 2, width, padding_idx=0)
        self.active_emb = nn.Embedding(4, width)
        self.side_emb = nn.Embedding(2, width)
        self.own_slot_emb = nn.Embedding(7, width, padding_idx=0)
        self.register_buffer(
            "side_ids", torch.tensor([0] * 6 + [1] * 6, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "own_slot_ids",
            torch.tensor([1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0], dtype=torch.long),
            persistent=False,
        )

        self.mon_scalars = nn.Linear(RICH_MON_SCALARS, width)
        self.global_proj = nn.Linear(RICH_GLOBAL_DIM, width)

        block = nn.TransformerEncoderLayer(
            width,
            4,
            2 * width,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.board = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)

        self.value_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
        self.slot0_head = nn.Sequential(
            nn.LayerNorm(2 * width),
            nn.Linear(2 * width, width),
            nn.GELU(),
            nn.Linear(width, self.slot_size),
        )
        self.slot0_action_emb = nn.Embedding(self.slot_size + 1, width)
        self.slot1_head = nn.Sequential(
            nn.LayerNorm(3 * width),
            nn.Linear(3 * width, width),
            nn.GELU(),
            nn.Linear(width, self.slot_size),
        )
        self.preview_head = nn.Sequential(
            nn.Linear(5 * width, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )

    @property
    def device(self) -> torch.device:
        return self.species_emb.weight.device

    def encode_board(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(context, own_tokens, active0_token, active1_token)``."""
        bsz = obs.shape[0]
        mons = obs[:, : 12 * RICH_MON_WIDTH].view(bsz, 12, RICH_MON_WIDTH)
        glob = obs[:, 12 * RICH_MON_WIDTH :]

        ids = mons[:, :, :RICH_MON_IDS].long()
        sp = ids[:, :, 0].clamp(0, self.species_emb.num_embeddings - 1)
        it = ids[:, :, 1].clamp(0, self.item_emb.num_embeddings - 1)
        ab = ids[:, :, 2].clamp(0, self.ability_emb.num_embeddings - 1)
        act = ids[:, :, 3].clamp(0, 3)
        mvs = ids[:, :, 4:8].clamp(0, self.move_emb.num_embeddings - 1)

        mon_tokens = (
            self.species_emb(sp)
            + self.item_emb(it)
            + self.ability_emb(ab)
            + self.active_emb(act)
            + self.move_emb(mvs).sum(dim=-2)
            + self.side_emb(self.side_ids)
            + self.own_slot_emb(self.own_slot_ids)
            + self.mon_scalars(mons[:, :, RICH_MON_IDS:])
        )
        glob_token = self.global_proj(glob).unsqueeze(1)
        encoded = self.board(torch.cat((glob_token, mon_tokens), dim=1))
        context = encoded[:, 0]
        own_tokens = encoded[:, 1:7]  # (B, 6, W)

        own_act = act[:, :6]
        mask0 = (own_act == 1).unsqueeze(-1).to(own_tokens.dtype)
        mask1 = (own_act == 2).unsqueeze(-1).to(own_tokens.dtype)
        active0 = (own_tokens * mask0).sum(dim=1)
        active1 = (own_tokens * mask1).sum(dim=1)
        return context, own_tokens, active0, active1

    def value_from_context(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        value_logit = self.value_head(context).squeeze(-1)
        value = 2.0 * torch.sigmoid(value_logit) - 1.0
        return value, value_logit

    def preview_logits(self, context: torch.Tensor, own_tokens: torch.Tensor) -> torch.Tensor:
        # Gather 4 own tokens per preview permutation: (B, 360, 4, W) -> (B, 360, 4*W)
        bsz = own_tokens.shape[0]
        gathered = own_tokens[:, self.preview_orders, :].reshape(
            bsz, self.preview_size, 4 * self.width
        )
        ctx = context[:, None, :].expand(-1, self.preview_size, -1)
        return self.preview_head(torch.cat((ctx, gathered), dim=-1)).squeeze(-1)

    def slot0_logits(self, context: torch.Tensor, active0: torch.Tensor) -> torch.Tensor:
        return self.slot0_head(torch.cat((context, active0), dim=-1))

    def slot1_logits(
        self, context: torch.Tensor, active1: torch.Tensor, a0: torch.Tensor
    ) -> torch.Tensor:
        a0_clamped = a0.clamp(0, self.slot_size)
        return self.slot1_head(torch.cat((context, active1, self.slot0_action_emb(a0_clamped)), dim=-1))

    @torch.no_grad()
    def values_from_obs(self, obs_np: np.ndarray) -> np.ndarray:
        if len(obs_np) == 0:
            return np.empty(0, dtype=np.float32)
        x = torch.from_numpy(np.ascontiguousarray(obs_np, dtype=np.float32)).to(self.device)
        ctx, _, _, _ = self.encode_board(x)
        val, _ = self.value_from_context(ctx)
        return val.cpu().numpy()

    @torch.no_grad()
    def top_joint_actions(
        self,
        obs_1d: np.ndarray,
        factored_mask: np.ndarray,
        min_switches: int,
        space: pikapro.ActionSpace,
        *,
        limit: int = 8,
    ) -> tuple[list[tuple[int, int]], np.ndarray, float]:
        """Rank legal joint `(a0, a1)` pairs by exact autoregressive log-probability.

        Returns ``(pairs, probs, value)`` where ``probs`` sums to 1 over ``pairs``.
        """
        s = self.slot_size
        x = torch.from_numpy(np.ascontiguousarray(obs_1d, dtype=np.float32)).unsqueeze(0).to(
            self.device
        )
        ctx, _, act0, act1 = self.encode_board(x)
        val, _ = self.value_from_context(ctx)
        value_f = float(val[0].item())

        m0 = np.asarray(factored_mask[:s], dtype=bool)
        legal_a0 = np.flatnonzero(m0)
        if len(legal_a0) == 0:
            return [], np.empty(0, dtype=np.float64), value_f

        l0 = self.slot0_logits(ctx, act0)[0].cpu().numpy()
        l0_masked = np.where(m0, l0, -1e9)
        lp0 = l0_masked - float(np.max(l0_masked))
        lp0 = lp0 - float(np.log(np.sum(np.exp(lp0[m0]))))

        # Rank all legal a0 candidates (top-8 is plenty since joint limit <= 8).
        a0_order = legal_a0[np.argsort(-lp0[legal_a0], kind="stable")][: max(limit, 8)]
        n0 = len(a0_order)
        m1_batch = space.pair_mask(
            np.broadcast_to(factored_mask[None, :], (n0, len(factored_mask))),
            np.full(n0, min_switches, dtype=np.uint8),
            a0_order,
        )
        ctx_rep = ctx.expand(n0, -1)
        act1_rep = act1.expand(n0, -1)
        a0_t = torch.from_numpy(a0_order.astype(np.int64)).to(self.device)
        l1_batch = self.slot1_logits(ctx_rep, act1_rep, a0_t).cpu().numpy()

        scored: list[tuple[float, int, int]] = []
        for idx, a0_val in enumerate(a0_order):
            m1 = m1_batch[idx]
            legal_a1 = np.flatnonzero(m1)
            if len(legal_a1) == 0:
                continue
            l1 = np.where(m1, l1_batch[idx], -1e9)
            l1_shift = l1 - float(np.max(l1[m1]))
            log_z = float(np.log(np.sum(np.exp(l1_shift[m1]))))
            for a1_val in legal_a1:
                joint_lp = float(lp0[a0_val]) + float(l1_shift[a1_val] - log_z)
                scored.append((joint_lp, int(a0_val), int(a1_val)))

        scored.sort(key=lambda item: -item[0])
        top = scored[:limit]
        if not top:
            return [], np.empty(0, dtype=np.float64), value_f
        top_lps = np.asarray([t[0] for t in top], dtype=np.float64)
        top_lps -= float(np.max(top_lps))
        probs = np.exp(top_lps)
        probs /= float(np.sum(probs))
        return [(t[1], t[2]) for t in top], probs, value_f

    @torch.no_grad()
    def choose_preview_index(self, obs_1d: np.ndarray, preview_mask: np.ndarray) -> int:
        x = torch.from_numpy(np.ascontiguousarray(obs_1d, dtype=np.float32)).unsqueeze(0).to(
            self.device
        )
        ctx, own_tokens, _, _ = self.encode_board(x)
        logits = self.preview_logits(ctx, own_tokens)[0]
        mask_t = torch.from_numpy(np.asarray(preview_mask, dtype=bool)).to(self.device)
        logits = logits.masked_fill(~mask_t, -torch.inf)
        return int(logits.argmax().item())

    def checkpoint(self) -> dict[str, object]:
        return {
            "rust_policy_version": RUST_POLICY_VERSION,
            "format_id": self.format_id,
            "width": self.width,
            "layers": self.layers,
            "dex_sizes": self.dex_sizes,
            "model": {k: v.detach().cpu() for k, v in self.state_dict().items()},
        }


def save_rust_policy(path: str | Path, model: RustFactoredPolicy) -> Path:
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    torch.save(model.checkpoint(), tmp)
    tmp.replace(path)
    return path


def load_rust_policy(path: str | Path, device: str = "cpu") -> RustFactoredPolicy:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    if ckpt.get("rust_policy_version") != RUST_POLICY_VERSION:
        raise ValueError(f"unsupported rust policy checkpoint: {path}")
    model = RustFactoredPolicy(
        width=int(ckpt["width"]),
        layers=int(ckpt["layers"]),
        format_id=str(ckpt.get("format_id", "championsvgc")),
        dex_sizes=dict(ckpt["dex_sizes"]),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    return model.eval()


def masked_log_prob_and_entropy(
    logits: torch.Tensor, mask: torch.Tensor, actions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute exact log-probability and entropy over masked categorical actions."""
    masked = logits.masked_fill(~mask, -1e9)
    log_probs = F.log_softmax(masked, dim=-1)
    probs = log_probs.exp() * mask.to(log_probs.dtype)
    # Clamp zero-prob entries safely so 0 * (-1e9) cannot produce NaN.
    entropy = -(probs * torch.where(mask, log_probs, torch.zeros_like(log_probs))).sum(dim=-1)
    chosen = log_probs.gather(-1, actions.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return chosen, entropy
