"""Autoregressive factored transformer policy over pikapro's native Rust `rich` observation.

Reads the flat 1,028-float `observation="rich"` tensor directly from:
- `pikapro.rl.VecEnv(..., observation="rich", masks="factored")` (self-play training)
- `pikapro.rl.ChannelSession(..., observation="rich", masks="factored")` (live Showdown play)
- `pikapro.Battle.encode_rich(side)` / `encode_state_rich(side)` (Rust lookahead search)

Architecture (v2 — move-order & team-slot equivariant):
- Encoder-only Transformer over the 12 Pokémon tokens + 1 global field/side token.
- Zero positional slot bias on own Pokémon or moves: every Pokémon and move generalizes
  100% identically regardless of team-slot ordering or move-slot ordering on unseen teams.
- Semantic move head scores each of the acting Pokémon's 4 moves directly from its move
  embedding, PP fraction, and disabled flag into the 10 `(target, mega)` variants.
- Semantic switch head scores each of the 6 own Pokémon directly from its contextualized
  Transformer token.
- Autoregressive Slot 1 conditioning encodes the semantic meaning of Slot 0's chosen move
  (move embedding + target/mega variant) or switch target (Pokémon token) rather than a
  blind action index.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pikapro
import pikapro.rl as rl
import torch
import torch.nn.functional as F
from torch import nn

RICH_MON_WIDTH = 88
RICH_MON_IDS = 8
RICH_MON_SCALARS = RICH_MON_WIDTH - RICH_MON_IDS  # 80
RICH_SIDE_WIDTH = 20
RICH_FIELD_WIDTH = 28
RICH_GLOBAL_DIM = 2 * RICH_SIDE_WIDTH + RICH_FIELD_WIDTH  # 68
RICH_OBS_SIZE = 12 * RICH_MON_WIDTH + RICH_GLOBAL_DIM  # 1124
RICH_FIELD_TURN_IDX = 12 * RICH_MON_WIDTH + 2 * RICH_SIDE_WIDTH + 20
RUST_POLICY_VERSION = 2

_MOVE_SLOTS = 4
_VARIANTS_PER_MOVE = 10  # 5 targets x 2 (normal, mega)
_NUM_MOVE_ACTIONS = _MOVE_SLOTS * _VARIANTS_PER_MOVE  # 40
_NUM_SWITCH_ACTIONS = 6
_PASS_ACTION = 46


@dataclass(frozen=True)
class BoardEncoding:
    """Contextualized board tensors produced once per observation batch."""

    context: torch.Tensor  # (B, W)
    own_tokens: torch.Tensor  # (B, 6, W)
    active0: torch.Tensor  # (B, W)
    active1: torch.Tensor  # (B, W)
    foe0: torch.Tensor  # (B, W)
    foe1: torch.Tensor  # (B, W)
    active0_mvs: torch.Tensor  # (B, 4) long move IDs
    active1_mvs: torch.Tensor  # (B, 4) long move IDs
    active0_mfeats: torch.Tensor  # (B, 4, 2) [pp_frac, disabled]
    active1_mfeats: torch.Tensor  # (B, 4, 2) [pp_frac, disabled]

    def index(self, idx: torch.Tensor) -> BoardEncoding:
        return BoardEncoding(
            context=self.context[idx],
            own_tokens=self.own_tokens[idx],
            active0=self.active0[idx],
            active1=self.active1[idx],
            foe0=self.foe0[idx],
            foe1=self.foe1[idx],
            active0_mvs=self.active0_mvs[idx],
            active1_mvs=self.active1_mvs[idx],
            active0_mfeats=self.active0_mfeats[idx],
            active1_mfeats=self.active1_mfeats[idx],
        )

    def expand_batch(self, n: int) -> BoardEncoding:
        return BoardEncoding(
            context=self.context.expand(n, -1),
            own_tokens=self.own_tokens.expand(n, -1, -1),
            active0=self.active0.expand(n, -1),
            active1=self.active1.expand(n, -1),
            foe0=self.foe0.expand(n, -1),
            foe1=self.foe1.expand(n, -1),
            active0_mvs=self.active0_mvs.expand(n, -1),
            active1_mvs=self.active1_mvs.expand(n, -1),
            active0_mfeats=self.active0_mfeats.expand(n, -1, -1),
            active1_mfeats=self.active1_mfeats.expand(n, -1, -1),
        )


def champions_dex_sizes(fmt: str = "championsvgc") -> dict[str, int]:
    return {
        "species": len(pikapro.names("species", fmt)),
        "moves": len(pikapro.names("moves", fmt)),
        "abilities": len(pikapro.names("abilities", fmt)),
        "items": len(pikapro.names("items", fmt)),
    }


class RustFactoredPolicy(nn.Module):
    """Equivariant encoder-only transformer with semantic autoregressive doubles heads."""

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
        self.dex_sizes = (
            dict(dex_sizes) if dex_sizes is not None else champions_dex_sizes(format_id)
        )

        space = pikapro.ActionSpace(format_id)
        self.slot_size = int(space.slot_size)
        self.preview_size = int(space.preview_size)
        if self.slot_size != _PASS_ACTION + 1:
            raise ValueError(f"unexpected championsvgc slot_size {self.slot_size}")
        table = rl.action_table(format_id)
        preview_orders = np.asarray(table["preview_orders"], dtype=np.int64)
        self.register_buffer("preview_orders", torch.from_numpy(preview_orders), persistent=False)

        self.species_emb = nn.Embedding(self.dex_sizes["species"] + 2, width, padding_idx=0)
        self.item_emb = nn.Embedding(self.dex_sizes["items"] + 2, width, padding_idx=0)
        self.ability_emb = nn.Embedding(self.dex_sizes["abilities"] + 2, width, padding_idx=0)
        self.move_emb = nn.Embedding(self.dex_sizes["moves"] + 2, width, padding_idx=0)
        self.active_emb = nn.Embedding(4, width)
        self.side_emb = nn.Embedding(2, width)
        self.register_buffer(
            "side_ids", torch.tensor([0] * 6 + [1] * 6, dtype=torch.long), persistent=False
        )

        # Remove per-slot PP/disabled from mon_scalars so the Transformer token is 100%
        # permutation-invariant to move slot order; PP/disabled feed directly into move_head
        # AND into the board token as a pooled move-state sum (`move_state_proj`).
        self.mon_scalars = nn.Linear(RICH_MON_SCALARS - 8, width)
        self.move_state_proj = nn.Linear(2, width)
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

        # Slot 0 context & semantic action heads
        self.slot0_ctx_proj = nn.Sequential(
            nn.LayerNorm(5 * width),
            nn.Linear(5 * width, width),
            nn.GELU(),
        )
        self.slot0_move_head = nn.Sequential(
            nn.LayerNorm(2 * width + 2),
            nn.Linear(2 * width + 2, width),
            nn.GELU(),
            nn.Linear(width, _VARIANTS_PER_MOVE),
        )
        self.slot0_switch_head = nn.Sequential(
            nn.LayerNorm(2 * width),
            nn.Linear(2 * width, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )
        self.slot0_pass_head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, 1),
        )

        # Semantic encoding of Slot 0's chosen action for autoregressive Slot 1 conditioning
        self.move_variant_emb = nn.Embedding(_VARIANTS_PER_MOVE, width)
        self.switch_chosen_emb = nn.Parameter(torch.zeros(width))
        self.pass_chosen_emb = nn.Parameter(torch.zeros(width))

        # Slot 1 context & semantic action heads
        self.slot1_ctx_proj = nn.Sequential(
            nn.LayerNorm(6 * width),
            nn.Linear(6 * width, width),
            nn.GELU(),
        )
        self.slot1_move_head = nn.Sequential(
            nn.LayerNorm(2 * width + 2),
            nn.Linear(2 * width + 2, width),
            nn.GELU(),
            nn.Linear(width, _VARIANTS_PER_MOVE),
        )
        self.slot1_switch_head = nn.Sequential(
            nn.LayerNorm(2 * width),
            nn.Linear(2 * width, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )
        self.slot1_pass_head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, 1),
        )

        self.preview_head = nn.Sequential(
            nn.Linear(5 * width, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )

    @property
    def device(self) -> torch.device:
        return self.species_emb.weight.device

    def encode_board(self, obs: torch.Tensor) -> BoardEncoding:
        """Encode a batch of 1,028-float Rust `rich` observations into `BoardEncoding`."""
        bsz = obs.shape[0]
        mons = obs[:, : 12 * RICH_MON_WIDTH].view(bsz, 12, RICH_MON_WIDTH)
        glob = obs[:, 12 * RICH_MON_WIDTH :]

        ids = mons[:, :, :RICH_MON_IDS].long()
        sp = ids[:, :, 0].clamp(0, self.species_emb.num_embeddings - 1)
        it = ids[:, :, 1].clamp(0, self.item_emb.num_embeddings - 1)
        ab = ids[:, :, 2].clamp(0, self.ability_emb.num_embeddings - 1)
        act = ids[:, :, 3].clamp(0, 3)
        mvs = ids[:, :, 4:8].clamp(0, self.move_emb.num_embeddings - 1)

        # Columns 66..70 are move0..3_pp, 70..74 are move0..3_disabled.
        # Pair each move's (pp, disabled) with its own embedding so the mon token is
        # 100% invariant to permuting move slots 0..3.
        move_pp = mons[:, :, 66:70]
        move_dis = mons[:, :, 70:74]
        move_feats = torch.stack((move_pp, move_dis), dim=-1)  # (B, 12, 4, 2)
        move_present = (mvs > 0).unsqueeze(-1).to(obs.dtype)
        move_pool = (
            (self.move_emb(mvs) + self.move_state_proj(move_feats)) * move_present
        ).sum(dim=-2)

        scalars_no_mv = torch.cat((mons[:, :, RICH_MON_IDS:66], mons[:, :, 74:]), dim=-1)
        mon_tokens = (
            self.species_emb(sp)
            + self.item_emb(it)
            + self.ability_emb(ab)
            + self.active_emb(act)
            + move_pool
            + self.side_emb(self.side_ids)
            + self.mon_scalars(scalars_no_mv)
        )
        glob_token = self.global_proj(glob).unsqueeze(1)
        encoded = self.board(torch.cat((glob_token, mon_tokens), dim=1))
        context = encoded[:, 0]
        own_tokens = encoded[:, 1:7]  # (B, 6, W)
        foe_tokens = encoded[:, 7:13]  # (B, 6, W)

        own_act = act[:, :6]
        foe_act = act[:, 6:12]
        m_own0 = (own_act == 1).unsqueeze(-1)
        m_own1 = (own_act == 2).unsqueeze(-1)
        m_foe0 = (foe_act == 1).unsqueeze(-1)
        m_foe1 = (foe_act == 2).unsqueeze(-1)

        active0 = (own_tokens * m_own0.to(own_tokens.dtype)).sum(dim=1)
        active1 = (own_tokens * m_own1.to(own_tokens.dtype)).sum(dim=1)
        foe0 = (foe_tokens * m_foe0.to(foe_tokens.dtype)).sum(dim=1)
        foe1 = (foe_tokens * m_foe1.to(foe_tokens.dtype)).sum(dim=1)

        own_mvs = mvs[:, :6, :]  # (B, 6, 4)
        own_mfeats = move_feats[:, :6, :, :]  # (B, 6, 4, 2)
        active0_mvs = (own_mvs * m_own0.long()).sum(dim=1)
        active1_mvs = (own_mvs * m_own1.long()).sum(dim=1)
        active0_mfeats = (own_mfeats * m_own0.unsqueeze(-1).to(obs.dtype)).sum(dim=1)
        active1_mfeats = (own_mfeats * m_own1.unsqueeze(-1).to(obs.dtype)).sum(dim=1)

        return BoardEncoding(
            context=context,
            own_tokens=own_tokens,
            active0=active0,
            active1=active1,
            foe0=foe0,
            foe1=foe1,
            active0_mvs=active0_mvs,
            active1_mvs=active1_mvs,
            active0_mfeats=active0_mfeats,
            active1_mfeats=active1_mfeats,
        )

    def value_from_context(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        value_logit = self.value_head(context).squeeze(-1)
        value = 2.0 * torch.sigmoid(value_logit) - 1.0
        return value, value_logit

    def preview_logits(self, enc: BoardEncoding) -> torch.Tensor:
        bsz = enc.own_tokens.shape[0]
        gathered = enc.own_tokens[:, self.preview_orders, :].reshape(
            bsz, self.preview_size, 4 * self.width
        )
        ctx = enc.context[:, None, :].expand(-1, self.preview_size, -1)
        return self.preview_head(torch.cat((ctx, gathered), dim=-1)).squeeze(-1)

    def _score_slot(
        self,
        slot_ctx: torch.Tensor,
        active_mvs: torch.Tensor,
        active_mfeats: torch.Tensor,
        own_tokens: torch.Tensor,
        move_head: nn.Module,
        switch_head: nn.Module,
        pass_head: nn.Module,
    ) -> torch.Tensor:
        bsz = slot_ctx.shape[0]
        mv_ctx = slot_ctx[:, None, :].expand(-1, _MOVE_SLOTS, -1)
        mv_in = torch.cat((mv_ctx, self.move_emb(active_mvs), active_mfeats), dim=-1)
        mv_logits = move_head(mv_in).reshape(bsz, _NUM_MOVE_ACTIONS)

        sw_ctx = slot_ctx[:, None, :].expand(-1, _NUM_SWITCH_ACTIONS, -1)
        sw_in = torch.cat((sw_ctx, own_tokens), dim=-1)
        sw_logits = switch_head(sw_in).squeeze(-1)

        pass_logit = pass_head(slot_ctx)
        return torch.cat((mv_logits, sw_logits, pass_logit), dim=-1)

    def slot0_logits(self, enc: BoardEncoding) -> torch.Tensor:
        slot0_ctx = self.slot0_ctx_proj(
            torch.cat((enc.context, enc.active0, enc.active1, enc.foe0, enc.foe1), dim=-1)
        )
        return self._score_slot(
            slot0_ctx,
            enc.active0_mvs,
            enc.active0_mfeats,
            enc.own_tokens,
            self.slot0_move_head,
            self.slot0_switch_head,
            self.slot0_pass_head,
        )

    def _encode_slot0_action(self, enc: BoardEncoding, a0: torch.Tensor) -> torch.Tensor:
        """Build a semantic `(B, W)` vector describing Slot 0's chosen move or switch."""
        bsz = a0.shape[0]
        a0_safe = a0.clamp(0, _PASS_ACTION)
        is_move = a0_safe < _NUM_MOVE_ACTIONS
        is_switch = (a0_safe >= _NUM_MOVE_ACTIONS) & (a0_safe < _PASS_ACTION)

        # Move branch: embed actual chosen move ID + target/mega variant
        mv_slot = (a0_safe // _VARIANTS_PER_MOVE).clamp(0, _MOVE_SLOTS - 1)
        mv_var = (a0_safe % _VARIANTS_PER_MOVE).clamp(0, _VARIANTS_PER_MOVE - 1)
        chosen_mv_id = enc.active0_mvs.gather(1, mv_slot.unsqueeze(-1)).squeeze(-1)
        mv_repr = self.move_emb(chosen_mv_id) + self.move_variant_emb(mv_var)

        # Switch branch: embed contextualized token of the chosen bench Pokémon
        sw_slot = (a0_safe - _NUM_MOVE_ACTIONS).clamp(0, _NUM_SWITCH_ACTIONS - 1)
        sw_idx = sw_slot.view(bsz, 1, 1).expand(-1, 1, self.width)
        sw_repr = enc.own_tokens.gather(1, sw_idx).squeeze(1) + self.switch_chosen_emb

        pass_repr = self.pass_chosen_emb.unsqueeze(0).expand(bsz, -1)
        out = torch.where(is_switch.unsqueeze(-1), sw_repr, pass_repr)
        out = torch.where(is_move.unsqueeze(-1), mv_repr, out)
        return out

    def slot1_logits(self, enc: BoardEncoding, a0: torch.Tensor) -> torch.Tensor:
        a0_repr = self._encode_slot0_action(enc, a0)
        slot1_ctx = self.slot1_ctx_proj(
            torch.cat(
                (enc.context, enc.active1, enc.active0, enc.foe0, enc.foe1, a0_repr), dim=-1
            )
        )
        return self._score_slot(
            slot1_ctx,
            enc.active1_mvs,
            enc.active1_mfeats,
            enc.own_tokens,
            self.slot1_move_head,
            self.slot1_switch_head,
            self.slot1_pass_head,
        )

    @torch.no_grad()
    def values_from_obs(self, obs_np: np.ndarray) -> np.ndarray:
        if len(obs_np) == 0:
            return np.empty(0, dtype=np.float32)
        x = torch.from_numpy(np.ascontiguousarray(obs_np, dtype=np.float32)).to(self.device)
        enc = self.encode_board(x)
        val, _ = self.value_from_context(enc.context)
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
        """Rank legal joint `(a0, a1)` pairs by exact autoregressive log-probability."""
        s = self.slot_size
        x = (
            torch.from_numpy(np.ascontiguousarray(obs_1d, dtype=np.float32))
            .unsqueeze(0)
            .to(self.device)
        )
        enc = self.encode_board(x)
        val, _ = self.value_from_context(enc.context)
        value_f = float(val[0].item())

        m0 = np.asarray(factored_mask[:s], dtype=bool)
        legal_a0 = np.flatnonzero(m0)
        if len(legal_a0) == 0:
            return [], np.empty(0, dtype=np.float64), value_f

        l0 = self.slot0_logits(enc)[0].cpu().numpy()
        l0_masked = np.where(m0, l0, -1e9)
        lp0 = l0_masked - float(np.max(l0_masked))
        lp0 = lp0 - float(np.log(np.sum(np.exp(lp0[m0]))))

        a0_order = legal_a0[np.argsort(-lp0[legal_a0], kind="stable")][: max(limit, 8)]
        n0 = len(a0_order)
        m1_batch = space.pair_mask(
            np.broadcast_to(factored_mask[None, :], (n0, len(factored_mask))),
            np.full(n0, min_switches, dtype=np.uint8),
            a0_order,
        )
        enc_rep = enc.expand_batch(n0)
        a0_t = torch.from_numpy(a0_order.astype(np.int64)).to(self.device)
        l1_batch = self.slot1_logits(enc_rep, a0_t).cpu().numpy()

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
    def top_preview_indices(
        self, obs_1d: np.ndarray, preview_mask: np.ndarray, *, limit: int = 10
    ) -> tuple[list[int], np.ndarray]:
        """Rank legal team-preview indices by policy probability (descending)."""
        x = (
            torch.from_numpy(np.ascontiguousarray(obs_1d, dtype=np.float32))
            .unsqueeze(0)
            .to(self.device)
        )
        enc = self.encode_board(x)
        logits = self.preview_logits(enc)[0].cpu().numpy()
        mask = np.asarray(preview_mask, dtype=bool)
        legal = np.flatnonzero(mask)
        if len(legal) == 0:
            return [], np.empty(0, dtype=np.float64)
        masked = np.where(mask, logits, -1e9)
        shifted = masked - float(np.max(masked[legal]))
        exp_legal = np.exp(shifted[legal])
        probs_legal = exp_legal / float(np.sum(exp_legal))
        order = np.argsort(-probs_legal, kind="stable")[:limit]
        return [int(legal[i]) for i in order], probs_legal[order].astype(np.float64)

    @torch.no_grad()
    def choose_preview_index(self, obs_1d: np.ndarray, preview_mask: np.ndarray) -> int:
        top, _ = self.top_preview_indices(obs_1d, preview_mask, limit=1)
        if not top:
            return 0
        return top[0]

    def checkpoint(self) -> dict[str, object]:
        return {
            "rust_policy_version": RUST_POLICY_VERSION,
            "format_id": self.format_id,
            "width": self.width,
            "layers": self.layers,
            "dex_sizes": self.dex_sizes,
            "rich_mon_width": RICH_MON_WIDTH,
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
        raise ValueError(
            f"unsupported rust policy checkpoint version {ckpt.get('rust_policy_version')}: {path}"
        )
    model = RustFactoredPolicy(
        width=int(ckpt["width"]),
        layers=int(ckpt["layers"]),
        format_id=str(ckpt.get("format_id", "championsvgc")),
        dex_sizes=dict(ckpt["dex_sizes"]),
    ).to(device)
    state = dict(ckpt["model"])
    # Backwards-compatible zero-padding when loading an older rich-obs checkpoint (e.g.
    # 80-float mon rows -> 88-float mon rows with appended VGC counter features). Since
    # new columns are appended at the tail of `scalars_no_mv`, zero-initialized input
    # columns produce bit-identical logits and values until fine-tuned.
    saved_w = state.get("mon_scalars.weight")
    target_w = model.mon_scalars.weight
    if saved_w is not None and saved_w.shape[0] == target_w.shape[0] and saved_w.shape[1] < target_w.shape[1]:
        padded = torch.zeros_like(target_w, device=saved_w.device, dtype=saved_w.dtype)
        padded[:, : saved_w.shape[1]] = saved_w
        state["mon_scalars.weight"] = padded
    model.load_state_dict(state)
    return model.eval()


def masked_log_prob_and_entropy(
    logits: torch.Tensor, mask: torch.Tensor, actions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute exact log-probability and entropy over masked categorical actions."""
    masked = logits.masked_fill(~mask, -1e9)
    log_probs = F.log_softmax(masked, dim=-1)
    probs = log_probs.exp() * mask.to(log_probs.dtype)
    entropy = -(probs * torch.where(mask, log_probs, torch.zeros_like(log_probs))).sum(dim=-1)
    chosen = log_probs.gather(-1, actions.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return chosen, entropy
