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

from vgc.damage import to_id
from vgc.data import load_moves

RICH_MON_WIDTH = 88
RICH_MON_IDS = 8
RICH_MON_SCALARS = RICH_MON_WIDTH - RICH_MON_IDS  # 80
RICH_SIDE_WIDTH = 20
RICH_FIELD_WIDTH = 28
RICH_GLOBAL_DIM = 2 * RICH_SIDE_WIDTH + RICH_FIELD_WIDTH  # 68
RICH_OBS_SIZE = 12 * RICH_MON_WIDTH + RICH_GLOBAL_DIM  # 1124
RICH_FIELD_TURN_IDX = 12 * RICH_MON_WIDTH + 2 * RICH_SIDE_WIDTH + 20
RUST_POLICY_VERSION = 3
SUPPORTED_RUST_POLICY_VERSIONS = (2, 3)

_MOVE_META_DIM = 32
_SPEED_FEAT_DIM = 4
_TYPES_ORDER = (
    "normal",
    "fire",
    "water",
    "electric",
    "grass",
    "ice",
    "fighting",
    "poison",
    "ground",
    "flying",
    "psychic",
    "bug",
    "rock",
    "ghost",
    "dragon",
    "dark",
    "steel",
    "fairy",
)
_TYPE_TO_IDX = {t: i for i, t in enumerate(_TYPES_ORDER)}
_PROTECT_MOVES = frozenset(
    {
        "protect",
        "detect",
        "spikyshield",
        "kingsshield",
        "banefulbunker",
        "obstruct",
        "silktrap",
        "burningbulwark",
        "wideguard",
        "quickguard",
        "matblock",
    }
)


def build_move_metadata_table(fmt: str = "championsvgc", num_rows: int | None = None) -> np.ndarray:
    """Build a frozen `(num_moves + 2, 32)` static move metadata table indexed by 1-based pikapro move ID.

    Row 0 (absent/padding) is all zeros. Provides immediate Day-1 knowledge of move type,
    category, base power, accuracy, priority, spread targeting, and Protect/contact/sound flags
    without needing millions of games per rare move.
    """
    names = pikapro.names("moves", fmt)
    rows = int(num_rows) if num_rows is not None else len(names) + 2
    table = np.zeros((rows, _MOVE_META_DIM), dtype=np.float32)
    moves_db = load_moves()
    for idx, raw_name in enumerate(names):
        row = idx + 1
        if row >= rows:
            break
        mid = to_id(raw_name)
        entry = moves_db.get(mid)
        if not entry:
            continue
        t_idx = _TYPE_TO_IDX.get(to_id(entry.get("type")))
        if t_idx is not None:
            table[row, t_idx] = 1.0
        cat = str(entry.get("category") or "").lower()
        if cat == "physical":
            table[row, 18] = 1.0
        elif cat == "special":
            table[row, 19] = 1.0
        else:
            table[row, 20] = 1.0
        bp = float(entry.get("basePower") or 0.0)
        table[row, 21] = min(bp / 150.0, 2.0)
        acc = entry.get("accuracy")
        table[row, 22] = 1.0 if acc is True or acc is None else min(float(acc) / 100.0, 1.0)
        prio = float(entry.get("priority") or 0.0)
        table[row, 23] = max(-1.0, min(1.0, prio / 5.0))
        target = str(entry.get("target") or "")
        table[row, 24] = 1.0 if target == "allAdjacentFoes" else 0.0
        table[row, 25] = 1.0 if target in ("allAdjacent", "all") else 0.0
        table[row, 26] = (
            1.0
            if target in ("self", "allySide", "adjacentAlly", "adjacentAllyOrSelf", "allies")
            else 0.0
        )
        flags = entry.get("flags") or {}
        table[row, 27] = 1.0 if flags.get("protect") else 0.0
        table[row, 28] = 1.0 if mid in _PROTECT_MOVES else 0.0
        table[row, 29] = 1.0 if flags.get("contact") else 0.0
        table[row, 30] = 1.0 if flags.get("sound") else 0.0
        table[row, 31] = 1.0 if flags.get("bypasssub") else 0.0
    return table


def _lookup_1based_id(names: list[str], target_id: str) -> int:
    for idx, raw in enumerate(names):
        if to_id(raw) == target_id:
            return idx + 1
    return -1

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
    """Equivariant encoder-only transformer with semantic autoregressive doubles heads.

    Supports both v2 (legacy `v3`/`v3b`/`v3c` checkpoints) and v3 (static move metadata,
    computed effective-speed & Trick Room speed-order cues, permutation-symmetric Team
    Preview head with LayerNorm, and configurable Transformer FFN width).
    """

    def __init__(
        self,
        width: int = 192,
        layers: int = 6,
        *,
        format_id: str = "championsvgc",
        dex_sizes: dict[str, int] | None = None,
        policy_version: int = RUST_POLICY_VERSION,
        ffn_mult: int | None = None,
    ) -> None:
        super().__init__()
        if policy_version not in SUPPORTED_RUST_POLICY_VERSIONS:
            raise ValueError(f"unsupported policy_version {policy_version}")
        self.policy_version = int(policy_version)
        self.width = int(width)
        self.layers = int(layers)
        self.ffn_mult = int(ffn_mult) if ffn_mult is not None else (3 if self.policy_version >= 3 else 2)
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
        # Precompute unique `(frozenset(lead2), frozenset(back2))` canonical representatives
        # (90 unique groups out of 360 permutations) so v3 `preview_head` runs 4x faster and
        # uses 4x less GPU memory with bit-exact identical output across all 360 orders.
        group_key_to_id: dict[tuple[tuple[int, ...], tuple[int, ...]], int] = {}
        unique_orders: list[np.ndarray] = []
        order_to_group = np.empty(len(preview_orders), dtype=np.int64)
        for oi, row in enumerate(preview_orders):
            key = (tuple(sorted(int(x) for x in row[:2])), tuple(sorted(int(x) for x in row[2:4])))
            gid = group_key_to_id.get(key)
            if gid is None:
                gid = len(unique_orders)
                group_key_to_id[key] = gid
                unique_orders.append(row)
            order_to_group[oi] = gid
        self.register_buffer(
            "preview_unique_orders",
            torch.from_numpy(np.asarray(unique_orders, dtype=np.int64)),
            persistent=False,
        )
        self.register_buffer(
            "preview_group_ids", torch.from_numpy(order_to_group), persistent=False
        )

        self.species_emb = nn.Embedding(self.dex_sizes["species"] + 2, width, padding_idx=0)
        self.item_emb = nn.Embedding(self.dex_sizes["items"] + 2, width, padding_idx=0)
        self.ability_emb = nn.Embedding(self.dex_sizes["abilities"] + 2, width, padding_idx=0)
        self.move_emb = nn.Embedding(self.dex_sizes["moves"] + 2, width, padding_idx=0)
        self.active_emb = nn.Embedding(4, width)
        self.side_emb = nn.Embedding(2, width)
        self.register_buffer(
            "side_ids", torch.tensor([0] * 6 + [1] * 6, dtype=torch.long), persistent=False
        )

        if self.policy_version >= 3:
            move_meta_np = build_move_metadata_table(format_id, self.move_emb.num_embeddings)
            self.register_buffer("move_meta", torch.from_numpy(move_meta_np), persistent=False)
            self.move_meta_proj = nn.Linear(_MOVE_META_DIM, width)
            self.speed_proj = nn.Linear(_SPEED_FEAT_DIM, width)
            item_names = pikapro.names("items", format_id)
            ability_names = pikapro.names("abilities", format_id)
            self._scarf_item_id = _lookup_1based_id(item_names, "choicescarf")
            self._chlorophyll_id = _lookup_1based_id(ability_names, "chlorophyll")
            self._swiftswim_id = _lookup_1based_id(ability_names, "swiftswim")
            self._sandrush_id = _lookup_1based_id(ability_names, "sandrush")
            self._slushrush_id = _lookup_1based_id(ability_names, "slushrush")
        else:
            self.move_meta_proj = None
            self.speed_proj = None
            self._scarf_item_id = -1
            self._chlorophyll_id = -1
            self._swiftswim_id = -1
            self._sandrush_id = -1
            self._slushrush_id = -1

        # Remove per-slot PP/disabled from mon_scalars so the Transformer token is 100%
        # permutation-invariant to move slot order; PP/disabled feed directly into move_head
        # AND into the board token as a pooled move-state sum (`move_state_proj`).
        self.mon_scalars = nn.Linear(RICH_MON_SCALARS - 8, width)
        self.move_state_proj = nn.Linear(2, width)
        self.global_proj = nn.Linear(RICH_GLOBAL_DIM, width)

        block = nn.TransformerEncoderLayer(
            width,
            4,
            self.ffn_mult * width,
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

        if self.policy_version >= 3:
            self.preview_head = nn.Sequential(
                nn.LayerNorm(5 * width),
                nn.Linear(5 * width, width),
                nn.GELU(),
                nn.Linear(width, 1),
            )
        else:
            self.preview_head = nn.Sequential(
                nn.Linear(5 * width, width),
                nn.GELU(),
                nn.Linear(width, 1),
            )

    @property
    def device(self) -> torch.device:
        return self.species_emb.weight.device

    def _move_repr(self, mvs: torch.Tensor) -> torch.Tensor:
        emb = self.move_emb(mvs)
        if self.move_meta_proj is None:
            return emb
        meta = self.move_meta[mvs]
        present = (mvs > 0).unsqueeze(-1).to(emb.dtype)
        return emb + self.move_meta_proj(meta) * present

    def _compute_speed_features(
        self,
        mons: torch.Tensor,
        glob: torch.Tensor,
        it: torch.Tensor,
        ab: torch.Tensor,
        act: torch.Tensor,
    ) -> torch.Tensor:
        """Compute fog-safe effective Speed & Trick Room speed-order features on GPU `(B, 12, 4)`."""
        present = mons[:, :, 8]
        # Exact Speed for own mons (`col 53`), fog-safe Level-50 estimate (`base + 36`) for foes (`col 59`).
        own_raw = mons[:, :6, 53] * 255.0
        foe_raw = mons[:, 6:12, 59] * 255.0 + 36.0
        raw_spe = torch.cat((own_raw, foe_raw), dim=1)

        # Stat stage multiplier from Speed boost (`col 27`, stored as stage / 6.0).
        stage = torch.round(mons[:, :, 27] * 6.0)
        stage_mult = torch.where(stage >= 0.0, (2.0 + stage) * 0.5, 2.0 / (2.0 - stage))

        # Paralysis (`col 18`), side Tailwind (`glob 3` own, `glob 23` foe), Choice Scarf, weather abilities.
        par_mult = torch.where(mons[:, :, 18] > 0.5, 0.5, 1.0)
        own_tw = glob[:, 3:4].expand(-1, 6)
        foe_tw = glob[:, 23:24].expand(-1, 6)
        tw_mult = 1.0 + torch.cat((own_tw, foe_tw), dim=1)

        scarf_mult = (
            torch.where(it == self._scarf_item_id, 1.5, 1.0)
            if self._scarf_item_id > 0
            else torch.ones_like(raw_spe)
        )
        sun = ((glob[:, 40] > 0.5) | (glob[:, 44] > 0.5)).unsqueeze(-1)
        rain = ((glob[:, 41] > 0.5) | (glob[:, 45] > 0.5)).unsqueeze(-1)
        sand = (glob[:, 42] > 0.5).unsqueeze(-1)
        snow = (glob[:, 43] > 0.5).unsqueeze(-1)
        w_boost = torch.zeros_like(it, dtype=torch.bool)
        if self._chlorophyll_id > 0:
            w_boost = w_boost | (sun & (ab == self._chlorophyll_id))
        if self._swiftswim_id > 0:
            w_boost = w_boost | (rain & (ab == self._swiftswim_id))
        if self._sandrush_id > 0:
            w_boost = w_boost | (sand & (ab == self._sandrush_id))
        if self._slushrush_id > 0:
            w_boost = w_boost | (snow & (ab == self._slushrush_id))
        weather_mult = torch.where(w_boost, 2.0, 1.0)

        eff_spe = (raw_spe * stage_mult * par_mult * tw_mult * scarf_mult * weather_mult) * present

        # Normalized active move-order rank in [0, 1] (automatically inverted under Trick Room!).
        # Exclude fainted Pokémon (`col 11 >= 0.5`): `PlayerView` keeps `active_slot > 0`
        # on fainted slots during force_switch and 2v1/1v1 endgames, whereas `encode_state_rich`
        # clears `active_slot` on fainted mons.
        is_active = (act > 0) & (mons[:, :, 11] < 0.5)
        is_tr = glob[:, 53:54]  # (B, 1)
        # Signed speed for ordering: higher moves first normally; lower moves first in Trick Room.
        order_key = torch.where(is_tr > 0.5, -eff_spe, eff_spe)
        # Pairwise comparison across the 12 rows, masked to other active mons.
        other_active = is_active.unsqueeze(1)  # (B, 1, 12)
        faster_than = (order_key.unsqueeze(2) > order_key.unsqueeze(1)) & other_active
        tied_with = (order_key.unsqueeze(2) == order_key.unsqueeze(1)) & other_active
        n_other_active = (is_active.to(eff_spe.dtype).sum(dim=1, keepdim=True) - 1.0).clamp(min=1.0)
        # Exclude self from tied_with by subtracting 1 when active.
        rank = (
            faster_than.to(eff_spe.dtype).sum(dim=2)
            + 0.5 * (tied_with.to(eff_spe.dtype).sum(dim=2) - is_active.to(eff_spe.dtype)).clamp(min=0.0)
        ) / n_other_active
        rank = rank * is_active.to(eff_spe.dtype)

        spe_norm = (eff_spe / 300.0).clamp(max=3.0)
        spe_log = torch.log1p(eff_spe.clamp(min=0.0)) / 6.0
        tr_rank = rank * is_tr
        return torch.stack((spe_norm, spe_log, rank, tr_rank), dim=-1)

    def encode_board(self, obs: torch.Tensor) -> BoardEncoding:
        """Encode a batch of 1,124-float Rust `rich` observations into `BoardEncoding`."""
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
            (self._move_repr(mvs) + self.move_state_proj(move_feats)) * move_present
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
        if self.speed_proj is not None:
            speed_feats = self._compute_speed_features(mons, glob, it, ab, act)
            mon_tokens = mon_tokens + self.speed_proj(speed_feats)
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
        if self.policy_version >= 3:
            # Evaluate the permutation-invariant head only on the 90 unique `(lead_pair, back_pair)`
            # groups, then expand to all 360 permutations (4x faster, 4x less GPU RAM, bit-exact).
            gathered = enc.own_tokens[:, self.preview_unique_orders, :]  # (B, 90, 4, W)
            n_groups = gathered.shape[1]
            ctx = enc.context[:, None, :].expand(-1, n_groups, -1)
            l0 = gathered[:, :, 0, :]
            l1 = gathered[:, :, 1, :]
            b0 = gathered[:, :, 2, :]
            b1 = gathered[:, :, 3, :]
            feats = torch.cat(
                (ctx, l0 + l1, (l0 - l1).abs(), b0 + b1, (b0 - b1).abs()), dim=-1
            )
            group_logits = self.preview_head(feats).squeeze(-1)  # (B, 90)
            return group_logits[:, self.preview_group_ids]
        gathered = enc.own_tokens[:, self.preview_orders, :]
        ctx = enc.context[:, None, :].expand(-1, self.preview_size, -1)
        return self.preview_head(
            torch.cat((ctx, gathered.reshape(bsz, self.preview_size, 4 * self.width)), dim=-1)
        ).squeeze(-1)

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
        mv_in = torch.cat((mv_ctx, self._move_repr(active_mvs), active_mfeats), dim=-1)
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

        # Move branch: embed actual chosen move ID + static metadata + target/mega variant
        mv_slot = (a0_safe // _VARIANTS_PER_MOVE).clamp(0, _MOVE_SLOTS - 1)
        mv_var = (a0_safe % _VARIANTS_PER_MOVE).clamp(0, _VARIANTS_PER_MOVE - 1)
        chosen_mv_id = enc.active0_mvs.gather(1, mv_slot.unsqueeze(-1)).squeeze(-1)
        mv_repr = self._move_repr(chosen_mv_id) + self.move_variant_emb(mv_var)

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
            "rust_policy_version": self.policy_version,
            "format_id": self.format_id,
            "width": self.width,
            "layers": self.layers,
            "ffn_mult": self.ffn_mult,
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
    ver = ckpt.get("rust_policy_version")
    if ver not in SUPPORTED_RUST_POLICY_VERSIONS:
        raise ValueError(
            f"unsupported rust policy checkpoint version {ver}: {path}"
        )
    ffn_mult = ckpt.get("ffn_mult")
    model = RustFactoredPolicy(
        width=int(ckpt["width"]),
        layers=int(ckpt["layers"]),
        format_id=str(ckpt.get("format_id", "championsvgc")),
        dex_sizes=dict(ckpt["dex_sizes"]),
        policy_version=int(ver),
        ffn_mult=int(ffn_mult) if ffn_mult is not None else (3 if int(ver) >= 3 else 2),
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


def upgrade_v2_to_v3(src: RustFactoredPolicy) -> RustFactoredPolicy:
    """Function-preserving warm-start upgrade from a `policy_version=2` model to `v3`.

    Copies the entire Transformer trunk, embeddings, value head, and Slot 0/1 heads
    bit-for-bit, and zero-initializes the new additive `move_meta_proj` and `speed_proj`
    projections so Day-0 in-battle logits and win probabilities are bit-exact identical
    to `src` while gradients flow into both new branches starting on Step 1.
    """
    if src.policy_version != 2:
        raise ValueError(f"upgrade_v2_to_v3 expects policy_version=2, got {src.policy_version}")
    dst = RustFactoredPolicy(
        width=src.width,
        layers=src.layers,
        format_id=src.format_id,
        dex_sizes=src.dex_sizes,
        policy_version=3,
        ffn_mult=src.ffn_mult,
    ).to(src.device)
    src_state = src.state_dict()
    dst_state = dst.state_dict()
    for key, tensor in src_state.items():
        if key.startswith("preview_head."):
            continue
        if key in dst_state and dst_state[key].shape == tensor.shape:
            dst_state[key] = tensor.detach().clone().to(dst.device)
    dst.load_state_dict(dst_state)
    with torch.no_grad():
        assert dst.move_meta_proj is not None and dst.speed_proj is not None
        dst.move_meta_proj.weight.zero_()
        dst.move_meta_proj.bias.zero_()
        dst.speed_proj.weight.zero_()
        dst.speed_proj.bias.zero_()
    return dst.eval()


def fit_v3_preview_head_from_teacher(
    student: RustFactoredPolicy,
    teacher: RustFactoredPolicy,
    teams: list[str],
    *,
    boards: int = 8192,
    steps: int = 120,
    batch_size: int = 256,
    lr: float = 3e-3,
    seed: int = 37401,
    threads: int = 8,
) -> dict[str, float]:
    """Fit ONLY `student.preview_head` to match `teacher`'s Team Preview policy.

    Freezes every parameter outside `preview_head` so in-battle logits and values stay
    bit-exact identical to `teacher`, and minimizes `KL(p_teacher || p_student)` over
    real Rust Team Preview boards (automatically learning the optimal 90-group
    symmetrized projection of a 360-way v2 teacher).
    """
    dev = student.device
    s = student.slot_size
    pool = rl.TeamPool(teams, student.format_id)
    n_envs = min(256, max(16, boards // 4))
    env = rl.VecEnv(
        student.format_id,
        n=n_envs,
        teams=pool,
        threads=threads,
        seed=seed,
        opponent="external",
        ots=0.0,
        preview="learned",
        masks="factored",
        observation="rich",
        autoreset="next_step",
        side_randomization=True,
    )
    obs_chunks: list[np.ndarray] = []
    mask_chunks: list[np.ndarray] = []
    collected = 0
    while collected < boards:
        step = env.reset()
        for slot in (0, 1):
            pv_m = step.masks[:, slot, 2 * s :]
            has_pv = pv_m.any(axis=-1)
            if has_pv.any():
                idx = np.flatnonzero(has_pv)
                obs_chunks.append(step.obs[idx, slot].copy())
                mask_chunks.append(pv_m[idx].copy())
                collected += len(idx)
                if collected >= boards:
                    break

    obs_np = np.concatenate(obs_chunks, axis=0)[:boards]
    mask_np = np.concatenate(mask_chunks, axis=0)[:boards]

    # Precompute frozen `enc` (context + own_tokens) and teacher target probabilities once!
    teacher.eval()
    student.eval()
    ctx_list: list[torch.Tensor] = []
    own_list: list[torch.Tensor] = []
    tgt_list: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(obs_np), 256):
            xb = torch.from_numpy(obs_np[start : start + 256]).to(dev)
            mb = torch.from_numpy(mask_np[start : start + 256]).to(dev)
            enc_s = student.encode_board(xb)
            enc_t = teacher.encode_board(xb.to(teacher.device))
            t_logits = teacher.preview_logits(enc_t).to(dev).masked_fill(~mb, -1e9)
            t_probs = F.softmax(t_logits, dim=-1)
            ctx_list.append(enc_s.context.detach())
            own_list.append(enc_s.own_tokens.detach())
            tgt_list.append(t_probs.detach())
    ctx_all = torch.cat(ctx_list, dim=0)
    own_all = torch.cat(own_list, dim=0)
    tgt_all = torch.cat(tgt_list, dim=0)
    mask_all = torch.from_numpy(mask_np).to(dev)

    saved_req = {name: p.requires_grad for name, p in student.named_parameters()}
    for name, p in student.named_parameters():
        p.requires_grad_(name.startswith("preview_head."))

    opt = torch.optim.Adam(student.preview_head.parameters(), lr=float(lr))
    student.preview_head.train()
    rng_t = torch.Generator(device="cpu").manual_seed(seed)
    n_total = len(obs_np)
    pv_micro = 64 if dev.type == "mps" else batch_size
    last_kl = 0.0
    try:
        for _ in range(steps):
            perm = torch.randint(0, n_total, (min(batch_size, n_total),), generator=rng_t)
            opt.zero_grad(set_to_none=True)
            step_kl = 0.0
            total_rows = len(perm)
            for sub_s in range(0, total_rows, pv_micro):
                sub = perm[sub_s : sub_s + pv_micro].to(dev)
                w = len(sub) / total_rows
                sub_enc = BoardEncoding(
                    context=ctx_all[sub],
                    own_tokens=own_all[sub],
                    active0=ctx_all[sub],
                    active1=ctx_all[sub],
                    foe0=ctx_all[sub],
                    foe1=ctx_all[sub],
                    active0_mvs=torch.empty(0, device=dev, dtype=torch.long),
                    active1_mvs=torch.empty(0, device=dev, dtype=torch.long),
                    active0_mfeats=torch.empty(0, device=dev),
                    active1_mfeats=torch.empty(0, device=dev),
                )
                s_logits = student.preview_logits(sub_enc).masked_fill(~mask_all[sub], -1e9)
                s_lp = F.log_softmax(s_logits, dim=-1)
                t_p = tgt_all[sub]
                t_lp = torch.log(t_p.clamp(min=1e-12))
                kl = (t_p * (t_lp - s_lp)).masked_fill(~mask_all[sub], 0.0).sum(dim=-1).mean()
                (kl * w).backward()
                step_kl += float(kl.detach().item()) * w
            torch.nn.utils.clip_grad_norm_(student.preview_head.parameters(), 1.0)
            opt.step()
            last_kl = step_kl
    finally:
        for name, p in student.named_parameters():
            p.requires_grad_(saved_req[name])
        student.eval()

    # Measure final canonical 90-group top-1 agreement against teacher.
    with torch.no_grad():
        eval_n = min(1024, n_total)
        sub_enc = BoardEncoding(
            context=ctx_all[:eval_n],
            own_tokens=own_all[:eval_n],
            active0=ctx_all[:eval_n],
            active1=ctx_all[:eval_n],
            foe0=ctx_all[:eval_n],
            foe1=ctx_all[:eval_n],
            active0_mvs=torch.empty(0, device=dev, dtype=torch.long),
            active1_mvs=torch.empty(0, device=dev, dtype=torch.long),
            active0_mfeats=torch.empty(0, device=dev),
            active1_mfeats=torch.empty(0, device=dev),
        )
        s_logits = student.preview_logits(sub_enc).masked_fill(~mask_all[:eval_n], -1e9)
        s_top = student.preview_group_ids[s_logits.argmax(dim=-1)]
        t_top = student.preview_group_ids[tgt_all[:eval_n].argmax(dim=-1)]
        agree = float((s_top == t_top).float().mean().item())
    return {"preview_kl": round(last_kl, 5), "preview_top1_group_agree": round(agree, 4), "boards": float(n_total)}


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
