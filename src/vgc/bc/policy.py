"""Deploy the trained BC v2 checkpoint (`vgc.bc.model.BcPolicyNet`) as a candidate
RE-RANKER blended with the heuristic evaluator/search scores -- NOT a standalone policy.
The heuristics stay load-bearing (they're the only thing that knows exact damage math);
the network only supplies learned judgment about which of the heuristic's own top
candidates a human-plausible player would actually pick.

Unlike `vgc.bc.dataset`/`model`/`train` (which hard-fail on import without the `train`
extra, torch), THIS module must stay importable -- and every function in it callable
without crashing -- even when torch isn't installed, because `vgc.agent` imports from it
unconditionally. Only `torch`/`vgc.bc.model.BcPolicyNet` themselves are guarded; anything
that doesn't need a loaded model (`battle_state_record`, the token-mapping helpers) works
identically either way. `load_bc_policy` degrades to returning `None` (never raises) when
torch or the checkpoint file is missing, or when the checkpoint's saved
`ENCODER_LAYOUT_VERSION` doesn't match the running encoder's -- callers (`score_orders`,
`vgc.agent.VgcPlayer.decide()`) already treat a `None` policy as "BC reranking disabled,
scores unchanged."

## `battle_state_record(battle, config) -> dict`

The live adapter from a poke-env `DoubleBattle` to the SAME schema-2 state shape
`vgc.replay_parse` emits (`{"state": {"our": ..., "opp": ..., "field": ...}}`), so it can
be fed straight into `vgc.bc.encoding.encode_state` -- the exact function the training
pipeline uses. Field-name drift between this adapter and `vgc.replay_parse`'s schema is
the single biggest silent-failure risk here (wrong-but-not-crashing feature values), so
every table this reuses (`_weather_str`/`_terrain_str`/`_screens_from`, `_resolve_species`)
is IMPORTED from `vgc.evaluator`/`vgc.replay_parse` rather than re-derived, and
`tests/test_bc_policy.py` asserts this adapter's output round-trips through the same
`encode_state` path as a real schema-2 fixture record.

Our own side is built from `battle.active_pokemon`/`battle.team` -- HP fractions, item,
ability, and all 4 moves are exactly known (it's our own team). The opponent side is
built from `battle.opponent_active_pokemon`/`battle.opponent_team` -- REVEALED
information only (`opp_mon.moves` as `revealed_moves`, item/ability `None` until poke-env
has actually seen them). Deliberately does NOT fill unrevealed opponent moves from
`vgc.sets.opponent_move_ids`'s set-prior corpus fill the way `vgc.evaluator` does for its
own Protect/switch heuristics: the model was trained on moves ACTUALLY USED in the replay
corpus, not prior-filled ones, so feeding it prior-filled moves here would shift its input
distribution away from what it learned on.

## `score_orders(policy, battle, scored_orders, config) -> list[ScoredOrder]`

Re-ranks ONLY the top `config.bc_rerank_top_k` heuristic-ranked candidates (cheap relative
to a full board search, and the heuristic's ranking below that cutoff is essentially
never going to be the actual best move anyway -- same cost/benefit reasoning as
`vgc.search`'s `search_our_candidates`). One batched forward pass computes both heads'
log-probabilities for BOTH of our active slots ONCE (2 rows: one per slot -- the current
board state, and hence the model's raw output, is identical across every candidate in
`scored_orders`, only WHICH action each candidate proposes for that slot differs, so
there is no need to re-run the network per candidate; each candidate's blended score is
just a lookup into those 2 rows at its own proposed move/target token indices). A
candidate whose move isn't in the checkpoint's vocab (shouldn't normally happen --
`data/champions/moves.json` is the same corpus both `vgc.bc.encoding` and every legal-move
enumerator draw from -- but defensively possible after a data update) skips the BC
adjustment entirely and keeps its heuristic score unchanged, noted in its breakdown.

Reproduces `vgc.search`'s earlier Fix-1 lesson (a partially-rescored top-K block must
never let the untouched tail "win" the whole ranking): after re-ranking the head block,
every tail entry is pinned strictly below the head block's new minimum score, in its
original relative order.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from poke_env.battle.double_battle import DoubleBattle
from poke_env.battle.field import Field
from poke_env.battle.move import Move
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.side_condition import SideCondition
from poke_env.player.battle_order import DoubleBattleOrder, SingleBattleOrder

from vgc.actions import describe_order
from vgc.bc.encoding import ENCODER_LAYOUT_VERSION, SLOT_FEATURE_DIM, encode_state, flatten_state
from vgc.damage import to_id
from vgc.data import load_moves
from vgc.decision_trace import record_note
from vgc.evaluator import (
    ScoredOrder,
    _SINGLE_TARGETS,
    _SPREAD_TARGETS_FOES_ONLY,
    _SPREAD_TARGETS_HITTING_ALLY,
    _screens_from,
    _terrain_str,
    _weather_str,
)
from vgc.models import PolicyConfig
from vgc.replay_parse import _resolve_species
from vgc.sets import normalize_item, normalize_status

try:
    import torch

    from vgc.bc.model import BcPolicyNet

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised via test_bc_policy.py's torch-less path
    _TORCH_AVAILABLE = False

_LOGGER = logging.getLogger(__name__)


@dataclass
class BcPolicy:
    """A loaded BC checkpoint, ready for `score_orders`. `model` is a `BcPolicyNet` in
    `eval()` mode on CPU (a 2.2MB MLP needs no GPU -- keeping ladder inference
    dependency-light matters more than the marginal speed a GPU would add). The vocab
    lists (and their derived `*_to_idx` maps) are the CHECKPOINT's own saved lists, not
    the running `vgc.bc.encoding` module's globals -- self-consistent with whatever this
    specific checkpoint was actually trained on, even in the (should-never-happen, since
    `ENCODER_LAYOUT_VERSION` is supposed to be bumped alongside any vocab change) case
    where the running module's vocab has silently drifted from the checkpoint's.
    """

    model: object  # BcPolicyNet -- typed loosely so this class stays definable torch-less
    species_vocab: list[str]
    move_vocab: list[str]
    item_vocab: list[str]
    ability_vocab: list[str]
    target_vocab: list[str]
    encoder_layout_version: str
    move_to_idx: dict[str, int] = field(default_factory=dict)
    target_to_idx: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.move_to_idx:
            self.move_to_idx = {token: idx for idx, token in enumerate(self.move_vocab)}
        if not self.target_to_idx:
            self.target_to_idx = {token: idx for idx, token in enumerate(self.target_vocab)}


_POLICY_CACHE: dict[str, BcPolicy | None] = {}


def load_bc_policy(checkpoint_path: str | Path) -> BcPolicy | None:
    """Load (and cache, per resolved path string) a `BcPolicy` from `checkpoint_path`.
    Never raises: torch missing, the file missing, a corrupt/unreadable checkpoint, or a
    saved `encoder_layout_version` that doesn't match this codebase's running
    `ENCODER_LAYOUT_VERSION` all log ONE warning and return `None` -- callers treat that
    identically to "BC reranking disabled."
    """
    cache_key = str(checkpoint_path)
    if cache_key in _POLICY_CACHE:
        return _POLICY_CACHE[cache_key]
    policy = _load_bc_policy_uncached(cache_key)
    _POLICY_CACHE[cache_key] = policy
    return policy


def _load_bc_policy_uncached(checkpoint_path: str) -> BcPolicy | None:
    if not _TORCH_AVAILABLE:
        _LOGGER.warning(
            "BC policy disabled: torch is not installed (checkpoint=%s) -- run "
            "`uv sync --extra train` to enable it",
            checkpoint_path,
        )
        return None
    path = Path(checkpoint_path)
    if not path.exists():
        _LOGGER.warning("BC policy disabled: checkpoint not found at %s", checkpoint_path)
        return None
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001 - a corrupt/foreign checkpoint must never crash a battle
        _LOGGER.warning(
            "BC policy disabled: failed to load checkpoint %s", checkpoint_path, exc_info=True
        )
        return None

    saved_version = checkpoint.get("encoder_layout_version")
    if saved_version != ENCODER_LAYOUT_VERSION:
        _LOGGER.warning(
            "BC policy disabled: checkpoint layout version %r does not match the running "
            "encoder's %r (checkpoint=%s) -- retrain or point bc_checkpoint_path at a "
            "compatible checkpoint",
            saved_version,
            ENCODER_LAYOUT_VERSION,
            checkpoint_path,
        )
        return None

    try:
        model = BcPolicyNet(
            species_vocab_size=len(checkpoint["species_vocab"]),
            item_vocab_size=len(checkpoint["item_vocab"]),
            ability_vocab_size=len(checkpoint["ability_vocab"]),
            move_vocab_size=len(checkpoint["move_vocab"]),
            target_vocab_size=len(checkpoint["target_vocab"]),
        )
        model.load_state_dict(checkpoint["state_dict"])
    except Exception:  # noqa: BLE001 - see above
        _LOGGER.warning(
            "BC policy disabled: failed to rebuild model from checkpoint %s",
            checkpoint_path,
            exc_info=True,
        )
        return None
    model.eval()

    return BcPolicy(
        model=model,
        species_vocab=list(checkpoint["species_vocab"]),
        move_vocab=list(checkpoint["move_vocab"]),
        item_vocab=list(checkpoint["item_vocab"]),
        ability_vocab=list(checkpoint["ability_vocab"]),
        target_vocab=list(checkpoint["target_vocab"]),
        encoder_layout_version=saved_version,
    )


# --- battle_state_record: the live adapter ------------------------------------------


def _active_mon_dict(pokemon: Pokemon | None) -> dict | None:
    if pokemon is None or pokemon.fainted:
        return None
    species_id, mega = _resolve_species(to_id(pokemon.species) or "")
    boosts = {
        stat: value
        for stat, value in (pokemon.boosts or {}).items()
        if stat in ("atk", "def", "spa", "spd", "spe") and value
    }
    revealed_moves = sorted(
        {move_id for move_id in (to_id(mid) for mid in (pokemon.moves or {}).keys()) if move_id}
    )
    return {
        "species": species_id,
        "hp_fraction": pokemon.current_hp_fraction,
        "status": normalize_status(pokemon.status),
        "boosts": boosts,
        # Exactly known for our own team; `None` (poke-env's "unknown_item" sentinel
        # normalizes to `None` via `normalize_item`) until actually revealed for the
        # opponent -- no special-casing needed, poke-env's own Pokemon fields already
        # carry this asymmetry, see module docstring.
        "item": normalize_item(pokemon.item),
        "ability": to_id(pokemon.ability) if pokemon.ability else None,
        "mega": mega,
        "revealed_moves": revealed_moves,
    }


def _bench_list(team: dict, active: list[Pokemon | None]) -> list[dict]:
    active_ids = {id(mon) for mon in active if mon is not None}
    bench: list[dict] = []
    for mon in (team or {}).values():
        if id(mon) in active_ids or mon.fainted:
            continue
        species_id, _mega = _resolve_species(to_id(mon.species) or "")
        bench.append(
            {
                "species_id": species_id,
                "hp_fraction": mon.current_hp_fraction,
                "status": normalize_status(mon.status),
            }
        )
    return bench


def _side_dict(active: list[Pokemon | None], team: dict, side_conditions) -> dict:
    conditions = set(_screens_from(side_conditions))
    if SideCondition.TAILWIND in side_conditions:
        conditions.add("tailwind")
    return {
        "active": [_active_mon_dict(mon) for mon in active],
        "bench": _bench_list(team, active),
        "side_conditions": sorted(conditions),
    }


def _field_dict(battle: DoubleBattle) -> dict:
    return {
        "weather": _weather_str(battle),
        "terrain": _terrain_str(battle),
        "trick_room": Field.TRICK_ROOM in battle.fields,
        "turn": battle.turn,
    }


def battle_state_record(battle: DoubleBattle, config: PolicyConfig | None = None) -> dict:
    """Build the SAME schema-2 `{"state": {...}}` shape `vgc.replay_parse` emits, from a
    LIVE `DoubleBattle`, ready for `vgc.bc.encoding.encode_state`. `config` is accepted
    for calling-convention consistency with the rest of this module (every battle-context
    builder in this codebase takes a `PolicyConfig`) but currently unused -- nothing about
    this adapter is a tunable judgment call, it's a mechanical field-name mapping.
    """
    del config  # unused -- see docstring
    our_active = list(battle.active_pokemon or [None, None])
    while len(our_active) < 2:
        our_active.append(None)
    opp_active = list(battle.opponent_active_pokemon or [None, None])
    while len(opp_active) < 2:
        opp_active.append(None)

    our_team = getattr(battle, "team", {}) or {}
    opp_team = getattr(battle, "opponent_team", {}) or {}

    state = {
        "our": _side_dict(our_active, our_team, battle.side_conditions),
        "opp": _side_dict(opp_active, opp_team, battle.opponent_side_conditions),
        "field": _field_dict(battle),
    }
    return {"state": state}


# --- action -> (move_token, target_token) mapping -----------------------------------


def _target_token(move_data: dict, move_target: int) -> str:
    """Mirrors `vgc.bc.encoding.encode_target`'s mapping, but FORWARD from a legal
    candidate order's own `move_target`/move data rather than backward from a replay
    protocol line -- see `vgc.evaluator._resolve_targets` (same target-kind constants,
    same position-number convention: -1/-2 = our slot 1/2, 1/2 = opponent slot 1/2).
    """
    target_kind = move_data.get("target")
    if target_kind in _SPREAD_TARGETS_HITTING_ALLY or target_kind in _SPREAD_TARGETS_FOES_ONLY:
        return "spread"
    if target_kind in _SINGLE_TARGETS:
        if move_target in (1, 2):
            return f"opp{move_target - 1}"
        if move_target in (-1, -2):
            return "ally"
        # No explicit target attached -- every enumerated candidate here is already
        # legal, so this shouldn't normally happen for a single-target move; fall back
        # to the same "first opposing slot" default `_resolve_targets` effectively uses.
        return "opp0"
    return "self_or_field"


def _single_tokens(
    single: SingleBattleOrder | None, policy: BcPolicy
) -> tuple[int, int, bool, str | None]:
    """`(move_token_idx, target_token_idx, skip, skip_reason)` for one slot's order.
    `skip=True` means the move id isn't in the checkpoint's vocab -- the caller must
    leave that WHOLE joint order's heuristic score untouched (see module docstring).
    """
    if single is None:
        return policy.move_to_idx["<pass>"], policy.target_to_idx["<none>"], False, None
    target = single.order
    if isinstance(target, Move):
        move_id = to_id(target.id)
        move_idx = policy.move_to_idx.get(move_id or "")
        if move_idx is None:
            return 0, 0, True, f"move {move_id!r} not in BC checkpoint vocab"
        move_data = load_moves().get(move_id)
        if move_data is None:
            return 0, 0, True, f"move {move_id!r} has no move data"
        target_str = _target_token(move_data, getattr(single, "move_target", 0) or 0)
        target_idx = policy.target_to_idx.get(target_str, policy.target_to_idx["<none>"])
        return move_idx, target_idx, False, None
    if isinstance(target, Pokemon):
        return policy.move_to_idx["<switch>"], policy.target_to_idx["<none>"], False, None
    # Pass (target is None or any other non-Move/non-Pokemon order payload).
    return policy.move_to_idx["<pass>"], policy.target_to_idx["<none>"], False, None


def _order_tokens(
    order: DoubleBattleOrder, policy: BcPolicy
) -> tuple[list[int], list[int], bool, str | None]:
    move_tokens: list[int] = []
    target_tokens: list[int] = []
    skip = False
    reason: str | None = None
    for single in (order.first_order, order.second_order):
        move_idx, target_idx, slot_skip, slot_reason = _single_tokens(single, policy)
        move_tokens.append(move_idx)
        target_tokens.append(target_idx)
        if slot_skip:
            skip = True
            reason = slot_reason
    return move_tokens, target_tokens, skip, reason


# --- score_orders: the actual reranker ------------------------------------------------


def score_orders(
    policy: BcPolicy | None,
    battle: DoubleBattle,
    scored_orders: list[ScoredOrder],
    config: PolicyConfig,
) -> list[ScoredOrder]:
    """Re-rank the top `config.bc_rerank_top_k` of `scored_orders` by blending in the BC
    model's log-probability of each candidate's joint action; see module docstring for
    the full algorithm. `policy is None` (torch/checkpoint unavailable, or
    `scored_orders` empty) returns `scored_orders` unchanged.
    """
    if policy is None or not scored_orders:
        return scored_orders

    top_k = max(1, config.bc_rerank_top_k)
    head = scored_orders[:top_k]
    tail = scored_orders[top_k:]

    record = battle_state_record(battle, config)
    index_array, scalar_array = flatten_state(encode_state(record))

    # The board state (hence the model's raw per-slot output) is IDENTICAL across every
    # candidate in `head` -- only which action each candidate proposes for that slot
    # differs. So the forward pass only needs 2 rows total (one per active slot), not
    # one per candidate; every candidate's score is a lookup into these 2 rows at its
    # own proposed move/target token indices. Still exactly "one batched forward pass."
    index_batch = np.tile(index_array, (2, 1))
    slot_onehots = np.eye(SLOT_FEATURE_DIM, dtype=np.float32)
    scalar_batch = np.concatenate([np.tile(scalar_array, (2, 1)), slot_onehots], axis=1)

    with torch.no_grad():
        move_logits, target_logits = policy.model(
            torch.from_numpy(index_batch), torch.from_numpy(scalar_batch)
        )
        move_logp = torch.log_softmax(move_logits, dim=-1)
        target_logp = torch.log_softmax(target_logits, dim=-1)

    reranked: list[ScoredOrder] = []
    for entry in head:
        move_tokens, target_tokens, skip, reason = _order_tokens(entry.order, policy)
        breakdown = dict(entry.breakdown)
        if skip:
            breakdown["bc_skipped"] = True
            breakdown["bc_skip_reason"] = reason
            reranked.append(ScoredOrder(order=entry.order, score=entry.score, breakdown=breakdown))
            continue
        bc_logprob = 0.0
        for slot in (0, 1):
            bc_logprob += float(move_logp[slot, move_tokens[slot]])
            bc_logprob += float(target_logp[slot, target_tokens[slot]])
        new_score = entry.score + config.bc_blend_weight * bc_logprob
        breakdown["bc_logprob"] = round(bc_logprob, 4)
        breakdown["bc_move_tokens"] = [policy.move_vocab[i] for i in move_tokens]
        breakdown["bc_target_tokens"] = [policy.target_vocab[i] for i in target_tokens]
        reranked.append(ScoredOrder(order=entry.order, score=new_score, breakdown=breakdown))

    # Stable sort: ties keep their pre-rerank relative order (mirrors search.py's own
    # sort convention).
    reranked.sort(key=lambda scored: scored.score, reverse=True)

    tail_result: list[ScoredOrder] = list(tail)
    if tail and reranked:
        # Fix-1 lesson (vgc.search's earlier regression): a partially-rescored top-K
        # block must never let the untouched tail "win" the ranking -- pin every tail
        # entry strictly below the reranked block's new minimum, in original order.
        floor = min(scored.score for scored in reranked) - 1.0
        tail_result = [
            ScoredOrder(order=entry.order, score=floor - i, breakdown=entry.breakdown)
            for i, entry in enumerate(tail)
        ]

    result = reranked + tail_result

    top_candidates = [
        {
            "order": describe_order(entry.order),
            "score": round(entry.score, 3),
            "bc_logprob": entry.breakdown.get("bc_logprob"),
        }
        for entry in reranked[: config.trace_top_k]
    ]
    record_note("bc_top_candidates", top_candidates)

    return result
