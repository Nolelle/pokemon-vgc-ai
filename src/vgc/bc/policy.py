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
torch or the checkpoint file is missing, or when the checkpoint's saved layout version is
NEITHER the running encoder's current `ENCODER_LAYOUT_VERSION` NOR a recognized entry in
`vgc.bc.encoding.LEGACY_ENCODER_LAYOUT_VERSIONS` -- callers (`score_orders`,
`vgc.agent.VgcPlayer.decide()`) already treat a `None` policy as "BC reranking disabled,
scores unchanged."

## Serving a legacy (`bc-encoding-v2`) checkpoint

A checkpoint like `bc_policy_v3.pt`/`bc_policy_v3sp.pt` was trained BEFORE `bc-encoding-
v4`'s preview-context fields existed, but is NOT refused just because the running
encoder has since moved to v4 (schema-4 replay/self-play data + `battle_state_record`/
`exchange_state_record` always emit the FULL v4-shaped state regardless of which
checkpoint is loaded -- the state producers never change). `load_bc_policy` sets
`BcPolicy.legacy_v2_layout` from the checkpoint's own saved `encoder_layout_version`
(never from the running module's current one), and every caller that turns a live/
simulated state into model input (`score_orders`, `position_value`,
`position_values_batch`) reads that flag via `_flatten_for_policy` to pick
`vgc.bc.encoding.flatten_state_v2` (legacy) or `flatten_state` (current) -- the SAME
`encode_state(record)` dict either way, just a different subset concatenated into the
final tensors (see `flatten_state_v2`'s docstring for why this doesn't need a second
`encode_state`). `vgc.bc.model.BcPolicyNet(legacy_v2_layout=True, ...)` mirrors this on
the model side (skips the preview-species trunk input its weights never had).

## `battle_state_record(battle, config) -> dict`

The live adapter from a poke-env `DoubleBattle` to the SAME schema-4 state shape
`vgc.replay_parse` emits (`{"state": {"our": ..., "opp": ..., "field": ...}}`), so it can
be fed straight into `vgc.bc.encoding.encode_state` -- the exact function the training
pipeline uses. Field-name drift between this adapter and `vgc.replay_parse`'s schema is
the single biggest silent-failure risk here (wrong-but-not-crashing feature values), so
every table this reuses (`_weather_str`/`_terrain_str`/`_screens_from`, `_resolve_species`)
is IMPORTED from `vgc.evaluator`/`vgc.replay_parse` rather than re-derived, and
`tests/test_bc_policy.py` asserts this adapter's output round-trips through the same
`encode_state` path as a real schema-4 fixture record.

Our own side is built from `battle.active_pokemon`/`battle.team` -- HP fractions, item,
ability, and all 4 moves are exactly known (it's our own team). The opponent side is
built from `battle.opponent_active_pokemon`/`battle.opponent_team` -- REVEALED
information only (`opp_mon.moves` as `revealed_moves`, item/ability `None` until poke-env
has actually seen them). Deliberately does NOT fill unrevealed opponent moves from
`vgc.sets.opponent_move_ids`'s set-prior corpus fill the way `vgc.evaluator` does for its
own Protect/switch heuristics: the model was trained on moves ACTUALLY USED in the replay
corpus, not prior-filled ones, so feeding it prior-filled moves here would shift its input
distribution away from what it learned on.

`preview_species`/`unseen_count` (schema 4) are filled by `_preview_species_for_our_side`/
`_preview_species_for_opp_side` -- see those functions' docstrings for why OUR side reads
`battle.team` (poke-env's `teampreview_team` is unreliable/effectively unpopulated in the
installed version) while the OPPONENT side reads `battle.teampreview_opponent_team`
(reliably populated from real `|poke|` lines).

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
from vgc.bc.encoding import (
    ENCODER_LAYOUT_VERSION,
    LEGACY_ENCODER_LAYOUT_VERSIONS,
    SLOT_FEATURE_DIM,
    STATE_SCALAR_DIM,
    STATE_SCALAR_DIM_V2,
    encode_state,
    flatten_state,
    flatten_state_v2,
)
from vgc.damage import PokemonState, to_id
from vgc.data import load_moves
from vgc.decision_trace import record_note
from vgc.evaluator import (
    ScoredOrder,
    _Context,
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
    # Which heads THIS checkpoint actually carries (`vgc.bc.model.BcPolicyNet`'s `heads`
    # arg it was built with) -- a checkpoint saved before the value head existed has no
    # `"heads"` key at all, treated as `("move", "target")` (see `load_bc_policy`), so
    # `has_value_head` correctly comes out False for it rather than guessing.
    heads: tuple[str, ...] = ("move", "target")
    # True for a checkpoint saved under a `vgc.bc.encoding.LEGACY_ENCODER_LAYOUT_VERSIONS`
    # entry (currently just `"bc-encoding-v2"`) -- see module docstring's "Serving a
    # legacy checkpoint" section. `load_bc_policy` sets this from the checkpoint's OWN
    # saved `encoder_layout_version`, never from the running encoder's current one.
    legacy_v2_layout: bool = False
    move_to_idx: dict[str, int] = field(default_factory=dict)
    target_to_idx: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.move_to_idx:
            self.move_to_idx = {token: idx for idx, token in enumerate(self.move_vocab)}
        if not self.target_to_idx:
            self.target_to_idx = {token: idx for idx, token in enumerate(self.target_vocab)}

    @property
    def has_value_head(self) -> bool:
        return "value" in self.heads


def _flatten_for_policy(policy: "BcPolicy", record: dict) -> tuple[np.ndarray, np.ndarray]:
    """`(index_array, scalar_array)` for `record` (a schema-4-shaped `{"state": {...}}`
    dict, from `battle_state_record`/`exchange_state_record`), using WHICHEVER layout
    `policy`'s checkpoint actually expects -- see module docstring's "Serving a legacy
    checkpoint" section. Every caller that flattens live/simulated state for a forward
    pass (`score_orders`, `position_value`, `position_values_batch`) goes through this
    instead of calling `flatten_state` directly, so a legacy checkpoint's narrower
    layout is never accidentally bypassed.
    """
    state = encode_state(record)
    if policy.legacy_v2_layout:
        return flatten_state_v2(state)
    return flatten_state(state)


_POLICY_CACHE: dict[str, BcPolicy | None] = {}


def load_bc_policy(checkpoint_path: str | Path) -> BcPolicy | None:
    """Load (and cache, per resolved path string) a `BcPolicy` from `checkpoint_path`.
    Never raises: torch missing, the file missing, a corrupt/unreadable checkpoint, or a
    saved layout version that's NEITHER the running encoder's current
    `ENCODER_LAYOUT_VERSION` NOR a recognized `LEGACY_ENCODER_LAYOUT_VERSIONS` entry all
    log ONE warning and return `None` -- callers treat that identically to "BC reranking
    disabled." A recognized legacy version is served via `BcPolicy.legacy_v2_layout`
    (see module docstring's "Serving a legacy checkpoint" section), not refused.
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
    is_legacy = saved_version in LEGACY_ENCODER_LAYOUT_VERSIONS
    if saved_version != ENCODER_LAYOUT_VERSION and not is_legacy:
        _LOGGER.warning(
            "BC policy disabled: checkpoint layout version %r does not match the running "
            "encoder's %r (and isn't a recognized legacy version %r) (checkpoint=%s) -- "
            "retrain or point bc_checkpoint_path at a compatible checkpoint",
            saved_version,
            ENCODER_LAYOUT_VERSION,
            sorted(LEGACY_ENCODER_LAYOUT_VERSIONS),
            checkpoint_path,
        )
        return None
    if is_legacy:
        _LOGGER.info(
            "BC policy: checkpoint %s uses legacy layout %r -- serving via "
            "BcPolicyNet(legacy_v2_layout=True)/flatten_state_v2",
            checkpoint_path,
            saved_version,
        )

    # Pre-value-head checkpoints (saved before this feature existed) have no "heads"
    # key at all -- treat that as exactly what it is, a move+target-only model, rather
    # than guessing it might have a value head it doesn't.
    heads = tuple(checkpoint.get("heads", ("move", "target")))
    scalar_dim = (STATE_SCALAR_DIM_V2 if is_legacy else STATE_SCALAR_DIM) + SLOT_FEATURE_DIM
    try:
        model = BcPolicyNet(
            species_vocab_size=len(checkpoint["species_vocab"]),
            item_vocab_size=len(checkpoint["item_vocab"]),
            ability_vocab_size=len(checkpoint["ability_vocab"]),
            move_vocab_size=len(checkpoint["move_vocab"]),
            target_vocab_size=len(checkpoint["target_vocab"]),
            scalar_dim=scalar_dim,
            legacy_v2_layout=is_legacy,
            heads=heads,
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
        heads=heads,
        legacy_v2_layout=is_legacy,
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


def _preview_species_for_our_side(battle: DoubleBattle) -> tuple[list[str], int]:
    """`(species_ids, unseen_count)` for OUR OWN previewed roster (schema-4 parity with
    `vgc.replay_parse`'s `preview_species`/`unseen_count`). Sourced from `battle.team`,
    NOT `battle.teampreview_team` -- the installed poke-env version populates `team`
    with all 6 of our own team members from the very first battle request (our own
    roster is fully known from the start, revealed or not), while `teampreview_team`
    is left empty in practice (never populated by the `|poke|`-line handler, which only
    ever appends to the OPPONENT's preview list -- see `_preview_species_for_opp_side`).
    `Pokemon.revealed` (poke-env's own "has this specific team slot ever been switched
    in this game" flag, set by `Pokemon.switch_in`) is exactly `vgc.replay_parse`'s
    `appeared_order` membership test for OUR side.
    """
    team = getattr(battle, "team", None) or {}
    species_ids: list[str] = []
    unseen_count = 0
    for mon in team.values():
        species_id, _mega = _resolve_species(to_id(mon.species) or "")
        if species_id:
            species_ids.append(species_id)
        if not getattr(mon, "revealed", False):
            unseen_count += 1
    return species_ids, unseen_count


def _preview_species_for_opp_side(battle: DoubleBattle) -> tuple[list[str], int]:
    """`(species_ids, unseen_count)` for the OPPONENT's previewed roster. Sourced from
    `battle.teampreview_opponent_team` (populated from real `|poke|` Team Preview reveal
    lines -- reliably available for the opponent side, unlike our own, see
    `_preview_species_for_our_side`). `battle.opponent_team` only ever contains species
    that have actually appeared (poke-env has no equivalent of our own side's
    always-fully-known `team`), so unseen is preview-species minus that appeared overlap.
    """
    preview = getattr(battle, "teampreview_opponent_team", None) or []
    species_ids: list[str] = []
    for mon in preview:
        species_id, _mega = _resolve_species(to_id(mon.species) or "")
        if species_id:
            species_ids.append(species_id)
    opp_team = getattr(battle, "opponent_team", None) or {}
    appeared = {_resolve_species(to_id(mon.species) or "")[0] for mon in opp_team.values()}
    unseen_count = sum(1 for species_id in species_ids if species_id not in appeared)
    return species_ids, unseen_count


def _side_dict(
    active: list[Pokemon | None],
    team: dict,
    side_conditions,
    preview_species: list[str],
    unseen_count: int,
) -> dict:
    conditions = set(_screens_from(side_conditions))
    if SideCondition.TAILWIND in side_conditions:
        conditions.add("tailwind")
    return {
        "active": [_active_mon_dict(mon) for mon in active],
        "bench": _bench_list(team, active),
        "side_conditions": sorted(conditions),
        "preview_species": preview_species,
        "unseen_count": unseen_count,
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

    our_preview_species, our_unseen_count = _preview_species_for_our_side(battle)
    opp_preview_species, opp_unseen_count = _preview_species_for_opp_side(battle)

    state = {
        "our": _side_dict(
            our_active, our_team, battle.side_conditions, our_preview_species, our_unseen_count
        ),
        "opp": _side_dict(
            opp_active,
            opp_team,
            battle.opponent_side_conditions,
            opp_preview_species,
            opp_unseen_count,
        ),
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
    `scored_orders` empty), or a policy that doesn't carry BOTH the move and target
    heads (e.g. `bc_checkpoint_path` pointing at a value-only checkpoint -- see
    `vgc.bc.model`'s `heads` config), returns `scored_orders` unchanged.
    """
    if (
        policy is None
        or not scored_orders
        or "move" not in policy.heads
        or "target" not in policy.heads
    ):
        return scored_orders

    top_k = max(1, config.bc_rerank_top_k)
    head = scored_orders[:top_k]
    tail = scored_orders[top_k:]

    record = battle_state_record(battle, config)
    index_array, scalar_array = _flatten_for_policy(policy, record)

    # The board state (hence the model's raw per-slot output) is IDENTICAL across every
    # candidate in `head` -- only which action each candidate proposes for that slot
    # differs. So the forward pass only needs 2 rows total (one per active slot), not
    # one per candidate; every candidate's score is a lookup into these 2 rows at its
    # own proposed move/target token indices. Still exactly "one batched forward pass."
    index_batch = np.tile(index_array, (2, 1))
    slot_onehots = np.eye(SLOT_FEATURE_DIM, dtype=np.float32)
    scalar_batch = np.concatenate([np.tile(scalar_array, (2, 1)), slot_onehots], axis=1)

    with torch.no_grad():
        move_logits, target_logits, _value_logit = policy.model(
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


# --- position_value: the outcome value head's read on a state, for vgc.search --------


def position_value(policy: BcPolicy | None, record_like_state: dict) -> float | None:
    """`sigmoid(value_logit)` -- the value head's estimated P(the state's `"our"` side
    wins), for a schema-3-shaped `{"state": {...}}` dict (the same contract
    `battle_state_record` produces and `encode_state` consumes). Returns `None` when
    `policy` is `None`, or the loaded checkpoint doesn't carry a value head at all (see
    `BcPolicy.has_value_head`) -- callers (`vgc.search`) treat `None` as "no value-head
    signal available this session," matching `load_bc_policy`'s other None-means-
    disabled cases. Never raises for the same reason those don't.

    The value head doesn't have its own "no slot" input -- every training sample (move,
    target, AND value alike) carries a real 2-dim active-slot one-hot
    (`vgc.bc.dataset`'s per-(record, slot) sampling), so calling the model with an
    all-zero one would be out-of-distribution for it. Instead this averages the
    prediction over BOTH slot one-hots (mirrors `score_orders`' 2-row batch for the same
    underlying reason: the board state is identical either way, only the slot bit
    differs) -- a whole-position judgment shouldn't arbitrarily favor one active slot's
    perspective, and the model never saw a genuinely slot-less input during training.
    """
    if policy is None or not policy.has_value_head:
        return None
    index_array, scalar_array = _flatten_for_policy(policy, record_like_state)
    index_batch = np.tile(index_array, (2, 1))
    slot_onehots = np.eye(SLOT_FEATURE_DIM, dtype=np.float32)
    scalar_batch = np.concatenate([np.tile(scalar_array, (2, 1)), slot_onehots], axis=1)
    with torch.no_grad():
        _move_logits, _target_logits, value_logit = policy.model(
            torch.from_numpy(index_batch), torch.from_numpy(scalar_batch)
        )
        probability = torch.sigmoid(value_logit).mean().item()
    return float(probability)


def position_values_batch(
    policy: BcPolicy | None, record_like_states: list[dict]
) -> list[float | None]:
    """Batched sibling of `position_value`: ONE forward pass scores ALL of
    `record_like_states` (each still gets `position_value`'s 2-row slot-averaging
    treatment -- see its docstring for why), instead of one forward pass per state.
    Returns a list index-aligned with `record_like_states`; every entry is `None` under
    the exact same conditions `position_value` would return `None` for each (policy
    missing or no value head) -- in that case every entry is `None`, no partial
    batching. `[]` in, `[]` out.

    Exists purely for `vgc.search.search_joint_orders`'s `use_value_head` path, which
    evaluates up to `search_our_candidates * len(responses)` positions per decision --
    measured at ~40ms/decision calling `position_value` once per position (a real local
    battle, `search_our_candidates=10` x ~8 responses), well over this feature's <5ms
    latency budget purely from per-call Python/torch dispatch overhead, not FLOPs (the
    2.2MB model itself is trivially fast). One batched call amortizes that overhead
    across every position at once.
    """
    if policy is None or not policy.has_value_head or not record_like_states:
        return [None] * len(record_like_states)
    n = len(record_like_states)
    index_arrays = []
    scalar_arrays = []
    for record in record_like_states:
        index_array, scalar_array = _flatten_for_policy(policy, record)
        index_arrays.append(index_array)
        scalar_arrays.append(scalar_array)
    index_stack = np.stack(index_arrays)
    scalar_stack = np.stack(scalar_arrays)
    # Each state gets 2 consecutive rows (slot-0 one-hot, slot-1 one-hot) -- see
    # position_value's docstring for why a slot-less input isn't an option.
    index_batch = np.repeat(index_stack, 2, axis=0)
    slot_onehots = np.tile(np.eye(SLOT_FEATURE_DIM, dtype=np.float32), (n, 1))
    scalar_batch = np.concatenate([np.repeat(scalar_stack, 2, axis=0), slot_onehots], axis=1)
    with torch.no_grad():
        _move_logits, _target_logits, value_logit = policy.model(
            torch.from_numpy(index_batch), torch.from_numpy(scalar_batch)
        )
        probabilities = torch.sigmoid(value_logit).reshape(n, 2).mean(dim=1)
    return [float(p) for p in probabilities.tolist()]


# --- exchange_state_record: vgc.search's post-exchange adapter for the value head ----


def exchange_state_record(
    our_states: list[PokemonState | None],
    opp_states: list[PokemonState | None],
    ctx: _Context,
    *,
    cache: dict | None = None,
) -> dict:
    """Build a schema-3-shaped `{"state": {...}}` dict (same contract
    `battle_state_record`/`encode_state` use) from a SIMULATED exchange's resulting
    `PokemonState`s (`vgc.search.resolve_exchange`'s `ExchangeResult.our_states`/
    `opp_states`, or `ctx.our_states`/`opp_states` themselves for the PRE-exchange
    position) -- the counterpart to `battle_state_record`'s live-battle adapter, sourced
    from damage-calc `PokemonState`s instead of poke-env `Pokemon` objects (a simulated
    exchange never produces the latter).

    `cache`: an optional dict a caller building MANY records against the same `ctx`
    (`vgc.search.search_joint_orders`'s hot path -- up to `search_our_candidates *
    len(responses)` calls per decision) can pass in and reuse across calls. Side
    conditions/field are 100% `ctx`-derived and never vary between exchanges within one
    decision; bench membership only varies when a candidate switches (so its cache key
    includes the active-species set) -- both were measured as the dominant per-call cost
    once the value head's own forward pass was batched (see `position_values_batch`'s
    docstring), since a bare Python loop + `_resolve_species` call per bench mon adds up
    across ~100 calls/decision even though each one is cheap in isolation. `None`
    (the default) computes everything fresh every call, correct but slower -- fine for
    a single-shot caller like `position_value`'s own v_before computation.

    v1 scope, deliberately approximate in two documented ways (not oversights):
    - `revealed_moves` for an active slot whose species DIDN'T change this exchange
      comes from `ctx`'s live Pokemon object (`.moves.keys()`, exactly known/revealed so
      far) -- a slot that SWITCHED during the exchange gets an empty `revealed_moves`
      list instead (no way to recover a fresh switch-in's revealed moveset from a bare
      `PokemonState`, and "nothing revealed yet" is directionally correct anyway for a
      mon that hasn't acted this hypothetical turn).
    - Bench comes from `ctx.battle.team`/`ctx.battle.opponent_team` (a benched mon takes
      no action during a single exchange, so its CURRENT live state is exact, not an
      approximation), excluding whichever species ended up active in `our_states`/
      `opp_states`.
    Field conditions (weather/terrain/trick room/side conditions) are `ctx`'s snapshot,
    NOT re-derived per exchange -- `vgc.search.resolve_exchange` already tracks its own
    `weather_for_exchange` override (a mega evolution granting Drought/Drizzle
    mid-exchange) separately for its own damage math; this adapter doesn't thread that
    override through, a known small gap for the (rare) mega-into-new-weather case.
    """

    def _resolved_active(state: PokemonState | None, live_mon) -> dict | None:
        if state is None:
            return None
        species_id, mega = _resolve_species(state.species_id)
        boosts = {
            stat: value
            for stat, value in (state.boosts or {}).items()
            if stat in ("atk", "def", "spa", "spd", "spe") and value
        }
        revealed_moves: list[str] = []
        if live_mon is not None and to_id(getattr(live_mon, "species", None)) == species_id:
            revealed_moves = sorted(
                {
                    move_id
                    for move_id in (to_id(mid) for mid in (getattr(live_mon, "moves", None) or {}))
                    if move_id
                }
            )
        max_hp = state.max_hp()
        return {
            "species": species_id,
            "hp_fraction": (state.hp_or_max() / max_hp) if max_hp else 0.0,
            "status": state.status,
            "boosts": boosts,
            "item": state.item,
            "ability": state.ability,
            "mega": mega,
            "revealed_moves": revealed_moves,
        }

    def _bench_uncached(team: dict, active_species: set[str]) -> list[dict]:
        bench: list[dict] = []
        for mon in (team or {}).values():
            if mon is None or getattr(mon, "fainted", False):
                continue
            species_id, _mega = _resolve_species(to_id(mon.species) or "")
            if species_id in active_species:
                continue
            bench.append(
                {
                    "species_id": species_id,
                    "hp_fraction": mon.current_hp_fraction,
                    "status": normalize_status(mon.status),
                }
            )
        return bench

    def _bench(side: str, team: dict, active_species: set[str]) -> list[dict]:
        if cache is None:
            return _bench_uncached(team, active_species)
        key = ("bench", side, frozenset(active_species))
        if key not in cache:
            cache[key] = _bench_uncached(team, active_species)
        return cache[key]

    our_active = [
        _resolved_active(our_states[i] if i < len(our_states) else None, ctx.our_pokemon[i])
        for i in range(2)
    ]
    opp_active = [
        _resolved_active(opp_states[i] if i < len(opp_states) else None, ctx.opp_pokemon[i])
        for i in range(2)
    ]
    our_active_species = {mon["species"] for mon in our_active if mon is not None}
    opp_active_species = {mon["species"] for mon in opp_active if mon is not None}

    if cache is not None and "conditions_and_field" in cache:
        our_conditions_sorted, opp_conditions_sorted, field = cache["conditions_and_field"]
    else:
        our_conditions = set(_screens_from(ctx.battle.side_conditions))
        if SideCondition.TAILWIND in ctx.battle.side_conditions:
            our_conditions.add("tailwind")
        opp_conditions = set(_screens_from(ctx.battle.opponent_side_conditions))
        if SideCondition.TAILWIND in ctx.battle.opponent_side_conditions:
            opp_conditions.add("tailwind")
        our_conditions_sorted = sorted(our_conditions)
        opp_conditions_sorted = sorted(opp_conditions)
        field = {
            "weather": ctx.weather,
            "terrain": ctx.terrain,
            "trick_room": ctx.trick_room,
            "turn": getattr(ctx.battle, "turn", 0),
        }
        if cache is not None:
            cache["conditions_and_field"] = (our_conditions_sorted, opp_conditions_sorted, field)

    # Schema 4: preview_species/unseen_count are 100% battle-global facts (like
    # side_conditions/field above), never varying between exchanges within one decision
    # -- see battle_state_record's docstring for why our own side reads `ctx.battle.team`
    # while the opponent side reads `ctx.battle.teampreview_opponent_team`.
    if cache is not None and "preview" in cache:
        our_preview_species, our_unseen_count, opp_preview_species, opp_unseen_count = cache[
            "preview"
        ]
    else:
        our_preview_species, our_unseen_count = _preview_species_for_our_side(ctx.battle)
        opp_preview_species, opp_unseen_count = _preview_species_for_opp_side(ctx.battle)
        if cache is not None:
            cache["preview"] = (
                our_preview_species,
                our_unseen_count,
                opp_preview_species,
                opp_unseen_count,
            )

    state = {
        "our": {
            "active": our_active,
            "bench": _bench("our", getattr(ctx.battle, "team", None), our_active_species),
            "side_conditions": our_conditions_sorted,
            "preview_species": our_preview_species,
            "unseen_count": our_unseen_count,
        },
        "opp": {
            "active": opp_active,
            "bench": _bench("opp", getattr(ctx.battle, "opponent_team", None), opp_active_species),
            "side_conditions": opp_conditions_sorted,
            "preview_species": opp_preview_species,
            "unseen_count": opp_unseen_count,
        },
        "field": field,
    }
    return {"state": state}
