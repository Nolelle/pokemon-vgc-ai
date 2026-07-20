"""Pure encoders from a `data/bc/decisions.jsonl` record (`vgc.replay_parse`'s schema,
schema 2+ required -- see that module's docstring) to fixed-size numpy arrays. NO torch
import here -- keep this importable without the `train` extra (`uv sync --extra train`);
`vgc.bc.dataset`/`model`/`train` do the torch-dependent work on top of what this module
produces.

## v2 changes from v1 (`ENCODER_LAYOUT_VERSION` bumped `"bc-encoding-v1"` ->
`"bc-encoding-v2"`, additive redesign)

v1 only encoded the 4 active slots' species/HP/status/boosts/mega plus a bench SUMMARY
(count + mean HP). v2 adds: per-active-slot item, ability, and up to 4 revealed moves
(now available on `vgc.replay_parse`'s schema-2 active-mon dicts); per-side bench
IDENTITY (species + HP, not just a count/mean -- also now available on schema 2's bench
entries, though bench `status` is present in the data but NOT yet consumed by this
layout version); and a second prediction target, `encode_target`, for WHERE a move
goes (not just which move).

## v4 changes from v2/v3 (`ENCODER_LAYOUT_VERSION` bumped to `"bc-encoding-v4"`,
additive; there was no separate v3 state-encoding layout -- the outcome-value head
`encode_value`/`vgc.bc.model`'s `value` head added in that phase read `record["won"]`
directly and never touched `encode_state`'s output, so the layout version wasn't bumped
then, see that section below)

Everything up to v2 is blind to anything that hasn't appeared on the field yet: neither
side's previewed six nor unrevealed bench members are in the state at all, even though
`vgc.replay_parse`'s schema 4 now carries exactly that per-battle context
(`preview_species`/`unseen_count`, see that module's docstring). v4 adds, per side:

- `species_idx_preview` (new INDEX key, 12 total: 6 our + 6 opp): `SPECIES_VOCAB` index
  of each of that side's up-to-6 previewed species (Team Preview order), `"<pad>"` for
  an unfilled/missing preview slot -- shares `species_embedding` with the active/bench
  species indices (`vgc.bc.model` mean-pools each side's 6 slots into one vector,
  `"<pad>"`-masked, mirroring how `move_idx` is already mean-pooled per active slot).
  This is deliberately RAW species identity, not a derived archetype label -- the
  network is expected to learn associations (e.g. "these two species together usually
  mean a rain team") from the embedding itself, the same way it already does for
  active/bench species.
- `alive_known_count` (2,), `alive_known_mean_hp` (2,), `preview_unseen_count` (2,) (new
  SCALAR keys, `[our, opp]` each): resource-state summaries computable directly from the
  snapshot's own `active`/`bench`/`unseen_count` fields, with NO dependency on
  `vgc.evaluator`/`vgc.gameplan` (this module stays pure/torch-free/battle-free by
  design -- see the module docstring's opening paragraph). "Known" mons are exactly
  what's already in the state (active non-empty slots + bench entries -- bench, per
  `vgc.replay_parse`'s docstring, only ever lists species that have actually appeared
  and aren't fainted), so "alive known" needs no new data, just a sum/mean over what's
  already there.

## v3 (no state-layout change)

Phase 3's outcome-value head (`encode_value`, `vgc.bc.model`'s `"value"` head) was added
without bumping `ENCODER_LAYOUT_VERSION` -- see that function's docstring below for why
its compatibility is independent of the state-encoding layout.

## Vocabularies

Built deterministically at import time from COMMITTED data -- nothing is persisted by
this module itself:

- `SPECIES_VOCAB` = `["<pad>", "<unk>"] + sorted(vgc.data.load_species().keys())` --
  unchanged from v1. `"<pad>"` (index 0) is used for BOTH an empty active slot and an
  empty (padded) bench slot; `"<unk>"` (index 1) is any species string that isn't a
  recognized champions-mod species id (defensive, shouldn't normally happen).
- `MOVE_VOCAB` = `["<pad>", "<switch>", "<pass>", "<unk>"] + sorted(vgc.data.
  load_moves().keys())`. v2 adds `"<pad>"` (index 0, NEW vs v1) since this same vocab
  now ALSO backs the per-slot revealed-moves list (`encode_state`'s `move_idx`), which
  needs a padding token for slots with fewer than 4 revealed moves -- `"<switch>"`/
  `"<pass>"` remain the two non-move action kinds (`encode_action`); `"<unk>"` remains
  RESERVED for symmetry but never assigned by `encode_action` (an unrecognized move id
  returns `None` instead, so a caller skips that sample -- see `encode_action`'s
  docstring). NOTE: because `"<pad>"` was inserted, every `MOVE_VOCAB` index differs
  from v1's -- this is exactly why checkpoints save their own vocab list rather than
  assuming a fixed mapping (see `ENCODER_LAYOUT_VERSION`).
- `ITEM_VOCAB` = `["<unk>", "<none>"] + sorted(vgc.data.load_items().keys())`.
  `"<unk>"` is assigned whenever a mon's `item` field is `None` -- **which, given
  `vgc.replay_parse`'s CURRENT schema, covers both "never revealed" and "revealed then
  consumed/removed" (`-enditem`/Trick-away), since both collapse to the same `None` in
  the state dict.** `"<none>"` is reserved in the vocab (per this feature's spec) but is
  NOT currently reachable from real data -- distinguishing "confirmed itemless" from
  "unrevealed" would need a `vgc.replay_parse` schema change beyond what schema 2 added
  (e.g. a tri-state or a sentinel for "confirmed empty"), which was out of scope here.
  This is a known, documented limitation, not a bug.
- `ABILITY_VOCAB` = `["<unk>"] + sorted({to_id(name) for species in
  vgc.data.load_species().values() for name in species["abilities"].values()})`.
  `"<unk>"` covers an unrevealed ability (a mon's `ability` field is `None`) the same
  way `SPECIES_VOCAB`'s `"<unk>"` covers an unrecognized species -- no `"<none>"`
  equivalent here since abilities are never "revealed as absent" the way an item can be
  consumed (every Pokemon has exactly one active ability at all times).
- `TARGET_VOCAB` = `["opp0", "opp1", "ally", "self_or_field", "spread", "<none>"]` --
  see `encode_target`.

## `encode_state(record) -> dict[str, np.ndarray]` layout

Per-side-relative (`record["state"]["our"]`/`["opp"]`), fixed-size arrays. 4-slot arrays
are always `[our_slot0, our_slot1, opp_slot0, opp_slot1]`; 8-slot bench arrays are
always `[our_bench0..3, opp_bench0..3]` (each side's own bench entries in
`vgc.replay_parse`'s `appeared_order` order, padded/truncated to exactly 4 -- in this
bring-6-pick-4 format at most 2 bench slots are ever realistically populated per side,
but the array is sized 4 regardless, per this feature's spec, for headroom).

| key                     | shape  | dtype   | meaning |
|-------------------------|--------|---------|---------|
| `species_idx_active`    | (4,)   | int64   | `SPECIES_VOCAB` index per active slot |
| `species_idx_bench`     | (8,)   | int64   | `SPECIES_VOCAB` index per bench slot, `"<pad>"` for an unfilled slot |
| `species_idx_preview`   | (12,)  | int64   | `SPECIES_VOCAB` index per previewed slot, `[our0..5, opp0..5]`, `"<pad>"` for an unfilled/missing preview slot (v4, NEW) |
| `item_idx`              | (4,)   | int64   | `ITEM_VOCAB` index per active slot |
| `ability_idx`           | (4,)   | int64   | `ABILITY_VOCAB` index per active slot |
| `move_idx`              | (4,4)  | int64   | `MOVE_VOCAB` index of each active slot's up-to-4 revealed moves (`"<pad>"` for the rest); slot order matches the other 4-slot arrays, move order is `vgc.replay_parse`'s sorted `revealed_moves`, truncated to 4 |
| `hp_fraction`           | (4,)   | float32 | 0.0-1.0, 0.0 for an empty slot |
| `status`                | (4,7)  | float32 | one-hot over `["none","brn","par","psn","tox","slp","frz"]` |
| `boosts`                | (4,5)  | float32 | `[atk,def,spa,spd,spe]` stage / 6.0 |
| `mega`                  | (4,)   | float32 | 1.0/0.0 |
| `bench_hp_fraction`     | (8,)   | float32 | 0.0-1.0, 0.0 for an unfilled bench slot |
| `fainted_count`         | (2,)   | float32 | `[our, opp]` -- count of that side's CURRENTLY EMPTY active slots (0, 1, or 2), / 2.0. This is a per-snapshot signal ("this side has an empty slot right now"), NOT a full-game running tally of every faint that's ever happened -- the latter isn't derivable from a single state snapshot (a fainted mon simply disappears from both `active` and `bench`) without a `vgc.replay_parse` schema change beyond schema 2's scope. |
| `weather`               | (5,)   | float32 | one-hot over `["none","sun","rain","sand","snow"]` |
| `terrain`               | (5,)   | float32 | one-hot over `["none","electric","grassy","psychic","misty"]` |
| `trick_room`            | (1,)   | float32 | 1.0/0.0 |
| `side_conditions`       | (8,)   | float32 | `[our_tailwind, our_reflect, our_lightscreen, our_auroraveil, opp_tailwind, opp_reflect, opp_lightscreen, opp_auroraveil]` |
| `turn`                  | (1,)   | float32 | `min(1.0, state["field"]["turn"] / 20.0)` |
| `alive_known_count`     | (2,)   | float32 | `[our, opp]` -- count of that side's currently-alive KNOWN mons (non-empty active slots + bench entries -- bench already excludes fainted mons, see `vgc.replay_parse`'s docstring) / 4.0 (v4, NEW; scaled by the bring-6-pick-4 format's max of 4 mons a side can ever have known+alive at once) |
| `alive_known_mean_hp`   | (2,)   | float32 | `[our, opp]` -- mean `hp_fraction` across that side's currently-alive known mons (active + bench), 0.0 if none (v4, NEW) |
| `preview_unseen_count`  | (2,)   | float32 | `[our, opp]` -- `state["our"/"opp"]["unseen_count"]` / 6.0 (v4, NEW; see `vgc.replay_parse` schema 4's docstring for why this doesn't trend to 0) |

`flatten_state(state) -> (index_array, scalar_array)` concatenates every INDEX key
(`species_idx_active`, `species_idx_bench`, `species_idx_preview`, `item_idx`,
`ability_idx`, `move_idx` flattened) into one `(INDEX_DIM,)` int64 array in that exact
order -- see `INDEX_SPECIES_ACTIVE_SLICE`/etc. below for the exact offsets `vgc.bc.model`
slices by -- and every remaining (scalar) key into one `(STATE_SCALAR_DIM,)` float32
array, in the table's top-to-bottom order (skipping the 6 index keys). The model
consumes exactly these two tensors (plus `vgc.bc.dataset`'s slot-index one-hot appended
to the scalar array) -- see `vgc.bc.model`'s module docstring.

"Missing/None -> zeros" everywhere, same contract as v1.

## `encode_action(record, slot) -> int | None`

Unchanged from v1 (still keys into `MOVE_VOCAB`, which now has one extra reserved
`"<pad>"` entry that shifts every other index but not the logic) -- see v1's docstring
text below, preserved verbatim:

The `MOVE_VOCAB` index of `record["action"][f"slot{slot}"]`'s action, ONLY for
`decision_kind == "turn"` records. Returns `None` (caller skips the sample) when: the
record isn't a "turn" decision, the named slot has no action dict, or its
`kind == "move"` but the `move_id` isn't a recognized champions-mod move.

## `encode_target(record, slot) -> int | None`

The `TARGET_VOCAB` index of WHERE that slot's action goes:
- `kind in ("switch", "pass")` -> always `"<none>"` (a real, always-assigned class: these
  action kinds have no targeting decision at all -- NOT treated as "unknown", since we
  DO know there's no target here, as opposed to a move whose target genuinely couldn't
  be determined).
- `kind == "move"` -> `action["target_slot"]` mapped directly (`"self"` -> the vocab's
  `"self_or_field"` entry; `"opp0"`/`"opp1"`/`"ally"`/`"spread"` map 1:1). If
  `target_slot` is `None` (an unparseable target -- e.g. a blocked `|cant|`-attempted
  move, which `vgc.replay_parse` always records with `target_slot: None` since a
  blocked move's real intended target was never announced by the protocol), returns
  `None` -- the "None-skip contract" matches `encode_action`'s: the caller should NOT
  train the target head on a state where the true target is genuinely unknown, but MAY
  still train the move head on the same sample (see `vgc.bc.dataset`).
- Anything else -> `None`.

## `encode_value(record) -> float | None`

`1.0` if `record["won"]` is `True`, `0.0` if `False`, `None` if the key is missing
entirely (a pre-schema-3 `vgc.replay_parse` record). Unlike `encode_action`/
`encode_target`, NOT gated to `decision_kind == "turn"` -- "who won the game" is a
per-record fact regardless of decision kind (`vgc.bc.dataset.BcTurnDataset` only ever
calls this for "turn" records today, matching the move/target heads' training data, but
the function itself doesn't assume that). The `None` case masks the value loss out the
same way `encode_target`'s `None` masks the target loss (see `vgc.bc.dataset`) --
matters for training on any decisions.jsonl built before schema 3.

## `ENCODER_LAYOUT_VERSION`

Bumped to `"bc-encoding-v4"` (see the v4 changes section above). Saved into every
training checkpoint alongside the vocab lists -- bump again whenever this module's STATE
encoding layout changes. `encode_value` doesn't touch state encoding at all (it reads
`record["won"]` directly, nothing from `encode_state`'s output), so adding it did NOT
bump this version -- a value-headed checkpoint's compatibility is still gated purely on
the state layout, exactly as before.
"""

from __future__ import annotations

import numpy as np

from vgc.damage import to_id
from vgc.data import load_items, load_moves, load_species

# --- vocabularies (deterministic, derived at import time from committed data) --------

SPECIES_VOCAB: list[str] = ["<pad>", "<unk>"] + sorted(load_species().keys())
SPECIES_TO_IDX: dict[str, int] = {token: idx for idx, token in enumerate(SPECIES_VOCAB)}

MOVE_VOCAB: list[str] = ["<pad>", "<switch>", "<pass>", "<unk>"] + sorted(load_moves().keys())
MOVE_TO_IDX: dict[str, int] = {token: idx for idx, token in enumerate(MOVE_VOCAB)}


def _all_ability_ids() -> list[str]:
    names: set[str] = set()
    for species in load_species().values():
        for raw_name in (species.get("abilities") or {}).values():
            normalized = to_id(raw_name)
            if normalized:
                names.add(normalized)
    return sorted(names)


ITEM_VOCAB: list[str] = ["<unk>", "<none>"] + sorted(load_items().keys())
ITEM_TO_IDX: dict[str, int] = {token: idx for idx, token in enumerate(ITEM_VOCAB)}

ABILITY_VOCAB: list[str] = ["<unk>"] + _all_ability_ids()
ABILITY_TO_IDX: dict[str, int] = {token: idx for idx, token in enumerate(ABILITY_VOCAB)}

TARGET_VOCAB: list[str] = ["opp0", "opp1", "ally", "self_or_field", "spread", "<none>"]
TARGET_TO_IDX: dict[str, int] = {token: idx for idx, token in enumerate(TARGET_VOCAB)}

ENCODER_LAYOUT_VERSION = "bc-encoding-v4"

# --- encode_state's fixed vocab-like orderings ----------------------------------------

_STATUS_ORDER = ("none", "brn", "par", "psn", "tox", "slp", "frz")
_STATUS_TO_IDX = {status: idx for idx, status in enumerate(_STATUS_ORDER)}
_BOOST_STATS = ("atk", "def", "spa", "spd", "spe")
_BOOST_STAGE_SCALE = 6.0
_WEATHER_ORDER = ("none", "sun", "rain", "sand", "snow")
_WEATHER_TO_IDX = {weather: idx for idx, weather in enumerate(_WEATHER_ORDER)}
_TERRAIN_ORDER = ("none", "electric", "grassy", "psychic", "misty")
_TERRAIN_TO_IDX = {terrain: idx for idx, terrain in enumerate(_TERRAIN_ORDER)}
_SIDE_CONDITION_FLAGS = ("tailwind", "reflect", "lightscreen", "auroraveil")
_TURN_SCALE = 20.0

_MAX_BENCH_SLOTS = 4
_MAX_REVEALED_MOVES = 4
# Bring-6-pick-4: a side's full previewed roster is always 6, and at most 4 of them can
# ever be known+alive on the field at once -- see the v4 docstring section above.
_MAX_PREVIEW_SLOTS = 6
_MAX_KNOWN_ALIVE = 4.0

# --- index array layout (documented offsets vgc.bc.model slices by) -------------------

INDEX_SPECIES_ACTIVE_SLICE = slice(0, 4)
INDEX_SPECIES_BENCH_SLICE = slice(4, 12)
INDEX_SPECIES_PREVIEW_SLICE = slice(12, 24)
INDEX_ITEM_SLICE = slice(24, 28)
INDEX_ABILITY_SLICE = slice(28, 32)
INDEX_MOVES_SLICE = slice(32, 48)
INDEX_DIM = 48  # 4 + 8 + 12 + 4 + 4 + 16

# Scalar feature dim `flatten_state` produces (every encode_state key EXCEPT the 6
# index keys above): hp_fraction (4) + status (4*7=28) + boosts (4*5=20) + mega (4) +
# bench_hp_fraction (8) + fainted_count (2) + weather (5) + terrain (5) + trick_room
# (1) + side_conditions (8) + turn (1) + alive_known_count (2) + alive_known_mean_hp (2)
# + preview_unseen_count (2) [v4, NEW: last 3 terms].
STATE_SCALAR_DIM = 4 + 28 + 20 + 4 + 8 + 2 + 5 + 5 + 1 + 8 + 1 + 2 + 2 + 2
# vgc.bc.dataset appends a 2-dim one-hot (which of our 2 active slots is deciding) on
# top of flatten_state's scalar output before feeding vgc.bc.model.BcPolicyNet -- both
# modules import this constant rather than hardcoding "2" so they can't silently drift.
SLOT_FEATURE_DIM = 2


def _encode_active_slot(mon: dict | None) -> dict[str, object]:
    if not mon:
        return {
            "species_idx": SPECIES_TO_IDX["<pad>"],
            "item_idx": ITEM_TO_IDX["<unk>"],
            "ability_idx": ABILITY_TO_IDX["<unk>"],
            "move_idx": [MOVE_TO_IDX["<pad>"]] * _MAX_REVEALED_MOVES,
            "hp_fraction": 0.0,
            "status": [0.0] * len(_STATUS_ORDER),
            "boosts": [0.0] * len(_BOOST_STATS),
            "mega": 0.0,
        }
    species_idx = SPECIES_TO_IDX.get(mon.get("species"), SPECIES_TO_IDX["<unk>"])
    item_idx = (
        ITEM_TO_IDX.get(mon.get("item"), ITEM_TO_IDX["<unk>"])
        if mon.get("item")
        else ITEM_TO_IDX["<unk>"]
    )
    ability_idx = (
        ABILITY_TO_IDX.get(mon.get("ability"), ABILITY_TO_IDX["<unk>"])
        if mon.get("ability")
        else ABILITY_TO_IDX["<unk>"]
    )
    revealed_moves = list(mon.get("revealed_moves") or [])[:_MAX_REVEALED_MOVES]
    move_idx = [MOVE_TO_IDX.get(move_id, MOVE_TO_IDX["<unk>"]) for move_id in revealed_moves]
    move_idx += [MOVE_TO_IDX["<pad>"]] * (_MAX_REVEALED_MOVES - len(move_idx))
    status_vec = [0.0] * len(_STATUS_ORDER)
    status_vec[_STATUS_TO_IDX.get(mon.get("status") or "none", 0)] = 1.0
    boosts = mon.get("boosts") or {}
    boost_vec = [float(boosts.get(stat, 0)) / _BOOST_STAGE_SCALE for stat in _BOOST_STATS]
    return {
        "species_idx": species_idx,
        "item_idx": item_idx,
        "ability_idx": ability_idx,
        "move_idx": move_idx,
        "hp_fraction": float(mon.get("hp_fraction") or 0.0),
        "status": status_vec,
        "boosts": boost_vec,
        "mega": 1.0 if mon.get("mega") else 0.0,
    }


def _encode_bench(bench: list[dict] | None) -> tuple[list[int], list[float]]:
    bench = list(bench or [])[:_MAX_BENCH_SLOTS]
    species_idx = [
        SPECIES_TO_IDX.get(mon.get("species_id"), SPECIES_TO_IDX["<unk>"]) for mon in bench
    ]
    hp_fraction = [float(mon.get("hp_fraction") or 0.0) for mon in bench]
    pad_count = _MAX_BENCH_SLOTS - len(bench)
    species_idx += [SPECIES_TO_IDX["<pad>"]] * pad_count
    hp_fraction += [0.0] * pad_count
    return species_idx, hp_fraction


def _encode_preview(preview_species: list[str] | None) -> list[int]:
    """(v4) `SPECIES_VOCAB` index per previewed species, truncated/padded to exactly
    `_MAX_PREVIEW_SLOTS` -- mirrors `_encode_bench`'s truncate-then-pad shape, but for
    `vgc.replay_parse` schema 4's `preview_species` (always up to 6, Team Preview
    order) rather than the appeared-only bench list.
    """
    preview_species = list(preview_species or [])[:_MAX_PREVIEW_SLOTS]
    species_idx = [
        SPECIES_TO_IDX.get(species_id, SPECIES_TO_IDX["<unk>"]) for species_id in preview_species
    ]
    species_idx += [SPECIES_TO_IDX["<pad>"]] * (_MAX_PREVIEW_SLOTS - len(species_idx))
    return species_idx


def _encode_side(side: dict | None) -> dict[str, object]:
    side = side or {}
    active = list(side.get("active") or [])
    while len(active) < 2:
        active.append(None)
    slots = [_encode_active_slot(active[i]) for i in range(2)]
    fainted_count = sum(1 for mon in active[:2] if not mon) / 2.0
    bench_raw = list(side.get("bench") or [])
    bench_species_idx, bench_hp_fraction = _encode_bench(bench_raw)
    conditions = set(side.get("side_conditions") or [])
    condition_flags = [1.0 if flag in conditions else 0.0 for flag in _SIDE_CONDITION_FLAGS]
    preview_species_idx = _encode_preview(side.get("preview_species"))

    # v4 resource-state summary: "known" mons are exactly what's already in the state
    # (non-empty active slots + bench, which per vgc.replay_parse's docstring only ever
    # lists species that have appeared and aren't fainted) -- no evaluator/gameplan
    # dependency needed, see module docstring's v4 section.
    alive_hp_fractions = [float(mon.get("hp_fraction") or 0.0) for mon in active[:2] if mon] + [
        float(mon.get("hp_fraction") or 0.0) for mon in bench_raw
    ]
    alive_known_count = len(alive_hp_fractions)
    alive_known_mean_hp = sum(alive_hp_fractions) / alive_known_count if alive_known_count else 0.0
    unseen_count = float(side.get("unseen_count") or 0)

    return {
        "slots": slots,
        "bench_species_idx": bench_species_idx,
        "bench_hp_fraction": bench_hp_fraction,
        "fainted_count": fainted_count,
        "condition_flags": condition_flags,
        "preview_species_idx": preview_species_idx,
        "alive_known_count": alive_known_count / _MAX_KNOWN_ALIVE,
        "alive_known_mean_hp": alive_known_mean_hp,
        "preview_unseen_count": unseen_count / _MAX_PREVIEW_SLOTS,
    }


def encode_state(record: dict) -> dict[str, np.ndarray]:
    """See module docstring for the exact layout of every returned key."""
    state = record.get("state") or {}
    our = _encode_side(state.get("our"))
    opp = _encode_side(state.get("opp"))
    field = state.get("field") or {}

    all_slots = our["slots"] + opp["slots"]  # [our0, our1, opp0, opp1]

    weather_vec = [0.0] * len(_WEATHER_ORDER)
    weather_vec[_WEATHER_TO_IDX.get(field.get("weather") or "none", 0)] = 1.0
    terrain_vec = [0.0] * len(_TERRAIN_ORDER)
    terrain_vec[_TERRAIN_TO_IDX.get(field.get("terrain") or "none", 0)] = 1.0
    turn_value = min(1.0, float(field.get("turn") or 0) / _TURN_SCALE)

    return {
        "species_idx_active": np.array([slot["species_idx"] for slot in all_slots], dtype=np.int64),
        "species_idx_bench": np.array(
            our["bench_species_idx"] + opp["bench_species_idx"], dtype=np.int64
        ),
        "species_idx_preview": np.array(
            our["preview_species_idx"] + opp["preview_species_idx"], dtype=np.int64
        ),
        "item_idx": np.array([slot["item_idx"] for slot in all_slots], dtype=np.int64),
        "ability_idx": np.array([slot["ability_idx"] for slot in all_slots], dtype=np.int64),
        "move_idx": np.array([slot["move_idx"] for slot in all_slots], dtype=np.int64),
        "hp_fraction": np.array([slot["hp_fraction"] for slot in all_slots], dtype=np.float32),
        "status": np.array([slot["status"] for slot in all_slots], dtype=np.float32),
        "boosts": np.array([slot["boosts"] for slot in all_slots], dtype=np.float32),
        "mega": np.array([slot["mega"] for slot in all_slots], dtype=np.float32),
        "bench_hp_fraction": np.array(
            our["bench_hp_fraction"] + opp["bench_hp_fraction"], dtype=np.float32
        ),
        "fainted_count": np.array([our["fainted_count"], opp["fainted_count"]], dtype=np.float32),
        "weather": np.array(weather_vec, dtype=np.float32),
        "terrain": np.array(terrain_vec, dtype=np.float32),
        "trick_room": np.array([1.0 if field.get("trick_room") else 0.0], dtype=np.float32),
        "side_conditions": np.array(
            our["condition_flags"] + opp["condition_flags"], dtype=np.float32
        ),
        "turn": np.array([turn_value], dtype=np.float32),
        "alive_known_count": np.array(
            [our["alive_known_count"], opp["alive_known_count"]], dtype=np.float32
        ),
        "alive_known_mean_hp": np.array(
            [our["alive_known_mean_hp"], opp["alive_known_mean_hp"]], dtype=np.float32
        ),
        "preview_unseen_count": np.array(
            [our["preview_unseen_count"], opp["preview_unseen_count"]], dtype=np.float32
        ),
    }


def flatten_state(state: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """`(index_array, scalar_array)`: `index_array` is `(INDEX_DIM,)` int64 (every
    embedding-lookup key, in `INDEX_*_SLICE` order); `scalar_array` is
    `(STATE_SCALAR_DIM,)` float32 (everything else, in the module docstring table's
    order). See `vgc.bc.model` for how these two tensors are consumed.
    """
    index_array = np.concatenate(
        [
            state["species_idx_active"],
            state["species_idx_bench"],
            state["species_idx_preview"],
            state["item_idx"],
            state["ability_idx"],
            state["move_idx"].reshape(-1),
        ]
    ).astype(np.int64)
    scalar_array = np.concatenate(
        [
            state["hp_fraction"],
            state["status"].reshape(-1),
            state["boosts"].reshape(-1),
            state["mega"],
            state["bench_hp_fraction"],
            state["fainted_count"],
            state["weather"],
            state["terrain"],
            state["trick_room"],
            state["side_conditions"],
            state["turn"],
            state["alive_known_count"],
            state["alive_known_mean_hp"],
            state["preview_unseen_count"],
        ]
    ).astype(np.float32)
    return index_array, scalar_array


def encode_action(record: dict, slot: int) -> int | None:
    """See module docstring's "`encode_action`" section."""
    if record.get("decision_kind") != "turn":
        return None
    action = (record.get("action") or {}).get(f"slot{slot}")
    if not action:
        return None
    kind = action.get("kind")
    if kind == "pass":
        return MOVE_TO_IDX["<pass>"]
    if kind == "switch":
        return MOVE_TO_IDX["<switch>"]
    if kind == "move":
        move_id = action.get("move_id")
        if not move_id:
            return None
        return MOVE_TO_IDX.get(move_id)  # None (not "<unk>") for an unrecognized move id
    return None


def encode_target(record: dict, slot: int) -> int | None:
    """See module docstring's "`encode_target`" section."""
    if record.get("decision_kind") != "turn":
        return None
    action = (record.get("action") or {}).get(f"slot{slot}")
    if not action:
        return None
    kind = action.get("kind")
    if kind in ("switch", "pass"):
        return TARGET_TO_IDX["<none>"]
    if kind == "move":
        target_slot = action.get("target_slot")
        if target_slot is None:
            return None
        if target_slot == "self":
            return TARGET_TO_IDX["self_or_field"]
        return TARGET_TO_IDX.get(target_slot)
    return None


def encode_value(record: dict) -> float | None:
    """See module docstring's "`encode_value`" section."""
    won = record.get("won")
    if won is None:
        return None
    return 1.0 if won else 0.0
