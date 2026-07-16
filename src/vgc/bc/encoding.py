"""Pure encoders from a `data/bc/decisions.jsonl` record (`vgc.replay_parse`'s schema)
to fixed-size numpy arrays. NO torch import here -- keep this importable without the
`train` extra (`uv sync --extra train`); `vgc.bc.dataset`/`model`/`train` do the
torch-dependent work on top of what this module produces.

## Vocabularies

Built deterministically at import time from COMMITTED data -- nothing is persisted by
this module itself, so the vocab is always reproducible from `data/champions/*.json`
alone (species/moves.json's keys, sorted):

- `SPECIES_VOCAB` = `["<pad>", "<unk>"] + sorted(vgc.data.load_species().keys())`.
  `"<pad>"` (index 0) is the embedding for an EMPTY active slot (no Pokemon there --
  see `encode_state`'s "Missing/None -> zeros" contract, index 0 zeroed out is exactly
  what a fresh `nn.Embedding` gives an untouched row anyway, but the explicit token
  keeps the mapping documented rather than accidental). `"<unk>"` (index 1) is any
  species string that shows up in a decision record but isn't a recognized champions-
  mod species id (shouldn't normally happen -- `vgc.replay_parse` only ever writes
  `to_id`-normalized real species ids -- but a corpus can outlive a species.json
  regeneration, so this is a defensive fallback, not a dead code path).
- `MOVE_VOCAB` = `["<switch>", "<pass>", "<unk>"] + sorted(vgc.data.load_moves().keys())`.
  `"<switch>"`/`"<pass>"` are the two non-move action kinds a turn-decision slot can
  take (see `encode_action`). `"<unk>"` is RESERVED for symmetry with `SPECIES_VOCAB`
  but never actually assigned by `encode_action` -- an unrecognized move id returns
  `None` instead (see `encode_action`'s docstring), so a caller skips that sample
  rather than training the model to treat "unknown" as a predictable class.

## `encode_state(record) -> dict[str, np.ndarray]` layout

Per-side-relative (`record["state"]["our"]`/`["opp"]`, already oriented from the
deciding player's own perspective by `vgc.replay_parse`) fixed-size arrays, in this
exact order for every 4-slot array below: `[our_slot0, our_slot1, opp_slot0, opp_slot1]`.

| key               | shape | dtype    | meaning |
|-------------------|-------|----------|---------|
| `species_idx`      | (4,)  | int64    | `SPECIES_VOCAB` index per active slot |
| `hp_fraction`      | (4,)  | float32  | 0.0-1.0, 0.0 for an empty slot |
| `status`           | (4,7) | float32  | one-hot over `["none","brn","par","psn","tox","slp","frz"]`, all-zero for an empty slot |
| `boosts`           | (4,5) | float32  | `[atk,def,spa,spd,spe]` stage / 6.0 (so +-6 maps to +-1.0) |
| `mega`             | (4,)  | float32  | 1.0 if that slot's mon is mega-evolved, else 0.0 |
| `bench_count`      | (2,)  | float32  | `[our, opp]` -- count of `vgc.replay_parse`'s "bench" list (species that have actually appeared, per its own module docstring -- NOT the full previewed 6) |
| `bench_mean_hp`    | (2,)  | float32  | `[our, opp]` mean bench HP fraction, 0.0 if no bench entries |
| `weather`          | (5,)  | float32  | one-hot over `["none","sun","rain","sand","snow"]` |
| `terrain`          | (5,)  | float32  | one-hot over `["none","electric","grassy","psychic","misty"]` |
| `trick_room`       | (1,)  | float32  | 1.0/0.0 |
| `side_conditions`  | (8,)  | float32  | `[our_tailwind, our_reflect, our_lightscreen, our_auroraveil, opp_tailwind, opp_reflect, opp_lightscreen, opp_auroraveil]` |
| `turn`             | (1,)  | float32  | `min(1.0, state["field"]["turn"] / 20.0)` |

`species_idx` is meant to be looked up through an `nn.Embedding` (see `vgc.bc.model`);
every other key is a plain scalar/one-hot feature. `flatten_state` concatenates every
key EXCEPT `species_idx` into one `(STATE_SCALAR_DIM,)` vector (species stay a separate
int64 array since they need embedding lookup, not concatenation).

"Missing/None -> zeros" everywhere: an empty active slot, a missing `state`/`field`/side
dict, absent boosts, etc. all resolve to the all-zero encoding for that piece rather than
raising -- a decision record is trusted to match `vgc.replay_parse`'s schema, but this
module never assumes a key is present.

## `encode_action(record, slot) -> int | None`

The `MOVE_VOCAB` index of `record["action"][f"slot{slot}"]`'s action, ONLY for
`decision_kind == "turn"` records (forced_switch/teampreview aren't move-vocab
decisions in this v1 scope -- `vgc.bc.dataset.BcTurnDataset` only ever samples "turn"
records anyway). Returns `None` (caller skips the sample) when: the record isn't a
"turn" decision, the named slot has no action dict, or its `kind == "move"` but the
`move_id` isn't a recognized champions-mod move -- see `MOVE_VOCAB`'s note above for
why that's `None` and not `"<unk>"`.

## `ENCODER_LAYOUT_VERSION`

A version string saved into every training checkpoint (`vgc.bc.train`) alongside the
vocab lists -- bump it whenever this module's encoding layout changes, so a checkpoint
trained against an older layout is never silently loaded against a newer one.
"""

from __future__ import annotations

import numpy as np

from vgc.data import load_moves, load_species

# --- vocabularies (deterministic, derived at import time from committed data) --------

SPECIES_VOCAB: list[str] = ["<pad>", "<unk>"] + sorted(load_species().keys())
SPECIES_TO_IDX: dict[str, int] = {token: idx for idx, token in enumerate(SPECIES_VOCAB)}

MOVE_VOCAB: list[str] = ["<switch>", "<pass>", "<unk>"] + sorted(load_moves().keys())
MOVE_TO_IDX: dict[str, int] = {token: idx for idx, token in enumerate(MOVE_VOCAB)}

ENCODER_LAYOUT_VERSION = "bc-encoding-v1"

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

# Scalar feature dim `flatten_state` produces (everything in encode_state's dict EXCEPT
# species_idx, which stays a separate int64 array for nn.Embedding lookup): hp_fraction
# (4) + status (4*7=28) + boosts (4*5=20) + mega (4) + bench_count (2) + bench_mean_hp
# (2) + weather (5) + terrain (5) + trick_room (1) + side_conditions (8) + turn (1).
STATE_SCALAR_DIM = 4 + 28 + 20 + 4 + 2 + 2 + 5 + 5 + 1 + 8 + 1
# vgc.bc.dataset appends a 2-dim one-hot (which of our 2 active slots is deciding) on
# top of flatten_state's output before feeding vgc.bc.model.BcPolicyNet -- both modules
# import this constant rather than hardcoding "2" so they can't silently drift apart.
SLOT_FEATURE_DIM = 2


def _encode_active_slot(mon: dict | None) -> dict[str, object]:
    if not mon:
        return {
            "species_idx": SPECIES_TO_IDX["<pad>"],
            "hp_fraction": 0.0,
            "status": [0.0] * len(_STATUS_ORDER),
            "boosts": [0.0] * len(_BOOST_STATS),
            "mega": 0.0,
        }
    species_idx = SPECIES_TO_IDX.get(mon.get("species"), SPECIES_TO_IDX["<unk>"])
    status_vec = [0.0] * len(_STATUS_ORDER)
    status_vec[_STATUS_TO_IDX.get(mon.get("status") or "none", 0)] = 1.0
    boosts = mon.get("boosts") or {}
    boost_vec = [float(boosts.get(stat, 0)) / _BOOST_STAGE_SCALE for stat in _BOOST_STATS]
    return {
        "species_idx": species_idx,
        "hp_fraction": float(mon.get("hp_fraction") or 0.0),
        "status": status_vec,
        "boosts": boost_vec,
        "mega": 1.0 if mon.get("mega") else 0.0,
    }


def _encode_side(side: dict | None) -> dict[str, object]:
    side = side or {}
    active = list(side.get("active") or [])
    while len(active) < 2:
        active.append(None)
    slots = [_encode_active_slot(active[i]) for i in range(2)]
    bench = side.get("bench") or []
    bench_count = float(len(bench))
    bench_mean_hp = (
        float(sum(mon.get("hp_fraction", 0.0) for mon in bench) / len(bench)) if bench else 0.0
    )
    conditions = set(side.get("side_conditions") or [])
    condition_flags = [1.0 if flag in conditions else 0.0 for flag in _SIDE_CONDITION_FLAGS]
    return {
        "slots": slots,
        "bench_count": bench_count,
        "bench_mean_hp": bench_mean_hp,
        "condition_flags": condition_flags,
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
        "species_idx": np.array([slot["species_idx"] for slot in all_slots], dtype=np.int64),
        "hp_fraction": np.array([slot["hp_fraction"] for slot in all_slots], dtype=np.float32),
        "status": np.array([slot["status"] for slot in all_slots], dtype=np.float32),
        "boosts": np.array([slot["boosts"] for slot in all_slots], dtype=np.float32),
        "mega": np.array([slot["mega"] for slot in all_slots], dtype=np.float32),
        "bench_count": np.array([our["bench_count"], opp["bench_count"]], dtype=np.float32),
        "bench_mean_hp": np.array([our["bench_mean_hp"], opp["bench_mean_hp"]], dtype=np.float32),
        "weather": np.array(weather_vec, dtype=np.float32),
        "terrain": np.array(terrain_vec, dtype=np.float32),
        "trick_room": np.array([1.0 if field.get("trick_room") else 0.0], dtype=np.float32),
        "side_conditions": np.array(
            our["condition_flags"] + opp["condition_flags"], dtype=np.float32
        ),
        "turn": np.array([turn_value], dtype=np.float32),
    }


def flatten_state(state: dict[str, np.ndarray]) -> np.ndarray:
    """Concatenates every `encode_state` key EXCEPT `species_idx` into one
    `(STATE_SCALAR_DIM,)` float32 vector, in the fixed order documented in the module
    docstring's table (top to bottom, skipping `species_idx`).
    """
    return np.concatenate(
        [
            state["hp_fraction"],
            state["status"].reshape(-1),
            state["boosts"].reshape(-1),
            state["mega"],
            state["bench_count"],
            state["bench_mean_hp"],
            state["weather"],
            state["terrain"],
            state["trick_room"],
            state["side_conditions"],
            state["turn"],
        ]
    ).astype(np.float32)


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
