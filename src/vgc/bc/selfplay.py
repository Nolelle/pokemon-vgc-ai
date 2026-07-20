"""Self-play data recording (`selfplay/run_selfplay.py`'s per-game decision recorder) --
grows the BC value head's training data ~10x beyond the downloaded human replay corpus
with free local bot-vs-bot games. Every recorded record aims to be a schema-4
`vgc.replay_parse`-shaped `"turn"` decision (`decision_kind`, `state`, `action`, `turn`,
`won`, `rating: None`, `replay_id`, `player`) so `vgc.bc.dataset.BcTurnDataset` can
consume it via `vgc.bc.encoding.encode_state`/`encode_action`/`encode_target`/
`encode_value` exactly like a real corpus record -- see `vgc.bc.dataset`'s
`allow_null_rating` flag (this data is always `rating: None`, since there's no ladder
Elo for a local self-play game; that flag is the explicit, non-silent way a caller opts
into training on null-rated records instead of them being dropped by default).

## Open Team Sheets: deliberately REJECTED for self-play

`vgc.models.PolicyConfig.accept_open_team_sheet` defaults to `True` (we always want to
see a REAL opponent's revealed sheet on the ladder). But in a self-play game, BOTH
players' configs would default to accepting -- and unlike the real ladder (where OTS
triggers on ~0.2% of games, see `vgc.replay_parse`'s module docstring), two of our own
bots would then ALWAYS mutually accept, which would make poke-env populate the
opponent's full revealed team via `battle.apply_teambuilder_team` at Team Preview time,
well before any of those Pokemon actually switch in. That would make self-play's
`battle_state_record` opponent-side state (bench identity, `unseen_count`) look nothing
like the real corpus data it's meant to supplement (built from ~0.2%-OTS human replays)
or like real ladder play -- a systematic train/serve mismatch, not a realistic case this
format actually produces at any real rate. `run_selfplay.py` therefore forces
`accept_open_team_sheet=False` on every self-play `PolicyConfig` variant, so both sides
only ever learn about each other from in-battle reveals, exactly like the corpus.

## `order_to_action_dict`

Converts a *chosen* `DoubleBattleOrder` (whatever `VgcPlayer.decide()` actually returned)
into the SAME `{"slot0": {...}, "slot1": {...}}` shape `vgc.replay_parse`'s "turn"
records carry, so `vgc.bc.encoding.encode_action`/`encode_target` read it identically
regardless of whether the underlying decision came from a parsed replay or a live
self-play game. Mirrors `vgc.bc.policy._single_tokens`/`_target_token`'s existing
order-to-target-kind logic (same `Move`/`Pokemon`/pass dispatch, same forward
target-kind mapping from `vgc.evaluator`'s target-kind constants) but emits raw
`vgc.replay_parse`-vocabulary strings (`"self"`, not `"self_or_field"`) instead of
`vgc.bc.encoding` vocab indices -- those raw strings are exactly what `encode_target`
already knows how to map (`"self"` -> `TARGET_TO_IDX["self_or_field"]`), so this stays a
pure schema-shape converter with no encoder-vocabulary dependency at all.

## `RecordingVgcPlayer`

A `VgcPlayer` subclass that overrides ONLY `decide()` (never `decide_teampreview()` --
self-play doesn't need teampreview records, see module docstring's schema note) to
buffer one record per turn decision, then finalizes (labels every buffered record for
that battle with the real outcome and appends them to `out_path`) from
`_battle_finished_callback`. Unlike `ladder/run_ladder.py`'s `LadderPlayer` (which defers
its own finalization by a tick because it additionally needs `battle.rating`, which only
populates on a LATER message), `battle.won`/`battle.turn` are already fully populated by
the time `_battle_finished_callback` fires (poke-env calls `battle.won_by(...)` before
invoking it) -- self-play records don't have or need a rating, so no deferral is needed
here, and finalization can happen synchronously.

Crash-safety: records for a battle are only written to disk once that battle actually
finishes (opened, appended, and closed in one `with` block) -- a mid-run crash loses at
most the one battle in flight, never any already-completed battle's data.
"""

from __future__ import annotations

import json
from pathlib import Path

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.move import Move
from poke_env.battle.pokemon import Pokemon
from poke_env.player.battle_order import BattleOrder, DoubleBattleOrder, SingleBattleOrder

from vgc.agent import VgcPlayer
from vgc.bc.policy import battle_state_record
from vgc.damage import to_id
from vgc.data import load_moves
from vgc.evaluator import _SINGLE_TARGETS, _SPREAD_TARGETS_FOES_ONLY, _SPREAD_TARGETS_HITTING_ALLY
from vgc.replay_parse import SCHEMA_VERSION, _resolve_species


def _forward_target_slot(move_data: dict | None, move_target: int) -> str | None:
    """Forward (order -> label) counterpart of `vgc.replay_parse._target_slot_label`:
    given a legal chosen move's own data + `move_target` position number (poke-env's
    -2/-1/1/2 convention, same as `vgc.evaluator._resolve_targets`), returns the SAME
    `"self"`/`"ally"`/`"opp0"`/`"opp1"`/`"spread"` vocabulary `vgc.replay_parse` produces
    when reconstructing a target backward from a replay protocol line. `None` if
    `move_data` is missing (an unrecognized move id -- shouldn't happen for a
    legally-enumerated order, but defensive).
    """
    if move_data is None:
        return None
    target_kind = move_data.get("target")
    if target_kind in _SPREAD_TARGETS_HITTING_ALLY or target_kind in _SPREAD_TARGETS_FOES_ONLY:
        return "spread"
    if target_kind in _SINGLE_TARGETS:
        if move_target in (1, 2):
            return f"opp{move_target - 1}"
        if move_target in (-1, -2):
            return "ally"
        return "opp0"
    return "self"


def _single_order_to_action(single: SingleBattleOrder | None) -> dict[str, object]:
    """One slot's `vgc.replay_parse`-shaped action dict from a chosen `SingleBattleOrder`
    (or `None`/non-Move/non-Pokemon payload, all of which mean "pass" -- mirrors
    `vgc.bc.policy._single_tokens`'s exact same fallback contract).
    """
    if single is None:
        return {"kind": "pass"}
    target = single.order
    if isinstance(target, Move):
        move_id = to_id(target.id)
        move_data = load_moves().get(move_id) if move_id else None
        target_slot = _forward_target_slot(move_data, getattr(single, "move_target", 0) or 0)
        return {
            "kind": "move",
            "move_id": move_id,
            "target_slot": target_slot,
            "mega": bool(getattr(single, "mega", False)),
        }
    if isinstance(target, Pokemon):
        species_id, _mega = _resolve_species(to_id(target.species) or "")
        return {"kind": "switch", "switch_species": species_id}
    return {"kind": "pass"}


def order_to_action_dict(order: DoubleBattleOrder) -> dict[str, dict[str, object]]:
    """`{"slot0": {...}, "slot1": {...}}` for a chosen joint order -- see module
    docstring.
    """
    return {
        "slot0": _single_order_to_action(order.first_order),
        "slot1": _single_order_to_action(order.second_order),
    }


class RecordingVgcPlayer(VgcPlayer):
    """`VgcPlayer` that records every `decide()` call as a schema-4-shaped `"turn"`
    decision record, appending them (labeled with the battle's real outcome) to
    `out_path` once each battle finishes. See module docstring for the full design.
    """

    def __init__(
        self,
        *,
        out_path: str | Path,
        replay_tag: str,
        config=None,
        **player_kwargs,
    ) -> None:
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        # A short, stable-per-instance tag (e.g. the sampled team/config variant's name)
        # so `record["replay_id"]` stays informative even though there's no real replay
        # id -- combined with poke-env's own `battle_tag` (unique per local server
        # session) for actual uniqueness.
        self.replay_tag = replay_tag
        self._pending: dict[str, list[dict[str, object]]] = {}
        self.games_recorded = 0
        self.records_written = 0
        super().__init__(config=config, **player_kwargs)

    def decide(self, battle: AbstractBattle) -> BattleOrder:
        order = super().decide(battle)
        # Gated on the ORDER's shape, not `isinstance(battle, DoubleBattle)` -- this
        # project's format is always doubles (see vgc/config.py's FORMAT_ID) so that
        # would never actually differ in production, but `battle_state_record`/
        # `order_to_action_dict` are both already duck-typed against a battle/order's
        # actual attributes (never an isinstance check), so requiring the concrete
        # poke-env DoubleBattle class here would only make this harder to unit test
        # against a lightweight fake battle for no real behavioral benefit.
        if isinstance(order, DoubleBattleOrder):
            self._buffer_decision(battle, order)
        return order

    def _buffer_decision(self, battle: AbstractBattle, order: DoubleBattleOrder) -> None:
        record = battle_state_record(battle, self.config)
        record["schema"] = SCHEMA_VERSION
        record["decision_kind"] = "turn"
        record["player"] = battle.player_role
        record["turn"] = battle.turn
        record["rating"] = None
        record["replay_id"] = f"{self.replay_tag}-{battle.battle_tag}"
        record["action"] = order_to_action_dict(order)
        self._pending.setdefault(battle.battle_tag, []).append(record)

    def _battle_finished_callback(self, battle: AbstractBattle) -> None:
        records = self._pending.pop(battle.battle_tag, [])
        self.games_recorded += 1
        if not records:
            return
        won = bool(battle.won)
        with self.out_path.open("a") as out_file:
            for record in records:
                record["won"] = won
                out_file.write(json.dumps(record, sort_keys=True) + "\n")
        self.records_written += len(records)
