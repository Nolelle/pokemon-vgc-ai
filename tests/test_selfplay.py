"""Unit tests for `vgc.bc.selfplay`: `order_to_action_dict`/`_forward_target_slot`
(torch-free, pure order->dict conversion) and `RecordingVgcPlayer`'s buffering/labeling
(a real `Player` instance constructed with `start_listening=False` so no network I/O
happens -- no local server needed, matching this module's own "no server in unit tests"
contract).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from poke_env.battle.move import Move
from poke_env.battle.pokemon import Pokemon
from poke_env.player.battle_order import DoubleBattleOrder, PassBattleOrder, SingleBattleOrder

from vgc.bc.selfplay import RecordingVgcPlayer, _forward_target_slot, order_to_action_dict
from vgc.data import load_moves
from vgc.models import PolicyConfig

# --- order_to_action_dict / _forward_target_slot: pure conversion --------------------


def _single(order, move_target: int = 0, mega: bool = False) -> SingleBattleOrder:
    return SingleBattleOrder(order=order, move_target=move_target, mega=mega)


def _double(first: SingleBattleOrder | None, second: SingleBattleOrder | None) -> DoubleBattleOrder:
    return DoubleBattleOrder(
        first_order=first if first is not None else PassBattleOrder(),
        second_order=second if second is not None else PassBattleOrder(),
    )


def test_forward_target_slot_single_target_move_maps_to_opp_slot() -> None:
    move_data = load_moves()["dragonclaw"]
    assert _forward_target_slot(move_data, 1) == "opp0"
    assert _forward_target_slot(move_data, 2) == "opp1"


def test_forward_target_slot_single_target_at_ally_maps_to_ally() -> None:
    move_data = load_moves()["dragonclaw"]
    assert _forward_target_slot(move_data, -1) == "ally"
    assert _forward_target_slot(move_data, -2) == "ally"


def test_forward_target_slot_spread_move_maps_to_spread() -> None:
    assert _forward_target_slot(load_moves()["earthquake"], 0) == "spread"
    assert _forward_target_slot(load_moves()["icywind"], 0) == "spread"


def test_forward_target_slot_self_targeting_move_maps_to_self() -> None:
    # NOTE: raw "self" (vgc.replay_parse's own vocabulary), NOT vgc.bc.policy's
    # already-vocab-mapped "self_or_field" -- see module docstring.
    assert _forward_target_slot(load_moves()["protect"], 0) == "self"


def test_forward_target_slot_none_move_data_returns_none() -> None:
    assert _forward_target_slot(None, 0) is None


def test_order_to_action_dict_move_slot() -> None:
    move = Move("earthquake", gen=9)
    order = _double(_single(move, move_target=0), None)
    action = order_to_action_dict(order)
    assert action["slot0"] == {
        "kind": "move",
        "move_id": "earthquake",
        "target_slot": "spread",
        "mega": False,
    }
    assert action["slot1"] == {"kind": "pass"}


def test_order_to_action_dict_switch_slot() -> None:
    target = Pokemon(gen=9, species="klefki")
    order = _double(None, _single(target))
    action = order_to_action_dict(order)
    assert action["slot1"] == {"kind": "switch", "switch_species": "klefki"}


def test_order_to_action_dict_mega_flag_propagates() -> None:
    move = Move("dragonclaw", gen=9)
    order = _double(_single(move, move_target=1, mega=True), None)
    action = order_to_action_dict(order)
    assert action["slot0"]["mega"] is True


def test_order_to_action_dict_pass_both_slots() -> None:
    order = _double(None, None)
    action = order_to_action_dict(order)
    assert action == {"slot0": {"kind": "pass"}, "slot1": {"kind": "pass"}}


def test_order_to_action_dict_mega_species_resolved_to_base() -> None:
    target = Pokemon(gen=9, species="charizardmegax")
    order = _double(_single(target), None)
    action = order_to_action_dict(order)
    assert action["slot0"] == {"kind": "switch", "switch_species": "charizard"}


# --- RecordingVgcPlayer: buffering + finalize-on-battle-end --------------------------


@dataclass
class _FakeMon:
    species: str
    item: str | None = None
    ability: str | None = None
    boosts: dict = field(default_factory=dict)
    status: Any = None
    current_hp_fraction: float = 1.0
    fainted: bool = False
    moves: dict = field(default_factory=dict)
    revealed: bool = True


class _FakeBattle:
    """Just enough of a `DoubleBattle` for `battle_state_record` +
    `RecordingVgcPlayer`'s own bookkeeping (`player_role`/`turn`/`battle_tag`/`won`).
    """

    def __init__(self, *, battle_tag: str, turn: int = 3, won: bool | None = None):
        self.battle_tag = battle_tag
        self.turn = turn
        self.won = won
        self.player_role = "p1"
        our = _FakeMon(species="garchomp", moves={"earthquake": None})
        opp = _FakeMon(species="charizard", moves={"heatwave": None})
        self.active_pokemon = [our, None]
        self.opponent_active_pokemon = [opp, None]
        self.team = {"our0": our}
        self.opponent_team = {"opp0": opp}
        self.teampreview_opponent_team: list = []
        self.side_conditions = []
        self.opponent_side_conditions = []
        self.fields = []
        self.weather = []


def _make_player(tmp_path: Path, *, replay_tag: str = "selfplay") -> RecordingVgcPlayer:
    team = (Path(__file__).parent.parent / "teams" / "dev.packed.txt").read_text().strip()
    return RecordingVgcPlayer(
        out_path=tmp_path / "records.jsonl",
        replay_tag=replay_tag,
        config=PolicyConfig(accept_open_team_sheet=False),
        team=team,
        battle_format="gen9championsvgc2026regmb",
        start_listening=False,
    )


def _fake_order() -> SimpleNamespace:
    move = Move("earthquake", gen=9)
    return _double(_single(move, move_target=0), None)


def test_buffer_decision_then_finish_writes_labeled_record(tmp_path: Path) -> None:
    player = _make_player(tmp_path)
    battle = _FakeBattle(battle_tag="battle-1", turn=4, won=True)
    order = _fake_order()

    player._buffer_decision(battle, order)
    assert battle.battle_tag in player._pending
    assert not player.out_path.exists()  # nothing written until the battle finishes

    player._battle_finished_callback(battle)

    assert player.out_path.exists()
    lines = player.out_path.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["decision_kind"] == "turn"
    assert record["won"] is True
    assert record["turn"] == 4
    assert record["rating"] is None
    assert record["player"] == "p1"
    assert record["replay_id"] == "selfplay-battle-1"
    assert record["action"]["slot0"]["kind"] == "move"
    assert "state" in record and "our" in record["state"] and "opp" in record["state"]
    assert battle.battle_tag not in player._pending  # consumed


def test_finish_labels_loss_correctly(tmp_path: Path) -> None:
    player = _make_player(tmp_path)
    battle = _FakeBattle(battle_tag="battle-lost", won=False)
    player._buffer_decision(battle, _fake_order())
    player._battle_finished_callback(battle)

    record = json.loads(player.out_path.read_text().strip())
    assert record["won"] is False
    assert record["outcome"] == "loss"


def test_finish_keeps_draw_distinct_from_loss(tmp_path: Path) -> None:
    player = _make_player(tmp_path)
    battle = _FakeBattle(battle_tag="battle-draw", won=None)
    player._buffer_decision(battle, _fake_order())
    player._battle_finished_callback(battle)
    record = json.loads(player.out_path.read_text().strip())
    assert record["outcome"] == "draw"
    assert record["won"] is None


def test_multiple_turns_in_one_battle_all_get_the_same_outcome_label(tmp_path: Path) -> None:
    player = _make_player(tmp_path)
    battle = _FakeBattle(battle_tag="battle-multi", turn=1, won=True)
    player._buffer_decision(battle, _fake_order())
    battle.turn = 2
    player._buffer_decision(battle, _fake_order())
    battle.turn = 3
    player._buffer_decision(battle, _fake_order())

    player._battle_finished_callback(battle)

    lines = player.out_path.read_text().strip().splitlines()
    assert len(lines) == 3
    records = [json.loads(line) for line in lines]
    assert [r["turn"] for r in records] == [1, 2, 3]
    assert all(r["won"] is True for r in records)


def test_finish_with_no_buffered_records_writes_nothing(tmp_path: Path) -> None:
    player = _make_player(tmp_path)
    battle = _FakeBattle(battle_tag="battle-empty", won=True)
    player._battle_finished_callback(battle)  # decide() never called for this battle
    assert not player.out_path.exists()
    assert player.games_recorded == 1


def test_two_battles_append_crash_safely_to_the_same_file(tmp_path: Path) -> None:
    player = _make_player(tmp_path)
    battle_a = _FakeBattle(battle_tag="battle-a", won=True)
    player._buffer_decision(battle_a, _fake_order())
    player._battle_finished_callback(battle_a)

    battle_b = _FakeBattle(battle_tag="battle-b", won=False)
    player._buffer_decision(battle_b, _fake_order())
    player._battle_finished_callback(battle_b)

    lines = player.out_path.read_text().strip().splitlines()
    assert len(lines) == 2
    records = [json.loads(line) for line in lines]
    assert records[0]["replay_id"] == "selfplay-battle-a"
    assert records[1]["replay_id"] == "selfplay-battle-b"
    assert player.records_written == 2
    assert player.games_recorded == 2


def test_decide_buffers_exactly_once_and_returns_the_underlying_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    player = _make_player(tmp_path)
    battle = _FakeBattle(battle_tag="battle-decide", turn=1)
    fixed_order = _fake_order()
    monkeypatch.setattr("vgc.bc.selfplay.VgcPlayer.decide", lambda self, b: fixed_order)

    returned = player.decide(battle)

    assert returned is fixed_order
    assert len(player._pending[battle.battle_tag]) == 1
