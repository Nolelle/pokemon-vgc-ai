"""Unit tests for `vgc.bc`, the Phase 3 behavior-cloning pipeline scaffold (v2: richer
per-slot features, a target head).

`vgc.bc.encoding` has no torch dependency, so its tests run unconditionally (this is
itself part of what's being verified -- see "torch must remain optional" in
`vgc.bc`'s package docstring). Dataset/model/training tests each call
`pytest.importorskip("torch")` locally (NOT at module level) so a torch-less
environment still collects and runs the encoder tests instead of skipping the whole
file.
"""

from __future__ import annotations

import json
import time

import pytest

from vgc.bc.encoding import (
    ABILITY_TO_IDX,
    INDEX_DIM,
    ITEM_TO_IDX,
    MOVE_TO_IDX,
    SLOT_FEATURE_DIM,
    SPECIES_TO_IDX,
    STATE_SCALAR_DIM,
    TARGET_TO_IDX,
    encode_action,
    encode_state,
    encode_target,
    flatten_state,
)

# --- shared synthetic record builder --------------------------------------------------


def _mon(
    species: str,
    hp_fraction: float = 1.0,
    status=None,
    boosts=None,
    item=None,
    ability=None,
    mega=False,
    revealed_moves=None,
):
    return {
        "species": species,
        "hp_fraction": hp_fraction,
        "status": status,
        "boosts": boosts or {},
        "item": item,
        "ability": ability,
        "mega": mega,
        "revealed_moves": revealed_moves or [],
    }


def _bench_mon(species_id: str, hp_fraction: float = 1.0, status=None) -> dict:
    return {"species_id": species_id, "hp_fraction": hp_fraction, "status": status}


def _turn_record(
    *,
    replay_id: str = "test-replay-1",
    player: str = "p1",
    rating: int | None = 1300,
    turn: int = 3,
    our_active: list,
    opp_active: list,
    action: dict,
    our_side_conditions: list | None = None,
    opp_side_conditions: list | None = None,
    weather: str | None = None,
    terrain: str | None = None,
    trick_room: bool = False,
    our_bench: list | None = None,
    opp_bench: list | None = None,
    won: bool | None = False,
    our_preview_species: list | None = None,
    opp_preview_species: list | None = None,
    our_unseen_count: int | None = None,
    opp_unseen_count: int | None = None,
) -> dict:
    record = {
        "replay_id": replay_id,
        "rating": rating,
        "schema": 4,
        "player": player,
        "turn": turn,
        "decision_kind": "turn",
        "state": {
            "our": {
                "active": our_active,
                "bench": our_bench or [],
                "side_conditions": our_side_conditions or [],
                "preview_species": our_preview_species or [],
                "unseen_count": our_unseen_count or 0,
            },
            "opp": {
                "active": opp_active,
                "bench": opp_bench or [],
                "side_conditions": opp_side_conditions or [],
                "preview_species": opp_preview_species or [],
                "unseen_count": opp_unseen_count or 0,
            },
            "field": {
                "weather": weather,
                "terrain": terrain,
                "trick_room": trick_room,
                "turn": turn,
            },
        },
        "action": action,
    }
    if won is not None:
        record["won"] = won
    return record


# --- encode_state: fixed sizes, determinism, missing/unknown handling ----------------


def test_encode_state_fixed_shapes() -> None:
    record = _turn_record(
        our_active=[_mon("garchomp"), _mon("klefki")],
        opp_active=[_mon("charizard"), None],
        action={"slot0": {"kind": "move", "move_id": "earthquake"}, "slot1": {"kind": "pass"}},
    )
    state = encode_state(record)

    assert state["species_idx_active"].shape == (4,)
    assert state["species_idx_bench"].shape == (8,)
    assert state["species_idx_preview"].shape == (12,)
    assert state["item_idx"].shape == (4,)
    assert state["ability_idx"].shape == (4,)
    assert state["move_idx"].shape == (4, 4)
    assert state["hp_fraction"].shape == (4,)
    assert state["status"].shape == (4, 7)
    assert state["boosts"].shape == (4, 5)
    assert state["mega"].shape == (4,)
    assert state["bench_hp_fraction"].shape == (8,)
    assert state["fainted_count"].shape == (2,)
    assert state["weather"].shape == (5,)
    assert state["terrain"].shape == (5,)
    assert state["trick_room"].shape == (1,)
    assert state["side_conditions"].shape == (8,)
    assert state["turn"].shape == (1,)
    assert state["alive_known_count"].shape == (2,)
    assert state["alive_known_mean_hp"].shape == (2,)
    assert state["preview_unseen_count"].shape == (2,)

    index_array, scalar_array = flatten_state(state)
    assert index_array.shape == (INDEX_DIM,)
    assert scalar_array.shape == (STATE_SCALAR_DIM,)


def test_encode_state_is_deterministic() -> None:
    record = _turn_record(
        our_active=[
            _mon("garchomp", hp_fraction=0.7, boosts={"atk": 2}, revealed_moves=["earthquake"]),
            _mon("klefki"),
        ],
        opp_active=[_mon("charizard", status="brn"), None],
        action={"slot0": {"kind": "move", "move_id": "earthquake"}, "slot1": {"kind": "pass"}},
    )
    state_a = encode_state(record)
    state_b = encode_state(record)
    for key in state_a:
        assert (state_a[key] == state_b[key]).all()


def test_encode_state_missing_slot_is_all_zero_except_pad_tokens() -> None:
    record = _turn_record(
        our_active=[_mon("garchomp"), None],
        opp_active=[None, None],
        action={"slot0": {"kind": "move", "move_id": "earthquake"}, "slot1": {"kind": "pass"}},
    )
    state = encode_state(record)
    for idx in (1, 2, 3):
        assert state["species_idx_active"][idx] == SPECIES_TO_IDX["<pad>"]
        assert state["item_idx"][idx] == ITEM_TO_IDX["<unk>"]
        assert state["ability_idx"][idx] == ABILITY_TO_IDX["<unk>"]
        assert (state["move_idx"][idx] == MOVE_TO_IDX["<pad>"]).all()
        assert state["hp_fraction"][idx] == 0.0
        assert state["status"][idx].sum() == 0.0
        assert (state["boosts"][idx] == 0.0).all()
        assert state["mega"][idx] == 0.0
    # Both active slots empty for opp -> both opp fainted_count contributions are "empty".
    assert state["fainted_count"][1] == 1.0  # opp: 2/2 slots empty -> 1.0
    assert state["fainted_count"][0] == 0.5  # our: 1/2 slots empty -> 0.5


def test_encode_state_unknown_species_maps_to_unk() -> None:
    record = _turn_record(
        our_active=[_mon("not_a_real_species_xyz"), None],
        opp_active=[None, None],
        action={"slot0": {"kind": "pass"}, "slot1": {"kind": "pass"}},
    )
    state = encode_state(record)
    assert state["species_idx_active"][0] == SPECIES_TO_IDX["<unk>"]
    assert state["hp_fraction"][0] == 1.0


def test_encode_state_hand_built_values() -> None:
    record = _turn_record(
        our_active=[
            _mon(
                "garchomp",
                hp_fraction=0.5,
                status="brn",
                boosts={"atk": 3},
                mega=True,
                item="garchompite",
                ability="roughskin",
                revealed_moves=["dragonclaw", "earthquake"],
            ),
            None,
        ],
        opp_active=[None, None],
        action={"slot0": {"kind": "pass"}, "slot1": {"kind": "pass"}},
        our_side_conditions=["tailwind", "reflect"],
        weather="sun",
        terrain="electric",
        trick_room=True,
        turn=10,
    )
    state = encode_state(record)

    assert state["hp_fraction"][0] == pytest.approx(0.5)
    assert state["status"][0].tolist() == [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert state["boosts"][0].tolist() == pytest.approx([0.5, 0.0, 0.0, 0.0, 0.0])
    assert state["mega"][0] == 1.0
    assert state["item_idx"][0] == ITEM_TO_IDX["garchompite"]
    assert state["ability_idx"][0] == ABILITY_TO_IDX["roughskin"]
    moves = {MOVE_TO_IDX["dragonclaw"], MOVE_TO_IDX["earthquake"]}
    assert set(state["move_idx"][0].tolist()) - {MOVE_TO_IDX["<pad>"]} == moves
    assert state["weather"].tolist() == [0.0, 1.0, 0.0, 0.0, 0.0]
    assert state["terrain"].tolist() == [0.0, 1.0, 0.0, 0.0, 0.0]
    assert state["trick_room"][0] == 1.0
    assert state["side_conditions"].tolist() == [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert state["turn"][0] == pytest.approx(10 / 20.0)


def test_encode_state_turn_scaling_caps_at_one() -> None:
    record = _turn_record(
        our_active=[None, None],
        opp_active=[None, None],
        action={"slot0": {"kind": "pass"}, "slot1": {"kind": "pass"}},
        turn=99,
    )
    state = encode_state(record)
    assert state["turn"][0] == 1.0


# --- item/ability: unrevealed -> <unk>, revealed -> real index ------------------------


def test_encode_state_item_unrevealed_is_unk() -> None:
    record = _turn_record(
        our_active=[_mon("garchomp", item=None), None],
        opp_active=[None, None],
        action={"slot0": {"kind": "pass"}, "slot1": {"kind": "pass"}},
    )
    state = encode_state(record)
    assert state["item_idx"][0] == ITEM_TO_IDX["<unk>"]


def test_encode_state_item_revealed_maps_to_real_index() -> None:
    record = _turn_record(
        our_active=[_mon("garchomp", item="lifeorb"), None],
        opp_active=[None, None],
        action={"slot0": {"kind": "pass"}, "slot1": {"kind": "pass"}},
    )
    state = encode_state(record)
    assert state["item_idx"][0] == ITEM_TO_IDX["lifeorb"]
    assert state["item_idx"][0] not in (ITEM_TO_IDX["<unk>"], ITEM_TO_IDX["<none>"])


def test_encode_state_ability_unrevealed_is_unk() -> None:
    record = _turn_record(
        our_active=[_mon("garchomp", ability=None), None],
        opp_active=[None, None],
        action={"slot0": {"kind": "pass"}, "slot1": {"kind": "pass"}},
    )
    state = encode_state(record)
    assert state["ability_idx"][0] == ABILITY_TO_IDX["<unk>"]


def test_encode_state_ability_revealed_maps_to_real_index() -> None:
    record = _turn_record(
        our_active=[_mon("garchomp", ability="roughskin"), None],
        opp_active=[None, None],
        action={"slot0": {"kind": "pass"}, "slot1": {"kind": "pass"}},
    )
    state = encode_state(record)
    assert state["ability_idx"][0] == ABILITY_TO_IDX["roughskin"]


# --- revealed-move pooling: padding, truncation, all-unrevealed -----------------------


def test_encode_state_no_revealed_moves_is_all_pad() -> None:
    record = _turn_record(
        our_active=[_mon("garchomp", revealed_moves=[]), None],
        opp_active=[None, None],
        action={"slot0": {"kind": "pass"}, "slot1": {"kind": "pass"}},
    )
    state = encode_state(record)
    assert (state["move_idx"][0] == MOVE_TO_IDX["<pad>"]).all()


def test_encode_state_fewer_than_four_revealed_moves_pads_rest() -> None:
    record = _turn_record(
        our_active=[_mon("garchomp", revealed_moves=["earthquake"]), None],
        opp_active=[None, None],
        action={"slot0": {"kind": "pass"}, "slot1": {"kind": "pass"}},
    )
    state = encode_state(record)
    move_row = state["move_idx"][0].tolist()
    assert move_row.count(MOVE_TO_IDX["earthquake"]) == 1
    assert move_row.count(MOVE_TO_IDX["<pad>"]) == 3


def test_encode_state_more_than_four_revealed_moves_truncates_to_four() -> None:
    # Not realistically possible in-game (max 4 real moves), but the encoder should be
    # defensively robust to it rather than crash/overflow the fixed-size array.
    record = _turn_record(
        our_active=[
            _mon(
                "garchomp",
                revealed_moves=["earthquake", "dragonclaw", "protect", "ironhead", "swordsdance"],
            ),
            None,
        ],
        opp_active=[None, None],
        action={"slot0": {"kind": "pass"}, "slot1": {"kind": "pass"}},
    )
    state = encode_state(record)
    assert state["move_idx"][0].shape == (4,)
    assert MOVE_TO_IDX["<pad>"] not in state["move_idx"][0].tolist()


# --- bench identity: species + hp fraction, padded to 4 per side ----------------------


def test_encode_state_bench_identity_and_padding() -> None:
    record = _turn_record(
        our_active=[_mon("garchomp"), None],
        opp_active=[None, None],
        action={"slot0": {"kind": "pass"}, "slot1": {"kind": "pass"}},
        our_bench=[_bench_mon("klefki", hp_fraction=0.5), _bench_mon("sylveon", hp_fraction=1.0)],
    )
    state = encode_state(record)
    # our bench occupies indices 0-3 of species_idx_bench/bench_hp_fraction.
    assert state["species_idx_bench"][0] == SPECIES_TO_IDX["klefki"]
    assert state["species_idx_bench"][1] == SPECIES_TO_IDX["sylveon"]
    assert state["bench_hp_fraction"][0] == pytest.approx(0.5)
    assert state["bench_hp_fraction"][1] == pytest.approx(1.0)
    # Remaining 2 our-bench slots and all 4 opp-bench slots are padding.
    for idx in (2, 3, 4, 5, 6, 7):
        assert state["species_idx_bench"][idx] == SPECIES_TO_IDX["<pad>"]
        assert state["bench_hp_fraction"][idx] == 0.0


# --- v4: preview species (6-slot, per side) + resource-state scalars -----------------


def test_encode_state_preview_species_indices_and_padding() -> None:
    record = _turn_record(
        our_active=[_mon("garchomp"), None],
        opp_active=[None, None],
        action={"slot0": {"kind": "pass"}, "slot1": {"kind": "pass"}},
        our_preview_species=["garchomp", "klefki", "incineroar"],
        opp_preview_species=["charizard", "gholdengo"],
    )
    state = encode_state(record)
    assert state["species_idx_preview"].shape == (12,)
    # our0..5 occupy indices 0-5, opp0..5 occupy indices 6-11.
    assert state["species_idx_preview"][0] == SPECIES_TO_IDX["garchomp"]
    assert state["species_idx_preview"][1] == SPECIES_TO_IDX["klefki"]
    assert state["species_idx_preview"][2] == SPECIES_TO_IDX["incineroar"]
    for idx in (3, 4, 5):
        assert state["species_idx_preview"][idx] == SPECIES_TO_IDX["<pad>"]
    assert state["species_idx_preview"][6] == SPECIES_TO_IDX["charizard"]
    assert state["species_idx_preview"][7] == SPECIES_TO_IDX["gholdengo"]
    for idx in (8, 9, 10, 11):
        assert state["species_idx_preview"][idx] == SPECIES_TO_IDX["<pad>"]


def test_encode_state_preview_species_missing_defaults_to_all_pad() -> None:
    record = _turn_record(
        our_active=[_mon("garchomp"), None],
        opp_active=[None, None],
        action={"slot0": {"kind": "pass"}, "slot1": {"kind": "pass"}},
    )
    state = encode_state(record)
    assert (state["species_idx_preview"] == SPECIES_TO_IDX["<pad>"]).all()


def test_encode_state_preview_unseen_count_scaled_by_six() -> None:
    record = _turn_record(
        our_active=[_mon("garchomp"), None],
        opp_active=[None, None],
        action={"slot0": {"kind": "pass"}, "slot1": {"kind": "pass"}},
        our_unseen_count=2,
        opp_unseen_count=4,
    )
    state = encode_state(record)
    assert state["preview_unseen_count"].tolist() == pytest.approx([2 / 6, 4 / 6])


def test_encode_state_alive_known_count_and_mean_hp_from_active_and_bench() -> None:
    record = _turn_record(
        our_active=[_mon("garchomp", hp_fraction=0.5), _mon("klefki", hp_fraction=1.0)],
        opp_active=[_mon("charizard", hp_fraction=0.25), None],
        action={"slot0": {"kind": "pass"}, "slot1": {"kind": "pass"}},
        our_bench=[_bench_mon("incineroar", hp_fraction=0.75)],
    )
    state = encode_state(record)
    # our: 2 active + 1 bench = 3 alive known mons, mean hp (0.5+1.0+0.75)/3.
    assert state["alive_known_count"][0] == pytest.approx(3 / 4.0)
    assert state["alive_known_mean_hp"][0] == pytest.approx((0.5 + 1.0 + 0.75) / 3)
    # opp: 1 active, no bench -> mean hp is just that slot's fraction.
    assert state["alive_known_count"][1] == pytest.approx(1 / 4.0)
    assert state["alive_known_mean_hp"][1] == pytest.approx(0.25)


def test_encode_state_alive_known_mean_hp_zero_when_no_known_mons() -> None:
    record = _turn_record(
        our_active=[None, None],
        opp_active=[None, None],
        action={"slot0": {"kind": "pass"}, "slot1": {"kind": "pass"}},
    )
    state = encode_state(record)
    assert state["alive_known_count"][0] == 0.0
    assert state["alive_known_mean_hp"][0] == 0.0


# --- encode_action: move id, switch/pass, unknown -> None ----------------------------


def test_encode_action_move_id() -> None:
    record = _turn_record(
        our_active=[_mon("garchomp"), None],
        opp_active=[None, None],
        action={"slot0": {"kind": "move", "move_id": "earthquake"}, "slot1": {"kind": "pass"}},
    )
    assert encode_action(record, 0) == MOVE_TO_IDX["earthquake"]
    assert encode_action(record, 1) == MOVE_TO_IDX["<pass>"]


def test_encode_action_switch() -> None:
    record = _turn_record(
        our_active=[_mon("garchomp"), None],
        opp_active=[None, None],
        action={"slot0": {"kind": "switch", "switch_species": "klefki"}, "slot1": {"kind": "pass"}},
    )
    assert encode_action(record, 0) == MOVE_TO_IDX["<switch>"]


def test_encode_action_unknown_move_returns_none() -> None:
    record = _turn_record(
        our_active=[_mon("garchomp"), None],
        opp_active=[None, None],
        action={
            "slot0": {"kind": "move", "move_id": "not_a_real_move_xyz"},
            "slot1": {"kind": "pass"},
        },
    )
    assert encode_action(record, 0) is None


def test_encode_action_non_turn_record_returns_none() -> None:
    record = _turn_record(
        our_active=[_mon("garchomp"), None],
        opp_active=[None, None],
        action={"slot0": {"kind": "move", "move_id": "earthquake"}, "slot1": {"kind": "pass"}},
    )
    record["decision_kind"] = "teampreview"
    assert encode_action(record, 0) is None


def test_encode_action_missing_slot_returns_none() -> None:
    record = _turn_record(
        our_active=[_mon("garchomp"), None],
        opp_active=[None, None],
        action={"slot0": {"kind": "move", "move_id": "earthquake"}},
    )
    assert encode_action(record, 1) is None


# --- encode_target: opp0/opp1/ally/self_or_field/spread/<none>, None-skip ------------


def test_encode_target_opp_and_ally_and_spread() -> None:
    record = _turn_record(
        our_active=[_mon("garchomp"), _mon("klefki")],
        opp_active=[_mon("charizard"), _mon("incineroar")],
        action={
            "slot0": {"kind": "move", "move_id": "earthquake", "target_slot": "opp0"},
            "slot1": {"kind": "move", "move_id": "helpinghand", "target_slot": "ally"},
        },
    )
    assert encode_target(record, 0) == TARGET_TO_IDX["opp0"]
    assert encode_target(record, 1) == TARGET_TO_IDX["ally"]


def test_encode_target_self_maps_to_self_or_field() -> None:
    record = _turn_record(
        our_active=[_mon("klefki"), None],
        opp_active=[None, None],
        action={"slot0": {"kind": "move", "move_id": "protect", "target_slot": "self"}},
    )
    assert encode_target(record, 0) == TARGET_TO_IDX["self_or_field"]


def test_encode_target_spread() -> None:
    record = _turn_record(
        our_active=[_mon("garchomp"), None],
        opp_active=[None, None],
        action={"slot0": {"kind": "move", "move_id": "earthquake", "target_slot": "spread"}},
    )
    assert encode_target(record, 0) == TARGET_TO_IDX["spread"]


def test_encode_target_switch_and_pass_are_none_class() -> None:
    record = _turn_record(
        our_active=[_mon("garchomp"), _mon("klefki")],
        opp_active=[None, None],
        action={
            "slot0": {"kind": "switch", "switch_species": "sylveon"},
            "slot1": {"kind": "pass"},
        },
    )
    assert encode_target(record, 0) == TARGET_TO_IDX["<none>"]
    assert encode_target(record, 1) == TARGET_TO_IDX["<none>"]


def test_encode_target_unknown_target_returns_none_for_skip() -> None:
    # Mirrors a blocked |cant|-attempted move: kind="move" but no target_slot info.
    record = _turn_record(
        our_active=[_mon("garchomp"), None],
        opp_active=[None, None],
        action={"slot0": {"kind": "move", "move_id": "dragonclaw", "target_slot": None}},
    )
    assert encode_target(record, 0) is None


def test_encode_target_non_turn_record_returns_none() -> None:
    record = _turn_record(
        our_active=[_mon("garchomp"), None],
        opp_active=[None, None],
        action={"slot0": {"kind": "move", "move_id": "earthquake", "target_slot": "opp0"}},
    )
    record["decision_kind"] = "forced_switch"
    assert encode_target(record, 0) is None


# --- BcTurnDataset: split determinism + no replay overlap + target masking ----------


def _write_jsonl(tmp_path, records: list[dict]):
    path = tmp_path / "decisions.jsonl"
    with path.open("w") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")
    return path


def _synthetic_records(n_replays: int, turns_per_replay: int = 3, rating: int = 1300) -> list[dict]:
    records = []
    for replay_idx in range(n_replays):
        replay_id = f"replay-{replay_idx}"
        # Alternate winner so the value label has both classes -- p1 wins even replays,
        # p2 wins odd ones (mirrors vgc.replay_parse's per-record "won" contract: True
        # for the winning player's records, False for the other player's).
        p1_won = replay_idx % 2 == 0
        for turn in range(1, turns_per_replay + 1):
            for player in ("p1", "p2"):
                won = p1_won if player == "p1" else not p1_won
                records.append(
                    _turn_record(
                        replay_id=replay_id,
                        player=player,
                        rating=rating,
                        turn=turn,
                        our_active=[_mon("garchomp"), _mon("klefki")],
                        opp_active=[_mon("charizard"), _mon("incineroar")],
                        action={
                            "slot0": {
                                "kind": "move",
                                "move_id": "earthquake",
                                "target_slot": "opp0",
                            },
                            "slot1": {"kind": "move", "move_id": "protect", "target_slot": "self"},
                        },
                        won=won,
                    )
                )
    return records


def test_bc_turn_dataset_split_is_deterministic(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    from vgc.bc.dataset import BcTurnDataset

    path = _write_jsonl(tmp_path, _synthetic_records(n_replays=40))

    train_a = BcTurnDataset(path, min_rating=1000, split="train")
    train_b = BcTurnDataset(path, min_rating=1000, split="val")
    train_c = BcTurnDataset(path, min_rating=1000, split="train")

    assert train_a.replays_included == train_c.replays_included
    assert len(train_a) == len(train_c)
    assert train_a.replays_included.isdisjoint(train_b.replays_included)
    assert train_a.replays_included | train_b.replays_included == {f"replay-{i}" for i in range(40)}
    del torch  # only imported to trigger the skip; unused otherwise


def test_bc_turn_dataset_no_replay_straddles_splits(tmp_path) -> None:
    pytest.importorskip("torch")
    from vgc.bc.dataset import BcTurnDataset

    path = _write_jsonl(tmp_path, _synthetic_records(n_replays=60))

    train_ds = BcTurnDataset(path, min_rating=1000, split="train", val_fraction=0.3)
    val_ds = BcTurnDataset(path, min_rating=1000, split="val", val_fraction=0.3)

    assert train_ds.replays_included.isdisjoint(val_ds.replays_included)
    assert len(train_ds) > 0
    assert len(val_ds) > 0


def test_bc_turn_dataset_min_rating_filters_records(tmp_path) -> None:
    pytest.importorskip("torch")
    from vgc.bc.dataset import BcTurnDataset

    low = _synthetic_records(n_replays=5, rating=1000)
    high = _synthetic_records(n_replays=5, rating=1400)
    for record in high:
        record["replay_id"] = f"high-{record['replay_id']}"
    path = _write_jsonl(tmp_path, low + high)

    dataset = BcTurnDataset(path, min_rating=1200, split="train", val_fraction=0.0)
    included_prefixes = {replay_id.split("-")[0] for replay_id in dataset.replays_included}
    assert included_prefixes == {"high"}


def test_bc_turn_dataset_skips_unrecognized_move_actions(tmp_path) -> None:
    pytest.importorskip("torch")
    from vgc.bc.dataset import BcTurnDataset

    record = _turn_record(
        replay_id="replay-skip",
        rating=1300,
        our_active=[_mon("garchomp"), _mon("klefki")],
        opp_active=[_mon("charizard"), _mon("incineroar")],
        action={
            "slot0": {"kind": "move", "move_id": "not_a_real_move_xyz"},
            "slot1": {"kind": "move", "move_id": "protect", "target_slot": "self"},
        },
    )
    path = _write_jsonl(tmp_path, [record])

    dataset = BcTurnDataset(path, min_rating=1000, split="train", val_fraction=0.0)
    assert len(dataset) == 1  # only slot1 encodes successfully
    assert dataset.skipped == 1


def test_bc_turn_dataset_masks_samples_with_unknown_target(tmp_path) -> None:
    pytest.importorskip("torch")
    from vgc.bc.dataset import BcTurnDataset

    record = _turn_record(
        replay_id="replay-masked-target",
        rating=1300,
        our_active=[_mon("garchomp"), _mon("klefki")],
        opp_active=[_mon("charizard"), None],
        action={
            # A real, known target -- has_target should be True, target_idx meaningful.
            "slot0": {"kind": "move", "move_id": "earthquake", "target_slot": "opp0"},
            # A blocked |cant|-attempted move -- move label present, target unknown.
            "slot1": {"kind": "move", "move_id": "protect", "target_slot": None},
        },
    )
    path = _write_jsonl(tmp_path, [record])
    dataset = BcTurnDataset(path, min_rating=1000, split="train", val_fraction=0.0)

    assert len(dataset) == 2  # neither sample dropped -- only the target is masked
    has_target_values = sorted(dataset[i][4].item() for i in range(len(dataset)))
    assert has_target_values == [0.0, 1.0]


# --- BcTurnDataset: value label (schema 3 "won") + turn field -----------------------


def test_bc_turn_dataset_value_label_matches_won(tmp_path) -> None:
    pytest.importorskip("torch")
    from vgc.bc.dataset import BcTurnDataset

    won_record = _turn_record(
        replay_id="replay-won",
        rating=1300,
        turn=5,
        our_active=[_mon("garchomp"), _mon("klefki")],
        opp_active=[_mon("charizard"), None],
        action={
            "slot0": {"kind": "move", "move_id": "earthquake", "target_slot": "opp0"},
            "slot1": {"kind": "move", "move_id": "protect", "target_slot": "self"},
        },
        won=True,
    )
    lost_record = _turn_record(
        replay_id="replay-lost",
        rating=1300,
        turn=7,
        our_active=[_mon("garchomp"), _mon("klefki")],
        opp_active=[_mon("charizard"), None],
        action={
            "slot0": {"kind": "move", "move_id": "earthquake", "target_slot": "opp0"},
            "slot1": {"kind": "move", "move_id": "protect", "target_slot": "self"},
        },
        won=False,
    )
    path = _write_jsonl(tmp_path, [won_record, lost_record])
    dataset = BcTurnDataset(path, min_rating=1000, split="train", val_fraction=0.0)

    assert len(dataset) == 4  # 2 slots x 2 records
    by_turn = {int(dataset[i][7].item()): dataset[i] for i in range(len(dataset))}
    won_sample = by_turn[5]
    lost_sample = by_turn[7]
    assert won_sample[5].item() == 1.0  # value_label
    assert won_sample[6].item() == 1.0  # has_value
    assert lost_sample[5].item() == 0.0
    assert lost_sample[6].item() == 1.0


def test_bc_turn_dataset_missing_won_key_masks_value_out(tmp_path) -> None:
    pytest.importorskip("torch")
    from vgc.bc.dataset import BcTurnDataset

    record = _turn_record(
        replay_id="replay-pre-schema3",
        rating=1300,
        our_active=[_mon("garchomp"), None],
        opp_active=[_mon("charizard"), None],
        action={"slot0": {"kind": "move", "move_id": "earthquake", "target_slot": "opp0"}},
        won=None,  # simulates a pre-schema-3 record with no "won" key at all
    )
    assert "won" not in record
    path = _write_jsonl(tmp_path, [record])
    dataset = BcTurnDataset(path, min_rating=1000, split="train", val_fraction=0.0)

    assert len(dataset) == 1
    assert dataset[0][6].item() == 0.0  # has_value == False
    assert dataset.no_value_count == 1


# --- model/training smoke test: three heads, masked target/value loss ----------------


def test_bc_policy_net_forward_returns_three_heads_by_default() -> None:
    torch = pytest.importorskip("torch")
    from vgc.bc.model import BcPolicyNet

    model = BcPolicyNet()
    batch_size = 4
    index_array = torch.zeros((batch_size, INDEX_DIM), dtype=torch.long)
    scalars = torch.zeros((batch_size, STATE_SCALAR_DIM + SLOT_FEATURE_DIM), dtype=torch.float32)

    move_logits, target_logits, value_logit = model(index_array, scalars)

    assert move_logits.shape == (batch_size, len(model.move_head.bias))
    assert target_logits.shape == (batch_size, len(model.target_head.bias))
    assert value_logit.shape == (batch_size,)


def test_bc_policy_net_value_only_head_config_omits_move_and_target() -> None:
    torch = pytest.importorskip("torch")
    from vgc.bc.model import BcPolicyNet

    model = BcPolicyNet(heads=("value",))
    assert model.move_head is None
    assert model.target_head is None
    assert model.value_head is not None

    batch_size = 3
    index_array = torch.zeros((batch_size, INDEX_DIM), dtype=torch.long)
    scalars = torch.zeros((batch_size, STATE_SCALAR_DIM + SLOT_FEATURE_DIM), dtype=torch.float32)
    move_logits, target_logits, value_logit = model(index_array, scalars)
    assert move_logits is None
    assert target_logits is None
    assert value_logit.shape == (batch_size,)
    # A value-only model carries no move/target parameters at all.
    state_dict_keys = model.state_dict().keys()
    assert not any(key.startswith("move_head") for key in state_dict_keys)
    assert not any(key.startswith("target_head") for key in state_dict_keys)


def test_bc_policy_net_multi_task_loss_decreases_over_gradient_steps(tmp_path) -> None:
    """Masked multi-task loss smoke test: move + target + value losses jointly, loss
    should decrease over a handful of gradient steps the same way the single/two-head
    version always has.
    """
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader

    from vgc.bc.dataset import BcTurnDataset
    from vgc.bc.model import BcPolicyNet

    path = _write_jsonl(tmp_path, _synthetic_records(n_replays=30))
    dataset = BcTurnDataset(path, min_rating=1000, split="train", val_fraction=0.0)
    assert len(dataset) > 0
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = BcPolicyNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    move_criterion = torch.nn.CrossEntropyLoss()
    target_criterion = torch.nn.CrossEntropyLoss(reduction="none")
    value_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

    losses: list[float] = []
    steps = 0
    start = time.time()
    while steps < 30:
        for (
            index_array,
            scalars,
            move_idx,
            target_idx,
            has_target,
            value_label,
            has_value,
            _turn,
        ) in loader:
            optimizer.zero_grad()
            move_logits, target_logits, value_logit = model(index_array, scalars)
            move_loss = move_criterion(move_logits, move_idx)

            per_sample_target_loss = target_criterion(target_logits, target_idx)
            target_mask_sum = has_target.sum()
            target_loss = (
                (per_sample_target_loss * has_target).sum() / target_mask_sum
                if target_mask_sum > 0
                else torch.zeros(())
            )

            per_sample_value_loss = value_criterion(value_logit, value_label)
            value_mask_sum = has_value.sum()
            value_loss = (
                (per_sample_value_loss * has_value).sum() / value_mask_sum
                if value_mask_sum > 0
                else torch.zeros(())
            )

            loss = move_loss + 0.5 * target_loss + 1.0 * value_loss
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            steps += 1
            if steps >= 30:
                break
    elapsed = time.time() - start

    assert sum(losses[-5:]) / 5 < sum(losses[:5]) / 5
    assert elapsed < 5.0  # fast enough that this doesn't need @pytest.mark.slow


def test_bc_policy_net_handles_batch_with_no_targets_or_values_at_all() -> None:
    """A batch where every sample's target AND value is masked out must not
    divide-by-zero for either head.
    """
    torch = pytest.importorskip("torch")
    from vgc.bc.model import BcPolicyNet

    model = BcPolicyNet()
    batch_size = 4
    index_array = torch.zeros((batch_size, INDEX_DIM), dtype=torch.long)
    scalars = torch.zeros((batch_size, STATE_SCALAR_DIM + SLOT_FEATURE_DIM), dtype=torch.float32)
    target_idx = torch.zeros((batch_size,), dtype=torch.long)
    has_target = torch.zeros((batch_size,), dtype=torch.float32)  # all masked out
    value_label = torch.zeros((batch_size,), dtype=torch.float32)
    has_value = torch.zeros((batch_size,), dtype=torch.float32)  # all masked out

    move_logits, target_logits, value_logit = model(index_array, scalars)
    target_criterion = torch.nn.CrossEntropyLoss(reduction="none")
    per_sample_target_loss = target_criterion(target_logits, target_idx)
    mask_sum = has_target.sum()
    target_loss = (
        (per_sample_target_loss * has_target).sum() / mask_sum if mask_sum > 0 else torch.zeros(())
    )
    value_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
    per_sample_value_loss = value_criterion(value_logit, value_label)
    value_mask_sum = has_value.sum()
    value_loss = (
        (per_sample_value_loss * has_value).sum() / value_mask_sum
        if value_mask_sum > 0
        else torch.zeros(())
    )
    assert move_logits.shape == (batch_size, len(model.move_head.bias))
    assert float(target_loss.item()) == 0.0
    assert float(value_loss.item()) == 0.0


def test_compute_auc_perfect_separation_is_one() -> None:
    pytest.importorskip("torch")
    from vgc.bc.train import compute_auc

    scores = [0.1, 0.2, 0.8, 0.9]
    labels = [0, 0, 1, 1]
    assert compute_auc(scores, labels) == pytest.approx(1.0)


def test_compute_auc_inverted_separation_is_zero() -> None:
    pytest.importorskip("torch")
    from vgc.bc.train import compute_auc

    scores = [0.9, 0.8, 0.2, 0.1]
    labels = [0, 0, 1, 1]
    assert compute_auc(scores, labels) == pytest.approx(0.0)


def test_compute_auc_random_ish_is_near_half() -> None:
    pytest.importorskip("torch")
    from vgc.bc.train import compute_auc

    scores = [0.5, 0.5, 0.5, 0.5]
    labels = [0, 1, 0, 1]
    assert compute_auc(scores, labels) == pytest.approx(0.5)


def test_compute_auc_all_one_class_returns_half() -> None:
    pytest.importorskip("torch")
    from vgc.bc.train import compute_auc

    assert compute_auc([0.1, 0.9], [1, 1]) == 0.5
    assert compute_auc([], []) == 0.5
