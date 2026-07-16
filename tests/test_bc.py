"""Unit tests for `vgc.bc`, the Phase 3 behavior-cloning pipeline scaffold.

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
    MOVE_TO_IDX,
    SPECIES_TO_IDX,
    STATE_SCALAR_DIM,
    encode_action,
    encode_state,
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
):
    return {
        "species": species,
        "hp_fraction": hp_fraction,
        "status": status,
        "boosts": boosts or {},
        "item": item,
        "ability": ability,
        "mega": mega,
    }


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
) -> dict:
    return {
        "replay_id": replay_id,
        "rating": rating,
        "player": player,
        "turn": turn,
        "decision_kind": "turn",
        "state": {
            "our": {
                "active": our_active,
                "bench": our_bench or [],
                "side_conditions": our_side_conditions or [],
            },
            "opp": {
                "active": opp_active,
                "bench": opp_bench or [],
                "side_conditions": opp_side_conditions or [],
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


# --- encode_state: fixed sizes, determinism, missing/unknown handling ----------------


def test_encode_state_fixed_shapes() -> None:
    record = _turn_record(
        our_active=[_mon("garchomp"), _mon("klefki")],
        opp_active=[_mon("charizard"), None],
        action={"slot0": {"kind": "move", "move_id": "earthquake"}, "slot1": {"kind": "pass"}},
    )
    state = encode_state(record)

    assert state["species_idx"].shape == (4,)
    assert state["hp_fraction"].shape == (4,)
    assert state["status"].shape == (4, 7)
    assert state["boosts"].shape == (4, 5)
    assert state["mega"].shape == (4,)
    assert state["bench_count"].shape == (2,)
    assert state["bench_mean_hp"].shape == (2,)
    assert state["weather"].shape == (5,)
    assert state["terrain"].shape == (5,)
    assert state["trick_room"].shape == (1,)
    assert state["side_conditions"].shape == (8,)
    assert state["turn"].shape == (1,)

    flat = flatten_state(state)
    assert flat.shape == (STATE_SCALAR_DIM,)


def test_encode_state_is_deterministic() -> None:
    record = _turn_record(
        our_active=[_mon("garchomp", hp_fraction=0.7, boosts={"atk": 2}), _mon("klefki")],
        opp_active=[_mon("charizard", status="brn"), None],
        action={"slot0": {"kind": "move", "move_id": "earthquake"}, "slot1": {"kind": "pass"}},
    )
    state_a = encode_state(record)
    state_b = encode_state(record)
    for key in state_a:
        assert (state_a[key] == state_b[key]).all()


def test_encode_state_missing_slot_is_all_zero_except_pad_species() -> None:
    record = _turn_record(
        our_active=[_mon("garchomp"), None],
        opp_active=[None, None],
        action={"slot0": {"kind": "move", "move_id": "earthquake"}, "slot1": {"kind": "pass"}},
    )
    state = encode_state(record)
    # Slot index 1 (our1) and 2,3 (opp0, opp1) are all empty.
    for idx in (1, 2, 3):
        assert state["species_idx"][idx] == SPECIES_TO_IDX["<pad>"]
        assert state["hp_fraction"][idx] == 0.0
        assert state["status"][idx].sum() == 0.0
        assert (state["boosts"][idx] == 0.0).all()
        assert state["mega"][idx] == 0.0


def test_encode_state_unknown_species_maps_to_unk() -> None:
    record = _turn_record(
        our_active=[_mon("not_a_real_species_xyz"), None],
        opp_active=[None, None],
        action={"slot0": {"kind": "pass"}, "slot1": {"kind": "pass"}},
    )
    state = encode_state(record)
    assert state["species_idx"][0] == SPECIES_TO_IDX["<unk>"]
    # Still real HP info even though the species itself is unrecognized.
    assert state["hp_fraction"][0] == 1.0


def test_encode_state_hand_built_values() -> None:
    record = _turn_record(
        our_active=[
            _mon("garchomp", hp_fraction=0.5, status="brn", boosts={"atk": 3}, mega=True),
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
    # status one-hot: ["none","brn","par","psn","tox","slp","frz"] -- brn is index 1.
    assert state["status"][0].tolist() == [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # boosts: [atk,def,spa,spd,spe] / 6.0 -- atk=+3 -> 0.5.
    assert state["boosts"][0].tolist() == pytest.approx([0.5, 0.0, 0.0, 0.0, 0.0])
    assert state["mega"][0] == 1.0
    # weather one-hot: ["none","sun","rain","sand","snow"] -- sun is index 1.
    assert state["weather"].tolist() == [0.0, 1.0, 0.0, 0.0, 0.0]
    # terrain one-hot: ["none","electric","grassy","psychic","misty"] -- electric is index 1.
    assert state["terrain"].tolist() == [0.0, 1.0, 0.0, 0.0, 0.0]
    assert state["trick_room"][0] == 1.0
    # side_conditions: [our_tailwind, our_reflect, our_lightscreen, our_auroraveil,
    #                    opp_tailwind, opp_reflect, opp_lightscreen, opp_auroraveil]
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


def test_encode_state_bench_summary() -> None:
    record = _turn_record(
        our_active=[_mon("garchomp"), None],
        opp_active=[None, None],
        action={"slot0": {"kind": "pass"}, "slot1": {"kind": "pass"}},
        our_bench=[
            {"species": "klefki", "hp_fraction": 0.5},
            {"species": "sylveon", "hp_fraction": 1.0},
        ],
    )
    state = encode_state(record)
    assert state["bench_count"][0] == 2.0
    assert state["bench_mean_hp"][0] == pytest.approx(0.75)
    assert state["bench_count"][1] == 0.0
    assert state["bench_mean_hp"][1] == 0.0


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


# --- BcTurnDataset: split determinism + no replay overlap ----------------------------


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
        for turn in range(1, turns_per_replay + 1):
            for player in ("p1", "p2"):
                records.append(
                    _turn_record(
                        replay_id=replay_id,
                        player=player,
                        rating=rating,
                        turn=turn,
                        our_active=[_mon("garchomp"), _mon("klefki")],
                        opp_active=[_mon("charizard"), _mon("incineroar")],
                        action={
                            "slot0": {"kind": "move", "move_id": "earthquake"},
                            "slot1": {"kind": "move", "move_id": "protect"},
                        },
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
    # Sanity: the two splits together cover every replay, with no double-counting.
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
    for i, record in enumerate(high):
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
            "slot1": {"kind": "move", "move_id": "protect"},
        },
    )
    path = _write_jsonl(tmp_path, [record])

    dataset = BcTurnDataset(path, min_rating=1000, split="train", val_fraction=0.0)
    assert len(dataset) == 1  # only slot1 encodes successfully
    assert dataset.skipped == 1


# --- model/training smoke test --------------------------------------------------------


def test_bc_policy_net_loss_decreases_over_gradient_steps(tmp_path) -> None:
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
    criterion = torch.nn.CrossEntropyLoss()

    losses: list[float] = []
    steps = 0
    start = time.time()
    while steps < 30:
        for species_idx, scalars, actions in loader:
            optimizer.zero_grad()
            logits = model(species_idx, scalars)
            loss = criterion(logits, actions)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            steps += 1
            if steps >= 30:
                break
    elapsed = time.time() - start

    # Average of the last 5 steps should be clearly below the average of the first 5 --
    # a synthetic dataset with exactly 2 possible actions is trivially learnable.
    assert sum(losses[-5:]) / 5 < sum(losses[:5]) / 5
    assert elapsed < 5.0  # fast enough that this doesn't need @pytest.mark.slow
