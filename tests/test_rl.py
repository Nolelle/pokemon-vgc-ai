"""Candidate encoding, legal masking, GAE, and PPO update smoke tests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from poke_env.battle.move import Move  # noqa: E402
from poke_env.battle.pokemon import Pokemon  # noqa: E402

from selfplay.train_ppo import (  # noqa: E402
    allocate_games,
    load_training_checkpoint,
    save_checkpoint,
)
from vgc.bc.encoding import (  # noqa: E402
    MOVE_TO_IDX,
    INDEX_DIM,
    SLOT_FEATURE_DIM,
    SPECIES_TO_IDX,
    STATE_SCALAR_DIM,
    TARGET_TO_IDX,
)
from vgc.battle_memory import BattleMemory, TurnMemory  # noqa: E402
from vgc.rl.encoding import (  # noqa: E402
    CandidateFeatures,
    HISTORY_SCALAR_DIM,
    encode_battle_history,
    encode_candidates,
    pad_candidate_features,
)
from vgc.rl.model import CandidatePolicyValueNet  # noqa: E402
from vgc.rl.opponents import (  # noqa: E402
    choose_opponent,
    discover_snapshots,
    load_snapshot,
    save_snapshot,
)
from vgc.rl.ppo import (  # noqa: E402
    PpoConfig,
    RolloutBuffer,
    RolloutStep,
    ppo_update,
    select_action,
)


def _single(target, *, move_target=0, mega=False):
    return SimpleNamespace(
        order=target,
        move_target=move_target,
        mega=mega,
        z_move=False,
        dynamax=False,
        terastallize=False,
    )


def _joint(first, second):
    return SimpleNamespace(first_order=first, second_order=second)


def _candidate_features(count: int) -> CandidateFeatures:
    return CandidateFeatures(
        move_indices=np.full((count, 2), MOVE_TO_IDX["protect"], dtype=np.int64),
        target_indices=np.full((count, 2), TARGET_TO_IDX["self_or_field"], dtype=np.int64),
        switch_species_indices=np.full((count, 2), SPECIES_TO_IDX["<pad>"], dtype=np.int64),
        flags=np.zeros((count, 2, 4), dtype=np.float32),
    )


def _state() -> tuple[np.ndarray, np.ndarray]:
    # The exact semantic values do not matter for the tensor/optimization smoke; valid
    # vocab indices and the real encoder widths do.
    return (
        np.zeros(INDEX_DIM, dtype=np.int64),
        np.zeros(STATE_SCALAR_DIM + SLOT_FEATURE_DIM, dtype=np.float32),
    )


def _history() -> np.ndarray:
    return np.zeros(HISTORY_SCALAR_DIM, dtype=np.float32)


def test_encode_candidates_preserves_joint_move_targets_switch_and_mega() -> None:
    attack = Move("dragonclaw", gen=9)
    switch = Pokemon(gen=9, species="incineroar")
    protect = Move("protect", gen=9)
    orders = [
        _joint(_single(attack, move_target=2, mega=True), _single(switch)),
        _joint(_single(protect), _single(None)),
    ]

    encoded = encode_candidates(orders)

    assert len(encoded) == 2
    assert encoded.move_indices[0].tolist() == [
        MOVE_TO_IDX["dragonclaw"],
        MOVE_TO_IDX["<switch>"],
    ]
    assert encoded.target_indices[0, 0] == TARGET_TO_IDX["opp1"]
    assert encoded.switch_species_indices[0, 1] == SPECIES_TO_IDX["incineroar"]
    assert encoded.flags[0, 0, 0] == 1.0
    assert encoded.move_indices[1, 1] == MOVE_TO_IDX["<pass>"]


def test_pad_candidate_features_marks_only_real_candidates_legal() -> None:
    one = _candidate_features(1)
    three = _candidate_features(3)
    _moves, _targets, _species, _flags, mask = pad_candidate_features([one, three])
    assert mask.tolist() == [[True, False, False], [True, True, True]]


def test_masked_policy_never_samples_padded_candidate() -> None:
    model = CandidatePolicyValueNet()
    state_indices, state_scalars = _state()
    candidates = _candidate_features(2)
    moves, targets, species, flags, mask = pad_candidate_features([candidates])
    # Add a third padded candidate and ensure it remains impossible even over many draws.
    moves = np.pad(moves, ((0, 0), (0, 1), (0, 0)))
    targets = np.pad(targets, ((0, 0), (0, 1), (0, 0)))
    species = np.pad(species, ((0, 0), (0, 1), (0, 0)))
    flags = np.pad(flags, ((0, 0), (0, 1), (0, 0), (0, 0)))
    mask = np.pad(mask, ((0, 0), (0, 1)), constant_values=False)

    sampled = set()
    for _ in range(50):
        action, _log_prob, _value = select_action(
            model,
            torch.as_tensor(state_indices[None, :]),
            torch.as_tensor(state_scalars[None, :]),
            torch.as_tensor(_history()[None, :]),
            torch.as_tensor(moves),
            torch.as_tensor(targets),
            torch.as_tensor(species),
            torch.as_tensor(flags),
            torch.as_tensor(mask),
        )
        sampled.add(int(action.item()))
    assert sampled <= {0, 1}


def test_model_rejects_state_with_no_legal_candidate() -> None:
    model = CandidatePolicyValueNet()
    state_indices, state_scalars = _state()
    candidates = _candidate_features(1)
    moves, targets, species, flags, _mask = pad_candidate_features([candidates])
    with pytest.raises(ValueError, match="at least one legal"):
        model(
            torch.as_tensor(state_indices[None, :]),
            torch.as_tensor(state_scalars[None, :]),
            torch.as_tensor(_history()[None, :]),
            torch.as_tensor(moves),
            torch.as_tensor(targets),
            torch.as_tensor(species),
            torch.as_tensor(flags),
            torch.zeros((1, 1), dtype=torch.bool),
        )


def test_finish_episode_propagates_terminal_win_backward() -> None:
    buffer = RolloutBuffer()
    indices, scalars = _state()
    for _ in range(3):
        buffer.add(
            RolloutStep(
                state_indices=indices,
                state_scalars=scalars,
                history_scalars=_history(),
                candidates=_candidate_features(2),
                action_index=0,
                old_log_prob=-0.69,
                old_value=0.0,
            )
        )
    buffer.finish_episode(1.0, PpoConfig(gamma=1.0, gae_lambda=1.0))
    assert [step.reward for step in buffer.steps] == [0.0, 0.0, 1.0]
    assert [step.return_value for step in buffer.steps] == pytest.approx([1.0, 1.0, 1.0])


def test_ppo_update_changes_parameters_and_returns_finite_metrics() -> None:
    torch.manual_seed(0)
    model = CandidatePolicyValueNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    buffer = RolloutBuffer()
    indices, scalars = _state()
    candidates = _candidate_features(3)
    moves, targets, species, flags, mask = pad_candidate_features([candidates])
    with torch.no_grad():
        action, log_prob, value = select_action(
            model,
            torch.as_tensor(indices[None, :]),
            torch.as_tensor(scalars[None, :]),
            torch.as_tensor(_history()[None, :]),
            torch.as_tensor(moves),
            torch.as_tensor(targets),
            torch.as_tensor(species),
            torch.as_tensor(flags),
            torch.as_tensor(mask),
        )
    for outcome in (1.0, -1.0, 1.0, 1.0):
        buffer.add(
            RolloutStep(
                state_indices=indices.copy(),
                state_scalars=scalars.copy(),
                history_scalars=_history(),
                candidates=candidates,
                action_index=int(action.item()),
                old_log_prob=float(log_prob.item()),
                old_value=float(value.item()),
            )
        )
        buffer.finish_episode(outcome, PpoConfig())

    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    metrics = ppo_update(
        model,
        optimizer,
        buffer,
        PpoConfig(epochs=2, minibatch_size=2),
    )

    assert all(np.isfinite(value) for value in metrics.values())
    assert metrics["steps"] == 4.0
    assert any(
        not torch.equal(before[name], parameter) for name, parameter in model.named_parameters()
    )


def test_encode_battle_history_captures_longitudinal_signals() -> None:
    memory = BattleMemory(battle_tag="battle-test")
    memory.opponent_moves["amoonguss"]["protect"] = 1
    memory.opponent_moves["amoonguss"]["spore"] = 2
    memory.opponent_protects["amoonguss"] = 1
    memory.opponent_switches["incineroar"] = 2
    memory.opponent_targets["charizard"] = 3
    memory.opponent_targets["farigiraf"] = 1
    memory.our_orders = [(1, "protect / heatwave"), (2, "protect / heatwave")]
    memory.turns = [
        TurnMemory(
            turn=1,
            our_active=("charizard", "farigiraf"),
            opponent_active=("amoonguss", "incineroar"),
            our_hp={"charizard": 100.0, "farigiraf": 100.0},
            opponent_hp={"amoonguss": 100.0, "incineroar": 100.0},
            weather=("sun",),
        ),
        TurnMemory(
            turn=2,
            our_active=("charizard", "venusaur"),
            opponent_active=("amoonguss", "torkoal"),
            our_hp={"charizard": 50.0, "venusaur": 100.0},
            opponent_hp={"amoonguss": 25.0, "torkoal": 100.0},
            weather=("rain",),
        ),
    ]

    encoded = encode_battle_history(memory)

    assert encoded.shape == (HISTORY_SCALAR_DIM,)
    assert encoded[3] == pytest.approx(1 / 3)  # observed Protect rate
    assert encoded[5] == pytest.approx(0.75)  # target concentration
    assert encoded[7] == 1.0  # repeated our last order
    assert encoded[9] == pytest.approx(2 / 3)  # repeated opponent move concentration
    assert encoded[10] > 0.0  # opponent switch diversity
    assert encoded[11] == pytest.approx(0.5)  # two distinct observed targets
    assert encoded[12] > 0.0  # weather changed
    assert encoded[13] > 0.0 and encoded[14] > 0.0  # both sides changed an active


def test_rollout_buffer_discards_only_unfinished_tail_and_merges_finished_workers() -> None:
    indices, scalars = _state()

    def step() -> RolloutStep:
        return RolloutStep(
            state_indices=indices.copy(),
            state_scalars=scalars.copy(),
            history_scalars=_history(),
            candidates=_candidate_features(2),
            action_index=0,
            old_log_prob=-0.69,
            old_value=0.0,
        )

    worker = RolloutBuffer()
    worker.add(step())
    worker.finish_episode(1.0, PpoConfig())
    worker.add(step())
    worker.add(step())
    assert worker.discard_unfinished_episode() == 2
    assert len(worker) == 1

    combined = RolloutBuffer()
    combined.extend_finished(worker)
    assert len(combined) == 1


def test_allocate_games_is_balanced_exact_and_never_creates_empty_workers() -> None:
    assert allocate_games(10, 4) == [3, 3, 2, 2]
    assert allocate_games(2, 8) == [1, 1]
    assert sum(allocate_games(17, 3)) == 17
    with pytest.raises(ValueError):
        allocate_games(0, 1)


def test_snapshot_pool_is_bounded_and_round_trips_model(tmp_path) -> None:
    torch.manual_seed(7)
    model = CandidatePolicyValueNet()
    pool = tmp_path / "pool"

    first = save_snapshot(pool, model, generation=0, max_snapshots=2)
    save_snapshot(pool, model, generation=1, max_snapshots=2)
    overwritten = save_snapshot(pool, model, generation=2, max_snapshots=2)

    assert first == overwritten
    assert len(discover_snapshots(pool)) == 2
    loaded = load_snapshot(overwritten)
    for key, value in model.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key])


def test_opponent_pool_sampling_keeps_heuristic_anchor_and_snapshot_option(tmp_path) -> None:
    import random

    snapshot = tmp_path / "snapshot_slot_00.pt"
    snapshots = [snapshot]
    heuristic = choose_opponent(snapshots, heuristic_fraction=1.0, rng=random.Random(0))
    historical = choose_opponent(snapshots, heuristic_fraction=0.0, rng=random.Random(0))

    assert heuristic.kind == "heuristic"
    assert historical.kind == "snapshot"
    assert historical.checkpoint_path == snapshot


def test_training_checkpoint_restores_model_optimizer_and_progress(tmp_path) -> None:
    model = CandidatePolicyValueNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=7e-4)
    path = tmp_path / "latest.pt"
    save_checkpoint(
        path,
        model,
        optimizer,
        iteration=3,
        games_seen=48,
        ppo_config=PpoConfig(),
    )
    expected = {key: value.clone() for key, value in model.state_dict().items()}
    with torch.no_grad():
        next(model.parameters()).add_(10.0)

    iteration, games_seen, restored_config = load_training_checkpoint(
        path, model, optimizer, device="cpu"
    )

    assert (iteration, games_seen) == (3, 48)
    assert restored_config == PpoConfig()
    for key, value in expected.items():
        assert torch.equal(value, model.state_dict()[key])
