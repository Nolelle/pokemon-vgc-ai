"""Candidate encoding, legal masking, GAE, and PPO update smoke tests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from poke_env.battle.move import Move  # noqa: E402
from poke_env.battle.pokemon import Pokemon  # noqa: E402

from vgc.bc.encoding import (  # noqa: E402
    MOVE_TO_IDX,
    INDEX_DIM,
    SLOT_FEATURE_DIM,
    SPECIES_TO_IDX,
    STATE_SCALAR_DIM,
    TARGET_TO_IDX,
)
from vgc.rl.encoding import (  # noqa: E402
    CandidateFeatures,
    encode_candidates,
    pad_candidate_features,
)
from vgc.rl.model import CandidatePolicyValueNet  # noqa: E402
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
        switch_species_indices=np.full(
            (count, 2), SPECIES_TO_IDX["<pad>"], dtype=np.int64
        ),
        flags=np.zeros((count, 2, 4), dtype=np.float32),
    )


def _state() -> tuple[np.ndarray, np.ndarray]:
    # The exact semantic values do not matter for the tensor/optimization smoke; valid
    # vocab indices and the real encoder widths do.
    return (
        np.zeros(INDEX_DIM, dtype=np.int64),
        np.zeros(STATE_SCALAR_DIM + SLOT_FEATURE_DIM, dtype=np.float32),
    )


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
    assert any(not torch.equal(before[name], parameter) for name, parameter in model.named_parameters())
