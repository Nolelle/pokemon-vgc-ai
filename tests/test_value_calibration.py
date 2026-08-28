from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from vgc.rl.encoding import (
    INFORMATION_INDEX_DIM,
    INFORMATION_SCALAR_DIM,
    META_SCALAR_DIM,
    InformationFeatures,
)
from vgc.rl.model import CandidatePolicyValueNet
from vgc.rl.mechanics_encoding import MechanicsFeatures
from vgc.rl.value_calibration import (
    ValueCalibrationConfig,
    calibrate_value_head,
    evaluate_value_model,
    split_samples_by_team_archetype,
)


def _sample(index: int, team: str, outcome: float):
    encoded = SimpleNamespace(
        state_indices=np.zeros(48, dtype=np.int64),
        state_scalars=np.full(94, index / 100.0, dtype=np.float32),
        history_scalars=np.zeros(16, dtype=np.float32),
        meta_scalars=np.zeros(META_SCALAR_DIM, dtype=np.float32),
        information=InformationFeatures(
            indices=np.zeros(INFORMATION_INDEX_DIM, dtype=np.int64),
            scalars=np.zeros(INFORMATION_SCALAR_DIM, dtype=np.float32),
        ),
        mechanics=MechanicsFeatures(
            tokens=np.frombuffer(b"{}", dtype=np.uint8).astype(np.int64) + 1
        ),
    )
    return SimpleNamespace(
        battle_id=f"battle-{index // 2}",
        team_id=team,
        outcome=outcome,
        encoded=encoded,
    )


def _samples():
    teams = ("sun/a", "sun/b", "rain/a", "rain/b")
    return [
        _sample(index, teams[(index // 2) % len(teams)], 1.0 if index % 4 else -1.0)
        for index in range(16)
    ]


def test_team_archetype_split_keeps_entire_teams_disjoint() -> None:
    train, validation = split_samples_by_team_archetype(
        _samples(), validation_fraction=0.5, seed=7
    )

    train_teams = {sample.team_id for sample in train}
    validation_teams = {sample.team_id for sample in validation}
    assert train_teams.isdisjoint(validation_teams)
    assert {team.split("/", 1)[0] for team in train_teams} == {"sun", "rain"}
    assert {team.split("/", 1)[0] for team in validation_teams} == {"sun", "rain"}


def test_value_calibration_changes_only_value_head_and_bounds_output() -> None:
    train, validation = split_samples_by_team_archetype(
        _samples(), validation_fraction=0.5, seed=7
    )
    torch.manual_seed(3)
    model = CandidatePolicyValueNet(
        use_meta_features=True,
        use_information_features=True,
        use_tactical_features=True,
        use_mechanics_features=True,
    )
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}

    result = calibrate_value_head(
        model,
        train,
        validation,
        ValueCalibrationConfig(epochs=2, batch_size=4, patience=2, seed=3),
        device="cpu",
    )
    metrics = evaluate_value_model(model, validation, batch_size=4, device="cpu")

    assert result["value_output_transform"] == "tanh"
    assert -1.0 <= metrics["value_min"] <= metrics["value_max"] <= 1.0
    assert any(
        not torch.equal(value, before[key])
        for key, value in model.state_dict().items()
        if key.startswith("value_head.")
    )
    assert all(
        torch.equal(value, before[key])
        for key, value in model.state_dict().items()
        if not key.startswith("value_head.")
    )


def test_identity_value_transform_remains_backward_compatible() -> None:
    model = CandidatePolicyValueNet()
    assert model.value_output_transform == "identity"
