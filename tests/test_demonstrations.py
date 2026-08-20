from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from vgc.bc.encoding import INDEX_DIM, SLOT_FEATURE_DIM, STATE_SCALAR_DIM
from vgc.rl.demonstrations import (
    annotate_samples,
    load_demonstrations,
    save_demonstrations,
    split_samples_grouped,
)
from vgc.rl.distill import DistillationSample
from vgc.rl.encoding import (
    HISTORY_SCALAR_DIM,
    INFORMATION_INDEX_DIM,
    INFORMATION_SCALAR_DIM,
    TACTICAL_FEATURE_DIM,
    CandidateFeatures,
    InformationFeatures,
)


def _sample(battle_id: str, *, team_id: str | None = None) -> DistillationSample:
    candidates = CandidateFeatures(
        move_indices=np.zeros((2, 2), dtype=np.int64),
        target_indices=np.zeros((2, 2), dtype=np.int64),
        switch_species_indices=np.zeros((2, 2), dtype=np.int64),
        flags=np.zeros((2, 2, 4), dtype=np.float32),
        tactical=np.zeros((2, TACTICAL_FEATURE_DIM), dtype=np.float32),
    )
    return DistillationSample(
        battle_id=battle_id,
        state_indices=np.zeros(INDEX_DIM, dtype=np.int64),
        state_scalars=np.zeros(STATE_SCALAR_DIM + SLOT_FEATURE_DIM, dtype=np.float32),
        history_scalars=np.zeros(HISTORY_SCALAR_DIM, dtype=np.float32),
        candidates=candidates,
        teacher_action_index=1,
        information=InformationFeatures(
            indices=np.zeros(INFORMATION_INDEX_DIM, dtype=np.int64),
            scalars=np.zeros(INFORMATION_SCALAR_DIM, dtype=np.float32),
        ),
        team_id=team_id,
    )


def test_demonstration_round_trip_preserves_complete_joint_examples(tmp_path) -> None:
    samples = annotate_samples(
        [_sample("battle-a")], team_id="team-a", opponent_team_id="team-b"
    )
    path = tmp_path / "demonstrations.pt"
    save_demonstrations(path, samples)
    loaded = load_demonstrations(path)
    assert len(loaded) == 1
    assert loaded[0].teacher_action_index == 1
    assert loaded[0].team_id == "team-a"
    assert loaded[0].information is not None
    assert loaded[0].candidates.tactical is not None


def test_grouped_split_never_crosses_battles_or_teams() -> None:
    samples = [
        _sample(f"battle-{team}-{battle}", team_id=f"team-{team}")
        for team in range(4)
        for battle in range(2)
        for _turn in range(2)
    ]
    train, validation = split_samples_grouped(
        samples, val_fraction=0.25, seed=3, group_by="battle"
    )
    assert {sample.battle_id for sample in train}.isdisjoint(
        {sample.battle_id for sample in validation}
    )

    train, validation = split_samples_grouped(
        samples, val_fraction=0.25, seed=3, group_by="team"
    )
    assert {sample.team_id for sample in train}.isdisjoint(
        {sample.team_id for sample in validation}
    )


def test_complete_context_validation_rejects_missing_information(tmp_path) -> None:
    sample = _sample("battle-a")
    incomplete = DistillationSample(
        battle_id=sample.battle_id,
        state_indices=sample.state_indices,
        state_scalars=sample.state_scalars,
        history_scalars=sample.history_scalars,
        candidates=sample.candidates,
        teacher_action_index=sample.teacher_action_index,
    )
    with pytest.raises(ValueError, match="information"):
        save_demonstrations(tmp_path / "bad.pt", [incomplete])
