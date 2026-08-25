"""Unit tests for schema-v2.x teacher score storage on distillation samples."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from vgc.evaluator import ScoredOrder  # noqa: E402
from vgc.rl.distill import DistillationSample, build_score_metadata  # noqa: E402
from vgc.rl.demonstrations import (  # noqa: E402
    load_demonstrations,
    save_demonstrations,
)
from vgc.rl.encoding import CandidateFeatures, InformationFeatures  # noqa: E402


def _entry(score: float, *, searched: bool | None) -> ScoredOrder:
    breakdown = {} if searched is None else {"searched": searched}
    return ScoredOrder(order=object(), score=score, breakdown=breakdown)


def test_scores_and_mask_reflect_the_exchange_search_flags():
    scored = [_entry(10.0, searched=True), _entry(7.0, searched=False), _entry(3.0, searched=False)]
    scores, mask = build_score_metadata(scored)
    assert scores.tolist() == [10.0, 7.0, 3.0]
    assert mask.tolist() == [True, False, False]


def test_myopic_teacher_marks_everything_searched():
    scored = [_entry(5.0, searched=None), _entry(1.0, searched=None)]
    _scores, mask = build_score_metadata(scored)
    assert mask.all()


def test_empty_entries_return_none():
    assert build_score_metadata([]) is None


def test_sample_with_scores_round_trips_through_a_demonstrations_file(tmp_path):
    sample = DistillationSample(
        battle_id="b1",
        state_indices=np.zeros(4, dtype=np.int64),
        state_scalars=np.zeros(4, dtype=np.float32),
        history_scalars=np.zeros(4, dtype=np.float32),
        candidates=CandidateFeatures(
            move_indices=np.zeros((2, 2), dtype=np.int64),
            target_indices=np.zeros((2, 2), dtype=np.int64),
            switch_species_indices=np.zeros((2, 2), dtype=np.int64),
            flags=np.zeros((2, 2, 4), dtype=np.float32),
            tactical=np.zeros((2, 42), dtype=np.float32),
        ),
        teacher_action_index=0,
        information=InformationFeatures(
            indices=np.zeros(84, dtype=np.int64), scalars=np.zeros(384, dtype=np.float32)
        ),
        turn=2,
        legal_action_count=2,
        candidate_myopic_ranks=np.arange(2, dtype=np.int64),
        candidate_tags=np.zeros((2, 5), dtype=np.int8),
        search_scores=np.asarray([9.0, -1.0], dtype=np.float32),
        searched_mask=np.asarray([True, False]),
    )
    path = tmp_path / "demo.pt"
    save_demonstrations(path, [sample])
    loaded = load_demonstrations(path)

    assert len(loaded) == 1
    restored = loaded[0]
    assert restored.search_scores.tolist() == [9.0, -1.0]
    assert restored.searched_mask.tolist() == [True, False]
    assert restored.candidate_myopic_ranks.tolist() == [0, 1]
