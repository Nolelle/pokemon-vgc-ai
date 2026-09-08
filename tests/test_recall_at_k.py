"""Unit tests for the shortlist-recall metric that gates neural candidate selection."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from vgc.evaluation import clustered_interval  # noqa: E402
from vgc.rl.distill import DistillationSample, recall_at_k, turn_bucket  # noqa: E402
from vgc.rl.encoding import CandidateFeatures  # noqa: E402


class _RankedStub:
    """Stands in for the policy net: returns caller-supplied logits, ignoring inputs."""

    use_meta_features = False
    use_information_features = False
    use_tactical_features = False

    def __init__(self, logits: np.ndarray) -> None:
        self._logits = torch.as_tensor(logits, dtype=torch.float32)
        self._cursor = 0

    def eval(self) -> None:
        return None

    def __call__(self, *args, **kwargs):
        mask = args[7]
        rows = mask.shape[0]
        chunk = self._logits[self._cursor : self._cursor + rows]
        self._cursor += rows
        return chunk, torch.zeros(rows)


def _sample(candidates: int, *, team: str = "pool/a", turn: int = 1) -> DistillationSample:
    return DistillationSample(
        battle_id="b",
        state_indices=np.zeros(4, dtype=np.int64),
        state_scalars=np.zeros(4, dtype=np.float32),
        history_scalars=np.zeros(4, dtype=np.float32),
        candidates=CandidateFeatures(
            move_indices=np.zeros((candidates, 2), dtype=np.int64),
            target_indices=np.zeros((candidates, 2), dtype=np.int64),
            switch_species_indices=np.zeros((candidates, 2), dtype=np.int64),
            flags=np.zeros((candidates, 2, 4), dtype=np.float32),
        ),
        teacher_action_index=0,
        team_id=team,
        opponent_team_id="pool/b",
        turn=turn,
        legal_action_count=candidates,
    )


def _run(logits, samples, ks=(1, 3, 5, 10), monkeypatch=None, width=None):
    import vgc.rl.distill as distill

    # The real model returns one logit per PADDED candidate slot and masks the illegal
    # ones itself, so the stub's logits and the mask must share a width.
    padded_width = width if width is not None else int(np.shape(logits)[1])

    def fake_batch(chunk, device):
        width = padded_width
        mask = torch.zeros((len(chunk), width), dtype=torch.bool)
        for row, s in enumerate(chunk):
            mask[row, : s.legal_action_count] = True
        return {
            "state_indices": None,
            "state_scalars": None,
            "history_scalars": None,
            "move_indices": None,
            "target_indices": None,
            "switch_species_indices": None,
            "flags": None,
            "candidate_mask": mask,
            "meta_scalars": None,
            "information_indices": None,
            "information_scalars": None,
            "tactical_features": None,
            "teacher_actions": torch.as_tensor(
                [s.teacher_action_index for s in chunk], dtype=torch.long
            ),
        }

    monkeypatch.setattr(distill, "_tensor_batch", fake_batch)
    return recall_at_k(_RankedStub(logits), samples, ks=ks, batch_size=len(samples), device="cpu")


def test_turn_bucket_matches_the_counterfactual_collector_split():
    assert turn_bucket(1) == "early" and turn_bucket(2) == "early"
    assert turn_bucket(3) == "mid" and turn_bucket(4) == "mid"
    assert turn_bucket(5) == "late" and turn_bucket(99) == "late"


def test_teacher_ranked_first_hits_every_k(monkeypatch):
    logits = np.array([[5.0, 1.0, 0.0, -1.0]])
    result = _run(logits, [_sample(4)], ks=(1, 3), monkeypatch=monkeypatch)
    assert result["recall"]["1"] == 1.0
    assert result["recall"]["3"] == 1.0
    assert result["teacher_rank"]["mean"] == pytest.approx(1.0)


def test_teacher_ranked_last_misses_small_k_and_hits_large_k(monkeypatch):
    # Teacher action (index 0) is the worst of four by the model's scores -> rank 4.
    logits = np.array([[-5.0, 1.0, 2.0, 3.0]])
    result = _run(logits, [_sample(4)], ks=(1, 3, 5), monkeypatch=monkeypatch)
    assert result["recall"]["1"] == 0.0
    assert result["recall"]["3"] == 0.0
    assert result["recall"]["5"] == 1.0
    assert result["teacher_rank"]["mean"] == pytest.approx(4.0)


def test_recall_is_monotone_in_k(monkeypatch):
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(12, 16))
    samples = [_sample(16) for _ in range(12)]
    result = _run(logits, samples, ks=(1, 3, 5, 10), monkeypatch=monkeypatch)
    values = [result["recall"][str(k)] for k in (1, 3, 5, 10)]
    assert values == sorted(values)


def test_padded_candidates_are_not_counted_as_outranking_the_teacher(monkeypatch):
    # Row has only 2 legal actions; the padded columns carry high scores that must be
    # masked out, otherwise the teacher looks worse-ranked than it is.
    logits = np.array([[1.0, 0.0, 99.0, 99.0]])
    result = _run(logits, [_sample(2)], ks=(1,), monkeypatch=monkeypatch, width=4)
    assert result["recall"]["1"] == 1.0
    assert result["teacher_rank"]["max"] == pytest.approx(1.0)


def test_trivial_states_are_reported_so_easy_collections_cannot_hide(monkeypatch):
    samples = [_sample(2), _sample(30)]
    logits = np.zeros((2, 30))
    logits[0, 0] = 1.0
    logits[1, 0] = 1.0
    result = _run(logits, samples, ks=(10,), monkeypatch=monkeypatch)
    # The 2-action state cannot fail recall@10; the 30-action one can.
    assert result["trivial"]["10"] == 1
    assert result["samples"] == 2


def test_hits_align_with_samples_so_they_can_be_clustered_by_team(monkeypatch):
    samples = [_sample(8, team="pool/a"), _sample(8, team="pool/a"), _sample(8, team="pool/b")]
    logits = np.zeros((3, 8))
    logits[0, 0] = 5.0  # team a: hit
    logits[1, 3] = 5.0  # team a: miss at k=1
    logits[2, 0] = 5.0  # team b: hit
    result = _run(logits, samples, ks=(1,), monkeypatch=monkeypatch)
    hits = result["hits"]["1"]
    assert len(hits) == len(samples)
    clusters = [(int(hits[0]) + int(hits[1]), 2), (int(hits[2]), 1)]
    low, high = clustered_interval(clusters)
    assert 0.0 <= low <= high <= 1.0


def test_empty_and_invalid_k_are_rejected(monkeypatch):
    with pytest.raises(ValueError):
        _run(np.zeros((1, 2)), [], ks=(1,), monkeypatch=monkeypatch)
    with pytest.raises(ValueError):
        _run(np.zeros((1, 2)), [_sample(2)], ks=(0,), monkeypatch=monkeypatch)
