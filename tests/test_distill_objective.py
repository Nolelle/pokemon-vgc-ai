"""Unit tests for the retention-oriented training knobs in vgc.rl.distill."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from vgc.bc.encoding import (  # noqa: E402
    INDEX_DIM,
    SLOT_FEATURE_DIM,
    STATE_SCALAR_DIM,
)
from vgc.rl.distill import (  # noqa: E402
    DistillationConfig,
    DistillationSample,
    _action_count_sampling_probabilities,
    distill_policy,
    evaluate_agreement,
)
from vgc.rl.encoding import HISTORY_SCALAR_DIM, CandidateFeatures  # noqa: E402
from vgc.rl.model import CandidatePolicyValueNet  # noqa: E402


def _sample(count: int, *, battle: str = "b") -> DistillationSample:
    moves = np.zeros((max(count, 1), 2), dtype=np.int64)
    return DistillationSample(
        battle_id=battle,
        state_indices=np.zeros(INDEX_DIM, dtype=np.int64),
        state_scalars=np.zeros(STATE_SCALAR_DIM + SLOT_FEATURE_DIM, dtype=np.float32),
        history_scalars=np.zeros(HISTORY_SCALAR_DIM, dtype=np.float32),
        candidates=CandidateFeatures(
            move_indices=moves,
            target_indices=np.zeros((max(count, 1), 2), dtype=np.int64),
            switch_species_indices=np.zeros((max(count, 1), 2), dtype=np.int64),
            flags=np.zeros((max(count, 1), 2, 4), dtype=np.float32),
        ),
        teacher_action_index=0,
        turn=1,
        legal_action_count=count,
    )


# --- _action_count_sampling_probabilities -----------------------------------------------


def test_sampling_probabilities_equalize_known_bins_and_preserve_unknown_share():
    samples = [
        _sample(10),
        _sample(12),
        _sample(30),
        _sample(40),
        _sample(50),
        _sample(60),
        _sample(0),
        _sample(0, battle="b2"),
    ]
    probabilities = _action_count_sampling_probabilities(samples)

    assert probabilities is not None
    assert probabilities.sum() == pytest.approx(1.0)
    known_mass = 6 / 8
    small_bin = probabilities[0] + probabilities[1]
    large_bin = probabilities[2:6].sum()
    unknown_mass = probabilities[6:].sum()
    assert small_bin == pytest.approx(known_mass / 2)  # two bins share the known mass
    assert large_bin == pytest.approx(known_mass / 2)
    assert unknown_mass == pytest.approx(1 - known_mass)
    # Within a bin every sample draws equally regardless of its bin's rarity overall.
    assert probabilities[0] == pytest.approx(probabilities[1])
    assert probabilities[2] == pytest.approx(probabilities[5])


def test_sampling_rebalance_disabled_without_metadata_or_single_bin():
    assert _action_count_sampling_probabilities([_sample(0), _sample(0)]) is None
    assert _action_count_sampling_probabilities([_sample(10), _sample(12)]) is None


# --- evaluate_agreement recall fields ---------------------------------------------------


def test_evaluate_agreement_reports_recall_and_rank_statistics():
    samples = [_sample(3, battle=f"b{i}") for i in range(4)]
    model = CandidatePolicyValueNet()

    metrics = evaluate_agreement(model, samples, batch_size=4, device="cpu")

    assert "recall_at_5" in metrics and "recall_at_10" in metrics
    assert metrics["recall_at_10"] == pytest.approx(1.0)  # 3 legal actions < K
    assert "teacher_rank_median" in metrics and "teacher_rank_p95" in metrics


# --- checkpoint_metric = "recall_at_k" ---------------------------------------------------


def test_recall_selection_prefers_the_high_recall_epoch_despite_worse_loss(monkeypatch):
    import vgc.rl.distill as distill_module

    torch.manual_seed(0)
    samples = [_sample(30, battle=f"battle-{i}") for i in range(4)]
    model = CandidatePolicyValueNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    scripted = [
        {"recall_at_10": 0.5, "recall_at_5": 0.4, "loss": 1.0},
        {"recall_at_10": 0.9, "recall_at_5": 0.8, "loss": 5.0},  # best by recall...
        {"recall_at_10": 0.8, "recall_at_5": 0.7, "loss": 0.1},  # ...despite worst CE
    ]

    def fake_evaluate(m, _samples, *, batch_size, device):
        entry = dict(scripted[len(fake_evaluate.calls)])
        entry.update(
            accuracy=entry["recall_at_10"],
            teacher_probability=0.5,
            top3_accuracy=entry["recall_at_5"],
            entropy=0.1,
            samples=float(len(samples)),
        )
        fake_evaluate.calls.append(entry)
        return entry

    fake_evaluate.calls = []
    monkeypatch.setattr(distill_module, "evaluate_agreement", fake_evaluate)

    config = DistillationConfig(
        epochs=3, batch_size=4, seed=0, checkpoint_metric="recall_at_k"
    )
    metrics = distill_module.distill_policy(
        model, optimizer, samples, config, device="cpu", val_samples=samples
    )

    assert metrics["best_epoch"] == 2
    assert metrics["best_val_recall_at_10"] == pytest.approx(0.9)
    assert metrics["checkpoint_metric"] == "recall_at_k"


def test_loss_selection_stays_the_default_when_configured(monkeypatch):
    import vgc.rl.distill as distill_module

    torch.manual_seed(1)
    samples = [_sample(30, battle=f"battle-{i}") for i in range(4)]
    model = CandidatePolicyValueNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def fake_evaluate(m, _samples, *, batch_size, device):
        return {
            "accuracy": 0.5,
            "teacher_probability": 0.5,
            "loss": float(len(fake_evaluate.calls)),
            "top3_accuracy": 0.5,
            "entropy": 0.1,
            "samples": float(len(samples)),
        }

    fake_evaluate.calls = []
    monkeypatch.setattr(distill_module, "evaluate_agreement", fake_evaluate)

    config = DistillationConfig(epochs=3, batch_size=4, seed=0, checkpoint_metric="loss")
    metrics = distill_module.distill_policy(
        model, optimizer, samples, config, device="cpu", val_samples=samples
    )
    assert metrics["best_epoch"] == 1  # lowest loss wins, first epoch here
    assert "best_val_recall_at_10" not in metrics


# --- hard_example_weight ------------------------------------------------------------------


def test_hard_example_weighting_reranks_every_epoch(monkeypatch):
    import vgc.rl.distill as distill_module

    torch.manual_seed(2)
    samples = [_sample(30, battle=f"battle-{i}") for i in range(4)]
    model = CandidatePolicyValueNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    calls = {"n": 0}

    def fake_ranks(_model, _samples, *, batch_size, device):
        calls["n"] += 1
        return np.asarray([15, 1, 20, 2], dtype=np.int64)

    monkeypatch.setattr(distill_module, "_teacher_ranks", fake_ranks)

    config = DistillationConfig(
        epochs=3,
        batch_size=4,
        seed=0,
        hard_example_weight=3.0,
        hard_example_rank=10,
    )
    metrics = distill_policy(model, optimizer, samples, config, device="cpu")

    assert metrics["epochs"] == 3.0
    assert calls["n"] == 3  # one re-ranking pass per epoch


# --- soft targets ------------------------------------------------------------------------


def _scored_sample(count: int, scores: list[float], *, battle: str = "b") -> DistillationSample:
    from dataclasses import replace

    return replace(
        _sample(count, battle=battle),
        search_scores=np.asarray(scores, dtype=np.float32),
        searched_mask=np.ones(count, dtype=bool),
    )


def test_soft_target_distribution_spreads_over_near_ties_and_zeroes_padding():
    from vgc.rl.distill import soft_target_distribution

    scores = torch.as_tensor([[10.0, 0.0, float("nan")]], dtype=torch.float32)
    mask = torch.as_tensor([[True, True, False]])
    target = soft_target_distribution(scores, mask, temperature=10.0)

    assert target[0, 2].item() == 0.0  # padded column carries no mass
    expected_top = float(np.exp(1.0) / (1.0 + np.exp(1.0)))  # gap of 10 at T=10
    assert target[0, 0].item() == pytest.approx(expected_top, rel=1e-4)
    assert target.sum().item() == pytest.approx(1.0)


def test_all_missing_scores_produce_zero_rows_for_hard_fallback():
    from vgc.rl.distill import soft_target_distribution

    scores = torch.full((1, 3), float("nan"))
    mask = torch.ones(1, 3, dtype=torch.bool)
    target = soft_target_distribution(scores, mask, temperature=16.0)
    assert torch.isnan(target).all() or target.abs().sum().item() == 0.0


def test_tensor_batch_pads_scores_and_flags_availability():
    from vgc.rl.distill import _tensor_batch

    wide = _scored_sample(30, [5.0, 1.0], battle="wide")
    plain = _sample(2, battle="plain")
    batch = _tensor_batch([wide, plain], device="cpu")

    assert batch["search_scores"].shape[1] == 30
    assert torch.isfinite(batch["search_scores"][0, :2]).all()
    assert not torch.isfinite(batch["search_scores"][1]).any()


def test_soft_targets_train_without_crash_on_mixed_batches():
    torch.manual_seed(3)
    scored = [
        _scored_sample(30, [40.0, 38.5] + [float(i) for i in range(28)], battle=f"s{i}")
        for i in range(4)
    ]
    unscored = [_sample(30, battle=f"u{i}") for i in range(4)]
    model = CandidatePolicyValueNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    config = DistillationConfig(
        epochs=2,
        batch_size=8,
        seed=0,
        soft_targets=True,
        soft_target_temperature=16.0,
    )
    metrics = distill_policy(model, optimizer, scored + unscored, config, device="cpu")

    assert metrics["loss"] >= 0.0
    assert np.isfinite(metrics["loss"])
