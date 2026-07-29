"""Unit tests for the honest teacher-bootstrap validation path (`--bootstrap-val-games`
in `selfplay/train_ppo.py`).

Background: gating the bootstrap on `split_samples_by_battle(samples, ...)` -- a split
of the TRAINING collection itself -- overlaps the training state distribution and
massively overstates agreement (measured: reported val_after.accuracy 0.955 vs. only
0.217 accuracy / 0.220 teacher_probability on a separately, freshly collected set).
`--bootstrap-val-games` fixes this by gating on a separately collected fresh set
instead; these tests are pure-Python/dict-level and never touch a live server.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import selfplay.train_ppo as train_ppo  # noqa: E402
from selfplay.train_ppo import (  # noqa: E402
    BOOTSTRAP_VAL_SEED_OFFSET,
    bootstrap_validation_seed,
    build_bootstrap_artifact,
)


def test_bootstrap_validation_seed_is_deterministic() -> None:
    assert bootstrap_validation_seed(42) == bootstrap_validation_seed(42)


def test_bootstrap_validation_seed_differs_from_training_seed() -> None:
    for seed in (0, 1, 42, 777, 123_456):
        assert bootstrap_validation_seed(seed) != seed


def test_bootstrap_validation_seed_uses_named_offset_constant() -> None:
    assert bootstrap_validation_seed(5) == 5 + BOOTSTRAP_VAL_SEED_OFFSET


def _metrics(accuracy: float, teacher_probability: float = 0.5) -> dict[str, float]:
    return {
        "accuracy": accuracy,
        "top3_accuracy": accuracy,
        "teacher_probability": teacher_probability,
        "loss": 1.0,
        "entropy": 1.0,
        "samples": 10.0,
    }


def test_build_bootstrap_artifact_rejects_unknown_validation_source() -> None:
    with pytest.raises(ValueError):
        build_bootstrap_artifact(
            validation_source="bogus",
            collection={"games": 10},
            fresh_collection=None,
            train_battles=5,
            val_battles=2,
            train_samples=50,
            val_samples=20,
            before=_metrics(0.3),
            training={"loss": 0.1},
            train_after=_metrics(0.9),
            val_after=_metrics(0.3),
            min_val_accuracy=0.0,
            min_improvement=0.0,
            min_teacher_probability=0.0,
            elapsed_seconds=1.0,
        )


def test_build_bootstrap_artifact_requires_fresh_collection_iff_fresh_source() -> None:
    # fresh_collection provided but source says training_split -- inconsistent.
    with pytest.raises(ValueError):
        build_bootstrap_artifact(
            validation_source="training_split",
            collection={"games": 10},
            fresh_collection={"games": 5},
            train_battles=5,
            val_battles=2,
            train_samples=50,
            val_samples=20,
            before=_metrics(0.3),
            training={"loss": 0.1},
            train_after=_metrics(0.9),
            val_after=_metrics(0.3),
            min_val_accuracy=0.0,
            min_improvement=0.0,
            min_teacher_probability=0.0,
            elapsed_seconds=1.0,
        )
    # fresh source but no fresh_collection -- also inconsistent.
    with pytest.raises(ValueError):
        build_bootstrap_artifact(
            validation_source="fresh_collection",
            collection={"games": 10},
            fresh_collection=None,
            train_battles=5,
            val_battles=2,
            train_samples=50,
            val_samples=20,
            before=_metrics(0.3),
            training={"loss": 0.1},
            train_after=_metrics(0.9),
            val_after=_metrics(0.3),
            min_val_accuracy=0.0,
            min_improvement=0.0,
            min_teacher_probability=0.0,
            elapsed_seconds=1.0,
        )


def test_build_bootstrap_artifact_training_split_path_has_no_fresh_fields() -> None:
    result = build_bootstrap_artifact(
        validation_source="training_split",
        collection={"games": 10, "samples": 70},
        fresh_collection=None,
        train_battles=8,
        val_battles=2,
        train_samples=56,
        val_samples=14,
        before=_metrics(0.3),
        training={"loss": 0.1},
        train_after=_metrics(0.9),
        val_after=_metrics(0.955, teacher_probability=0.9),
        min_val_accuracy=0.0,
        min_improvement=0.0,
        min_teacher_probability=0.0,
        elapsed_seconds=1.0,
    )
    assert result["validation_source"] == "training_split"
    assert "fresh_collection" not in result
    assert "train_vs_fresh_val_accuracy_gap" not in result
    # val_after is the authoritative (if leaky) metric on this path.
    assert result["val_after"]["accuracy"] == pytest.approx(0.955)
    assert result["passed"] is True


def test_build_bootstrap_artifact_fresh_path_reports_train_vs_fresh_gap() -> None:
    result = build_bootstrap_artifact(
        validation_source="fresh_collection",
        collection={"games": 200, "samples": 1400},
        fresh_collection={"games": 60, "samples": 420},
        train_battles=200,
        val_battles=60,
        train_samples=1400,
        val_samples=420,
        before=_metrics(0.2, teacher_probability=0.22),
        training={"loss": 0.1},
        train_after=_metrics(0.955, teacher_probability=0.9),
        val_after=_metrics(0.217, teacher_probability=0.220),
        min_val_accuracy=0.0,
        min_improvement=0.0,
        min_teacher_probability=0.0,
        elapsed_seconds=1.0,
    )
    assert result["validation_source"] == "fresh_collection"
    assert result["fresh_collection"] == {"games": 60, "samples": 420}
    # Headline diagnostic: the memorization gap between train_after and the
    # authoritative (fresh) val_after.
    assert result["train_vs_fresh_val_accuracy_gap"] == pytest.approx(0.955 - 0.217)
    assert result["val_after"]["accuracy"] == pytest.approx(0.217)


def test_build_bootstrap_artifact_gate_uses_authoritative_val_after() -> None:
    # min_val_accuracy set between the leaky (0.955) and honest (0.217) numbers --
    # only passes if val_after (already the authoritative set by contract) is used.
    result = build_bootstrap_artifact(
        validation_source="fresh_collection",
        collection={"games": 200},
        fresh_collection={"games": 60},
        train_battles=200,
        val_battles=60,
        train_samples=1400,
        val_samples=420,
        before=_metrics(0.2, teacher_probability=0.22),
        training={"loss": 0.1},
        train_after=_metrics(0.955, teacher_probability=0.9),
        val_after=_metrics(0.217, teacher_probability=0.220),
        min_val_accuracy=0.5,
        min_improvement=0.0,
        min_teacher_probability=0.0,
        elapsed_seconds=1.0,
    )
    assert result["passed"] is False


def test_main_rejects_negative_bootstrap_val_games(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_ppo.py",
            "--iterations",
            "1",
            "--eval-games",
            "0",
            "--bootstrap-val-games",
            "-1",
        ],
    )
    with pytest.raises(SystemExit):
        train_ppo.main()
