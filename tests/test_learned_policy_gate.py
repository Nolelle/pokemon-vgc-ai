from __future__ import annotations

from pathlib import Path

from offline.evaluate_learned_policy import build_gate_report


def _result(*, rate: float, low: float, fallback_count: int = 0) -> dict[str, object]:
    return {
        "games": 1000,
        "win_rate": rate,
        "clustered_interval": [low, min(1.0, rate + 0.03)],
        "min_opponent_win_rate": 0.55,
        "fallback_count": fallback_count,
        "opponent_results": {
            "vgc": {
                "games": 167,
                "wins": 95,
                "win_rate": 95 / 167,
                "wilson_interval": [0.51, 0.64],
            }
        },
    }


def test_gate_requires_rl_provenance_strength_generalization_and_zero_fallbacks(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"exact model bytes")
    payload = {
        "use_information_features": True,
        "use_tactical_features": True,
        "evaluation": {"win_rate": 0.56},
        "iteration": 5,
        "games_seen": 20_000,
    }
    report = build_gate_report(
        checkpoint=checkpoint,
        checkpoint_payload=payload,
        train_result=_result(rate=0.58, low=0.54),
        holdout_result=_result(rate=0.56, low=0.52),
        incumbent="vgc",
        minimum_games=1000,
        minimum_rl_games=10_000,
        overall_threshold=0.50,
        incumbent_threshold=0.50,
        minimum_opponent_rate=0.45,
        maximum_generalization_gap=0.05,
    )

    assert report["passed"] is True
    assert len(report["checkpoint_sha256"]) == 64
    assert all(report["checks"].values())

    failed = build_gate_report(
        checkpoint=checkpoint,
        checkpoint_payload={**payload, "games_seen": 0},
        train_result=_result(rate=0.70, low=0.65),
        holdout_result=_result(rate=0.56, low=0.49, fallback_count=1),
        incumbent="vgc",
        minimum_games=1000,
        minimum_rl_games=10_000,
        overall_threshold=0.50,
        incumbent_threshold=0.50,
        minimum_opponent_rate=0.45,
        maximum_generalization_gap=0.05,
    )

    assert failed["passed"] is False
    assert failed["checks"]["minimum_rl_experience"] is False
    assert failed["checks"]["heldout_clustered_lower_bound"] is False
    assert failed["checks"]["generalization_gap"] is False
    assert failed["checks"]["zero_fallbacks"] is False
