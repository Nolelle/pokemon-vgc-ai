"""Tests for ``selfplay/train_fixed_mirror.py``."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from selfplay import train_fixed_mirror
from vgc.rl.model import CandidatePolicyValueNet


def test_build_ppo_config_matches_fixed_mirror_protocol() -> None:
    config = train_fixed_mirror.build_ppo_config()
    assert config.gamma == 1.0
    assert config.teacher_anchor_weight == 0.0
    assert config.reward_shaping_coef == 0.0
    assert config.entropy_weight == 0.01


def test_parse_args_default_opponent_is_random() -> None:
    args = train_fixed_mirror.parse_args([])
    assert args.opponent == "random"
    assert args.init_from is None


def test_eval_only_accepts_zero_iterations() -> None:
    args = train_fixed_mirror.parse_args(
        ["--iterations", "0", "--eval-games", "500", "--eval-opponents", "random,vgc"]
    )
    assert args.iterations == 0
    assert args.eval_games == 500
    assert train_fixed_mirror._parse_eval_opponents(args.eval_opponents, args.opponent) == [
        "random",
        "vgc",
    ]


def test_parse_eval_opponents_falls_back_to_training_opponent() -> None:
    assert train_fixed_mirror._parse_eval_opponents("", "random") == ["random"]
    assert train_fixed_mirror._parse_eval_opponents("heuristic,vgc", "random") == [
        "heuristic",
        "vgc",
    ]


def test_parse_eval_opponents_defaults_to_mix_names() -> None:
    mix = train_fixed_mirror.parse_opponent_mix(train_fixed_mirror.DEFAULT_MIX)
    names = [name for name, _ in mix]
    assert train_fixed_mirror._parse_eval_opponents("", "random", mix_names=names) == names
    assert train_fixed_mirror._parse_eval_opponents("vgc", "random", mix_names=names) == ["vgc"]


def test_default_mix_matches_mixed_opponent_protocol() -> None:
    mix = train_fixed_mirror.parse_opponent_mix(train_fixed_mirror.DEFAULT_MIX)
    weights = {name: weight for name, weight in mix}
    assert weights == {
        "random": pytest.approx(0.1),
        "maxpower": pytest.approx(0.1),
        "heuristic": pytest.approx(0.2),
        "vgc_myopic": pytest.approx(0.2),
        "vgc_shallow": pytest.approx(0.2),
        "vgc": pytest.approx(0.2),
    }
    assert abs(sum(weights.values()) - 1.0) < 1e-12


def test_parse_opponent_mix_normalizes_and_rejects_bad_entries() -> None:
    mix = train_fixed_mirror.parse_opponent_mix("random=1,heuristic=3")
    assert mix == [("random", 0.25), ("heuristic", 0.75)]
    with pytest.raises(ValueError, match="name=weight"):
        train_fixed_mirror.parse_opponent_mix("random")
    with pytest.raises(ValueError, match="positive"):
        train_fixed_mirror.parse_opponent_mix("random=0,heuristic=0")


def test_sample_opponent_is_seed_deterministic() -> None:
    mix = train_fixed_mirror.parse_opponent_mix(train_fixed_mirror.DEFAULT_MIX)
    first_rng = random.Random(7)
    second_rng = random.Random(7)
    first = [train_fixed_mirror.sample_opponent(mix, first_rng) for _ in range(200)]
    second = [train_fixed_mirror.sample_opponent(mix, second_rng) for _ in range(200)]
    assert first == second
    assert len(set(first)) > 1


def test_parse_eval_at_games_sorts_and_dedupes() -> None:
    assert train_fixed_mirror.parse_eval_at_games("25000,10000,50000,10000") == [
        10000,
        25000,
        50000,
    ]
    assert train_fixed_mirror.parse_eval_at_games("") == []
    with pytest.raises(ValueError, match="positive"):
        train_fixed_mirror.parse_eval_at_games("0")


def test_pool_eval_metrics_mean_and_min() -> None:
    pool = train_fixed_mirror.pool_eval_metrics(
        [
            {"opponent": "random", "win_rate": 0.95},
            {"opponent": "heuristic", "win_rate": 0.35},
            {"opponent": "vgc", "win_rate": 0.90},
        ]
    )
    assert pool["mean_win_rate"] == pytest.approx((0.95 + 0.35 + 0.90) / 3)
    assert pool["min_win_rate"] == 0.35
    assert pool["min_opponent"] == "heuristic"
    assert pool["kind"] == "pool"


def test_parse_args_accepts_mix_and_eval_at_games() -> None:
    args = train_fixed_mirror.parse_args(
        [
            "--mix",
            train_fixed_mirror.DEFAULT_MIX,
            "--eval-at-games",
            "10000,25000,50000",
            "--eval-games",
            "500",
        ]
    )
    assert args.mix == train_fixed_mirror.DEFAULT_MIX
    assert args.eval_at_games == "10000,25000,50000"
    assert args.eval_games == 500


@pytest.mark.integration
def test_smoke_trains_one_iteration_and_writes_checkpoint(tmp_path: Path) -> None:
    from vgc.rl.env import DEFAULT_SHOWDOWN_REPO

    if not (DEFAULT_SHOWDOWN_REPO / "dist" / "sim" / "index.js").exists():
        pytest.skip(f"no built showdown sim at {DEFAULT_SHOWDOWN_REPO}")
    torch = pytest.importorskip("torch")
    torch = pytest.importorskip("torch")

    out_dir = tmp_path / "smoke"
    train_fixed_mirror.main(
        [
            "--iterations",
            "1",
            "--games-per-iteration",
            "2",
            "--eval-games",
            "0",
            "--seed",
            "99",
            "--out-dir",
            str(out_dir),
        ]
    )

    metrics_path = out_dir / "metrics.jsonl"
    checkpoint_path = out_dir / "latest.pt"
    assert metrics_path.exists()
    assert checkpoint_path.exists()

    row = json.loads(metrics_path.read_text().strip())
    assert row["iteration"] == 1
    assert row["games_seen"] == 2
    assert row["steps"] > 0
    assert "ppo" in row

    model = CandidatePolicyValueNet(use_meta_features=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    iteration, games_seen, restored = train_fixed_mirror.load_checkpoint(
        checkpoint_path, model, optimizer, device="cpu"
    )
    assert iteration == 1
    assert games_seen == 2
    assert restored.gamma == 1.0
    assert restored.teacher_anchor_weight == 0.0


def test_init_from_cli_is_accepted() -> None:
    args = train_fixed_mirror.parse_args(["--init-from", "/tmp/latest.pt", "--opponent", "maxpower"])
    assert args.init_from == Path("/tmp/latest.pt")
    assert args.opponent == "maxpower"
