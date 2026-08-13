"""Tests for ``selfplay/train_fixed_mirror.py``."""

from __future__ import annotations

import json
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


def test_parse_eval_opponents_falls_back_to_training_opponent() -> None:
    assert train_fixed_mirror._parse_eval_opponents("", "random") == ["random"]
    assert train_fixed_mirror._parse_eval_opponents("heuristic,vgc", "random") == [
        "heuristic",
        "vgc",
    ]


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
