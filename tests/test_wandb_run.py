from __future__ import annotations

import argparse
import math
from pathlib import Path

from vgc.rl.wandb_run import (
    WandbRun,
    add_wandb_args,
    config_from_args,
    flatten_metrics,
    nonfinite_metric_names,
    stall_seconds_per_game,
    start_wandb_run,
)


def test_flatten_metrics_nests_ppo_and_intervals() -> None:
    flat = flatten_metrics(
        {
            "iteration": 3,
            "games_seen": 192,
            "ppo": {"loss": 0.5, "entropy": 1.2},
            "clustered_interval": (0.4, 0.6),
            "workers": [{"error": "nope"}],
            "snapshot": "/tmp/latest.pt",
        }
    )
    assert flat["iteration"] == 3.0
    assert flat["ppo/loss"] == 0.5
    assert flat["clustered_interval/low"] == 0.4
    assert flat["clustered_interval/high"] == 0.6
    assert "workers" not in flat
    assert "snapshot" not in flat


def test_nonfinite_metric_names_only_flags_health_keys() -> None:
    names = nonfinite_metric_names(
        {
            "train/ppo/loss": math.nan,
            "train/games_seen": math.inf,
            "eval/win_rate": math.nan,
            "train/wins": 4.0,
        }
    )
    assert names == ["train/ppo/loss", "eval/win_rate"]


def test_stall_needs_history_then_flags_a_spike() -> None:
    history = [10.0, 11.0, 9.5]
    assert not stall_seconds_per_game(12.0, history)
    assert stall_seconds_per_game(50.0, history)
    assert not stall_seconds_per_game(50.0, [10.0])


def test_disabled_run_is_a_noop() -> None:
    run = WandbRun(None, step_metric="games_seen")
    run.log_train({"games_seen": 32, "ppo": {"loss": 1.0}, "elapsed_seconds": 8.0}, games_this_iter=32)
    run.log_eval({"games_seen": 32, "win_rate": 0.5})
    run.log_epoch(1, {"val/loss": 0.2})
    run.alert("should not fire", "nope")
    run.finish()
    assert not run.enabled


def test_start_without_flag_does_not_need_wandb() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("runs/demo"))
    add_wandb_args(parser)
    args = parser.parse_args([])
    run = start_wandb_run(args, job_type="ppo", step_metric="games_seen")
    assert not run.enabled
    assert args.wandb is False


def test_start_with_flag_without_wandb_exits(monkeypatch) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("runs/demo"))
    add_wandb_args(parser)
    args = parser.parse_args(["--wandb"])
    monkeypatch.setattr("vgc.rl.wandb_run.wandb", None)
    try:
        start_wandb_run(args, job_type="ppo", step_metric="games_seen")
    except SystemExit as exc:
        assert "uv sync --extra train" in str(exc)
    else:
        raise AssertionError("expected SystemExit when wandb is missing")


def test_config_from_args_stringifies_paths() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("runs/demo"))
    parser.add_argument("--seed", type=int, default=7)
    add_wandb_args(parser)
    args = parser.parse_args(["--seed", "9"])
    config = config_from_args(args)
    assert config["seed"] == 9
    assert config["out_dir"] == "runs/demo"
    assert config["wandb"] is False
