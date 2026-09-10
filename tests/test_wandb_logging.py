from __future__ import annotations

import os
from argparse import Namespace

import pytest

from vgc.wandb_logging import (
    WandbSession,
    flatten_metrics,
    should_enable_wandb,
)


@pytest.mark.parametrize(
    ("explicit", "api_key", "disabled", "expected"),
    [
        (True, "", "", True),
        (False, "abc", "", False),
        (None, "abc", "", True),
        (None, "", "", False),
        (None, "abc", "true", False),
    ],
)
def test_should_enable_wandb(
    explicit: bool | None,
    api_key: str,
    disabled: str,
    expected: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_DISABLED", raising=False)
    if api_key:
        monkeypatch.setenv("WANDB_API_KEY", api_key)
    if disabled:
        monkeypatch.setenv("WANDB_DISABLED", disabled)
    assert should_enable_wandb(explicit) is expected


def test_flatten_metrics_skips_nested_lists() -> None:
    flat = flatten_metrics(
        {
            "ppo": {"loss": 0.5, "entropy": 1.2},
            "counts": {"wins": 3},
            "ignored": [1, 2, 3],
        }
    )
    assert flat == {
        "ppo/loss": 0.5,
        "ppo/entropy": 1.2,
        "counts/wins": 3,
    }


def test_wandb_session_disabled_is_noop() -> None:
    session = WandbSession(enabled=False, job_type="unit_test", config={"seed": 1})
    assert session.active is False
    session.log({"train/loss": 1.0}, step=1)
    session.log_summary({"best": 0.9})
    session.finish()


def test_wandb_session_from_cli_without_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_DISABLED", raising=False)
    args = Namespace(wandb=None, wandb_project=None, wandb_run_name=None)
    session = WandbSession.from_cli(args, job_type="unit_test", config={"seed": 1})
    assert session.enabled is False
    session.finish()


def test_wandb_session_explicit_flag_uses_offline_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_DISABLED", raising=False)
    monkeypatch.setenv("WANDB_MODE", "offline")
    args = Namespace(wandb=True, wandb_project="test-project", wandb_run_name="run-a")
    session = WandbSession.from_cli(args, job_type="unit_test", config={"seed": 1})
    assert session.enabled is True
    session.log({"train/loss": 0.25}, step=1)
    session.finish()
