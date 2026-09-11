"""Optional Weights & Biases overlay for long training runs.

JSONL under ``--out-dir`` remains the source of truth. This module only mirrors
scalars so a browser/phone dashboard can watch loss, eval win rate, and seed
overlays, and so NaN updates or stalled iterations can email/Slack you.

Disabled unless ``--wandb`` is passed. Importing this module does not require
the wandb package.
"""

from __future__ import annotations

import argparse
import math
import statistics
from collections.abc import Mapping
from pathlib import Path
from typing import Any

DEFAULT_WANDB_PROJECT = "pokemon-vgc-ai"
STALL_RATIO = 4.0
STALL_MIN_SAMPLES = 2
SKIP_KEYS = frozenset(
    {
        "workers",
        "snapshot",
        "learner_team_counts",
        "opponent_teams",
        "learner_teams",
        "val_history",
        "collection",
        "fresh_collection",
        "dataset_metadata",
        "artifact_evidence",
        "development_team_sha256",
    }
)
NAN_KEY_HINTS = (
    "loss",
    "entropy",
    "grad_norm",
    "approx_kl",
    "clip_fraction",
    "win_rate",
    "accuracy",
    "teacher_probability",
)

try:
    import wandb
except ImportError:  # pragma: no cover - wandb is an optional train extra
    wandb = None


def add_wandb_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="mirror training scalars to Weights & Biases (JSONL is still written)",
    )
    parser.add_argument(
        "--wandb-project",
        default=DEFAULT_WANDB_PROJECT,
        help="W&B project name (default: %(default)s)",
    )
    parser.add_argument(
        "--wandb-entity",
        default=None,
        help="W&B entity (user or team); default is the logged-in account",
    )
    parser.add_argument(
        "--wandb-run-name",
        default=None,
        help="W&B run name; default is the --out-dir folder name",
    )
    parser.add_argument(
        "--wandb-group",
        default=None,
        help="W&B group name; pass the same group on seed replicates to overlay charts",
    )


def config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for key, value in vars(args).items():
        if key.startswith("_"):
            continue
        converted = _config_value(value)
        if converted is not _MISSING:
            config[key] = converted
    return config


_MISSING = object()


def _config_value(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        converted = [_config_value(item) for item in value]
        if any(item is _MISSING for item in converted):
            return _MISSING
        return converted
    if isinstance(value, list) and len(value) <= 32:
        converted = [_config_value(item) for item in value]
        if any(item is _MISSING for item in converted):
            return _MISSING
        return converted
    return _MISSING


def flatten_metrics(payload: Mapping[str, object], *, prefix: str = "") -> dict[str, float]:
    """Turn a nested metrics row into W&B scalars. Non-numeric leaves are dropped."""

    flat: dict[str, float] = {}
    for key, value in payload.items():
        if key in SKIP_KEYS:
            continue
        name = f"{prefix}{key}" if prefix else str(key)
        if isinstance(value, bool):
            flat[name] = float(value)
        elif isinstance(value, int):
            flat[name] = float(value)
        elif isinstance(value, float):
            flat[name] = value
        elif isinstance(value, Mapping):
            flat.update(flatten_metrics(value, prefix=f"{name}/"))
        elif isinstance(value, (list, tuple)) and 0 < len(value) <= 8:
            if all(_is_number(item) for item in value):
                if len(value) == 2 and name.endswith("interval"):
                    flat[f"{name}/low"] = float(value[0])
                    flat[f"{name}/high"] = float(value[1])
                else:
                    for index, item in enumerate(value):
                        flat[f"{name}/{index}"] = float(item)
    return flat


def nonfinite_metric_names(metrics: Mapping[str, float]) -> list[str]:
    names: list[str] = []
    for key, value in metrics.items():
        if not _looks_like_health_metric(key):
            continue
        if not math.isfinite(value):
            names.append(key)
    return names


def stall_seconds_per_game(
    current: float,
    history: list[float],
    *,
    ratio: float = STALL_RATIO,
    min_samples: int = STALL_MIN_SAMPLES,
) -> bool:
    if current <= 0.0 or len(history) < min_samples:
        return False
    baseline = statistics.median(history[-5:])
    if baseline <= 0.0:
        return False
    return current >= ratio * baseline


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _looks_like_health_metric(name: str) -> bool:
    lowered = name.lower()
    return any(hint in lowered for hint in NAN_KEY_HINTS)


class WandbRun:
    """Thin wrapper: no-ops when W&B is off so trainers can call it unconditionally."""

    def __init__(self, run: Any | None, *, step_metric: str) -> None:
        self._run = run
        self.step_metric = step_metric
        self._seconds_per_game: list[float] = []

    @property
    def enabled(self) -> bool:
        return self._run is not None

    @property
    def url(self) -> str | None:
        if self._run is None:
            return None
        return getattr(self._run, "url", None)

    def log_train(
        self,
        row: Mapping[str, object],
        *,
        games_this_iter: int | None = None,
        prefix: str = "train",
    ) -> None:
        metrics = flatten_metrics(row, prefix=f"{prefix}/")
        self._attach_step(metrics, row)
        elapsed = row.get("elapsed_seconds")
        if _is_number(elapsed) and games_this_iter and games_this_iter > 0:
            seconds_per_game = float(elapsed) / games_this_iter
            metrics[f"{prefix}/seconds_per_game"] = seconds_per_game
            if stall_seconds_per_game(seconds_per_game, self._seconds_per_game):
                self.alert(
                    "Stalled iteration",
                    f"{seconds_per_game:.1f}s/game vs median "
                    f"{statistics.median(self._seconds_per_game[-5:]):.1f}s/game "
                    f"over the last {min(5, len(self._seconds_per_game))} iterations",
                    level="WARN",
                )
            self._seconds_per_game.append(seconds_per_game)
        worker_errors = row.get("worker_errors")
        if worker_errors is None:
            batch = row.get("batch")
            if isinstance(batch, Mapping):
                worker_errors = batch.get("worker_errors")
        if _is_number(worker_errors) and int(worker_errors) > 0:
            self.alert(
                "Worker errors",
                f"{int(worker_errors)} worker(s) failed this iteration",
                level="WARN",
            )
        self._log(metrics)

    def log_eval(self, row: Mapping[str, object], *, prefix: str = "eval") -> None:
        metrics = flatten_metrics(row, prefix=f"{prefix}/")
        self._attach_step(metrics, row)
        self._log(metrics)

    def log_epoch(self, epoch: int, metrics: Mapping[str, object]) -> None:
        payload = flatten_metrics(metrics)
        payload["epoch"] = float(epoch)
        self._log(payload)

    def alert(self, title: str, text: str, *, level: str = "ERROR") -> None:
        if self._run is None:
            return
        alert_level = _alert_level(level)
        self._run.alert(title=title[:64], text=text, level=alert_level)

    def finish(self) -> None:
        if self._run is None:
            return
        self._run.finish()
        self._run = None

    def _attach_step(self, metrics: dict[str, float], row: Mapping[str, object]) -> None:
        for alias in (self.step_metric, "games_seen", "training_games", "epoch"):
            value = row.get(alias)
            if _is_number(value):
                metrics[self.step_metric] = float(value)
                return

    def _log(self, metrics: Mapping[str, float]) -> None:
        if self._run is None or not metrics:
            return
        bad = nonfinite_metric_names(metrics)
        if bad:
            details = ", ".join(f"{name}={metrics[name]}" for name in bad)
            self.alert("NaN/Inf metric", details, level="ERROR")
        self._run.log(dict(metrics))


def start_wandb_run(
    args: argparse.Namespace,
    *,
    job_type: str,
    step_metric: str,
) -> WandbRun:
    if not getattr(args, "wandb", False):
        return WandbRun(None, step_metric=step_metric)
    if wandb is None:
        raise SystemExit(
            "--wandb requires the wandb package. Install it with "
            "`uv sync --extra train`, then run `wandb login` once."
        )
    out_dir = getattr(args, "out_dir", None)
    if isinstance(out_dir, Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        wandb_dir = str(out_dir)
    else:
        wandb_dir = None
    try:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=args.wandb_run_name or (out_dir.name if isinstance(out_dir, Path) else None),
            group=args.wandb_group or None,
            job_type=job_type,
            config=config_from_args(args),
            dir=wandb_dir,
            tags=[job_type],
        )
    except Exception as exc:
        raise SystemExit(
            f"wandb.init failed: {exc}\n"
            "Run `wandb login` once (https://wandb.ai/authorize), then retry with --wandb."
        ) from exc
    if run is None:
        return WandbRun(None, step_metric=step_metric)
    run.define_metric(step_metric)
    run.define_metric("*", step_metric=step_metric)
    url = getattr(run, "url", None)
    if url:
        print(f"wandb: {url}", flush=True)
    return WandbRun(run, step_metric=step_metric)


def _alert_level(level: str) -> Any:
    if wandb is None:
        return level
    mapping = {
        "INFO": wandb.AlertLevel.INFO,
        "WARN": wandb.AlertLevel.WARN,
        "ERROR": wandb.AlertLevel.ERROR,
    }
    return mapping.get(level.upper(), wandb.AlertLevel.ERROR)
