"""Optional Weights & Biases integration for training scripts.

Enabled when ``--wandb`` is passed or ``WANDB_API_KEY`` is set. Disabled when
``--no-wandb`` is passed or ``WANDB_DISABLED`` is truthy. If initialization fails,
training continues without remote logging.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from vgc.config import FORMAT_ID

DEFAULT_PROJECT = "pokemon-vgc-ai"


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def should_enable_wandb(explicit: bool | None = None) -> bool:
    """Return whether W&B logging should be attempted for this run."""

    if _truthy(os.environ.get("WANDB_DISABLED")):
        return False
    if explicit is True:
        return True
    if explicit is False:
        return False
    return bool(os.environ.get("WANDB_API_KEY", "").strip())


def add_wandb_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "enable Weights & Biases logging "
            "(default: on when WANDB_API_KEY is set, off otherwise)"
        ),
    )
    parser.add_argument(
        "--wandb-project",
        default=None,
        help="W&B project name (default: WANDB_PROJECT env or pokemon-vgc-ai)",
    )
    parser.add_argument(
        "--wandb-run-name",
        default=None,
        help="W&B run name (default: WANDB_RUN_NAME env or script-derived name)",
    )


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def flatten_metrics(
    metrics: dict[str, Any],
    *,
    prefix: str = "",
    sep: str = "/",
) -> dict[str, float | int | str | bool]:
    """Flatten nested metric dicts for W&B scalar logging."""

    flat: dict[str, float | int | str | bool] = {}

    def _walk(key: str, value: Any) -> None:
        if isinstance(value, dict):
            if not value:
                return
            for child_key, child_value in value.items():
                child_name = f"{key}{sep}{child_key}" if key else str(child_key)
                _walk(child_name, child_value)
            return
        if isinstance(value, (list, tuple, set)):
            return
        if isinstance(value, (bool, int, float, str)):
            flat[key] = value

    for key, value in metrics.items():
        name = f"{prefix}{sep}{key}" if prefix else str(key)
        _walk(name, value)
    return flat


def config_from_namespace(args: argparse.Namespace) -> dict[str, Any]:
    payload = {key: _json_safe(value) for key, value in vars(args).items()}
    payload["format_id"] = FORMAT_ID
    return payload


# Late import guard: Path is used in _json_safe
from pathlib import Path  # noqa: E402


class WandbSession:
    """Thin wrapper that no-ops when W&B is disabled or unavailable."""

    def __init__(
        self,
        *,
        enabled: bool,
        job_type: str,
        config: dict[str, Any] | None = None,
        project: str | None = None,
        run_name: str | None = None,
        tags: list[str] | None = None,
    ) -> None:
        self.enabled = enabled
        self._run = None
        if not enabled:
            return

        try:
            import wandb
        except ImportError:
            print(
                "WARNING: wandb is not installed; continuing without remote logging. "
                "Install with `uv sync --extra train`."
            )
            self.enabled = False
            return

        init_kwargs: dict[str, Any] = {
            "project": project or os.environ.get("WANDB_PROJECT", DEFAULT_PROJECT),
            "job_type": job_type,
            "config": _json_safe(config or {}),
        }
        entity = os.environ.get("WANDB_ENTITY", "").strip()
        if entity:
            init_kwargs["entity"] = entity
        resolved_name = run_name or os.environ.get("WANDB_RUN_NAME", "").strip()
        if resolved_name:
            init_kwargs["name"] = resolved_name
        if tags:
            init_kwargs["tags"] = tags

        try:
            self._run = wandb.init(**init_kwargs)
        except Exception as exc:  # pragma: no cover - network/login failures
            print(f"WARNING: wandb.init failed ({exc}); continuing without remote logging.")
            self.enabled = False
            self._run = None

    @classmethod
    def from_cli(
        cls,
        args: argparse.Namespace,
        *,
        job_type: str,
        config: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> WandbSession:
        explicit = getattr(args, "wandb", None)
        project = getattr(args, "wandb_project", None)
        run_name = getattr(args, "wandb_run_name", None)
        return cls(
            enabled=should_enable_wandb(explicit),
            job_type=job_type,
            config=config,
            project=project,
            run_name=run_name,
            tags=tags,
        )

    @property
    def active(self) -> bool:
        return self.enabled and self._run is not None

    def log(self, metrics: dict[str, Any], *, step: int | None = None) -> None:
        if not self.active:
            return
        payload = flatten_metrics(metrics)
        if not payload:
            return
        try:
            import wandb

            wandb.log(payload, step=step)
        except Exception as exc:  # pragma: no cover - best-effort logging
            print(f"WARNING: wandb.log failed ({exc}); continuing training.")

    def log_summary(self, metrics: dict[str, Any]) -> None:
        if not self.active:
            return
        payload = flatten_metrics(metrics)
        if not payload:
            return
        try:
            import wandb

            for key, value in payload.items():
                wandb.run.summary[key] = value
        except Exception as exc:  # pragma: no cover - best-effort logging
            print(f"WARNING: wandb summary update failed ({exc}); continuing training.")

    def finish(self, exit_code: int = 0) -> None:
        if not self.active:
            return
        try:
            import wandb

            wandb.finish(exit_code=exit_code)
        except Exception as exc:  # pragma: no cover - best-effort cleanup
            print(f"WARNING: wandb.finish failed ({exc}).")
        finally:
            self._run = None

    def __enter__(self) -> WandbSession:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.finish(exit_code=1 if exc is not None else 0)


def log_json_artifact(session: WandbSession, path: Path, name: str | None = None) -> None:
    if not session.active or not path.is_file():
        return
    try:
        import wandb

        wandb.save(str(path), base_path=str(path.parent), policy="now")
        if name is not None:
            artifact = wandb.Artifact(name=name, type="metrics")
            artifact.add_file(str(path))
            wandb.log_artifact(artifact)
    except Exception as exc:  # pragma: no cover - optional artifact upload
        print(f"WARNING: wandb artifact upload failed ({exc}).")


def dumps_config(config: dict[str, Any]) -> str:
    return json.dumps(_json_safe(config), indent=2, sort_keys=True)
