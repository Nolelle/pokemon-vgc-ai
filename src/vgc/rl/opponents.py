"""Historical policy-snapshot pool for PPO self-play opponents."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

try:
    import torch
except ImportError as exc:  # pragma: no cover - train extra is optional
    raise ImportError(
        "vgc.rl.opponents requires the 'train' extra (torch) -- run `uv sync --extra train`."
    ) from exc

from vgc.rl.model import CandidatePolicyValueNet

RL_ARCHITECTURE_VERSION = "candidate-policy-value-v3-meta"


@dataclass(frozen=True)
class OpponentSpec:
    kind: str  # "heuristic" or "snapshot"
    label: str
    checkpoint_path: Path | None = None


def discover_snapshots(pool_dir: Path) -> list[Path]:
    return sorted(pool_dir.glob("snapshot_slot_*.pt")) if pool_dir.exists() else []


def choose_opponent(
    snapshots: list[Path],
    *,
    heuristic_fraction: float,
    rng: random.Random,
) -> OpponentSpec:
    """Sample the fixed heuristic anchor or a uniformly random historical policy."""

    if not 0.0 <= heuristic_fraction <= 1.0:
        raise ValueError("heuristic_fraction must be between 0 and 1")
    if not snapshots or rng.random() < heuristic_fraction:
        return OpponentSpec(kind="heuristic", label="heuristic")
    path = rng.choice(snapshots)
    return OpponentSpec(kind="snapshot", label=path.stem, checkpoint_path=path)


def save_snapshot(
    pool_dir: Path,
    model: CandidatePolicyValueNet,
    *,
    generation: int,
    max_snapshots: int,
) -> Path:
    """Write a bounded ring slot so long runs do not grow checkpoint storage forever."""

    if max_snapshots <= 0:
        raise ValueError("max_snapshots must be positive")
    pool_dir.mkdir(parents=True, exist_ok=True)
    slot = generation % max_snapshots
    path = pool_dir / f"snapshot_slot_{slot:02d}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "architecture": RL_ARCHITECTURE_VERSION,
            "generation": generation,
            "use_meta_features": model.use_meta_features,
            "use_information_features": model.use_information_features,
            "use_tactical_features": model.use_tactical_features,
            "head_dropout": model.head_dropout_p,
            "value_output_transform": model.value_output_transform,
            "head_width": getattr(model, "head_width", None),
        },
        path,
    )
    return path


def load_snapshot(path: Path, *, device: str = "cpu") -> CandidatePolicyValueNet:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    architecture = checkpoint.get("architecture")
    if architecture != RL_ARCHITECTURE_VERSION:
        raise ValueError(
            f"unsupported RL snapshot architecture {architecture!r} in {path}; "
            f"expected {RL_ARCHITECTURE_VERSION!r}"
        )
    use_meta_features = bool(checkpoint.get("use_meta_features", False))
    use_information_features = bool(checkpoint.get("use_information_features", False))
    use_tactical_features = bool(checkpoint.get("use_tactical_features", False))
    model = CandidatePolicyValueNet(
        use_meta_features=use_meta_features,
        use_information_features=use_information_features,
        use_tactical_features=use_tactical_features,
        head_dropout=float(checkpoint.get("head_dropout", 0.0)),
        value_output_transform=str(checkpoint.get("value_output_transform", "identity")),
        head_width=checkpoint.get("head_width"),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model
