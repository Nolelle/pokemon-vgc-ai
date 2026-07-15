"""Append structured experiment notes for future write-ups. Direct port of
pokemon-tcg-ai's offline/log_experiment.py append-only JSONL pattern.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from vgc.config import RUNS_DIR  # noqa: E402

DEFAULT_LOG = RUNS_DIR / "experiments.jsonl"


def log_experiment(name: str, summary: str, metrics_path: Path | None, output: Path) -> None:
    entry: dict[str, object] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "name": name,
        "summary": summary,
    }
    if metrics_path is not None:
        entry["metrics_path"] = str(metrics_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a") as file:
        file.write(json.dumps(entry, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--metrics-path", type=Path)
    parser.add_argument("--out", type=Path, default=DEFAULT_LOG)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    log_experiment(args.name, args.summary, args.metrics_path, args.out)
    print(args.out)
