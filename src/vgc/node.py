"""Locate a modern Node.js executable for the local Pokemon Showdown checkout."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

NODE_ENV_VAR = "VGC_NODE"
MIN_NODE_MAJOR = 22


def _parse_node_major(version: str) -> int | None:
    match = re.fullmatch(r"v?(\d+)(?:\.\d+){0,2}", version.strip())
    return int(match.group(1)) if match else None


def _node_major(path: Path) -> int | None:
    try:
        result = subprocess.run(
            [str(path), "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return _parse_node_major(result.stdout)


def _candidate_paths() -> list[Path]:
    candidates: list[Path] = []
    if configured := os.environ.get(NODE_ENV_VAR):
        candidates.append(Path(configured).expanduser())

    # Prefer an installed Node 22 over whatever older executable happens to be first on
    # PATH. This machine currently has Node 16 at /usr/local/bin/node as well as Node 22
    # under nvm, and modern Showdown's compiled simulator cannot run on the former.
    nvm_root = Path.home() / ".nvm" / "versions" / "node"
    candidates.extend(sorted(nvm_root.glob("v22.*/bin/node"), reverse=True))

    if path_node := shutil.which("node"):
        candidates.append(Path(path_node))

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if resolved not in seen:
            seen.add(resolved)
            unique.append(candidate)
    return unique


def find_node(minimum_major: int = MIN_NODE_MAJOR) -> str:
    """Return a usable Node executable, rejecting versions older than ``minimum_major``.

    Set ``VGC_NODE`` to override discovery. The override is still version-checked so a
    stale local setting fails clearly instead of producing an unrelated JavaScript
    syntax error later.
    """

    rejected: list[str] = []
    for candidate in _candidate_paths():
        major = _node_major(candidate)
        if major is not None and major >= minimum_major:
            return str(candidate)
        rejected.append(f"{candidate} (Node {major if major is not None else 'unreadable'})")

    details = ", ".join(rejected) or "no candidates found"
    raise FileNotFoundError(
        f"Node {minimum_major}+ is required for Pokemon Showdown; checked: {details}. "
        f"Set {NODE_ENV_VAR} to a compatible executable."
    )


def node_environment(node: str) -> dict[str, str]:
    """Return an environment where child ``node`` commands use the same executable.

    Pokemon Showdown's launcher may invoke ``node build`` internally, so calling the
    launcher with an absolute Node 22 path is not sufficient when PATH still resolves
    ``node`` to an older installation.
    """

    environment = os.environ.copy()
    node_dir = str(Path(node).resolve().parent)
    existing_path = environment.get("PATH")
    environment["PATH"] = (
        node_dir if not existing_path else node_dir + os.pathsep + existing_path
    )
    return environment
