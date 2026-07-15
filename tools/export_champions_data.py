#!/usr/bin/env python
"""Thin wrapper: invokes tools/export_champions_data.mjs against the local Showdown
repo and reports row counts. This is the single supported way to (re)generate
data/champions/*.json -- run it any time the showdown repo's champions mod changes.

Usage:
    .venv/bin/python tools/export_champions_data.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vgc.config import DATA_DIR, SHOWDOWN_REPO  # noqa: E402
from vgc.node import find_node  # noqa: E402

EXPORT_SCRIPT = Path(__file__).resolve().parent / "export_champions_data.mjs"

def main() -> int:
    if not SHOWDOWN_REPO.exists():
        print(f"showdown repo not found at {SHOWDOWN_REPO} (see vgc/config.py)", file=sys.stderr)
        return 1
    dist_sim = SHOWDOWN_REPO / "dist" / "sim" / "dex.js"
    if not dist_sim.exists():
        print(
            f"{dist_sim} not found -- run `node build` in {SHOWDOWN_REPO} first",
            file=sys.stderr,
        )
        return 1

    node = find_node()
    result = subprocess.run(
        [node, str(EXPORT_SCRIPT), str(SHOWDOWN_REPO), str(DATA_DIR)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout, file=sys.stdout)
        print(result.stderr, file=sys.stderr)
        return result.returncode

    counts = json.loads(result.stdout.strip().splitlines()[-1])
    print(f"Exported champions-mod data to {DATA_DIR}:")
    for name, count in counts.items():
        print(f"  {name}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
