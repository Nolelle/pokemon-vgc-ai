#!/usr/bin/env python
"""Make the local Showdown checkout and data/champions match public master, then gate.

Does nothing if already in parity. Otherwise pulls + force-builds Showdown, re-exports the
Champions data and mechanics catalogue, re-pins the catalogue hash, commits the refresh and
runs the three readiness gates. Stops for review if the catalogue changed shape. See
`vgc.showdown_sync` for the full contract.

Usage:
    .venv/bin/python tools/sync_showdown.py
    .venv/bin/python tools/sync_showdown.py --accept-catalog-changes   # after reviewing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vgc.showdown_sync import SyncBlocked, sync_showdown  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--accept-catalog-changes",
        action="store_true",
        help="re-pin even though the mechanics catalogue changed shape (review it first)",
    )
    parser.add_argument(
        "--skip-gates", action="store_true", help="refresh and commit, but do not run the gates"
    )
    args = parser.parse_args(argv)
    try:
        result = sync_showdown(
            accept_catalog_changes=args.accept_catalog_changes, run_gates=not args.skip_gates
        )
    except SyncBlocked as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 1
    print(f"verdict: {result.status} (showdown {result.showdown_commit[:9]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
