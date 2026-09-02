#!/usr/bin/env python
"""Print whether the local Showdown checkout still matches public master."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vgc.config import SHOWDOWN_REPO  # noqa: E402
from vgc.showdown_parity import (  # noqa: E402
    check_showdown_parity,
    format_parity_report,
    load_pinned_commit,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="skip git fetch and compare against the cached upstream ref",
    )
    args = parser.parse_args(argv)

    report = check_showdown_parity(
        SHOWDOWN_REPO,
        load_pinned_commit(),
        fetch=not args.no_fetch,
    )
    print(format_parity_report(report))
    print(f"verdict: {'PASS' if report.ready else 'BLOCKED'}")
    return 0 if report.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
