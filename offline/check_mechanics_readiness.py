#!/usr/bin/env python
"""Print the mechanics gate and fail unless every required family is exact."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from vgc.mechanics_gate import mechanics_readiness  # noqa: E402


def main() -> int:
    result = mechanics_readiness()
    print(f"catalog current: {result.catalog_current}")
    print(f"declared ready: {result.declared_ready}")
    print(f"exact ({len(result.exact)}): {', '.join(result.exact)}")
    print(f"partial ({len(result.partial)}): {', '.join(result.partial)}")
    print(f"missing ({len(result.missing)}): {', '.join(result.missing)}")
    print(
        "scopes: "
        + ", ".join(
            f"{scope}={'READY' if ready else 'BLOCKED'}" for scope, ready in result.scopes
        )
    )
    print(f"verdict: {'PASS' if result.ready else 'BLOCKED'}")
    return 0 if result.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
