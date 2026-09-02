#!/usr/bin/env python
"""Print the mechanics gate and fail unless every required family is exact."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vgc import mechanics_gate  # noqa: E402
from vgc.config import SHOWDOWN_REPO  # noqa: E402
from vgc.gate_evidence import (  # noqa: E402
    file_sha256,
    format_family_status,
    git_dirty,
    pytest_collection_failed,
    run_gate_pytest,
    write_gate_artifact,
)
from vgc.showdown_parity import (  # noqa: E402
    check_showdown_parity,
    format_parity_report,
    load_pinned_commit,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="skip pytest execution and do not write the gate artifact",
    )
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="skip git fetch when checking public Showdown parity",
    )
    args = parser.parse_args(argv)

    coverage = json.loads(mechanics_gate.COVERAGE_PATH.read_text())
    for family in coverage.get("families") or []:
        print(format_family_status(family))

    result = mechanics_gate.mechanics_readiness()
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

    static_ready = (
        result.catalog_current
        and result.declared_ready
        and not result.partial
        and not result.missing
        and all(ready for _scope, ready in result.scopes)
    )
    if args.static_only:
        print(f"verdict: {'PASS' if static_ready else 'BLOCKED'}")
        return 0 if static_ready else 1

    if not static_ready:
        print("verdict: BLOCKED")
        return 1

    parity = check_showdown_parity(
        SHOWDOWN_REPO,
        load_pinned_commit(),
        fetch=not args.no_fetch,
    )
    print(format_parity_report(parity))
    if not parity.ready:
        print("verdict: BLOCKED")
        return 1

    test_node_ids = mechanics_gate.mechanics_gate_test_node_ids()
    print(f"running {len(test_node_ids)} gate test node(s)")
    completed = run_gate_pytest(test_node_ids)
    output = (completed.stdout or "") + (completed.stderr or "")
    if output.strip():
        print(output.rstrip())
    if pytest_collection_failed(completed.returncode) or completed.returncode != 0:
        print("verdict: BLOCKED")
        return 1

    if git_dirty():
        print("verdict: BLOCKED (dirty working tree; commit or stash before writing gate artifact)")
        return 1

    write_gate_artifact(
        mechanics_gate.GATE_ARTIFACT_PATH,
        coverage_hash=file_sha256(mechanics_gate.COVERAGE_PATH),
        dependency_hash=file_sha256(mechanics_gate.CATALOG_PATH),
        tests_run=test_node_ids,
        showdown_head=parity.local_head,
        parity_checked_utc=datetime.now(UTC).isoformat(),
    )
    refreshed = mechanics_gate.mechanics_readiness()
    print(f"artifact current: {refreshed.artifact_current}")
    print(f"verdict: {'PASS' if refreshed.ready else 'BLOCKED'}")
    return 0 if refreshed.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
