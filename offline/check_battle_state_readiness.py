#!/usr/bin/env python
"""Print the Part B battle-state gate and fail unless every scope is ready."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vgc import battle_state_gate  # noqa: E402
from vgc.gate_evidence import (  # noqa: E402
    file_sha256,
    format_family_status,
    git_dirty,
    pytest_collection_failed,
    run_gate_pytest,
    write_gate_artifact,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="skip pytest execution and do not write the gate artifact",
    )
    args = parser.parse_args(argv)

    coverage = json.loads(battle_state_gate.COVERAGE_PATH.read_text())
    for family in coverage.get("families") or []:
        print(format_family_status(family))

    result = battle_state_gate.battle_state_readiness()
    print(f"declared ready: {result.declared_ready}")
    print(f"joint priors current: {result.priors_current}")
    print(f"decision replay schema current: {result.schema_current}")
    print(f"multi-belief defaults ready: {result.branch_defaults_ready}")
    print(f"exact ({len(result.exact)}): {', '.join(result.exact)}")
    print(f"partial ({len(result.partial)}): {', '.join(result.partial)}")
    print(f"missing ({len(result.missing)}): {', '.join(result.missing)}")
    print(
        "scopes: "
        + ", ".join(f"{name}={'READY' if ready else 'BLOCKED'}" for name, ready in result.scopes)
    )

    static_ready = (
        result.declared_ready
        and result.priors_current
        and result.schema_current
        and result.branch_defaults_ready
        and not result.partial
        and not result.missing
        and bool(result.scopes)
        and all(ready for _name, ready in result.scopes)
    )
    if args.static_only:
        print(f"verdict: {'PASS' if static_ready else 'BLOCKED'}")
        return 0 if static_ready else 1

    if not static_ready:
        print("verdict: BLOCKED")
        return 1

    test_node_ids = battle_state_gate.battle_state_gate_test_node_ids()
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
        battle_state_gate.GATE_ARTIFACT_PATH,
        coverage_hash=file_sha256(battle_state_gate.COVERAGE_PATH),
        dependency_hash=file_sha256(battle_state_gate.SET_PRIORS_PATH),
        tests_run=test_node_ids,
    )
    refreshed = battle_state_gate.battle_state_readiness()
    print(f"artifact current: {refreshed.artifact_current}")
    print(f"verdict: {'PASS' if refreshed.ready else 'BLOCKED'}")
    return 0 if refreshed.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
