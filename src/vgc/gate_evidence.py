"""Shared helpers for fail-closed readiness gates backed by named pytest evidence."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from vgc.config import REPO_ROOT

VALID_STATUSES = frozenset({"exact", "partial", "missing"})
GAP_NO_FAMILY_TEST = "no family-specific Showdown-executed test"
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def git_dirty() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return bool(result.stdout.strip())


def validate_family_tests(families: list[dict], *, gate_name: str) -> None:
    for family in families:
        family_id = family.get("id")
        status = family.get("status")
        tests = family.get("tests")
        if tests is None:
            raise ValueError(f"{gate_name} family {family_id!r} is missing a tests list")
        if not isinstance(tests, list) or not all(isinstance(item, str) for item in tests):
            raise ValueError(f"{gate_name} family {family_id!r} has an invalid tests list")
        if status == "exact" and not tests:
            raise ValueError(
                f"{gate_name} family {family_id!r} is exact but has no backing tests"
            )


def collect_test_node_ids(families: list[dict]) -> list[str]:
    collected: list[str] = []
    seen: set[str] = set()
    for family in families:
        for node_id in family.get("tests") or []:
            if node_id not in seen:
                seen.add(node_id)
                collected.append(node_id)
    return collected


def run_gate_pytest(test_node_ids: list[str]) -> subprocess.CompletedProcess[str]:
    if not test_node_ids:
        raise ValueError("gate pytest run requested with zero test node ids")
    command = [
        str(PYTHON),
        "-m",
        "pytest",
        *test_node_ids,
        "-m",
        "",
        "-q",
        "-p",
        "no:cacheprovider",
    ]
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def pytest_collection_failed(returncode: int) -> bool:
    """Pytest uses exit code 4 for usage/collection errors and 5 when nothing ran."""

    return returncode in (4, 5)


def load_gate_artifact(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def artifact_matches(
    artifact: dict,
    *,
    coverage_hash: str,
    dependency_hash: str,
    artifact_path: Path | None = None,
) -> bool:
    if artifact.get("coverage_sha256") != coverage_hash:
        return False
    if artifact.get("dependency_sha256") != dependency_hash:
        return False
    if artifact.get("dirty") is not False:
        return False
    try:
        return artifact.get("git_commit") == git_head()
    except subprocess.CalledProcessError:
        return False


def write_gate_artifact(
    path: Path,
    *,
    coverage_hash: str,
    dependency_hash: str,
    tests_run: list[str],
    showdown_head: str | None = None,
    parity_checked_utc: str | None = None,
) -> dict:
    payload = {
        "git_commit": git_head(),
        "dirty": git_dirty(),
        "coverage_sha256": coverage_hash,
        "dependency_sha256": dependency_hash,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "tests_run": tests_run,
        "showdown_head": showdown_head,
        "parity_checked_utc": parity_checked_utc,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def format_family_status(family: dict) -> str:
    family_id = family["id"]
    status = family["status"]
    tests = family.get("tests") or []
    if status == "partial" and family.get("gap"):
        return f"{family_id}: {status} ({family['gap']})"
    if tests:
        return f"{family_id}: {status} ({len(tests)} test(s))"
    return f"{family_id}: {status}"
