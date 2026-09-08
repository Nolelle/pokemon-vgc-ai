"""Fail-closed Problem C gate for legal action generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from vgc.config import DATA_DIR, REPO_ROOT
from vgc.gate_evidence import (
    artifact_matches,
    collect_test_node_ids,
    file_sha256,
    load_gate_artifact,
    validate_family_tests,
)

COVERAGE_PATH = DATA_DIR / "action_coverage.json"
GATE_ARTIFACT_PATH = REPO_ROOT / "runs" / "gates" / "action_gate.json"
VALID_STATUSES = frozenset({"exact", "partial", "missing"})


@dataclass(frozen=True)
class ActionReadiness:
    declared_ready: bool
    exact: tuple[str, ...]
    partial: tuple[str, ...]
    missing: tuple[str, ...]
    scopes: tuple[tuple[str, bool], ...]
    artifact_current: bool = False
    tests_run: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        return (
            self.declared_ready
            and not self.partial
            and not self.missing
            and bool(self.scopes)
            and all(ready for _scope, ready in self.scopes)
            and self.artifact_current
        )


class ActionNotReadyError(RuntimeError):
    pass


def _artifact_current(
    coverage_path: Path,
    artifact_path: Path,
) -> tuple[bool, tuple[str, ...]]:
    artifact = load_gate_artifact(artifact_path)
    if artifact is None:
        return False, ()
    current = artifact_matches(
        artifact,
        coverage_hash=file_sha256(coverage_path),
        dependency_hash=file_sha256(coverage_path),
    )
    tests_run = tuple(artifact.get("tests_run") or ())
    return current, tests_run


def action_readiness(
    coverage_path: Path | None = None,
    artifact_path: Path | None = None,
) -> ActionReadiness:
    coverage_path = coverage_path or COVERAGE_PATH
    artifact_path = artifact_path or GATE_ARTIFACT_PATH
    coverage = json.loads(coverage_path.read_text())
    families = coverage.get("families") or []
    validate_family_tests(families, gate_name="action")
    grouped: dict[str, list[str]] = {status: [] for status in VALID_STATUSES}
    seen: set[str] = set()
    for family in families:
        family_id = family.get("id")
        status = family.get("status")
        if not isinstance(family_id, str) or not family_id or family_id in seen:
            raise ValueError(f"invalid or duplicate action family: {family_id!r}")
        if status not in VALID_STATUSES:
            raise ValueError(f"invalid action status for {family_id}: {status!r}")
        seen.add(family_id)
        grouped[status].append(family_id)
    if not seen:
        raise ValueError("action coverage contains no families")
    artifact_current, tests_run = _artifact_current(coverage_path, artifact_path)
    return ActionReadiness(
        declared_ready=coverage.get("release_ready") is True,
        exact=tuple(sorted(grouped["exact"])),
        partial=tuple(sorted(grouped["partial"])),
        missing=tuple(sorted(grouped["missing"])),
        scopes=tuple(
            sorted((name, value is True) for name, value in (coverage.get("scopes") or {}).items())
        ),
        artifact_current=artifact_current,
        tests_run=tests_run,
    )


def action_gate_test_node_ids(coverage_path: Path | None = None) -> list[str]:
    coverage = json.loads((coverage_path or COVERAGE_PATH).read_text())
    families = coverage.get("families") or []
    validate_family_tests(families, gate_name="action")
    return collect_test_node_ids(families)


def assert_action_ready(
    action: str,
    artifact_path: Path | None = None,
) -> None:
    result = action_readiness(artifact_path=artifact_path)
    if result.ready:
        return
    reasons: list[str] = []
    if not result.declared_ready:
        reasons.append("release_ready is false")
    if result.partial:
        reasons.append(f"{len(result.partial)} families are partial")
    if result.missing:
        reasons.append(f"{len(result.missing)} families are missing")
    blocked = [name for name, ready in result.scopes if not ready]
    if blocked:
        reasons.append(f"blocked scopes: {', '.join(blocked)}")
    if not result.artifact_current:
        reasons.append("the action gate artifact is missing or stale")
    raise ActionNotReadyError(
        f"Blocked {action}: " + "; ".join(reasons) + ". "
        "Run `.venv/bin/python offline/check_action_readiness.py`."
    )


def enforce_action_gate_for_cli(action: str) -> None:
    try:
        assert_action_ready(action)
    except ActionNotReadyError as exc:
        raise SystemExit(str(exc)) from None
