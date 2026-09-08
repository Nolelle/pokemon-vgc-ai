"""Fail-closed release gate for mechanics completeness.

Training comparisons and ladder results are not meaningful readiness evidence while the
shared battle forecast omits legal mechanics. The catalogue hash prevents a Showdown
data update from silently bypassing review.
"""

from __future__ import annotations

import hashlib
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

CATALOG_PATH = DATA_DIR / "mechanics_catalog.json"
COVERAGE_PATH = DATA_DIR / "mechanics_coverage.json"
GATE_ARTIFACT_PATH = REPO_ROOT / "runs" / "gates" / "mechanics_gate.json"
VALID_STATUSES = frozenset({"exact", "partial", "missing"})


@dataclass(frozen=True)
class MechanicsReadiness:
    catalog_current: bool
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
            self.catalog_current
            and self.declared_ready
            and not self.partial
            and not self.missing
            and all(ready for _scope, ready in self.scopes)
            and self.artifact_current
        )


class MechanicsNotReadyError(RuntimeError):
    """Raised when public play is attempted before the mechanics gate passes."""


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _artifact_current(
    coverage_path: Path,
    catalog_path: Path,
    artifact_path: Path,
) -> tuple[bool, tuple[str, ...]]:
    artifact = load_gate_artifact(artifact_path)
    if artifact is None:
        return False, ()
    current = artifact_matches(
        artifact,
        coverage_hash=file_sha256(coverage_path),
        dependency_hash=file_sha256(catalog_path),
    )
    tests_run = tuple(artifact.get("tests_run") or ())
    return current, tests_run


def mechanics_readiness(
    catalog_path: Path | None = None,
    coverage_path: Path | None = None,
    artifact_path: Path | None = None,
) -> MechanicsReadiness:
    # Resolved at call time, not bound at import time, so a test can point the gate at
    # a regressed coverage file and confirm it still fails closed.
    catalog_path = catalog_path or CATALOG_PATH
    coverage_path = coverage_path or COVERAGE_PATH
    artifact_path = artifact_path or GATE_ARTIFACT_PATH
    catalog_bytes = catalog_path.read_bytes()
    catalog = json.loads(catalog_bytes)
    coverage = _load(coverage_path)
    families = coverage.get("families") or []
    validate_family_tests(families, gate_name="mechanics")
    if coverage.get("catalog_schema_version") != catalog.get("schema_version"):
        catalog_current = False
    else:
        catalog_current = (
            coverage.get("catalog_sha256") == hashlib.sha256(catalog_bytes).hexdigest()
        )
    grouped: dict[str, list[str]] = {status: [] for status in VALID_STATUSES}
    seen: set[str] = set()
    for family in families:
        family_id = family.get("id")
        status = family.get("status")
        if not isinstance(family_id, str) or not family_id or family_id in seen:
            raise ValueError(f"invalid or duplicate mechanics family id: {family_id!r}")
        if status not in VALID_STATUSES:
            raise ValueError(f"invalid status for mechanics family {family_id}: {status!r}")
        seen.add(family_id)
        grouped[status].append(family_id)
    if not seen:
        raise ValueError("mechanics coverage contains no families")
    artifact_current, tests_run = _artifact_current(coverage_path, catalog_path, artifact_path)
    return MechanicsReadiness(
        catalog_current=catalog_current,
        declared_ready=coverage.get("release_ready") is True,
        exact=tuple(sorted(grouped["exact"])),
        partial=tuple(sorted(grouped["partial"])),
        missing=tuple(sorted(grouped["missing"])),
        scopes=tuple(
            sorted(
                (scope, details.get("ready") is True)
                for scope, details in coverage.get("readiness_scopes", {}).items()
            )
        ),
        artifact_current=artifact_current,
        tests_run=tests_run,
    )


def mechanics_gate_test_node_ids(coverage_path: Path | None = None) -> list[str]:
    coverage = _load(coverage_path or COVERAGE_PATH)
    families = coverage.get("families") or []
    validate_family_tests(families, gate_name="mechanics")
    return collect_test_node_ids(families)


def assert_mechanics_ready(
    action: str = "public ladder play",
    artifact_path: Path | None = None,
) -> None:
    readiness = mechanics_readiness(artifact_path=artifact_path)
    if readiness.ready:
        return
    reasons: list[str] = []
    if not readiness.catalog_current:
        reasons.append("the Showdown catalogue changed after its last review")
    if readiness.partial:
        reasons.append(f"{len(readiness.partial)} mechanics families are only partial")
    if readiness.missing:
        reasons.append(f"{len(readiness.missing)} mechanics families are missing")
    if not readiness.declared_ready:
        reasons.append("release_ready is false")
    blocked_scopes = [scope for scope, ready in readiness.scopes if not ready]
    if blocked_scopes:
        reasons.append(f"blocked scopes: {', '.join(blocked_scopes)}")
    if not readiness.artifact_current:
        reasons.append("the mechanics gate artifact is missing or stale")
    raise MechanicsNotReadyError(
        f"Blocked {action}: " + "; ".join(reasons) + ". "
        "Run `.venv/bin/python offline/check_mechanics_readiness.py` for the full list."
    )


def enforce_mechanics_gate_for_cli(action: str) -> None:
    """Clean command-line failure without exposing an internal Python traceback."""

    try:
        assert_mechanics_ready(action)
    except MechanicsNotReadyError as exc:
        raise SystemExit(str(exc)) from None
