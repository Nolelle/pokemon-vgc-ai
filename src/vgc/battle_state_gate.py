"""Fail-closed Part B gate for player-view state and hidden-information beliefs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from vgc.battle_state_replay import DECISION_REPLAY_SCHEMA
from vgc.config import DATA_DIR, REPO_ROOT
from vgc.gate_evidence import (
    artifact_matches,
    collect_test_node_ids,
    file_sha256,
    load_gate_artifact,
    validate_family_tests,
)
from vgc.models import PolicyConfig

COVERAGE_PATH = DATA_DIR / "battle_state_coverage.json"
SET_PRIORS_PATH = REPO_ROOT / "data" / "usage" / "set_priors.json"
GATE_ARTIFACT_PATH = REPO_ROOT / "runs" / "gates" / "battle_state_gate.json"
VALID_STATUSES = frozenset({"exact", "partial", "missing"})


@dataclass(frozen=True)
class BattleStateReadiness:
    declared_ready: bool
    priors_current: bool
    schema_current: bool
    branch_defaults_ready: bool
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
            and self.priors_current
            and self.schema_current
            and self.branch_defaults_ready
            and not self.partial
            and not self.missing
            and bool(self.scopes)
            and all(ready for _scope, ready in self.scopes)
            and self.artifact_current
        )


class BattleStateNotReadyError(RuntimeError):
    pass


def _artifact_current(
    coverage_path: Path,
    set_priors_path: Path,
    artifact_path: Path,
) -> tuple[bool, tuple[str, ...]]:
    artifact = load_gate_artifact(artifact_path)
    if artifact is None:
        return False, ()
    current = artifact_matches(
        artifact,
        coverage_hash=file_sha256(coverage_path),
        dependency_hash=file_sha256(set_priors_path),
    )
    tests_run = tuple(artifact.get("tests_run") or ())
    return current, tests_run


def battle_state_readiness(
    coverage_path: Path | None = None,
    set_priors_path: Path | None = None,
    artifact_path: Path | None = None,
) -> BattleStateReadiness:
    coverage_path = coverage_path or COVERAGE_PATH
    set_priors_path = set_priors_path or SET_PRIORS_PATH
    artifact_path = artifact_path or GATE_ARTIFACT_PATH
    coverage = json.loads(coverage_path.read_text())
    priors_bytes = set_priors_path.read_bytes()
    families = coverage.get("families") or []
    validate_family_tests(families, gate_name="battle-state")
    grouped: dict[str, list[str]] = {status: [] for status in VALID_STATUSES}
    seen: set[str] = set()
    for family in families:
        family_id = family.get("id")
        status = family.get("status")
        if not isinstance(family_id, str) or not family_id or family_id in seen:
            raise ValueError(f"invalid or duplicate battle-state family: {family_id!r}")
        if status not in VALID_STATUSES:
            raise ValueError(f"invalid battle-state status for {family_id}: {status!r}")
        seen.add(family_id)
        grouped[status].append(family_id)
    if not seen:
        raise ValueError("battle-state coverage contains no families")
    config = PolicyConfig()
    artifact_current, tests_run = _artifact_current(coverage_path, set_priors_path, artifact_path)
    return BattleStateReadiness(
        declared_ready=coverage.get("release_ready") is True,
        priors_current=(
            coverage.get("set_priors_sha256")
            == hashlib.sha256(priors_bytes).hexdigest()
        ),
        schema_current=coverage.get("decision_replay_schema") == DECISION_REPLAY_SCHEMA,
        branch_defaults_ready=(
            config.exact_search_spread_hypotheses >= 2
            and config.exact_search_set_hypotheses >= 2
            and config.exact_search_bring_hypotheses >= 2
            and config.exact_search_total_hypotheses >= 2
        ),
        exact=tuple(sorted(grouped["exact"])),
        partial=tuple(sorted(grouped["partial"])),
        missing=tuple(sorted(grouped["missing"])),
        scopes=tuple(
            sorted((name, value is True) for name, value in (coverage.get("scopes") or {}).items())
        ),
        artifact_current=artifact_current,
        tests_run=tests_run,
    )


def battle_state_gate_test_node_ids(coverage_path: Path | None = None) -> list[str]:
    coverage = json.loads((coverage_path or COVERAGE_PATH).read_text())
    families = coverage.get("families") or []
    validate_family_tests(families, gate_name="battle-state")
    return collect_test_node_ids(families)


def assert_battle_state_ready(
    action: str,
    artifact_path: Path | None = None,
) -> None:
    result = battle_state_readiness(artifact_path=artifact_path)
    if result.ready:
        return
    reasons: list[str] = []
    if not result.declared_ready:
        reasons.append("release_ready is false")
    if not result.priors_current:
        reasons.append("joint opponent priors changed after review")
    if not result.schema_current:
        reasons.append("decision replay schema changed after review")
    if not result.branch_defaults_ready:
        reasons.append("default search collapses material opponent uncertainty")
    if result.partial:
        reasons.append(f"{len(result.partial)} families are partial")
    if result.missing:
        reasons.append(f"{len(result.missing)} families are missing")
    blocked = [name for name, ready in result.scopes if not ready]
    if blocked:
        reasons.append(f"blocked scopes: {', '.join(blocked)}")
    if not result.artifact_current:
        reasons.append("the battle-state gate artifact is missing or stale")
    raise BattleStateNotReadyError(
        f"Blocked {action}: " + "; ".join(reasons) + ". "
        "Run `.venv/bin/python offline/check_battle_state_readiness.py`."
    )


def enforce_battle_state_gate_for_cli(action: str) -> None:
    try:
        assert_battle_state_ready(action)
    except BattleStateNotReadyError as exc:
        raise SystemExit(str(exc)) from None
