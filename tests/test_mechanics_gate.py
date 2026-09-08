from __future__ import annotations

import json
from pathlib import Path

import pytest

import offline.check_mechanics_readiness as mechanics_readiness_cli
from vgc.gate_evidence import file_sha256, write_gate_artifact
from vgc.mechanics_gate import (
    CATALOG_PATH,
    COVERAGE_PATH,
    MechanicsNotReadyError,
    assert_mechanics_ready,
    enforce_mechanics_gate_for_cli,
    mechanics_readiness,
)
from vgc.showdown_parity import ParityReport

VALID_ENGINE_STATES = frozenset(
    {
        "official_showdown",
        "official_showdown_clone",
        "official_showdown_clone_with_belief_branches",
        "lossless_public_json_tokens",
        "captured",
    }
)
FAKE_GIT_HEAD = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"


def _patch_clean_git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("vgc.gate_evidence.git_head", lambda: FAKE_GIT_HEAD)
    monkeypatch.setattr("vgc.gate_evidence.git_dirty", lambda: False)


def test_checked_in_catalog_is_current_and_families_are_classified() -> None:
    result = mechanics_readiness()

    assert result.catalog_current
    assert result.exact
    assert not result.partial
    assert not result.missing
    assert result.declared_ready
    # Static readiness is complete; `ready` additionally needs a fresh clean-HEAD
    # artifact from a full gate run, which the unit suite does not produce.
    assert result.ready == result.artifact_current


def test_every_exact_family_names_a_real_engine_and_its_evidence() -> None:
    """"Exact" has to point at something, or the gate is just a boolean someone set."""

    coverage = json.loads(COVERAGE_PATH.read_text())
    for family in coverage["families"]:
        if family["status"] != "exact":
            continue
        assert family["state"] in VALID_ENGINE_STATES, family
        assert family.get("verification") not in (None, "", "none"), family
        assert family.get("tests")


def test_exact_family_without_tests_raises_value_error(tmp_path: Path) -> None:
    coverage = json.loads(COVERAGE_PATH.read_text())
    coverage["families"][0]["tests"] = []
    regressed = tmp_path / "coverage.json"
    regressed.write_text(json.dumps(coverage))

    with pytest.raises(ValueError, match="no backing tests"):
        mechanics_readiness(CATALOG_PATH, regressed)


def test_a_single_regressed_family_blocks_the_whole_release(tmp_path: Path) -> None:
    """The gate must stay fail-closed now that it is green."""

    coverage = json.loads(COVERAGE_PATH.read_text())
    coverage["families"][0]["status"] = "partial"
    regressed = tmp_path / "coverage.json"
    regressed.write_text(json.dumps(coverage))

    assert not mechanics_readiness(CATALOG_PATH, regressed).ready


def test_a_blocked_scope_alone_blocks_the_release(tmp_path: Path) -> None:
    coverage = json.loads(COVERAGE_PATH.read_text())
    coverage["readiness_scopes"]["live_decisions"]["ready"] = False
    regressed = tmp_path / "coverage.json"
    regressed.write_text(json.dumps(coverage))

    assert not mechanics_readiness(CATALOG_PATH, regressed).ready


def test_cli_gate_fails_without_internal_traceback_contract(tmp_path, monkeypatch) -> None:
    coverage = json.loads(COVERAGE_PATH.read_text())
    coverage["release_ready"] = False
    regressed = tmp_path / "coverage.json"
    regressed.write_text(json.dumps(coverage))
    monkeypatch.setattr("vgc.mechanics_gate.COVERAGE_PATH", regressed)

    with pytest.raises(SystemExit, match="Blocked model training"):
        enforce_mechanics_gate_for_cli("model training")


def test_catalog_hash_change_invalidates_review(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"schema_version": 1, "changed": True}))
    coverage = tmp_path / "coverage.json"
    coverage.write_text(
        json.dumps(
            {
                "catalog_schema_version": 1,
                "catalog_sha256": "stale",
                "release_ready": True,
                "families": [
                    {
                        "id": "example",
                        "status": "exact",
                        "state": "captured",
                        "tests": ["tests/test_mechanics_gate.py::test_catalog_hash_change_invalidates_review"],
                    }
                ],
            }
        )
    )

    result = mechanics_readiness(catalog, coverage)

    assert not result.catalog_current
    assert not result.ready


def test_unknown_test_node_id_blocks_gate_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    coverage = json.loads(COVERAGE_PATH.read_text())
    for family in coverage["families"]:
        if family["status"] == "partial":
            family["status"] = "exact"
            family.pop("gap", None)
            if not family.get("tests"):
                family["tests"] = [
                    "tests/test_mechanics_gate.py::test_catalog_hash_change_invalidates_review"
                ]
    coverage["families"][0]["tests"] = ["tests/does_not_exist.py::test_missing"]
    regressed = tmp_path / "coverage.json"
    regressed.write_text(json.dumps(coverage))
    monkeypatch.setattr("vgc.mechanics_gate.COVERAGE_PATH", regressed)
    monkeypatch.setattr(
        mechanics_readiness_cli,
        "check_showdown_parity",
        lambda *_args, **_kwargs: ParityReport(
            local_head="abc",
            local_dirty=False,
            pinned_commit="abc",
            pinned_matches_head=True,
            upstream_ref="origin/master",
            fetched=True,
            fetch_error=None,
            missing_upstream_commits=(),
        ),
    )

    assert mechanics_readiness_cli.main([]) == 1


def test_assert_mechanics_ready_requires_current_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_path = tmp_path / "mechanics_gate.json"
    monkeypatch.setattr("vgc.mechanics_gate.GATE_ARTIFACT_PATH", artifact_path)

    with pytest.raises(MechanicsNotReadyError, match="artifact is missing or stale"):
        assert_mechanics_ready("test action", artifact_path=artifact_path)


def test_assert_mechanics_ready_passes_with_current_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_clean_git(monkeypatch)
    artifact_path = tmp_path / "mechanics_gate.json"
    monkeypatch.setattr("vgc.mechanics_gate.GATE_ARTIFACT_PATH", artifact_path)
    coverage = json.loads(COVERAGE_PATH.read_text())
    for family in coverage["families"]:
        family["status"] = "exact"
        family.pop("gap", None)
        if not family.get("tests"):
            family["tests"] = ["tests/test_mechanics_gate.py::test_catalog_hash_change_invalidates_review"]
    regressed = tmp_path / "coverage.json"
    regressed.write_text(json.dumps(coverage))
    monkeypatch.setattr("vgc.mechanics_gate.COVERAGE_PATH", regressed)

    write_gate_artifact(
        artifact_path,
        coverage_hash=file_sha256(regressed),
        dependency_hash=file_sha256(CATALOG_PATH),
        tests_run=["tests/test_mechanics_gate.py::test_catalog_hash_change_invalidates_review"],
    )

    assert_mechanics_ready("test action", artifact_path=artifact_path)


def test_assert_mechanics_ready_rejects_dirty_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_clean_git(monkeypatch)
    artifact_path = tmp_path / "mechanics_gate.json"
    monkeypatch.setattr("vgc.mechanics_gate.GATE_ARTIFACT_PATH", artifact_path)
    coverage = json.loads(COVERAGE_PATH.read_text())
    for family in coverage["families"]:
        family["status"] = "exact"
        family.pop("gap", None)
        if not family.get("tests"):
            family["tests"] = ["tests/test_mechanics_gate.py::test_catalog_hash_change_invalidates_review"]
    regressed = tmp_path / "coverage.json"
    regressed.write_text(json.dumps(coverage))
    monkeypatch.setattr("vgc.mechanics_gate.COVERAGE_PATH", regressed)

    artifact_path.write_text(
        json.dumps(
            {
                "git_commit": FAKE_GIT_HEAD,
                "dirty": True,
                "coverage_sha256": file_sha256(regressed),
                "dependency_sha256": file_sha256(CATALOG_PATH),
                "timestamp_utc": "2026-09-01T00:00:00+00:00",
                "tests_run": [],
            }
        )
        + "\n"
    )

    with pytest.raises(MechanicsNotReadyError, match="artifact is missing or stale"):
        assert_mechanics_ready("test action", artifact_path=artifact_path)
