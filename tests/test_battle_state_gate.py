from __future__ import annotations

import json
from pathlib import Path

import pytest

from vgc.battle_state_gate import (
    COVERAGE_PATH,
    SET_PRIORS_PATH,
    BattleStateNotReadyError,
    assert_battle_state_ready,
    battle_state_readiness,
)
from vgc.gate_evidence import file_sha256, write_gate_artifact

FAKE_GIT_HEAD = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"


def _patch_clean_git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("vgc.gate_evidence.git_head", lambda: FAKE_GIT_HEAD)
    monkeypatch.setattr("vgc.gate_evidence.git_dirty", lambda: False)


def _all_exact_coverage() -> dict:
    coverage = json.loads(COVERAGE_PATH.read_text())
    coverage["release_ready"] = True
    for family in coverage["families"]:
        family["status"] = "exact"
        family.pop("gap", None)
    return coverage


def test_current_battle_state_gate_is_classified_but_not_ready_without_artifact() -> None:
    result = battle_state_readiness()
    assert not result.declared_ready
    assert result.priors_current
    assert result.schema_current
    assert result.branch_defaults_ready
    assert len(result.exact) == 14
    assert len(result.partial) == 2
    assert result.missing == ()
    assert not result.artifact_current
    assert not result.ready


def test_every_exact_family_names_automated_verification() -> None:
    coverage = json.loads(COVERAGE_PATH.read_text())
    for family in coverage["families"]:
        if family["status"] != "exact":
            continue
        assert family["id"]
        assert family["verification"]
        assert family.get("tests")


def test_exact_family_without_tests_raises_value_error(tmp_path: Path) -> None:
    coverage = json.loads(COVERAGE_PATH.read_text())
    coverage["families"][0]["tests"] = []
    path = tmp_path / "coverage.json"
    path.write_text(json.dumps(coverage))

    with pytest.raises(ValueError, match="no backing tests"):
        battle_state_readiness(path, SET_PRIORS_PATH)


def test_coverage_regression_blocks_the_gate(tmp_path: Path) -> None:
    coverage = json.loads(COVERAGE_PATH.read_text())
    coverage["families"][0]["status"] = "partial"
    path = tmp_path / "coverage.json"
    path.write_text(json.dumps(coverage))

    result = battle_state_readiness(path, SET_PRIORS_PATH)

    assert not result.ready
    assert result.partial


def test_joint_prior_drift_blocks_the_gate(tmp_path: Path) -> None:
    changed = tmp_path / "set_priors.json"
    changed.write_text(SET_PRIORS_PATH.read_text() + "\n")

    result = battle_state_readiness(COVERAGE_PATH, changed)

    assert not result.ready
    assert not result.priors_current


def test_runtime_gate_fails_closed_after_declared_regression(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    coverage = json.loads(COVERAGE_PATH.read_text())
    coverage["release_ready"] = False
    path = tmp_path / "coverage.json"
    path.write_text(json.dumps(coverage))
    monkeypatch.setattr("vgc.battle_state_gate.COVERAGE_PATH", path)

    with pytest.raises(BattleStateNotReadyError, match="release_ready"):
        assert_battle_state_ready("model training")


def test_public_and_training_entrypoints_enforce_part_b_gate() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = [
        "ladder/run_ladder.py",
        "selfplay/train_imitation.py",
        "selfplay/train_counterfactual_q.py",
        "selfplay/train_fixed_mirror.py",
        "selfplay/train_full_pipeline.py",
        "selfplay/train_ppo.py",
        "selfplay/train_search_value.py",
    ]
    for relative in paths:
        source = (root / relative).read_text()
        assert "enforce_battle_state_gate_for_cli" in source, relative


def test_unknown_test_node_id_blocks_gate_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    coverage = _all_exact_coverage()
    coverage["families"][0]["tests"] = ["tests/does_not_exist.py::test_missing"]
    regressed = tmp_path / "coverage.json"
    regressed.write_text(json.dumps(coverage))
    monkeypatch.setattr("vgc.battle_state_gate.COVERAGE_PATH", regressed)

    from offline.check_battle_state_readiness import main

    assert main([]) == 1


def test_assert_battle_state_ready_requires_current_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_path = tmp_path / "battle_state_gate.json"
    monkeypatch.setattr("vgc.battle_state_gate.GATE_ARTIFACT_PATH", artifact_path)

    with pytest.raises(BattleStateNotReadyError, match="artifact is missing or stale"):
        assert_battle_state_ready("model training", artifact_path=artifact_path)


def test_assert_battle_state_ready_passes_with_current_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_clean_git(monkeypatch)
    artifact_path = tmp_path / "battle_state_gate.json"
    coverage_path = tmp_path / "coverage.json"
    coverage_path.write_text(json.dumps(_all_exact_coverage()))
    monkeypatch.setattr("vgc.battle_state_gate.COVERAGE_PATH", coverage_path)
    monkeypatch.setattr("vgc.battle_state_gate.GATE_ARTIFACT_PATH", artifact_path)

    write_gate_artifact(
        artifact_path,
        coverage_hash=file_sha256(coverage_path),
        dependency_hash=file_sha256(SET_PRIORS_PATH),
        tests_run=["tests/test_battle_state_gate.py::test_joint_prior_drift_blocks_the_gate"],
    )

    assert_battle_state_ready("model training", artifact_path=artifact_path)


def test_assert_battle_state_ready_rejects_dirty_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_clean_git(monkeypatch)
    artifact_path = tmp_path / "battle_state_gate.json"
    coverage_path = tmp_path / "coverage.json"
    coverage_path.write_text(json.dumps(_all_exact_coverage()))
    monkeypatch.setattr("vgc.battle_state_gate.COVERAGE_PATH", coverage_path)
    monkeypatch.setattr("vgc.battle_state_gate.GATE_ARTIFACT_PATH", artifact_path)

    artifact_path.write_text(
        json.dumps(
            {
                "git_commit": FAKE_GIT_HEAD,
                "dirty": True,
                "coverage_sha256": file_sha256(coverage_path),
                "dependency_sha256": file_sha256(SET_PRIORS_PATH),
                "timestamp_utc": "2026-09-01T00:00:00+00:00",
                "tests_run": [],
            }
        )
        + "\n"
    )

    with pytest.raises(BattleStateNotReadyError, match="artifact is missing or stale"):
        assert_battle_state_ready("model training", artifact_path=artifact_path)
