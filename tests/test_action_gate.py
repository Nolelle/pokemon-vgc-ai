from __future__ import annotations

import json
from pathlib import Path

import pytest

import offline.check_action_readiness as action_readiness_cli
from vgc.action_gate import (
    COVERAGE_PATH,
    ActionNotReadyError,
    assert_action_ready,
    action_readiness,
)
from vgc.gate_evidence import file_sha256, write_gate_artifact
from vgc.showdown_parity import ParityReport

FAKE_GIT_HEAD = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"


def _patch_clean_git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("vgc.gate_evidence.git_head", lambda: FAKE_GIT_HEAD)
    monkeypatch.setattr("vgc.gate_evidence.git_dirty", lambda: False)


def test_current_action_gate_is_classified_but_not_ready_without_artifact(
    tmp_path: Path,
) -> None:
    result = action_readiness(artifact_path=tmp_path / "action_gate.json")
    assert result.declared_ready
    assert len(result.exact) == 7
    assert len(result.partial) == 0
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


def test_coverage_regression_blocks_the_gate(tmp_path: Path) -> None:
    coverage = json.loads(COVERAGE_PATH.read_text())
    coverage["families"][0]["status"] = "partial"
    path = tmp_path / "coverage.json"
    path.write_text(json.dumps(coverage))

    result = action_readiness(path, tmp_path / "action_gate.json")

    assert not result.ready
    assert result.partial


def test_runtime_gate_fails_closed_after_declared_regression(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    coverage = json.loads(COVERAGE_PATH.read_text())
    coverage["release_ready"] = False
    path = tmp_path / "coverage.json"
    path.write_text(json.dumps(coverage))
    monkeypatch.setattr("vgc.action_gate.COVERAGE_PATH", path)

    with pytest.raises(ActionNotReadyError, match="release_ready"):
        assert_action_ready("model training")


def test_public_and_training_entrypoints_enforce_action_gate() -> None:
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
        assert "enforce_action_gate_for_cli" in source, relative


def test_unknown_test_node_id_blocks_gate_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    coverage = json.loads(COVERAGE_PATH.read_text())
    coverage["families"][0]["tests"] = ["tests/does_not_exist.py::test_missing"]
    regressed = tmp_path / "coverage.json"
    regressed.write_text(json.dumps(coverage))
    monkeypatch.setattr("vgc.action_gate.COVERAGE_PATH", regressed)
    monkeypatch.setattr(
        action_readiness_cli,
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

    assert action_readiness_cli.main([]) == 1


def test_assert_action_ready_requires_current_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_path = tmp_path / "action_gate.json"
    monkeypatch.setattr("vgc.action_gate.GATE_ARTIFACT_PATH", artifact_path)

    with pytest.raises(ActionNotReadyError, match="artifact is missing or stale"):
        assert_action_ready("model training", artifact_path=artifact_path)


def test_assert_action_ready_passes_with_current_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_clean_git(monkeypatch)
    artifact_path = tmp_path / "action_gate.json"
    monkeypatch.setattr("vgc.action_gate.GATE_ARTIFACT_PATH", artifact_path)

    write_gate_artifact(
        artifact_path,
        coverage_hash=file_sha256(COVERAGE_PATH),
        dependency_hash=file_sha256(COVERAGE_PATH),
        tests_run=["tests/test_action_gate.py::test_coverage_regression_blocks_the_gate"],
    )

    assert_action_ready("model training", artifact_path=artifact_path)
