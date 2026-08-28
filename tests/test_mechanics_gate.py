from __future__ import annotations

import json

import pytest

from vgc.mechanics_gate import (
    CATALOG_PATH,
    COVERAGE_PATH,
    assert_mechanics_ready,
    enforce_mechanics_gate_for_cli,
    mechanics_readiness,
)


VALID_ENGINE_STATES = frozenset(
    {
        "official_showdown",
        "official_showdown_clone",
        "official_showdown_clone_with_belief_branches",
        "lossless_public_json_tokens",
        "captured",
    }
)


def test_checked_in_catalog_is_current_and_every_family_is_exact() -> None:
    result = mechanics_readiness()

    assert result.catalog_current
    assert result.exact
    assert not result.partial
    assert not result.missing
    assert result.declared_ready
    assert result.ready


def test_every_exact_family_names_a_real_engine_and_its_evidence() -> None:
    """"Exact" has to point at something, or the gate is just a boolean someone set."""

    coverage = json.loads(COVERAGE_PATH.read_text())
    for family in coverage["families"]:
        assert family["state"] in VALID_ENGINE_STATES, family
        assert family.get("verification") not in (None, "", "none"), family


def test_release_gate_passes_for_the_checked_in_coverage() -> None:
    assert_mechanics_ready("test action")


def test_a_single_regressed_family_blocks_the_whole_release(tmp_path) -> None:
    """The gate must stay fail-closed now that it is green."""

    coverage = json.loads(COVERAGE_PATH.read_text())
    coverage["families"][0]["status"] = "partial"
    regressed = tmp_path / "coverage.json"
    regressed.write_text(json.dumps(coverage))

    assert not mechanics_readiness(CATALOG_PATH, regressed).ready


def test_a_blocked_scope_alone_blocks_the_release(tmp_path) -> None:
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


def test_catalog_hash_change_invalidates_review(tmp_path) -> None:
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
                    {"id": "example", "status": "exact", "state": "captured"}
                ],
            }
        )
    )

    result = mechanics_readiness(catalog, coverage)

    assert not result.catalog_current
    assert not result.ready
