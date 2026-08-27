from __future__ import annotations

import json

import pytest

from vgc.mechanics_gate import (
    MechanicsNotReadyError,
    assert_mechanics_ready,
    enforce_mechanics_gate_for_cli,
    mechanics_readiness,
)


def test_checked_in_catalog_is_current_but_release_is_blocked() -> None:
    result = mechanics_readiness()

    assert result.catalog_current
    assert result.exact
    assert result.partial
    assert result.missing
    assert not result.declared_ready
    assert not result.ready


def test_release_gate_fails_closed_with_useful_counts() -> None:
    with pytest.raises(MechanicsNotReadyError, match="mechanics families are missing"):
        assert_mechanics_ready("test action")


def test_cli_gate_fails_without_internal_traceback_contract() -> None:
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
