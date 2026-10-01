"""The auto-sync may re-pin on its own only when the catalogue's content is unchanged."""

from __future__ import annotations

import hashlib
import json

from vgc.showdown_sync import catalog_changes, repin_catalog_hash


def _catalog(commit: str, encore_hooks: list[str]) -> dict:
    return {
        "generated_from": {"showdown_commit": commit, "mod": "champions"},
        "callbacks": {"condition.onOverrideAction": encore_hooks},
    }


def test_a_new_showdown_commit_alone_is_not_a_catalogue_change() -> None:
    assert catalog_changes(_catalog("aaa", ["encore"]), _catalog("bbb", ["encore"])) == []


def test_a_removed_engine_hook_needs_review() -> None:
    changes = catalog_changes(_catalog("aaa", ["encore"]), _catalog("bbb", []))
    assert changes == ["removed callbacks/condition.onOverrideAction: 'encore'"]


def test_other_generated_from_fields_are_still_compared() -> None:
    old = _catalog("aaa", [])
    new = _catalog("aaa", [])
    new["generated_from"]["mod"] = "champions2"
    assert catalog_changes(old, new) == [
        "changed generated_from/mod: 'champions' -> 'champions2'"
    ]


def test_repin_writes_the_catalogue_hash(tmp_path) -> None:
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps(_catalog("bbb", [])))
    coverage = tmp_path / "coverage.json"
    coverage.write_text('{\n  "catalog_sha256": "old",\n  "families": []\n}\n')
    digest = repin_catalog_hash(coverage, catalog)
    assert digest == hashlib.sha256(catalog.read_bytes()).hexdigest()
    assert json.loads(coverage.read_text())["catalog_sha256"] == digest
