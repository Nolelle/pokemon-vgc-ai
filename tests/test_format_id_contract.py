"""Fail-closed `format_id` contract: parsed replay datasets must say which regulation they
came from, and training must refuse to mix datasets that do not (CLAUDE.md: "Mix formats
in training only with an explicit `format_id` on every dataset").
"""

from __future__ import annotations

import json
import sys

import pytest

from tools import parse_replays
from tools.parse_replays import FormatIdError, resolve_format_id

MB = "gen9championsvgc2026regmb"
MC = "gen9championsvgc2026regmc"


def test_resolve_format_id_fails_on_disagreement_or_missing(tmp_path):
    assert resolve_format_id({"formatid": MC}, tmp_path / MC) == MC
    assert resolve_format_id({}, tmp_path / MC) == MC
    assert resolve_format_id({"formatid": MC}, tmp_path / "scratch") == MC
    with pytest.raises(FormatIdError, match="disagrees"):
        resolve_format_id({"formatid": MB}, tmp_path / MC)
    with pytest.raises(FormatIdError, match="no formatid"):
        resolve_format_id({}, tmp_path / "scratch")


def test_parse_replays_aborts_without_writing_on_mismatch(tmp_path, monkeypatch):
    replays_dir = tmp_path / MC
    replays_dir.mkdir()
    (replays_dir / "a.json").write_text(json.dumps({"id": "a", "formatid": MB, "log": "|"}))
    out = tmp_path / "decisions.jsonl"
    out.write_text("previous run\n")
    monkeypatch.setattr(
        sys, "argv", ["parse_replays", "--replays-dir", str(replays_dir), "--out", str(out)]
    )
    assert parse_replays.main() == 2
    assert out.read_text() == "previous run\n"
    assert not out.with_suffix(".jsonl.tmp").exists()


def _write(path, format_ids):
    with path.open("w") as file:
        for i, format_id in enumerate(format_ids):
            record = {"decision_kind": "teampreview", "replay_id": f"r{i}"}
            if format_id:
                record["format_id"] = format_id
            file.write(json.dumps(record) + "\n")
    return path


def test_dataset_refuses_partial_labels_and_unlabelled_mixes(tmp_path):
    dataset = pytest.importorskip("vgc.bc.dataset")

    with pytest.raises(dataset.FormatMixError, match="lack format_id"):
        dataset.BcTurnDataset(_write(tmp_path / "partial.jsonl", [MC, None]))
    with pytest.raises(dataset.FormatMixError, match="lack format_id"):
        dataset.BcTurnDataset(_write(tmp_path / "blank.jsonl", [MC, "  "]))

    mc = dataset.BcTurnDataset(_write(tmp_path / "mc.jsonl", [MC, MC]))
    mb = dataset.BcTurnDataset(_write(tmp_path / "mb.jsonl", [MB]))
    legacy = dataset.BcTurnDataset(_write(tmp_path / "legacy.jsonl", [None]))
    assert mc.format_ids == {MC}
    assert legacy.format_ids == set()

    # A lone legacy file still loads; mixing it with anything does not.
    assert dataset.check_format_mix([legacy]) == set()
    with pytest.raises(dataset.FormatMixError, match="without format_id"):
        dataset.check_format_mix([mc, legacy])
    # Explicitly labelled formats may be mixed, visibly.
    with pytest.warns(UserWarning, match="mixes formats"):
        assert dataset.check_format_mix([mc, mb]) == {MB, MC}
