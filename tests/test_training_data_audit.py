from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("torch")

from offline.audit_training_data import audit
from vgc.rl.demonstrations import (
    INFORMATION_CONTRACT_VERSION,
    file_sha256,
    save_demonstrations,
    save_split_manifest,
)
from vgc.rl.distill import DistillationSample, PUBLIC_TEACHER_SOURCE_ID
from vgc.rl.encoding import CandidateFeatures, InformationFeatures
from vgc.rl.mechanics_encoding import MechanicsFeatures


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _sample(battle: str, team: str, opponent: str) -> DistillationSample:
    return DistillationSample(
        battle_id=battle,
        state_indices=np.zeros(4, dtype=np.int64),
        state_scalars=np.zeros(4, dtype=np.float32),
        history_scalars=np.zeros(4, dtype=np.float32),
        candidates=CandidateFeatures(
            move_indices=np.zeros((2, 2), dtype=np.int64),
            target_indices=np.zeros((2, 2), dtype=np.int64),
            switch_species_indices=np.zeros((2, 2), dtype=np.int64),
            flags=np.zeros((2, 2, 4), dtype=np.float32),
            tactical=np.zeros((2, 42), dtype=np.float32),
        ),
        teacher_action_index=1,
        decision_index=1,
        request_kind="move",
        information=InformationFeatures(
            indices=np.zeros(84, dtype=np.int64),
            scalars=np.zeros(384, dtype=np.float32),
        ),
        mechanics=MechanicsFeatures(
            tokens=np.frombuffer(b"{}", dtype=np.uint8).astype(np.int64) + 1
        ),
        source_id=PUBLIC_TEACHER_SOURCE_ID,
        team_id=team,
        opponent_team_id=opponent,
        turn=1,
        legal_action_count=2,
        candidate_myopic_ranks=np.arange(2, dtype=np.int64),
        candidate_tags=np.zeros((2, 5), dtype=np.int8),
        search_scores=np.asarray([1.0, 2.0], dtype=np.float32),
        searched_mask=np.ones(2, dtype=np.bool_),
        candidate_descriptions=("protect / protect", "tackle@1 / protect"),
        teacher_action_description="tackle@1 / protect",
        team_sha256=_hash(team),
        opponent_team_sha256=_hash(opponent),
    )


def _metadata() -> dict[str, object]:
    return {
        "created_at_utc": "2026-08-30T00:00:00+00:00",
        "repository_commit": "a" * 40,
        "repository_dirty": False,
        "showdown_commit": "b" * 40,
        "showdown_dirty": False,
        "format_id": "gen9championsvgc2026regmb",
        "collector": "test",
        "requested_games": 2,
        "seed": 1,
        "team_source": "test",
        "opponents": ["test"],
        "policy_config": {},
        "information_contract": INFORMATION_CONTRACT_VERSION,
        "teacher_source": PUBLIC_TEACHER_SOURCE_ID,
    }


def _valid_dataset(tmp_path):
    dataset = tmp_path / "demonstrations.pt"
    manifest = tmp_path / "split_manifest.json"
    train = [_sample("train-battle", "train-a", "train-b")]
    validation = [_sample("validation-battle", "validation-a", "validation-b")]
    save_demonstrations(dataset, [*train, *validation], metadata=_metadata())
    save_split_manifest(
        manifest,
        dataset_path=dataset,
        train=train,
        validation=validation,
        group_by="team",
        seed=1,
        val_fraction=0.5,
    )
    return dataset, manifest


def test_audit_passes_complete_disjoint_dataset(tmp_path) -> None:
    dataset, manifest = _valid_dataset(tmp_path)

    report = audit(dataset, manifest)

    assert report["verdict"] == "PASS"
    assert report["errors"] == []
    assert report["counts"]["sample_count"] == 2
    assert report["counts"]["split_samples"] == {"train": 1, "validation": 1}


def test_audit_fails_when_team_content_crosses_split(tmp_path) -> None:
    dataset, manifest = _valid_dataset(tmp_path)
    import torch

    payload = torch.load(dataset, map_location="cpu", weights_only=False)
    payload["samples"][1] = _sample("validation-battle", "validation-a", "train-b")
    torch.save(payload, dataset)
    split = json.loads(manifest.read_text())
    from vgc.rl.demonstrations import file_sha256

    split["dataset_sha256"] = file_sha256(dataset)
    manifest.write_text(json.dumps(split))

    report = audit(dataset, manifest)

    assert report["verdict"] == "FAIL"
    assert any("packed-team fingerprints cross" in error for error in report["errors"])


def test_audit_fails_duplicate_decision_and_illegal_teacher_reference(tmp_path) -> None:
    dataset, manifest = _valid_dataset(tmp_path)
    import torch

    payload = torch.load(dataset, map_location="cpu", weights_only=False)
    duplicate = replace(
        payload["samples"][0],
        teacher_action_description="not-a-legal-candidate",
    )
    payload["samples"].append(duplicate)
    payload["sample_count"] += 1
    torch.save(payload, dataset)
    split = json.loads(manifest.read_text())
    from vgc.rl.demonstrations import file_sha256

    split["dataset_sha256"] = file_sha256(dataset)
    split["train_samples"] += 1
    manifest.write_text(json.dumps(split))

    report = audit(dataset, manifest)

    assert report["verdict"] == "FAIL"
    assert any("teacher action does not match" in error for error in report["errors"])
    assert any("duplicate battle/decision" in error for error in report["errors"])


def _shard_pair(tmp_path):
    """Two shard files sharing one battle id, as real sharded collection does."""
    from offline.merge_collection_shards import load_shard_datasets

    first = tmp_path / "shard-a" / "demonstrations.pt"
    second = tmp_path / "shard-b" / "demonstrations.pt"
    first.parent.mkdir()
    second.parent.mkdir()
    save_demonstrations(
        first, [_sample("imitation-train-000000", "team-a", "team-b")], metadata=_metadata()
    )
    save_demonstrations(
        second, [_sample("imitation-train-000000", "team-c", "team-d")], metadata=_metadata()
    )
    return first, second, load_shard_datasets


def test_shard_loader_namespaces_colliding_battle_ids(tmp_path) -> None:
    first, second, load_shard_datasets = _shard_pair(tmp_path)

    samples, _ = load_shard_datasets([first, second])

    assert [sample.battle_id for sample in samples] == [
        f"{file_sha256(first)}:imitation-train-000000",
        f"{file_sha256(second)}:imitation-train-000000",
    ]


def test_shard_loader_rejects_heterogeneous_metadata(tmp_path) -> None:
    first, second, load_shard_datasets = _shard_pair(tmp_path)
    import torch

    payload = torch.load(second, map_location="cpu", weights_only=False)
    payload["metadata"] = {**payload["metadata"], "format_id": "something-else"}
    torch.save(payload, second)

    with pytest.raises(SystemExit, match="disagree on format_id"):
        load_shard_datasets([first, second])


def test_audit_passes_multi_file_manifest(tmp_path) -> None:
    first = tmp_path / "shard-a.pt"
    second = tmp_path / "shard-b.pt"
    manifest = tmp_path / "split_manifest.json"
    train = [_sample("battle-a", "team-a", "team-b")]
    validation = [_sample("battle-b", "team-c", "team-d")]
    save_demonstrations(first, train, metadata=_metadata())
    save_demonstrations(second, validation, metadata=_metadata())
    save_split_manifest(
        manifest,
        dataset_paths=[first, second],
        train=train,
        validation=validation,
        group_by="team",
        seed=1,
        val_fraction=0.5,
    )

    report = audit([first, second], manifest)

    assert report["verdict"] == "PASS", report["errors"]
    assert report["counts"]["sample_count"] == 2
    assert isinstance(report["dataset"], list)


def test_audit_fails_multi_file_fingerprint_mismatch(tmp_path) -> None:
    import torch

    first = tmp_path / "shard-a.pt"
    second = tmp_path / "shard-b.pt"
    manifest = tmp_path / "split_manifest.json"
    train = [_sample("battle-a", "team-a", "team-b")]
    validation = [_sample("battle-b", "team-c", "team-d")]
    save_demonstrations(first, train, metadata=_metadata())
    save_demonstrations(second, validation, metadata=_metadata())
    save_split_manifest(
        manifest,
        dataset_paths=[first, second],
        train=train,
        validation=validation,
        group_by="team",
        seed=1,
        val_fraction=0.5,
    )
    payload = torch.load(second, map_location="cpu", weights_only=False)
    payload["samples"].append(_sample("battle-c", "team-c", "team-d"))
    torch.save(payload, second)

    report = audit([first, second], manifest)

    assert report["verdict"] == "FAIL"
    assert any("fingerprint does not match" in error for error in report["errors"])


def test_load_training_dataset_single_keeps_ids_multi_namespaces(tmp_path) -> None:
    from selfplay.train_imitation import load_training_dataset

    first = tmp_path / "shard-a" / "demonstrations.pt"
    second = tmp_path / "shard-b" / "demonstrations.pt"
    first.parent.mkdir()
    second.parent.mkdir()
    save_demonstrations(first, [_sample("game-0", "team-a", "team-b")], metadata=_metadata())
    save_demonstrations(second, [_sample("game-0", "team-c", "team-d")], metadata=_metadata())

    single_samples, _ = load_training_dataset([first])
    assert [sample.battle_id for sample in single_samples] == [f"{file_sha256(first)}:game-0"]

    multi_samples, multi_metadata = load_training_dataset([first, second])
    assert [sample.battle_id for sample in multi_samples] == [
        f"{file_sha256(first)}:game-0",
        f"{file_sha256(second)}:game-0",
    ]
    assert len(multi_metadata["source_datasets"]) == 2
