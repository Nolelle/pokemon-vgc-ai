"""Release checks reject weak, stale, swapped, and development-only evidence."""

import json
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from vgc.artifact_evidence import atomic_write_json, file_reference  # noqa: E402
from vgc.model_release import enforce_model_release_for_cli, validate_model_release  # noqa: E402
from vgc.models import PolicyConfig  # noqa: E402
from vgc.rl.opponents import RL_ARCHITECTURE_VERSION  # noqa: E402


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    config = PolicyConfig()
    evidence = {"schema": "fixture", "information_contract": "public_search_v1"}
    monkeypatch.setattr("vgc.model_release.decision_evidence", lambda *a, **kw: evidence)

    def save(name, value):
        path = tmp_path / f"{name}.json"
        atomic_write_json(path, value)
        return file_reference(path)

    audit = save("audit", {"verdict": "PASS", "errors": []})
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "architecture": RL_ARCHITECTURE_VERSION,
            "use_mechanics_features": True,
            "trained_outputs": {"action_preferences": True, "winning_chance": False},
            "training_audit_sha256": audit["sha256"],
            "development_team_sha256": ["training-team"],
        },
        checkpoint,
    )
    model_hash = file_reference(checkpoint)["sha256"]
    protocol = save("protocol", {
        "schema": "vgc-final-protocol-v1", "partition_role": "final_test",
        "frozen_commit": "pre-evaluation-commit", "team_sha256": ["final-team"],
        "minimum_paired_games": 1500, "minimum_teams": 20,
        "recall_threshold": 0.98, "noninferiority_margin": 0.02,
        "minimum_time_reduction": 0.15, "maximum_decision_ms": 10000,
    })
    common = {
        "checkpoint_sha256": model_hash, "decision_evidence": evidence,
        "protocol_sha256": protocol["sha256"], "partition_role": "final_test",
    }
    release = {
        "schema": "vgc-model-release-v1", "checkpoint_sha256": model_hash,
        "decision_evidence": evidence, "protocol": protocol, "training_audit": audit,
    }
    reports = {
        "recall": {"guided_safety_slots": 4, "by_k_guided": {"10": {
            "clustered_lower_bound": 0.985, "teams": 150,
            "decisions_without_guidance_metadata": 0,
        }}},
        "strength": {
            "mode": "hybrid", "search_k": 10, "safety_slots": 4,
            "paired_strength": {"paired_games": 1500, "teams": 150,
                                "team_clustered_difference_interval": [-0.01, 0.01]},
            "fallbacks": {"shadow": 0, "hybrid": 0},
            **{arm: {"retention": {"guided_at_10": {
                "team_clustered_interval": [0.985, 0.995],
            }}} for arm in ("shadow", "hybrid")},
        },
        "timing": {"same_hardware": True, "equal_budget": True,
                   "time_reduction_lower_bound": 0.2, "maximum_decision_ms": 1000},
        "local_smoke": {"games": 2, "fallbacks": 0, "completed": True},
        "restore": {"loaded": True, "backup_verified": True,
                    "restored_checkpoint_sha256": model_hash},
    }
    for name, data in reports.items():
        release[name] = save(name, {**common, **data})
    path = tmp_path / "release.json"
    atomic_write_json(path, release)
    return path, checkpoint, config, release


def test_bound_bundle_passes(bundle):
    path, checkpoint, config, _ = bundle
    validate_model_release(path, checkpoint, config, safety_slots=4)


@pytest.mark.parametrize("name,field,value", [
    ("recall", "partition_role", "development"),
    ("strength", "checkpoint_sha256", "different-model"),
    ("strength", "fallbacks", {"shadow": 0, "hybrid": 1}),
    ("timing", "time_reduction_lower_bound", 0.01),
    ("timing", "time_reduction_lower_bound", float("nan")),
    ("restore", "backup_verified", False),
    ("local_smoke", "completed", False),
])
def test_bad_measurement_rejected_even_with_updated_report_hash(bundle, name, field, value):
    path, checkpoint, config, release = bundle
    report_path = path.parent / f"{name}.json"
    report = json.loads(report_path.read_text())
    report[field] = value
    # A malformed producer can emit NaN; the verifier must still reject it.
    report_path.write_text(json.dumps(report))
    release[name] = file_reference(report_path)
    atomic_write_json(path, release)
    with pytest.raises(ValueError):
        validate_model_release(path, checkpoint, config, safety_slots=4)


def test_swapped_model_rejected(bundle):
    path, checkpoint, config, _ = bundle
    checkpoint.write_bytes(b"different bytes")
    with pytest.raises(ValueError, match="model mismatch"):
        validate_model_release(path, checkpoint, config, safety_slots=4)


def test_modified_report_without_new_hash_rejected(bundle):
    path, checkpoint, config, _ = bundle
    (path.parent / "recall.json").write_text("{}")
    with pytest.raises(ValueError, match="fingerprint"):
        validate_model_release(path, checkpoint, config, safety_slots=4)


def test_stale_decision_dependencies_rejected(bundle, monkeypatch):
    path, checkpoint, config, _ = bundle
    monkeypatch.setattr("vgc.model_release.decision_evidence", lambda *a, **kw: {})
    with pytest.raises(ValueError, match="stale"):
        validate_model_release(path, checkpoint, config, safety_slots=4)


def test_untrained_winning_chance_cannot_be_enabled(bundle):
    path, checkpoint, config, _ = bundle
    with pytest.raises(ValueError, match="untrained"):
        validate_model_release(path, checkpoint, replace(config, use_value_head=True), safety_slots=4)


def test_weak_sample_size_rejected(bundle):
    path, checkpoint, config, release = bundle
    p = path.parent / "strength.json"
    d = json.loads(p.read_text())
    d["paired_strength"]["paired_games"] = 100
    atomic_write_json(p, d)
    release["strength"] = file_reference(p)
    atomic_write_json(path, release)
    with pytest.raises(ValueError, match="too few paired"):
        validate_model_release(path, checkpoint, config, safety_slots=4)


def test_missing_release_fails_closed():
    with pytest.raises(SystemExit, match="--model-release"):
        enforce_model_release_for_cli(None, None, PolicyConfig(), safety_slots=4)
