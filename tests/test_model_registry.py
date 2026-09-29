"""The model register must not call a checkpoint loadable when the current data says otherwise.

A Champions data re-export changes embedding tables without changing the architecture
string, so "compatible" is only true relative to the exact ordered tokens in today's
`data/champions/*.json`. Compatible entries record an order-sensitive fingerprint of that
vocabulary; the static check needs no torch and no checkpoint files. When the files are
present (they live in the gitignored `runs/`), the entry is also checked against the real
bytes and a real `load_snapshot`.
"""

import json
from pathlib import Path

import pytest

from vgc.model_vocabulary import (
    VOCABULARY_KEY,
    current_vocabulary,
    vocabulary_fingerprint,
    vocabulary_mismatches,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY = json.loads((REPO_ROOT / "data/models/registry.json").read_text())
LOADABLE = [m for m in REGISTRY["models"] if m.get("loader")]


@pytest.mark.parametrize("entry", LOADABLE, ids=lambda m: m["id"])
def test_compatible_entries_match_current_data_vocabulary(entry):
    loader = entry["loader"]
    if loader["status"] != "compatible":
        assert loader.get("reason") or loader["status"] == "rejected_architecture"
        return
    assert loader.get("vocabulary_sha256") == vocabulary_fingerprint(current_vocabulary()), (
        f"{entry['id']}: compatible entry must record the current data vocabulary's "
        "fingerprint (vgc.model_vocabulary.vocabulary_fingerprint)"
    )


# What load_snapshot's refusal must mention for each non-compatible status, so an unrelated
# failure (a corrupt file, a different defect) cannot pass for the recorded reason.
_REFUSAL_REASON = {"incompatible_vocabulary": "vocabulary", "rejected_architecture": "architecture"}


@pytest.mark.parametrize("entry", LOADABLE, ids=lambda m: m["id"])
def test_registry_status_matches_real_checkpoint_when_present(entry):
    torch = pytest.importorskip("torch")
    from vgc.artifact_evidence import file_sha256
    from vgc.rl.opponents import load_snapshot

    path = REPO_ROOT / entry["checkpoint"]["path"]
    if not path.exists():
        pytest.skip(f"checkpoint not present in this checkout: {path}")
    assert file_sha256(path) == entry["checkpoint"]["sha256"], "registry hash is stale"
    status = entry["loader"]["status"]
    if status != "compatible":
        with pytest.raises(ValueError, match=_REFUSAL_REASON[status]):
            load_snapshot(path)
        return
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert not vocabulary_mismatches(payload.get(VOCABULARY_KEY))
    load_snapshot(path)


def _snapshot(tmp_path, vocabulary):
    torch = pytest.importorskip("torch")
    from vgc.rl.model import CandidatePolicyValueNet
    from vgc.rl.opponents import RL_ARCHITECTURE_VERSION

    payload = {
        "architecture": RL_ARCHITECTURE_VERSION,
        "model_state_dict": CandidatePolicyValueNet(use_mechanics_features=True).state_dict(),
        "use_mechanics_features": True,
    }
    if vocabulary is not None:
        payload[VOCABULARY_KEY] = vocabulary
    path = tmp_path / "snapshot.pt"
    torch.save(payload, path)
    return path


def test_load_snapshot_accepts_current_vocabulary(tmp_path):
    pytest.importorskip("torch")
    from vgc.rl.opponents import load_snapshot

    load_snapshot(_snapshot(tmp_path, current_vocabulary()))


def test_load_snapshot_refuses_reordered_vocabulary_of_equal_size(tmp_path):
    pytest.importorskip("torch")
    from vgc.rl.opponents import load_snapshot

    vocabulary = current_vocabulary()
    items = vocabulary["item"]
    items[2], items[3] = items[3], items[2]
    with pytest.raises(ValueError, match="different item tokens"):
        load_snapshot(_snapshot(tmp_path, vocabulary))


def test_load_snapshot_refuses_checkpoint_without_recorded_vocabulary(tmp_path):
    pytest.importorskip("torch")
    from vgc.rl.opponents import load_snapshot

    with pytest.raises(ValueError, match="records no data vocabulary"):
        load_snapshot(_snapshot(tmp_path, None))


def test_bc_policy_disabled_for_stale_vocabulary(tmp_path):
    torch = pytest.importorskip("torch")
    from vgc.bc.encoding import ENCODER_LAYOUT_VERSION
    from vgc.bc.policy import load_bc_policy

    vocabulary = current_vocabulary()
    vocabulary["species"] = vocabulary["species"][:-1]  # the M-B BC shape: one species short
    path = tmp_path / "bc.pt"
    torch.save(
        {
            "encoder_layout_version": ENCODER_LAYOUT_VERSION,
            **{f"{name}_vocab": tokens for name, tokens in vocabulary.items()},
            "target_vocab": [],
            "state_dict": {},
        },
        path,
    )
    assert load_bc_policy(path) is None
