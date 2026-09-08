"""Content identities for reproducible model evidence; paths are not identities."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def file_reference(path: Path) -> dict[str, str]:
    path = Path(path).resolve()
    return {"path": str(path), "sha256": file_sha256(path)}


def atomic_write_json(path: Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def decision_evidence(config, *, safety_slots: int, root: Path = REPO_ROOT) -> dict:
    """Fingerprint decision code/data/settings, excluding docs and scratch artifacts.

    The conservative source boundary includes the vgc package. This deliberately
    invalidates approval after a package change, but never after a docs-only edit.
    Showdown runtime parity is separately enforced by the existing readiness gates.
    """
    from vgc.rl.public_search import PUBLIC_SEARCH_CONTRACT_VERSION

    root = Path(root)
    files = set((root / "src/vgc").rglob("*.py"))
    for directory in ("data/champions", "data/usage", "data/meta"):
        files.update((root / directory).glob("*.json"))
    for name in ("tools/sim_worker.mjs", "ladder/run_ladder.py", "uv.lock"):
        if (root / name).is_file():
            files.add(root / name)
    fingerprints = {str(p.relative_to(root)): file_sha256(p) for p in sorted(files)}
    dependencies = {}
    for name in ("poke-env", "numpy", "torch"):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    settings = asdict(config)
    # Logging does not change a chosen order and differs between test and ladder.
    settings.pop("log_decisions", None)
    settings = json.loads(json.dumps(settings, default=str))
    payload = {
        "schema": "vgc-decision-evidence-v1",
        "information_contract": PUBLIC_SEARCH_CONTRACT_VERSION,
        "files": fingerprints,
        "dependencies": dependencies,
        "policy_config": settings,
        "safety_slots": safety_slots,
    }
    return {**payload, "sha256": json_sha256(payload)}


def read_verified_json(reference: dict, *, base: Path) -> dict:
    path = Path(reference["path"])
    if not path.is_absolute():
        path = base / path
    if file_sha256(path) != reference.get("sha256"):
        raise ValueError(f"artifact fingerprint mismatch: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"artifact must be a JSON object: {path}")
    return value
