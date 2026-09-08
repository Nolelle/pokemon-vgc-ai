"""Lossless neural input for the complete public mechanics snapshot.

The older learned inputs are useful summaries, but summaries necessarily discard
facts.  This encoder takes the canonical :mod:`vgc.mechanics_state` snapshot and turns
its complete JSON representation into byte tokens.  No hashing, fixed vocabulary, or
maximum length is involved, so two different snapshots always produce different token
sequences and a newly exposed field is included automatically.

Token 0 is reserved for batch padding.  Every UTF-8 byte is shifted by one, giving the
model tokens 1..256.  Keeping the wire representation simple also makes the contract
easy to audit and decode in tests.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from vgc.mechanics_state import BattleMechanicsState, snapshot_battle

MECHANICS_ENCODING_VERSION = "public-mechanics-json-v1"
MECHANICS_PAD_TOKEN = 0
MECHANICS_TOKEN_VOCAB_SIZE = 257


@dataclass(frozen=True)
class MechanicsFeatures:
    """A reversible, complete public battle observation for the neural model."""

    tokens: np.ndarray
    schema_version: str = MECHANICS_ENCODING_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != MECHANICS_ENCODING_VERSION:
            raise ValueError(
                f"unsupported mechanics encoding {self.schema_version!r}; "
                f"expected {MECHANICS_ENCODING_VERSION!r}"
            )
        if self.tokens.ndim != 1 or self.tokens.dtype != np.int64:
            raise ValueError("mechanics tokens must be a one-dimensional int64 array")
        if not len(self.tokens) or np.any(self.tokens <= 0) or np.any(self.tokens >= 257):
            raise ValueError("mechanics tokens must contain only shifted UTF-8 bytes 1..256")

    @property
    def digest(self) -> str:
        """Stable identity used to prove stored and live observations match."""

        return hashlib.sha256(self.to_json_bytes()).hexdigest()

    def to_json_bytes(self) -> bytes:
        return bytes((self.tokens - 1).astype(np.uint8).tolist())

    def decoded(self) -> dict[str, object]:
        return json.loads(self.to_json_bytes())


def canonical_mechanics_json(state: BattleMechanicsState) -> bytes:
    """Serialize every snapshot field deterministically, with no lossy fallback."""

    return json.dumps(
        asdict(state),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def encode_mechanics_state(state: BattleMechanicsState) -> MechanicsFeatures:
    raw = np.frombuffer(canonical_mechanics_json(state), dtype=np.uint8).astype(np.int64)
    return MechanicsFeatures(tokens=raw + 1)


def encode_mechanics_context(battle) -> MechanicsFeatures:
    """Snapshot and losslessly encode only what this player can publicly observe."""

    return encode_mechanics_state(snapshot_battle(battle))


def pad_mechanics_features(
    features: Sequence[MechanicsFeatures],
) -> tuple[np.ndarray, np.ndarray]:
    """Pad a batch without truncation and return ``(tokens, real-token mask)``."""

    if not features:
        raise ValueError("cannot pad an empty mechanics batch")
    width = max(len(feature.tokens) for feature in features)
    tokens = np.full((len(features), width), MECHANICS_PAD_TOKEN, dtype=np.int64)
    mask = np.zeros((len(features), width), dtype=np.bool_)
    for row, feature in enumerate(features):
        size = len(feature.tokens)
        tokens[row, :size] = feature.tokens
        mask[row, :size] = True
    return tokens, mask
