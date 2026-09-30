"""Which game-data vocabulary a saved model was trained with.

Every learned model embeds species/items/abilities/moves by their index in
`vgc.bc.encoding`'s vocabularies, which are built from `data/champions/*.json`. A data
re-export (e.g. Reg M-B -> M-C on 2026-09-09) can add, remove or reorder tokens without
changing any architecture string, so a checkpoint is only valid against the exact ordered
token lists it was trained with. Row counts are not enough: an equal-sized table with a
different order loads cleanly and reads every index as the wrong thing.

RL checkpoints store the lists under `VOCABULARY_KEY`; BC checkpoints have always stored
them as `species_vocab`/`item_vocab`/`ability_vocab`/`move_vocab`. A checkpoint with no
recorded vocabulary cannot be verified and is refused. Torch-free on purpose, so the
registry check runs without the `train` extra.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from vgc.bc.encoding import ABILITY_VOCAB, ITEM_VOCAB, MOVE_VOCAB, SPECIES_VOCAB

VOCABULARY_KEY = "data_vocabulary"

# Embedding-weight key (relative to the BC/state encoder) for each vocabulary.
EMBEDDING_KEYS = {
    "species": "species_embedding.weight",
    "item": "item_embedding.weight",
    "ability": "ability_embedding.weight",
    "move": "move_embedding.weight",
}


def current_vocabulary() -> dict[str, list[str]]:
    """Ordered tokens the current Champions data export gives each embedding table."""

    return {
        "species": list(SPECIES_VOCAB),
        "item": list(ITEM_VOCAB),
        "ability": list(ABILITY_VOCAB),
        "move": list(MOVE_VOCAB),
    }


def vocabulary_fingerprint(vocabulary: Mapping[str, list[str]]) -> str:
    """Order-sensitive SHA-256 of a vocabulary, for recording in the model registry."""

    canonical = json.dumps({name: list(vocabulary[name]) for name in EMBEDDING_KEYS})
    return hashlib.sha256(canonical.encode()).hexdigest()


def bc_checkpoint_vocabulary(checkpoint: Mapping) -> dict[str, list[str]] | None:
    """The vocabulary a BC checkpoint recorded, or None if it recorded none."""

    keys = {name: f"{name}_vocab" for name in EMBEDDING_KEYS}
    if not all(key in checkpoint for key in keys.values()):
        return None
    return {name: list(checkpoint[key]) for name, key in keys.items()}


def vocabulary_mismatches(saved: Mapping[str, list[str]] | None) -> list[str]:
    """Names of the tables whose saved tokens differ from today's (all of them if none)."""

    current = current_vocabulary()
    if saved is None:
        return list(current)
    return [name for name, tokens in current.items() if list(saved.get(name, ())) != tokens]


def _row_counts(state_dict: Mapping, prefix: str) -> str:
    current = current_vocabulary()
    parts = []
    for name, key in EMBEDDING_KEYS.items():
        weight = state_dict.get(prefix + key)
        if weight is not None and weight.shape[0] != len(current[name]):
            parts.append(
                f"{name}: checkpoint {weight.shape[0]} rows vs current {len(current[name])}"
            )
    return "; ".join(parts)


def require_current_vocabulary(
    checkpoint: Mapping, path: str | Path, *, state_prefix: str = "state_encoder."
) -> None:
    """Raise ValueError unless an RL checkpoint was trained on today's exact vocabulary."""

    saved = checkpoint.get(VOCABULARY_KEY)
    mismatched = vocabulary_mismatches(saved)
    if not mismatched:
        return
    rows = _row_counts(checkpoint.get("model_state_dict", {}), state_prefix)
    reason = (
        "records no data vocabulary, so its token order cannot be verified"
        if saved is None
        else f"was trained on different {'/'.join(mismatched)} tokens"
    )
    raise ValueError(
        f"checkpoint {path} {reason} against the current data/champions export"
        + (f" ({rows})" if rows else "")
        + "; retrain it on the current format data"
    )
