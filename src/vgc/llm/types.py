"""Plain data types shared across the LLM layer."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Option:
    """One numbered joint order the model may pick. `id` looks like 'P07'."""

    id: str
    order: str  # e.g. "/choose move earthquake, switch 3"
    note: str = ""  # one-line engine note (score, gap to top, kind)
    kind: str = "attack"  # attack | switch | protect


@dataclass(frozen=True)
class ContextPacket:
    """What the model sees. `fixed_text` is the cacheable prefix, `turn_text` changes."""

    fixed_text: str
    turn_text: str
    option_ids: tuple[str, ...]
    request_id: str = ""  # one per turn/attempt chain; late answers for old ids are dropped
    turn: int | None = None

    @property
    def full_text(self) -> str:
        return self.fixed_text + "\n\n" + self.turn_text

    @property
    def prompt_sha256(self) -> str:
        return hashlib.sha256(self.full_text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Proposal:
    id: str
    why: str = ""


@dataclass(frozen=True)
class Advice:
    plan: str
    proposals: tuple[Proposal, ...]


@dataclass(frozen=True)
class RawResult:
    """What a client hands back, before any validation."""

    text: str | None  # the structured-output JSON string, if any
    status: str = "completed"  # completed | incomplete | failed
    request_id: str = ""  # echo of ContextPacket.request_id
    input_tokens: int = 0  # includes cached and cache-write tokens
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0  # includes thinking tokens
    refusal: str | None = None
    response_id: str = ""


@dataclass
class CallRecord:
    """One JSONL row per call (plus one per late arrival after we gave up)."""

    request_id: str
    model: str
    level: str
    prompt_sha256: str
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    cost_usd: float = 0.0
    status: str = "ok"
    error: str = ""
    turn: int | None = None
    attempt: int = 0
    invalid_ids: int = 0
    ungrounded_names: list[str] = field(default_factory=list)
    ts: float = 0.0
