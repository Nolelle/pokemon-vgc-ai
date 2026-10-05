"""LLM clients: a protocol, a scripted fake for tests, and the real OpenAI Responses client.

Only `OpenAIResponsesClient.complete` ever touches the network, and it imports `openai`
lazily, so nothing else in `vgc.llm` needs the package installed.
"""

from __future__ import annotations

import json
import time
from typing import Any, Protocol

from vgc.llm.types import ContextPacket, RawResult


class LLMError(Exception):
    """Base class for transport-level failures a client may raise."""


class RateLimited(LLMError):
    pass


class ConnectionFailed(LLMError):
    pass


class LLMTimeout(LLMError):
    pass


class LLMClient(Protocol):
    def complete(
        self, packet: ContextPacket, level: str, max_output_tokens: int, timeout_s: float
    ) -> RawResult: ...


def response_schema(option_ids: tuple[str, ...] | list[str]) -> dict[str, Any]:
    """Strict structured-output schema: IDs are an enum of the CURRENT option IDs.

    The 'at most 3' limit is enforced by the harness, not the schema (strict mode support
    for array size keywords varies).
    """
    return {
        "type": "object",
        "properties": {
            "plan": {"type": "string"},
            "proposals": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "enum": list(option_ids)},
                        "why": {"type": "string"},
                    },
                    "required": ["id", "why"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["plan", "proposals"],
        "additionalProperties": False,
    }


# ---------------------------------------------------------------------------------------
# Fake client
# ---------------------------------------------------------------------------------------

SCENARIOS = (
    "valid", "malformed_json", "unknown_id", "duplicate_ids", "stale_id", "refusal", "empty",
    "incomplete_max_output_tokens", "rate_limited", "connection_error", "slow", "stalled",
    "late_after_cancel", "plausible_false_reasoning",
    "preview_three_species", "preview_bad_leads",
)  # fmt: skip


def _answer(ids: list[str], plan: str = "Pressure the right-hand target.", why: str = "good") -> str:
    return json.dumps(
        {"plan": plan, "proposals": [{"id": i, "why": why} for i in ids]}
    )


def _preview_answer(bring: list[str], leads: list[str]) -> str:
    return json.dumps(
        {"plan": "Lead with the pair that fits their likely bring.", "bring": bring,
         "leads": leads, "why": "fake preview answer"}
    )


class FakeLLMClient:
    """Scripted client. `scenario` is one of SCENARIOS; `calls` records every request."""

    def __init__(
        self,
        scenario: str = "valid",
        *,
        sleep_s: float = 0.05,
        stale_ids: tuple[str, ...] = ("P97",),
    ) -> None:
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario {scenario!r}")
        self.scenario = scenario
        self.sleep_s = sleep_s
        self.stale_ids = stale_ids
        self.calls: list[ContextPacket] = []

    def complete(
        self, packet: ContextPacket, level: str, max_output_tokens: int, timeout_s: float
    ) -> RawResult:
        self.calls.append(packet)
        s = self.scenario
        ids = list(packet.option_ids)
        if packet.kind == "preview":
            return self._preview(packet, max_output_tokens, timeout_s)
        usage = {
            "input_tokens": len(packet.full_text) // 4,
            "output_tokens": 80,
            "request_id": packet.request_id,
        }

        def ok(text: str | None, **kw: Any) -> RawResult:
            return RawResult(text=text, **{**usage, **kw})

        if s == "valid":
            return ok(_answer(ids[:2]))
        if s == "malformed_json":
            return ok('{"plan": "x", "proposals": [{"id": ')
        if s == "unknown_id":
            return ok(_answer(["P99", "Z01"]))
        if s == "duplicate_ids":
            return ok(_answer([ids[0]] * 3))
        if s == "stale_id":
            return ok(_answer([*self.stale_ids, ids[0]]))
        if s == "refusal":
            return ok(None, refusal="I can't help with that.")
        if s == "empty":
            return ok("")
        if s == "incomplete_max_output_tokens":
            return ok(None, status="incomplete", output_tokens=max_output_tokens)
        if s == "rate_limited":
            raise RateLimited("429 too many requests")
        if s == "connection_error":
            raise ConnectionFailed("connection reset")
        if s == "slow":
            time.sleep(self.sleep_s)
            return ok(_answer(ids[:1]))
        if s == "stalled":
            time.sleep(timeout_s + 0.15)
            raise LLMTimeout("request timed out")
        if s == "late_after_cancel":
            time.sleep(timeout_s + 0.3)  # ignores its own timeout
            return ok(_answer(ids[:1]))
        # plausible_false_reasoning: valid ID, reasoning names things not in the packet
        return ok(
            _answer(
                ids[:1],
                plan="Garchomp is faster, so Close Combat from Lucario with Choice Scarf wins.",
                why="Choice Scarf Garchomp outspeeds",
            )
        )

    def _preview(self, packet: ContextPacket, max_output_tokens: int, timeout_s: float) -> RawResult:
        """Scripted team-preview answers. `packet.option_ids` are OUR six species names.

        The "valid" answer brings names[2:6] and leads names[5], names[3] (so it differs from
        the natural 1234 order); the other scenarios break one rule each.
        """
        s = self.scenario
        names = list(packet.option_ids)
        usage = {"input_tokens": len(packet.full_text) // 4, "output_tokens": 80,
                 "request_id": packet.request_id}

        def ok(text: str | None, **kw: Any) -> RawResult:
            return RawResult(text=text, **{**usage, **kw})

        bring = names[2:6]
        leads = [names[5], names[3]]
        if s in ("valid", "slow", "plausible_false_reasoning"):
            if s == "slow":
                time.sleep(self.sleep_s)
            return ok(_preview_answer(bring, leads))
        if s == "malformed_json":
            return ok('{"plan": "x", "bring": [')
        if s in ("unknown_id", "stale_id"):
            return ok(_preview_answer([*bring[:3], "Missingno"], leads))
        if s == "duplicate_ids":
            return ok(_preview_answer([names[0]] * 4, [names[0], names[0]]))
        if s == "preview_three_species":
            return ok(_preview_answer(bring[:3], bring[:2]))
        if s == "preview_bad_leads":
            return ok(_preview_answer(bring, [names[0], names[1]]))
        if s == "refusal":
            return ok(None, refusal="I can't help with that.")
        if s == "empty":
            return ok("")
        if s == "incomplete_max_output_tokens":
            return ok(None, status="incomplete", output_tokens=max_output_tokens)
        if s == "rate_limited":
            raise RateLimited("429 too many requests")
        if s == "connection_error":
            raise ConnectionFailed("connection reset")
        if s == "stalled":
            time.sleep(timeout_s + 0.15)
            raise LLMTimeout("request timed out")
        time.sleep(timeout_s + 0.3)  # late_after_cancel: ignores its own timeout
        return ok(_preview_answer(bring, leads))


# ---------------------------------------------------------------------------------------
# Real client
# ---------------------------------------------------------------------------------------


class OpenAIResponsesClient:
    """Luna via the OpenAI Responses API. The only code here that uses the network."""

    def __init__(self, model: str = "gpt-6-luna", api_key: str | None = None) -> None:
        self.model = model
        self._api_key = api_key
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            import openai  # lazy: the package is optional

            # max_retries=0: the harness owns retry policy and the turn clock.
            self._client = openai.OpenAI(api_key=self._api_key, max_retries=0)
        return self._client

    def complete(
        self, packet: ContextPacket, level: str, max_output_tokens: int, timeout_s: float
    ) -> RawResult:
        client = self._get_client()
        # No temperature/top_p: not allowed when reasoning effort != none, and we do not
        # use them at 'none' either, for one code path.
        kwargs: dict[str, Any] = {
            "model": self.model,
            "instructions": packet.fixed_text,  # cacheable prefix
            "input": packet.turn_text,
            "reasoning": {"effort": level},
            "max_output_tokens": max_output_tokens,  # caps thinking + answer
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "vgc_preview" if packet.kind == "preview" else "vgc_advice",
                    "strict": True,
                    "schema": packet.schema or response_schema(packet.option_ids),
                }
            },
            "store": False,
            "timeout": timeout_s,
        }
        try:
            resp = client.responses.create(**kwargs)
        except Exception as exc:  # map by class name so we never import openai at module load
            name = type(exc).__name__
            if name == "RateLimitError":
                raise RateLimited(str(exc)) from exc
            if name == "APITimeoutError":
                raise LLMTimeout(str(exc)) from exc
            if name == "APIConnectionError":
                raise ConnectionFailed(str(exc)) from exc
            raise LLMError(f"{name}: {exc}") from exc
        return self._to_raw(resp, packet)

    @staticmethod
    def _to_raw(resp: Any, packet: ContextPacket) -> RawResult:
        usage = getattr(resp, "usage", None)
        details = getattr(usage, "input_tokens_details", None)
        refusal = None
        for item in getattr(resp, "output", None) or []:
            for part in getattr(item, "content", None) or []:
                if getattr(part, "type", "") == "refusal":
                    refusal = getattr(part, "refusal", "") or "refusal"
        text = getattr(resp, "output_text", None) or None
        return RawResult(
            text=text,
            status=str(getattr(resp, "status", "completed") or "completed"),
            request_id=packet.request_id,
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            cached_tokens=int(getattr(details, "cached_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            refusal=refusal,
            response_id=str(getattr(resp, "id", "") or ""),
        )
