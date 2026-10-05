"""Run one advisory LLM call under a hard deadline and return validated advice or None.

`advise()` never raises. The caller always has an engine fallback, so every failure mode
(bad JSON, unknown/stale IDs, refusal, timeout, rate limit, spend cap) just returns None
after being logged. One JSONL `CallRecord` is appended per call.

Deadline: the call runs in a daemon worker thread and we stop waiting at the deadline even
if the client ignores its own timeout. An answer that arrives later is recorded as
`late_ignored` and never used. A request id per turn guards against a late answer for an
old turn being applied to a newer one.
"""

from __future__ import annotations

import dataclasses
import json
import re
import threading
import time
from collections.abc import Callable, Collection, Iterable
from pathlib import Path

from vgc.llm.client import (
    ConnectionFailed,
    LLMClient,
    LLMError,
    LLMTimeout,
    RateLimited,
    response_schema,
)
from vgc.llm.config import LLMConfig
from vgc.llm.spend import SpendMeter, cost_usd, max_input_tokens
from vgc.llm.types import Advice, CallRecord, ContextPacket, Proposal, RawResult

_latest_lock = threading.Lock()
_latest_request_id: str | None = None

# Statuses worth one more attempt (transport / format trouble, not model refusals).
_RETRYABLE = {"malformed_json", "empty", "rate_limited", "connection_error"}


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


_vocab_cache: frozenset[str] | None = None


def default_vocab() -> frozenset[str]:
    """Normalized species/move/item names (len >= 5) from data/champions; empty if absent."""
    global _vocab_cache
    if _vocab_cache is None:
        names: set[str] = set()
        try:
            from vgc import data

            for table in (data.load_species(), data.load_moves(), data.load_items()):
                for entry in table.values():
                    n = _normalize(str(entry.get("name", "")))
                    if len(n) >= 5:
                        names.add(n)
        except Exception:
            names = set()
        _vocab_cache = frozenset(names)
    return _vocab_cache


def ungrounded_names(
    reasoning: Iterable[str], packet_text: str, vocab: Collection[str] | None = None
) -> list[str]:
    """Names of Pokemon/moves/items in the reasoning that never appear in the packet.

    Looks at 1-3 word windows of the reasoning. Heuristic and logged only: it can false-flag
    common-word move names, so it never blocks advice.
    """
    names = default_vocab() if vocab is None else vocab
    packet_norm = _normalize(packet_text)
    flagged: list[str] = []
    for text in reasoning:
        words = re.findall(r"[A-Za-z0-9']+", text)
        for n in (1, 2, 3):
            for i in range(len(words) - n + 1):
                cand = _normalize("".join(words[i : i + n]))
                if cand in names and cand not in packet_norm and cand not in flagged:
                    flagged.append(cand)
    return flagged


def _input_bound(packet: ContextPacket) -> int:
    """Upper bound on input tokens for everything the real client sends."""
    schema = json.dumps(packet.schema or response_schema(packet.option_ids))
    return max_input_tokens(packet.fixed_text, packet.turn_text, schema)


def _is_current(request_id: str) -> bool:
    with _latest_lock:
        return _latest_request_id in (None, request_id)


def _append(log_path: str | Path | None, rec: CallRecord) -> None:
    if log_path is None:
        return
    try:
        rec.ts = time.time()
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(dataclasses.asdict(rec)) + "\n")
    except Exception:
        pass  # logging must never break a turn


def _parse(
    raw: RawResult, packet: ContextPacket, cfg: LLMConfig
) -> tuple[Advice | None, str, int]:
    """Return (advice, status, n_invalid_ids)."""
    if raw.refusal:
        return None, "refusal", 0
    if raw.status == "incomplete":
        return None, "incomplete", 0
    if raw.status not in ("completed", "ok"):
        return None, f"status_{raw.status}", 0
    if not raw.text or not raw.text.strip():
        return None, "empty", 0
    if packet.kind == "preview":
        return _parse_structured(raw, packet)
    try:
        data = json.loads(raw.text)
        plan = data["plan"]
        items = data["proposals"]
        if not isinstance(plan, str) or not isinstance(items, list):
            raise TypeError("bad shape")
    except (ValueError, KeyError, TypeError):
        return None, "malformed_json", 0
    current = set(packet.option_ids)
    seen: set[str] = set()
    kept: list[Proposal] = []
    invalid = 0
    for item in items:
        pid = item.get("id") if isinstance(item, dict) else None
        if not isinstance(pid, str) or pid not in current:
            invalid += 1  # unknown, or stale from an earlier turn's option list
            continue
        if pid in seen:
            continue
        seen.add(pid)
        why = item.get("why", "")
        kept.append(Proposal(pid, why if isinstance(why, str) else ""))
        if len(kept) >= cfg.max_proposals:
            break
    if not kept:
        return None, "no_valid_ids", invalid
    return Advice(plan=plan, proposals=tuple(kept)), "ok", invalid


def _parse_structured(raw: RawResult, packet: ContextPacket) -> tuple[Advice | None, str, int]:
    """Non-move answers (team preview): valid JSON object, then the packet's own validator."""
    try:
        data = json.loads(raw.text or "")
        if not isinstance(data, dict) or not isinstance(data.get("plan"), str):
            raise TypeError("bad shape")
    except (ValueError, TypeError):
        return None, "malformed_json", 0
    if packet.validate is not None:
        error = packet.validate(data)
        if error:
            return None, "invalid_answer", 1
    return Advice(plan=data["plan"], proposals=(), data=data), "ok", 0


class _Call:
    """One client call on a daemon thread, so we can stop waiting at the deadline."""

    def __init__(self, client: LLMClient, packet: ContextPacket, level: str, max_out: int,
                 timeout_s: float, clock: Callable[[], float] = time.monotonic) -> None:
        self.clock = clock
        self.completed_at = float("inf")
        self.raw: RawResult | None = None
        self.exc: BaseException | None = None
        self.done = threading.Event()
        self.abandoned = False
        self.on_late = None  # set by the harness; called if we finish after being abandoned
        self._lock = threading.Lock()
        self._args = (client, packet, level, max_out, timeout_s)
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        client, packet, level, max_out, timeout_s = self._args
        try:
            self.raw = client.complete(packet, level, max_out, timeout_s)
        except BaseException as exc:  # noqa: BLE001 - reported, never raised to the caller
            self.exc = exc
        with self._lock:
            self.completed_at = self.clock()  # when the worker finished, not when we looked
            self.done.set()
            late = self.abandoned
        if late and self.on_late is not None:
            self.on_late(self)

    def abandon(self) -> bool:
        """Mark as given up on. Returns False if it finished just in time."""
        with self._lock:
            if self.done.is_set():
                return False
            self.abandoned = True
            return True


def advise(
    packet: ContextPacket,
    client: LLMClient,
    level: str,
    budget_s: float,
    spend_meter: SpendMeter | None = None,
    log_path: str | Path | None = None,
    *,
    config: LLMConfig | None = None,
    vocab: Collection[str] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> Advice | None:
    """Ask the LLM for up to 3 option IDs within `budget_s` seconds, or return None."""
    try:
        return _advise(packet, client, level, budget_s, spend_meter, log_path,
                       config or LLMConfig(), vocab, clock)
    except Exception as exc:  # last-resort guard: the engine fallback must always win
        _append(log_path, CallRecord(packet.request_id, (config or LLMConfig()).model, level,
                                     packet.prompt_sha256, status="harness_error",
                                     error=f"{type(exc).__name__}: {exc}", turn=packet.turn,
                                     kind=packet.kind))
        return None


def _advise(packet, client, level, budget_s, meter, log_path, cfg, vocab, clock) -> Advice | None:
    global _latest_request_id
    start = clock()
    deadline = start + budget_s
    with _latest_lock:
        _latest_request_id = packet.request_id
    max_out = cfg.max_output_tokens_for(level)

    for attempt in (0, 1):
        remaining = deadline - clock()
        if attempt == 1 and remaining < cfg.retry_min_remaining_s:
            return None
        if remaining <= cfg.deadline_margin_s:
            return None

        def record(status: str, raw: RawResult | None = None, error: str = "",
                   invalid: int = 0, flagged: list[str] | None = None,
                   cost: float = 0.0, t0: float = 0.0) -> None:
            _append(log_path, CallRecord(
                request_id=packet.request_id, model=cfg.model, level=level,
                prompt_sha256=packet.prompt_sha256,
                input_tokens=raw.input_tokens if raw else 0,
                cached_tokens=raw.cached_tokens if raw else 0,
                output_tokens=raw.output_tokens if raw else 0,
                latency_s=round(clock() - t0, 4), cost_usd=cost, status=status,
                error=error, turn=packet.turn, attempt=attempt, invalid_ids=invalid,
                ungrounded_names=flagged or [], kind=packet.kind,
            ))

        t0 = clock()
        res = None
        if meter is not None:
            res = meter.reserve(_input_bound(packet), max_out)
            if res is None:
                record("spend_cap", error="local spend cap would be exceeded", t0=t0)
                return None

        call = _Call(client, packet, level, max_out,
                     max(0.05, remaining - cfg.deadline_margin_s), clock)

        def late(c: _Call, _t0=t0, _attempt=attempt) -> None:
            _append(log_path, CallRecord(
                request_id=packet.request_id, model=cfg.model, level=level,
                prompt_sha256=packet.prompt_sha256,
                input_tokens=c.raw.input_tokens if c.raw else 0,
                output_tokens=c.raw.output_tokens if c.raw else 0,
                latency_s=round(clock() - _t0, 4), status="late_ignored",
                error=type(c.exc).__name__ if c.exc else "", turn=packet.turn,
                attempt=_attempt, kind=packet.kind))

        call.on_late = late
        call.done.wait(timeout=max(0.0, deadline - clock()))
        if call.abandon():  # False means it finished just in time; use its result
            # Billing unknown: charge the worst case rather than under-count.
            if meter is not None and res is not None:
                meter.settle(res)
            record("timeout", error="deadline reached", t0=t0)
            return None

        raw, exc = call.raw, call.exc
        if raw is not None and call.completed_at > deadline:
            # Finished after the deadline but before we looked (e.g. caller descheduled):
            # too late to use. Book the real cost, never the answer.
            cost = cost_usd(cfg.model, raw.input_tokens, raw.cached_tokens, raw.output_tokens,
                            raw.cache_write_tokens)
            if meter is not None and res is not None:
                meter.settle(res, cost)
            record("late_ignored", raw, error="completed after deadline", cost=cost, t0=t0)
            return None
        if exc is not None:
            status = ("rate_limited" if isinstance(exc, RateLimited)
                      else "connection_error" if isinstance(exc, ConnectionFailed)
                      else "timeout" if isinstance(exc, LLMTimeout)
                      else "client_error")
            if meter is not None and res is not None:
                # Timeouts may still be billed; other transport errors were never accepted.
                meter.settle(res) if status == "timeout" else meter.cancel(res)
            record(status, error=f"{type(exc).__name__}: {exc}"
                   if isinstance(exc, LLMError) else type(exc).__name__, t0=t0)
            if status in _RETRYABLE:
                continue
            return None

        assert raw is not None
        cost = cost_usd(cfg.model, raw.input_tokens, raw.cached_tokens, raw.output_tokens,
                        raw.cache_write_tokens)
        if meter is not None and res is not None:
            meter.settle(res, cost)  # incomplete / refused calls are still billed
        if raw.request_id and raw.request_id != packet.request_id or not _is_current(
            packet.request_id
        ):
            record("stale_request", raw, cost=cost, t0=t0)
            return None
        advice, status, invalid = _parse(raw, packet, cfg)
        flagged: list[str] = []
        if advice is not None:
            texts = [advice.plan, *(p.why for p in advice.proposals)]
            if advice.data and isinstance(advice.data.get("why"), str):
                texts.append(advice.data["why"])
            flagged = ungrounded_names(texts, packet.full_text, vocab)
        record(status, raw, invalid=invalid, flagged=flagged, cost=cost, t0=t0)
        if advice is not None or status not in _RETRYABLE:
            return advice
    return None
