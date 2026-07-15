"""Lightweight per-decision trace recorder for diagnostics.

Minimal port of pokemon-tcg-ai's `src/decision_trace.py` ContextVar pattern: a cheap,
additive side-channel. `vgc.agent.VgcPlayer.choose_move` opens one `DecisionTrace` per
top-level decision via a `ContextVar`, and future evaluator code (Phase 2) can record
scalar facts onto it as a side effect of work it already does, read back afterwards via
`get_last_trace()`.

When no trace is active (e.g. tests calling evaluator functions directly, without going
through `choose_move`), every recording call below is a single `is None` check and
returns immediately -- no behavior or measurable cost change for any existing caller.

Tracing is only enabled when the `VGC_TRACE` environment variable is set (to any
non-empty value), so it never fires by default in normal battles or CI.
"""

from __future__ import annotations

import os
from contextvars import ContextVar, Token
from dataclasses import dataclass, field

TRACE_ENV_VAR = "VGC_TRACE"


def trace_enabled() -> bool:
    """Read the flag dynamically so session runners can enable tracing at runtime."""

    return bool(os.environ.get(TRACE_ENV_VAR))


@dataclass
class DecisionTrace:
    """Facts recorded during a single top-level `choose_move()` call."""

    turn: int | None = None
    chosen_order: str | None = None
    fallback_used: bool = False
    fallback_reason: str | None = None
    notes: dict[str, object] = field(default_factory=dict)


_trace_var: ContextVar[DecisionTrace | None] = ContextVar("vgc_decision_trace", default=None)
_LAST_TRACE: DecisionTrace | None = None


def start_trace() -> Token | None:
    """Begin a fresh trace for one `choose_move()` call. Always pair with `finish_trace`.

    No-op (returns None) when `VGC_TRACE` is unset, so callers still must handle a None
    token -- `finish_trace(None)` is a safe no-op too.
    """
    if not trace_enabled():
        return None
    return _trace_var.set(DecisionTrace())


def current_trace() -> DecisionTrace | None:
    """Return the active trace, or None if no trace is currently open (or disabled)."""
    return _trace_var.get()


def finish_trace(token: Token | None) -> DecisionTrace | None:
    """End the trace started by `start_trace`, stashing it for `get_last_trace()`."""
    global _LAST_TRACE
    if token is None:
        return None
    trace = _trace_var.get()
    _trace_var.reset(token)
    _LAST_TRACE = trace
    return trace


def get_last_trace() -> DecisionTrace | None:
    """Return the most recently finished trace, or None if tracing is disabled/unused."""
    return _LAST_TRACE


def record_note(key: str, value: object) -> None:
    trace = _trace_var.get()
    if trace is None:
        return
    trace.notes[key] = value


def record_fallback(reason: str) -> None:
    trace = _trace_var.get()
    if trace is None:
        return
    trace.fallback_used = True
    trace.fallback_reason = reason
