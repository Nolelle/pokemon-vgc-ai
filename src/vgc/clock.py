"""Clock guard: read Showdown's battle timer and bound every decision by it.

Showdown's "VGC Timer" rule (see `server/room-battle.ts`) gives each player a ~420 s bank
for the whole game (plus 90 s of grace at the start), caps a single decision at 55 s (90 s
for team preview), and charges time in 5 s ticks. Before this module nothing in the bot
read that timer, so one slow search could burn the bank and forfeit the game.

Everything here is pure (no poke-env, no I/O) so it can be unit tested:

* `parse_timer_message` / `ClockState` -- what the server told us about our clock.
* `budget_seconds` -- how long the next decision may think.
* `run_with_deadline` -- run a decision function under that budget with a fallback.

An unknown clock (no timer message seen: the offline direct env, or the timer not yet on)
yields no budget and no enforcement, so offline behavior is unchanged.
"""

from __future__ import annotations

import contextvars
import math
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

T = TypeVar("T")

# Set (inside a guarded worker's context) once its deadline has passed: lets a late
# decide() skip side effects such as recording a choice that was never sent.
_cancel_var: contextvars.ContextVar[threading.Event | None] = contextvars.ContextVar(
    "vgc_clock_cancel", default=None
)


def cancelled() -> bool:
    """True when the current decision already missed its deadline and was discarded."""

    event = _cancel_var.get()
    return event is not None and event.is_set()


def bind_cancel(event: threading.Event) -> None:
    """Bind ``event`` as the cancel flag of the current context (worker side)."""

    _cancel_var.set(event)


# Showdown charges the timer in whole ticks (`TICK_TIME` in room-battle.ts).
TICK_S = 5.0

DECISION_KINDS = ("normal", "critical", "forced_switch", "preview")

# Real strings the server sends (room-battle.ts nextRequest / timer toggles):
#   |inactive|Time left: 55 sec this turn | 420 sec total
#   |inactive|Time left: 90 sec this turn | 420 sec total | 90 sec grace
#   |inactiveoff|Battle timer is now OFF.
_TIME_LEFT_RE = re.compile(
    r"^Time left:\s*(\d+)\s*sec this turn\s*\|\s*(\d+)\s*sec total(?:\s*\|\s*(\d+)\s*sec grace)?"
)


@dataclass(frozen=True)
class ClockState:
    """Our own clock as last announced by the server.

    ``bank_s`` is the total time left, excluding grace (the server's own "total");
    ``turn_cap_s`` is the most this single request may take; ``updated_at`` is the
    ``time.monotonic()`` instant the message arrived, so elapsed time can be subtracted.
    """

    bank_s: float
    turn_cap_s: float
    grace_s: float = 0.0
    updated_at: float = 0.0

    def bank_now(self, now: float | None = None) -> float:
        """Bank minus wall time since the announcement (never negative)."""

        if now is None:
            now = time.monotonic()
        return max(0.0, self.bank_s - max(0.0, now - self.updated_at))

    def turn_cap_now(self, now: float | None = None) -> float:
        if now is None:
            now = time.monotonic()
        return max(0.0, self.turn_cap_s - max(0.0, now - self.updated_at))


def parse_timer_message(
    message: list[str] | str, now: float | None = None
) -> ClockState | None | bool:
    """Parse one protocol line (split on ``|`` as poke-env does, or a raw string).

    Returns a `ClockState` for a ``Time left`` line, ``False`` for the timer being turned
    off (``|inactiveoff|``), and ``None`` for anything else (including the unrelated
    "<name> has N seconds left" chatter that goes to the whole room).
    """

    if isinstance(message, str):
        parts = message.split("|")
        message = parts[1:] if parts and parts[0] == "" else parts
        message = ["", *message]
    if len(message) < 2:
        return None
    kind = message[1]
    if kind == "inactiveoff":
        return False
    if kind != "inactive" or len(message) < 3:
        return None
    # The text itself contains "|" separators, so rejoin everything after the kind.
    text = "|".join(message[2:]).strip()
    match = _TIME_LEFT_RE.match(text)
    if match is None:
        return None
    return ClockState(
        bank_s=float(match.group(2)),
        turn_cap_s=float(match.group(1)),
        grace_s=float(match.group(3) or 0),
        updated_at=time.monotonic() if now is None else now,
    )


# VGC Timer rule (data/rulesets.ts): 420 s bank, 90 s grace, 55 s per turn, 90 s first turn.
STARTING_BANK_S = 420.0
STARTING_GRACE_S = 90.0
MAX_TURN_S = 55.0
MAX_FIRST_TURN_S = 90.0


class ClockTracker:
    """Our best estimate of the bank at the start of each decision, for one battle.

    Showdown sends ``|request|`` BEFORE ``|inactive|Time left...`` for the same turn, and
    poke-env runs the decision inside the request line's handler, so the "Time left"
    line for request K is only seen AFTER decision K. The estimate for decision J is
    therefore: the last announced bank (start of request K), minus the ticks our own
    decisions K..J-1 are assumed to have consumed (elapsed wall time rounded UP to whole
    ticks, pessimistic). Grace is spent before the bank, so it offsets the first debit.
    Each new announcement re-anchors the estimate, so errors do not accumulate.

    ``assume_full_bank`` seeds a fresh 420 s bank / 90 s grace before any message (for a
    player that started the timer itself); otherwise the clock stays unknown until the
    server announces one, and nothing is enforced.
    """

    def __init__(self, assume_full_bank: bool = False) -> None:
        self.decisions_started = 0
        self._debits: dict[int, float] = {}  # request group (first decision idx) -> ticks
        self._anchor: ClockState | None = None
        self._anchor_idx = 0
        self.timer_off = False
        # A "request group" is one server request: its first decision plus any retries
        # (invalid-choice / unavailable-choice), which keep the original start time.
        self._pending_arrival: float | None = None
        self._noted_any = False
        self._group_first: int | None = None
        self._group_arrival = 0.0
        self._group_timed = False  # True when the arrival time was seen, not assumed
        if assume_full_bank:
            self._anchor = ClockState(
                STARTING_BANK_S, MAX_FIRST_TURN_S, STARTING_GRACE_S, time.monotonic()
            )

    def note_request(self, now: float | None = None, update: bool = False) -> None:
        """A ``|request|`` line arrived. ``update`` requests do not restart the timer."""

        self._noted_any = True
        if not update:
            self._pending_arrival = time.monotonic() if now is None else now

    def observe(self, message: list[str] | str, now: float | None = None) -> None:
        parsed = parse_timer_message(message, now)
        if parsed is False:
            self._anchor = None
            self.timer_off = True
        elif isinstance(parsed, ClockState):
            self._anchor = parsed
            if self._pending_arrival is not None:
                # Request seen, decision not started yet (e.g. OTS wait): it is the next group.
                self._anchor_idx = self.decisions_started
            else:
                self._anchor_idx = self._group_first or 0
            self.timer_off = False

    def begin_decision(self, now: float | None = None) -> tuple[int, ClockState | None]:
        """Register a decision; return its index and the clock estimated at request start.

        The returned `ClockState` is stamped with the request's ARRIVAL time, so
        `bank_now`/`turn_cap_now` subtract everything already spent on this request (a
        delayed team preview, or a retry of the same request).
        """

        if now is None:
            now = time.monotonic()
        idx = self.decisions_started
        self.decisions_started += 1
        if self._pending_arrival is not None:
            self._group_first, self._group_arrival = idx, self._pending_arrival
            self._group_timed = True
            self._pending_arrival = None
        elif not self._noted_any or self._group_first is None:
            self._group_first, self._group_arrival, self._group_timed = idx, now, False
        group = self._group_first
        anchor = self._anchor
        if anchor is None:
            return idx, None
        debit = 0.0
        for g, spent in self._debits.items():
            if self._anchor_idx <= g < group:
                debit += max(0.0, spent - anchor.grace_s) if g == self._anchor_idx else spent
        if self._anchor_idx == group:
            cap = anchor.turn_cap_s
        else:
            cap = MAX_FIRST_TURN_S if group == 0 else MAX_TURN_S
        return idx, ClockState(max(0.0, anchor.bank_s - debit), cap, 0.0, self._group_arrival)

    def end_decision(self, idx: int, elapsed_s: float, now: float | None = None) -> None:
        """Record what this request has cost so far, rounded UP to whole ticks."""

        if self._group_timed:
            if now is None:
                now = time.monotonic()
            spent = now - self._group_arrival
        else:
            spent = elapsed_s
        group = self._group_first if self._group_first is not None else idx
        self._debits[group] = math.ceil(max(0.0, spent) / TICK_S - 1e-9) * TICK_S


@dataclass(frozen=True)
class Budget:
    """Outcome of `budget_seconds`: a time limit, or an instruction to skip deciding."""

    seconds: float | None  # None = no limit (unknown clock / guard off)
    fallback_only: bool = False
    bank_left_s: float | None = None
    kind: str = "normal"


def _floor_to_tick(seconds: float) -> float:
    return math.floor(seconds / TICK_S + 1e-9) * TICK_S


def budget_seconds(
    clock: ClockState | None,
    kind: str,
    config: Any,
    now: float | None = None,
) -> Budget:
    """How long the next decision may think.

    Assume the game is deliberately long (``clock_assumed_remaining_decisions`` more
    decisions, regardless of how many Pokemon are left -- Protect/switch loops mean
    material is not a bound), reserve one full tick per assumed decision, keep a safety
    margin, and split what is left evenly. The result is floored to whole ticks (the
    server rounds charges up to ticks), then capped by the per-kind ceiling and by the
    request's own cap minus the margin. A derived budget of one tick or less means the
    bank cannot afford thinking at all: ``fallback_only``.

    Worst case, each decision costs its budget plus one tick, so N decisions at this
    budget never exceed the bank (see tests/test_clock.py).
    """

    if clock is None or not getattr(config, "clock_guard_enabled", True):
        return Budget(seconds=None, kind=kind)
    caps = {
        "normal": config.clock_cap_normal_s,
        "critical": config.clock_cap_critical_s,
        "forced_switch": config.clock_cap_forced_switch_s,
        "preview": config.clock_cap_preview_s,
    }
    cap = caps.get(kind, config.clock_cap_normal_s)
    bank = clock.bank_now(now)
    remaining = max(1, int(config.clock_assumed_remaining_decisions))
    margin = config.clock_safety_margin_s
    derived = (bank - config.clock_reserve_per_decision_s * remaining - margin) / remaining
    derived = _floor_to_tick(derived)
    turn_room = clock.turn_cap_now(now) - margin
    if derived <= TICK_S or turn_room <= 0:
        return Budget(seconds=0.0, fallback_only=True, bank_left_s=bank, kind=kind)
    return Budget(seconds=min(derived, cap, turn_room), bank_left_s=bank, kind=kind)


class WorkerSlot:
    """Allows at most one guarded decision worker at a time per player.

    A worker that missed its deadline cannot be killed, and subclasses share unlocked
    simulator connections, so a new decision must not start while the old one is alive.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def try_acquire(self) -> bool:
        return self._lock.acquire(blocking=False)

    def release(self) -> None:
        self._lock.release()

    def busy(self) -> bool:
        return self._lock.locked()


@dataclass
class DeadlineResult:
    value: Any
    reason: str  # none | deadline | exception | fallback-only | previous-worker-busy
    elapsed_s: float
    error: BaseException | None = None


def run_with_deadline(
    decide: Callable[[], T],
    fallback: Callable[[], T],
    budget: Budget,
    *,
    clock: Callable[[], float] = time.monotonic,
    slot: WorkerSlot | None = None,
) -> DeadlineResult:
    """Run ``decide`` within ``budget``; otherwise return ``fallback``'s value.

    The fallback is computed FIRST (it must be cheap and legal), so it is already in hand
    when the deadline passes. With no limit (unknown clock) ``decide`` simply runs inline
    and a raised exception propagates to the caller's existing exception handling --
    behavior identical to before the guard existed.

    With a limit, ``decide`` runs on a daemon worker thread. Python cannot kill a thread,
    so on timeout the worker keeps running in the background; its result is discarded and
    can never be sent, because only this function's return value is ever used. (The late
    worker still burns CPU until it finishes and may touch battle memory; it is read-only
    with respect to the battle itself.) An exception inside ``decide`` is reported as
    ``reason="exception"`` with the fallback's value and the error attached.
    """

    start = clock()
    if slot is not None and slot.busy():
        # A previous worker (one that missed its deadline) is still running.
        return DeadlineResult(fallback(), "previous-worker-busy", clock() - start)
    if budget.seconds is None:
        return DeadlineResult(decide(), "none", clock() - start)
    safe = fallback()
    if budget.fallback_only:
        return DeadlineResult(safe, "fallback-only", clock() - start)
    if slot is not None and not slot.try_acquire():
        return DeadlineResult(safe, "previous-worker-busy", clock() - start)

    box: dict[str, Any] = {}
    done = threading.Event()

    def _worker() -> None:
        try:
            box["value"] = decide()
        except BaseException as exc:  # noqa: BLE001 - reported to the caller
            box["error"] = exc
        finally:
            if slot is not None:
                slot.release()
            done.set()

    thread = threading.Thread(target=_worker, name="vgc-decide", daemon=True)
    try:
        thread.start()
    except BaseException:
        if slot is not None:
            slot.release()
        raise
    remaining = max(0.0, budget.seconds - (clock() - start))
    if not done.wait(remaining):
        return DeadlineResult(safe, "deadline", clock() - start)
    if "error" in box:
        return DeadlineResult(safe, "exception", clock() - start, error=box["error"])
    return DeadlineResult(box["value"], "none", clock() - start)
