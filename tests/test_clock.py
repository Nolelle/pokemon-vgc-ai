"""Clock guard: budget formula, timer-message parsing, and the deadline wrapper.

The message strings are the real ones from Showdown's `server/room-battle.ts`
(`nextRequest`): `|inactive|Time left: N sec this turn | M sec total [| G sec grace]`.
"""

from __future__ import annotations

import math
import threading
import time
from types import SimpleNamespace

from vgc.agent import VgcPlayer
from vgc.clock import (
    TICK_S,
    Budget,
    ClockState,
    ClockTracker,
    budget_seconds,
    cancelled,
    parse_timer_message,
    run_with_deadline,
)
from vgc.models import PolicyConfig

CFG = PolicyConfig()


def _state(bank: float, cap: float = 55.0) -> ClockState:
    return ClockState(bank_s=bank, turn_cap_s=cap, updated_at=0.0)


# --- parsing ---------------------------------------------------------------------------


def test_parse_time_left_lines():
    # poke-env splits on "|", so the three clauses arrive as separate list items.
    plain = parse_timer_message(["", "inactive", "Time left: 55 sec this turn ", " 420 sec total"])
    assert (plain.turn_cap_s, plain.bank_s, plain.grace_s) == (55.0, 420.0, 0.0)
    first = parse_timer_message(
        "|inactive|Time left: 90 sec this turn | 420 sec total | 90 sec grace"
    )
    assert (first.turn_cap_s, first.bank_s, first.grace_s) == (90.0, 420.0, 90.0)


def test_parse_ignores_chatter_and_detects_timer_off():
    assert parse_timer_message("|inactive|Edmund has 25 seconds left this turn.") is None
    assert parse_timer_message("|inactive|Battle timer is ON: inactive players will lose.") is None
    assert parse_timer_message("|move|p1a: X|Protect|p1a: X") is None
    assert parse_timer_message("|inactiveoff|Battle timer is now OFF.") is False


# --- budget ----------------------------------------------------------------------------


def test_unknown_clock_or_disabled_guard_has_no_limit():
    assert budget_seconds(None, "normal", CFG).seconds is None
    off = PolicyConfig(clock_guard_enabled=False)
    assert budget_seconds(_state(420), "normal", off).seconds is None


def test_budget_formula_and_caps():
    # (420 - 5*25 - 5) / 25 = 11.6 -> floored to the 10 s tick, under the 12 s cap.
    assert budget_seconds(_state(420), "normal", CFG, now=0.0).seconds == 10.0
    # Forced switches are capped at 4 s even though the bank allows 10 s.
    assert budget_seconds(_state(420), "forced_switch", CFG, now=0.0).seconds == 4.0
    # A rich bank hits the per-kind cap; critical may exceed normal.
    assert budget_seconds(_state(900), "normal", CFG, now=0.0).seconds == 12.0
    assert budget_seconds(_state(900), "critical", CFG, now=0.0).seconds == 20.0
    # The request's own cap (minus margin) also limits it.
    assert budget_seconds(_state(900, cap=9), "normal", CFG, now=0.0).seconds == 4.0


def test_poor_bank_means_fallback_only():
    # (140 - 125 - 5)/25 = 0.4 -> floored to 0 s: cannot afford to think.
    poor = budget_seconds(_state(140), "normal", CFG, now=0.0)
    assert poor.fallback_only and poor.seconds == 0.0
    assert budget_seconds(_state(420, cap=5), "normal", CFG, now=0.0).fallback_only


def test_long_game_never_exhausts_the_bank_under_pessimistic_charging():
    # Worst case: every decision uses its whole budget AND is charged one extra tick;
    # a fallback-only decision still costs one tick.
    bank = 420.0
    kinds = ["preview"] + ["normal", "critical", "forced_switch"] * 15  # 46 decisions
    for kind in kinds:
        budget = budget_seconds(_state(bank), kind, CFG, now=0.0)
        bank -= TICK_S if budget.fallback_only else budget.seconds + TICK_S
        assert bank > 0, f"bank exhausted at {kind}"


# --- tracker: the "Time left" line arrives AFTER its own decision ------------------------


def test_tracker_debits_previous_decision_and_uses_grace():
    tracker = ClockTracker(assume_full_bank=True)
    idx, state = tracker.begin_decision(now=0.0)
    assert (state.bank_s, state.turn_cap_s) == (420.0, 90.0)
    tracker.end_decision(idx, 12.0)  # preview: ticks drain grace first, bank untouched
    tracker.observe("|inactive|Time left: 90 sec this turn | 420 sec total | 90 sec grace", now=1.0)
    _, state = tracker.begin_decision(now=2.0)
    assert state.bank_s == 420.0 and state.turn_cap_s == 55.0  # 12 s charge < 90 s grace
    # Unknown clock stays unknown until the server announces one.
    assert ClockTracker().begin_decision()[1] is None


def test_delayed_preview_and_retries_do_not_restart_the_request_allowance():
    tracker = ClockTracker(assume_full_bank=True)
    tracker.note_request(now=100.0)  # |request| arrives; OTS handling delays the decision
    tracker.observe(
        "|inactive|Time left: 90 sec this turn | 420 sec total | 90 sec grace", now=100.1
    )
    idx, state = tracker.begin_decision(now=185.0)  # 85 s later
    assert state.turn_cap_now(185.0) == 5.0  # not a fresh 90 s
    budget = budget_seconds(state, "preview", CFG, now=185.0)
    assert budget.fallback_only  # 5 s left minus the 5 s margin: no time to think
    tracker.end_decision(idx, 0.01, now=185.0)
    # An invalid-choice retry of the SAME request keeps the original arrival time.
    _, retry = tracker.begin_decision(now=200.0)
    assert retry.updated_at == 100.0 and retry.turn_cap_now(200.0) == 0.0
    # A fresh (non-update) request starts a new allowance; an update resend does not.
    tracker.note_request(now=300.0, update=True)
    _, still = tracker.begin_decision(now=301.0)
    assert still.updated_at == 100.0
    tracker.note_request(now=400.0)
    tracker.observe("|inactive|Time left: 55 sec this turn | 300 sec total", now=400.1)
    _, fresh = tracker.begin_decision(now=401.0)
    assert fresh.updated_at == 400.0 and fresh.bank_s == 300.0
    assert fresh.turn_cap_s == 55.0


# --- deadline wrapper ------------------------------------------------------------------


def test_fast_decide_returns_its_own_result():
    out = run_with_deadline(lambda: "decided", lambda: "fallback", Budget(seconds=1.0))
    assert (out.value, out.reason) == ("decided", "none")


def test_slow_decide_returns_fallback_and_late_result_is_discarded():
    gate = threading.Event()
    finished = []

    def slow():
        gate.wait(5)
        finished.append("late")
        return "late-result"

    started = time.monotonic()
    out = run_with_deadline(slow, lambda: "fallback", Budget(seconds=0.2))
    assert (out.value, out.reason) == ("fallback", "deadline")
    assert time.monotonic() - started < 1.0
    gate.set()  # let the worker finish; nothing can overwrite the returned value
    time.sleep(0.1)
    assert finished == ["late"] and out.value == "fallback"


def test_exception_and_fallback_only_and_no_limit_paths():
    def boom():
        raise RuntimeError("x")

    err = run_with_deadline(boom, lambda: "fallback", Budget(seconds=1.0))
    assert (err.value, err.reason) == ("fallback", "exception")
    only = run_with_deadline(lambda: "decided", lambda: "fb", Budget(0.0, fallback_only=True))
    assert (only.value, only.reason) == ("fb", "fallback-only")
    assert run_with_deadline(lambda: 7, lambda: 0, Budget(seconds=None)).value == 7


# --- VgcPlayer integration (no server) ---------------------------------------------------


def _player(**cfg) -> VgcPlayer:
    player = VgcPlayer.__new__(VgcPlayer)
    player.config = PolicyConfig(**cfg)
    player.fallback_count = 0
    player.clock_log = []
    player._assume_timer_on = True
    player._clock_trackers = {}
    return player


def test_player_guard_sends_fallback_on_timeout_and_decide_when_fast():
    battle = SimpleNamespace(battle_tag="b1", turn=3)
    player = _player(clock_cap_normal_s=0.2)
    seen = []

    def slow():
        time.sleep(0.6)
        seen.append(cancelled())
        return "slow"

    assert player._guarded(battle, "normal", slow, lambda: "fb") == "fb"
    entry = player.clock_log[-1]
    assert entry["fallback_reason"] == "deadline" and entry["decision_kind"] == "normal"
    assert entry["budget_s"] == 0.2 and player.fallback_count == 1
    time.sleep(0.6)
    assert seen == [True]  # the late worker knows it was discarded

    assert player._guarded(battle, "normal", lambda: "fast", lambda: "fb") == "fast"
    assert player.clock_log[-1]["fallback_reason"] == "none"


def test_player_guard_is_inert_without_a_timer():
    player = _player()
    player._assume_timer_on = False
    battle = SimpleNamespace(battle_tag="offline", turn=1)
    assert player._guarded(battle, "normal", lambda: "x", lambda: "fb") == "x"
    entry = player.clock_log[-1]
    assert entry["budget_s"] is None and entry["fallback_reason"] == "none"
    assert math.isclose(entry["elapsed_ms"], 0.0, abs_tol=50.0)


def test_busy_worker_blocks_new_worker_and_late_record_choice_is_ignored():
    from vgc.battle_memory import BattleMemory

    battle = SimpleNamespace(battle_tag="b2", turn=4)
    player = _player(clock_cap_normal_s=0.2)
    memory = BattleMemory(battle_tag="b2")
    release = threading.Event()
    started: list[str] = []

    def slow():
        started.append("first")
        release.wait(5)
        memory.record_choice(4, "late-unsent-order")  # must be ignored once cancelled
        return "late"

    assert player._guarded(battle, "normal", slow, lambda: "fb") == "fb"
    assert player.clock_log[-1]["fallback_reason"] == "deadline"

    def second():
        started.append("second")
        return "second"

    # The first worker is still sleeping: no new worker, straight to the fallback.
    assert player._guarded(battle, "normal", second, lambda: "fb2") == "fb2"
    assert player.clock_log[-1]["fallback_reason"] == "previous-worker-busy"
    assert started == ["first"]
    release.set()
    time.sleep(0.2)
    assert memory.our_orders == []
    # Once the old worker has finished, decisions run again.
    assert player._guarded(battle, "normal", second, lambda: "fb2") == "second"
