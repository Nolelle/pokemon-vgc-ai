"""LLM move proposer: ask for up to 3 of our joint orders, hand them back to the search.

`propose_orders` never raises and never decides anything. It returns engine orders the LLM
named by option ID; `vgc.search.search_joint_orders` adds them to its candidate shortlist
and scores them like any other candidate, so the engine keeps the last word. Every failure
(skipped turn, bad answer, timeout, spend cap, missing key, missing facts module) returns
`[]`, which means "search exactly as if the LLM did not exist".

Wiring: `VgcPlayer.decide` passes `make_order_proposer(...)` to `search_joint_orders` only
when `PolicyConfig.llm_proposer_enabled` is set. The wait is sized from the guarded
decision's remaining clock (`vgc.clock.time_left`) minus `llm_safety_margin_s`; the cheap
fallback order was already computed before the decision worker started.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from typing import Any

from vgc.clock import time_left
from vgc.decision_trace import record_note
from vgc.llm.client import FakeLLMClient, LLMClient, OpenAIResponsesClient
from vgc.llm.config import LLMConfig
from vgc.llm.harness import advise
from vgc.llm.packet import new_request_id
from vgc.llm.spend import SpendMeter, reset_shared_meters, shared_meter
from vgc.llm.thinking import choose_level
from vgc.models import PolicyConfig

MAX_PROPOSALS = 3

_lock = threading.Lock()
_runtimes: dict[tuple, tuple[LLMClient, SpendMeter]] = {}
_plans: dict[str, str] = {}


def reset_proposer_state() -> None:
    """Forget the cached client, spend meter and team plans (tests, config changes)."""
    with _lock:
        _runtimes.clear()
        _plans.clear()
    reset_shared_meters()


def llm_config_for(config: PolicyConfig) -> LLMConfig:
    return LLMConfig(
        model=config.llm_model,
        max_proposals=MAX_PROPOSALS,
        spend_cap_usd=config.llm_budget_cap_usd,
    )


def _runtime(config: PolicyConfig) -> tuple[LLMClient, SpendMeter]:
    """One client per process per setting; one spend meter per spend file (see spend.py)."""
    fake = bool(config.llm_fake_scenario)
    # Fake runs use a private in-memory meter, so their cap is part of the identity; real
    # runs share the file's meter, which enforces the smallest cap any config asked for.
    key = (config.llm_model, config.llm_fake_scenario, config.llm_spend_file,
           config.llm_budget_cap_usd if fake or not config.llm_spend_file else None)
    with _lock:
        if key not in _runtimes:
            if fake:
                client: LLMClient = FakeLLMClient(config.llm_fake_scenario)
                meter = SpendMeter(None, config.llm_budget_cap_usd, config.llm_model)
            else:
                client = OpenAIResponsesClient(config.llm_model)
                meter = shared_meter(
                    config.llm_spend_file, config.llm_budget_cap_usd, config.llm_model
                )
            _runtimes[key] = (client, meter)
        elif not fake and config.llm_spend_file:
            # Lowers the shared meter's cap if this config asks for less.
            shared_meter(config.llm_spend_file, config.llm_budget_cap_usd, config.llm_model)
        return _runtimes[key]


def _team_plan(config: PolicyConfig) -> str:
    name = config.llm_team_plan
    if not name:
        return ""
    with _lock:
        if name in _plans:
            return _plans[name]
    try:
        from vgc.llm.facts import load_team_plan

        plan = str(load_team_plan(name) or "")
    except Exception:  # noqa: BLE001 - a missing plan just means no plan text
        plan = ""
    with _lock:
        _plans[name] = plan
    return plan


def _order_text(order: Any) -> str:
    text = getattr(order, "message", None)
    return str(text if text else order)


def _decision_kind(battle: Any) -> tuple[str, bool]:
    """('turn' | 'forced_switch', critical) -- reuses the clock guard's classifier."""
    try:
        from vgc.agent import _move_kind

        kind = _move_kind(battle)
    except Exception:  # noqa: BLE001
        return "turn", False
    return ("forced_switch", False) if kind == "forced_switch" else ("turn", kind == "critical")


def propose_orders(
    battle: Any,
    scored: Sequence[Any],
    config: PolicyConfig,
    budget_s: float,
    memory: Any = None,
) -> list[Any]:
    """Return at most 3 of `scored`'s orders the LLM proposed (best first), or []."""
    started = time.monotonic()
    note: dict[str, Any] = {"budget_s": round(float(budget_s), 2)}
    try:
        orders = _propose(battle, scored, config, budget_s, memory, note)
    except Exception as exc:  # noqa: BLE001 - the engine fallback must always win
        note["status"] = "error"
        note["error"] = f"{type(exc).__name__}: {exc}"
        orders = []
    note["n_orders"] = len(orders)
    note["elapsed_s"] = round(time.monotonic() - started, 3)
    record_note("llm_proposer", note)
    return orders[:MAX_PROPOSALS]


def _propose(battle, scored, config, budget_s, memory, note) -> list[Any]:
    if len(scored) < 2:
        note["status"] = "skipped_single_option"
        return []
    kind, critical = _decision_kind(battle)
    gap = float(scored[0].score) - float(scored[1].score)
    llm_cfg = llm_config_for(config)
    level = choose_level(kind, budget_s, gap, len(scored), critical=critical, config=llm_cfg)
    note.update(kind=kind, gap=round(gap, 2), level=level)
    if level is None:
        note["status"] = "skipped"
        return []

    from vgc.llm.facts import build_packet  # lazy: written separately, optional at import

    prep_started = time.monotonic()
    turn = getattr(battle, "turn", None)
    packet, options = build_packet(
        battle,
        scored,
        llm_config=llm_cfg,
        team_plan=_team_plan(config),
        memory=memory,
        policy_config=config,
        request_id=new_request_id(turn if isinstance(turn, int) else None),
    )
    # Packet construction can be slow; `advise` starts its own deadline, so re-derive what is
    # really left (our own wait budget AND the guarded decision's clock) right before the
    # paid call, and skip it rather than launch a request that outlives the decision.
    remaining = float(budget_s) - (time.monotonic() - prep_started)
    left = time_left()
    if left is not None:
        remaining = min(remaining, left - config.llm_safety_margin_s)
    note["remaining_s"] = round(remaining, 2)
    if remaining < llm_cfg.min_call_budget_s:
        note["status"] = "skipped_slow_prep"
        return []
    client, meter = _runtime(config)
    advice = advise(
        packet, client, level, remaining, meter, config.llm_log_path, config=llm_cfg
    )
    if advice is None:
        note["status"] = "no_advice"
        return []

    by_text = {_order_text(entry.order): entry.order for entry in scored}
    option_order = {option.id: option.order for option in options}
    orders: list[Any] = []
    ids: list[str] = []
    for proposal in advice.proposals:
        order = by_text.get(option_order.get(proposal.id, ""))
        if order is None or any(order is existing for existing in orders):
            continue
        orders.append(order)
        ids.append(proposal.id)
    note["status"] = "ok" if orders else "no_mappable_ids"
    note["ids"] = ids
    return orders


def make_order_proposer(
    battle: Any, config: PolicyConfig, memory: Any = None
) -> Callable[[list[Any]], list[Any]]:
    """Build the callback `search_joint_orders(order_proposer=...)` expects.

    The wait budget is whatever the guarded decision has left minus the safety margin, or
    `llm_offline_budget_s` when no timer is announced.
    """
    def run(scored: list[Any]) -> list[Any]:
        left = time_left()
        budget = config.llm_offline_budget_s if left is None else left - config.llm_safety_margin_s
        return propose_orders(battle, scored, config, max(0.0, budget), memory)

    return run
