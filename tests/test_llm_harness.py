"""LLM harness: every failure mode ends in valid advice with current IDs, or None."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

from vgc.llm.client import SCENARIOS, FakeLLMClient, OpenAIResponsesClient, response_schema
from vgc.llm.config import LLMConfig
from vgc.llm.harness import advise, ungrounded_names
from vgc.llm.packet import build_options, build_packet
from vgc.llm.spend import SpendMeter
from vgc.llm.thinking import choose_level

CFG = LLMConfig(deadline_margin_s=0.05)
BUDGET_S = 0.6  # tiny so stalled/late scenarios finish fast; retries need >= 6 s so none run


def _packet(n: int = 12):
    orders = [
        SimpleNamespace(order=f"/choose move m{i} 1, move m{i} 2", score=100.0 - i)
        for i in range(n)
    ]
    orders[5].order = "/choose switch 3, move protect"
    options = build_options(orders, max_options=30)
    return build_packet({"rules": "VGC"}, {"board": ["Charizard 100%"]}, options, turn=4)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_every_scenario_gives_current_ids_or_none(scenario, tmp_path):
    packet = _packet()
    log = tmp_path / "calls.jsonl"
    client = FakeLLMClient(scenario)
    start = time.monotonic()
    advice = advise(packet, client, "none", BUDGET_S, log_path=log, config=CFG)
    assert time.monotonic() - start < BUDGET_S + 0.2
    if advice is not None:
        assert advice.proposals and len(advice.proposals) <= 3
        ids = [p.id for p in advice.proposals]
        assert set(ids) <= set(packet.option_ids) and len(set(ids)) == len(ids)
    else:
        assert scenario not in {"valid", "slow", "duplicate_ids", "plausible_false_reasoning"}
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert rows and rows[0]["request_id"] == packet.request_id
    assert rows[0]["status"] in {"ok"} or advice is None
    if scenario == "plausible_false_reasoning":
        assert rows[0]["ungrounded_names"]  # flagged and logged, but advice still returned
        assert advice is not None


def test_stale_id_dropped_valid_one_kept():
    packet = _packet()
    advice = advise(packet, FakeLLMClient("stale_id"), "none", BUDGET_S, config=CFG)
    assert advice is not None and [p.id for p in advice.proposals] == [packet.option_ids[0]]


def test_stalled_call_cannot_block_past_budget(tmp_path):
    log = tmp_path / "c.jsonl"
    start = time.monotonic()
    out = advise(_packet(), FakeLLMClient("late_after_cancel"), "none", 0.3, log_path=log,
                 config=CFG)
    assert out is None and time.monotonic() - start < 0.5
    time.sleep(0.6)  # late answer lands after we gave up and is only logged
    statuses = [json.loads(x)["status"] for x in log.read_text().splitlines()]
    assert statuses == ["timeout", "late_ignored"]


def test_options_include_switch_and_protect_topups():
    orders = [SimpleNamespace(order=f"/choose move a{i} 1, move b{i} 1", score=50 - i)
              for i in range(40)]
    orders[38].order = "/choose switch 3, move c 1"
    orders[39].order = "/choose move protect, move protect"
    opts = build_options(orders, max_options=10)
    assert len(opts) == 10 and opts[0].id == "P01"
    assert {o.kind for o in opts} >= {"switch", "protect"}


def test_spend_meter_refuses_past_cap_and_persists(tmp_path):
    path = tmp_path / "spend.json"
    meter = SpendMeter(path, cap_usd=0.01)
    res = meter.reserve(10_000, 4_000)  # worst case ~ $0.003
    assert res is not None
    meter.settle(res, 0.009)
    assert meter.reserve(10_000, 4_000) is None  # 0.009 + 0.003 > 0.01
    assert SpendMeter(path, cap_usd=0.01).spent_usd == pytest.approx(0.009)  # survives restart
    path.write_text("garbage")
    assert SpendMeter(path, cap_usd=0.01).reserve(1, 1) is None  # corrupt file fails closed


def test_advise_refuses_when_cap_would_break(tmp_path):
    client = FakeLLMClient("valid")
    meter = SpendMeter(None, cap_usd=0.0)
    log = tmp_path / "c.jsonl"
    assert advise(_packet(), client, "none", BUDGET_S, meter, log, config=CFG) is None
    assert client.calls == [] and json.loads(log.read_text())["status"] == "spend_cap"


def test_spend_is_booked_from_usage(tmp_path):
    meter = SpendMeter(None, cap_usd=1.0)
    assert advise(_packet(), FakeLLMClient("valid"), "none", BUDGET_S, meter, config=CFG)
    assert 0 < meter.spent_usd < 0.001 and meter.calls == 1


@pytest.mark.parametrize(
    "kind,budget,gap,n,critical,expected",
    [
        ("turn", 30, 5, 1, False, None),  # single legal option
        ("forced_switch", 30, 5, 4, False, None),
        ("turn", 5.9, 5, 20, False, None),  # no time
        ("turn", 30, 30.0, 20, False, None),  # engine clearly right
        ("turn", 30, 15, 20, False, "none"),  # default
        ("turn", 9, 3, 20, False, "none"),  # close but too little clock for 'low'
        ("turn", 12, 3, 20, False, "low"),
        ("preview", 25, None, 15, False, "medium"),
        ("turn", 25, 3, 20, True, "medium"),
        ("preview", 12, None, 15, False, "none"),
    ],
)
def test_choose_level(kind, budget, gap, n, critical, expected):
    assert choose_level(kind, budget, gap, n, critical=critical) == expected


def test_choose_level_never_high_live():
    cfg = LLMConfig(max_live_level="low")
    assert choose_level("preview", 60, None, 15, config=cfg) == "low"


def test_real_client_request_shape_and_error_mapping():
    captured = {}

    class Resp:
        status, id, output, output_text = "completed", "r1", [], '{"plan":"p","proposals":[]}'
        usage = SimpleNamespace(
            input_tokens=100, output_tokens=7, input_tokens_details=SimpleNamespace(cached_tokens=40)
        )

    class Responses:
        def create(self, **kw):
            captured.update(kw)
            return Resp()

    client = OpenAIResponsesClient()
    client._client = SimpleNamespace(responses=Responses())
    packet = _packet()
    raw = client.complete(packet, "low", 900, 5.0)
    assert captured["store"] is False and captured["reasoning"] == {"effort": "low"}
    assert "temperature" not in captured and captured["max_output_tokens"] == 900
    fmt = captured["text"]["format"]
    assert fmt["strict"] and fmt["schema"] == response_schema(packet.option_ids)
    assert (raw.input_tokens, raw.cached_tokens, raw.output_tokens) == (100, 40, 7)


def test_grounding_flags_unknown_names_only():
    vocab = {"garchomp", "closecombat", "charizard"}
    flagged = ungrounded_names(["Garchomp uses Close Combat, Charizard is fine"],
                               "Charizard 100%", vocab)
    assert flagged == ["garchomp", "closecombat"]


def test_maximal_usage_never_pushes_settled_total_past_cap(tmp_path):
    """Reservation bounds input by UTF-8 bytes, so even worst-case real usage fits the cap."""
    from vgc.llm.harness import _input_bound
    from vgc.llm.spend import cost_usd

    packet = _packet()
    bound = _input_bound(packet)
    assert bound >= len(packet.full_text.encode("utf-8"))
    max_out = CFG.max_output_tokens_for("none")
    worst = SpendMeter(None, cap_usd=1.0).worst_case_usd(bound, max_out)
    meter = SpendMeter(tmp_path / "s.json", cap_usd=worst * 2.5)
    settled_ok = 0
    while (res := meter.reserve(bound, max_out)) is not None:
        # maximal usage: one token per byte, all billed as cache writes, output at the cap
        meter.settle(res, cost_usd("gpt-6-luna", bound, 0, max_out, cache_write_tokens=bound))
        settled_ok += 1
        assert meter.spent_usd <= meter.cap_usd
    assert settled_ok == 2


def test_crash_after_reserve_counts_reservation_as_spent(tmp_path):
    path = tmp_path / "spend.json"
    meter = SpendMeter(path, cap_usd=0.01)
    res = meter.reserve(10_000, 4_000)
    assert res is not None  # ...process dies here, before settle()
    recovered = SpendMeter(path, cap_usd=0.01)
    assert recovered.spent_usd == pytest.approx(res.amount_usd)
    assert recovered.reserve(10_000, 4_000) is not None  # 0.003 + 0.003 < 0.01
    assert recovered.reserve(10_000, 8_000) is None  # remaining budget reflects the hold
    assert SpendMeter(path, cap_usd=0.01).spent_usd >= res.amount_usd  # still persisted


def test_result_finished_after_deadline_is_not_used(tmp_path):
    """Worker finishes past the deadline before the waiting caller looks: still rejected."""
    now = [0.0]
    inner = FakeLLMClient("valid")

    class Slowish:
        def complete(self, packet, level, max_output_tokens, timeout_s):
            out = inner.complete(packet, level, max_output_tokens, timeout_s)
            now[0] += 10.0  # completion time lands past the 0.6 s budget
            return out

    log = tmp_path / "c.jsonl"
    meter = SpendMeter(None, cap_usd=1.0)
    out = advise(_packet(), Slowish(), "none", BUDGET_S, meter, log, config=CFG,
                 clock=lambda: now[0])
    assert out is None
    assert json.loads(log.read_text())["status"] == "late_ignored"
    assert meter.calls == 1 and meter.spent_usd > 0
