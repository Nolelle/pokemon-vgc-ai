"""LLM proposer wiring: proposals only ADD candidates to the search; every failure is a no-op.

`vgc.llm.facts` is written separately, so a minimal stand-in is injected here.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import types
from types import SimpleNamespace

import pytest

import vgc.agent as agent_module
import vgc.llm.proposer as proposer_module
import vgc.search as search_module
from vgc.actions import describe_order
from vgc.agent import VgcPlayer
from vgc.evaluator import ScoredOrder
from vgc.llm import packet as packet_module
from vgc.llm.proposer import make_order_proposer, propose_orders
from vgc.llm.thinking import choose_level as real_choose_level
from vgc.models import PolicyConfig
from vgc.search import ExchangeResult, OppResponse, _OppSlotAction

TAGS = "abcdef"


def _order(tag: str) -> SimpleNamespace:
    single = SimpleNamespace(
        order=tag, mega=False, z_move=False, dynamax=False, terastallize=False, move_target=0
    )
    return SimpleNamespace(first_order=single, second_order=single, message=f"/choose move {tag}")


def _scored() -> list[ScoredOrder]:
    # Small gaps so the engine is "unsure"; best first.
    return [ScoredOrder(_order(t), 100.0 - 2 * i, {}) for i, t in enumerate(TAGS)]


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    proposer_module.reset_proposer_state()

    def build_packet(battle, scored, *, llm_config, team_plan, memory=None, request_id, **_kw):
        options = packet_module.build_options(scored, llm_config.max_options)
        packet = packet_module.build_packet(
            {"plan": team_plan or "none"}, {"board": ["x"]}, options, request_id=request_id
        )
        return packet, options

    facts = types.ModuleType("vgc.llm.facts")
    facts.build_packet = build_packet
    facts.load_team_plan = lambda name: f"plan:{name}"
    monkeypatch.setitem(sys.modules, "vgc.llm.facts", facts)
    # Skip the real thinking-level policy (needs >= 6 s of budget) so tests stay fast.
    monkeypatch.setattr(proposer_module, "choose_level", lambda *a, **k: "none")
    real_cfg = proposer_module.llm_config_for
    monkeypatch.setattr(
        proposer_module, "llm_config_for",
        lambda config: dataclasses.replace(real_cfg(config), min_call_budget_s=0.3),
    )
    monkeypatch.setattr(proposer_module, "_decision_kind", lambda battle: ("turn", False))
    yield
    proposer_module.reset_proposer_state()


def _cfg(tmp_path, scenario="valid", **kw) -> PolicyConfig:
    return PolicyConfig(
        llm_proposer_enabled=True,
        llm_fake_scenario=scenario,
        llm_log_path=str(tmp_path / "calls.jsonl"),
        **kw,
    )


def _patch_search(monkeypatch, entries) -> None:
    monkeypatch.setattr(search_module, "score_joint_orders", lambda _b, _c: list(entries))
    monkeypatch.setattr(search_module, "build_context", lambda _b, _c: object())
    response = OppResponse(slot0=_OppSlotAction(kind="none"), slot1=_OppSlotAction(kind="none"))
    monkeypatch.setattr(search_module, "_enumerate_opp_responses", lambda _c, _cfg: [response])
    monkeypatch.setattr(search_module, "resolve_exchange", lambda *a, **k: ExchangeResult())
    monkeypatch.setattr(search_module, "_record_search_trace", lambda scored, config: None)


def test_valid_proposals_map_to_engine_orders_and_are_logged(tmp_path) -> None:
    scored = _scored()
    config = _cfg(tmp_path)
    orders = propose_orders(SimpleNamespace(turn=3), scored, config, 1.0)
    # The fake 'valid' client proposes the first two option IDs, i.e. the engine's top two.
    assert [o is scored[i].order for i, o in enumerate(orders)] == [True, True]
    rows = [json.loads(x) for x in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert rows and rows[0]["status"] == "ok"


@pytest.mark.parametrize("scenario", ["malformed_json", "stalled", "unknown_id", "refusal"])
def test_failures_return_no_orders(tmp_path, scenario) -> None:
    assert propose_orders(SimpleNamespace(turn=3), _scored(), _cfg(tmp_path, scenario), 0.6) == []


def test_forced_switch_and_clear_gap_skip_the_call(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(proposer_module, "choose_level", real_choose_level)  # real policy
    monkeypatch.setattr(proposer_module, "_decision_kind", lambda b: ("forced_switch", False))
    config = _cfg(tmp_path)
    assert propose_orders(SimpleNamespace(turn=3), _scored(), config, 30.0) == []
    client, _ = proposer_module._runtime(config)
    assert client.calls == []
    monkeypatch.setattr(proposer_module, "_decision_kind", lambda b: ("turn", False))
    clear = [ScoredOrder(_order("a"), 100.0, {}), ScoredOrder(_order("b"), 0.0, {})]
    assert propose_orders(SimpleNamespace(turn=3), clear, config, 30.0) == []
    assert client.calls == []


def test_search_adds_proposals_to_shortlist_and_engine_scores_them(monkeypatch, tmp_path) -> None:
    entries = _scored()
    _patch_search(monkeypatch, entries)
    base = PolicyConfig(search_our_candidates=1, search_diverse_candidates=False,
                        use_rolling_horizon=False)

    def run(**kw):
        return search_module.search_joint_orders(object(), base, **kw)

    def key(result):
        return [(describe_order(e.order), e.score) for e in result]

    # Disabled / zero extras: identical to the plain call.
    assert key(run(order_proposer=None, extra_candidates=0)) == key(run())

    # Proposer names two orders outside the top-1 shortlist: both get searched, flagged.
    proposed = [entries[3].order, entries[4].order]
    result = run(order_proposer=lambda ranked: proposed)
    searched = [e for e in result if e.breakdown["searched"]]
    assert len(searched) == 3
    flagged = [e for e in result if e.breakdown.get("llm_proposed")]
    assert {id(e.order) for e in flagged} == {id(o) for o in proposed}
    assert {id(e.order) for e in result} == {id(e.order) for e in entries}  # nothing lost/legal

    # A crashing proposer degrades to the plain search.
    def boom(_ranked):
        raise RuntimeError("x")

    assert key(run(order_proposer=boom)) == key(run())

    # Equal-time control arm widens the shortlist by N, in ranking order.
    control = run(extra_candidates=2)
    assert sum(e.breakdown["searched"] for e in control) == 3


def test_end_to_end_valid_fake_through_player_search(monkeypatch, tmp_path) -> None:
    entries = _scored()
    _patch_search(monkeypatch, entries)
    config = _cfg(
        tmp_path, search_our_candidates=1, search_diverse_candidates=False,
        use_rolling_horizon=False,
    )
    battle = SimpleNamespace(turn=2)
    result = search_module.search_joint_orders(
        battle, config, order_proposer=make_order_proposer(battle, config)
    )
    assert any(e.breakdown.get("llm_proposed") for e in result)
    assert result[0].order in [e.order for e in entries]  # the chosen order is a legal one


def test_player_only_passes_proposer_when_enabled(monkeypatch) -> None:
    calls: list[dict] = []
    monkeypatch.setattr(
        agent_module, "search_joint_orders", lambda battle, config, **kw: calls.append(kw) or []
    )
    memory = SimpleNamespace()
    for cfg, expected in [
        (PolicyConfig(), {}),
        (PolicyConfig(llm_control_extra_candidates=4), {"extra_candidates": 4}),
        (PolicyConfig(llm_proposer_enabled=True), {"order_proposer"}),
    ]:
        player = VgcPlayer.__new__(VgcPlayer)
        player.config = cfg
        player._search(SimpleNamespace(), memory)
        assert set(calls[-1]) == set(expected)
        if isinstance(expected, dict):
            assert calls[-1] == expected


def test_slow_packet_build_skips_the_paid_call(monkeypatch, tmp_path) -> None:
    import time as time_module

    real_build = sys.modules["vgc.llm.facts"].build_packet

    def slow_build(*args, **kwargs):
        time_module.sleep(0.5)  # eats the whole budget before `advise` could start its clock
        return real_build(*args, **kwargs)

    monkeypatch.setattr(sys.modules["vgc.llm.facts"], "build_packet", slow_build)
    config = _cfg(tmp_path)
    assert propose_orders(SimpleNamespace(turn=3), _scored(), config, 0.6) == []
    client, _ = proposer_module._runtime(config)
    assert client.calls == []  # no request was launched

    # Same for the guarded decision's own clock: little time left -> no call.
    monkeypatch.setattr(sys.modules["vgc.llm.facts"], "build_packet", real_build)
    monkeypatch.setattr(proposer_module, "time_left", lambda: 0.35)
    assert propose_orders(SimpleNamespace(turn=3), _scored(), config, 5.0) == []
    assert client.calls == []
    monkeypatch.setattr(proposer_module, "time_left", lambda: None)
    assert propose_orders(SimpleNamespace(turn=3), _scored(), config, 5.0)  # control: works


def test_spend_file_shared_across_caps(monkeypatch, tmp_path) -> None:
    from vgc.llm.spend import SpendMeter, shared_meter

    path = tmp_path / "spend.json"
    a = shared_meter(path, 20.0)
    b = shared_meter(str(path), 19.0)
    assert a is b and a.cap_usd == 19.0  # one tally, smallest cap wins
    # Reserve most of the cap; the other "config" must see it and be refused.
    worst = a.worst_case_usd(10_000_000, 10_000_000)
    n = int(19.0 / worst)
    held = [a.reserve(10_000_000, 10_000_000) for _ in range(n)]
    assert all(held) and b.reserve(10_000_000, 10_000_000) is None
    for res in held:
        a.settle(res)
    assert json.loads(path.read_text())["spent_usd"] == pytest.approx(worst * n)

    # Independent meters (e.g. two processes) merge through the file instead of overwriting.
    p2 = tmp_path / "two.json"
    m1, m2 = SpendMeter(p2, 1.0), SpendMeter(p2, 1.0)
    r1, r2 = m1.reserve(1_000, 1_000), m2.reserve(1_000, 1_000)
    m1.settle(r1, 0.10)
    m2.settle(r2, 0.20)
    data = json.loads(p2.read_text())
    assert data["spent_usd"] == pytest.approx(0.30) and data["calls"] == 2 and not data["reserved"]

    # Through the proposer: two configs, same file, caps 20 and 19, share one meter.
    proposer_module.reset_proposer_state()
    monkeypatch.setattr(proposer_module, "OpenAIResponsesClient", lambda model: object())
    f = str(tmp_path / "p.json")
    c1 = PolicyConfig(llm_spend_file=f, llm_budget_cap_usd=20.0)
    c2 = PolicyConfig(llm_spend_file=f, llm_budget_cap_usd=19.0)
    m_a = proposer_module._runtime(c1)[1]
    m_b = proposer_module._runtime(c2)[1]
    assert m_a is m_b and m_a.cap_usd == 19.0
