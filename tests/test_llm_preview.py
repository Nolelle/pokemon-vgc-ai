"""LLM team-preview advisor: off = identical order; valid fake answer = used; any bad one =
heuristic order (the fallback), with the call logged as kind "preview"."""

from __future__ import annotations

import dataclasses
import json

import pytest

import vgc.llm.proposer as proposer_module
from vgc.agent import VgcPlayer
from vgc.decision_trace import finish_trace, start_trace
from vgc.llm.preview import build_preview_packet, order_string, validate_answer
from vgc.models import PolicyConfig
from vgc.team_preview import build_team_order

from test_team_preview import _FakeBattle, _opp_team, _our_team

NAMES = ["Charizard", "Farigiraf", "Venusaur", "Garchomp", "Incineroar", "Sylveon"]
# FakeLLMClient "valid" preview: bring names[2:6], leads names[5], names[3] -> 6 4 3 5.
FAKE_VALID_ORDER = "/team 6435"


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("VGC_TRACE", "1")
    proposer_module.reset_proposer_state()
    real_cfg = proposer_module.llm_config_for
    monkeypatch.setattr(
        proposer_module, "llm_config_for",
        lambda config: dataclasses.replace(real_cfg(config), min_call_budget_s=0.3),
    )
    yield
    proposer_module.reset_proposer_state()


def _decide(tmp_path, scenario="valid", enabled=True, budget=20.0):
    config = PolicyConfig(
        llm_preview_enabled=enabled, llm_fake_scenario=scenario,
        llm_log_path=str(tmp_path / "calls.jsonl"), llm_preview_budget_s=budget,
    )
    battle = _FakeBattle(_our_team(), _opp_team())
    heuristic = build_team_order(battle, config)
    token = start_trace()
    order = VgcPlayer(config=config, start_listening=False).decide_teampreview(battle)
    trace = finish_trace(token)
    return battle, heuristic, order, trace.notes.get("llm_preview")


def _log(tmp_path):
    path = tmp_path / "calls.jsonl"
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def test_disabled_is_identical_and_makes_no_call(tmp_path) -> None:
    _battle, heuristic, order, note = _decide(tmp_path, enabled=False)
    assert order == heuristic and note is None and _log(tmp_path) == []


def test_valid_answer_replaces_heuristic_order(tmp_path) -> None:
    battle, heuristic, order, note = _decide(tmp_path)
    assert heuristic != FAKE_VALID_ORDER  # the test must be able to tell them apart
    assert order == FAKE_VALID_ORDER
    assert note["used"] and note["bring"] == NAMES[2:6] and note["leads"] == [NAMES[5], NAMES[3]]
    assert [m._selected_in_teampreview for m in battle.team.values()] == [
        False, False, True, True, True, True
    ]
    rows = _log(tmp_path)
    assert len(rows) == 1 and rows[0]["kind"] == "preview" and rows[0]["status"] == "ok"


@pytest.mark.parametrize(
    "scenario",
    ["preview_three_species", "unknown_id", "preview_bad_leads", "duplicate_ids",
     "malformed_json", "refusal", "empty", "rate_limited"],
)
def test_invalid_answers_keep_the_heuristic_order(tmp_path, scenario) -> None:
    _battle, heuristic, order, note = _decide(tmp_path, scenario)
    assert order == heuristic and not note["used"] and note["fallback_reason"] == "no_advice"


def test_stalled_call_keeps_the_heuristic_order(tmp_path) -> None:
    _battle, heuristic, order, note = _decide(tmp_path, "stalled", budget=0.6)
    assert order == heuristic and not note["used"]


def test_bad_level_keeps_the_heuristic_order(tmp_path) -> None:
    config = PolicyConfig(llm_preview_enabled=True, llm_preview_level="high",
                          llm_fake_scenario="valid", llm_log_path=str(tmp_path / "c.jsonl"))
    battle = _FakeBattle(_our_team(), _opp_team())
    heuristic = build_team_order(battle, config)
    player = VgcPlayer(config=config, start_listening=False)
    assert player.decide_teampreview(battle) == heuristic


def test_prompt_is_blind_and_schema_enumerates_our_six() -> None:
    config = PolicyConfig()
    battle = _FakeBattle(_our_team(), _opp_team())
    heuristic = build_team_order(battle, config)
    packet = build_preview_packet(battle, config, NAMES, "t000-x")
    text = packet.full_text
    assert heuristic not in text and "/team" not in text
    assert "GUESS" in text and "ESTIMATE" in text
    assert packet.schema["properties"]["bring"]["items"]["enum"] == NAMES
    assert len(text) // 4 < 6000


def test_validate_and_order_string() -> None:
    ok = {"plan": "p", "bring": NAMES[:4], "leads": [NAMES[3], NAMES[0]], "why": "w"}
    assert validate_answer(ok, NAMES) is None
    assert order_string(NAMES, ok["bring"], ok["leads"]) == "/team 4123"
