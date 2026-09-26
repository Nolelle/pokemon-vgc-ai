"""System One (Jev) integration: mock HTTP only, no live API."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from urllib.error import URLError

from poke_env.battle import DoubleBattle

from vgc.decision_trace import finish_trace, get_last_trace, start_trace
from vgc.evaluator import score_joint_orders
from vgc.models import PolicyConfig
from vgc.system_one.client import TYPESAFE_API_KEY_ENV
from vgc.system_one.jev import TRACE_KEY
from vgc.system_one.questions import (
    QUESTION_MAIN_THREAT,
    QUESTION_OPPONENT_MODE,
    QUESTION_PLAY_SLOW,
)


def _minimal_double_battle() -> DoubleBattle:
    battle = MagicMock(spec=DoubleBattle)
    battle.battle_tag = "battle-gen9-test-1"
    battle.turn = 3
    battle.fields = set()
    battle.weather = set()
    battle.side_conditions = set()
    battle.opponent_side_conditions = set()
    battle.active_pokemon = [None, None]
    battle.opponent_active_pokemon = [None, None]
    battle.teambuilder_team = []
    battle.teampreview_opponent_team = []
    battle.opponent_team = {}
    return battle


def test_flag_off_never_calls_http(monkeypatch):
    monkeypatch.setenv("VGC_TRACE", "1")
    monkeypatch.setenv(TYPESAFE_API_KEY_ENV, "test-key")
    battle = _minimal_double_battle()
    config = PolicyConfig(use_jev_system_one=False)

    with patch("vgc.system_one.jev.evaluate_system_one") as mock_eval:
        with patch("vgc.evaluator.enumerate_joint_orders", return_value=[]):
            score_joint_orders(battle, config)
    mock_eval.assert_not_called()


def test_mocked_success_on_trace(monkeypatch):
    monkeypatch.setenv("VGC_TRACE", "1")
    monkeypatch.setenv(TYPESAFE_API_KEY_ENV, "test-key")
    battle = _minimal_double_battle()
    config = PolicyConfig(use_jev_system_one=True)
    mock_response = {
        "model": "jev-1.13.0",
        "answers": {
            QUESTION_OPPONENT_MODE: {
                "type": "choice",
                "choice": "aggressive",
                "probabilities": {"aggressive": 0.7, "defensive": 0.3},
                "confidence": 0.5,
            },
            QUESTION_PLAY_SLOW: {"type": "noul", "noul": 0.2},
            QUESTION_MAIN_THREAT: {
                "type": "choice",
                "choice": "none_visible",
                "probabilities": {"none_visible": 1.0},
                "confidence": 0.9,
            },
        },
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }

    token = start_trace()
    with patch("vgc.system_one.jev.evaluate_system_one", return_value=mock_response):
        with patch("vgc.evaluator.enumerate_joint_orders", return_value=[]):
            score_joint_orders(battle, config)
    finish_trace(token)

    trace = get_last_trace()
    assert trace is not None
    system_one = trace.notes.get(TRACE_KEY)
    assert isinstance(system_one, dict)
    assert system_one["status"] == "ok"
    assert QUESTION_OPPONENT_MODE in system_one["answers"]


def test_missing_key_skipped_without_raise(monkeypatch):
    monkeypatch.setenv("VGC_TRACE", "1")
    monkeypatch.delenv(TYPESAFE_API_KEY_ENV, raising=False)
    battle = _minimal_double_battle()
    config = PolicyConfig(use_jev_system_one=True)

    token = start_trace()
    with patch("vgc.system_one.jev.evaluate_system_one") as mock_eval:
        with patch("vgc.evaluator.enumerate_joint_orders", return_value=[]):
            score_joint_orders(battle, config)
    finish_trace(token)
    mock_eval.assert_not_called()

    trace = get_last_trace()
    assert trace is not None
    assert trace.notes[TRACE_KEY] == {"status": "skipped", "reason": "missing_api_key"}


def test_timeout_recorded_without_raise(monkeypatch):
    from vgc.system_one.client import SystemOneClientError

    monkeypatch.setenv("VGC_TRACE", "1")
    monkeypatch.setenv(TYPESAFE_API_KEY_ENV, "test-key")
    battle = _minimal_double_battle()
    config = PolicyConfig(use_jev_system_one=True)

    token = start_trace()
    with patch(
        "vgc.system_one.jev.evaluate_system_one",
        side_effect=SystemOneClientError("request timed out"),
    ):
        with patch("vgc.evaluator.enumerate_joint_orders", return_value=[]):
            score_joint_orders(battle, config)
    finish_trace(token)

    trace = get_last_trace()
    assert trace is not None
    entry = trace.notes[TRACE_KEY]
    assert entry["status"] == "error"
    assert "timed out" in entry["reason"]


def test_client_maps_url_error_to_client_error(monkeypatch):
    from vgc.system_one.client import SystemOneClientError, evaluate_system_one

    monkeypatch.setenv(TYPESAFE_API_KEY_ENV, "k")

    def fake_urlopen(_request, timeout=0):
        raise URLError("timed out")

    with patch("vgc.system_one.client.urllib.request.urlopen", fake_urlopen):
        try:
            evaluate_system_one("state", {"q": {"type": "noul", "instructions": "x"}}, api_key="k")
        except SystemOneClientError as exc:
            assert "timed out" in str(exc)
        else:
            raise AssertionError("expected SystemOneClientError")
