"""Orchestrate one batched Jev call per decision and attach results to the trace."""

from __future__ import annotations

from typing import Any

from poke_env.battle import DoubleBattle

from vgc.battle_memory import BattleMemory
from vgc.decision_trace import record_note
from vgc.models import PolicyConfig
from vgc.system_one.client import (
    DEFAULT_TIMEOUT_SECONDS,
    SystemOneClientError,
    evaluate_system_one,
    read_api_key_from_env,
)
from vgc.system_one.questions import SYSTEM_ONE_QUESTION_IDS, build_questions
from vgc.system_one.state import build_public_board_state

TRACE_KEY = "system_one"


def _normalize_answers(raw: dict[str, Any]) -> dict[str, object]:
    answers = raw.get("answers")
    if not isinstance(answers, dict):
        return {}
    normalized: dict[str, object] = {}
    for question_id in SYSTEM_ONE_QUESTION_IDS:
        entry = answers.get(question_id)
        if not isinstance(entry, dict):
            continue
        normalized[question_id] = dict(entry)
    return normalized


def maybe_attach_system_one_judgments(
    battle: DoubleBattle,
    memory: BattleMemory | None,
    config: PolicyConfig,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> None:
    """Run System One when enabled; never raises (battle-safe side channel)."""

    if not config.use_jev_system_one:
        return

    api_key = read_api_key_from_env()
    if api_key is None:
        record_note(
            TRACE_KEY,
            {"status": "skipped", "reason": "missing_api_key"},
        )
        return

    mem = memory if memory is not None else BattleMemory(battle_tag=battle.battle_tag)
    try:
        state = build_public_board_state(battle, mem, config)
        questions = build_questions(battle)
        response = evaluate_system_one(
            state,
            questions,
            api_key=api_key,
            timeout=timeout,
        )
        record_note(
            TRACE_KEY,
            {
                "status": "ok",
                "model": response.get("model"),
                "answers": _normalize_answers(response),
                "usage": response.get("usage"),
            },
        )
    except SystemOneClientError as exc:
        record_note(
            TRACE_KEY,
            {"status": "error", "reason": str(exc)},
        )
    except Exception as exc:  # noqa: BLE001 - must not disturb search/evaluator
        record_note(
            TRACE_KEY,
            {"status": "error", "reason": f"unexpected: {exc!r}"},
        )
