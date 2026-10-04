"""Decide whether to call the LLM and how hard it should think."""

from __future__ import annotations

from vgc.llm.config import LEVELS, LLMConfig

_FORCED = {"forced_switch", "forced"}


def _clamp(level: str, config: LLMConfig) -> str:
    cap = LEVELS.index(config.max_live_level)
    return LEVELS[min(LEVELS.index(level), cap)]


def choose_level(
    decision_kind: str,
    budget_s: float,
    engine_score_gap: float | None,
    n_legal: int,
    *,
    critical: bool = False,
    config: LLMConfig | None = None,
) -> str | None:
    """Return a reasoning effort, or None meaning "do not call".

    decision_kind: 'turn', 'preview', or 'forced_switch'.
    engine_score_gap: engine top minus runner-up (None if unknown, e.g. at preview).
    """
    cfg = config or LLMConfig()
    if n_legal <= 1 or decision_kind in _FORCED:
        return None  # nothing to decide
    if budget_s < cfg.min_call_budget_s:
        return None  # no time to wait for an answer
    if engine_score_gap is not None and engine_score_gap >= cfg.clear_gap:
        return None  # engine is clearly right; do not spend money or time
    if (decision_kind == "preview" or critical) and budget_s >= cfg.medium_min_budget_s:
        return _clamp("medium", cfg)
    if (
        engine_score_gap is not None
        and engine_score_gap <= cfg.close_gap
        and budget_s >= cfg.low_min_budget_s
    ):
        return _clamp("low", cfg)
    return "none"
