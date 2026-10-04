"""Knobs for the LLM advisory layer.

Kept in its own frozen dataclass (not `PolicyConfig`) until the layer is wired into play.
All score-gap thresholds are in the engine's own score units and are placeholders to be
calibrated from replay logs before the LLM is allowed to influence a move.
"""

from __future__ import annotations

from dataclasses import dataclass

# Ordered weakest -> strongest. Luna's `reasoning.effort` values.
LEVELS: tuple[str, ...] = ("none", "low", "medium", "high", "xhigh", "max")


@dataclass(frozen=True)
class LLMConfig:
    model: str = "gpt-6-luna"
    # Options shown to the model per turn (engine top picks + switch/Protect top-ups).
    max_options: int = 30
    # Proposals we keep from an answer.
    max_proposals: int = 3

    # --- timing (seconds of the turn clock) ---
    # Below this much clock we do not call at all.
    min_call_budget_s: float = 6.0
    # A second attempt is only made if at least this much clock is still left.
    retry_min_remaining_s: float = 6.0
    # Held back from the per-call timeout so we return before the deadline.
    deadline_margin_s: float = 0.25

    # --- thinking level choice (see thinking.choose_level) ---
    # Engine top beats runner-up by at least this: the call is not worth making.
    clear_gap: float = 30.0
    # Top-two gap at or below this counts as a close decision.
    close_gap: float = 8.0
    low_min_budget_s: float = 10.0
    medium_min_budget_s: float = 20.0
    # High and above are never used live, whatever the caller asks for.
    max_live_level: str = "medium"

    # Cap on thinking + answer tokens, per level. If hit the call is 'incomplete', gives
    # no answer, and is still billed, so keep these tight.
    max_output_tokens_none: int = 500
    max_output_tokens_low: int = 1500
    max_output_tokens_medium: int = 3500
    max_output_tokens_high: int = 8000
    max_output_tokens_xhigh: int = 16000
    max_output_tokens_max: int = 32000

    # Hard local spend cap in USD (persisted by SpendMeter).
    spend_cap_usd: float = 20.0

    def max_output_tokens_for(self, level: str) -> int:
        return int(getattr(self, f"max_output_tokens_{level}"))
