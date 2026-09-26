"""TypeSafe System One question definitions for VGC turn judgments.

All question text, option rubrics, and confidence thresholds for downstream A/B
consumption live here — callers batch them into one HTTP request per decision.
"""

from __future__ import annotations

from poke_env.battle import DoubleBattle

from vgc.damage import to_id

# Question ids (API map keys; not sent to the model).
QUESTION_OPPONENT_MODE = "opponent_mode"
QUESTION_PLAY_SLOW = "play_slow"
QUESTION_MAIN_THREAT = "main_threat"

SYSTEM_ONE_QUESTION_IDS = (
    QUESTION_OPPONENT_MODE,
    QUESTION_PLAY_SLOW,
    QUESTION_MAIN_THREAT,
)

# Confidence floors for a future same-session A/B that may gate on Jev output.
# Unused for move selection in this integration (system two stays authoritative).
OPPONENT_MODE_CONFIDENCE_FLOOR = 0.35
PLAY_SLOW_NOUL_EXTREME_MARGIN = 0.25
MAIN_THREAT_CONFIDENCE_FLOOR = 0.30

_OPPONENT_MODE_CRITERIA: dict[str, str] = {
    "aggressive": "Pressing damage, KOs, or fast tempo; little setup or stalling.",
    "defensive": "Protecting, pivoting, or reducing damage taken this turn.",
    "setup_or_support": "Setting up stats, field, or enabling the partner.",
    "mixed_or_unclear": "No single mode clearly dominates from public information.",
}


def build_questions(battle: DoubleBattle) -> dict[str, dict[str, object]]:
    """Typed question payloads for one batched System One call."""

    threat_criteria: dict[str, str] = {}
    for idx, mon in enumerate(battle.opponent_active_pokemon or []):
        if mon is None or mon.fainted:
            continue
        species = to_id(mon.species)
        key = f"slot_{idx}_{species}"
        threat_criteria[key] = (
            f"Opponent active slot {idx + 1}: {species} "
            f"(HP ~{int(round(mon.current_hp_fraction * 100))}%)."
        )
    if not threat_criteria:
        threat_criteria["none_visible"] = "No opposing active Pokemon are visible."

    return {
        QUESTION_OPPONENT_MODE: {
            "type": "choice",
            "instructions": (
                "From the public Pokemon VGC doubles board, which opponent game plan "
                "best matches what they are trying to do this turn?"
            ),
            "criteria": dict(_OPPONENT_MODE_CRITERIA),
        },
        QUESTION_PLAY_SLOW: {
            "type": "noul",
            "instructions": (
                "Should our side prioritize a slow line this turn — Protect, pivot, "
                "or stall — over maximal immediate damage?"
            ),
            "criteria": {
                "true": "Slow tempo is appropriate given pressure and field state.",
                "false": "We should press offense or tempo rather than play slow.",
            },
        },
        QUESTION_MAIN_THREAT: {
            "type": "choice",
            "instructions": (
                "Which opposing active Pokemon is the main threat our line must answer "
                "this turn?"
            ),
            "criteria": threat_criteria,
        },
    }
