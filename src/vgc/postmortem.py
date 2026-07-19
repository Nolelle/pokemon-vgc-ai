"""Automatic first-principles loss classification from saved decision traces."""

from __future__ import annotations

from dataclasses import dataclass

from vgc.principles import SPEED_CONTROL_MOVES


@dataclass(frozen=True)
class LossReason:
    category: str
    confidence: float
    evidence: str

    def as_dict(self) -> dict[str, object]:
        return {
            "category": self.category,
            "confidence": round(self.confidence, 3),
            "evidence": self.evidence,
        }


def _slot_breakdowns(trace: dict) -> list[dict]:
    chosen = (trace.get("notes") or {}).get("chosen_breakdown") or {}
    return [chosen.get("slot0") or {}, chosen.get("slot1") or {}]


def classify_loss(traces: list[dict]) -> dict[str, object]:
    """Return a ranked explanation using the document's eight post-loss categories.

    This is diagnostic attribution, not ground truth. Every returned reason includes the
    trace fact that triggered it so replay review can confirm or reject the hypothesis.
    """
    if not traces:
        reason = LossReason("unknown_information", 0.25, "no decision trace was available")
        return {"primary": reason.category, "reasons": [reason.as_dict()]}

    reasons: list[LossReason] = []
    preview_notes = (traces[0].get("notes") or {}).get("team_preview_choice") or {}
    plan = preview_notes.get("plan") or {}
    lead_functions = set(preview_notes.get("lead_functions") or plan.get("lead_functions") or [])
    if preview_notes and len(lead_functions) < 2:
        reasons.append(
            LossReason(
                "wrong_lead",
                0.8,
                f"lead supplied only {len(lead_functions)} strategic functions",
            )
        )
    if preview_notes and (
        preview_notes.get("closer_picked") is False
        or (
            plan.get("opponent_engines")
            and int(preview_notes.get("engine_targets_covered", 0)) == 0
        )
    ):
        reasons.append(
            LossReason(
                "wrong_four",
                0.8,
                "selected four omitted the closer or lacked an answer to the opposing engine",
            )
        )

    turn_traces = [trace for trace in traces if int(trace.get("turn") or 0) > 0]
    speed_disadvantage_turns = 0
    high_uncertainty_turns = 0
    high_danger_unprotected_turns = 0
    used_speed_control = False
    sacrificed_closer = False
    worst_response_collapse = False
    fallback = False
    for trace in turn_traces:
        notes = trace.get("notes") or {}
        principles = notes.get("principles_turn") or {}
        our_speeds = [float(value) for value in principles.get("our_speed") or [] if value]
        opp_speeds = [float(value) for value in principles.get("opponent_speed") or [] if value]
        if our_speeds and opp_speeds:
            if sum(opp_speeds) / len(opp_speeds) > sum(our_speeds) / len(our_speeds):
                speed_disadvantage_turns += 1
        if sum(principles.get("opponent_uncertainty") or []) >= 4:
            high_uncertainty_turns += 1

        chosen_order = str(trace.get("chosen_order") or "")
        if any(move_id in chosen_order for move_id in SPEED_CONTROL_MOVES):
            used_speed_control = True
        danger = max(principles.get("double_target_threat") or [0.0])
        if danger >= 100.0 and "protect" not in chosen_order and "switch->" not in chosen_order:
            high_danger_unprotected_turns += 1
        for slot in _slot_breakdowns(trace):
            if float(slot.get("win_con_preservation_penalty", 0.0)) > 0.0:
                sacrificed_closer = True
        search_candidates = notes.get("search_top_candidates") or []
        if search_candidates:
            worst = search_candidates[0].get("worst_response")
            exchange = search_candidates[0].get("exchange_value")
            if worst and exchange is not None and float(exchange) <= -75.0:
                worst_response_collapse = True
        fallback = fallback or bool(trace.get("fallback_used"))

    n_turns = max(1, len(turn_traces))
    if speed_disadvantage_turns >= max(1, n_turns // 2) and not used_speed_control:
        reasons.append(
            LossReason(
                "poor_speed_control",
                0.75,
                f"slower on {speed_disadvantage_turns}/{n_turns} turns without using speed control",
            )
        )
    if high_danger_unprotected_turns:
        reasons.append(
            LossReason(
                "lost_positioning",
                min(0.9, 0.55 + 0.1 * high_danger_unprotected_turns),
                f"stayed exposed to a combined KO threat on {high_danger_unprotected_turns} turns",
            )
        )
    if sacrificed_closer:
        reasons.append(
            LossReason(
                "sacrificed_closer",
                0.9,
                "the chosen attack incurred the explicit win-condition preservation penalty",
            )
        )
    if worst_response_collapse:
        reasons.append(
            LossReason(
                "unnecessary_prediction",
                0.7,
                "the selected line had a severely losing reasonable opponent response",
            )
        )
    if high_uncertainty_turns >= max(1, n_turns // 2):
        reasons.append(
            LossReason(
                "unknown_information",
                0.6,
                f"at least four opponent set details were unknown on {high_uncertainty_turns} turns",
            )
        )
    if fallback:
        reasons.append(
            LossReason(
                "missed_damage_calculation",
                0.5,
                "a policy fallback prevented the intended calculated decision",
            )
        )

    if not reasons:
        reasons.append(
            LossReason(
                "lost_positioning",
                0.3,
                "no single high-confidence trigger fired; replay positioning review is required",
            )
        )
    reasons.sort(key=lambda reason: reason.confidence, reverse=True)
    return {"primary": reasons[0].category, "reasons": [reason.as_dict() for reason in reasons]}

