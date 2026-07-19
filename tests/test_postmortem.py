from __future__ import annotations

from vgc.postmortem import classify_loss


def test_classifies_wrong_lead_and_wrong_four_from_preview_plan() -> None:
    traces = [
        {
            "turn": 0,
            "notes": {
                "team_preview_choice": {
                    "lead_functions": ["immediate_pressure"],
                    "closer_picked": False,
                    "engine_targets_covered": 0,
                    "plan": {"opponent_engines": ["trick_room"]},
                }
            },
        }
    ]
    result = classify_loss(traces)
    categories = {reason["category"] for reason in result["reasons"]}
    assert {"wrong_lead", "wrong_four"} <= categories


def test_classifies_speed_positioning_closer_prediction_and_uncertainty() -> None:
    traces = [
        {"turn": 0, "notes": {"team_preview_choice": {"lead_functions": ["a", "b"]}}},
        {
            "turn": 1,
            "chosen_order": "heatwave / leafstorm@1",
            "notes": {
                "principles_turn": {
                    "our_speed": [80.0, 90.0],
                    "opponent_speed": [150.0, 160.0],
                    "double_target_threat": [140.0, 30.0],
                    "opponent_uncertainty": [3, 3],
                },
                "chosen_breakdown": {
                    "slot0": {"win_con_preservation_penalty": 40.0},
                    "slot1": {},
                },
                "search_top_candidates": [
                    {
                        "worst_response": "protect / trickroom",
                        "exchange_value": -90.0,
                    }
                ],
            },
        },
    ]
    result = classify_loss(traces)
    categories = {reason["category"] for reason in result["reasons"]}
    assert {
        "poor_speed_control",
        "lost_positioning",
        "sacrificed_closer",
        "unnecessary_prediction",
        "unknown_information",
    } <= categories


def test_empty_trace_is_still_classified() -> None:
    assert classify_loss([])["primary"] == "unknown_information"

