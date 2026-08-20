from __future__ import annotations

import argparse

import numpy as np
import pytest

from offline.evaluate_search_value import _parse_weights, parse_args
from offline.evaluate_search_value_gate import parse_args as parse_gate_args
from vgc.bc.encoding import SPECIES_TO_IDX
from vgc.rl.encoding import (
    INFORMATION_INDEX_DIM,
    INFORMATION_SCALAR_DIM,
    INFORMATION_TEAM_SLOTS,
    OWN_SCALARS_PER_MON,
    InformationFeatures,
)
from vgc.rl.search_value import _patch_information, summarize_value_records


def _information() -> InformationFeatures:
    indices = np.zeros(INFORMATION_INDEX_DIM, dtype=np.int64)
    indices[0] = SPECIES_TO_IDX["charizard"]
    opp_start = INFORMATION_TEAM_SLOTS * (3 + 4)
    indices[opp_start] = SPECIES_TO_IDX["garchomp"]
    scalars = np.zeros(INFORMATION_SCALAR_DIM, dtype=np.float32)
    scalars[0] = 1.0
    scalars[1] = 1.0
    scalars[2] = 1.0
    opp_scalar_start = INFORMATION_TEAM_SLOTS * OWN_SCALARS_PER_MON
    scalars[opp_scalar_start] = 1.0
    scalars[opp_scalar_start + 1] = 1.0
    scalars[opp_scalar_start + 2] = 1.0
    scalars[opp_scalar_start + 3] = 1.0
    return InformationFeatures(indices=indices, scalars=scalars)


def test_future_information_patch_updates_only_public_position_facts() -> None:
    base = _information()
    record = {
        "state": {
            "our": {
                "active": [
                    {
                        "species": "charizard",
                        "hp_fraction": 0.25,
                        "status": "brn",
                    }
                ],
                "bench": [],
            },
            "opp": {
                "active": [
                    {
                        "species": "garchomp",
                        "hp_fraction": 0.0,
                        "status": None,
                    }
                ],
                "bench": [],
            },
        }
    }

    patched = _patch_information(base, record)

    assert patched.indices is not base.indices
    assert np.array_equal(patched.indices, base.indices)
    assert patched.scalars[1] == 1.0
    assert patched.scalars[2] == 0.25
    assert patched.scalars[3] == 0.0
    assert patched.scalars[5] == 1.0
    opp_start = INFORMATION_TEAM_SLOTS * OWN_SCALARS_PER_MON
    assert patched.scalars[opp_start + 2] == 0.0
    assert patched.scalars[opp_start + 3] == 0.0
    assert patched.scalars[opp_start + 4] == 1.0
    assert base.scalars[2] == 1.0


def _decision(battle: int, turn: int, outcome: float, value: float) -> dict[str, object]:
    return {
        "battle_id": f"battle-{battle}",
        "team_id": f"team-{battle % 4}",
        "turn": turn,
        "outcome": outcome,
        "value_before": value,
        "value_latency_ms": 2.0,
        "baseline_winner": "A",
        "shadow_winners": {"0.1": "A" if battle % 2 else "B"},
        "actions": [
            {
                "action": "A",
                "search_score": float(turn),
                "rl_value_after_aggregate": value,
            },
            {
                "action": "B",
                "search_score": float(turn) - 1.0,
                "rl_value_after_aggregate": -value,
            },
        ],
    }


def test_summary_uses_expected_return_calibration_and_game_disjoint_prediction() -> None:
    records = [
        _decision(battle, turn, 1.0 if battle % 2 else -1.0, 0.8 if battle % 2 else -0.8)
        for battle in range(20)
        for turn in (1, 2)
    ]

    summary = summarize_value_records(records, (0.1,))

    assert summary["calibration"]["target"] == "expected_terminal_return_minus1_to_plus1"
    assert summary["calibration"]["expected_return_mse"] == pytest.approx(0.04)
    assert summary["calibration"]["win_probability_brier"] == pytest.approx(0.01)
    assert summary["incremental_predictive_value"]["available"] is True
    assert summary["shadow_choice_changes"]["0.1"]["decisions_changed"] == 20
    assert summary["outcome_label_contract"].startswith("only the baseline")


def test_value_weight_cli_requires_positive_declared_weights() -> None:
    assert _parse_weights("0.25,0.1,0.25") == (0.1, 0.25)
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_weights("0,-1")
    assert parse_args(["--value-weights", "0.05,0.1"]).value_weights == (0.05, 0.1)
    assert parse_gate_args(["--value-weight", "0.05"]).value_weight == 0.05
