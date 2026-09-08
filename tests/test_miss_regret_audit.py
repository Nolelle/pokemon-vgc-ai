"""Unit tests for the miss-regret audit over recorded shadow/hybrid decisions."""

from __future__ import annotations

import json

import pytest

from offline.audit_miss_regret import action_count_bucket, summarize_regret


def _record(
    *,
    retained: bool,
    regret: float | None,
    team: str = "pool/a",
    actions: int = 150,
    turn: int = 3,
) -> dict:
    return {
        "team_id": team,
        "legal_action_count": actions,
        "turn": turn,
        "retention": {"neural_at_10": {"retained": retained, "known_regret": regret}},
    }


def _records() -> list[dict]:
    return [
        _record(retained=True, regret=None),
        # Free miss: best kept action within 5 points of the winner.
        _record(retained=False, regret=3.0),
        # Costly miss.
        _record(retained=False, regret=80.0),
        # Miss with unknown regret (winner not among reference scores).
        _record(retained=False, regret=None),
        _record(retained=True, regret=None, team="pool/b", actions=8, turn=1),
    ]


def test_bucket_labels_match_the_recall_screen_bins():
    assert action_count_bucket(0) == "unknown"
    assert action_count_bucket(10) == "0-25"
    assert action_count_bucket(26) == "26-100"
    assert action_count_bucket(200) == "101-200"
    assert action_count_bucket(201) == "201+"


def test_regret_stats_cover_misses_only():
    summary = summarize_regret(_records(), metric="neural_at_10", thresholds=[5.0])
    assert summary["decisions"] == 5
    assert summary["retained"] == 2
    assert summary["misses"] == 3
    assert summary["misses_with_known_regret"] == 2
    assert summary["regret_distribution"]["mean"] == pytest.approx((3.0 + 80.0) / 2)
    assert summary["regret_distribution"]["max"] == pytest.approx(80.0)


def test_negligible_threshold_excuses_only_cheap_known_misses():
    summary = summarize_regret(_records(), metric="neural_at_10", thresholds=[5.0])
    block = summary["negligible_thresholds"]["le_5"]
    assert block["misses_within_threshold"] == 1
    overall = block["effective_retention_overall"]
    # 3 of 5 decisions are retained or cheaply missed; the 80-point miss and the
    # unknown-regret miss both stay misses.
    assert overall["rate"] == pytest.approx(3 / 5)
    assert block["effective_retention_among_misses"]["rate"] == pytest.approx(1 / 3)


def test_unknown_regret_cannot_be_excused_by_a_threshold():
    records = [_record(retained=False, regret=None)]
    summary = summarize_regret(records, metric="neural_at_10", thresholds=[100.0])
    block = summary["negligible_thresholds"]["le_100"]
    assert block["misses_within_threshold"] == 0
    assert block["effective_retention_overall"]["rate"] == 0.0
    assert summary["misses_with_unknown_regret"] == 1


def test_strata_buckets_split_decisions_and_share_of_misses():
    summary = summarize_regret(_records(), metric="neural_at_10", thresholds=[5.0])
    big = summary["by_action_count_bucket"]["101-200"]
    assert big["decisions"] == 4
    assert big["recall"] == pytest.approx(1 / 4)
    assert big["le_5_share_of_misses"] == pytest.approx(1 / 3)
    easy = summary["by_action_count_bucket"]["0-25"]
    assert easy["decisions"] == 1 and easy["recall"] == 1.0
    early = summary["by_phase_bucket"]["phase:early"]
    assert early["decisions"] == 1
    mid = summary["by_phase_bucket"]["phase:mid"]
    assert mid["decisions"] == 4


def test_records_missing_the_metric_are_counted_not_crashed():
    records = [{"team_id": "x", "retention": {}}] + _records()
    summary = summarize_regret(records, metric="guided_at_10", thresholds=[5.0])
    assert summary["decisions"] == 0
    assert summary["records_without_metric"] == 6


def test_invalid_metric_is_rejected():
    with pytest.raises(ValueError):
        summarize_regret(_records(), metric="top10", thresholds=[5.0])


def test_summary_is_json_serializable(tmp_path):
    summary = summarize_regret(_records(), metric="neural_at_10", thresholds=[5.0, 20.0])
    text = json.dumps(summary, sort_keys=True)
    assert "vgc-miss-regret-audit-v1" in text
