from __future__ import annotations

from offline.evaluate_belief_shortlist_recall import summarize


def _record(
    own_team: str,
    *,
    incumbent_hit: bool,
    candidate_hit: bool,
    turn: int = 2,
    sets_differ: bool | None = None,
    posterior_differs_from_prior: bool = False,
    speed_observations: int = 0,
    damage_observations: int = 0,
) -> dict:
    if sets_differ is None:
        sets_differ = incumbent_hit != candidate_hit
    return {
        "own_team": own_team,
        "opp_team": f"opp-of-{own_team}",
        "turn": turn,
        "n": 40,
        "winner": "move a, move b",
        "incumbent_hit": incumbent_hit,
        "candidate_hit": candidate_hit,
        "incumbent_rank": 1 if incumbent_hit else 12,
        "candidate_rank": 1 if candidate_hit else 12,
        "hypotheses_used": 3,
        "posterior_differs_from_prior": posterior_differs_from_prior,
        "speed_observations": speed_observations,
        "damage_observations": damage_observations,
        "sets_differ": sets_differ,
        "wide_search_seconds": 1.0,
    }


def _balanced_records(
    teams: int,
    per_team: int,
    *,
    incumbent_hit: bool,
    candidate_hit: bool,
) -> list[dict]:
    records = []
    for team_index in range(teams):
        team = f"team-{team_index:03d}"
        for turn in range(per_team):
            records.append(
                _record(
                    team,
                    incumbent_hit=incumbent_hit,
                    candidate_hit=candidate_hit,
                    turn=2 + turn,
                )
            )
    return records


def test_verdict_pass_when_mixture_matches_point_estimate() -> None:
    records = _balanced_records(40, 3, incumbent_hit=True, candidate_hit=True)

    summary = summarize(records)

    assert summary["decisions"] == 120
    assert summary["clusters"] == 40
    assert summary["recall_incumbent"] == 1.0
    assert summary["recall_candidate"] == 1.0
    assert summary["difference_mean"] == 0.0
    assert summary["difference_interval"][0] >= -0.02
    assert summary["verdict"] == "PASS"


def test_verdict_fail_when_mixture_drops_the_winner() -> None:
    records = _balanced_records(40, 3, incumbent_hit=True, candidate_hit=False)

    summary = summarize(records)

    assert summary["decisions"] == 120
    assert summary["clusters"] == 40
    assert summary["difference_mean"] == -1.0
    assert summary["difference_interval"][1] < -0.02
    assert summary["verdict"] == "FAIL"


def test_verdict_indeterminate_below_cluster_or_decision_floor() -> None:
    too_few_clusters = _balanced_records(39, 3, incumbent_hit=True, candidate_hit=True)
    too_few_decisions = _balanced_records(40, 2, incumbent_hit=True, candidate_hit=True)

    assert summarize(too_few_clusters)["verdict"] == "INDETERMINATE"
    assert summarize(too_few_clusters)["clusters"] == 39
    assert summarize(too_few_clusters)["decisions"] == 117
    assert summarize(too_few_decisions)["verdict"] == "INDETERMINATE"
    assert summarize(too_few_decisions)["clusters"] == 40
    assert summarize(too_few_decisions)["decisions"] == 80


def test_empty_records_are_indeterminate() -> None:
    summary = summarize([])

    assert summary["decisions"] == 0
    assert summary["clusters"] == 0
    assert summary["verdict"] == "INDETERMINATE"


def test_summarize_clusters_by_own_team_not_by_decision() -> None:
    records = [
        _record("alpha", incumbent_hit=True, candidate_hit=False, turn=2),
        _record("alpha", incumbent_hit=True, candidate_hit=False, turn=4),
        _record("beta", incumbent_hit=False, candidate_hit=True, turn=2),
        _record("beta", incumbent_hit=False, candidate_hit=True, turn=4),
        _record("gamma", incumbent_hit=True, candidate_hit=True, turn=2),
    ]

    summary = summarize(records, min_clusters=3, min_decisions=5)

    assert summary["clusters"] == 3
    assert summary["decisions"] == 5
    assert summary["recall_incumbent"] == 3 / 5
    assert summary["recall_candidate"] == 3 / 5
    assert summary["difference_mean"] == 0.0
    assert summary["sets_differ"] == 4
    assert summary["sets_differ_incumbent_hits"] == 2
    assert summary["sets_differ_candidate_hits"] == 2


def test_posterior_and_skip_fields_pass_through() -> None:
    records = [
        _record(
            "alpha",
            incumbent_hit=True,
            candidate_hit=True,
            posterior_differs_from_prior=True,
            speed_observations=2,
            damage_observations=1,
            sets_differ=False,
        ),
        _record(
            "beta",
            incumbent_hit=True,
            candidate_hit=True,
            posterior_differs_from_prior=False,
            speed_observations=0,
            damage_observations=0,
            sets_differ=False,
        ),
    ]

    summary = summarize(
        records,
        skips={"n_le_10": 3, "exception:SimWorkerError": 1},
        min_clusters=1,
        min_decisions=1,
    )

    assert summary["skips"] == {"n_le_10": 3, "exception:SimWorkerError": 1}
    assert summary["posterior_differs_from_prior"] == 1
    assert summary["decisions_with_speed_or_damage_evidence"] == 1
    assert summary["posterior_differs_with_evidence"] == 1
    assert summary["sets_differ"] == 0
