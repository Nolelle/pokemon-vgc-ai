import json

from tools.inventory_replay_pretraining import (
    assign_splits,
    outcome_candidate,
    outcome_matches,
    preview_complete,
    raw_metadata,
    reserve_teams,
    summarize_row,
)


def test_ratings_use_changes_and_conflicts_are_unknown(tmp_path):
    path = tmp_path / "x.json"
    path.write_text(json.dumps({"id": "x", "log": "\n".join([
        "|player|p2|",
        "|player|p1|A &amp; B||1900", "|player|p2|C||2000",
        "|raw|A &amp; B's rating: 1500 &rarr; <strong>1510</strong>",
        "|raw|A &amp; B's rating: 1501 &rarr; <strong>1510</strong>",
        "|raw|C's rating: 1600 &rarr; <strong>1590</strong>",
        "|-message|C lost due to inactivity.", "|win|A &amp; B",
    ])}))
    result = raw_metadata(path)
    assert result["ratings"] == {"p1": None, "p2": 1600}
    assert result["ending"] == "timer_offline"
    assert result["winner"] == "p1"


def test_ties_duplicates_and_both_role_reservation():
    games = [{"id": str(i), "time": i // 2, "log_sha256": str(i), "exclusions": [],
              "rosters": {"p1": [f"a{i}"], "p2": [f"b{i}"]}} for i in range(20)]
    games[1]["log_sha256"] = games[0]["log_sha256"]
    assign_splits(games)
    assert games[0]["split"] == games[1]["split"] == "excluded"
    assert games[16]["split"] == games[17]["split"]
    test = next(g for g in games if g["split"] == "test")
    for g in games:
        if g["split"] == "test":
            g["rosters"] = {"p1": ["reserved"], "p2": ["reserved"]}
    earlier = [g for g in games if g["split"] == "train"][:2]
    earlier[0]["rosters"]["p1"] = ["reserved"]
    earlier[1]["rosters"]["p2"] = ["reserved"]
    reserved = reserve_teams(games)
    assert reserved == {("reserved",)}
    assert all(g["split"] == "reserved_team_excluded" for g in earlier)
    assert test["team_challenge"]


def test_preview_and_unknown_are_not_complete_actions():
    row = {"decision_kind": "teampreview", "state": {"our": {"preview": list("abcdef")}},
           "action": {"picked": list("abc"), "lead_order": list("ab")}}
    assert not preview_complete(row)
    row["action"]["picked"] = list("abcd")
    assert preview_complete(row)
    counts = summarize_row({"decision_kind": "turn", "state": {},
                            "action_status": {"slot0": "unknown", "slot1": "no_action_required"},
                            "action": {"slot0": {"kind": "pass"}}})
    assert counts["slot_unknown"] == 1
    assert counts["bad_state_rows"] == 1
    assert counts["observed_move"] == counts["observed_switch"] == 0


def test_outcome_mismatch_is_not_verified():
    game = {"winner": "p1", "outcomes": {"p1": {("loss", False)}}}
    assert not outcome_matches(game, "p1")
    game["outcomes"]["p1"] = {("win", True)}
    assert outcome_matches(game, "p1")


def test_timer_and_mismatch_are_never_outcome_candidates():
    game = {"ending": "timer_offline", "exclusions": []}
    assert not outcome_candidate(game, "train")
    game["ending"] = "played_out"
    game["exclusions"] = ["outcome_mismatch_or_unresolved"]
    assert not outcome_candidate(game, "train")
    game["exclusions"] = []
    assert outcome_candidate(game, "train")
