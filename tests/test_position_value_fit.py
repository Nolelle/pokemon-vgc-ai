import gzip
import json

import numpy as np

from offline.fit_position_value_weights import _side_features, load, select_replays


def test_compact_load_matches_previous_implementation(tmp_path):
    side = {"active": [], "bench": [], "unseen_count": 6, "side_conditions": []}
    records = []
    for player in ("p1", "p2"):
        for turn, conditions in ((2, ["tailwind", "reflect"]), (1, ["tailwind"]), (3, [])):
            our = {**side, "side_conditions": conditions}
            records.append({"decision_kind": "turn", "outcome": "win", "won": True,
                            "replay_id": "one", "player": player, "turn": turn,
                            "state": {"our": our, "opp": side}})
    path = tmp_path / "records.jsonl.gz"
    with gzip.open(path, "wt") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
    actual = load(path)
    expected_rows = []
    for player in ("p1", "p2"):
        previous = {"our": set(), "opp": set()}
        for record in sorted((r for r in records if r["player"] == player), key=lambda r: r["turn"]):
            features = []
            for key in ("our", "opp"):
                state = record["state"][key]
                conditions = set(state["side_conditions"])
                features.append(_side_features(state, {c: c not in previous[key] for c in conditions}))
                previous[key] = conditions
            expected_rows.append([a - b for a, b in zip(*features)])
    np.testing.assert_array_equal(actual["X"], expected_rows)
    np.testing.assert_array_equal(actual["L"][[1, 2, 4, 5]], actual["X"][[0, 1, 3, 4]])


def test_filters_use_unescaped_names_and_pre_ratings(tmp_path):
    preview = "".join(f"|poke|{side}|Mon{i}\\n" for side in ("p1", "p2") for i in range(6))
    preview = preview.replace("\\n", "\n")
    def write(replay_id, rating):
        log = ("|player|p1|A & B|x|1500\n|player|p2|Other|x|1500\n" + preview
               + "|raw|A &amp; B's rating: 1300 &rarr; <strong>1290</strong>\n" + rating)
        (tmp_path / f"{replay_id}.json").write_text(json.dumps({"log": log}))
    write("one", "|raw|Other's rating: 1240 &rarr; <strong>1250</strong>\n")
    write("two", "")
    write("three", "|raw|Other's rating: 1200 &rarr; <strong>1210</strong>\n")
    selected, counts = select_replays({"one", "two", "three"}, tmp_path, min_rating=1234,
                                      exclude_accounts=set(), team_cap=None, seed=1)
    assert selected == {"one"}
    assert counts["excluded_missing_pre_rating"] == 1
    assert counts["excluded_below_rating"] == 1
    selected, counts = select_replays({"one", "two"}, tmp_path, min_rating=None,
                                      exclude_accounts={"ab"}, team_cap=None, seed=1)
    assert not selected and counts["excluded_accounts"] == 2
    selected, counts = select_replays({"one", "two"}, tmp_path, min_rating=None,
                                      exclude_accounts=set(), team_cap=1, seed=1)
    repeated, _ = select_replays({"two", "one"}, tmp_path, min_rating=None,
                                  exclude_accounts=set(), team_cap=1, seed=1)
    assert selected == repeated and len(selected) == 1 and counts["excluded_team_cap"] == 1


def test_individual_team_cap_checks_both_sides_and_is_reproducible(tmp_path):
    teams = {name: [f"{name}{i}" for i in range(6)] for name in "ABCD"}
    pairs = {"one": ("A", "B"), "two": ("C", "A"), "three": ("D", "B")}
    for game, (first, second) in pairs.items():
        log = "|player|p1|first|x|1500\n|player|p2|second|x|1500\n"
        log += "".join(f"|poke|{side}|{mon}\n" for side, team in
                       (("p1", first), ("p2", second)) for mon in teams[team])
        (tmp_path / f"{game}.json").write_text(json.dumps({"log": log}))
    kwargs = dict(min_rating=None, exclude_accounts=set(), team_cap=None,
                  individual_team_cap=1, seed=42)
    selected, info = select_replays(set(pairs), tmp_path, **kwargs)
    repeated, again = select_replays(set(reversed(list(pairs))), tmp_path, **kwargs)
    assert selected == repeated and info == again
    used = [team for game in selected for team in pairs[game]]
    assert len(used) == len(set(used))
    assert info["excluded_individual_team_cap"] == len(pairs) - len(selected)
    assert info["input_games"] == info["retained_games"] + info["excluded_individual_team_cap"]
