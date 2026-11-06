"""Critical split, seat, and influence boundaries of replay sampling."""
import pytest

from vgc.bc.replay_sampling import build_sampling_plan, sample_task_units


def game(a, b, *, split="train", ratings=None):
    return {"split": split, "rosters": {"p1": [a], "p2": [b]},
            "ratings": ratings or {"p1": 1700, "p2": 1400},
            "accounts": {"p1": a, "p2": b}}


def fixture():
    games = {"a": game("common", "other"), "b": game("common", "third"),
             "c": game("rare", "rarest")}
    counts = {(rid, side, "move"): 2 for rid in games for side in ("p1", "p2")}
    return games, counts


def masses(plan):
    result = {}
    for (rid, side, task), weight in plan.per_unit_weights.items():
        result[rid] = result.get(rid, 0) + weight * plan.group_counts[(rid, side, task)]
    return result


def test_heldout_additions_cannot_change_plan():
    games, counts = fixture()
    original = build_sampling_plan(games, counts)
    games["held"] = game("common", "common", split="test")
    counts[("held", "p1", "move")] = 100000
    assert build_sampling_plan(games, counts) == original
    draws = sample_task_units(original, "move", 100, 42)
    assert draws == sample_task_units(original, "move", 100, 42)
    assert all(rid != "held" and 0 <= ordinal < 2 for rid, _, _, ordinal in draws)


def test_unit_duplication_preserves_game_mass_and_normalization():
    games, counts = fixture()
    original = build_sampling_plan(games, counts)
    counts[("a", "p1", "move")] *= 20
    changed = build_sampling_plan(games, counts)
    assert masses(changed) == pytest.approx(masses(original))
    assert sum(masses(changed).values()) == pytest.approx(1)
    assert masses(changed)["c"] > masses(changed)["a"]
    assert changed.diagnostics["move"]["max_team_participation_after"] < 2 / 3


def test_expert_is_acting_seat_and_unknown_excluded():
    games, counts = fixture()
    games["b"]["ratings"]["p1"] = None
    plan = build_sampling_plan(games, counts, phase="expert")
    assert set(plan.group_counts) == {("a", "p1", "move"), ("c", "p1", "move")}
    assert ("third",) not in plan.roster_frequencies
    assert sum(masses(plan).values()) == pytest.approx(1)


def test_mirrors_count_roster_once_and_opponent_controls_weight():
    games = {"a": game("common", "common"), "b": game("rare", "common"),
             "c": game("other", "last")}
    counts = {(rid, "p1", "move"): 1 for rid in games}
    plan = build_sampling_plan(games, counts)
    assert plan.roster_frequencies[("common",)] == 2
    assert masses(plan)["a"] == masses(plan)["b"]
    assert masses(plan)["b"] < masses(plan)["c"]


def test_empty_and_invalid_requests():
    plan = build_sampling_plan({}, {})
    assert sample_task_units(plan, "move", 0, 1) == []
    with pytest.raises(ValueError):
        sample_task_units(plan, "move", 1, 1)
    with pytest.raises(ValueError):
        build_sampling_plan({}, {}, phase="invalid")


def test_per_task_eligibility():
    """1100-rated side contributes outcome units but no policy units; a showteam
    game contributes outcome only; expert unchanged."""
    from vgc.bc.replay_sampling import _eligible
    game = {"ratings": {"p1": 1100.0, "p2": 1750.0}, "showteam": False,
            "split": "train"}
    assert _eligible(game, "p1", "outcome", "broad", 1200, float("-inf"))
    assert not _eligible(game, "p1", "move", "broad", 1200, float("-inf"))
    assert not _eligible(game, "p1", "preview_brought", "broad", 1200, float("-inf"))
    assert _eligible(game, "p2", "move", "broad", 1200, float("-inf"))
    sheet = dict(game, showteam=True)
    assert not _eligible(sheet, "p2", "move", "broad", 1200, float("-inf"))
    assert not _eligible(sheet, "p2", "preview_leads", "broad", 1200, float("-inf"))
    assert _eligible(sheet, "p2", "outcome", "broad", 1200, float("-inf"))
    assert _eligible(game, "p2", "move", "expert") and not _eligible(game, "p1", "move", "expert")


def test_fixed_boundary_splits():
    """time < DEV_START -> train; < TEST_START -> validation; <= WINDOW_END ->
    test; after -> excluded with 'after_window'."""
    import sys
    sys.path.insert(0, "tools")
    from inventory_replay_pretraining import assign_splits  # noqa: PLC0415
    games = [{"id": f"g{i}", "time": t, "exclusions": [], "rosters": {},
              "log_sha256": f"h{i}"}
             for i, t in enumerate([100, 200, 300, 400, 500])]
    assign_splits(games, {"dev_start": 300, "test_start": 400, "window_end": 450})
    assert [g["base_split"] for g in games] == [
        "train", "train", "validation", "test", "excluded"]
    assert "after_window" in games[4]["exclusions"]
