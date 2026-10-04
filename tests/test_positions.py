"""A saved position must rebuild to the battle the engine actually faced.

Critical because every grade is computed from the rebuilt battle: if the saved bundle did
not reproduce the decision's public state, the exact-search verdict would be about some
other position. Needs the local Showdown engine (no server), hence ``integration``.
"""

from __future__ import annotations

import json

import pytest

from vgc.config import FORMAT_ID, REPO_ROOT
from vgc.mechanics_state import snapshot_battle
from vgc.models import PolicyConfig
from vgc.positions import (
    team_split,
    play_recorded_game,
    rebuild_position,
    sampleable_decisions,
    trimmed_bundle,
    verify_position,
)
from vgc.rl.agents import make_direct_agent
from vgc.rl.env import SimWorker

pytestmark = pytest.mark.integration

TEAMS = REPO_ROOT / "teams" / "owner"


def _active(side_state: dict) -> list[tuple[str, int | None]]:
    return [
        (mon["species_id"], mon["current_hp"])
        for mon in side_state["pokemon"]
        if mon["active"]
    ]


def test_recorded_position_rebuilds_to_the_decision_state():
    ours = (TEAMS / "psyspam_sand.packed.txt").read_text().strip()
    theirs = (TEAMS / "salamence_tw.packed.txt").read_text().strip()
    with SimWorker() as worker:
        game = play_recorded_game(
            worker,
            f"battle-{FORMAT_ID}-424242",
            our_team=ours,
            opp_team=theirs,
            opp_agent=make_direct_agent("vgc", theirs, config=PolicyConfig(format_id=FORMAT_ID)),
            seed=[11, 22, 33, 44],
            seat="p2",  # the less obvious seat: our side is not p1
        )
    indices = sampleable_decisions(game)
    assert indices, "game produced no gradeable decision"
    index = indices[-1]
    cut = trimmed_bundle(game.bundle, index)
    original = game.bundle["decisions"][index]

    # No later information and no engine answer in the stored bundle.
    assert len(cut["decisions"]) == index + 1
    assert cut["decisions"][index]["chosen_order"] is None
    assert len(cut["messages"]) == original["observation_cutoff"]
    json.dumps(cut)  # serialisable as saved

    assert verify_position(cut, index).ready
    battle, memory = rebuild_position(cut, index)
    rebuilt = json.loads(json.dumps(snapshot_battle(battle), default=lambda o: o.__dict__))
    saved = original["state"]
    assert _active(rebuilt["our_side"]) == _active(saved["our_side"])
    assert _active(rebuilt["opponent_side"]) == _active(saved["opponent_side"])
    assert memory is not None


@pytest.mark.parametrize("fraction", [0.3, 0.5])
def test_split_fallback_keys_on_species_not_file_name(fraction, tmp_path):
    base = (TEAMS / "psyspam_sand.packed.txt").read_text().strip()
    # Same six species, different moves/item: a near-copy under another file name.
    near_copy = base.replace("Protect", "Taunt")
    assert near_copy != base
    names = [f"team_{i:03d}.packed.txt" for i in range(40)]
    for name in names:
        assert team_split(name, None, fraction, packed_team=base) == team_split(
            "zzz.packed.txt", tmp_path / "missing.json", fraction, packed_team=near_copy
        )
    # split.json still wins over the fallback.
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"holdout_files": ["a.txt"], "train_files": ["b.txt"]}))
    assert team_split("a.txt", split, packed_team=base) == "test"
    assert team_split("b.txt", split, packed_team=base) == "tune"
