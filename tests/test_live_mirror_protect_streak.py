"""A rebuilt public mirror must keep the odds of a second consecutive Protect.

poke-env counts consecutive successful Protects (1 after one), but Showdown's `stall`
condition stores the odds denominator (3 after one, x3 per repeat). The worker used to
copy poke-env's count straight in, so a repeat Protect on a mirror root succeeded 1 time
in 1, and the exact judge chose back-to-back Protects that failed on the ladder
(2026-10-08 session: repeated failed Protects, including a double failure that lost a game).
"""

from __future__ import annotations

import pytest

from tests.test_live_mirror_branches import COMPACT_CONFIG, _require_showdown_and_pool
from vgc.mechanics_state import snapshot_battle
from vgc.rl.env import DEFAULT_SHOWDOWN_REPO, DirectBattle, SimWorker
from vgc.rl.live_mirror import LiveExactMirror

pytestmark = pytest.mark.integration

# Own team (rain_offense team_11) leads Pelipper + Archaludon; move 4 is Tailwind for
# Pelipper and Protect for Archaludon.
BOTH_MOVE_4 = "move 4, move 4"


def test_mirror_root_keeps_consecutive_protect_odds() -> None:
    own_team, opp_team = _require_showdown_and_pool()
    with SimWorker(DEFAULT_SHOWDOWN_REPO) as worker:
        source = DirectBattle.start(worker, "protect-streak-source", own_team, opp_team,
                                    seed=[1, 2, 3, 4])
        mirror = LiveExactMirror(own_team, COMPACT_CONFIG)
        root = None
        try:
            source.step({"p1": "team 1234", "p2": "team 1234"})
            source.step({"p1": BOTH_MOVE_4, "p2": "default"})
            assert set(source.sides_to_move()) == {"p1", "p2"}, "need a normal turn 2"
            archaludon = next(
                mon for mon in snapshot_battle(source.battles["p1"]).our_side.pokemon
                if mon.species_id == "archaludon"
            )
            assert archaludon.protect_counter == 1

            root = mirror.build(source.battles["p1"])
            successes = 0
            trials = 30
            for i in range(trials):
                clone = root.clone(f"{root.battle_id}-streak-{i}", seed=[i + 1, 7, 11, 13])
                try:
                    result = clone.step({"p1": BOTH_MOVE_4, "p2": "default"})
                    text = "\n".join(result.lines["p1"])
                    if "|-singleturn|p1b: Archaludon|Protect" in text:
                        successes += 1
                finally:
                    clone.close()
            # Real odds are 1 in 3 (P[>= 20 of 30] < 1e-4); the bug gave 30 of 30.
            assert 1 <= successes < 20, f"repeat Protect succeeded {successes}/{trials}"
        finally:
            if root is not None:
                root.close()
            mirror.close()
            source.close()
