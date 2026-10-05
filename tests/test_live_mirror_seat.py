"""A p2 observation must be searched on a board that matches it.

Until 2026-10-05 the mirror always seated us as p1. For a p2 observation the search then
read a template parser the public patch never updated: ~30% of p2 decisions searched the
wrong active Pokemon and ~24% the wrong weather, and exact-vs-exact A/A games went 63/37
to p1. Nothing failed -- branches still moved -- so only a seat-split A/A exposed it.
"""

from __future__ import annotations

import pytest

from vgc.config import FORMAT_ID, REPO_ROOT
from vgc.mechanics_state import snapshot_battle
from vgc.models import PolicyConfig
from vgc.rl.agents import DirectAgent, make_direct_agent
from vgc.rl.env import DEFAULT_SHOWDOWN_REPO, SimWorker
from vgc.rl.live_mirror import LiveExactMirror, mirror_side
from vgc.rl.match import play_battle


def _board(state):
    return (
        sorted(mon.species_id for mon in state.our_side.pokemon if mon.active),
        sorted(mon.species_id for mon in state.opponent_side.pokemon if mon.active),
        sorted(effect.id for effect in state.weather),
        sorted(effect.id for effect in state.fields),
        sorted(effect.id for effect in state.our_side.side_conditions),
        sorted(effect.id for effect in state.opponent_side.side_conditions),
    )


class _SeatProbe(DirectAgent):
    def __init__(self, inner: DirectAgent, team: str, checked: list) -> None:
        super().__init__(inner.player, name=f"probe-{id(self)}")
        self.team = team
        self.checked = checked

    def choose(self, battle):
        if not battle.teampreview and not any(battle.force_switch) and battle.turn >= 2:
            mirror = LiveExactMirror(self.team, PolicyConfig())
            try:
                root = mirror.build(battle)
                side = mirror_side(battle)
                view = root._decision_battles.get(side, root.battles[side])
                real = _board(snapshot_battle(battle))
                self.checked.append((battle.player_role, _board(snapshot_battle(view)), real))
                root.close()
            finally:
                mirror.close()
        return super().choose(battle)


@pytest.mark.integration
def test_mirror_decision_view_matches_the_observation_from_either_seat() -> None:
    if not DEFAULT_SHOWDOWN_REPO.exists():
        pytest.skip("local Pokemon Showdown checkout is unavailable")
    team = (REPO_ROOT / "teams" / "meta1.packed.txt").read_text().strip()
    checked: list = []
    with SimWorker(DEFAULT_SHOWDOWN_REPO) as worker:
        for game in range(2):
            agents = {
                side: _SeatProbe(
                    make_direct_agent("vgc", team, battle_format=FORMAT_ID), team, checked
                )
                for side in ("p1", "p2")
            }
            play_battle(
                worker, f"seat-{game}", agents, {"p1": team, "p2": team}, seed=[game + 1, 2, 3, 4]
            )

    seats = {role for role, _view, _real in checked}
    assert seats == {"p1", "p2"}
    mismatched = [(role, view, real) for role, view, real in checked if view != real]
    assert not mismatched, mismatched[:3]


@pytest.mark.integration
def test_mirror_keeps_real_account_names_so_a_won_branch_reads_as_won() -> None:
    """Branch parsers copy the observation (real names); `|win|` names the simulator's
    players. With simulator names `p1`/`p2`, a ladder account's winning branch scored
    -10,000. The mirror must start Showdown under the observation's own names."""

    if not DEFAULT_SHOWDOWN_REPO.exists():
        pytest.skip("local Pokemon Showdown checkout is unavailable")
    from vgc.rl.env import DirectBattle

    team = (REPO_ROOT / "teams" / "meta1.packed.txt").read_text().strip()
    with SimWorker(DEFAULT_SHOWDOWN_REPO) as worker:
        source = DirectBattle.start(
            worker, "named", team, team, seed=[1, 2, 3, 4],
            usernames={"p1": "Alice", "p2": "Bob"},
        )
        try:
            source.step({"p1": "team 1234", "p2": "team 1234"})
            for seat in ("p1", "p2"):
                observation = source.battles[seat]
                mirror = LiveExactMirror(team, PolicyConfig())
                try:
                    root = mirror.build(observation)
                    # Our name decides `won`; the opponent's only has to differ from it.
                    assert root.usernames[seat] == {"p1": "Alice", "p2": "Bob"}[seat]
                    assert len(set(root.usernames.values())) == 2
                    root.close()
                finally:
                    mirror.close()
        finally:
            source.close()
