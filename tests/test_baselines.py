from pathlib import Path

import pytest

from vgc.baselines import BASELINES, make_player
from vgc.config import FORMAT_ID, TEAMS_DIR
from vgc.agent import VgcPlayer


@pytest.mark.parametrize("baseline", sorted(BASELINES))
@pytest.mark.parametrize("accept", [False, True])
def test_all_baselines_honor_explicit_ots_setting(baseline: str, accept: bool) -> None:
    team = Path(TEAMS_DIR / "dev.packed.txt").read_text().strip()

    player = make_player(
        baseline,
        team,
        FORMAT_ID,
        accept_open_team_sheet=accept,
        start_listening=False,
    )

    assert player.accept_open_team_sheet is accept


def test_horizon_baseline_enables_both_planning_gates() -> None:
    team = Path(TEAMS_DIR / "dev.packed.txt").read_text().strip()
    player = make_player("vgc_horizon", team, FORMAT_ID, start_listening=False)

    assert isinstance(player, VgcPlayer)
    assert player.config.use_rolling_horizon is True
    assert player.config.search_diverse_candidates is True
