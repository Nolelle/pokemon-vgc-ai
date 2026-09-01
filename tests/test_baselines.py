from pathlib import Path

import pytest

from vgc.baselines import BASELINES, make_learned_player, make_player
from vgc.config import FORMAT_ID, TEAMS_DIR
from vgc.agent import VgcPlayer
from vgc.models import PolicyConfig


def test_default_policy_knows_own_team_but_rejects_opponent_sheet() -> None:
    config = PolicyConfig()

    assert config.use_own_team_spreads is True
    assert config.accept_open_team_sheet is False


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


def test_capability_ladder_rungs_are_strictly_nested() -> None:
    """Random < maxpower < heuristic are poke-env; the vgc rungs nest by search depth."""

    team = Path(TEAMS_DIR / "dev.packed.txt").read_text().strip()
    myopic = make_player("vgc_myopic", team, FORMAT_ID, start_listening=False)
    shallow = make_player("vgc_shallow", team, FORMAT_ID, start_listening=False)
    full = make_player("vgc", team, FORMAT_ID, start_listening=False)

    assert myopic.config.use_two_ply_search is False
    assert myopic.config.use_rolling_horizon is False
    assert shallow.config.use_two_ply_search is True
    assert shallow.config.use_rolling_horizon is False
    assert full.config.use_two_ply_search is True
    assert full.config.use_rolling_horizon is True


def test_learned_player_is_explicit_checkpoint_opt_in(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from vgc.rl.model import CandidatePolicyValueNet
    from vgc.rl.opponents import save_snapshot
    from vgc.rl.player import PpoVgcPlayer

    checkpoint = save_snapshot(
        tmp_path,
        CandidatePolicyValueNet(
            use_information_features=True,
            use_tactical_features=True,
            head_dropout=0.2,
        ),
        generation=0,
        max_snapshots=1,
    )
    team = Path(TEAMS_DIR / "dev.packed.txt").read_text().strip()
    player = make_learned_player(
        checkpoint,
        team,
        FORMAT_ID,
        start_listening=False,
    )

    assert isinstance(player, PpoVgcPlayer)
    assert player.deterministic is True
    assert player.model.use_information_features is True
    assert player.model.use_tactical_features is True
    assert player.model.head_dropout_p == pytest.approx(0.2)
    assert "learned" not in BASELINES
