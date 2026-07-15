"""Registry of baseline players usable as opponents/anchors in offline evals.

Every factory takes the same (team, battle_format, **kwargs) shape so
offline/run_matches.py can build either side of a match from a name string without
knowing which poke-env class backs it.
"""

from __future__ import annotations

from collections.abc import Callable

from poke_env.player.baselines import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer
from poke_env.player.player import Player

from vgc.agent import VgcPlayer
from vgc.config import FORMAT_ID
from vgc.models import PolicyConfig

PlayerFactory = Callable[..., Player]


def _make_random(team: str, battle_format: str, **kwargs) -> Player:
    return RandomPlayer(team=team, battle_format=battle_format, **kwargs)


def _make_maxpower(team: str, battle_format: str, **kwargs) -> Player:
    return MaxBasePowerPlayer(team=team, battle_format=battle_format, **kwargs)


def _make_heuristic(team: str, battle_format: str, **kwargs) -> Player:
    return SimpleHeuristicsPlayer(team=team, battle_format=battle_format, **kwargs)


def _make_vgc(team: str, battle_format: str, **kwargs) -> Player:
    # Phase 2b: VgcPlayer.decide()/decide_teampreview() default to the real heuristic
    # evaluator (vgc.evaluator.score_joint_orders / vgc.team_preview.build_team_order),
    # gated behind PolicyConfig.use_heuristic_evaluator (default True -- see vgc/agent.py).
    config = kwargs.pop("config", None) or PolicyConfig(format_id=battle_format)
    return VgcPlayer(config=config, team=team, battle_format=battle_format, **kwargs)


BASELINES: dict[str, PlayerFactory] = {
    "random": _make_random,
    "maxpower": _make_maxpower,
    "heuristic": _make_heuristic,
    "vgc": _make_vgc,
}


def make_player(
    name: str, team: str, battle_format: str = FORMAT_ID, **kwargs
) -> Player:
    """Construct a registered baseline player by name.

    :param name: one of BASELINES' keys ("random", "maxpower", "heuristic", "vgc").
    :param team: packed or paste-format Showdown team string.
    :param battle_format: Showdown format id; defaults to this project's target format.
    """
    try:
        factory = BASELINES[name]
    except KeyError as exc:
        available = ", ".join(sorted(BASELINES))
        raise ValueError(f"unknown baseline {name!r}; available: {available}") from exc
    return factory(team, battle_format, **kwargs)
