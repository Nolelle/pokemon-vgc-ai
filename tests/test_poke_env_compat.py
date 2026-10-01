"""poke-env 0.15 crashes on Showdown's Round-chain line unless it is normalized."""

from __future__ import annotations

import logging

import pytest
from poke_env.battle import DoubleBattle

from vgc.poke_env_compat import normalize_for_poke_env

# Real line from M-C replay corpus: Sylveon's Round moved up behind its ally's Round.
ROUND_CHAIN = "|move|p1b: Sylveon|Round|p2a: Hawlucha|[from] move: Round".split("|")


def _battle_seeing_unrevealed_sylveon() -> DoubleBattle:
    battle = DoubleBattle("battle-test", "p2user", logging.getLogger("test"), gen=9)
    battle.player_role = "p2"
    battle.parse_message("|player|p1|p1user||".split("|"))
    battle.parse_message("|player|p2|p2user||".split("|"))
    battle.parse_message("|switch|p1b: Sylveon|Sylveon, L50, F|100/100".split("|"))
    battle.parse_message("|switch|p2a: Hawlucha|Hawlucha, L50, M|100/100".split("|"))
    return battle


def test_poke_env_raises_on_raw_round_chain() -> None:
    with pytest.raises(KeyError):
        _battle_seeing_unrevealed_sylveon().parse_message(list(ROUND_CHAIN))


def test_normalized_round_chain_parses_and_reveals_round() -> None:
    battle = _battle_seeing_unrevealed_sylveon()
    battle.parse_message(normalize_for_poke_env(list(ROUND_CHAIN)))
    sylveon = battle.get_pokemon("p1b: Sylveon")
    assert "round" in sylveon.moves


def test_other_lines_are_untouched() -> None:
    line = "|move|p1a: Indeedee|Round|p2a: Hawlucha".split("|")
    assert normalize_for_poke_env(line) is line
    copycat = "|move|p1a: Indeedee|Round|p2a: Hawlucha|[from] move: Copycat".split("|")
    assert normalize_for_poke_env(copycat) is copycat
