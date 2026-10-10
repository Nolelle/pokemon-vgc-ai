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


# --- Psych Up / Costar: `-copyboost|RECEIVER|SOURCE` --------------------------------------


def _battle_with_boosted_slowbro() -> DoubleBattle:
    battle = DoubleBattle("battle-test", "p1user", logging.getLogger("test"), gen=9)
    battle.player_role = "p1"
    for line in (
        "|player|p1|p1user||",
        "|player|p2|p2user||",
        "|switch|p1a: Overqwil|Overqwil, L50, F|100/100",
        "|switch|p2a: Slowbro|Slowbro, L50, M|100/100",
        "|-boost|p2a: Slowbro|evasion|1",
        "|-boost|p2a: Slowbro|def|2",
        "|-boost|p1a: Overqwil|atk|1",
    ):
        battle.parse_message(line.split("|"))
    return battle


def test_psych_up_copies_target_boosts_onto_the_user() -> None:
    """Real shape from ladder game 2695881082: Showdown names the user first."""
    line = "|-copyboost|p1a: Overqwil|p2a: Slowbro|[from] move: Psych Up".split("|")
    battle = _battle_with_boosted_slowbro()
    battle.parse_message(normalize_for_poke_env(list(line)))
    user = battle.get_pokemon("p1a: Overqwil")
    target = battle.get_pokemon("p2a: Slowbro")
    assert (user.boosts["evasion"], user.boosts["def"], user.boosts["atk"]) == (1, 2, 0)
    assert (target.boosts["evasion"], target.boosts["def"]) == (1, 2)  # untouched


def test_poke_env_reads_copyboost_backwards_without_the_rewrite() -> None:
    line = "|-copyboost|p1a: Overqwil|p2a: Slowbro|[from] move: Psych Up".split("|")
    battle = _battle_with_boosted_slowbro()
    battle.parse_message(list(line))
    assert battle.get_pokemon("p2a: Slowbro").boosts["evasion"] == 0  # the bug
    assert battle.get_pokemon("p1a: Overqwil").boosts["evasion"] == 0


def test_costar_copies_the_ally_onto_the_user() -> None:
    battle = DoubleBattle("battle-test", "p1user", logging.getLogger("test"), gen=9)
    battle.player_role = "p1"
    for line in (
        "|player|p1|p1user||",
        "|player|p2|p2user||",
        "|switch|p1a: Flamigo|Flamigo, L50, M|100/100",
        "|switch|p1b: Dondozo|Dondozo, L50, M|100/100",
        "|switch|p2a: Slowbro|Slowbro, L50, M|100/100",
        "|-boost|p1a: Flamigo|spa|2",
        "|-copyboost|p1b: Dondozo|p1a: Flamigo|[from] ability: Costar",
    ):
        battle.parse_message(normalize_for_poke_env(line.split("|")))
    assert battle.get_pokemon("p1b: Dondozo").boosts["spa"] == 2
    assert battle.get_pokemon("p1a: Flamigo").boosts["spa"] == 2
