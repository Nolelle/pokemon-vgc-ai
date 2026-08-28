"""Unit tests for `vgc.sets` -- opponent state assembly, with a lightweight stand-in for
poke-env's `Pokemon` (see `vgc.sets.ObservedPokemon`'s Protocol) so these don't need a
real battle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from vgc.sets import load_usage_spreads, opponent_spread_hypotheses, opponent_state
from vgc.stats import default_opponent_nature, default_opponent_spread


@dataclass
class _FakePokemon:
    species: str
    item: str | None = None
    ability: str | None = None
    boosts: dict[str, int] = field(default_factory=dict)
    status: Any = None
    current_hp_fraction: float = 1.0


def test_falls_back_to_default_spread_when_no_usage_data() -> None:
    fake = _FakePokemon(species="garchomp", item="focussash", ability="roughskin")
    state = opponent_state(fake, usage={})
    assert state.sp_spread == default_opponent_spread("garchomp")
    assert state.nature == default_opponent_nature("garchomp")
    assert state.item == "focussash"
    assert state.ability == "roughskin"


def test_unknown_item_sentinel_normalized_to_none() -> None:
    fake = _FakePokemon(species="klefki", item="unknown_item")
    state = opponent_state(fake, usage={})
    assert state.item is None


def test_usage_data_overrides_default_when_present() -> None:
    usage = {"garchomp": [{"sp": {"hp": 4, "spe": 32}, "nature": "jolly", "weight": 10}]}
    fake = _FakePokemon(species="garchomp")
    state = opponent_state(fake, usage=usage)
    assert state.sp_spread == {"hp": 4, "spe": 32}
    assert state.nature == "jolly"


def test_usage_data_picks_highest_weight_entry() -> None:
    usage = {
        "garchomp": [
            {"sp": {"atk": 32}, "nature": "adamant", "weight": 1},
            {"sp": {"spe": 32}, "nature": "jolly", "weight": 99},
        ]
    }
    fake = _FakePokemon(species="garchomp")
    state = opponent_state(fake, usage=usage)
    assert state.sp_spread == {"spe": 32}
    assert state.nature == "jolly"


def test_curated_nature_override_wins_over_usage_guess() -> None:
    usage = {"garchomp": [{"sp": {"atk": 32}, "nature": "adamant", "weight": 1}]}
    fake = _FakePokemon(species="garchomp")

    state = opponent_state(fake, usage=usage, nature_override="jolly")

    assert state.sp_spread == {"atk": 32}
    assert state.nature == "jolly"


def test_boosts_filtered_to_damage_relevant_stats() -> None:
    fake = _FakePokemon(species="garchomp", boosts={"atk": 2, "accuracy": 1, "evasion": -1, "def": 0})
    state = opponent_state(fake, usage={})
    assert state.boosts == {"atk": 2}  # accuracy/evasion/zero-value dropped


def test_status_normalized_from_status_like_object() -> None:
    fake = _FakePokemon(species="garchomp", status=SimpleNamespace(name="BRN"))
    state = opponent_state(fake, usage={})
    assert state.status == "brn"


def test_fainted_status_dropped() -> None:
    fake = _FakePokemon(species="garchomp", status=SimpleNamespace(name="FNT"))
    state = opponent_state(fake, usage={})
    assert state.status is None


def test_current_hp_rescaled_from_fraction_not_raw_percent_scale() -> None:
    # poke-env reports opponent HP on a 0-100-ish scale, not real Champions HP -- make
    # sure opponent_state uses current_hp_fraction against OUR computed max_hp, not
    # some raw number from the fake.
    fake = _FakePokemon(species="garchomp", current_hp_fraction=0.5)
    state = opponent_state(fake, usage={})
    assert state.current_hp == round(state.max_hp() * 0.5)


def test_load_usage_spreads_missing_file_returns_empty_dict() -> None:
    load_usage_spreads.cache_clear()
    assert load_usage_spreads("/nonexistent/path/spreads.json") == {}


# --- opponent_spread_hypotheses: the belief distribution behind opponent_state -------
#
# opponent_state answers "what spread do they have?" with one guess. These cover the
# question the exact search actually needs: "what MIGHT they have, and how likely is
# each?" -- see vgc.sets.opponent_spread_hypotheses.

_THREE_SPREADS = {
    "garchomp": [
        {"sp": {"atk": 32, "spe": 32, "hp": 2}, "nature": "jolly", "weight": 6.0},
        {"sp": {"hp": 32, "def": 32, "spd": 2}, "nature": "impish", "weight": 3.0},
        {"sp": {"spa": 32, "spe": 32, "hp": 2}, "nature": "timid", "weight": 1.0},
    ]
}


def test_hypotheses_are_a_normalized_distribution_most_likely_first() -> None:
    hypotheses = opponent_spread_hypotheses("garchomp", _THREE_SPREADS)
    weights = [weight for _sp, _nature, weight in hypotheses]
    assert weights == [0.6, 0.3, 0.1]
    assert sum(weights) == 1.0
    assert [nature for _sp, nature, _w in hypotheses] == ["jolly", "impish", "timid"]


def test_top_hypothesis_matches_what_opponent_state_would_have_picked() -> None:
    # The point estimate must stay the mode of the distribution, or the belief branch
    # and the myopic evaluator would silently disagree about the same opponent.
    fake = _FakePokemon(species="garchomp")
    state = opponent_state(fake, usage=_THREE_SPREADS)
    top_spread, top_nature, _weight = opponent_spread_hypotheses("garchomp", _THREE_SPREADS)[0]
    assert state.sp_spread == top_spread
    assert state.nature == top_nature


def test_limit_keeps_the_most_likely_and_renormalizes() -> None:
    hypotheses = opponent_spread_hypotheses("garchomp", _THREE_SPREADS, limit=2)
    weights = [weight for _sp, _nature, weight in hypotheses]
    assert [nature for _sp, nature, _w in hypotheses] == ["jolly", "impish"]
    # 6 and 3 renormalize to 2/3 and 1/3 -- dropping the tail must not leave the
    # surviving branches summing to less than one.
    assert weights == [2.0 / 3.0, 1.0 / 3.0]


def test_species_without_usage_data_yields_one_certain_default() -> None:
    hypotheses = opponent_spread_hypotheses("garchomp", {})
    assert len(hypotheses) == 1
    spread, nature, weight = hypotheses[0]
    assert weight == 1.0
    assert spread == default_opponent_spread("garchomp")
    assert nature == default_opponent_nature("garchomp")


def test_zero_weight_entries_fall_back_rather_than_dividing_by_zero() -> None:
    usage = {"garchomp": [{"sp": {"hp": 1}, "nature": "brave", "weight": 0.0}]}
    hypotheses = opponent_spread_hypotheses("garchomp", usage)
    assert [weight for _sp, _nature, weight in hypotheses] == [1.0]
    assert hypotheses[0][1] == default_opponent_nature("garchomp")


def test_limit_below_one_is_rejected() -> None:
    import pytest

    with pytest.raises(ValueError, match="at least 1"):
        opponent_spread_hypotheses("garchomp", _THREE_SPREADS, limit=0)


def test_real_usage_data_is_genuinely_uncertain() -> None:
    # Guards the premise of the whole belief branch: if the shipped corpus turned out to
    # name one spread with near-certainty, collapsing to the point estimate would be
    # harmless and this machinery would be dead weight. It does not.
    usage = load_usage_spreads()
    if not usage:  # no corpus committed in this checkout
        return
    confidences = [opponent_spread_hypotheses(species, usage)[0][2] for species in usage]
    assert max(confidences) < 1.0
    assert sum(c <= 0.5 for c in confidences) > len(confidences) // 4
