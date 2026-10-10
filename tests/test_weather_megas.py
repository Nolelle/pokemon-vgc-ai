"""Weather-setting Megas in the quick scorer (`vgc.evaluator`): the complete weather-ability
table, the signed weather change, and single-stone Mega timing. Hand-built contexts, no
server (same style as tests/test_evaluator.py, whose helpers are reused)."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from test_evaluator import (
    _attack_ctx,
    _garchomp,
    _klefki,
    _single_for,
)
from vgc.damage import PokemonState
from vgc.data import load_moves
from vgc.evaluator import (
    _ability_weather,
    _Context,
    _score_attack_order,
    _score_status_mega,
    mega_evolved_state,
)
from vgc.models import PolicyConfig
from vgc.weather_abilities import team_weather_scores, weather_adjusted_accuracy


def _team_mon(species: str, ability: str, item: str | None = None, fainted: bool = False):
    return SimpleNamespace(
        species=species, ability=ability, item=item, fainted=fainted, first_turn=False
    )


def _with_team(ctx: _Context, *mons) -> _Context:
    battle = SimpleNamespace(side_conditions=[], team={f"p1: {i}": m for i, m in enumerate(mons)})
    return replace(ctx, battle=battle)


def _tyranitar_ctx(weather: str | None, *team_mons) -> _Context:
    attacker = PokemonState("tyranitar", item="tyranitarite", ability="sandstream")
    ctx = _attack_ctx(
        ally_state=None, opp_state=_klefki(), attacker_state=attacker, weather=weather
    )
    return _with_team(ctx, *team_mons)


_TYRANITAR = _team_mon("tyranitar", "sandstream", "tyranitarite")
_FILLER = _team_mon("garchomp", "roughskin")


# --- 1. complete weather ability table ---------------------------------------------------


def test_ability_weather_table_is_complete_and_legacy_is_not() -> None:
    complete = PolicyConfig()
    legacy = PolicyConfig(weather_abilities_complete=False)
    for ability, weather in (
        ("drought", "sun"),
        ("drizzle", "rain"),
        ("sandstream", "sand"),
        ("snowwarning", "snow"),
    ):
        assert _ability_weather(ability, complete) == weather
    assert _ability_weather("sandstream", legacy) is None
    assert _ability_weather("snowwarning", legacy) is None
    assert _ability_weather("drought", legacy) == "sun"
    assert _ability_weather("intimidate", complete) is None


@pytest.mark.parametrize(
    ("mega_species", "stone", "base"),
    [
        ("tyranitarmega", "tyranitarite", "tyranitar"),
        ("abomasnowmega", "abomasite", "abomasnow"),
        ("froslassmega", "froslassite", "froslass"),
    ],
)
def test_sand_and_snow_megas_change_weather_for_the_quick_scorer(
    mega_species: str, stone: str, base: str
) -> None:
    state = PokemonState(base, item=stone)
    assert mega_evolved_state(state).species_id == mega_species
    ctx = _attack_ctx(ally_state=None, opp_state=_klefki(), attacker_state=state)
    ctx = _with_team(ctx, _team_mon(base, "x", stone), _FILLER)

    _score, info = _score_status_mega(0, ctx, PolicyConfig())
    assert info["mega_weather_change"] is True
    assert info["mega_material"] is True
    legacy = PolicyConfig(weather_abilities_complete=False, mega_single_stone_no_hold=False)
    legacy_score, legacy_info = _score_status_mega(0, ctx, legacy)
    assert legacy_info["mega_weather_change"] is False
    assert legacy_score == -legacy.mega_unnecessary_penalty


def test_attack_order_scores_sand_mega_damage_against_sand_boosted_rock_foe() -> None:
    # The Mega's Sand Stream lands before its attack, so a special hit onto a Rock type is
    # scored against the foe's sand-boosted Sp. Def. Legacy scored it in no weather.
    foe = PokemonState("tyranitar", sp_spread={"hp": 32, "spd": 32}, nature="careful")
    attacker = PokemonState("tyranitar", item="tyranitarite", ability="sandstream")
    ctx = _attack_ctx(ally_state=None, opp_state=foe, attacker_state=attacker)
    ctx = _with_team(ctx, _TYRANITAR, _FILLER)
    move, single = _single_for("darkpulse", mega=True)
    data = load_moves()["darkpulse"]
    new = _score_attack_order(move, data, single, 0, ctx, PolicyConfig())
    old = _score_attack_order(
        move, data, single, 0, ctx, PolicyConfig(weather_abilities_complete=False)
    )
    assert new[1] < old[1]


# --- 3. weather accuracy helper -------------------------------------------------------------


def test_weather_adjusted_accuracy_matches_showdown_rules() -> None:
    assert weather_adjusted_accuracy("hurricane", "rain") == 1.0
    assert weather_adjusted_accuracy("thunder", "rain") == 1.0
    assert weather_adjusted_accuracy("hurricane", "sun") == 0.5
    assert weather_adjusted_accuracy("thunder", "sun") == 0.5
    assert weather_adjusted_accuracy("hurricane", None) == pytest.approx(0.7)
    assert weather_adjusted_accuracy("hurricane", "snow") == pytest.approx(0.7)
    assert weather_adjusted_accuracy("blizzard", "snow") == 1.0
    assert weather_adjusted_accuracy("blizzard", "rain") == pytest.approx(0.7)
    assert weather_adjusted_accuracy("earthquake", "rain") is None
    assert weather_adjusted_accuracy(None, "rain") is None


# --- 4. signed weather change -----------------------------------------------------------------


def test_team_weather_scores_count_setters_benefit_abilities_and_mega_forms() -> None:
    scores = team_weather_scores(
        [
            _team_mon("torkoal", "drought"),
            _team_mon("venusaur", "chlorophyll"),
            _team_mon("charizard", "blaze", "charizarditey"),  # Mega-Y brings Drought
            _team_mon("excadrill", "sandrush", fainted=True),  # fainted: ignored
            None,
        ]
    )
    assert scores == {"sun": 3}


def test_mega_that_replaces_a_preferred_weather_is_harmful_not_material() -> None:
    config = PolicyConfig()
    ctx = _tyranitar_ctx(
        "sun", _TYRANITAR, _team_mon("torkoal", "drought"), _team_mon("venusaur", "chlorophyll")
    )
    score, info = _score_status_mega(0, ctx, config)
    assert info["mega_weather_change"] is True
    assert info["mega_harmful_weather"] is True
    assert info["mega_material"] is False
    assert score == pytest.approx(-config.mega_harmful_weather_penalty, abs=0.01)

    # Legacy control: any weather change is material and unpenalized.
    legacy_score, legacy_info = _score_status_mega(0, ctx, PolicyConfig(mega_weather_signed=False))
    assert legacy_info["mega_material"] is True
    assert legacy_score == 0.0


def test_mega_weather_over_a_weather_our_team_does_not_use_stays_material() -> None:
    # Opposing rain is up; our team has no rain users, so replacing it with sand costs nothing.
    ctx = _tyranitar_ctx("rain", _TYRANITAR, _FILLER)
    score, info = _score_status_mega(0, ctx, PolicyConfig())
    assert info["mega_harmful_weather"] is False
    assert info["mega_material"] is True
    assert score == 0.0


def test_attack_order_applies_harmful_weather_penalty_for_mega() -> None:
    ctx = _tyranitar_ctx(
        "sun", _TYRANITAR, _team_mon("torkoal", "drought"), _team_mon("venusaur", "chlorophyll")
    )
    move, single = _single_for("rockslide", mega=True)
    data = load_moves()["rockslide"]
    signed = _score_attack_order(move, data, single, 0, ctx, PolicyConfig())
    unsigned = _score_attack_order(move, data, single, 0, ctx, PolicyConfig(mega_weather_signed=False))
    assert signed[0] < unsigned[0]


# --- 5. single-stone Mega timing (psyspam_sand) ----------------------------------------------


def test_single_stone_mega_holds_when_its_weather_is_already_up() -> None:
    # Protect on turn 1 with sand already up from base Tyranitar: the Mega would only
    # re-summon active sand, so holding keeps that re-summon in reserve (penalty stays).
    config = PolicyConfig()
    ctx = _tyranitar_ctx("sand", _TYRANITAR, _FILLER)
    score, info = _score_status_mega(0, ctx, config)
    assert info["mega_weather_change"] is False
    assert info["mega_material"] is False
    assert score == -config.mega_unnecessary_penalty


def test_single_stone_mega_evolves_when_its_weather_is_not_up() -> None:
    ctx = _tyranitar_ctx(None, _TYRANITAR, _FILLER)
    score, info = _score_status_mega(0, ctx, PolicyConfig())
    assert info["mega_weather_change"] is True
    assert info["mega_material"] is True
    assert score == 0.0


def test_single_stone_mega_with_no_weather_skips_the_hold_penalty() -> None:
    config = PolicyConfig()
    attacker = _garchomp(item="garchompite", ability="roughskin")
    ctx = _attack_ctx(ally_state=None, opp_state=_klefki(), attacker_state=attacker)
    garchomp = _team_mon("garchomp", "roughskin", "garchompite")
    one = _with_team(ctx, garchomp)
    two = _with_team(ctx, garchomp, _team_mon("charizard", "blaze", "charizarditey"))
    second_fainted = _with_team(
        ctx, garchomp, _team_mon("charizard", "blaze", "charizarditey", fainted=True)
    )
    one_score, _ = _score_status_mega(0, one, config)
    assert one_score > 0.0  # tiny tie-break toward the Mega twin, no penalty
    assert _score_status_mega(0, second_fainted, config)[0] == one_score
    assert _score_status_mega(0, two, config)[0] == -config.mega_unnecessary_penalty
    legacy = PolicyConfig(mega_single_stone_no_hold=False)
    assert _score_status_mega(0, one, legacy)[0] == -legacy.mega_unnecessary_penalty
