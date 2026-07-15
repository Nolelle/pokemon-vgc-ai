"""Unit tests for `vgc.damage` -- pure math, hand-computed expected values.

See `tests/test_damage_ground_truth.py` (integration) for verification against the real
sim rather than by-hand arithmetic.
"""

from __future__ import annotations

from vgc.damage import FieldState, PokemonState, damage_range, to_id


def test_to_id_normalizes_names() -> None:
    assert to_id("Life Orb") == "lifeorb"
    assert to_id("lifeorb") == "lifeorb"
    assert to_id(None) is None
    assert to_id("") == ""  # PokemonState.__post_init__ normalizes "" -> None itself


def test_status_move_deals_zero_damage_and_is_flagged_supported() -> None:
    attacker = PokemonState("garchomp", sp_spread={"hp": 30, "atk": 4, "spe": 32}, nature="adamant")
    defender = PokemonState("klefki", sp_spread={"hp": 32, "def": 32, "spd": 2}, nature="bold")
    result = damage_range(attacker, defender, "protect", FieldState())
    assert result.min_damage == 0 and result.max_damage == 0
    assert result.breakdown["move_supported"] is True
    assert result.breakdown["reason"] == "status_move"


def test_variable_base_power_move_is_flagged_unsupported() -> None:
    # Flail has no flat basePower in the exported data (relies on a basePowerCallback
    # this module doesn't implement -- see `_variable_base_power`'s explicit move_id
    # list, which does NOT include Flail/Reversal's "lower HP -> higher BP" formula) --
    # must not silently report 0 damage as if that were a real computed answer. (Gyro
    # Ball/Electro Ball/weight-based moves/Water Spout/Eruption/Weather Ball moved to
    # `test_damage_ground_truth.py` once this module started computing real values for
    # them.)
    attacker = PokemonState("garchomp", sp_spread={"hp": 30, "atk": 4, "spe": 32}, nature="adamant")
    defender = PokemonState("klefki", sp_spread={"hp": 32, "def": 32, "spd": 2}, nature="bold")
    result = damage_range(attacker, defender, "flail", FieldState())
    assert result.breakdown["move_supported"] is False
    assert result.min_damage == 0 and result.max_damage == 0


def test_type_immunity_short_circuits_to_zero() -> None:
    # Ghost-type moves never hit Normal-type Pokemon (typechart immune code, not an
    # ability-based immunity).
    attacker = PokemonState("garchomp", sp_spread={"hp": 30, "atk": 4, "spe": 32}, nature="adamant")
    defender = PokemonState("blissey", sp_spread={"hp": 32, "def": 32, "spd": 2}, nature="bold")
    result = damage_range(attacker, defender, "shadowball", FieldState())
    assert result.breakdown["immune"] is True
    assert result.min_damage == 0 and result.max_damage == 0


def test_unhandled_ability_is_flagged_but_causes_no_crash() -> None:
    attacker = PokemonState(
        "garchomp", sp_spread={"hp": 30, "atk": 4, "spe": 32}, nature="adamant", ability="roughskin"
    )
    defender = PokemonState("klefki", sp_spread={"hp": 32, "def": 32, "spd": 2}, nature="bold", ability="prankster")
    result = damage_range(attacker, defender, "earthquake", FieldState(is_doubles=False))
    assert result.breakdown["ability_handled"] is False
    # Graceful default: no modifier applied for the unhandled ability, not an error.
    assert result.min_damage > 0


def test_unhandled_item_is_flagged() -> None:
    attacker = PokemonState(
        "garchomp", sp_spread={"hp": 30, "atk": 4, "spe": 32}, nature="adamant", item="leftovers"
    )
    defender = PokemonState("klefki", sp_spread={"hp": 32, "def": 32, "spd": 2}, nature="bold")
    result = damage_range(attacker, defender, "earthquake", FieldState(is_doubles=False))
    assert result.breakdown["item_handled"] is False


def test_choice_scarf_is_handled_and_damage_neutral() -> None:
    attacker_no_item = PokemonState(
        "garchomp", sp_spread={"hp": 30, "atk": 4, "spe": 32}, nature="adamant"
    )
    attacker_scarf = PokemonState(
        "garchomp", sp_spread={"hp": 30, "atk": 4, "spe": 32}, nature="adamant", item="choicescarf"
    )
    defender = PokemonState("klefki", sp_spread={"hp": 32, "def": 32, "spd": 2}, nature="bold")
    field = FieldState(is_doubles=False)
    plain = damage_range(attacker_no_item, defender, "earthquake", field)
    scarfed = damage_range(attacker_scarf, defender, "earthquake", field)
    assert scarfed.breakdown["item_handled"] is True
    # Choice Scarf affects Speed only -- no damage-formula effect.
    assert plain.min_damage == scarfed.min_damage
    assert plain.max_damage == scarfed.max_damage


def test_rolls_are_monotonic_and_bracket_expected() -> None:
    attacker = PokemonState("garchomp", sp_spread={"hp": 30, "atk": 4, "spe": 32}, nature="adamant")
    defender = PokemonState("klefki", sp_spread={"hp": 32, "def": 32, "spd": 2}, nature="bold")
    result = damage_range(attacker, defender, "earthquake", FieldState(is_doubles=False))
    rolls = result.breakdown["rolls"]
    assert len(rolls) == 16
    assert rolls == sorted(rolls)
    assert rolls[0] == result.min_damage
    assert rolls[-1] == result.max_damage
    assert result.min_damage <= result.expected_damage <= result.max_damage


def test_percent_fields_match_max_hp() -> None:
    attacker = PokemonState("garchomp", sp_spread={"hp": 30, "atk": 4, "spe": 32}, nature="adamant")
    defender = PokemonState("klefki", sp_spread={"hp": 32, "def": 32, "spd": 2}, nature="bold")
    result = damage_range(attacker, defender, "earthquake", FieldState(is_doubles=False))
    max_hp = defender.max_hp()
    assert result.min_percent == result.min_damage / max_hp * 100
    assert result.max_percent == result.max_damage / max_hp * 100


def test_mega_forme_uses_mega_species_data() -> None:
    # garchompmega has different base stats/ability/types than the base garchomp entry
    # (see data/champions/species.json) -- calculate_stats/PokemonState must use the
    # mega's own entry, not the base forme's.
    base = PokemonState("garchomp", sp_spread={"hp": 30, "atk": 4, "spe": 32}, nature="adamant")
    mega = PokemonState("garchompmega", sp_spread={"hp": 30, "atk": 4, "spe": 32}, nature="adamant")
    assert base.stats()["atk"] != mega.stats()["atk"]
    assert mega.species_data()["abilities"]["0"] == "Sand Force"


def test_default_spread_used_when_none_given() -> None:
    # An opponent PokemonState with no sp_spread should fall back to
    # vgc.stats.default_opponent_spread rather than erroring.
    from vgc.stats import default_opponent_spread

    opponent = PokemonState("garchomp")
    assert opponent.stats()["hp"] == 108 + default_opponent_spread("garchomp")["hp"] + 75


# --- variable base power moves (see `_variable_base_power`) ----------------------------


def test_weather_ball_doubles_power_and_changes_type_by_weather() -> None:
    attacker = PokemonState("charizardmegay", sp_spread={"hp": 32, "spa": 32, "spe": 2}, nature="modest")
    # Water-type defender: Normal (no weather) is neutral (1x), Fire (sun) is resisted
    # (0.5x) -- a cheap way to confirm the move's type actually changed, not just its BP.
    defender = PokemonState("feraligatr", sp_spread={"hp": 32, "spd": 32, "def": 2}, nature="calm")

    no_weather = damage_range(attacker, defender, "weatherball", FieldState(weather=None, is_doubles=False))
    assert no_weather.breakdown["base_power"] == 50
    assert no_weather.breakdown["type_effectiveness"] == 1.0  # Normal vs. Water

    sun = damage_range(attacker, defender, "weatherball", FieldState(weather="sun", is_doubles=False))
    assert sun.breakdown["base_power"] == 100
    assert sun.breakdown["type_effectiveness"] == 0.5  # Fire vs. Water
    assert sun.breakdown["stab"] == 1.5  # Charizard-Y is Fire-type, so Weather Ball's new type is STAB
    assert sun.breakdown["weather_modifier"] == 1.5  # sun boosts the now-Fire-type move

    rain = damage_range(attacker, defender, "weatherball", FieldState(weather="rain", is_doubles=False))
    assert rain.breakdown["base_power"] == 100
    assert rain.breakdown["type_effectiveness"] == 0.5  # Water vs. Water is resisted
    assert rain.breakdown["stab"] == 1.0  # not Water-type
    assert rain.breakdown["weather_modifier"] == 1.5  # rain boosts the now-Water-type move

    sand = damage_range(attacker, defender, "weatherball", FieldState(weather="sand", is_doubles=False))
    assert sand.breakdown["base_power"] == 100  # still doubles, even though sand has no damage multiplier
    assert sand.breakdown["type_effectiveness"] == 1.0  # Rock vs. Water is neutral
    assert sand.breakdown["weather_modifier"] == 1.0  # no Fire/Water weather boost for Rock


def test_water_spout_scales_with_attacker_hp_fraction() -> None:
    attacker_full = PokemonState("feraligatr", sp_spread={"hp": 32, "spa": 32, "spd": 2}, nature="modest")
    max_hp = attacker_full.max_hp()
    attacker_half = PokemonState(
        "feraligatr", sp_spread={"hp": 32, "spa": 32, "spd": 2}, nature="modest", current_hp=max_hp // 2
    )
    defender = PokemonState("klefki", sp_spread={"hp": 32, "def": 32, "spd": 2}, nature="bold")
    field = FieldState(is_doubles=False)

    full_result = damage_range(attacker_full, defender, "waterspout", field)
    assert full_result.breakdown["base_power"] == 150

    half_result = damage_range(attacker_half, defender, "waterspout", field)
    assert half_result.breakdown["base_power"] == (150 * (max_hp // 2)) // max_hp
    assert half_result.breakdown["base_power"] < full_result.breakdown["base_power"]


def test_electro_ball_power_by_speed_ratio() -> None:
    # Feraligatr (Water, base 78 Spe) as the fast attacker, Torkoal (Fire, base 20 Spe)
    # as the slow defender -- Water isn't Electric-immune (unlike a Ground-type, which
    # would short-circuit the whole calculation to `immune` before BP is ever computed).
    fast_attacker = PokemonState("feraligatr", sp_spread={"hp": 32, "spa": 32, "spe": 2}, nature="modest")
    slow_defender = PokemonState("torkoal", sp_spread={"hp": 32, "spd": 32, "spe": 2}, nature="bold")
    field = FieldState(is_doubles=False)
    fast_result = damage_range(fast_attacker, slow_defender, "electroball", field)
    slow_result = damage_range(slow_defender, fast_attacker, "electroball", field)
    assert fast_result.breakdown["base_power"] > slow_result.breakdown["base_power"]
    assert fast_result.breakdown["base_power"] in (40, 60, 80, 120, 150)
    assert slow_result.breakdown["base_power"] == 40  # ratio < 1 -> the lowest bracket


def test_gyro_ball_power_by_inverse_speed_ratio() -> None:
    # Slow attacker, fast defender -> Gyro Ball rewards being slower.
    slow_attacker = PokemonState("klefki", sp_spread={"hp": 32, "atk": 32, "def": 2}, nature="brave")
    fast_defender = PokemonState("garchomp", sp_spread={"hp": 30, "spe": 32}, nature="jolly")
    field = FieldState(is_doubles=False)
    slow_result = damage_range(slow_attacker, fast_defender, "gyroball", field)
    fast_result = damage_range(fast_defender, slow_attacker, "gyroball", field)
    assert slow_result.breakdown["base_power"] > fast_result.breakdown["base_power"]
    assert slow_result.breakdown["base_power"] <= 150


def test_grass_knot_power_by_target_weight() -> None:
    # Feraligatr (88.8kg = 888 hg) lands in the [500, 1000) hg tier -> 80 BP.
    attacker = PokemonState("garchomp", sp_spread={"hp": 30, "spa": 32, "spe": 4}, nature="modest")
    defender = PokemonState("feraligatr", sp_spread={"hp": 32, "def": 32, "spd": 2}, nature="bold")
    result = damage_range(attacker, defender, "grassknot", FieldState(is_doubles=False))
    assert result.breakdown["base_power"] == 80
    # Klefki (3kg = 30 hg) is under the lowest 100 hg threshold -> 20 BP.
    light_defender = PokemonState("klefki", sp_spread={"hp": 32, "def": 32, "spd": 2}, nature="bold")
    light_result = damage_range(attacker, light_defender, "grassknot", FieldState(is_doubles=False))
    assert light_result.breakdown["base_power"] == 20


def test_heavy_slam_power_by_weight_ratio() -> None:
    # Garchomp (95kg = 950 hg) vs. Klefki (3kg = 30 hg): ratio ~31.7 -> the top 120 BP tier.
    attacker = PokemonState("garchomp", sp_spread={"hp": 30, "atk": 4, "spe": 32}, nature="adamant")
    defender = PokemonState("klefki", sp_spread={"hp": 32, "def": 32, "spd": 2}, nature="bold")
    result = damage_range(attacker, defender, "heavyslam", FieldState(is_doubles=False))
    assert result.breakdown["base_power"] == 120
    # Reversed matchup: Klefki vs. Garchomp -> ratio << 1 -> the bottom 40 BP tier.
    reversed_result = damage_range(defender, attacker, "heavyslam", FieldState(is_doubles=False))
    assert reversed_result.breakdown["base_power"] == 40


def test_boost_stages_are_clamped_to_plus_minus_six() -> None:
    attacker_neutral = PokemonState("garchomp", sp_spread={"hp": 30, "atk": 4, "spe": 32}, nature="adamant")
    attacker_overboosted = PokemonState(
        "garchomp", sp_spread={"hp": 30, "atk": 4, "spe": 32}, nature="adamant", boosts={"atk": 10}
    )
    attacker_at_cap = PokemonState(
        "garchomp", sp_spread={"hp": 30, "atk": 4, "spe": 32}, nature="adamant", boosts={"atk": 6}
    )
    defender = PokemonState("klefki", sp_spread={"hp": 32, "def": 32, "spd": 2}, nature="bold")
    field = FieldState(is_doubles=False)
    overboosted = damage_range(attacker_overboosted, defender, "earthquake", field)
    at_cap = damage_range(attacker_at_cap, defender, "earthquake", field)
    assert overboosted.min_damage == at_cap.min_damage
    assert overboosted.max_damage == at_cap.max_damage
    assert overboosted.min_damage > damage_range(attacker_neutral, defender, "earthquake", field).min_damage
