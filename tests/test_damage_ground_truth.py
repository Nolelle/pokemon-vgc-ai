"""Ground-truth verification of `vgc.damage.damage_range` against the real sim.

Each case drives a minimal `[Gen 9 Champions] (Doubles) Custom Game` battle via
`tools/sim_probe.mjs`, reads the actual damage dealt from the `|-damage|` protocol line,
and asserts it falls within `damage_range`'s `[min_damage, max_damage]` -- the random roll
means we can't assert an exact number, containment IS the point (see module docstring in
`src/vgc/damage.py` and the Phase 2a task spec).

Uses a small set of canonical species throughout (Garchomp, Klefki, Torkoal, Scizor,
Feraligatr, Dragonite, Azumarill) and leans hard on the fact that `Custom Game` doesn't
validate move/item/ability legality (see `tests/sim_harness.py`'s docstring) to force
whatever ability/item/move isolates one mechanic at a time onto a convenient body, mixing
species/abilities that would never legally co-occur on the ladder. That's intentional and
matches how a damage calculator is normally exercised ("what if this had Tinted Lens") --
it is NOT a claim that these combinations are legal teams.
"""

from __future__ import annotations

import pytest

from vgc.damage import DamageResult, FieldState, PokemonState, damage_range

from sim_harness import first_damage_delta, pokeset, run_scenario

pytestmark = pytest.mark.integration

SINGLES = "gen9championscustomgame"
DOUBLES = "gen9championsdoublescustomgame"


def assert_actual_in_range(result: DamageResult, actual: int) -> None:
    assert result.min_damage <= actual <= result.max_damage, (
        f"actual damage {actual} not in [{result.min_damage}, {result.max_damage}]; "
        f"breakdown={result.breakdown}"
    )


def _run_single_hit(
    attacker_set: dict,
    defender_set: dict,
    move: str,
    setup_evals: tuple[str, ...] = (),
    seed: tuple[int, int, int, int] = (1, 2, 3, 4),
) -> tuple[list[str], int]:
    """Minimal singles battle: team preview, optional `>eval` setup, attacker uses
    `move`, defender uses a harmless `splash`. Returns (log, actual damage on p2a).
    """
    commands = [
        "p1 team 1",
        "p2 team 1",
        *setup_evals,
        f"p1 move {move}",
        "p2 move splash",
    ]
    result = run_scenario(SINGLES, [attacker_set], [defender_set], commands, seed=seed)
    actual = first_damage_delta(result["log"], "p2a")
    return result["log"], actual


def _run_single_hit_expect_immune(
    attacker_set: dict,
    defender_set: dict,
    move: str,
    setup_evals: tuple[str, ...] = (),
    seed: tuple[int, int, int, int] = (1, 2, 3, 4),
) -> list[str]:
    """Like `_run_single_hit`, but for cases expected to be a total immunity -- no
    `|-damage|` line is emitted at all, so this just returns the log for the caller to
    assert `-immune` appears in (rather than trying to compute a damage delta of 0 from
    a `-damage` line that will never exist).
    """
    commands = [
        "p1 team 1",
        "p2 team 1",
        *setup_evals,
        f"p1 move {move}",
        "p2 move splash",
    ]
    result = run_scenario(SINGLES, [attacker_set], [defender_set], commands, seed=seed)
    return result["log"]


# --- physical / special, STAB / not, effectiveness ---------------------------------------


def test_physical_stab_super_effective() -> None:
    # Ground (STAB) vs Steel/Fairy: Steel is 2x weak to Ground, Fairy neutral -> 2x.
    attacker_set = pokeset(
        "Garchomp", ["earthquake"], nature="Adamant", sp={"hp": 30, "atk": 4, "spe": 32}, ability="Rough Skin"
    )
    defender_set = pokeset("Klefki", ["splash"], nature="Bold", sp={"hp": 32, "def": 32, "spd": 2})
    _, actual = _run_single_hit(attacker_set, defender_set, "earthquake")

    attacker = PokemonState("garchomp", sp_spread={"hp": 30, "atk": 4, "spe": 32}, nature="adamant")
    defender = PokemonState("klefki", sp_spread={"hp": 32, "def": 32, "spd": 2}, nature="bold")
    result = damage_range(attacker, defender, "earthquake", FieldState(is_doubles=False))
    assert result.breakdown["type_effectiveness"] == 2.0
    assert result.breakdown["stab"] == 1.5
    assert_actual_in_range(result, actual)


def test_physical_not_stab_resisted() -> None:
    # Normal (not STAB for Garchomp) vs Steel/Fairy: Steel resists Normal 0.5x -> 0.5x.
    attacker_set = pokeset(
        "Garchomp", ["bodyslam"], nature="Adamant", sp={"hp": 30, "atk": 4, "spe": 32}, ability="Rough Skin"
    )
    defender_set = pokeset("Klefki", ["splash"], nature="Bold", sp={"hp": 32, "def": 32, "spd": 2})
    _, actual = _run_single_hit(attacker_set, defender_set, "bodyslam")

    attacker = PokemonState("garchomp", sp_spread={"hp": 30, "atk": 4, "spe": 32}, nature="adamant")
    defender = PokemonState("klefki", sp_spread={"hp": 32, "def": 32, "spd": 2}, nature="bold")
    result = damage_range(attacker, defender, "bodyslam", FieldState(is_doubles=False))
    assert result.breakdown["type_effectiveness"] == 0.5
    assert result.breakdown["stab"] == 1.0
    assert_actual_in_range(result, actual)


def test_special_stab_double_super_effective() -> None:
    # Fire (STAB) vs Bug/Steel: both 2x weak to Fire -> 4x. White Smoke (not Drought) so
    # no incidental weather. Ember (40bp), not Flamethrower -- a 4x-effective STAB
    # Flamethrower one-shots even a maximally SpD-invested Scizor (its 177 max HP is
    # fixed, same Stat Points ceiling issue as test_life_orb_boosts_damage), which would
    # turn the `|-damage|` line into an uninformative `0 fnt`.
    attacker_set = pokeset(
        "Torkoal", ["ember"], nature="Modest", sp={"hp": 32, "spa": 32, "spd": 2}, ability="White Smoke"
    )
    defender_set = pokeset("Scizor", ["splash"], nature="Calm", sp={"hp": 32, "spd": 32, "def": 2})
    _, actual = _run_single_hit(attacker_set, defender_set, "ember")

    attacker = PokemonState("torkoal", sp_spread={"hp": 32, "spa": 32, "spd": 2}, nature="modest")
    defender = PokemonState("scizor", sp_spread={"hp": 32, "spd": 32, "def": 2}, nature="calm")
    result = damage_range(attacker, defender, "ember", FieldState(is_doubles=False))
    assert result.breakdown["type_effectiveness"] == 4.0
    assert_actual_in_range(result, actual)


def test_special_resisted() -> None:
    # Fire vs pure Water: resisted 0.5x.
    attacker_set = pokeset(
        "Torkoal", ["flamethrower"], nature="Modest", sp={"hp": 32, "spa": 32, "spd": 2}, ability="White Smoke"
    )
    defender_set = pokeset("Feraligatr", ["splash"], nature="Bold", sp={"hp": 32, "def": 32, "spd": 2})
    _, actual = _run_single_hit(attacker_set, defender_set, "flamethrower")

    attacker = PokemonState("torkoal", sp_spread={"hp": 32, "spa": 32, "spd": 2}, nature="modest")
    defender = PokemonState("feraligatr", sp_spread={"hp": 32, "def": 32, "spd": 2}, nature="bold")
    result = damage_range(attacker, defender, "flamethrower", FieldState(is_doubles=False))
    assert result.breakdown["type_effectiveness"] == 0.5
    assert_actual_in_range(result, actual)


# --- doubles spread move + weather + Life Orb (all at once, deliberately) ----------------


def test_doubles_spread_move_with_sun_and_life_orb() -> None:
    p1_team = [
        pokeset(
            "Torkoal", ["heatwave"], nature="Quiet", sp={"hp": 32, "spa": 32, "spd": 2},
            ability="Drought", item="Life Orb",
        ),
        pokeset("Whimsicott", ["protect"], nature="Timid", sp={"hp": 32, "spe": 32}, ability="Prankster"),
    ]
    p2_team = [
        pokeset("Clefable", ["followme"], nature="Bold", sp={"hp": 32, "def": 32, "spd": 2}, ability="Magic Guard"),
        pokeset("Grimmsnarl", ["bulkup"], nature="Relaxed", sp={"hp": 32, "atk": 32, "def": 2}, ability="Prankster"),
    ]
    commands = [
        "p1 team 12",
        "p2 team 12",
        "p1 move heatwave, move protect",
        "p2 move followme, move bulkup",
    ]
    result = run_scenario(DOUBLES, p1_team, p2_team, commands, seed=(5, 6, 7, 8))
    # Grimmsnarl (p2b) gets +1 Def from Bulk Up (Prankster priority) before Heat Wave
    # connects -- Clefable (p2a) is missed by Heat Wave under this seed (confirmed by
    # replaying tools/sim_probe.mjs during development), so only p2b is asserted here.
    actual = first_damage_delta(result["log"], "p2b")

    attacker = PokemonState(
        "torkoal", sp_spread={"hp": 32, "spa": 32, "spd": 2}, nature="quiet", ability="drought", item="lifeorb"
    )
    defender = PokemonState(
        "grimmsnarl", sp_spread={"hp": 32, "atk": 32, "def": 2}, nature="relaxed", boosts={"def": 1}
    )
    field = FieldState(weather="sun", is_doubles=True, num_targets=2)
    result_calc = damage_range(attacker, defender, "heatwave", field)
    assert result_calc.breakdown["spread_modifier"] == 0.75
    assert result_calc.breakdown["weather_modifier"] == 1.5
    assert_actual_in_range(result_calc, actual)


# --- weather -------------------------------------------------------------------------------


def test_weather_rain_boosts_water_move() -> None:
    # "Drizzle" forced onto Garchomp (not its real ability -- Custom Game doesn't
    # validate; see module docstring) purely to auto-set rain at switch-in.
    attacker_set = pokeset(
        "Garchomp", ["waterfall"], nature="Adamant", sp={"hp": 30, "atk": 4, "spe": 32}, ability="Drizzle"
    )
    defender_set = pokeset("Klefki", ["splash"], nature="Bold", sp={"hp": 32, "def": 32, "spd": 2})
    _, actual = _run_single_hit(attacker_set, defender_set, "waterfall")

    attacker = PokemonState("garchomp", sp_spread={"hp": 30, "atk": 4, "spe": 32}, nature="adamant")
    defender = PokemonState("klefki", sp_spread={"hp": 32, "def": 32, "spd": 2}, nature="bold")
    result = damage_range(attacker, defender, "waterfall", FieldState(weather="rain", is_doubles=False))
    assert result.breakdown["weather_modifier"] == 1.5
    assert result.breakdown["type_effectiveness"] == 1.0  # Water vs Steel/Fairy is neutral
    assert_actual_in_range(result, actual)


def test_weather_rain_cuts_fire_move() -> None:
    attacker_set = pokeset(
        "Torkoal", ["flamethrower"], nature="Modest", sp={"hp": 32, "spa": 32, "spd": 2}, ability="Drizzle"
    )
    defender_set = pokeset("Klefki", ["splash"], nature="Bold", sp={"hp": 32, "def": 32, "spd": 2})
    _, actual = _run_single_hit(attacker_set, defender_set, "flamethrower")

    attacker = PokemonState("torkoal", sp_spread={"hp": 32, "spa": 32, "spd": 2}, nature="modest")
    defender = PokemonState("klefki", sp_spread={"hp": 32, "def": 32, "spd": 2}, nature="bold")
    result = damage_range(attacker, defender, "flamethrower", FieldState(weather="rain", is_doubles=False))
    assert result.breakdown["weather_modifier"] == 0.5
    assert_actual_in_range(result, actual)


# --- status / screens ------------------------------------------------------------------


def test_burn_halves_physical_damage() -> None:
    attacker_set = pokeset(
        "Garchomp", ["earthquake"], nature="Adamant", sp={"hp": 30, "atk": 4, "spe": 32}, ability="Rough Skin"
    )
    defender_set = pokeset("Klefki", ["splash"], nature="Bold", sp={"hp": 32, "def": 32, "spd": 2})
    _, actual = _run_single_hit(
        attacker_set, defender_set, "earthquake", setup_evals=("eval p1active.setStatus('brn')",)
    )

    attacker = PokemonState(
        "garchomp", sp_spread={"hp": 30, "atk": 4, "spe": 32}, nature="adamant", status="brn"
    )
    defender = PokemonState("klefki", sp_spread={"hp": 32, "def": 32, "spd": 2}, nature="bold")
    result = damage_range(attacker, defender, "earthquake", FieldState(is_doubles=False))
    assert result.breakdown["burn_modifier"] == 0.5
    assert_actual_in_range(result, actual)


def test_reflect_halves_physical_damage_singles() -> None:
    attacker_set = pokeset(
        "Garchomp", ["earthquake"], nature="Adamant", sp={"hp": 30, "atk": 4, "spe": 32}, ability="Rough Skin"
    )
    defender_set = pokeset("Klefki", ["splash"], nature="Bold", sp={"hp": 32, "def": 32, "spd": 2})
    _, actual = _run_single_hit(
        attacker_set, defender_set, "earthquake", setup_evals=("eval p2.addSideCondition('reflect', 'debug')",)
    )

    attacker = PokemonState("garchomp", sp_spread={"hp": 30, "atk": 4, "spe": 32}, nature="adamant")
    defender = PokemonState("klefki", sp_spread={"hp": 32, "def": 32, "spd": 2}, nature="bold")
    field = FieldState(screens=frozenset({"reflect"}), is_doubles=False)
    result = damage_range(attacker, defender, "earthquake", field)
    assert result.breakdown["final_multiplier"] == pytest.approx(0.5)
    assert_actual_in_range(result, actual)


def test_light_screen_two_thirds_multiplier_in_doubles() -> None:
    p1_team = [
        pokeset(
            "Torkoal", ["flamethrower"], nature="Modest", sp={"hp": 32, "spa": 32, "spd": 2}, ability="White Smoke"
        ),
        pokeset("Whimsicott", ["protect"], nature="Timid", sp={"hp": 32, "spe": 32}, ability="Prankster"),
    ]
    p2_team = [
        pokeset("Feraligatr", ["splash"], nature="Bold", sp={"hp": 32, "def": 32, "spd": 2}, ability="Torrent"),
        pokeset("Klefki", ["splash"], nature="Bold", sp={"hp": 32, "def": 32, "spd": 2}, ability="Prankster"),
    ]
    commands = [
        "p1 team 12",
        "p2 team 12",
        "eval battle.sides[1].addSideCondition('lightscreen', 'debug')",
        "p1 move flamethrower 1, move protect",
        "p2 move splash, move splash",
    ]
    result = run_scenario(DOUBLES, p1_team, p2_team, commands)
    actual = first_damage_delta(result["log"], "p2a")

    attacker = PokemonState("torkoal", sp_spread={"hp": 32, "spa": 32, "spd": 2}, nature="modest")
    defender = PokemonState("feraligatr", sp_spread={"hp": 32, "def": 32, "spd": 2}, nature="bold")
    field = FieldState(screens=frozenset({"lightscreen"}), is_doubles=True)
    result_calc = damage_range(attacker, defender, "flamethrower", field)
    assert result_calc.breakdown["final_multiplier"] == pytest.approx(2732 / 4096)
    assert_actual_in_range(result_calc, actual)


# --- items -----------------------------------------------------------------------------


def test_life_orb_boosts_damage() -> None:
    attacker_set = pokeset(
        "Garchomp", ["earthquake"], nature="Adamant", sp={"hp": 30, "atk": 4, "spe": 32},
        ability="Rough Skin", item="Life Orb",
    )
    # Blissey (huge HP, Normal-typed so neutral to Ground) rather than Klefki here --
    # Life Orb's 1.3x on top of an already-super-effective Earthquake would one-shot
    # Klefki's fixed 164 max HP (Stat Points cap HP investment at 32, so Klefki can't be
    # made bulkier), and a fainted `-damage` line reports `0 fnt` instead of the real
    # remaining-HP delta, which breaks the ground-truth comparison.
    defender_set = pokeset("Blissey", ["splash"], nature="Bold", sp={"hp": 32, "def": 32, "spd": 2})
    _, actual = _run_single_hit(attacker_set, defender_set, "earthquake")

    attacker = PokemonState(
        "garchomp", sp_spread={"hp": 30, "atk": 4, "spe": 32}, nature="adamant", item="lifeorb"
    )
    defender = PokemonState("blissey", sp_spread={"hp": 32, "def": 32, "spd": 2}, nature="bold")
    result = damage_range(attacker, defender, "earthquake", FieldState(is_doubles=False))
    assert result.breakdown["final_multiplier"] == pytest.approx(5324 / 4096)
    assert_actual_in_range(result, actual)


def test_type_boost_item_charcoal() -> None:
    attacker_set = pokeset(
        "Torkoal", ["flamethrower"], nature="Modest", sp={"hp": 32, "spa": 32, "spd": 2},
        ability="White Smoke", item="Charcoal",
    )
    defender_set = pokeset("Feraligatr", ["splash"], nature="Bold", sp={"hp": 32, "def": 32, "spd": 2})
    _, actual = _run_single_hit(attacker_set, defender_set, "flamethrower")

    attacker = PokemonState(
        "torkoal", sp_spread={"hp": 32, "spa": 32, "spd": 2}, nature="modest", item="charcoal"
    )
    defender = PokemonState("feraligatr", sp_spread={"hp": 32, "def": 32, "spd": 2}, nature="bold")
    result = damage_range(attacker, defender, "flamethrower", FieldState(is_doubles=False))
    assert result.breakdown["base_power"] == int(90 * 4915 / 4096)
    assert_actual_in_range(result, actual)


def test_expert_belt_boosts_only_super_effective_hits() -> None:
    # Ember, not Flamethrower -- see test_special_stab_double_super_effective's comment
    # (a 4x-effective STAB Flamethrower + Expert Belt's own 1.2x one-shots Scizor).
    attacker_set = pokeset(
        "Torkoal", ["ember"], nature="Modest", sp={"hp": 32, "spa": 32, "spd": 2},
        ability="White Smoke", item="Expert Belt",
    )
    defender_set = pokeset("Scizor", ["splash"], nature="Calm", sp={"hp": 32, "spd": 32, "def": 2})
    _, actual = _run_single_hit(attacker_set, defender_set, "ember")

    attacker = PokemonState(
        "torkoal", sp_spread={"hp": 32, "spa": 32, "spd": 2}, nature="modest", item="expertbelt"
    )
    defender = PokemonState("scizor", sp_spread={"hp": 32, "spd": 32, "def": 2}, nature="calm")
    result = damage_range(attacker, defender, "ember", FieldState(is_doubles=False))
    assert result.breakdown["final_multiplier"] == pytest.approx(4915 / 4096)
    assert_actual_in_range(result, actual)


# --- variable base power moves (Weather Ball / Water Spout / Electro Ball / Gyro Ball /
# weight-based moves -- see `vgc.damage._variable_base_power`) --------------------------


def test_weather_ball_in_sun_on_charizard_y() -> None:
    # The flagship case this whole feature exists for: Charizard-Y's own ability is
    # Drought (data/champions/species.json), so forcing "Drought" onto the base Charizard
    # here (same on-switch-in-triggers-sun trick as test_weather_rain_boosts_water_move's
    # Drizzle) is not even a legality bypass -- it's Charizard-Y's real ability, just
    # applied one turn before it mega evolves so sun is already active when Weather Ball
    # (now 100 BP Fire-type) resolves: sun's 1.5x Fire boost AND this Pokemon's own Fire
    # STAB both stack on top of the doubled base power.
    attacker_set = pokeset(
        "Charizard", ["weatherball"], nature="Modest", sp={"hp": 32, "spa": 32, "spe": 2},
        ability="Drought", item="Charizardite Y",
    )
    # Blissey (huge HP, Normal-typed -- neutral to Fire) rather than a squishier mon, same
    # anti-faint reasoning as test_life_orb_boosts_damage.
    defender_set = pokeset("Blissey", ["splash"], nature="Bold", sp={"hp": 32, "def": 32, "spd": 2})
    log, actual = _run_single_hit(attacker_set, defender_set, "weatherball mega")
    assert any("|-mega|" in line for line in log), "expected Charizard-Y to mega evolve"

    attacker = PokemonState("charizardmegay", sp_spread={"hp": 32, "spa": 32, "spe": 2}, nature="modest")
    defender = PokemonState("blissey", sp_spread={"hp": 32, "def": 32, "spd": 2}, nature="bold")
    result = damage_range(attacker, defender, "weatherball", FieldState(weather="sun", is_doubles=False))
    assert result.breakdown["base_power"] == 100
    assert result.breakdown["stab"] == 1.5
    assert result.breakdown["weather_modifier"] == 1.5
    assert_actual_in_range(result, actual)


def test_weather_ball_in_rain_changes_type_to_water() -> None:
    attacker_set = pokeset(
        "Charizard", ["weatherball"], nature="Modest", sp={"hp": 32, "spa": 32, "spe": 2}, ability="Drizzle"
    )
    defender_set = pokeset("Blissey", ["splash"], nature="Bold", sp={"hp": 32, "def": 32, "spd": 2})
    _, actual = _run_single_hit(attacker_set, defender_set, "weatherball")

    attacker = PokemonState("charizard", sp_spread={"hp": 32, "spa": 32, "spe": 2}, nature="modest")
    defender = PokemonState("blissey", sp_spread={"hp": 32, "def": 32, "spd": 2}, nature="bold")
    result = damage_range(attacker, defender, "weatherball", FieldState(weather="rain", is_doubles=False))
    assert result.breakdown["base_power"] == 100
    assert result.breakdown["stab"] == 1.0  # base Charizard is Fire/Flying, not Water
    assert result.breakdown["weather_modifier"] == 1.5  # rain boosts the now-Water-type move
    assert_actual_in_range(result, actual)


def test_water_spout_at_full_hp() -> None:
    attacker_set = pokeset(
        "Feraligatr", ["waterspout"], nature="Modest", sp={"hp": 32, "spa": 32, "spd": 2}, ability="Torrent"
    )
    defender_set = pokeset("Blissey", ["splash"], nature="Bold", sp={"hp": 32, "def": 32, "spd": 2})
    _, actual = _run_single_hit(attacker_set, defender_set, "waterspout")

    attacker = PokemonState("feraligatr", sp_spread={"hp": 32, "spa": 32, "spd": 2}, nature="modest")
    defender = PokemonState("blissey", sp_spread={"hp": 32, "def": 32, "spd": 2}, nature="bold")
    result = damage_range(attacker, defender, "waterspout", FieldState(is_doubles=False))
    assert result.breakdown["base_power"] == 150
    assert_actual_in_range(result, actual)


def test_water_spout_at_half_hp() -> None:
    attacker_set = pokeset(
        "Feraligatr", ["waterspout"], nature="Modest", sp={"hp": 32, "spa": 32, "spd": 2}, ability="Torrent"
    )
    defender_set = pokeset("Blissey", ["splash"], nature="Bold", sp={"hp": 32, "def": 32, "spd": 2})
    _, actual = _run_single_hit(
        attacker_set, defender_set, "waterspout",
        setup_evals=("eval p1active.sethp(Math.floor(p1active.maxhp / 2))",),
    )

    attacker = PokemonState("feraligatr", sp_spread={"hp": 32, "spa": 32, "spd": 2}, nature="modest")
    max_hp = attacker.max_hp()
    attacker.current_hp = max_hp // 2
    defender = PokemonState("blissey", sp_spread={"hp": 32, "def": 32, "spd": 2}, nature="bold")
    result = damage_range(attacker, defender, "waterspout", FieldState(is_doubles=False))
    assert result.breakdown["base_power"] == (150 * (max_hp // 2)) // max_hp
    assert result.breakdown["base_power"] < 150
    assert_actual_in_range(result, actual)


def test_gyro_ball_slow_attacker_vs_fast_target() -> None:
    # Torkoal (base 20 Spe, 0 Speed SP, Brave -10%) vs. a maximally Speed-invested Jolly
    # Garchomp -- a large enough gap that Gyro Ball lands well above its 50 BP midpoint.
    attacker_set = pokeset(
        "Torkoal", ["gyroball"], nature="Brave", sp={"hp": 32, "atk": 32, "def": 2}, ability="White Smoke"
    )
    defender_set = pokeset("Garchomp", ["splash"], nature="Jolly", sp={"hp": 30, "spe": 32}, ability="Rough Skin")
    _, actual = _run_single_hit(attacker_set, defender_set, "gyroball")

    attacker = PokemonState("torkoal", sp_spread={"hp": 32, "atk": 32, "def": 2}, nature="brave")
    defender = PokemonState("garchomp", sp_spread={"hp": 30, "spe": 32}, nature="jolly")
    result = damage_range(attacker, defender, "gyroball", FieldState(is_doubles=False))
    assert result.breakdown["base_power"] > 100  # slow attacker vs. a much faster target
    assert_actual_in_range(result, actual)


def test_electro_ball_fast_attacker_vs_slow_target() -> None:
    # Mirror of the Gyro Ball case above: a maximally Speed-invested Timid Garchomp vs.
    # Torkoal (0 Speed SP, Brave -10%) is a large enough ratio to hit Electro Ball's top
    # (150 BP) bracket.
    attacker_set = pokeset(
        "Garchomp", ["electroball"], nature="Timid", sp={"hp": 30, "spa": 4, "spe": 32}, ability="Rough Skin"
    )
    defender_set = pokeset("Torkoal", ["splash"], nature="Brave", sp={"hp": 32, "def": 32, "spd": 2})
    _, actual = _run_single_hit(attacker_set, defender_set, "electroball")

    attacker = PokemonState("garchomp", sp_spread={"hp": 30, "spa": 4, "spe": 32}, nature="timid")
    defender = PokemonState("torkoal", sp_spread={"hp": 32, "def": 32, "spd": 2}, nature="brave")
    result = damage_range(attacker, defender, "electroball", FieldState(is_doubles=False))
    assert result.breakdown["base_power"] >= 120  # fast attacker vs. a much slower target
    assert_actual_in_range(result, actual)


def test_grass_knot_on_a_heavy_target() -> None:
    # Dragonite (210 kg = 2100 hg) clears the top weight tier -> 120 BP.
    attacker_set = pokeset(
        "Torkoal", ["grassknot"], nature="Modest", sp={"hp": 32, "spa": 32, "spd": 2}, ability="White Smoke"
    )
    defender_set = pokeset("Dragonite", ["splash"], nature="Bold", sp={"hp": 32, "def": 32, "spd": 2})
    _, actual = _run_single_hit(attacker_set, defender_set, "grassknot")

    attacker = PokemonState("torkoal", sp_spread={"hp": 32, "spa": 32, "spd": 2}, nature="modest")
    defender = PokemonState("dragonite", sp_spread={"hp": 32, "def": 32, "spd": 2}, nature="bold")
    result = damage_range(attacker, defender, "grassknot", FieldState(is_doubles=False))
    assert result.breakdown["base_power"] == 120
    assert_actual_in_range(result, actual)


def test_heavy_slam_by_weight_ratio() -> None:
    # Dragonite (2100 hg) vs. Klefki (30 hg): ratio ~70 -> the top 120 BP tier.
    attacker_set = pokeset(
        "Dragonite", ["heavyslam"], nature="Adamant", sp={"hp": 32, "atk": 32, "spe": 2}
    )
    defender_set = pokeset("Klefki", ["splash"], nature="Bold", sp={"hp": 32, "def": 32, "spd": 2})
    _, actual = _run_single_hit(attacker_set, defender_set, "heavyslam")

    attacker = PokemonState("dragonite", sp_spread={"hp": 32, "atk": 32, "spe": 2}, nature="adamant")
    defender = PokemonState("klefki", sp_spread={"hp": 32, "def": 32, "spd": 2}, nature="bold")
    result = damage_range(attacker, defender, "heavyslam", FieldState(is_doubles=False))
    assert result.breakdown["base_power"] == 120
    assert_actual_in_range(result, actual)


# --- mega forme --------------------------------------------------------------------------


def test_mega_forme_attacker() -> None:
    attacker_set = pokeset(
        "Garchomp", ["earthquake"], nature="Adamant", sp={"hp": 30, "atk": 4, "spe": 32},
        ability="Rough Skin", item="Garchompite",
    )
    defender_set = pokeset("Klefki", ["splash"], nature="Bold", sp={"hp": 32, "def": 32, "spd": 2})
    log, actual = _run_single_hit(attacker_set, defender_set, "earthquake mega")
    assert any("|-mega|" in line for line in log), "expected a mega evolution to occur"

    # Stats/ability/types after mega evolving come from the mega forme's own species
    # entry (garchompmega), same Stat Points/nature as the base forme (see CLAUDE.md).
    attacker = PokemonState(
        "garchompmega", sp_spread={"hp": 30, "atk": 4, "spe": 32}, nature="adamant", ability="sandforce"
    )
    defender = PokemonState("klefki", sp_spread={"hp": 32, "def": 32, "spd": 2}, nature="bold")
    result = damage_range(attacker, defender, "earthquake", FieldState(is_doubles=False))
    assert_actual_in_range(result, actual)


# --- ability-based type immunities ------------------------------------------------------


def test_water_absorb_immunity() -> None:
    attacker_set = pokeset(
        "Garchomp", ["waterfall"], nature="Adamant", sp={"hp": 30, "atk": 4, "spe": 32}, ability="Rough Skin"
    )
    defender_set = pokeset(
        "Klefki", ["splash"], nature="Bold", sp={"hp": 32, "def": 32, "spd": 2}, ability="Water Absorb"
    )
    log = _run_single_hit_expect_immune(attacker_set, defender_set, "waterfall")
    assert any("-immune" in line for line in log)

    attacker = PokemonState("garchomp", sp_spread={"hp": 30, "atk": 4, "spe": 32}, nature="adamant")
    defender = PokemonState(
        "klefki", sp_spread={"hp": 32, "def": 32, "spd": 2}, nature="bold", ability="waterabsorb"
    )
    result = damage_range(attacker, defender, "waterfall", FieldState(is_doubles=False))
    assert result.breakdown["immune"] is True
    assert result.min_damage == 0 and result.max_damage == 0


def test_levitate_immunity() -> None:
    attacker_set = pokeset(
        "Garchomp", ["earthquake"], nature="Adamant", sp={"hp": 30, "atk": 4, "spe": 32}, ability="Rough Skin"
    )
    defender_set = pokeset(
        "Klefki", ["splash"], nature="Bold", sp={"hp": 32, "def": 32, "spd": 2}, ability="Levitate"
    )
    log = _run_single_hit_expect_immune(attacker_set, defender_set, "earthquake")
    assert any("-immune" in line for line in log)

    attacker = PokemonState("garchomp", sp_spread={"hp": 30, "atk": 4, "spe": 32}, nature="adamant")
    defender = PokemonState(
        "klefki", sp_spread={"hp": 32, "def": 32, "spd": 2}, nature="bold", ability="levitate"
    )
    result = damage_range(attacker, defender, "earthquake", FieldState(is_doubles=False))
    assert result.breakdown["immune"] is True


# --- other whitelisted abilities --------------------------------------------------------


def test_multiscale_halves_damage_at_full_hp() -> None:
    attacker_set = pokeset(
        "Torkoal", ["icebeam"], nature="Modest", sp={"hp": 32, "spa": 32, "spd": 2}, ability="White Smoke"
    )
    defender_set = pokeset(
        "Dragonite", ["splash"], nature="Bold", sp={"hp": 32, "def": 32, "spd": 2}, ability="Multiscale"
    )
    _, actual = _run_single_hit(attacker_set, defender_set, "icebeam")

    attacker = PokemonState("torkoal", sp_spread={"hp": 32, "spa": 32, "spd": 2}, nature="modest")
    defender = PokemonState(
        "dragonite", sp_spread={"hp": 32, "def": 32, "spd": 2}, nature="bold", ability="multiscale"
    )
    result = damage_range(attacker, defender, "icebeam", FieldState(is_doubles=False))
    assert result.breakdown["final_multiplier"] == pytest.approx(0.5)
    assert_actual_in_range(result, actual)


def test_solid_rock_reduces_super_effective_damage() -> None:
    attacker_set = pokeset(
        "Garchomp", ["earthquake"], nature="Adamant", sp={"hp": 30, "atk": 4, "spe": 32}, ability="Rough Skin"
    )
    defender_set = pokeset(
        "Klefki", ["splash"], nature="Bold", sp={"hp": 32, "def": 32, "spd": 2}, ability="Solid Rock"
    )
    _, actual = _run_single_hit(attacker_set, defender_set, "earthquake")

    attacker = PokemonState("garchomp", sp_spread={"hp": 30, "atk": 4, "spe": 32}, nature="adamant")
    defender = PokemonState(
        "klefki", sp_spread={"hp": 32, "def": 32, "spd": 2}, nature="bold", ability="solidrock"
    )
    result = damage_range(attacker, defender, "earthquake", FieldState(is_doubles=False))
    assert result.breakdown["final_multiplier"] == pytest.approx(0.75)
    assert_actual_in_range(result, actual)


def test_tinted_lens_boosts_resisted_damage() -> None:
    attacker_set = pokeset(
        "Garchomp", ["icebeam"], nature="Modest", sp={"hp": 32, "spa": 32, "spd": 2},
        ability="Tinted Lens", item="Wise Glasses",
    )
    defender_set = pokeset("Klefki", ["splash"], nature="Bold", sp={"hp": 32, "def": 32, "spd": 2})
    _, actual = _run_single_hit(attacker_set, defender_set, "icebeam")

    attacker = PokemonState(
        "garchomp", sp_spread={"hp": 32, "spa": 32, "spd": 2}, nature="modest",
        ability="tintedlens", item="wiseglasses",
    )
    defender = PokemonState("klefki", sp_spread={"hp": 32, "def": 32, "spd": 2}, nature="bold")
    result = damage_range(attacker, defender, "icebeam", FieldState(is_doubles=False))
    assert result.breakdown["type_effectiveness"] == 0.5
    assert result.breakdown["final_multiplier"] == pytest.approx(2.0)
    assert_actual_in_range(result, actual)


def test_thick_fat_halves_fire_damage() -> None:
    attacker_set = pokeset(
        "Torkoal", ["flamethrower"], nature="Modest", sp={"hp": 32, "spa": 32, "spd": 2}, ability="White Smoke"
    )
    defender_set = pokeset(
        "Klefki", ["splash"], nature="Bold", sp={"hp": 32, "def": 32, "spd": 2}, ability="Thick Fat"
    )
    _, actual = _run_single_hit(attacker_set, defender_set, "flamethrower")

    attacker = PokemonState("torkoal", sp_spread={"hp": 32, "spa": 32, "spd": 2}, nature="modest")
    defender = PokemonState(
        "klefki", sp_spread={"hp": 32, "def": 32, "spd": 2}, nature="bold", ability="thickfat"
    )
    result = damage_range(attacker, defender, "flamethrower", FieldState(is_doubles=False))
    assert result.breakdown["final_multiplier"] == pytest.approx(0.5)
    assert_actual_in_range(result, actual)


def test_huge_power_doubles_attack_stat() -> None:
    attacker_set = pokeset(
        "Azumarill", ["waterfall"], nature="Adamant", sp={"hp": 32, "atk": 32, "spe": 2}, ability="Huge Power"
    )
    defender_set = pokeset("Klefki", ["splash"], nature="Bold", sp={"hp": 32, "def": 32, "spd": 2})
    _, actual = _run_single_hit(attacker_set, defender_set, "waterfall")

    attacker = PokemonState(
        "azumarill", sp_spread={"hp": 32, "atk": 32, "spe": 2}, nature="adamant", ability="hugepower"
    )
    defender = PokemonState("klefki", sp_spread={"hp": 32, "def": 32, "spd": 2}, nature="bold")
    result = damage_range(attacker, defender, "waterfall", FieldState(is_doubles=False))
    assert_actual_in_range(result, actual)


def test_technician_boosts_low_power_move() -> None:
    attacker_set = pokeset(
        "Scizor", ["quickattack"], nature="Adamant", sp={"hp": 32, "atk": 32, "spe": 2}, ability="Technician"
    )
    defender_set = pokeset("Klefki", ["splash"], nature="Bold", sp={"hp": 32, "def": 32, "spd": 2})
    _, actual = _run_single_hit(attacker_set, defender_set, "quickattack")

    attacker = PokemonState(
        "scizor", sp_spread={"hp": 32, "atk": 32, "spe": 2}, nature="adamant", ability="technician"
    )
    defender = PokemonState("klefki", sp_spread={"hp": 32, "def": 32, "spd": 2}, nature="bold")
    result = damage_range(attacker, defender, "quickattack", FieldState(is_doubles=False))
    assert result.breakdown["base_power"] == 60  # 40 * 1.5, Technician's <=60bp threshold
    assert_actual_in_range(result, actual)


def test_adaptability_boosts_stab_to_double() -> None:
    attacker_set = pokeset(
        "Garchomp", ["earthquake"], nature="Adamant", sp={"hp": 30, "atk": 4, "spe": 32}, ability="Adaptability"
    )
    # Blissey (huge HP) rather than Klefki -- see test_life_orb_boosts_damage's comment;
    # Adaptability's 2x STAB is strong enough to KO Klefki's fixed 164 max HP here too.
    defender_set = pokeset("Blissey", ["splash"], nature="Bold", sp={"hp": 32, "def": 32, "spd": 2})
    _, actual = _run_single_hit(attacker_set, defender_set, "earthquake")

    attacker = PokemonState(
        "garchomp", sp_spread={"hp": 30, "atk": 4, "spe": 32}, nature="adamant", ability="adaptability"
    )
    defender = PokemonState("blissey", sp_spread={"hp": 32, "def": 32, "spd": 2}, nature="bold")
    result = damage_range(attacker, defender, "earthquake", FieldState(is_doubles=False))
    assert result.breakdown["stab"] == 2.0
    assert_actual_in_range(result, actual)


def test_guts_boosts_attack_when_statused() -> None:
    attacker_set = pokeset(
        "Garchomp", ["earthquake"], nature="Adamant", sp={"hp": 30, "atk": 4, "spe": 32}, ability="Guts"
    )
    # Blissey (huge HP) rather than Klefki -- see test_life_orb_boosts_damage's comment;
    # Guts' 1.5x Atk on top of an already-super-effective Earthquake is strong enough to
    # KO Klefki's fixed 164 max HP here too.
    defender_set = pokeset("Blissey", ["splash"], nature="Bold", sp={"hp": 32, "def": 32, "spd": 2})
    # Paralysis, not burn -- isolates Guts' Atk boost from burn's (Guts-negated) halving.
    _, actual = _run_single_hit(
        attacker_set, defender_set, "earthquake", setup_evals=("eval p1active.setStatus('par')",)
    )

    attacker = PokemonState(
        "garchomp", sp_spread={"hp": 30, "atk": 4, "spe": 32}, nature="adamant", ability="guts", status="par"
    )
    defender = PokemonState("blissey", sp_spread={"hp": 32, "def": 32, "spd": 2}, nature="bold")
    result = damage_range(attacker, defender, "earthquake", FieldState(is_doubles=False))
    assert result.breakdown["attack_value"] == int(169 * 1.5)
    assert_actual_in_range(result, actual)


# --- terrain -----------------------------------------------------------------------------


def test_electric_terrain_boosts_grounded_attacker() -> None:
    attacker_set = pokeset(
        "Garchomp", ["thunderbolt"], nature="Modest", sp={"hp": 32, "spa": 32, "spd": 2}, ability="Rough Skin"
    )
    defender_set = pokeset("Feraligatr", ["splash"], nature="Bold", sp={"hp": 32, "def": 32, "spd": 2})
    _, actual = _run_single_hit(
        attacker_set, defender_set, "thunderbolt", setup_evals=("eval battle.field.setTerrain('electricterrain', 'debug')",)
    )

    attacker = PokemonState("garchomp", sp_spread={"hp": 32, "spa": 32, "spd": 2}, nature="modest")
    defender = PokemonState("feraligatr", sp_spread={"hp": 32, "def": 32, "spd": 2}, nature="bold")
    result = damage_range(attacker, defender, "thunderbolt", FieldState(terrain="electric", is_doubles=False))
    assert result.breakdown["terrain_modifier"] == 1.3
    assert_actual_in_range(result, actual)


def test_misty_terrain_halves_dragon_move_on_grounded_defender() -> None:
    attacker_set = pokeset(
        "Garchomp", ["dragonclaw"], nature="Adamant", sp={"hp": 30, "atk": 4, "spe": 32}, ability="Rough Skin"
    )
    defender_set = pokeset("Feraligatr", ["splash"], nature="Bold", sp={"hp": 32, "def": 32, "spd": 2})
    _, actual = _run_single_hit(
        attacker_set, defender_set, "dragonclaw", setup_evals=("eval battle.field.setTerrain('mistyterrain', 'debug')",)
    )

    attacker = PokemonState("garchomp", sp_spread={"hp": 30, "atk": 4, "spe": 32}, nature="adamant")
    defender = PokemonState("feraligatr", sp_spread={"hp": 32, "def": 32, "spd": 2}, nature="bold")
    result = damage_range(attacker, defender, "dragonclaw", FieldState(terrain="misty", is_doubles=False))
    assert result.breakdown["terrain_modifier"] == 0.5
    assert_actual_in_range(result, actual)


# --- Wonder Guard (no legal holder in this mod -- see report; ability is still
# implemented and exercised here via the same validation-bypass trick as everything
# else in this file) --------------------------------------------------------------------


def test_wonder_guard_blocks_non_super_effective_hit() -> None:
    attacker_set = pokeset(
        "Garchomp", ["waterfall"], nature="Adamant", sp={"hp": 30, "atk": 4, "spe": 32}, ability="Rough Skin"
    )
    defender_set = pokeset(
        "Klefki", ["splash"], nature="Bold", sp={"hp": 32, "def": 32, "spd": 2}, ability="Wonder Guard"
    )
    log = _run_single_hit_expect_immune(attacker_set, defender_set, "waterfall")
    assert any("-immune" in line for line in log)

    attacker = PokemonState("garchomp", sp_spread={"hp": 30, "atk": 4, "spe": 32}, nature="adamant")
    defender = PokemonState(
        "klefki", sp_spread={"hp": 32, "def": 32, "spd": 2}, nature="bold", ability="wonderguard"
    )
    result = damage_range(attacker, defender, "waterfall", FieldState(is_doubles=False))
    assert result.breakdown["immune"] is True


def test_wonder_guard_allows_super_effective_hit() -> None:
    attacker_set = pokeset(
        "Garchomp", ["earthquake"], nature="Adamant", sp={"hp": 30, "atk": 4, "spe": 32}, ability="Rough Skin"
    )
    defender_set = pokeset(
        "Klefki", ["splash"], nature="Bold", sp={"hp": 32, "def": 32, "spd": 2}, ability="Wonder Guard"
    )
    _, actual = _run_single_hit(attacker_set, defender_set, "earthquake")
    assert actual > 0

    attacker = PokemonState("garchomp", sp_spread={"hp": 30, "atk": 4, "spe": 32}, nature="adamant")
    defender = PokemonState(
        "klefki", sp_spread={"hp": 32, "def": 32, "spd": 2}, nature="bold", ability="wonderguard"
    )
    result = damage_range(attacker, defender, "earthquake", FieldState(is_doubles=False))
    assert result.breakdown["immune"] is False
    assert_actual_in_range(result, actual)
