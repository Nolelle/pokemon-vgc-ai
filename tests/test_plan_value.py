"""Hand-written checklist for the measured plan value (`vgc.plan_value`, the probe).

Each case builds ONE set with the real builder path (probe -> per-set entry) and checks the
mechanic the owner says weather/terrain are worth: the engine should show a clear gain where a
condition enables the move, and ~nothing where it does not. Integration-marked (it runs the
local Showdown engine; no server). Sets are not legality-checked -- the probe plays Custom Game.
"""

from __future__ import annotations

import pytest

from vgc import plan_value as pv

pytestmark = pytest.mark.integration

SINGLE, DOUBLE = 0, 1


def _gain(entry: dict, weather: str, terrain: str, profile: int = SINGLE) -> float:
    return entry["gain"][pv.CONDITIONS.index((weather, terrain))][profile]


def _bare(entry: dict, profile: int = SINGLE) -> float:
    return entry["per_turn"][0][profile]


@pytest.fixture(scope="module")
def measure():
    with pv.ProbeWorker() as worker:
        yield lambda packed: worker.measure(pv.parse_packed_set(packed), seeds=(1,))


def test_electro_shot_fires_in_one_turn_in_rain(measure):
    entry = measure("Archaludon||Leftovers|Stamina|ElectroShot,Protect|Modest|2,,,32,,32||||50|")
    # Bare field: turn 1 charges, turn 2 hits. Rain: both turns hit, the second at +2.
    assert _gain(entry, "raindance", "none") > 0.8 * _bare(entry)
    assert abs(_gain(entry, "sunnyday", "none")) < 1.0


def test_solar_beam_skips_charge_in_sun(measure):
    entry = measure("Venusaur||Leftovers|Chlorophyll|SolarBeam,Protect|Modest|2,,,32,,32||||50|")
    assert _gain(entry, "sunnyday", "none") > 0.8 * _bare(entry)
    # Rain halves Solar Beam's power on top of the charge turn.
    assert _gain(entry, "raindance", "none") < 0


def test_thunder_never_misses_in_rain(measure):
    entry = measure("Thundurus||Leftovers|Prankster|Thunder,Protect|Modest|2,,,32,,32||||50|")
    # 70% -> 100% accuracy is +43% of the bare expectation; the engine computes it, not us.
    assert _gain(entry, "raindance", "none") == pytest.approx(0.43 * _bare(entry), rel=0.05)
    assert _gain(entry, "sunnyday", "none") < 0  # 50% accuracy in sun


def test_expanding_force_spreads_and_boosts_in_psychic_terrain(measure):
    entry = measure("Indeedee||Leftovers|PsychicSurge|ExpandingForce,Protect|Modest|2,,,32,,32|M|||50|")
    one, two = _gain(entry, "none", "psychicterrain"), _gain(
        entry, "none", "psychicterrain", DOUBLE
    )
    assert one > 0.4 * _bare(entry)  # 1.5x power
    assert two > 1.5 * one  # and it now hits both foes (minus spread reduction)


def test_terrain_pulse_gains_in_every_terrain(measure):
    entry = measure("Blastoise||Leftovers|Torrent|TerrainPulse,Protect|Modest|2,,,32,,32||||50|")
    for terrain in ("electricterrain", "grassyterrain", "psychicterrain", "mistyterrain"):
        assert _gain(entry, "none", terrain) > 0.3 * _bare(entry), terrain


def test_set_with_no_field_moves_gains_nothing_from_weather(measure):
    entry = measure("Snorlax||Leftovers|Thick Fat|BodySlam,Earthquake|Adamant|2,32,,,,||||50|")
    for weather in ("sunnyday", "raindance", "sandstorm", "snowscape"):
        assert abs(_gain(entry, weather, "none")) < 1.0, weather
