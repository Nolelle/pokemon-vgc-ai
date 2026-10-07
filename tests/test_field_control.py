"""Critical behaviour of `vgc.field_control.field_control_value`."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from vgc.field_control import field_control_value
from vgc.models import PolicyConfig
from vgc.stats import calculate_stats

CONFIG = PolicyConfig()
TURN = 8


def _mon(species, moves=("protect",), *, active=True, ours=True, ability=None, item=None):
    stats = calculate_stats(species, sp={"spe": 20}, nature="serious")
    return SimpleNamespace(
        species_id=species,
        base_species_id=species,
        fainted=False,
        active=active,
        ability_id=ability,
        ability_known=ours,
        item_id=item,
        item_known=ours,
        status=None,
        boosts=(("spe", 0),),
        stats=tuple(stats.items()) if ours else (),
        moves=tuple(SimpleNamespace(id=m, disabled=False) for m in moves),
    )


def _effect(effect_id, start):
    return SimpleNamespace(id=effect_id, turns=start)


def _state(ours, theirs, *, weather=(), fields=(), our_cond=(), their_cond=()):
    return SimpleNamespace(
        turn=TURN,
        finished=False,
        weather=tuple(weather),
        fields=tuple(fields),
        our_side=SimpleNamespace(pokemon=tuple(ours), side_conditions=tuple(our_cond)),
        opponent_side=SimpleNamespace(pokemon=tuple(theirs), side_conditions=tuple(their_cond)),
    )


def _slow_vs_fast(**kwargs):
    slow = [_mon("torkoal"), _mon("ursaluna")]
    fast = [_mon("dragapult", ours=False), _mon("greninja", ours=False)]
    return slow, fast, kwargs


def test_trick_room_helps_slow_side_and_swap_negates():
    slow, fast, _ = _slow_vs_fast()
    room = (_effect("trickroom", TURN - 1),)  # set last turn: covers turns 8..11
    slow_ours = field_control_value(_state(slow, fast, fields=room), CONFIG)
    # same board with the fast side as "ours"
    fast_ours = [_mon("dragapult"), _mon("greninja")]
    slow_theirs = [_mon("torkoal", ours=False), _mon("ursaluna", ours=False)]
    fast_val = field_control_value(_state(fast_ours, slow_theirs, fields=room), CONFIG)
    none_val = field_control_value(_state(slow, fast), CONFIG)
    assert slow_ours > 0 > fast_val
    assert slow_ours > none_val  # Trick Room flips a lost speed race
    # Swapping the roles of the two teams negates the value (spread data differs per
    # side, so allow a small tolerance).
    assert abs(slow_ours + fast_val) < 1.0


def test_tailwind_value_scales_with_turns_left():
    ours = [_mon("kingambit"), _mon("incineroar")]
    theirs = [_mon("dragapult", ours=False), _mon("greninja", ours=False)]

    def value(start):
        cond = () if start is None else (_effect("tailwind", start),)
        return field_control_value(_state(ours, theirs, our_cond=cond), CONFIG)

    three_left = value(TURN - 1)  # set last turn, duration 4 -> turns 5,6,7
    one_left = value(TURN - 3)  # -> turn 5 only
    expired = value(TURN - 4)  # -> ended before turn 5
    assert three_left > one_left > value(None)
    assert expired == value(None)


def test_rain_helps_water_attackers_and_sun_hurts():
    ours = [_mon("kingdra", ("hydropump", "surf")), _mon("milotic", ("muddywater",))]
    theirs = [_mon("incineroar", ("flareblitz",), ours=False)]
    theirs[0].ability_known = True
    base = field_control_value(_state(ours, theirs), CONFIG)
    rain = field_control_value(_state(ours, theirs, weather=(_effect("raindance", 4),)), CONFIG)
    sun = field_control_value(_state(ours, theirs, weather=(_effect("sunnyday", 4),)), CONFIG)
    assert rain > base > sun


def test_expired_conditions_contribute_nothing():
    slow, fast, _ = _slow_vs_fast()
    base = field_control_value(_state(slow, fast), CONFIG)
    expired_room = (_effect("trickroom", TURN - 6),)  # last active turn 6 < 8
    expired_terrain = (_effect("grassyterrain", TURN - 6),)
    state = _state(slow, fast, fields=expired_room + expired_terrain)
    assert field_control_value(state, CONFIG) == base


def test_runtime_under_five_ms():
    ours = [_mon("torkoal", ("heatwave", "weatherball")), _mon("ursaluna"),
            _mon("kingdra", ("hydropump",), active=False), _mon("milotic", active=False)]
    theirs = [_mon("dragapult", ("shadowball",), ours=False), _mon("greninja", ours=False),
              _mon("incineroar", ours=False, active=False), _mon("rillaboom", ours=False,
                                                                 active=False)]
    state = _state(
        ours, theirs, weather=(_effect("sunnyday", 4),),
        fields=(_effect("trickroom", 4), _effect("grassyterrain", 4)),
        our_cond=(_effect("tailwind", 4),),
    )
    field_control_value(state, CONFIG)  # warm the caches
    started = time.perf_counter()
    for _ in range(200):
        field_control_value(state, CONFIG)
    assert (time.perf_counter() - started) / 200 < 0.005


def test_speed_modifiers_round_like_showdown_so_ties_stay_ties() -> None:
    from vgc.field_control import _showdown_modify

    assert _showdown_modify(101, 1.5) == 151  # Scarfed 101 ties an unboosted 151
    assert _showdown_modify(151, 0.5) == 75  # paralysis rounds half down
    assert _showdown_modify(100, 2.0 * 1.5) == 300


# --- measured plan value (PolicyConfig.exact_search_field_measured_plan) ---------------------


def _plan_cache(packed: str, rain_gain: float) -> dict:
    from vgc import plan_value as pv

    gain = [[0.0] * 3 for _ in pv.CONDITIONS]
    gain[pv.CONDITIONS.index(("raindance", "none"))] = [rain_gain] * 3
    key = pv.set_key(pv.parse_packed_set(packed))
    return {"entries": {key: {"species": "archaludon", "mega": None, "gain": gain}}}


def test_measured_plan_credits_what_the_estimate_misses_and_falls_back_when_missing():
    from dataclasses import replace

    from vgc import plan_value as pv

    packed = "Archaludon||Leftovers|Stamina|ElectroShot,Protect|Modest|2,,,32,,32||||50|"
    pv.clear_registry()
    pv.register_own_team(packed, _plan_cache(packed, 40.0))
    on_config = replace(CONFIG, exact_search_field_measured_plan=True)
    theirs = [_mon("dragapult", ours=False), _mon("greninja", ours=False)]
    rain = (_effect("raindance", TURN - 1),)

    def value(config, ours):
        return field_control_value(_state(ours, theirs, weather=rain), config)

    ours = [_mon("archaludon", ("electroshot", "protect")), _mon("torkoal")]
    # 40 %HP/turn x fit weight 0.25 x (1 + .8 + .64) = ~24 points the estimate cannot see.
    assert value(on_config, ours) - value(CONFIG, ours) > 20
    # A bench Pokemon counts at the reserve weight, an unregistered one falls back and is counted.
    benched = [_mon("archaludon", ("electroshot", "protect"), active=False), _mon("torkoal")]
    half = value(on_config, benched) - value(CONFIG, benched)
    assert 8 < half < 16
    pv.FALLBACKS.clear()
    unknown = [_mon("kingdra", ("hydropump",)), _mon("milotic", ("muddywater",))]
    assert value(on_config, unknown) == value(CONFIG, unknown)
    assert pv.FALLBACKS["kingdra"] > 0
    pv.clear_registry()


# --- measured speed payoff (PolicyConfig.exact_search_field_measured_speed) -------------------


def test_measured_speed_payoff_replaces_generic_trick_room_term_and_falls_back_when_uncached():
    from dataclasses import replace

    from vgc import speed_payoff as sp

    sp.clear_registry()
    on_config = replace(CONFIG, exact_search_field_measured_speed=True)
    slow, fast, _ = _slow_vs_fast()
    room = (_effect("trickroom", TURN - 1),)
    state = _state(slow, fast, fields=room)
    # Nothing cached: every condition keeps the generic term, so the flag changes nothing.
    assert field_control_value(state, on_config) == field_control_value(state, CONFIG)
    # Cache all four Pokemon: Trick Room is now worth the AVERAGE of the two views instead.
    _cache_speed(
        {"torkoal": 20.0, "ursaluna": 20.0, "dragapult": -10.0, "greninja": -10.0}, field="tr"
    )
    bare = _state(slow, fast)
    # Each payoff is a net duel number, so ours (+20 +20) and theirs (-(-10 -10)) are two views
    # of one exchange: (40 + 20) / 2 = 30 %HP/turn, NOT 60. Swing = 30 x fit weight 0.25 x
    # (1 + .8 + .64 turns) = 18.3 on top of the bare board (the generic term is removed while
    # the room is measured).
    swing = field_control_value(state, on_config) - field_control_value(bare, on_config)
    assert swing == pytest.approx(0.25 * 30 * (1 + 0.8 + 0.64), abs=0.1)
    assert field_control_value(bare, on_config) == field_control_value(bare, CONFIG)
    sp.clear_registry()


def _cache_speed(values: dict[str, float], *, field: str) -> None:
    """Register measured entries for the test Pokemon, one payoff field set per species."""
    from vgc import speed_payoff as sp

    for species, value in values.items():
        args = {"tw": 0.0, "tw_against": 0.0, "tr": 0.0}
        args[field] = value
        entry = sp.SpeedEntry(species, **args)
        sp._TEAMS.setdefault("test-team", ({}, {}))[0][(species, frozenset(("protect",)))] = entry
        sp._USAGE[species] = entry


def test_overlapping_speed_controls_are_never_credited_as_a_measured_benefit():
    """Tailwind payoffs were measured on a bare field: two Tailwinds cancel, and Tailwind
    inside Trick Room is not a benefit (Codex review 2026-10-06)."""
    from dataclasses import replace

    from vgc import speed_payoff as sp

    sp.clear_registry()
    on_config = replace(CONFIG, exact_search_field_measured_speed=True)
    slow, fast, _ = _slow_vs_fast()
    tailwind = (_effect("tailwind", TURN - 1),)
    room = (_effect("trickroom", TURN - 1),)
    bare = _state(slow, fast)
    _cache_speed({"torkoal": 30.0, "ursaluna": 30.0, "dragapult": 30.0, "greninja": 30.0}, field="tw")

    def value(state, config=on_config):
        return field_control_value(state, config)

    # Our Tailwind alone is credited: both sets gain 30, the foes' own tw says nothing about
    # being hit by it (tw_against = 0), so the averaged value is (60 + 0) / 2 = 30.
    ours_only = _state(slow, fast, our_cond=tailwind)
    assert value(ours_only) - value(bare) == pytest.approx(0.25 * 30 * (1 + 0.8 + 0.64), abs=0.1)
    # Both sides under Tailwind: every Speed doubles, nothing changes, nothing is credited.
    both = _state(slow, fast, our_cond=tailwind, their_cond=tailwind)
    assert value(both) == pytest.approx(value(bare))
    assert value(both, CONFIG) == pytest.approx(value(bare, CONFIG), abs=1e-6)
    # Tailwind inside Trick Room: the generic term orders it (Tailwind makes us SLOWER under the
    # room), identically with the measured flag on or off, and no measured Tailwind credit.
    both_controls = _state(slow, fast, fields=room, our_cond=tailwind)
    assert value(both_controls) == pytest.approx(value(both_controls, CONFIG))
    sp.clear_registry()


def test_measured_caches_never_read_another_teams_set() -> None:
    """Both arms of an A/B share a process: a lookup must read only the bound team.

    Species+moves alone collided across 160 of 320 test teams (Codex review 2026-10-06).
    """
    from vgc import speed_payoff as sp
    from vgc.team_scope import bind_own_team, team_key
    import contextvars

    sp.clear_registry()
    mine = sp.SpeedEntry("incineroar", 0.0, 0.0, 5.0)
    theirs = sp.SpeedEntry("incineroar", 0.0, 0.0, 50.0)
    key = ("incineroar", frozenset(("fakeout",)))
    sp._TEAMS.setdefault(team_key("TEAM A"), ({}, {}))[0][key] = mine
    sp._TEAMS.setdefault(team_key("TEAM B"), ({}, {}))[0][key] = theirs

    def read(team: str | None):
        def inner():
            bind_own_team(team)
            return sp.lookup_own("incineroar", ("fakeout",))

        return contextvars.copy_context().run(inner)

    assert read("TEAM A") is mine
    assert read("TEAM B") is theirs
    assert read(None) is None  # two teams registered, none bound: no guessing
    sp.clear_registry()
