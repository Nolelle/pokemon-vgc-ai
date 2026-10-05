"""Critical behaviour of `vgc.field_control.field_control_value`."""

from __future__ import annotations

import time
from types import SimpleNamespace

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
