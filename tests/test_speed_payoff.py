"""Critical behaviour of the measured speed payoff (`vgc.speed_payoff`, the probe).

Integration-marked: it runs the local Showdown engine (no server). Sets are not legality-
checked -- the probe plays Custom Game. Payoffs are read per reference foe (`per_foe`), so each
case is one duel whose answer is a matter of who moves first and what that move does.
"""

from __future__ import annotations

import pytest

from vgc import speed_payoff as sp

pytestmark = pytest.mark.integration

# Fast, strong foe: outspeeds the slow attacker with or without Tailwind.
FAST_FOE = "Salamence-Mega||Salamencite|Intimidate|DoubleEdge,HyperVoice|Hasty|2,32,,,,32||||50|"
# Slow, hard-hitting foe that a fast attacker outspeeds on a bare field.
SLOW_FOE = "Kingambit||ChoiceBand|Defiant|KowtowCleave,IronHead|Adamant|32,32,,,2,||||50|"
# Slower than Kingambit; Tailwind makes it faster.
SLOWER_FOE = "Hatterene||LifeOrb|MagicBounce|Moonblast,DazzlingGleam|Quiet|27,,7,32,,|||S||"
FAST_ATTACKER = "Garchomp||Leftovers|RoughSkin|DragonClaw,RockSlide|Jolly|2,32,,,,32|F|||50|"
SLOW_ATTACKER = "Kingambit||ChopleBerry|Defiant|KowtowCleave,IronHead|Adamant|32,32,,,2,||||50|"


def _panel(*packed: str) -> list[dict]:
    return [{"species": text.split("|")[0], "packed": text, "weight": 1.0} for text in packed]


@pytest.fixture(scope="module")
def measure():
    with sp.ProbeWorker() as worker:
        yield lambda subject, *foes: worker.measure(
            sp.parse_packed_set(subject), _panel(*foes)
        )


def test_slow_attacker_gains_under_trick_room_and_loses_to_foe_tailwind(measure):
    # Kingambit is slower than Salamence-Mega but faster than Hatterene.
    entry = measure(SLOW_ATTACKER, FAST_FOE, SLOWER_FOE)
    assert entry["per_foe"]["tr"][0] > 5, entry["per_foe"]  # moves before Salamence under TR
    assert entry["per_foe"]["tw_against"][1] < -5, entry["per_foe"]  # Hatterene Tailwind


def test_fast_attacker_gains_under_tailwind_and_loses_under_trick_room(measure):
    # Garchomp outspeeds Kingambit, but not Mega Salamence.
    entry = measure(FAST_ATTACKER, SLOW_FOE, FAST_FOE)
    assert entry["per_foe"]["tr"][0] < -5, entry["per_foe"]  # Kingambit now hits first
    assert entry["per_foe"]["tw"][1] > 5, entry["per_foe"]  # Garchomp now beats Salamence


def test_speed_payoff_needs_no_outcome_change_when_order_is_unchanged(measure):
    # Salamence-Mega already outspeeds Kingambit: its own Tailwind changes nothing.
    entry = measure(FAST_FOE, SLOW_FOE)
    assert entry["per_foe"]["tw"][0] == pytest.approx(0.0, abs=0.01)


def test_cache_key_is_stable_and_sensitive():
    pset = sp.parse_packed_set(FAST_ATTACKER)
    same = sp.parse_packed_set(FAST_ATTACKER.replace("DragonClaw,RockSlide", "RockSlide,DragonClaw"))
    other = sp.parse_packed_set(FAST_ATTACKER.replace("Jolly", "Adamant"))
    assert sp.set_key(pset, "abc") == sp.set_key(same, "abc")  # move order never matters
    assert sp.set_key(pset, "abc") != sp.set_key(other, "abc")
    assert sp.set_key(pset, "abc") != sp.set_key(pset, "abd")  # a new panel invalidates
