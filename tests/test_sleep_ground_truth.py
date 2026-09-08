"""Champions-mod sleep timing checked against the real Showdown simulator."""

from __future__ import annotations

import pytest

from tests.sim_harness import pokeset, run_scenario

pytestmark = pytest.mark.integration


def _sleep_log(seed: tuple[int, int, int, int]) -> list[str]:
    """Spore a slower target, then let it try to act on the next two turns."""

    result = run_scenario(
        "gen9championscustomgame",
        [pokeset("Snorlax", ["Tackle"], ability="Immunity")],
        [pokeset("Gengar", ["Spore", "Protect"], ability="Cursed Body")],
        [
            "p1 team 1",
            "p2 team 1",
            "p1 move 1",
            "p2 move 1",
            "p1 move 1",
            "p2 move 2",
            "p1 move 1",
            "p2 move 2",
        ],
        seed=seed,
    )
    return result["log"]


def test_champions_sleep_has_only_two_or_three_denied_action_opportunities() -> None:
    # These fixed Showdown seeds deliberately cover both legal duration branches.
    short_sleep = _sleep_log((2, 2, 3, 4))
    long_sleep = _sleep_log((1, 2, 3, 4))

    assert sum("|cant|p1a: Snorlax|slp" in line for line in short_sleep) == 1
    assert sum("|cant|p1a: Snorlax|slp" in line for line in long_sleep) == 2
    assert any("|-curestatus|p1a: Snorlax|slp" in line for line in short_sleep)
    assert any("|-curestatus|p1a: Snorlax|slp" in line for line in long_sleep)
