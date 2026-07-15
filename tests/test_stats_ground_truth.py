"""Ground-truth verification of `vgc.stats.calculate_stats` against the real sim.

Drives `[Gen 9 Champions] Custom Game` (debug-enabled, so `>eval` works) via
`tools/sim_probe.mjs` and dumps each Pokemon's actual `maxhp`/`storedStats` computed by
`data/mods/champions/scripts.ts::statModify`, then asserts our formula matches exactly
(not a range -- stat calculation is deterministic, unlike damage rolls).
"""

from __future__ import annotations

import json

import pytest

from vgc.config import FORMAT_ID
from vgc.stats import calculate_stats

from sim_harness import pokeset, run_scenario

pytestmark = pytest.mark.integration


def _dump_stats(team: list[dict]) -> list[dict]:
    result = run_scenario(
        "gen9championscustomgame",
        team,
        [pokeset("Wynaut", ["splash"])],
        [
            f"p1 team {''.join(str(i + 1) for i in range(len(team)))}",
            "p2 team 1",
            "eval JSON.stringify(p1.pokemon.map(p => "
            "({species: p.species.id, maxhp: p.maxhp, stats: p.storedStats})))",
        ],
    )
    (payload,) = result["evalResults"]
    # `>eval`'s result formatting (`Utils.visualize` on a string) wraps the JSON.stringify
    # output in a literal leading/trailing `"` WITHOUT escaping the JSON's own internal
    # `"` characters, so this isn't valid JSON-of-JSON -- strip the wrapper quotes and
    # parse the (already-valid) inner JSON directly.
    assert payload.startswith('"') and payload.endswith('"'), payload
    return json.loads(payload[1:-1])


@pytest.mark.parametrize(
    ("species", "sp", "nature"),
    [
        ("Garchomp", {"hp": 30, "atk": 4, "spe": 32}, "Adamant"),
        ("Pikachu", {}, "Serious"),
        ("Clefable", {"hp": 32, "def": 32, "spd": 2}, "Bold"),
        ("Torkoal", {"hp": 32, "spa": 32, "spd": 2}, "Quiet"),
    ],
)
def test_stat_formula_matches_sim(species: str, sp: dict[str, int], nature: str) -> None:
    team = [pokeset(species, ["splash"], nature=nature, sp=sp)]
    dumped = _dump_stats(team)[0]

    expected = calculate_stats(species.lower(), sp=sp, nature=nature.lower())

    assert dumped["maxhp"] == expected["hp"]
    for stat in ("atk", "def", "spa", "spd", "spe"):
        assert dumped["stats"][stat] == expected[stat], (
            f"{species} {stat}: sim={dumped['stats'][stat]} expected={expected[stat]}"
        )


def test_format_is_doubles_and_uses_champions_mod() -> None:
    # Sanity check this suite is actually exercising the mod this project targets, not
    # some other Showdown format -- see CLAUDE.md's "never assume vanilla gen9" warning.
    assert FORMAT_ID == "gen9championsvgc2026regmb"
