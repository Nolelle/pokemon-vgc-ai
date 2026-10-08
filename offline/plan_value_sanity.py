#!/usr/bin/env python
"""Print the measured plan-value gains for the owner's six teams, for eyeballing.

For each Pokemon on a team it prints the gain (in % of the reference foe's max HP per turn,
versus the bare field) in the conditions that team's plan actually uses, for the one-foe and
two-foe profiles, with the best move there; then the Pokemon's three best conditions overall.
A Pokemon whose plan condition shows ~0 is one the measured term will not credit; a big gain
should match what the owner knows the set is for (e.g. Archaludon in rain).

Reads `data/usage/plan_value_cache.json` (build it with `tools/build_plan_value_cache.py`).

Usage:
    .venv/bin/python offline/plan_value_sanity.py [team_name ...]
"""

from __future__ import annotations

import sys

from vgc import plan_value as pv
from vgc.config import REPO_ROOT

TEAM_DIR = REPO_ROOT / "teams" / "owner"

# The conditions each owner team's plan runs under (teams/owner/plans/*.md). Tailwind and
# Trick Room are speed control, not covered by the probe, so salamence_tw/hatterene_tr list
# the terrain/weather modes their plans also use.
PLAN_CONDITIONS: dict[str, list[tuple[str, str, str]]] = {
    "psyspam_sand": [
        ("psychic terrain", "none", "psychicterrain"),
        ("sand", "sandstorm", "none"),
        ("sand+psychic", "sandstorm", "psychicterrain"),
    ],
    "gardevoir_psyspam": [
        ("psychic terrain", "none", "psychicterrain"),
        ("sun", "sunnyday", "none"),
    ],
    "hatterene_tr": [
        ("sun (Eruption)", "sunnyday", "none"),
        ("psychic terrain", "none", "psychicterrain"),
    ],
    "coaching_baxcalibur": [
        ("grassy terrain", "none", "grassyterrain"),
        ("snow", "snowscape", "none"),
    ],
    "terrain_pulse_blastoise": [
        ("psychic terrain", "none", "psychicterrain"),
        ("grassy terrain", "none", "grassyterrain"),
        ("rain", "raindance", "none"),
    ],
    "salamence_tw": [
        ("sun", "sunnyday", "none"),
        ("rain", "raindance", "none"),
        ("psychic terrain", "none", "psychicterrain"),
    ],
}


def _move_name(entry: dict, condition: int, profile: int) -> str:
    index = entry["best"][condition][profile]
    return entry["tested"][index] if index >= 0 else "-"


def _cell(entry: dict, weather: str, terrain: str) -> str:
    c = pv.CONDITIONS.index((weather, terrain))
    single, double = entry["gain"][c][0], entry["gain"][c][1]
    move = _move_name(entry, c, 1)
    return f"{single:+6.1f} / {double:+6.1f}  ({move})"


def main() -> None:
    wanted = set(sys.argv[1:]) or set(PLAN_CONDITIONS)
    cache = pv.read_cache()
    for team_name, plan in PLAN_CONDITIONS.items():
        if team_name not in wanted:
            continue
        path = TEAM_DIR / f"{team_name}.packed.txt"
        print(f"\n=== {team_name}  (gain %HP/turn, one foe / two foes, best move on two foes)")
        for pset in pv.parse_packed_team(path.read_text()):
            entry = cache["entries"].get(pv.set_key(pset))
            if entry is None:
                print(f"  {pset.species_id:<18} (not in cache)")
                continue
            # A stone holder is measured twice: base form, then (its own rows) the Mega form.
            forms = [(pset.species_id, entry)]
            if entry.get("mega_form"):
                forms.append((f"{entry['mega']}", {**entry, **entry["mega_form"]}))
            for name, form in forms:
                base = form["per_turn"][0]
                print(f"  {name:<18} bare field {base[0]:5.1f} / {base[1]:5.1f}")
                for label, weather, terrain in plan:
                    print(f"      {label:<16} {_cell(form, weather, terrain)}")
                ranked = sorted(
                    range(1, len(pv.CONDITIONS)), key=lambda c: -form["gain"][c][1]
                )[:3]
                top = ", ".join(
                    f"{'+'.join(x for x in pv.CONDITIONS[c] if x != 'none')} "
                    f"{form['gain'][c][1]:+.1f}"
                    for c in ranked
                )
                print(f"      best overall: {top}")


if __name__ == "__main__":
    main()
