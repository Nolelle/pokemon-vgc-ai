#!/usr/bin/env python
"""Print the measured Tailwind / Trick Room payoff for every Pokemon on the owner's teams.

Columns (all %HP per turn, panel-averaged; see `vgc/speed_payoff.py`):
    spe     our Speed vs the panel's mean (first/last of the panel's pairings it outspeeds)
    TW      payoff when OUR side has Tailwind
    TW-vs   payoff when the FOE has Tailwind (negative = it hurts us)
    TR      payoff under Trick Room

Expect: fast attackers gain from Tailwind and lose under Trick Room; slow bulky attackers gain
under Trick Room; sets that already outspeed the whole panel gain little from our Tailwind.

Reads `data/usage/speed_payoff_cache.json` (build: `tools/build_speed_payoff_cache.py`).

Usage:
    .venv/bin/python offline/speed_payoff_sanity.py [team_name ...]
"""

from __future__ import annotations

import sys

from vgc import plan_value as pv
from vgc import speed_payoff as sp
from vgc.config import REPO_ROOT

TEAM_DIR = REPO_ROOT / "teams" / "owner"
TEAMS = (
    "salamence_tw",
    "hatterene_tr",
    "psyspam_sand",
    "gardevoir_psyspam",
    "coaching_baxcalibur",
    "terrain_pulse_blastoise",
)


def main() -> None:
    wanted = set(sys.argv[1:]) or set(TEAMS)
    cache = sp.read_cache()
    panel_id = cache.get("panel_hash", "")
    print("panel:", ", ".join(foe["species"] for foe in cache.get("panel", [])))
    for team in TEAMS:
        if team not in wanted:
            continue
        print(f"\n=== {team}  (%HP per turn vs panel; spe = outspeeds N of {len(cache['panel'])})")
        print(f"  {'species':<18}{'spe':>5}{'TW':>8}{'TW-vs':>8}{'TR':>8}   ours best vs panel")
        for pset in pv.parse_packed_team((TEAM_DIR / f"{team}.packed.txt").read_text()):
            entry = cache["entries"].get(sp.set_key(pset, panel_id))
            if entry is None:
                print(f"  {pset.species_id:<18} (not in cache)")
                continue
            outspeeds = sum(1 for ours, foe in entry["speeds"] if ours > foe)
            moves = sorted(set(entry["ours_move"]))
            print(
                f"  {pset.species_id:<18}{outspeeds:>5}{entry['tw']:>+8.1f}"
                f"{entry['tw_against']:>+8.1f}{entry['tr']:>+8.1f}   {','.join(moves)[:40]}"
            )


if __name__ == "__main__":
    main()
