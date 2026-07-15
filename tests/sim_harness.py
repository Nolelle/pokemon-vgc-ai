"""Shared helper for driving `tools/sim_probe.mjs` from integration tests.

Used by both `tests/test_stats_ground_truth.py` and `tests/test_damage_ground_truth.py`.
Not a test module itself (no `test_` prefix, not collected by pytest).

Key fact this whole harness leans on: `[Gen 9 Champions] Custom Game`'s ruleset
(`config/formats.ts`) is `['Team Preview', 'Cancel Mod', 'Max Team Size = 24', 'Max Move
Count = 24', 'Max Level = 9999', 'Default Level = 50']` -- notably NOT `Standard`/
`Obtainable`, so `BattleStream` (unlike the TeamValidator `tools/export_champions_data.py`
and real ladder play go through) does not validate move legality, ability legality, or
item-species pairing at all. Confirmed empirically: a team spec giving Garchomp "Ice Beam"
and "Tinted Lens" (neither of which it can legally have) was accepted and simulated
correctly. This lets ground-truth cases freely mix whatever move/item/ability isolates
the mechanic under test onto a convenient, high-stat species, without being constrained by
`data/champions/learnsets.json` or real ability pairings -- exactly like a standard damage
calculator's "what if" mode.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

from vgc.config import REPO_ROOT, SHOWDOWN_REPO

SIM_PROBE = REPO_ROOT / "tools" / "sim_probe.mjs"

FULL_IVS = {"hp": 31, "atk": 31, "def": 31, "spa": 31, "spd": 31, "spe": 31}


def find_node() -> str:
    node = shutil.which("node")
    if node:
        return node
    # node is not always on PATH in fresh shells on this machine (see CLAUDE.md).
    fallback = Path.home() / ".nvm/versions/node/v22.22.0/bin/node"
    if fallback.exists():
        return str(fallback)
    raise FileNotFoundError("node not found on PATH and no nvm fallback exists")


def pokeset(
    species: str,
    moves: list[str],
    *,
    nature: str = "Serious",
    sp: dict[str, int] | None = None,
    ability: str = "",
    item: str = "",
    level: int = 50,
    gender: str = "M",
) -> dict:
    """Build a Teams.pack-compatible PokemonSet dict. `sp` is a Stat Point spread (see
    `vgc.stats`), e.g. `{"hp": 32, "atk": 32}` -- missing stats default to 0, matching
    `vgc.stats.calculate_stats`'s own default.
    """
    sp = sp or {}
    evs = {stat: sp.get(stat, 0) for stat in ("hp", "atk", "def", "spa", "spd", "spe")}
    return {
        "species": species,
        "name": species,
        "item": item,
        "ability": ability,
        "moves": moves,
        "nature": nature,
        "evs": evs,
        "ivs": dict(FULL_IVS),
        "level": level,
        "gender": gender,
    }


def run_scenario(
    format_id: str,
    p1_team: list[dict],
    p2_team: list[dict],
    commands: list[str],
    seed: tuple[int, int, int, int] = (1, 2, 3, 4),
) -> dict:
    """Run `tools/sim_probe.mjs` against the local showdown repo and return its parsed
    `{"log": [...], "evalResults": [...]}` output.
    """
    scenario = {
        "format": format_id,
        "seed": list(seed),
        "p1": {"name": "p1", "team": p1_team},
        "p2": {"name": "p2", "team": p2_team},
        "commands": commands,
    }
    result = subprocess.run(
        [find_node(), str(SIM_PROBE), str(SHOWDOWN_REPO), "-"],
        input=json.dumps(scenario),
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"sim_probe.mjs failed:\nstdout={result.stdout}\nstderr={result.stderr}")
    return json.loads(result.stdout)


_SWITCH_RE = re.compile(r"^\|(?:switch|drag)\|(?P<slot>p\d[ab]): [^|]*\|[^|]*\|(?P<hp>\d+)/(?P<max>\d+)")
_DAMAGE_RE = re.compile(r"^\|-damage\|(?P<slot>p\d[ab]): [^|]*\|(?P<hp>\d+)/(?P<max>\d+)")


def first_damage_delta(log: list[str], slot: str) -> int:
    """HP lost by `slot` (e.g. `"p2a"`) at its FIRST `|-damage|` line after its most
    recent `|switch|`/`|drag|` -- i.e. the damage from the single hit a ground-truth
    case cares about, ignoring any later residual (burn/weather/etc) damage that shows
    up further down the same log.
    """
    max_hp = None
    for line in log:
        switch_match = _SWITCH_RE.match(line)
        if switch_match and switch_match.group("slot") == slot:
            max_hp = int(switch_match.group("max"))
            continue
        damage_match = _DAMAGE_RE.match(line)
        if damage_match and damage_match.group("slot") == slot and max_hp is not None:
            return max_hp - int(damage_match.group("hp"))
    raise AssertionError(f"no -damage line found for {slot} in log:\n" + "\n".join(log))


def max_hp_from_switch(log: list[str], slot: str) -> int:
    for line in log:
        switch_match = _SWITCH_RE.match(line)
        if switch_match and switch_match.group("slot") == slot:
            return int(switch_match.group("max"))
    raise AssertionError(f"no switch-in found for {slot} in log:\n" + "\n".join(log))
