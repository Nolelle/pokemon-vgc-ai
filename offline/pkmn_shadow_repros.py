"""Minimal protocol repros for the @pkmn/client defects the shadow trial found.

Each case feeds a few real Showdown protocol lines (the audit's own wording) to a fresh
@pkmn/client `Battle` (our M-C data) and checks ONE fact against what Showdown's engine does.
A `DEFECT` line means the client's state is wrong after those lines; `ok` means it is right.
The expected values come from the engine (see each case's `why`), not from the client.

    .venv/bin/python offline/pkmn_shadow_repros.py
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pkmn_shadow  # noqa: E402

HEAD = [
    "|player|p1|Bot|1", "|player|p2|Opp|2", "|gametype|doubles", "|gen|9",
    "|tier|[Gen 9 Champions] VGC 2026 Reg M-C", "|rule|Species Clause: Limit one of each Pokemon",
    "|switch|p1a: Rillaboom|Rillaboom, L50, M|100/100",
    "|switch|p1b: Gourgeist|Gourgeist, L50, F|100/100",
    "|switch|p2a: Typhlosion|Typhlosion, L50, M|100/100",
    "|switch|p2b: Staraptor|Staraptor, L50, M|100/100",
]  # fmt: skip


@dataclass
class Case:
    name: str
    why: str
    lines: list[str]
    check: Callable[[dict[str, Any]], tuple[bool, Any]]  # (is_correct, observed)


def mon(snap: dict[str, Any], side: str, name: str) -> list[dict[str, Any]]:
    return [p for p in snap[side]["team"] if p["name"] == name]


CASES = [
    Case(
        "returning_pokemon_loses_its_history",
        "A Pokemon that U-turned out of slot b and comes back into slot a (empty after a faint) "
        "is the client's `lastPokemon`, which `getSwitchedPokemon` skips when the target slot is "
        "empty: a blank duplicate replaces it, dropping revealed moves/item/ability.",
        [
            "|move|p2b: Staraptor|Tackle|p1a: Rillaboom",
            "|faint|p2a: Typhlosion",
            "|move|p2b: Staraptor|U-turn|p1a: Rillaboom",
            "|switch|p2b: Lucario|Lucario, L50, M|100/100|[from] U-turn",
            "|switch|p2a: Staraptor|Staraptor, L50, M|80/100",
        ],
        lambda s: (
            sorted(m["id"] for m in mon(s, "p2", "Staraptor")[0]["moves"]) == ["tackle", "uturn"],
            [m["id"] for m in mon(s, "p2", "Staraptor")[0]["moves"]],
        ),
    ),
    Case(
        "worry_seed_ability_becomes_the_base_ability",
        "Worry Seed/Simple Beam/Entrainment change the ability for the current stint only "
        "(Showdown restores it on switch-out); the client's `-ability` handler also calls "
        "`activateAbility` without the not-base flag, so the unknown base ability is filled "
        "with the temporary one and survives the switch.",
        [
            "|move|p1a: Rillaboom|Worry Seed|p2a: Typhlosion",
            "|-ability|p2a: Typhlosion|Insomnia|Blaze|[from] move: Worry Seed",
            "|switch|p2a: Lucario|Lucario, L50, M|100/100",
            "|switch|p2a: Typhlosion|Typhlosion, L50, M|100/100",
        ],
        lambda s: (
            mon(s, "p2", "Typhlosion")[0]["ability"] != "insomnia",
            mon(s, "p2", "Typhlosion")[0]["ability"],
        ),
    ),
    Case(
        "plain_heal_resets_toxic_counter",
        "A `-heal` with no `[from]` (Leech Seed's drain, Recover, ...) is treated as Healing "
        "Wish/Lunar Dance and zeroes the toxic and sleep counters; Showdown's toxic stage is "
        "unaffected by healing.",
        [
            "|-status|p2a: Typhlosion|tox",
            "|upkeep",
            "|upkeep",
            "|-heal|p2a: Typhlosion|90/100 tox|[silent]",
        ],
        lambda s: (
            mon(s, "p2", "Typhlosion")[0]["toxicTurns"] == 2,
            mon(s, "p2", "Typhlosion")[0]["toxicTurns"],
        ),
    ),
    Case(
        "plain_heal_resets_sleep_counter",
        "Same root cause, sleep: two `cant|slp` turns then a silent heal.",
        [
            "|-status|p2a: Typhlosion|slp",
            "|cant|p2a: Typhlosion|slp",
            "|cant|p2a: Typhlosion|slp",
            "|-heal|p2a: Typhlosion|90/100 slp|[silent]",
        ],
        lambda s: (
            mon(s, "p2", "Typhlosion")[0]["sleepTurns"] == 2,
            mon(s, "p2", "Typhlosion")[0]["sleepTurns"],
        ),
    ),
    Case(
        "bug_bite_gives_the_attacker_the_eaten_berry_as_its_item",
        "Bug Bite/Pluck eat the TARGET's berry; the heal line `[from] item: Sitrus Berry` is on "
        "the attacker, and the client's `-heal` handler assigns the item to the healed Pokemon, "
        "replacing whatever the attacker really holds (Rillaboom holds nothing it showed).",
        [
            "|move|p1a: Rillaboom|Bug Bite|p2a: Typhlosion",
            "|-enditem|p2a: Typhlosion|Sitrus Berry|[from] stealeat|[move] Bug Bite|[of] p1a: Rillaboom",
            "|-heal|p1a: Rillaboom|60/100|[from] item: Sitrus Berry",
        ],
        lambda s: (mon(s, "p1", "Rillaboom")[0]["item"] == "", mon(s, "p1", "Rillaboom")[0]["item"]),
    ),
    Case(
        "symbiosis_item_pass_is_not_tracked",
        "`-activate|GIVER|ability: Symbiosis|ITEM|[of] RECEIVER`: the giver loses ITEM and the "
        "receiver holds it; the client records only the ability.",
        [
            "|move|p2b: Staraptor|Tackle|p1a: Rillaboom",
            "|-activate|p2a: Typhlosion|ability: Symbiosis|Quick Claw|[of] p2b: Staraptor",
        ],
        lambda s: (
            mon(s, "p2", "Staraptor")[0]["item"] == "quickclaw",
            mon(s, "p2", "Staraptor")[0]["item"],
        ),
    ),
    Case(
        "gastro_acid_leaves_no_volatile",
        "Gastro Acid sends only `-endability`; Showdown keeps a `gastroacid` volatile (ability "
        "suppressed until switch-out). The client blanks the ability but records no volatile, "
        "so its own `effectiveAbility()` gastroacid check can never fire.",
        ["|move|p1a: Rillaboom|Gastro Acid|p2a: Typhlosion", "|-endability|p2a: Typhlosion"],
        lambda s: (
            "gastroacid" in mon(s, "p2", "Typhlosion")[0]["volatiles"],
            sorted(mon(s, "p2", "Typhlosion")[0]["volatiles"]),
        ),
    ),
    Case(
        "trapped_volatile_never_ends",
        "`-activate|X|trapped` (Block/Mean Look) adds a `trapped` volatile; Showdown ends it "
        "silently when the trapper leaves, with no protocol line, so the client keeps it until "
        "the victim itself switches.",
        [
            "|move|p2b: Staraptor|Block|p1a: Rillaboom",
            "|-activate|p1a: Rillaboom|trapped",
            "|switch|p2b: Lucario|Lucario, L50, M|100/100",
        ],
        lambda s: (
            "trapped" not in mon(s, "p1", "Rillaboom")[0]["volatiles"],
            sorted(mon(s, "p1", "Rillaboom")[0]["volatiles"]),
        ),
    ),
    Case(
        "transform_copies_species_types_not_current_types",
        "Transform copies the target's CURRENT types (here Water, from Soak); the client's "
        "`copyTypesFrom(target, true)` uses the species' base types.",
        [
            "|move|p1a: Rillaboom|Soak|p1b: Gourgeist",
            "|-start|p1b: Gourgeist|typechange|Water|[from] move: Soak",
            "|move|p2a: Typhlosion|Transform|p1b: Gourgeist",
            "|-transform|p2a: Typhlosion|p1b: Gourgeist",
        ],
        lambda s: (
            mon(s, "p2", "Typhlosion")[0]["types"] == ["Water"],
            mon(s, "p2", "Typhlosion")[0]["types"],
        ),
    ),
    Case(
        "natural_cure_reveal_is_ignored",
        "`-curestatus|X|slp|[from] ability: Natural Cure` names the ability to both players; the "
        "client's `-curestatus` handler never reads `[from]`.",
        [
            "|-status|p2a: Typhlosion|slp",
            "|switch|p2a: Lucario|Lucario, L50, M|100/100",
            "|-curestatus|p2a: Typhlosion|slp|[from] ability: Natural Cure",
        ],
        lambda s: (
            mon(s, "p2", "Typhlosion")[0]["ability"] == "naturalcure",
            mon(s, "p2", "Typhlosion")[0]["ability"],
        ),
    ),
    Case(
        "yawn_dropped_when_any_status_lands",
        "Showdown keeps the (inert) `yawn` volatile until its timer ends even if the target "
        "meanwhile got another status; the client removes it in its `-status` handler.",
        [
            "|move|p1a: Rillaboom|Yawn|p2a: Typhlosion",
            "|-start|p2a: Typhlosion|move: Yawn",
            "|move|p1b: Gourgeist|Will-O-Wisp|p2a: Typhlosion",
            "|-status|p2a: Typhlosion|brn",
        ],
        lambda s: (
            "yawn" in mon(s, "p2", "Typhlosion")[0]["volatiles"],
            sorted(mon(s, "p2", "Typhlosion")[0]["volatiles"]),
        ),
    ),
    Case(
        "toxic_counter_is_not_capped",
        "Showdown caps the toxic stage at 15 (damage stops growing); the client's counter "
        "increments on every upkeep without a cap.",
        ["|-status|p2a: Typhlosion|tox"] + ["|upkeep"] * 20,
        lambda s: (
            mon(s, "p2", "Typhlosion")[0]["toxicTurns"] <= 15,
            mon(s, "p2", "Typhlosion")[0]["toxicTurns"],
        ),
    ),
    Case(
        "anger_point_ability_reveal_is_ignored",
        "`-setboost|X|atk|6|[from] ability: Anger Point` reveals the ability to both players; "
        "the client's `-setboost` handler never reads `[from]`.",
        [
            "|move|p1a: Rillaboom|Tackle|p2a: Typhlosion",
            "|-crit|p2a: Typhlosion",
            "|-setboost|p2a: Typhlosion|atk|6|[from] ability: Anger Point",
        ],
        lambda s: (
            mon(s, "p2", "Typhlosion")[0]["ability"] == "angerpoint",
            mon(s, "p2", "Typhlosion")[0]["ability"],
        ),
    ),
]


def run_case(reader: pkmn_shadow.PkmnReader, case: Case) -> tuple[bool, Any, list[Any]]:
    reader.request({"cmd": "open", "id": case.name, "perspective": "p1"})
    res = reader.request({"cmd": "feed", "id": case.name, "lines": HEAD + case.lines})
    snap = reader.request({"cmd": "snapshot", "id": case.name})["snapshot"]
    reader.request({"cmd": "close", "id": case.name})
    ok, observed = case.check(snap)
    return ok, observed, res["errors"]


def main() -> int:
    defects = 0
    with pkmn_shadow.PkmnReader() as reader:
        for case in CASES:
            ok, observed, errors = run_case(reader, case)
            defects += not ok
            print(f"{'ok    ' if ok else 'DEFECT'} {case.name}: observed {json.dumps(observed)}")
            if errors:
                print(f"       parse errors: {errors}")
            if not ok:
                print(f"       why: {case.why}")
    print(f"\n{defects}/{len(CASES)} repros show a client defect")
    return 0


if __name__ == "__main__":
    sys.exit(main())
