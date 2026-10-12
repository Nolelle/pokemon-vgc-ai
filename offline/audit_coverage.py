"""Which legal Champions entries did the audit battles actually exercise?

Three tiers per category: `present` (an ability / item on an active Pokemon in the dump),
`triggered` (protocol evidence the effect fired), and for moves `used` (a `|move|` line). The
legal sets come from data/champions (moves.json and items.json: `isNonstandard` null; abilities
from mechanics_catalog.json, there is no abilities.json), protocol message types from scanning
the Showdown checkout's sim/ and data/ for `add('-tag'` calls, and conditions from the move
data's volatileStatus / sideCondition / slotCondition plus the field conditions.
"""

from __future__ import annotations

import json
import re
from collections import Counter

from vgc.config import REPO_ROOT, SHOWDOWN_REPO

DATA = REPO_ROOT / "data" / "champions"


def legal_sets() -> dict[str, set[str]]:
    moves = json.loads((DATA / "moves.json").read_text())
    items = json.loads((DATA / "items.json").read_text())
    catalog = json.loads((DATA / "mechanics_catalog.json").read_text())
    volatiles, sides, slots = set(), set(), set()
    for m in moves.values():
        if m.get("isNonstandard") is not None:
            continue
        for key in ("volatileStatus", "selfVolatileStatus"):
            if m.get(key):
                volatiles.add(m[key])
        volatiles.update(m.get("secondaryVolatileStatuses") or [])
        if m.get("sideCondition"):
            sides.add(m["sideCondition"])
        if m.get("slotCondition"):
            slots.add(m["slotCondition"])
    return {
        "moves": {k for k, v in moves.items() if v.get("isNonstandard") is None},
        "abilities": set(catalog["legal_entities"]["abilities"]),
        "items": set(items),
        "volatiles": volatiles,
        "side_conditions": sides,
        "slot_conditions": slots,
        "field_conditions": {
            "raindance", "sunnyday", "sandstorm", "snowscape", "electricterrain", "grassyterrain",
            "mistyterrain", "psychicterrain", "trickroom", "gravity", "magicroom", "wonderroom",
            "fairylock", "echoedvoice", "mudsport", "watersport", "iondeluge",
        },  # fmt: skip
    }


def protocol_tags() -> set[str]:
    tags: set[str] = set()
    pattern = re.compile(r"""(?:this|battle)\.add(?:Move)?\(\s*['"](-?[a-z]+)['"]""")
    for sub in ("sim", "data"):
        root = SHOWDOWN_REPO / sub
        if not root.exists():
            continue
        for path in root.rglob("*.ts"):
            try:
                tags.update(pattern.findall(path.read_text(errors="ignore")))
            except OSError:
                continue
    return tags


def report_text(cov: dict[str, Counter]) -> str:
    legal = legal_sets()
    lines = ["== COVERAGE (legal entries exercised in the audit battles)"]

    def row(label: str, universe: set[str], hit: set[str]) -> None:
        hit = hit & universe
        pct = 100 * len(hit) / max(1, len(universe))
        lines.append(f"{label:28s} {len(hit):4d}/{len(universe):4d} = {pct:5.1f}%")
        missing = sorted(universe - hit)
        if missing:
            lines.append(f"    missing: {', '.join(missing[:60])}{' ...' if len(missing) > 60 else ''}")

    row("moves used", legal["moves"], set(cov["move_used"]))
    row("abilities present", legal["abilities"], set(cov["ability_present"]))
    row("abilities triggered", legal["abilities"], set(cov["ability_trigger"]))
    row("items present", legal["items"], set(cov["item_present"]))
    row("items triggered", legal["items"], set(cov["item_trigger"]))
    row("volatiles seen", {v.lower() for v in legal["volatiles"]}, set(cov["volatile"]))
    row("side conditions", {v.lower() for v in legal["side_conditions"]}, set(cov["side_condition"]))
    row("slot conditions", {v.lower() for v in legal["slot_conditions"]}, set(cov["slot_condition"]))
    row("field conditions", legal["field_conditions"], set(cov["field_condition"]))
    tags = {t for t in protocol_tags() if t}
    seen = {t for t in cov["tags"]}
    row("protocol message types", tags, seen)
    return "\n".join(lines)


def coverage_percentages(cov: dict[str, Counter]) -> dict[str, float]:
    legal = legal_sets()
    pairs = {
        "moves_used": (legal["moves"], "move_used"),
        "abilities_present": (legal["abilities"], "ability_present"),
        "items_present": (legal["items"], "item_present"),
    }
    return {
        key: 100 * len(set(cov[name]) & universe) / max(1, len(universe))
        for key, (universe, name) in pairs.items()
    }
