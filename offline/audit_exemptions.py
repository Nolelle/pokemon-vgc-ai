"""The audit's documented exemptions, each a PREDICATE on one specific finding.

An exemption never excuses a field or a category. It excuses one mismatch when a named,
demonstrated cause holds for THAT Pokemon at THAT time (a specific move, ability or protocol
message in its history, a specific state of the truth). Anything that satisfies no predicate is
reported as unexplained, so a new defect in an already-excused field still fails the audit.

Every exemption has an id (used as the `[id]` prefix of the group in the report), a one-line
reason, a regex over the finding's `field/kind/detail` key, and the predicate. Tests in
`tests/test_battle_parsing_audit.py` build synthetic findings and assert both that the
documented case is exempt and that near-misses are not.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

Predicate = Callable[[dict[str, Any], dict[str, Any]], bool]


@dataclass(frozen=True)
class Exemption:
    id: str
    key: str  # regex matched against "field/kind/detail" (fullmatch)
    reason: str
    kind: str  # "hidden" (Showdown does not show it) | "limitation" (shown, not modelled)
    predicate: Predicate


def finding_key(finding: dict[str, Any]) -> str:
    return "/".join(p for p in (finding["field"], finding["kind"], finding["detail"]) if p)


REGISTRY: list[Exemption] = []


def register(*, id: str, key: str, reason: str, kind: str) -> Callable[[Predicate], Predicate]:
    def deco(fn: Predicate) -> Predicate:
        REGISTRY.append(Exemption(id, key, reason, kind, fn))
        return fn

    return deco


def classify(finding: dict[str, Any], cx: dict[str, Any]) -> str | None:
    """The id of the first exemption whose key matches AND whose predicate holds, else None."""
    key = finding_key(finding)
    for ex in REGISTRY:
        if re.fullmatch(ex.key, key) and ex.predicate(finding, cx):
            return ex.id
    return None


# --- registered exemptions --------------------------------------------------------------------

_EXTENDERS = {
    "sunnyday": "heatrock",
    "raindance": "damprock",
    "sandstorm": "smoothrock",
    "snowscape": "icyrock",
    "snow": "icyrock",
    "electricterrain": "terrainextender",
    "grassyterrain": "terrainextender",
    "psychicterrain": "terrainextender",
    "mistyterrain": "terrainextender",
    "reflect": "lightclay",
    "lightscreen": "lightclay",
    "auroraveil": "lightclay",
}


@register(
    id="illusion_name_collision",
    key=r".*",
    kind="limitation",
    reason=(
        "a live Illusion copies an ACTIVE ally (every benched Pokemon had fainted), so two "
        "on-field Pokemon share one protocol ident and poke-env's team dict (keyed by ident) "
        "holds a single entry for both"
    ),
)
def _illusion_name_collision(finding: dict[str, Any], cx: dict[str, Any]) -> bool:
    tm = cx.get("tm") or {}
    if tm.get("collision"):
        return True
    # After the break the one shared entry is still polluted by both Pokemon, until the next
    # line that restates its HP / status / boosts. Only the two names that collided are exempt.
    names = (cx.get("collided") or {}).get(cx.get("role"), set())
    if finding["field"] == "active_slots":
        slots = cx["truth"]["sides"][cx["role"]]["slots"]
        return any(name in names for name in slots if name)
    return cx.get("name") in names and finding["field"] in _COLLISION_FIELDS


_COLLISION_FIELDS = frozenset(
    {"hp", "boosts", "status", "status_counter", "active_flag", "volatile", "protect_counter",
     "foe_moves", "preparing", "fainted"}
)  # fmt: skip


@register(
    id="hidden_foe_extender",
    key=r"duration/mismatch/(weather|field|side):[a-z]+",
    kind="hidden",
    reason=(
        "Damp/Heat/Smooth/Icy Rock, Terrain Extender and Light Clay lengthen a condition by 3 "
        "turns, and the protocol never says so; it is exempt only while a foe Pokemon that "
        "set it holds the matching item and the bot has not been shown that item"
    ),
)
def _hidden_foe_extender(finding: dict[str, Any], cx: dict[str, Any]) -> bool:
    extra = finding["extra"]
    if finding["truth"] - finding["belief"] != 3 or extra.get("setter") != extra.get("foe_role"):
        return False
    effect = extra["condition"].split(":", 1)[1]
    extender = _EXTENDERS.get(effect)
    if extender is None:
        return False
    foe_role = extra["foe_role"]
    truth_mons = cx["truth"]["sides"][foe_role]["mons"]
    holders = [
        name for name, tm in truth_mons.items() if extender in (tm["item"], tm["last_item"])
    ]
    if not holders:
        return False
    shown = {
        mon.name
        for mon in cx["snapshot"].opponent_side.pokemon
        if mon.item_state == "known" and mon.item_id == extender
    }
    return any(name not in shown for name in holders)
