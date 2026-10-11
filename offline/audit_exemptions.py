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


@register(
    id="unrevealed_disguise_residue",
    key=r"(foe_moves|ability|item|hp|fainted|active_flag|active_slots|status|boosts|volatile|protect_counter)/.*",
    kind="hidden",
    reason=(
        "while a Zoroark's Illusion is up everything it shows (moves, ability, item, damage, even "
        "its faint) is announced under the disguise's name; if the episode ends WITHOUT the "
        "Illusion breaking (it switches out, or faints, which clears the disguise silently), "
        "nothing ever tells the client those records were the Zoroark's, not the named Pokemon's"
    ),
)
def _unrevealed_disguise_residue(finding: dict[str, Any], cx: dict[str, Any]) -> bool:
    names = (cx.get("deceived") or {}).get(cx.get("role"), set())
    if finding["field"] == "active_slots":
        shown = [n for n in list(finding["truth"] or []) + list(finding["belief"] or []) if n]
        return cx.get("label") == "foe" and any(n in names for n in shown)
    return cx.get("label") == "foe" and cx.get("name") in names


_COLLISION_FIELDS = frozenset(
    {"hp", "boosts", "status", "status_counter", "active_flag", "volatile", "protect_counter",
     "foe_moves", "preparing", "fainted", "item", "ability"}
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


@register(
    id="receiver_announced_not_applied",
    key=r"ability/foe_wrong/.*",
    kind="limitation",
    reason=(
        "Showdown announces `-ability|MON|NEW|OLD|[from] ability: Receiver|[of] ALLY` but its "
        "own dump keeps the Receiver / Power of Alchemy holder's ability unchanged, so the "
        "announced ability is not the truth"
    ),
)
def _receiver_announced(finding: dict[str, Any], cx: dict[str, Any]) -> bool:
    know = cx.get("know")
    return bool(
        know
        and cx.get("label") == "foe"
        and know.has_event(cx["role"], cx["name"], "receiver_announced")
        and finding["truth"] in ("receiver", "powerofalchemy")
    )


@register(
    id="rest_sleep_counter",
    key=r"status_counter/mismatch/slp",
    kind="limitation",
    reason=(
        "sleep caused by Rest: Showdown's counter is one ahead of the protocol-derived count "
        "on the first sleeping turn; exempt only for a Pokemon that used Rest, off by exactly +1"
    ),
)
def _rest_sleep(finding: dict[str, Any], cx: dict[str, Any]) -> bool:
    know = cx.get("know")
    return bool(
        know
        and know.has_event(cx["role"], cx["name"], "rest")
        and finding["truth"] - finding["belief"] == 1
    )


@register(
    id="hidden_ally_skill_swap",
    key=r"ability/(foe_wrong|foe_known_but_unrevealed|foe_revealed_but_unknown)/.*",
    kind="hidden",
    reason=(
        "an ALLY Skill Swap announces neither ability (`|-activate|A|Skill Swap|||[of] B`), so a "
        "foe that took part holds an ability the client cannot know, and what it revealed afterwards (the swapped-in ability, kept in BattleMemory as if base) stays unverifiable for the rest of the battle"
    ),
)
def _hidden_ally_skill_swap(finding: dict[str, Any], cx: dict[str, Any]) -> bool:
    know = cx.get("know")
    return bool(
        know and cx.get("label") == "foe" and know.has_event(cx["role"], cx["name"], "hidden_skillswap")
    )


@register(
    id="yawn_kept_through_rest",
    key=r"volatile/missing/yawn",
    kind="limitation",
    reason=(
        "Rest on a yawned Pokemon sends no `-end|Yawn`; Showdown keeps the (now inert) volatile "
        "while poke-env ends it with the sleep, so only a sleeping Pokemon is exempt"
    ),
)
def _yawn_kept(finding: dict[str, Any], cx: dict[str, Any]) -> bool:
    tm = cx.get("tm") or {}
    return tm.get("status") == "slp"


@register(
    id="illusion_holder_fainted_same_step",
    key=r"(fainted|active_flag|hp|active_slots|boosts|status)/.*",
    kind="hidden",
    reason=(
        "a Zoroark that arrives disguised and faints in the SAME step (entry hazard, Memento) "
        "was never visible as a disguise at either end of the step; its faint is announced under "
        "the disguise's name"
    ),
)
def _holder_fainted_same_step(finding: dict[str, Any], cx: dict[str, Any]) -> bool:
    if cx.get("label") != "foe":
        return False
    role = cx.get("role")
    mons = cx["truth"]["sides"][role]["mons"]
    fainted_illusion = [m for m in mons.values() if m["fainted"] and m["base_ability"] == "illusion"]
    if not fainted_illusion:
        return False
    name = cx.get("name")
    if finding["field"] == "active_slots":
        return any("faint" in ln for ln in cx.get("lines", []))
    return any(f"faint|{role}" in ln and f": {name}" in ln for ln in cx.get("lines", []))
