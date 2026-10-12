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
    # Which battle-state readers the cause applies to. A "hidden" exemption (Showdown never
    # sends the information) is reader-independent; one that names a poke-env data structure
    # or repair is "pokeenv" only and must not excuse another reader's mismatch.
    readers: tuple[str, ...] = ("pokeenv", "pkmn")


def finding_key(finding: dict[str, Any]) -> str:
    return "/".join(p for p in (finding["field"], finding["kind"], finding["detail"]) if p)


REGISTRY: list[Exemption] = []


def register(
    *, id: str, key: str, reason: str, kind: str, readers: tuple[str, ...] = ("pokeenv", "pkmn")
) -> Callable[[Predicate], Predicate]:
    def deco(fn: Predicate) -> Predicate:
        REGISTRY.append(Exemption(id, key, reason, kind, fn, readers))
        return fn

    return deco


def classify(finding: dict[str, Any], cx: dict[str, Any]) -> str | None:
    """The id of the first exemption whose key matches AND whose predicate holds, else None."""
    key = finding_key(finding)
    reader = cx.get("reader", "pokeenv")
    for ex in REGISTRY:
        if reader in ex.readers and re.fullmatch(ex.key, key) and ex.predicate(finding, cx):
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
    readers=("pokeenv",),
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
     "foe_moves", "preparing", "fainted", "item", "ability", "types"}
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
        name
        for name, tm in truth_mons.items()
        if extender in (tm["item"], tm["last_item"], tm["set_item"])
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
    id="trace_mega_forme_unrecognised",
    key=r"species/mismatch/foe",
    kind="limitation",
    readers=("pokeenv",),
    reason=(
        "the bot recognises a Mega by its forme ability (poke-env does not rename the species); "
        "a Mega whose own ability is Trace (Meowstic-Mega, Alakazam-Mega) then shows another "
        "ability, so the match is lost. Follow-up: record the forme from the detailschange line"
    ),
)
def _trace_mega(finding: dict[str, Any], cx: dict[str, Any]) -> bool:
    from vgc.data import load_species

    entry = load_species().get(finding["truth"]) or {}
    abilities = {str(a).lower().replace(" ", "") for a in (entry.get("abilities") or {}).values()}
    return bool(entry.get("isMega")) and abilities == {"trace"}


@register(
    id="illusion_episode_within_one_step",
    key=r"(foe_moves|status|hp|ability|item|active_flag)/.*",
    kind="hidden",
    reason=(
        "a Zoroark that switched in disguised and broke in the SAME step is a disguise at neither "
        "end of it, so what it showed under the disguise's name cannot be attributed to it"
    ),
)
def _illusion_episode_within_one_step(finding: dict[str, Any], cx: dict[str, Any]) -> bool:
    know = cx.get("know")
    hit = bool(
        know
        and cx.get("label") == "foe"
        and f"illusion_break:{cx.get('role')}" in know.step_flags
    )
    if hit and cx.get("deceived") is not None and cx.get("name"):
        cx["deceived"][cx["role"]].add(cx["name"])  # the polluted record stays polluted
    return hit


@register(
    id="transformed_reveal_not_kept",
    key=r"ability/foe_revealed_but_unknown/.*",
    kind="limitation",
    readers=("pokeenv",),
    reason=(
        "a transformed Pokemon shows its TARGET's ability; poke-env keeps it only as a temporary "
        "ability that the snapshot reads back as unknown. Follow-up, not repaired"
    ),
)
def _transformed_reveal(finding: dict[str, Any], cx: dict[str, Any]) -> bool:
    know = cx.get("know")
    return bool(
        know
        and (
            know.has_event(cx["role"], cx["name"], "transformed")
            or know.has_event(cx["role"], cx["name"], "transform_target")
        )
    )


@register(
    id="bench_regenerator_heal",
    key=r"hp/foe_percent",
    kind="hidden",
    reason=(
        "Regenerator heals 1/3 on the way OUT without any protocol line (unless the ability was "
        "shown), so a benched foe's percentage is unknown until it returns"
    ),
)
def _bench_regenerator(finding: dict[str, Any], cx: dict[str, Any]) -> bool:
    tm = cx.get("tm") or {}
    return bool(
        cx.get("label") == "foe"
        and not tm.get("active")
        and "regenerator" in (tm.get("ability"), tm.get("base_ability"))
        and finding["truth"] > finding["belief"]
    )


@register(
    id="called_protect_counter",
    key=r"protect_counter/mismatch",
    kind="limitation",
    readers=("pokeenv",),
    reason=(
        "a Protect-family move CALLED by Copycat / Instruct / Sleep Talk advances Showdown's stall "
        "counter in ways the bot's per-move counter does not model; follow-up, not repaired"
    ),
)
def _called_protect(finding: dict[str, Any], cx: dict[str, Any]) -> bool:
    know = cx.get("know")
    return bool(know and know.has_event(cx["role"], cx["name"], "called_protect"))


@register(
    id="psych_up_from_disguise",
    key=r"boosts/mismatch/.*",
    kind="hidden",
    reason=(
        "Psych Up copies the boosts of the Pokemon it really hit; when that is a disguised "
        "Zoroark the client attributed them to the disguise's name, not to the Zoroark"
    ),
)
def _psych_up_from_disguise(finding: dict[str, Any], cx: dict[str, Any]) -> bool:
    if not any("-copyboost" in ln for ln in cx.get("lines", [])):
        return False
    prev = cx.get("prev_truth") or {}
    for side in (prev.get("sides") or {}).values():
        for tm in side["mons"].values():
            disguise = tm.get("illusion_name")
            if disguise and any(f": {disguise}" in ln for ln in cx["lines"] if "-copyboost" in ln):
                return True
    return False


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
    if cx.get("label") != "foe" or not cx.get("truth"):
        return False
    role = cx.get("role")
    mons = cx["truth"]["sides"][role]["mons"]
    fainted_illusion = [m for m in mons.values() if m["fainted"] and m["base_ability"] == "illusion"]
    if not fainted_illusion:
        return False
    name = cx.get("name")
    if name in {m["name"] for m in fainted_illusion} and any(
        f"faint|{role}" in ln for ln in cx.get("lines", [])
    ):
        # the holder itself: its faint was announced under some disguise's name
        if cx.get("deceived") is not None:
            cx["deceived"][role].add(name)
        return True
    if finding["field"] == "active_slots":
        return any("faint" in ln for ln in cx.get("lines", []))
    hit = any(f"faint|{role}" in ln and f": {name}" in ln for ln in cx.get("lines", []))
    if hit and cx.get("deceived") is not None:
        cx["deceived"][role].add(name)  # the record stays wrong on the following steps too
    return hit


@register(
    id="pkmn_not_tracked",
    key=r"(preparing|protect_counter)/.*",
    kind="limitation",
    readers=("pkmn",),
    reason=(
        "@pkmn/client has no field for these: it never handles `|-prepare|` (two-turn move "
        "charge state) and keeps no Protect/stall counter (only a one-turn `protect` volatile). "
        "Reported as NOT TRACKED, not as correct: a replacement reader would have to add them"
    ),
)
def _pkmn_not_tracked(finding: dict[str, Any], cx: dict[str, Any]) -> bool:
    return True
