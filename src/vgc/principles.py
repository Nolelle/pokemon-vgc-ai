"""Explicit vocabulary for the Champions first-principles policy.

The evaluator used to recognize strategic ideas only when a one-off scorer happened to
encode them.  This module gives team preview, turn scoring, search, traces, and the
postmortem code one shared set of names for the things the policy is required to reason
about: engines, action denial, speed control, positioning, setup, redirection, and team
roles.  The classifications are deliberately transparent move/ability-id rules over the
Champions export rather than learned labels, so every verdict remains inspectable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from vgc.damage import to_id
from vgc.data import load_moves

SELF_PROTECT_MOVES = frozenset(
    {
        "protect",
        "detect",
        "kingsshield",
        "spikyshield",
        "banefulbunker",
        "burningbulwark",
        "silktrap",
        "obstruct",
    }
)
REDIRECTION_MOVES = frozenset({"followme", "ragepowder"})
SPEED_CONTROL_MOVES = frozenset(
    {
        "tailwind",
        "trickroom",
        "icywind",
        "electroweb",
        "bulldoze",
        "scaryface",
        "thunderwave",
        "stringshot",
    }
)
DIRECT_DENIAL_MOVES = frozenset(
    {
        "fakeout",
        "sleeppowder",
        "spore",
        "hypnosis",
        "sing",
        "yawn",
        "taunt",
        "encore",
        "disable",
        "imprison",
    }
)
SETUP_MOVES = frozenset(
    {
        "agility",
        "amnesia",
        "bellydrum",
        "bulkup",
        "calmmind",
        "coil",
        "dragondance",
        "growth",
        "howl",
        "irondefense",
        "nastyplot",
        "quiverdance",
        "shellsmash",
        "shiftgear",
        "swordsdance",
    }
)
SCREEN_MOVES = frozenset({"reflect", "lightscreen", "auroraveil"})
PIVOT_MOVES = frozenset(
    {"partingshot", "uturn", "voltswitch", "flipturn", "batonpass", "chillyreception"}
)
RECOVERY_MOVES = frozenset(
    {"recover", "roost", "slackoff", "softboiled", "synthesis", "moonlight", "morningsun"}
)
WIDE_DEFENSE_MOVES = frozenset({"wideguard", "quickguard", "matblock"})
SLEEP_MOVES = frozenset({"sleeppowder", "spore", "hypnosis", "sing"})
BURN_MOVES = frozenset({"willowisp"})

WEATHER_SETTER_ABILITIES = {
    "drizzle": "rain",
    "drought": "sun",
    "sandstream": "sand",
    "snowwarning": "snow",
}
WEATHER_SPEED_ABILITIES = {
    "swiftswim": "rain",
    "chlorophyll": "sun",
    "sandrush": "sand",
    "slushrush": "snow",
}


def normalized_move_ids(mon) -> frozenset[str]:
    moves = getattr(mon, "moves", None) or {}
    raw_ids = moves.keys() if hasattr(moves, "keys") else moves
    return frozenset(filter(None, (to_id(move_id) for move_id in raw_ids)))


def damaging_move_ids(mon) -> frozenset[str]:
    data = load_moves()
    return frozenset(
        move_id
        for move_id in normalized_move_ids(mon)
        if move_id in data and data[move_id].get("category") != "Status"
    )


def spread_move_ids(mon) -> frozenset[str]:
    data = load_moves()
    spread_targets = {"allAdjacent", "allAdjacentFoes", "all"}
    return frozenset(
        move_id
        for move_id in damaging_move_ids(mon)
        if data[move_id].get("target") in spread_targets
    )


def priority_move_ids(mon) -> frozenset[str]:
    data = load_moves()
    return frozenset(
        move_id
        for move_id in damaging_move_ids(mon)
        if int(data[move_id].get("priority", 0)) > 0
    )


def status_move_ids(mon) -> frozenset[str]:
    data = load_moves()
    return frozenset(
        move_id
        for move_id in normalized_move_ids(mon)
        if move_id in data and data[move_id].get("category") == "Status"
    )


@dataclass(frozen=True)
class MonRoles:
    attacker: bool
    speed_control: bool
    action_denial: bool
    redirection: bool
    setup: bool
    pivot: bool
    protection: bool
    spread_pressure: bool
    priority_pressure: bool

    def active_functions(self) -> frozenset[str]:
        return frozenset(
            name
            for name, enabled in (
                ("immediate_pressure", self.attacker),
                ("speed_control", self.speed_control),
                ("action_denial", self.action_denial),
                ("redirection", self.redirection),
                ("setup", self.setup),
                ("pivot", self.pivot),
                ("protection", self.protection),
                ("spread_pressure", self.spread_pressure),
                ("priority_pressure", self.priority_pressure),
            )
            if enabled
        )


def classify_mon(mon) -> MonRoles:
    move_ids = normalized_move_ids(mon)
    ability = to_id(getattr(mon, "ability", None))
    attacks = damaging_move_ids(mon)
    data = load_moves()
    meaningful_attacks = {
        move_id
        for move_id in attacks
        if int(data.get(move_id, {}).get("basePower", 0) or 0) >= 55
    }
    return MonRoles(
        attacker=bool(meaningful_attacks),
        speed_control=bool(move_ids & SPEED_CONTROL_MOVES),
        action_denial=bool(move_ids & DIRECT_DENIAL_MOVES),
        redirection=bool(move_ids & REDIRECTION_MOVES),
        setup=bool(move_ids & SETUP_MOVES),
        pivot=bool(move_ids & PIVOT_MOVES) or ability in {"intimidate", "hospitality"},
        protection=bool(move_ids & (SELF_PROTECT_MOVES | REDIRECTION_MOVES | WIDE_DEFENSE_MOVES)),
        spread_pressure=bool(spread_move_ids(mon)),
        priority_pressure=bool(priority_move_ids(mon)),
    )


@dataclass(frozen=True)
class TeamSignals:
    engines: frozenset[str]
    engine_enabler_indices: frozenset[int]
    roles_by_index: tuple[MonRoles, ...]

    def summary(self) -> dict[str, object]:
        return {
            "engines": sorted(self.engines),
            "engine_enablers": sorted(self.engine_enabler_indices),
            "roles": [sorted(roles.active_functions()) for roles in self.roles_by_index],
        }


def detect_team_signals(team: Iterable) -> TeamSignals:
    mons = list(team)
    roles = tuple(classify_mon(mon) for mon in mons)
    engines: set[str] = set()
    enablers: set[int] = set()

    abilities = [to_id(getattr(mon, "ability", None)) for mon in mons]
    moves = [normalized_move_ids(mon) for mon in mons]
    weather_setters = {
        weather: {idx for idx, ability in enumerate(abilities) if ability_id == ability}
        for ability_id, weather in WEATHER_SETTER_ABILITIES.items()
    }
    weather_abusers = {
        weather: {idx for idx, ability in enumerate(abilities) if ability_id == ability}
        for ability_id, weather in WEATHER_SPEED_ABILITIES.items()
    }
    for weather in ("rain", "sun", "sand", "snow"):
        if weather_setters.get(weather) and weather_abusers.get(weather):
            engines.add(weather)
            enablers.update(weather_setters[weather])

    for idx, move_ids in enumerate(moves):
        if "trickroom" in move_ids:
            engines.add("trick_room")
            enablers.add(idx)
        if "tailwind" in move_ids:
            engines.add("tailwind")
            enablers.add(idx)
        if move_ids & SCREEN_MOVES:
            engines.add("screens")
            enablers.add(idx)
        if move_ids & REDIRECTION_MOVES:
            engines.add("redirection_setup")
            enablers.add(idx)
        if move_ids & SETUP_MOVES:
            engines.add("setup")
        if move_ids & DIRECT_DENIAL_MOVES:
            engines.add("action_denial")
        if roles[idx].spread_pressure:
            engines.add("spread_offense")
        if roles[idx].priority_pressure:
            engines.add("priority_offense")
        if roles[idx].pivot:
            engines.add("pivot_cycle")
            if abilities[idx] in {"intimidate", "hospitality"}:
                enablers.add(idx)

    return TeamSignals(
        engines=frozenset(engines),
        engine_enabler_indices=frozenset(enablers),
        roles_by_index=roles,
    )


def utility_kind(move_id: str) -> str | None:
    move_id = to_id(move_id)
    if move_id in SELF_PROTECT_MOVES:
        return "protect"
    if move_id in REDIRECTION_MOVES:
        return "redirection"
    if move_id in SPEED_CONTROL_MOVES:
        return "speed_control"
    if move_id in DIRECT_DENIAL_MOVES:
        return "action_denial"
    if move_id in SETUP_MOVES:
        return "setup"
    if move_id in SCREEN_MOVES:
        return "screen"
    if move_id in PIVOT_MOVES:
        return "pivot"
    if move_id in BURN_MOVES:
        return "burn"
    if move_id in WIDE_DEFENSE_MOVES:
        return "wide_defense"
    if move_id in RECOVERY_MOVES:
        return "recovery"
    return None


# Utility families whose whole value comes from landing on an OPPONENT: putting it to
# sleep, taunting it, slowing it, burning it, dropping its stats. Aimed at our own ally
# (legal in doubles -- `normal`-target moves can pick the partner) the same effect is
# a loss for us, never a neutral no-op.
_FOE_DIRECTED_UTILITY_KINDS = frozenset({"action_denial", "speed_control", "burn", "pivot"})
_SINGLE_TARGET_KINDS = frozenset({"normal", "any", "adjacentFoe"})
# The deliberate exceptions: an ally ability that turns the "harmful" status into a
# benefit (Guts/Flare Boost/Quick Feet on a burn; Volt Absorb/Motor Drive/Lightning Rod
# absorb Thunder Wave, Guts/Quick Feet profit from paralysis; Contrary turns Parting
# Shot's and Scary Face's drops into boosts -- Defiant/Competitive do not trigger on an
# ally's drop). The policy does not model that benefit, so these are scored neutral
# rather than penalized -- the order stays available, it is just not rewarded.
_ALLY_BENEFIT_ABILITIES = {
    "willowisp": frozenset({"guts", "flareboost", "quickfeet"}),
    "thunderwave": frozenset({"voltabsorb", "motordrive", "lightningrod", "guts", "quickfeet"}),
    "partingshot": frozenset({"contrary"}),
    "scaryface": frozenset({"contrary"}),
}


def harms_ally_target(move_id: str, ally_ability: str | None = None) -> bool:
    """True if ``move_id`` aimed at our own ally is a self-inflicted harm.

    Covers single-target status moves the policy values as foe-directed utility (sleep,
    Taunt/Encore/Yawn/Disable, Thunder Wave/Scary Face, Will-O-Wisp, Parting Shot). An
    ally ability in ``_ALLY_BENEFIT_ABILITIES`` is the deliberate-reason exception.
    """
    move_id = to_id(move_id)
    move_data = load_moves().get(move_id) or {}
    if move_data.get("category") != "Status" or move_data.get("target") not in _SINGLE_TARGET_KINDS:
        return False
    if utility_kind(move_id) not in _FOE_DIRECTED_UTILITY_KINDS:
        return False
    return to_id(ally_ability or "") not in _ALLY_BENEFIT_ABILITIES.get(move_id, frozenset())


def is_speed_drop_attack(move_id: str) -> bool:
    return to_id(move_id) in {"icywind", "electroweb", "bulldoze"}

