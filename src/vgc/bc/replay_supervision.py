"""Conservative observed-action labels from spectator logs; never certified orders.

The masks describe individual observations, not complete legal or submitted choices.
No private information is reconstructed and no target-choice labels are enabled.
"""
from __future__ import annotations

import html
import json
import re
from functools import lru_cache
from pathlib import Path

_SEATS = ("p1", "p2")
_TASKS = ("move", "target", "switch_destination", "preview_brought", "preview_leads", "outcome")
# Protocol effects that can prevent a fresh choice; move properties come from the
# Champions catalog. Never search chat, nicknames, or usernames for move names.
_LOCK_EFFECTS = {"encore", "lockedmove", "twoturnmove", "mustrecharge", "recharge",
                 "bide", "uproar", "rollout", "iceball"}


def _id(value):
    return re.sub(r"[^a-z0-9]", "", html.unescape(value).lower()) if isinstance(value, str) else ""


@lru_cache(maxsize=1)
def _species_data():
    path = Path(__file__).resolve().parents[3] / "data/champions/species.json"
    return json.loads(path.read_text())


@lru_cache(maxsize=1)
def _move_data():
    path = Path(__file__).resolve().parents[3] / "data/champions/moves.json"
    return json.loads(path.read_text())


def _move_can_force_later_choice(move_id):
    move = _move_data().get(move_id, {})
    return bool(move.get("flags", {}).get("charge") or move.get("flags", {}).get("recharge")
                or move.get("selfVolatileStatus") in _LOCK_EFFECTS
                or move.get("volatileStatus") in _LOCK_EFFECTS
                or move_id in {"rollout", "iceball"})


def _identity(details, species):
    identity = _id(details.split(",")[0])
    entry = species.get(identity)
    if not isinstance(entry, dict):
        return None
    parent = entry.get("battleOnly")
    if isinstance(parent, list):
        parent = parent[0] if parent else None
    if entry.get("isMega"):
        parent = parent or entry.get("baseSpecies")
    return _id(parent) if parent else identity


def audit_raw_log(log: str) -> dict:
    """Read per-turn evidence, independent preview appearances, and ending evidence."""
    result = {"turns": {}, "preview": {}, "winner": None, "ending": "unresolved",
              "diagnostics": []}
    if not isinstance(log, str):
        result["diagnostics"].append("invalid_log")
        return result
    species = _species_data()
    names, appeared, leads = {}, {s: [] for s in _SEATS}, {s: [] for s in _SEATS}
    turn = 0
    # A game-wide exclusion is conservative when a lock begins before the current turn.
    locked = False
    blocked_switch = False
    winner_name = None
    for line in log.splitlines():
        p = line.split("|")
        if len(p) < 2:
            continue
        tag = p[1]
        if tag in ("move", "cant"):
            index = 3 if tag == "move" else 4
            if len(p) > index and _move_can_force_later_choice(_id(p[index])):
                locked = True
            if any(x.lower().startswith("[from]") for x in p[index + 1:]):
                locked = True
        if tag in ("-start", "-activate", "cant") and len(p) > 3:
            effect = _id(p[3].removeprefix("move: "))
            if effect in _LOCK_EFFECTS:
                locked = True
        if tag == "-prepare":
            locked = True
        # Entry abilities/items can cause a replacement before any move resolves.
        # Excluding switches after any such event is deliberately conservative.
        if tag.startswith("-") or tag in ("replace", "swap"):
            blocked_switch = True
        if tag == "player" and len(p) > 3 and p[2] in _SEATS and _id(p[3]):
            names.setdefault(p[2], _id(p[3]))
        elif tag == "turn" and len(p) > 2:
            try:
                turn = int(p[2])
                if turn < 1:
                    raise ValueError
            except ValueError:
                result["diagnostics"].append("invalid_turn")
                turn = 0
            blocked_switch = False
        elif tag == "win" and len(p) > 2:
            winner_name = _id(p[2])
        elif tag in ("switch", "drag", "replace") and len(p) > 3:
            pos = re.match(r"^(p[12])([ab]):", p[2])
            if not pos:
                result["diagnostics"].append("invalid_position")
                continue
            side, slot = pos.group(1), str("ab".index(pos.group(2)))
            identity = _identity(p[3], species)
            if identity and identity not in appeared[side]:
                appeared[side].append(identity)
            if not turn and tag == "switch" and identity and len(leads[side]) < 2:
                leads[side].append(identity)
            if turn:
                ev = result["turns"].setdefault(str(turn), {}).setdefault(side, {}).setdefault(slot, [])
                ev.append({"tag": tag, "species": identity,
                           "early_switch": tag == "switch" and not blocked_switch})
            if tag == "drag":
                blocked_switch = True
        elif tag in ("move", "cant", "faint") and len(p) > 2:
            pos = re.match(r"^(p[12])([ab]):", p[2])
            if turn and pos:
                ev = result["turns"].setdefault(str(turn), {}).setdefault(pos.group(1), {}).setdefault(str("ab".index(pos.group(2))), [])
                ev.append({"tag": tag, "move": _id(p[3]) if len(p) > 3 else None,
                           "called": any("[from]" in x.lower() for x in p[4:])})
            blocked_switch = True
    result["preview"] = {s: {"picked": appeared[s][:4], "lead_order": leads[s]} for s in _SEATS}
    result["locked_or_called"] = locked
    matches = [s for s in _SEATS if winner_name and names.get(s) == winner_name]
    result["winner"] = matches[0] if len(matches) == 1 else None
    if re.search(r"^\|-message\|[^\n]*(lost due to inactivity|lost by being offline)", log, re.M | re.I):
        result["ending"] = "timer_offline"
    elif re.search(r"^\|-message\|[^\n]*forfeited\.", log, re.M | re.I):
        result["ending"] = "forfeit"
    elif re.search(r"^\|tie(?:\||$)", log, re.M):
        result["ending"] = "draw"
    elif result["winner"]:
        result["ending"] = "played_out"
    return result


def _mapping(value):
    return value if isinstance(value, dict) else {}


def build_supervision(record: dict, game_metadata: dict, audit: dict) -> dict:
    """Return partial labels and masks; disabled labels are always null."""
    targets = {k: [None, None] if k in _TASKS[:3] else None for k in _TASKS}
    masks = {k: [False, False] if k in _TASKS[:3] else False for k in _TASKS}
    reasons = {k: ["not_observed", "not_observed"] if k in _TASKS[:3] else "not_observed" for k in _TASKS}
    result = {"targets": targets, "masks": masks, "reasons": reasons,
              "kind": record.get("decision_kind") if isinstance(record, dict) else None,
              "supervision": "observed_partial_v1", "joint_choice_certified": False}
    if not all(isinstance(x, dict) for x in (record, game_metadata, audit)):
        return result
    kind, side, state = record.get("decision_kind"), record.get("player"), record.get("state")
    if (kind not in ("turn", "forced_switch", "teampreview") or side not in _SEATS
            or not isinstance(state, dict) or any(not isinstance(state.get(s), dict) for s in ("our", "opp", "field"))):
        return result
    action = record.get("action")
    if not isinstance(action, dict):
        action = {}
    if kind == "turn":
        turn = record.get("turn")
        events = _mapping(_mapping(_mapping(audit.get("turns")).get(str(turn))).get(side))
        statuses = _mapping(record.get("action_status"))
        for i in range(2):
            key = f"slot{i}"
            a = action.get(key)
            ev = events.get(str(i), [])
            ev = ev if isinstance(ev, list) and all(isinstance(x, dict) for x in ev) else []
            reasons["target"][i] = "submitted_target_not_certifiable"
            if not isinstance(a, dict) or statuses.get(key) != "observed":
                continue
            if a.get("kind") == "move":
                matching = [x for x in ev if x.get("tag") == "move"]
                if (not audit.get("locked_or_called", True) and len(matching) == 1
                        and not matching[0].get("called") and _id(a.get("move_id"))
                        and _id(a.get("move_id")) in _move_data()
                        and _id(a.get("move_id")) == matching[0].get("move")):
                    targets["move"][i] = a["move_id"]
                    masks["move"][i] = True
                    reasons["move"][i] = "observed_move_not_certified_choice"
                else:
                    reasons["move"][i] = "raw_mismatch_or_called_or_locked"
            elif a.get("kind") == "switch":
                if (ev and ev[0].get("tag") == "switch" and ev[0].get("early_switch")
                        and ev[0].get("species") and ev[0]["species"] == a.get("switch_species")):
                    targets["switch_destination"][i] = a["switch_species"]
                    masks["switch_destination"][i] = True
                    reasons["switch_destination"][i] = "observed_early_switch_not_certified_choice"
                else:
                    reasons["switch_destination"][i] = "raw_mismatch_or_involuntary_switch"
    elif kind == "teampreview":
        expected = _mapping(_mapping(game_metadata.get("expected_preview")).get(side))
        raw = _mapping(_mapping(audit.get("preview")).get(side))
        preview = state["our"].get("preview")
        picked, leads = action.get("picked"), action.get("lead_order")
        def valid(items, n):
            return (isinstance(items, list) and len(items) == n
                    and all(isinstance(x, str) and x for x in items) and len(set(items)) == n
                    and isinstance(preview, list) and set(items) <= set(preview))
        if (valid(picked, 4) and picked == raw.get("picked") == expected.get("picked")):
            targets["preview_brought"] = sorted(picked)
            masks["preview_brought"] = True
            reasons["preview_brought"] = "observed_appearances"
        if (valid(leads, 2) and leads == raw.get("lead_order") == expected.get("lead_order")
                and isinstance(picked, list) and picked[:2] == leads):
            targets["preview_leads"] = list(leads)
            masks["preview_leads"] = True
            reasons["preview_leads"] = "observed_opening_leads"
    winner, ending = game_metadata.get("winner"), game_metadata.get("ending")
    outcome = "win" if winner == side else "loss" if winner in _SEATS else None
    if (outcome and record.get("outcome") == outcome and record.get("won") is (outcome == "win")
            and winner == audit.get("winner") and ending == audit.get("ending")
            and ending in ("played_out", "forfeit")):
        targets["outcome"] = outcome
        masks["outcome"] = True
        reasons["outcome"] = "matched_raw_outcome"
    else:
        reasons["outcome"] = "unverified_or_excluded_ending"
    return result
