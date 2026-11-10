"""Raw nickname-based action ownership and reveal timing checks, independent of parser.

Checks every train/validation turn; no test gameplay is opened. This is a focused
audit, not a complete Showdown interpreter. Species normalization alone is shared.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import re
from pathlib import Path

from vgc.data import canonical_species_id, load_moves, load_species


def toid(value):
    return re.sub(r"[^a-z0-9]", "", value.lower())


def identity(token):
    match = re.fullmatch(r"(p[12])[ab]?: (.*)", token)
    return (match[1], match[2]) if match else None


def species(details):
    sid = toid(details.split(",")[0])
    entry = load_species().get(sid, {})
    if entry.get("isMega"):
        parent = entry.get("battleOnly") or entry["baseSpecies"]
        sid = toid(parent[0] if isinstance(parent, list) else parent)
    elif entry.get("battleOnly"):
        parent = entry["battleOnly"]
        sid = toid(parent[0] if isinstance(parent, list) else parent)
    return canonical_species_id(sid)


def evidence(log):
    names, first_moves, turns, events = {}, {}, {}, collections.defaultdict(list)
    first_abilities = {}
    active = {"p1": [None, None], "p2": [None, None]}
    by_nickname = collections.defaultdict(set)
    items = {}
    cant_reveals = {}
    turn = 0
    # (turn, actor) pairs whose move that turn was not a free choice: Encore from
    # its start turn through the turn it ends, and an Instruct-ed repetition.
    # Charge/Outrage continuations carry `[from] lockedmove` on the move line.
    locked, encored = set(), set()

    def resolve(token):
        nick = identity(token)
        if nick is None:
            return None
        if len(token) > 2 and token[2] in "ab":
            actor = active[nick[0]]["ab".index(token[2])]
            return actor if actor and actor[:2] == nick else None
        candidates = by_nickname[nick]
        return next(iter(candidates)) if len(candidates) == 1 else None

    for idx, line in enumerate(log.splitlines()):
        p = line.split("|")
        if len(p) < 2:
            continue
        tag = p[1]
        if tag == "turn":
            turn = int(p[2])
            locked.update((turn, actor) for actor in encored)
            turns[turn] = {
                "line": idx, "active": {s: list(v) for s, v in active.items()},
                "items": dict(items),
            }
            cant_reveals.clear()
        if len(p) < 3:
            continue
        if tag in ("switch", "drag", "replace") and identity(p[2]) and len(p) > 3:
            nick = identity(p[2])
            sid = species(p[3])
            who = (*nick, sid)
            encored.discard(active[nick[0]]["ab".index(p[2][2])])
            names[who] = sid
            by_nickname[nick].add(who)
            active[who[0]]["ab".index(p[2][2])] = who
        else:
            who = resolve(p[2])
        if tag in ("switch", "drag", "replace", "detailschange", "-formechange") and who and len(p) > 3:
            abilities = {toid(a) for a in load_species().get(
                toid(p[3].split(",")[0]), {}).get("abilities", {}).values()}
            if len(abilities) == 1:
                first_abilities.setdefault((who, abilities.pop()), idx)
        if who:
            of = next((resolve(x[4:].strip()) for x in p[3:] if x.startswith("[of]")), None)
            from_ability = next((toid(x.split(":", 1)[1]) for x in p[3:]
                                 if x.startswith("[from] ability:")), None)
            if tag == "-ability" and len(p) > 3:
                value = toid(p[3])
                first_abilities.setdefault((who, value), idx)
                if from_ability:
                    if value == from_ability:
                        if of:
                            first_abilities.setdefault((of, value), idx)
                    else:
                        first_abilities.setdefault((who, from_ability), idx)
                        if of:
                            first_abilities.setdefault((of, value), idx)
            else:
                # An unprefixed ability names the subject's ability; [of] on a
                # cant line names the blocked attacker and must not get it.
                for part in p[3:]:
                    if part.startswith("ability:"):
                        first_abilities.setdefault((who, toid(part.split(":", 1)[1])), idx)
                if from_ability:
                    first_abilities.setdefault((of or who, from_ability), idx)
        else:
            # Weather/terrain lines have a field name as their subject and name
            # the setter only in [of].
            of = next((resolve(x[4:].strip()) for x in p[3:] if x.startswith("[of]")), None)
            source = next((toid(x.split(":", 1)[1]) for x in p[3:]
                           if x.startswith("[from] ability:")), None)
            if of and source:
                first_abilities.setdefault((of, source), idx)
        if tag == "swap" and who and len(p) > 3:
            old, new = "ab".index(p[2][2]), int(p[3])
            active[who[0]][old], active[who[0]][new] = (
                active[who[0]][new], active[who[0]][old])
        if tag == "faint" and who:
            active[who[0]]["ab".index(p[2][2])] = None
            encored.discard(who)
        if who and len(p) > 3 and tag in ("-start", "-end", "-singleturn"):
            effect = toid(p[3].removeprefix("move: "))
            if effect == "encore" and tag == "-start":
                encored.add(who)
                locked.add((turn, who))
            elif effect == "encore" and tag == "-end":
                encored.discard(who)
            elif effect == "instruct" and tag == "-singleturn":
                locked.add((turn, who))
        if tag in ("move", "cant") and who:
            actor = who
            index = 3 if tag == "move" else 4
            if tag == "cant" and len(p) > 4 and p[3].startswith("ability:"):
                owner = next((x[4:].strip() for x in p[5:] if x.startswith("[of]")), None)
                if owner:
                    actor = resolve(owner)
            move = toid(p[index]) if len(p) > index else ""
            called = [x for x in p[index + 1:] if x.startswith("[from]")]
            native = not called or all(x in ("[from] move: Sleep Talk", "[from] Sleep Talk",
                                             "[from] move: Instruct") for x in called)
            if actor and move:
                # A blocked move named by `cant` is the actor's own only if it has
                # no move line this turn: after a native one it is a Dancer copy or
                # a reflection, after a called one it is that call being blocked
                # (Magic Bounce reflection stopped by Armor Tail, also on the turn
                # the reflector switched in); before one, the later native move
                # withdraws it.
                if tag == "cant" and events[turn, actor]:
                    native = False
                if tag == "move" and native and not called:
                    pending = cant_reveals.pop(actor, None)
                    if pending and pending[0] != move and first_moves.get((actor, pending[0])) == pending[1]:
                        del first_moves[actor, pending[0]]
                if native and move != "struggle":
                    if tag == "cant" and (actor, move) not in first_moves:
                        cant_reveals[actor] = (move, idx)
                    first_moves.setdefault((actor, move), idx)
                if tag == "move":
                    events[turn, actor].append((move, idx, line, bool(called)))
        if who and tag in ("-item", "-enditem") and len(p) > 3:
            items[who] = None if tag == "-enditem" else toid(p[3])
        if who and tag == "-mega" and len(p) > 4:
            items[who] = toid(p[4])
        # For damage [of] names the item holder (e.g. Rocky Helmet).
        # For Shell Bell healing [of] names the damaged victim, not the holder.
        item = next((x[len("[from] item:"):].strip() for x in p[3:]
                     if x.startswith("[from] item:")), None)
        if who and item:
            owner = who
            if tag == "-damage":
                owner = next((resolve(x[4:].strip()) for x in p[3:] if x.startswith("[of]")), who)
            item_id = toid(item)
            # A consumed berry's heal is not evidence that it is held again.
            if owner and not any(s in item_id for s in ("berry", "herb", "seed", "gem")):
                items[owner] = toid(item)
    return names, first_moves, first_abilities, turns, events, locked


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prepared", type=Path, required=True)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    counts, violations = collections.Counter(), collections.Counter()
    examples = collections.defaultdict(list)
    affected = collections.defaultdict(set)
    summary = json.loads((args.prepared / "summary.json").read_text())
    manifest = {g["id"]: g for g in map(json.loads, Path(summary["inputs"]["selection"]).open())}
    current = None

    def bad(key, row, detail):
        violations[key] += 1
        affected[key].add(row["replay_id"])
        if len(examples[key]) < 8:
            examples[key].append({
                "id": row["replay_id"], "side": row["player"], "turn": row["turn"],
                "source_line": row["source_line"], "detail": detail,
            })

    for split in ("train", "validation"):
        path = args.prepared / f"{split}.jsonl.gz"
        with path.open("rb") as handle:
            assert hashlib.file_digest(handle, "sha256").hexdigest() == summary["files_sha256"][path.name]
        with gzip.open(path, "rt") as handle:
            for text in handle:
                row = json.loads(text)
                rid = row["replay_id"]
                if current != rid:
                    raw_bytes = (args.raw / f"{rid}.json").read_bytes()
                    assert hashlib.sha256(raw_bytes).hexdigest() == manifest[rid]["file_sha256"]
                    names, first_moves, first_abilities, turns, events, locked = evidence(
                        json.loads(raw_bytes)["log"])
                    current = rid
                    counts["games"] += 1
                if row["decision_kind"] != "turn":
                    continue
                turn = turns[row["turn"]]
                cutoff = turn["line"]
                st = row["input"]["state"]
                sv = row["supervision"]
                side = row["player"]
                counts["turn_records"] += 1
                for rel, seat in (("our", side), ("opp", "p2" if side == "p1" else "p1")):
                    for slot, mon in enumerate(st[rel]["active"]):
                        actor = turn["active"][seat][slot]
                        expected_species = names.get(actor)
                        actual_species = mon and mon["species"]
                        counts["active_identity"] += 1
                        if expected_species != actual_species:
                            bad("active_identity", row, [rel, slot, expected_species, actual_species])
                            continue
                        if not mon:
                            continue
                        if rel == "our" and (row["turn"], actor) not in locked:
                            observed = events[row["turn"], actor]
                            if (len(observed) == 1 and observed[0][0] != "struggle"
                                    and not observed[0][3] and observed[0][0] in load_moves()):
                                counts["recoverable_move_mask"] += 1
                                if not sv["masks"]["move"][slot]:
                                    bad("recoverable_move_mask_missing", row, [slot, actor, observed])
                        if actor in turn["items"]:
                            counts["known_current_item"] += 1
                            if mon["item"] != turn["items"][actor]:
                                bad("known_current_item", row, [rel, slot, actor, turn["items"][actor], mon["item"]])
                        if rel == "opp":
                            if mon["item"]:
                                counts["opponent_item_before_decision"] += 1
                                if not turn["items"].get(actor):
                                    bad("opponent_item_before_decision", row,
                                        [actor, mon["item"]])
                            if mon["ability"]:
                                counts["opponent_ability_owner_before_decision"] += 1
                                first = first_abilities.get((actor, mon["ability"]))
                                if first is None or first >= cutoff:
                                    bad("opponent_ability_owner_before_decision", row,
                                        [actor, mon["ability"], first, cutoff])
                            for move in mon["revealed_moves"]:
                                counts["opponent_move_before_decision"] += 1
                                first = first_moves.get((actor, move))
                                if first is None or first >= cutoff:
                                    bad("opponent_move_before_decision", row, [actor, move, first, cutoff])
                            for (owner, move), first in first_moves.items():
                                if owner == actor and first < cutoff:
                                    counts["opponent_move_retained"] += 1
                                    if move not in mon["revealed_moves"]:
                                        bad("opponent_move_missing", row, [actor, move, first, cutoff])
                        elif sv["masks"]["move"][slot]:
                            if (row["turn"], actor) in locked:
                                bad("move_mask_on_locked_actor", row, [slot, actor])
                            label = sv["targets"]["move"][slot]
                            observed = events[row["turn"], actor]
                            counts["move_label_actor"] += 1
                            if len(observed) != 1 or observed[0][3]:
                                bad("move_label_not_single_native_move", row, [slot, actor, label, observed])
                            if label not in {e[0] for e in observed}:
                                bad("move_label_actor", row, [slot, actor, label, observed])
                            if label == "struggle":
                                bad("forced_struggle_policy_label", row, [slot, actor, label])
                            if len(mon["revealed_moves"]) > 4:
                                bad("move_options_over_four", row, [slot, actor, mon["revealed_moves"], label])
                for task, mask in sv["masks"].items():
                    target = sv["targets"][task]
                    pairs = zip(mask, target) if isinstance(mask, list) else [(mask, target)]
                    for enabled, label in pairs:
                        counts["mask_contract"] += 1
                        if type(enabled) is not bool or enabled != (label is not None):
                            bad("mask_contract", row, task)
        print(f"{split}: {dict(counts)}; violations: {dict(violations)}", flush=True)
    report = {
        "status": "fail" if violations else "pass", "counts": dict(counts),
        "violations": dict(violations), "examples": dict(examples),
        "affected_game_counts": {k: len(v) for k, v in affected.items()},
        "affected_game_ids": {k: sorted(v) for k, v in affected.items()},
        "prepared": str(args.prepared.resolve()),
        "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "limits": ["Train/validation turn decisions only; no final-test gameplay.",
                   "Identity, observed move labels, move reveal timing, known item state, masks.",
                   "Does not certify legal requests, submitted targets, or every ability transition."],
    }
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
