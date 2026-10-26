"""Check corrected opening states against raw logs and compare retained replay labels.

The opening checker is deliberately independent of vgc.replay_parse. It checks each
active slot's net stat stages and the field immediately before the first turn marker.
It is not a complete replay mechanics audit. Output includes content fingerprints.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import itertools
import json
import re
from pathlib import Path


def open_records(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        for line in handle:
            yield json.loads(line)


def fingerprint(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def opening(log: str) -> tuple[dict, dict]:
    weather_ids = {"SunnyDay": "sun", "DesolateLand": "sun", "RainDance": "rain",
                   "PrimordialSea": "rain", "Sandstorm": "sand", "Hail": "snow",
                   "Snowscape": "snow", "Snow": "snow"}
    terrain_ids = {f"move: {name} Terrain": name.lower()
                   for name in ("Electric", "Grassy", "Psychic", "Misty")}
    field = {"weather": None, "terrain": None, "trick_room": False, "turn": 1}
    stages: dict[str, dict] = {}
    for line in log.splitlines():
        if line.startswith("|turn|"):
            break
        parts = line.split("|")
        if len(parts) < 2:
            continue
        tag = parts[1]
        if tag == "-clearallboost":
            for pos in stages:
                stages[pos] = {}
        if len(parts) < 3:
            continue
        target = parts[2]
        pos = target[:3]
        if tag in ("switch", "drag"):
            stages[pos] = {}
        elif tag == "-weather" and "[upkeep]" not in parts[3:]:
            field["weather"] = None if target == "none" else weather_ids.get(target, target.lower())
        elif tag in ("-fieldstart", "-fieldend"):
            enabled = tag == "-fieldstart"
            if target == "move: Trick Room":
                field["trick_room"] = enabled
            elif target in terrain_ids:
                terrain = terrain_ids[target]
                if enabled or field["terrain"] == terrain:
                    field["terrain"] = terrain if enabled else None
        elif pos in stages:
            values = stages[pos]
            if tag == "-clearboost":
                stages[pos] = {}
            elif tag == "-clearnegativeboost":
                stages[pos] = {k: v for k, v in values.items() if v > 0}
            elif tag in ("-boost", "-unboost", "-setboost") and len(parts) > 4:
                try:
                    amount = int(parts[4])
                except ValueError:
                    continue
                value = amount if tag == "-setboost" else (
                    values.get(parts[3], 0) + amount * (-1 if tag == "-unboost" else 1))
                values[parts[3]] = max(-6, min(6, value))
    return field, stages


def raw_id(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def raw_identities(log: str, species: dict) -> tuple[dict, dict]:
    previews = {side: [] for side in ("p1", "p2")}
    appeared = {side: [] for side in previews}
    leads = {side: [] for side in previews}
    aliases = {side: {} for side in previews}
    before_turn = True
    for line in log.splitlines():
        if line.startswith("|turn|"):
            before_turn = False
        parts = line.split("|")
        if len(parts) < 4:
            continue
        if parts[1] == "poke" and parts[2] in previews:
            previews[parts[2]].append(raw_id(parts[3].split(",")[0]))
        if parts[1] not in ("switch", "drag", "replace"):
            continue
        side = parts[2][:2]
        if side not in previews:
            continue
        identity = raw_id(parts[3].split(",")[0])
        entry = species.get(identity, {})
        parent = entry.get("battleOnly")
        if isinstance(parent, list):
            parent = parent[0] if parent else None
        if entry.get("isMega"):
            old = raw_id(entry.get("baseSpecies", identity))
            identity = raw_id(parent or entry.get("baseSpecies", identity))
            # Male Meowstic is legitimate. Only allow a correction to a form that
            # actually belongs to this side's preview and whose generic base does not.
            if identity in previews[side] and old not in previews[side]:
                aliases[side][old] = identity
        elif parent:
            identity = raw_id(parent)
        if identity not in appeared[side]:
            appeared[side].append(identity)
        if before_turn and parts[1] in ("switch", "drag") and len(leads[side]) < 2:
            leads[side].append(identity)
    expected = {side: {"derived": True, "picked": appeared[side][:4],
                       "lead_order": leads[side]} for side in previews}
    return aliases, expected


def normalized_action(record: dict, aliases: dict) -> dict:
    action = json.loads(json.dumps(record["action"]))
    for value in [action, *[v for v in action.values() if isinstance(v, dict)]]:
        if "switch_species" in value:
            value["switch_species"] = aliases.get(value["switch_species"], value["switch_species"])
    if record["decision_kind"] == "teampreview":
        for key in ("picked", "lead_order"):
            mapped = [aliases.get(v, v) for v in action[key]]
            action[key] = list(dict.fromkeys(mapped))
    return action


def audit(data: Path, previous: Path, raw: Path) -> dict:
    counts: collections.Counter = collections.Counter()
    failures: collections.Counter = collections.Counter()
    examples = []
    seen = set()
    raw_hash = hashlib.sha256()
    species = json.loads((Path(__file__).resolve().parents[1]
                          / "data/champions/species.json").read_text())
    current_id, expected_field, expected_stages = None, None, None
    for new, old in itertools.zip_longest(open_records(data), open_records(previous)):
        if new is None or old is None:
            failures["record_count_mismatch"] += 1
            break
        counts["records"] += 1
        rid = new["replay_id"]
        if rid != current_id:
            content = (raw / f"{rid}.json").read_bytes()
            raw_hash.update(rid.encode() + b"\0" + hashlib.sha256(content).digest())
            log = json.loads(content)["log"]
            expected_field, expected_stages = opening(log)
            aliases, expected_preview = raw_identities(log, species)
            current_id = rid
        if new["schema"] != 7:
            failures["unexpected_schema"] += 1
        for key in ("replay_id", "player", "turn", "decision_kind", "outcome", "won",
                    "rating", "format_id", "action_status", "sets"):
            if new.get(key) != old.get(key):
                failures[f"changed_{key}"] += 1
        if new["action"] != old["action"]:
            counts["identity_corrected_actions"] += 1
            side_aliases = aliases[new["player"]]
            if new["decision_kind"] == "teampreview":
                valid_change = new["action"] == expected_preview[new["player"]]
                counts["raw_verified_preview_action_corrections"] += valid_change
            else:
                valid_change = normalized_action(new, side_aliases) == normalized_action(
                    old, side_aliases)
            if not valid_change:
                failures["unexpected_action_change"] += 1
                if len(examples) < 10:
                    examples.append({"replay_id": rid, "old_action": old["action"],
                                     "new_action": new["action"]})
        if new["state"] != old["state"]:
            counts["changed_states"] += 1
        seen.add(rid)
        if new["decision_kind"] != "turn" or new["turn"] != 1:
            continue
        if new["player"] == "p1":
            counts["opening_games"] += 1
            counts["opening_field_games"] += bool(
                expected_field["weather"] or expected_field["terrain"])
            counts["opening_boost_games"] += any(
                v for values in expected_stages.values() for v in values.values())
        counts["opening_perspectives"] += 1
        before = failures.total()
        if new["state"]["field"] != expected_field:
            failures["opening_field_mismatch"] += 1
        for side in ("p1", "p2"):
            relative = "our" if new["player"] == side else "opp"
            for slot, mon in enumerate(new["state"][relative]["active"]):
                pos = f"{side}{'ab'[slot]}"
                actual = {k: v for k, v in (mon or {}).get("boosts", {}).items() if v}
                expected = {k: v for k, v in expected_stages.get(pos, {}).items() if v}
                if actual != expected:
                    failures["opening_boost_mismatch"] += 1
        if failures.total() > before and len(examples) < 10:
            examples.append({"replay_id": rid, "player": new["player"]})
    return {
        "verdict": "PASS" if not failures else "FAIL",
        "counts": dict(counts), "replays": len(seen), "failures": dict(failures),
        "examples": examples,
        "data": {"path": str(data), "sha256": fingerprint(data)},
        "previous": {"path": str(previous), "sha256": fingerprint(previous)},
        "raw_manifest_sha256": raw_hash.hexdigest(),
        "limits": "Checks opening field/stages and label preservation, not all mechanics. "
                  "Public reveal timing is additionally covered by parser regression tests.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--raw-replays", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.data, args.previous, args.raw_replays)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
