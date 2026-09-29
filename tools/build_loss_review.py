"""Build a human loss-review sheet from saved public-ladder losses.

Step one of the Jev loss-review trial (see CLAUDE.md "TypeSafe/Jev usage"): a person
labels each loss by hand BEFORE any model sees it, so the labels are independent
ground truth. This script only turns each saved replay log into a readable,
turn-by-turn summary of what was publicly observed.

- Observed facts come from the replay's protocol log only.
- The bot's own rule-based loss guess (`vgc.postmortem.classify_loss`, stored in
  `runs/ladder.jsonl`) is kept in a separate field so reviewers can hide it and avoid
  anchoring on it.
- Opponent usernames and nicknames are dropped; Pokemon are named by species.
- A fixed, seeded subset is marked `holdout`. Those games are labelled like the rest,
  but are not looked at while designing Jev's questions, so they can test it fairly.

    .venv/bin/python tools/build_loss_review.py
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from vgc.config import FORMAT_ID, RUNS_DIR

DEFAULT_OUTPUT = RUNS_DIR / "review" / "loss_review_mc.json"
HOLDOUT_SEED = 20260929
HOLDOUT_FRACTION = 0.35

_LOG_RE = re.compile(r'class="battle-log-data">(.*?)</script>', re.S)


def _species(details: str) -> str:
    return details.split(",")[0].strip()


def _hp_pct(hp: str) -> str:
    hp = hp.split()[0]
    if hp.endswith("fnt") or hp == "0":
        return "0%"
    if "/" in hp:
        # Showdown can append a colour suffix to the max ("50/100g" in some logs).
        cur, top = (re.sub(r"\D", "", part) for part in hp.split("/"))
        return f"{round(100 * int(cur) / int(top))}%"
    return f"{hp}%"


def _load_ladder_losses(ladder_path: Path, format_id: str) -> list[dict]:
    rows = [json.loads(line) for line in ladder_path.read_text().splitlines() if line.strip()]
    return [
        row
        for row in rows
        if row.get("format") == format_id
        and row.get("lost")
        and "local" not in str(row.get("session_id"))
        and row.get("replay_path")
        and Path(row["replay_path"]).exists()
    ]


def summarize_log(log: str, our_side: str) -> dict:
    """Turn a protocol log into plain-English, public-only turn lines."""

    names: dict[str, str] = {}  # "p2a: Nick" -> species
    preview: dict[str, list[str]] = {"p1": [], "p2": []}
    brought: dict[str, list[str]] = {"p1": [], "p2": []}
    leads: dict[str, list[str]] = {"p1": [], "p2": []}
    turns: list[dict] = []
    current: dict = {"turn": 0, "events": []}
    started = False
    flags = {
        "crit_against_us": 0,
        "crit_for_us": 0,
        "our_misses": 0,
        "their_misses": 0,
        "our_full_para": 0,
        "their_full_para": 0,
    }

    def side_of(ident: str) -> str:
        return ident[:2]

    def who(ident: str) -> str:
        ident = ident.strip()
        slot_key = ident.split(":")[0][:3]
        species = names.get(ident) or names.get(slot_key) or ident.split(":")[-1].strip()
        owner = "Our" if side_of(ident) == our_side else "Their"
        return f"{owner} {species}"

    def add(text: str) -> None:
        current["events"].append(text)

    for raw in log.splitlines():
        parts = raw.split("|")
        if len(parts) < 2:
            continue
        tag = parts[1]
        if tag == "poke":
            preview[parts[2]].append(_species(parts[3]))
        elif tag == "start":
            started = True
        elif tag in ("switch", "drag"):
            ident, details = parts[2], parts[3]
            species = _species(details)
            names[ident] = species
            names[ident.split(":")[0][:3]] = species
            side = side_of(ident)
            if species not in brought[side]:
                brought[side].append(species)
            if current["turn"] == 0 and started:
                leads[side].append(species)
            else:
                add(f"{who(ident)} switched in ({_hp_pct(parts[4])})")
        elif tag == "detailschange":
            ident = parts[2]
            species = _species(parts[3])
            names[ident] = species
            names[ident.split(":")[0][:3]] = species
        elif tag == "turn":
            if current["events"] or current["turn"]:
                turns.append(current)
            current = {"turn": int(parts[2]), "events": []}
        elif tag == "move":
            has_target = len(parts) > 4 and parts[4].strip() and parts[4] != parts[2]
            target = f" on {who(parts[4])}" if has_target else ""
            if "[spread]" in raw:
                target = " (spread)"
            missed = " - missed" if "[miss]" in raw else ""
            add(f"{who(parts[2])} used {parts[3]}{target}{missed}")
        elif tag == "-damage":
            source = ""
            if len(parts) > 4 and parts[4].startswith("[from]"):
                source = f" from {parts[4][7:].strip()}"
            add(f"  {who(parts[2])} fell to {_hp_pct(parts[3])}{source}")
        elif tag == "-heal":
            add(f"  {who(parts[2])} healed to {_hp_pct(parts[3])}")
        elif tag == "faint":
            add(f"  ** {who(parts[2])} fainted")
        elif tag == "-crit":
            add(f"  CRITICAL HIT on {who(parts[2])}")
            flags["crit_against_us" if side_of(parts[2]) == our_side else "crit_for_us"] += 1
        elif tag == "-miss":
            add(f"  {who(parts[2])} missed")
            flags["our_misses" if side_of(parts[2]) == our_side else "their_misses"] += 1
        elif tag == "cant":
            add(f"{who(parts[2])} could not move ({parts[3]})")
            if parts[3] == "par":
                flags["our_full_para" if side_of(parts[2]) == our_side else "their_full_para"] += 1
        elif tag == "-supereffective":
            add(f"  super effective on {who(parts[2])}")
        elif tag == "-status":
            add(f"  {who(parts[2])} got status {parts[3]}")
        elif tag == "-mega":
            add(f"{who(parts[2])} Mega Evolved")
        elif tag == "-weather" and "[upkeep]" not in raw:
            add(f"Weather: {parts[2]}")
        elif tag in ("-fieldstart", "-fieldend"):
            add(
                f"Field {'started' if tag == '-fieldstart' else 'ended'}: "
                f"{parts[2].replace('move: ', '')}"
            )
        elif tag in ("-sidestart", "-sideend"):
            owner = "Our" if parts[2].startswith(our_side) else "Their"
            verb = "started" if tag == "-sidestart" else "ended"
            add(f"{owner} side {verb}: {parts[3].replace('move: ', '')}")
        elif tag in ("-boost", "-unboost"):
            sign = "+" if tag == "-boost" else "-"
            add(f"  {who(parts[2])} {parts[3]} {sign}{parts[4]}")
        elif tag == "win":
            add("Game over: we lost")
    if current["events"]:
        turns.append(current)

    their = "p2" if our_side == "p1" else "p1"
    return {
        "our_preview": preview[our_side],
        "their_preview": preview[their],
        "our_brought": brought[our_side],
        "their_revealed": brought[their],
        "our_leads": leads[our_side],
        "their_leads": leads[their],
        "turns": turns,
        "flags": flags,
    }


def build(ladder_path: Path, format_id: str) -> list[dict]:
    games = []
    for row in _load_ladder_losses(ladder_path, format_id):
        html = Path(row["replay_path"]).read_text()
        match = _LOG_RE.search(html)
        if not match:
            continue
        log = match.group(1)
        # Replay files are saved as "<our username> - <battle tag>.html".
        username = Path(row["replay_path"]).name.split(" - ")[0].lower()
        player_line = next(
            line
            for line in log.splitlines()
            if line.startswith("|player|") and line.split("|")[3].lower() == username
        )
        our_side = player_line.split("|")[2]
        summary = summarize_log(log, our_side)
        games.append(
            {
                "id": row["battle_tag"].rsplit("-", 1)[-1],
                "battle_tag": row["battle_tag"],
                "date": row["timestamp"][:10],
                "our_rating": row.get("rating"),
                "opponent_rating": row.get("opponent_rating"),
                "turns_played": row.get("turns"),
                **summary,
                "bot_guess": row.get("loss_classification"),
            }
        )
    rng = random.Random(HOLDOUT_SEED)
    holdout = set(rng.sample(range(len(games)), round(HOLDOUT_FRACTION * len(games))))
    for index, game in enumerate(games):
        game["holdout"] = index in holdout
    return games


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ladder", type=Path, default=RUNS_DIR / "ladder.jsonl")
    parser.add_argument("--format-id", default=FORMAT_ID)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    games = build(args.ladder, args.format_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(games, indent=1))
    print(
        f"wrote {len(games)} losses ({sum(g['holdout'] for g in games)} holdout) to {args.output}"
    )


if __name__ == "__main__":
    main()
