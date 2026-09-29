"""Count simple, checkable facts in public-ladder games: losses vs wins.

No judgement and no model: every fact is read straight off the replay's protocol log
(what a spectator would see), so anyone can verify it. A fact only points at a problem
if it is clearly more common in losses than in wins, so both groups are counted and
compared with a two-sided Fisher exact test. With ~23 games per group only large gaps
can register; treat every result as a lead to look at, not a finding.

    .venv/bin/python tools/loss_patterns.py
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

from vgc.config import FORMAT_ID, RUNS_DIR

DEFAULT_OUTPUT = RUNS_DIR / "review" / "loss_patterns_mc.json"
_LOG_RE = re.compile(r'class="battle-log-data">(.*?)</script>', re.S)

# Plain-English descriptions of each yes/no fact, per game.
FACTS = {
    "our_lead_ko_turn1": "one of our leads was knocked out on turn 1",
    "their_lead_ko_turn1": "one of their leads was knocked out on turn 1",
    "our_ko_before_acting": "one of ours was knocked out before it got to act that turn",
    "their_ko_before_acting": "one of theirs was knocked out before it got to act that turn",
    "we_got_first_ko": "we scored the first knockout",
    "they_set_tailwind": "they set up Tailwind",
    "we_set_tailwind": "we set up Tailwind",
    "they_set_trick_room": "they set up Trick Room",
    "we_set_trick_room": "we set up Trick Room",
    "crit_against_us": "at least one critical hit landed on us",
    "crit_for_us": "at least one of our hits was critical",
    "we_missed": "at least one of our attacks missed",
    "they_missed": "at least one of their attacks missed",
    "we_fully_paralysed": "one of ours was fully paralysed at least once",
}


def _species(details: str) -> str:
    return details.split(",")[0].strip()


def game_facts(log: str, our_side: str) -> dict:
    """Yes/no facts plus counts for one game, from our side's point of view."""

    facts = dict.fromkeys(FACTS, False)
    counts = Counter()
    active_at_start: set[str] = set()
    acted: set[str] = set()
    species_by_slot: dict[str, str] = {}
    leads: dict[str, list[str]] = {"p1": [], "p2": []}
    brought: dict[str, list[str]] = {"p1": [], "p2": []}
    turn = 0
    last_mover_side: str | None = None
    first_faint_side: str | None = None  # side whose Pokemon fainted first
    switched_this_turn: set[str] = set()

    def ours(ident: str) -> bool:
        return ident.startswith(our_side)

    for raw in log.splitlines():
        parts = raw.split("|")
        if len(parts) < 2:
            continue
        tag = parts[1]
        if tag in ("switch", "drag"):
            slot = parts[2].split(":")[0][:3]
            species = _species(parts[3])
            species_by_slot[slot] = species
            side = slot[:2]
            if species not in brought[side]:
                brought[side].append(species)
            if turn == 0:
                leads[side].append(species)
            else:
                switched_this_turn.add(slot)
        elif tag == "turn":
            turn = int(parts[2])
            active_at_start = set(species_by_slot)
            acted = set()
            switched_this_turn = set()
        elif tag in ("move", "cant"):
            slot = parts[2].split(":")[0][:3]
            acted.add(slot)
            last_mover_side = slot[:2]
            if tag == "cant" and parts[3] == "par" and ours(slot):
                facts["we_fully_paralysed"] = True
        elif tag == "faint":
            slot = parts[2].split(":")[0][:3]
            side_key = "our" if ours(slot) else "their"
            counts[f"{side_key}_faints"] += 1
            if first_faint_side is None:
                first_faint_side = slot[:2]
            if turn == 1:
                facts[f"{side_key}_lead_ko_turn1"] = True
            if (
                turn >= 1
                and slot in active_at_start
                and slot not in acted
                and slot not in switched_this_turn
            ):
                facts[f"{side_key}_ko_before_acting"] = True
                counts[f"{side_key}_ko_before_acting"] += 1
            species_by_slot.pop(slot, None)
        elif tag == "-crit":
            facts["crit_against_us" if ours(parts[2]) else "crit_for_us"] = True
        elif tag == "-miss":
            facts["we_missed" if ours(parts[2]) else "they_missed"] = True
        elif tag == "-sidestart" and "Tailwind" in parts[3]:
            facts["we_set_tailwind" if ours(parts[2]) else "they_set_tailwind"] = True
        elif tag == "-fieldstart" and "Trick Room" in parts[2] and last_mover_side:
            key = "we_set_trick_room" if last_mover_side == our_side else "they_set_trick_room"
            facts[key] = True

    facts["we_got_first_ko"] = first_faint_side is not None and first_faint_side != our_side
    their = "p2" if our_side == "p1" else "p1"
    return {
        "facts": facts,
        "counts": dict(counts),
        "turns": turn,
        "our_leads": leads[our_side],
        "their_leads": leads[their],
        "our_brought": brought[our_side],
        "their_brought": brought[their],
    }


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p-value for [[a, b], [c, d]]."""

    n = a + b + c + d
    row1, col1 = a + b, a + c

    def prob(x: int) -> float:
        return math.comb(col1, x) * math.comb(n - col1, row1 - x) / math.comb(n, row1)

    observed = prob(a)
    lo, hi = max(0, row1 + col1 - n), min(row1, col1)
    return min(1.0, sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= observed * (1 + 1e-9)))


def load_games(ladder_path: Path, format_id: str) -> list[dict]:
    games = []
    for line in ladder_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("format") != format_id or "local" in str(row.get("session_id")):
            continue
        replay = Path(row.get("replay_path") or "")
        if not replay.exists():
            continue
        match = _LOG_RE.search(replay.read_text())
        if not match:
            continue
        log = match.group(1)
        # Replay files are saved as "<our username> - <battle tag>.html".
        username = replay.name.split(" - ")[0].lower()
        our_side = next(
            line.split("|")[2]
            for line in log.splitlines()
            if line.startswith("|player|") and line.split("|")[3].lower() == username
        )
        games.append(
            {
                "id": row["battle_tag"].rsplit("-", 1)[-1],
                "won": bool(row.get("won")),
                **game_facts(log, our_side),
            }
        )
    return games


def compare(games: list[dict]) -> list[dict]:
    losses = [g for g in games if not g["won"]]
    wins = [g for g in games if g["won"]]
    rows = []
    for key, text in FACTS.items():
        in_losses = sum(g["facts"][key] for g in losses)
        in_wins = sum(g["facts"][key] for g in wins)
        rows.append(
            {
                "fact": key,
                "description": text,
                "losses": in_losses,
                "loss_games": len(losses),
                "wins": in_wins,
                "win_games": len(wins),
                "p_value": round(
                    fisher_exact(in_losses, len(losses) - in_losses, in_wins, len(wins) - in_wins),
                    4,
                ),
            }
        )
    rows.sort(key=lambda r: r["p_value"])
    return rows


def knockout_timing(games: list[dict]) -> list[dict]:
    """Share of knockouts that landed before the Pokemon could act, per knockout.

    The per-game "knocked out before acting" fact is inflated in losses simply because
    all four of our Pokemon faint in every loss. Dividing by the number of knockouts
    removes that. Knockouts within one game are not independent, so the p-value here
    is optimistic.
    """

    rows = []
    for side in ("our", "their"):
        by_result = {}
        for won in (False, True):
            group = [g for g in games if g["won"] == won]
            before = sum(g["counts"].get(f"{side}_ko_before_acting", 0) for g in group)
            total = sum(g["counts"].get(f"{side}_faints", 0) for g in group)
            by_result[won] = (before, total)
        (lb, lt), (wb, wt) = by_result[False], by_result[True]
        rows.append(
            {
                "side": side,
                "losses": [lb, lt],
                "wins": [wb, wt],
                "p_value": round(fisher_exact(lb, lt - lb, wb, wt - wb), 4),
            }
        )
    return rows


def species_table(games: list[dict], field: str) -> list[dict]:
    seen = Counter()
    lost = Counter()
    for game in games:
        for species in set(game[field]):
            seen[species] += 1
            lost[species] += not game["won"]
    return [
        {"species": s, "games": n, "losses": lost[s], "loss_rate": round(lost[s] / n, 2)}
        for s, n in seen.most_common()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ladder", type=Path, default=RUNS_DIR / "ladder.jsonl")
    parser.add_argument("--format-id", default=FORMAT_ID)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    games = load_games(args.ladder, args.format_id)
    losses = sum(not g["won"] for g in games)
    rows = compare(games)
    report = {
        "format_id": args.format_id,
        "games": len(games),
        "losses": losses,
        "wins": len(games) - losses,
        "facts": rows,
        "our_lead_pairs": Counter(
            " + ".join(sorted(g["our_leads"])) + (" (L)" if not g["won"] else " (W)") for g in games
        ).most_common(),
        "their_brought": species_table(games, "their_brought"),
        "our_brought": species_table(games, "our_brought"),
        "knockout_timing": knockout_timing(games),
        "per_game": games,
        "caveat": (
            "Small samples: with ~23 games per group only large gaps show up, and with "
            f"{len(rows)} facts checked, one p < 0.05 is expected by chance. Facts are "
            "correlations with losing, not proven causes."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1))

    print(f"{len(games)} games: {losses} losses, {len(games) - losses} wins\n")
    print(f"{'fact':<62} {'losses':>8} {'wins':>8} {'p':>7}")
    for row in rows:
        print(
            f"{row['description']:<62} {row['losses']:>3}/{row['loss_games']:<4} "
            f"{row['wins']:>3}/{row['win_games']:<4} {row['p_value']:>7.3f}"
        )
    print("\nshare of knockouts that landed before the Pokemon acted (fairer than the")
    print("per-game row above, which is inflated because every loss has 4 of ours faint):")
    for row in report["knockout_timing"]:
        (lb, lt), (wb, wt) = row["losses"], row["wins"]
        print(
            f"  {row['side']:>5}: losses {lb}/{lt} = {lb / max(lt, 1):.0%}, "
            f"wins {wb}/{wt} = {wb / max(wt, 1):.0%}, p = {row['p_value']:.3f} (optimistic)"
        )
    print(f"\n{report['caveat']}")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
