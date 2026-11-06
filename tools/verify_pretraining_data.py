#!/usr/bin/env python
"""Independent re-verification of a prepared replay-pretraining dataset.

Re-derives facts from the RAW replay logs (not from pipeline outputs) and checks the
selection manifest, prepared shards and sampling plans against them:

  A. selection: raw file hashes, split time boundaries, duplicates, pre-game ratings
     and winners re-derived from raw logs, reserved-team rule.
  B. prepared shards: split membership, no test/validation ids in training sampling,
     rating floors / open-sheet rule per task, outcome labels vs raw winner, forbidden
     keys in model inputs, preview label sanity, forced-switch destination leak,
     own-mega-this-turn leak, move option-set sizes.
  C. fog: for sampled games, every opponent move/item/ability/mega in a model input
     must be publicly revealed in the raw log BEFORE that decision's log line.

    .venv/bin/python tools/verify_pretraining_data.py --out runs/verify_20261106
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import html
import json
import random
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from vgc.data import canonical_species_id, load_species  # noqa: E402
from vgc.replay_parse import _resolve_species, _species_from_details  # noqa: E402

PRE_RATING = re.compile(r"^\|raw\|(.+?)'s rating: (\d+) &rarr;", re.M)
FORBIDDEN_INPUT_KEYS = {"rating", "pre_rating", "outcome", "won", "action", "winner",
                        "name", "player_name", "account", "own_info", "own_fully_confirmed"}
POLICY_TASKS = ("move", "switch_destination", "preview_brought", "preview_leads")


def toid(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def species_key(details: str) -> str:
    return canonical_species_id(_resolve_species(toid(_species_from_details(details)))[0] or "")


def single_ability(forme: str) -> str | None:
    abilities = {toid(a) for a in ((load_species().get(forme) or {}).get("abilities") or {}).values()}
    return abilities.pop() if len(abilities) == 1 else None


class Tally:
    def __init__(self):
        self.counts = collections.Counter()
        self.examples = collections.defaultdict(list)

    def bad(self, key: str, example) -> None:
        self.counts[key] += 1
        if len(self.examples[key]) < 5:
            self.examples[key].append(example)

    def ok(self, key: str) -> None:
        self.counts[key + "__checked"] += 1


def raw_facts(log: str):
    names, ratings = {}, collections.defaultdict(set)
    for line in log.splitlines():
        p = line.split("|")
        if len(p) > 3 and p[1] == "player" and p[2] in ("p1", "p2") and toid(p[3]):
            names.setdefault(p[2], toid(p[3]))
    for name, before in PRE_RATING.findall(log):
        for side, n in names.items():
            if n == toid(html.unescape(name)):
                ratings[side].add(int(before))
    wins = re.findall(r"^\|win\|([^\n]*)", log, re.M)
    winner = None
    if wins:
        sides = [s for s, n in names.items() if n == toid(wins[-1])]
        winner = sides[0] if len(sides) == 1 else None
    return {s: (next(iter(v)) if len(v) == 1 else None) for s, v in ratings.items()}, winner


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", type=Path, default=REPO / "data/bc/pretraining_regmc_s9_20261106")
    ap.add_argument("--raw", type=Path, default=REPO / "data/replays/gen9championsvgc2026regmc")
    ap.add_argument("--fog-games", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=20261106)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t = Tally()
    summary = json.loads((args.prepared / "summary.json").read_text())
    manifest_path = Path(summary["inputs"]["selection"])
    dataset_path = Path(summary["inputs"]["dataset"])
    sel_summary = json.loads((manifest_path.parent / "summary.json").read_text())

    # ---------------- A. selection ----------------
    games = {}
    for line in manifest_path.open():
        g = json.loads(line)
        if g["id"] in games:
            t.bad("A_duplicate_id", g["id"])
        games[g["id"]] = g
    kept = {rid: g for rid, g in games.items() if g["split"] in ("train", "validation", "test")}
    times = {s: [g["time"] for g in kept.values() if g["split"] == s] for s in ("train", "validation", "test")}
    b = sel_summary.get("boundaries") or {}
    print("boundaries in selection summary:", b)
    print("split time ranges:", {s: (min(v), max(v), len(v)) for s, v in times.items()})
    if not (max(times["train"]) < min(times["validation"]) and max(times["validation"]) < min(times["test"])):
        t.bad("A_split_times_overlap", {s: (min(v), max(v)) for s, v in times.items()})
    for s, fixed in (("validation", 1792406685), ("test", 1792870992)):
        if min(times[s]) < fixed:
            t.bad("A_split_before_fixed_boundary", (s, min(times[s])))
    if max(times["test"]) > 1793572357:
        t.bad("A_test_after_window_end", max(times["test"]))
    hashes = collections.defaultdict(set)
    for g in kept.values():
        hashes[g["log_sha256"]].add(g["split"])
    for h, s in hashes.items():
        if len(s) > 1:
            t.bad("A_same_log_in_two_splits", (h, sorted(s)))
    # Only the hashed 5% "team challenge" test rosters are held out of train/val
    # (same rule as inventory_replay_pretraining.reserve_teams).
    test_rosters = {tuple(r) for g in kept.values() if g["split"] == "test" for r in g["rosters"].values()}
    reserved = set(sorted(test_rosters, key=lambda r: hashlib.sha256(
        ("20261102:" + ",".join(r)).encode()).hexdigest())[:max(1, round(len(test_rosters) * .05))])
    for g in kept.values():
        if g["split"] in ("train", "validation") and any(tuple(r) in reserved for r in g["rosters"].values()):
            t.bad("A_train_val_roster_in_team_challenge", g["id"])
        if g.get("showteam"):
            t.bad("A_open_sheet_game_in_split", g["id"])
    # raw hash + independent rating/winner derivation for every kept game
    raw_ratings = {}
    for i, (rid, g) in enumerate(kept.items()):
        data = (args.raw / f"{rid}.json").read_bytes()
        t.ok("A_raw_hash")
        if hashlib.sha256(data).hexdigest() != g["file_sha256"]:
            t.bad("A_raw_hash", rid)
        raw = json.loads(data)
        if raw.get("formatid") != "gen9championsvgc2026regmc":
            t.bad("A_format", rid)
        ratings, winner = raw_facts(raw["log"])
        raw_ratings[rid] = ratings
        for side in ("p1", "p2"):
            t.ok("A_pre_rating")
            if ratings.get(side) != g["ratings"].get(side):
                t.bad("A_pre_rating", (rid, side, ratings.get(side), g["ratings"].get(side)))
        t.ok("A_winner")
        if winner != g.get("winner"):
            t.bad("A_winner", (rid, winner, g.get("winner")))
        if raw.get("rating") is not None and ratings:
            # listing rating is post-game; it should not systematically equal pre-game
            t.counts["A_listing_equals_some_pre_rating"] += raw["rating"] in ratings.values()
            t.counts["A_listing_rating_seen"] += 1
        if i % 20000 == 0:
            print(f"  A raw {i}/{len(kept)}", flush=True)

    # ---------------- s9 source records (log_line, action) ----------------
    src = {}
    with gzip.open(dataset_path, "rt") as f:
        for n, line in enumerate(f, 1):
            r = json.loads(line)
            src[n] = (r["replay_id"], r["player"], r.get("log_line"), r.get("action"), r["decision_kind"])
    print("source records:", len(src), flush=True)

    # ---------------- B. prepared shards ----------------
    rng = random.Random(args.seed)
    fog_ids = set(rng.sample(sorted(rid for rid, g in kept.items() if g["split"] != "test"), args.fog_games))
    fog_records = collections.defaultdict(list)
    option_sizes = collections.Counter()
    outcome_balance = collections.Counter()

    def walk_keys(obj, path=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in FORBIDDEN_INPUT_KEYS:
                    yield path + "." + k
                yield from walk_keys(v, path + "." + k)
        elif isinstance(obj, list):
            for v in obj:
                yield from walk_keys(v, path)

    for split in ("train", "validation", "test"):
        with gzip.open(args.prepared / f"{split}.jsonl.gz", "rt") as f:
            for r in f:
                r = json.loads(r)
                rid, side = r["replay_id"], r["player"]
                g = games.get(rid)
                t.ok("B_split_membership")
                if g is None or g["split"] != split or r["split"] != split:
                    t.bad("B_split_membership", (rid, split, g and g["split"]))
                    continue
                s_rid, s_side, log_line, action, kind = src[r["source_line"]]
                if (s_rid, s_side, kind) != (rid, side, r["decision_kind"]):
                    t.bad("B_source_line_join", (rid, r["source_line"]))
                    continue
                for bad_key in walk_keys(r["input"]):
                    t.bad("B_forbidden_input_key", (rid, bad_key))
                if r["pre_rating"] != raw_ratings[rid].get(side):
                    t.bad("B_pre_rating_vs_raw", (rid, side, r["pre_rating"], raw_ratings[rid].get(side)))
                sv = r["supervision"]
                if sv["masks"]["outcome"]:
                    t.ok("B_outcome_vs_raw_winner")
                    expect = "win" if g["winner"] == side else "loss"
                    if sv["targets"]["outcome"] != expect:
                        t.bad("B_outcome_vs_raw_winner", (rid, side, sv["targets"]["outcome"], g["winner"]))
                    outcome_balance[split, sv["targets"]["outcome"]] += 1
                st = r["input"].get("state") or {}
                if r["decision_kind"] == "teampreview" and sv["masks"]["preview_brought"]:
                    t.ok("B_preview_labels")
                    brought, leads = sv["targets"]["preview_brought"], sv["targets"]["preview_leads"]
                    if len(set(brought)) != 4 or (leads and (len(set(leads)) != 2 or not set(leads) <= set(brought))):
                        t.bad("B_preview_labels", (rid, side, brought, leads))
                if r["decision_kind"] in ("turn", "forced_switch"):
                    our = st["our"]
                    act_species = [m["species"] if m else None for m in our["active"]]
                    bench_species = [m["species_id"] for m in our["bench"]]
                    for slot in (0, 1):
                        dest = sv["targets"]["switch_destination"][slot]
                        if sv["masks"]["switch_destination"][slot]:
                            t.ok("B_destination_leak")
                            if dest in act_species or dest not in bench_species:
                                t.bad("B_destination_leak", (rid, side, r["turn"], r["decision_kind"], slot, dest, act_species, bench_species))
                        mv = sv["targets"]["move"][slot]
                        if sv["masks"]["move"][slot] and our["active"][slot]:
                            moves = our["active"][slot]["revealed_moves"]
                            option_sizes[split, len(moves), mv in moves] += 1
                        if (r["decision_kind"] == "turn" and isinstance(action, dict)
                                and (action.get(f"slot{slot}") or {}).get("mega")
                                and our["active"][slot] and our["active"][slot]["mega"]):
                            t.bad("B_own_mega_this_turn_already_in_state", (rid, side, r["turn"], slot))
                    if rid in fog_ids:
                        fog_records[rid].append((side, log_line, r["turn"], r["decision_kind"], st["opp"]))
        print(f"  B {split} done", flush=True)

    # sampling plans: only train ids; per-task eligibility
    for phase in ("broad", "expert"):
        with (args.prepared / f"sampling_{phase}.jsonl").open() as f:
            for line in f:
                row = json.loads(line)
                rid, side, task = row.get("replay_id"), row.get("player"), row.get("task")
                if rid is None:
                    continue
                g = games.get(rid)
                t.ok(f"B_sampling_{phase}")
                if g is None or g["split"] != "train":
                    t.bad(f"B_sampling_{phase}_non_train", (rid, g and g["split"]))
                    continue
                rating = raw_ratings[rid].get(side)
                if phase == "expert":
                    if rating is None or rating < 1600:
                        t.bad("B_sampling_expert_floor", (rid, side, task, rating))
                elif task in POLICY_TASKS:
                    if rating is None or rating < 1200:
                        t.bad("B_sampling_broad_policy_floor", (rid, side, task, rating))
                    if g.get("showteam"):
                        t.bad("B_sampling_policy_open_sheet", rid)
                elif task == "outcome":
                    t.counts["B_sampling_broad_outcome_below_1200"] += rating is not None and rating < 1200
                    if rating is None:
                        t.bad("B_sampling_outcome_unknown_rating", (rid, side))

    # ---------------- C. fog check on sampled games ----------------
    for rid, recs in fog_records.items():
        lines = json.loads((args.raw / f"{rid}.json").read_text())["log"].split("\n")
        # first public reveal line index per (side, species, kind, value)
        first = {}
        ident_species = {}
        for idx, line in enumerate(lines):
            p = line.split("|")
            if len(p) < 3:
                continue
            tag = p[1]
            if tag in ("switch", "drag", "replace") and len(p) > 3:
                ident_species[p[2][:2] + p[2][3:]] = species_key(p[3])
            if tag == "detailschange" and len(p) > 3:
                key = p[2][:2] + p[2][3:]
                first.setdefault((key, "mega"), idx)
            mentioned = [x for x in p[2:] if re.match(r"^(\[of\] )?p[12][ab]?: ", x)]
            for x in mentioned:
                x = x.replace("[of] ", "")
                key = x[:2] + x[x.index(":"):]  # "p2: Nick"
                for token in p[3:]:
                    tid = toid(token.split(":")[-1])
                    if tid:
                        first.setdefault((key, "token", tid), idx)
            if tag in ("move", "cant") and len(p) > 3:
                key = p[2][:2] + p[2][3:]
                # a blocked attempt (`|cant|X|move: Taunt|Trick Room`) also reveals the move
                first.setdefault((key, "move", toid(p[-1] if tag == "cant" else p[3])), idx)
        for side, log_line, turn, kind, opp in recs:
            oside = "p2" if side == "p1" else "p1"
            keys_by_species = collections.defaultdict(list)
            for key, sp in ident_species.items():
                if key.startswith(oside):
                    keys_by_species[sp].append(key)
            for mon in opp["active"]:
                if not mon:
                    continue
                keys = keys_by_species.get(canonical_species_id(mon["species"]), [])
                if not keys:
                    t.bad("C_opp_species_not_in_log", (rid, side, turn, mon["species"]))
                    continue
                def seen(kind_, value):
                    return min((first[(k, kind_, value)] if kind_ != "mega" else first[(k, "mega")]
                                for k in keys if ((k, kind_, value) if kind_ != "mega" else (k, "mega")) in first),
                               default=None)
                for mv in mon["revealed_moves"]:
                    t.ok("C_opp_move")
                    at = seen("move", mv)
                    if at is None or at >= log_line:
                        t.bad("C_opp_move", (rid, side, turn, kind, mon["species"], mv, at, log_line))
                for field in ("item", "ability"):
                    val = mon[field]
                    if not val:
                        continue
                    t.ok(f"C_opp_{field}")
                    at = seen("token", val)
                    if at is not None and at < log_line:
                        continue
                    if field == "ability":
                        forme = mon["species"] + ("mega" if mon["mega"] else "")
                        if single_ability(mon["species"]) == val or any(
                                single_ability(f) == val for f in load_species() if f.startswith(forme) and mon["mega"]):
                            continue
                    if field == "item" and mon["mega"]:
                        mega_at = seen("mega", None)
                        if mega_at is not None and mega_at < log_line and val.endswith("ite"):
                            continue
                    t.bad(f"C_opp_{field}", (rid, side, turn, kind, mon["species"], val, at, log_line))
                if mon["mega"]:
                    t.ok("C_opp_mega")
                    at = seen("mega", None)
                    if at is None or at >= log_line:
                        t.bad("C_opp_mega", (rid, side, turn, kind, mon["species"], at, log_line))

    out = {
        "counts": dict(sorted(t.counts.items())),
        "examples": t.examples,
        "outcome_balance": {f"{k[0]}:{k[1]}": v for k, v in sorted(outcome_balance.items())},
        "move_option_sizes": {f"{k[0]}:n={k[1]}:label_in={k[2]}": v for k, v in sorted(option_sizes.items())},
        "fog_games": len(fog_records),
    }
    (args.out / "verification.json").write_text(json.dumps(out, indent=1, default=str))
    for k, v in out["counts"].items():
        print(f"{k:55s} {v}")
    print("outcome balance:", out["outcome_balance"])
    print("move option sizes:", out["move_option_sizes"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
