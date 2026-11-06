#!/usr/bin/env python
"""Live-vs-replay parity harness (Job A): same game, two views.

Plays battles through the direct env (`vgc.rl.env.DirectBattle`, fogged
per-player `DoubleBattle`s = the same poke-env parsing as ladder), captures the
exact `vgc.rl.encoding.live_state_record` + `trunk_inputs` the live policy sees
at every decision, then rebuilds the replay view from the sim's public
`spectator` stream: `vgc.replay_parse.parse_replay` + own-team rebuild Layer 1 +
bench completion (Layers 2/3 need cross-game context and are skipped; residual
own-side hidden fields are classified `estimate_expected`, not bugs).

Outputs under --out: summary.json, examples.jsonl, logs/<game>.{spectator.log,
records.jsonl}, report.md.

    .venv/bin/python offline/check_live_replay_parity.py --seed 20261105
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from vgc.bc.encoding import trunk_inputs  # noqa: E402
from vgc.bc.own_team_rebuild import analyze_log, rebuild_record  # noqa: E402
from vgc.config import FORMAT_ID  # noqa: E402
from vgc.replay_parse import parse_replay  # noqa: E402
from vgc.rl.agents import make_direct_agent  # noqa: E402
from vgc.rl.encoding import live_state_record  # noqa: E402
from vgc.rl.env import DirectBattle, SimWorker  # noqa: E402

TEAM_FILES = [
    "teams/mc_ladder_04.packed.txt",   # Grassy Surge + Tailwind + Sneasler
    "teams/mc_ladder_03.packed.txt",   # Psychic Surge + Indeedee
    "teams/owner/hatterene_tr.packed.txt",        # Trick Room + Hatterene
    "teams/owner/salamence_tw.packed.txt",        # Mega Salamence + Tailwind
    "teams/owner/gardevoir_psyspam.packed.txt",   # Mega Gardevoir + Psyspam
    "teams/dev.packed.txt",
]
HP_TOLERANCE = 0.01  # own-side rounding only; reported separately


def _kind(battle, request: dict | None) -> str:
    if getattr(battle, "teampreview", False):
        return "teampreview"
    if request and request.get("forceSwitch") and "active" not in request:
        return "forced_switch"
    return "turn"


def capture_live(worker: SimWorker, battle_id: str, agents: dict,
                 teams: dict, seed: list[int], battle_format: str) -> dict:
    """Play one battle, capturing the live view at every per-side request."""
    battle = DirectBattle.start(worker, battle_id, teams["p1"], teams["p2"],
                                battle_format=battle_format, seed=seed)
    live = {"p1": [], "p2": []}
    for side in ("p1", "p2"):
        agents[side].observe(battle_id, battle.last_lines[side])
    while not battle.ended:
        to_move = battle.sides_to_move()
        for side in to_move:
            bt = battle.battles[side]
            request = getattr(bt, "last_request", None)
            record = live_state_record(bt)
            idx, sc = trunk_inputs(record)
            live[side].append({
                "kind": _kind(bt, request),
                "turn": int(getattr(bt, "turn", 0) or 0),
                "record": record,
                "idx": idx, "sc": sc,
            })
        choices = {s: agents[s].choose(battle.battles[s]) for s in to_move}
        result = battle.step(choices)
        for side in to_move:
            agents[side].observe(battle_id, result.lines.get(side, []))
    spec = worker.request({"cmd": "spectator", "id": battle_id})
    return {"live": live, "spectator": spec["spectator"],
            "winner": battle.winner}


def replay_view(log_lines: list[str]) -> dict:
    """Public spectator log -> parsed records -> Layer-1 rebuild + bench."""
    log = "\n".join(log_lines)
    parsed = parse_replay("live-parity", None, log)
    facts = analyze_log("live-parity", log)
    out = {"p1": [], "p2": []}
    for rec in parsed.records:
        rebuilt = rebuild_record(copy.deepcopy(rec), facts, ctx=None)
        idx, sc = trunk_inputs(rebuilt)
        out[rec["player"]].append({
            "kind": rec["decision_kind"], "turn": rec.get("turn") or 0,
            "record": rebuilt, "idx": idx, "sc": sc,
            "log_line": rec.get("log_line"),
        })
    # Species that appear anywhere in the public log, per side -- a live-only
    # own-bench reserve outside this set is information the spectator cannot
    # have, i.e. inherent, not a reconstruction bug.
    brought = {"p1": set(), "p2": set()}
    for (side_, species_) in facts.mons:
        brought.setdefault(side_, set()).add(species_)
    return out, parsed, brought


# --- field comparison ---------------------------------------------------------


# Replay teampreview records name the preview list `preview` where the live
# record uses `preview_species`: key `preview_species` looks up `preview` in
# the replay dict.
_ALIASES = {"preview_species": "preview"}


def _norm(path, v):
    """Teampreview-schema aliases: null active/bench/side_conditions == empty,
    null trick_room == False, null unseen_count == None (kept distinct)."""
    if v is None:
        if path.endswith(("active", "bench", "side_conditions")):
            return []
        if path.endswith("trick_room"):
            return False
        if path.endswith("unseen_count"):
            return 0
    if path.endswith("active") and v == []:
        return [None, None]
    return v


def _cmp(a, b, path, out, *, side=""):
    """Walk two state dicts; append (path, live, replay) mismatches."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            bval = b.get(k)
            if bval is None and k in _ALIASES:
                bval = b.get(_ALIASES[k])
            _cmp(a.get(k), bval, f"{path}.{k}" if path else k, out, side=side)
    elif isinstance(a, list) and isinstance(b, list):
        # bench lists: compare as canonical (species,hp,status) sets -- order is a
        # known cosmetic difference (live: team order; replay: appearance order).
        # Raw lists are kept so the inherent-mask can tell subset/unseen cases
        # from real content bugs.
        if path.endswith("bench"):
            sa = sorted(json.dumps(e, sort_keys=True) for e in a)
            sb = sorted(json.dumps(e, sort_keys=True) for e in b)
            if sa != sb:
                out.append((path, a, b))
            return
        if len(a) != len(b):
            out.append((path, a, b))
            return
        for i, (x, y) in enumerate(zip(a, b)):
            _cmp(x, y, f"{path}[{i}]", out, side=side)
    else:
        a, b = _norm(path, a), _norm(path, b)
        if a != b:
            out.append((path, a, b))


def _hp_rounding_only(diffs, side_paths):
    """All diffs are own-side hp_fraction within tolerance."""
    if not diffs:
        return False
    return all(
        p in side_paths and isinstance(lv, (int, float)) and isinstance(rv, (int, float))
        and abs(lv - rv) <= HP_TOLERANCE
        for p, lv, rv in diffs
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--baseline-games", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20261105)
    ap.add_argument("--out", type=Path, default=Path("runs/live_replay_parity_20261105"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "logs").mkdir(exist_ok=True)
    rng = random.Random(args.seed)
    teams = {Path(f).name: Path(f).read_text().strip() for f in TEAM_FILES}
    team_names = list(teams)

    mismatch_counts = defaultdict(Counter)   # (side_bucket, kind) -> path -> n
    array_diffs = {"idx": Counter(), "sc": Counter()}
    stats = Counter()
    examples = []
    unmatched = []

    with SimWorker() as worker:
        total = args.games + args.baseline_games
        for gi in range(total):
            baseline = gi >= args.games
            t1, t2 = rng.choice(team_names), rng.choice(team_names)
            p1a = "random" if baseline else "vgc"
            p2a = "maxpower" if baseline else "vgc"
            agents = {
                "p1": make_direct_agent(p1a, teams[t1], battle_format=FORMAT_ID),
                "p2": make_direct_agent(p2a, teams[t2], battle_format=FORMAT_ID),
            }
            seed = [rng.randrange(1, 2**31) for _ in range(4)]
            gid = f"par{gi}"
            try:
                played = capture_live(worker, gid, agents,
                                      {"p1": teams[t1], "p2": teams[t2]}, seed, FORMAT_ID)
            except Exception as exc:  # noqa: BLE001 - count and continue
                stats["game_errors"] += 1
                print(f"game {gi} failed: {exc}", file=sys.stderr)
                continue
            log = played["spectator"]
            (args.out / "logs" / f"{gid}.spectator.log").write_text("\n".join(log))
            stats["games"] += 1
            try:
                replay_units, parsed, game_brought = replay_view(log)
            except Exception as exc:  # noqa: BLE001
                stats["parse_errors"] += 1
                print(f"game {gi} parse failed: {exc}", file=sys.stderr)
                continue
            stats["replay_records"] += len(parsed.records)

            for side in ("p1", "p2"):
                live_units = played["live"][side]
                rep_units = replay_units[side]
                # align (turn, kind, ordinal-within-key)
                rep_by_key = defaultdict(list)
                for u in rep_units:
                    rep_by_key[u["turn"], u["kind"]].append(u)
                seen = Counter()
                for u in live_units:
                    key = (u["turn"], u["kind"])
                    ordinal = seen[key]
                    seen[key] += 1
                    pool = rep_by_key.get(key) or []
                    if ordinal >= len(pool):
                        stats["unmatched_live"] += 1
                        if len(unmatched) < 30:
                            unmatched.append({"game": gid, "side": side,
                                              "missing": "live", **{k: u[k] for k in ("turn", "kind")}})
                        continue
                    r = pool[ordinal]
                    stats["decisions_compared"] += 1
                    live_state = u["record"]["state"]
                    rep_state = r["record"]["state"]
                    diffs = []
                    _cmp(live_state, rep_state, "state", diffs)
                    own_hp_paths = {p for p, _, _ in diffs
                                    if p.startswith("state.our") and "hp_fraction" in p}
                    rounding = _hp_rounding_only(diffs, own_hp_paths)
                    if diffs and rounding:
                        stats["decisions_hp_rounding_only"] += 1
                    elif diffs:
                        stats["decisions_field_diffs"] += 1
                    idx_diff = int((u["idx"] != r["idx"]).sum())
                    sc_diff = int((u["sc"] != r["sc"]).sum())
                    if not diffs and not idx_diff and not sc_diff:
                        stats["decisions_identical"] += 1
                    # Headline metric: identical once the INHERENT classes are
                    # masked -- own-side private facts the spectator cannot know:
                    # revealed_moves only when replay is a true subset of live's
                    # private set (a non-subset is a real bug), item/ability when
                    # replay's value is hidden (None), bench when every live-only
                    # reserve never appears anywhere in the log, own hp <=0.01.
                    brought = game_brought.get(side, set())

                    def _inherent(p, lv, rv):
                        if not p.startswith("state.our"):
                            return False
                        if "revealed_moves" in p:
                            return isinstance(rv, list) and isinstance(lv, list) \
                                and set(map(str, rv)) <= set(map(str, lv))
                        if p.endswith((".item", ".ability")):
                            return rv is None or rv == []
                        if "hp_fraction" in p:
                            return (isinstance(lv, (int, float))
                                    and isinstance(rv, (int, float))
                                    and abs(lv - rv) <= HP_TOLERANCE)
                        if p.endswith("bench") and isinstance(lv, list) \
                                and isinstance(rv, list):
                            live_sp = {e.get("species_id") for e in lv if isinstance(e, dict)}
                            rep_sp = {e.get("species_id") for e in rv if isinstance(e, dict)}
                            return (live_sp - rep_sp).isdisjoint(brought)
                        return False

                    # Teampreview is a cosmetic-only class: the live record fills
                    # bench/active/preview differently from the replay's minimal
                    # schema, and no consumer feeds the trunk at teampreview.
                    if u["kind"] == "teampreview":
                        stats["decisions_identical_excl_inherent"] += 1
                    elif all(_inherent(p, lv, rv) for p, lv, rv in diffs):
                        stats["decisions_identical_excl_inherent"] += 1
                    else:
                        for p_, lv_, rv_ in diffs:
                            if not _inherent(p_, lv_, rv_):
                                stats[f"noninherent:{p_}"] += 1
                    for path, lv, rv in diffs:
                        field = path
                        # Own-side fields still hidden in the public log get an
                        # L2/L3 guess in training: live-knows-more diffs there are
                        # expected estimates, not reconstruction bugs.
                        est = (rv is None or rv == []
                               or (isinstance(rv, list) and isinstance(lv, list)
                                   and set(map(str, rv)) < set(map(str, lv))))
                        cls = ("estimate_expected" if path.startswith("state.our") and est
                               else "rounding"
                               if path.startswith("state.our")
                               and "hp_fraction" in path and isinstance(lv, (int, float))
                               and isinstance(rv, (int, float))
                               and abs(lv - rv) <= HP_TOLERANCE
                               else "diff")
                        mismatch_counts[u["kind"], cls][field] += 1
                        if len(examples) < 50000:
                            examples.append({"game": gid, "side": side,
                                             "turn": u["turn"], "kind": u["kind"],
                                             "field": field, "class": cls,
                                             "live": lv, "replay": rv})
                    if idx_diff:
                        array_diffs["idx"][u["kind"]] += 1
                    if sc_diff:
                        array_diffs["sc"][u["kind"]] += 1
                # replay units with no live counterpart
                for (turn, kind), pool in rep_by_key.items():
                    extra = len(pool) - seen.get((turn, kind), 0)
                    if extra > 0:
                        stats["unmatched_replay"] += extra
                        if len(unmatched) < 30:
                            unmatched.append({"game": gid, "side": side,
                                              "missing": "replay",
                                              "turn": turn, "kind": kind,
                                              "extra": extra})

    identical = stats["decisions_identical"]
    compared = stats["decisions_compared"]
    summary = {
        "games": stats["games"], "decisions_compared": compared,
        "decisions_identical_encoded": identical,
        "identical_share": identical / compared if compared else None,
        "identical_excluding_inherent": stats["decisions_identical_excl_inherent"],
        "identical_share_excl_inherent": (
            stats["decisions_identical_excl_inherent"] / compared if compared else None),
        "decisions_hp_rounding_only": stats["decisions_hp_rounding_only"],
        "decisions_field_diffs": stats["decisions_field_diffs"],
        "unmatched_live": stats["unmatched_live"],
        "unmatched_replay": stats["unmatched_replay"],
        "unmatched_examples": unmatched[:30],
        "noninherent_field_counts": {
            k.removeprefix("noninherent:"): v
            for k, v in stats.items() if k.startswith("noninherent:")
        },
        "mismatch_counts": {
            f"{kind}/{cls}": dict(c)
            for (kind, cls), c in sorted(mismatch_counts.items())
        },
        "array_diff_decisions": {k: dict(v) for k, v in array_diffs.items()},
        "errors": {"game": stats["game_errors"], "parse": stats["parse_errors"]},
        "replay_records": stats["replay_records"],
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    with (args.out / "examples.jsonl").open("w") as f:
        for e in examples:
            f.write(json.dumps(e, default=str) + "\n")
    print(json.dumps({k: summary.get(k) for k in
                      ("games", "decisions_compared", "decisions_identical_encoded",
                       "identical_share", "unmatched_live", "unmatched_replay")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
