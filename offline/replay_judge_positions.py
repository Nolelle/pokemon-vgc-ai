#!/usr/bin/env python
"""Re-run the live exact judge on saved public-ladder positions under different knobs.

Each ladder game's player-view state-replay bundle (runs/ladder/state-replays) is replayed
message by message; at every move decision the shipped fast search runs under the bundle's
own config (so the shortlist is the one the live bot judged), then the live exact judge
(`vgc.exact_judge.ExactJudge`) re-ranks it once per ARM -- an arm is a set of PolicyConfig
overrides -- on the same public mirror and the same random-future key (common random
numbers). One JSON line per decision records every arm's pick and exact values, so
pick-change rates between arms can be tabulated (see `summarise` for the tables).

Public information only, exactly like live play: the replay sees the messages the bot saw.

Arms (override with --arms name,name):
    base      the live configuration (4 replies, one-turn horizon)
    la        passive look-ahead (exact_judge_passive_lookahead)
    la_bt     look-ahead, a game ending on the extra turn scored by its board
    div       diverse replies (exact_search_diverse_replies), 4 replies
    wide      16 replies, no diversity (the reference for the reply questions)
    la_div    look-ahead + diverse replies
    la_wide   look-ahead + 16 replies (the reference for la_div)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vgc.actions import describe_order  # noqa: E402
from vgc.battle_state_replay import _feed_messages, _replay_player  # noqa: E402
from vgc.exact_judge import ExactJudge  # noqa: E402
from vgc.principles import SELF_PROTECT_MOVES, WIDE_DEFENSE_MOVES  # noqa: E402

MAIN_CHECKOUT = Path("/Users/edmundyu/code/projects/pokemon-vgc-ai")
FORMAT = "gen9championsvgc2026regmc"
PROTECT_FAMILY = SELF_PROTECT_MOVES | WIDE_DEFENSE_MOVES

ARMS: dict[str, dict] = {
    "base": {},
    "la": {"exact_judge_passive_lookahead": True},
    "la_bt": {
        "exact_judge_passive_lookahead": True,
        "exact_search_passive_lookahead_board_terminals": True,
    },
    "div": {"exact_search_diverse_replies": True},
    "wide": {"exact_judge_opp_candidates": 16},
    "la_div": {"exact_judge_passive_lookahead": True, "exact_search_diverse_replies": True},
    "la_s1": {
        "exact_judge_passive_lookahead": True,
        "exact_judge_passive_lookahead_samples": 1,
    },
    "la_s1_bt": {
        "exact_judge_passive_lookahead": True,
        "exact_judge_passive_lookahead_samples": 1,
        "exact_search_passive_lookahead_board_terminals": True,
    },
    "la_a1": {
        "exact_judge_passive_lookahead": True,
        "exact_judge_passive_lookahead_alternatives": 1,
    },
    "la_wide": {"exact_judge_passive_lookahead": True, "exact_judge_opp_candidates": 16},
}


def slots(desc: str) -> list[str]:
    """Per-slot move ids of a `describe_order` string ('' for a pass / switch)."""

    out = []
    for part in desc.split(" / "):
        part = part.strip()
        if part.startswith("switch->") or part.startswith("/choose"):
            out.append("")
        else:
            out.append(part.split("@")[0].removesuffix("-mega"))
    return out


def is_passive(desc: str) -> bool:
    acting = [s for s in slots(desc) if s]
    return bool(acting) and all(s in PROTECT_FAMILY for s in acting) and not any(
        p.strip().startswith("switch->") for p in desc.split(" / ")
    )


def has_protect(desc: str) -> bool:
    return any(s in PROTECT_FAMILY for s in slots(desc))


def repeat_protect(now: str, before: str | None) -> bool:
    if not before:
        return False
    return any(
        a in PROTECT_FAMILY and b in PROTECT_FAMILY
        for a, b in zip(slots(now), slots(before), strict=False)
    )


def load_games(args) -> list[dict]:
    rows = [json.loads(line) for line in args.ladder_jsonl.read_text().splitlines() if line.strip()]
    rows = [
        r
        for r in rows
        if r.get("format") == FORMAT
        and str(r.get("session_id", "")) == args.session
        and r.get("lost") in (True, False)
    ]
    if args.only:
        rows = [r for r in rows if any(tag in r["battle_tag"] for tag in args.only)]
    return rows[: args.limit] if args.limit else rows


def _opp_tailwind(battle) -> bool:
    return any("tailwind" in str(c).lower() for c in getattr(battle, "opponent_side_conditions", {}))


def _trick_room(battle) -> bool:
    return any("trick" in str(f).lower() for f in getattr(battle, "fields", {}))


async def replay_game(row: dict, args) -> dict:
    path = Path(row["state_replay_path"])
    if not path.exists():
        path = args.ladder_dir / "state-replays" / path.name
    bundle = json.loads(path.read_text())
    metas = bundle["decisions"]
    player, tag = _replay_player(bundle)
    base_cfg = replace(player.config, clock_guard_enabled=False)
    player.config = base_cfg
    arm_names = [a for a in args.arms if a in ARMS]
    records: list[dict] = []
    prev = {"turn": None, "order": None}

    def analyse(battle) -> None:
        recorder = player._decision_replay_recorder
        idx = len(recorder._stream(battle.battle_tag).decisions) - 1
        meta = metas[idx]
        turn = int(meta.get("turn") or 0)
        if meta["phase"] == "team_preview":
            return
        recorded = str(meta.get("chosen_order"))
        prior = prev["order"] if prev["turn"] is not None and prev["turn"] == turn - 1 else None
        prev["turn"], prev["order"] = turn, recorded
        if len(meta.get("legal_actions") or []) <= 1:
            return
        memory = getattr(battle, "_vgc_battle_memory", None)
        fast = player._fast_search(battle, memory)
        if not fast:
            return
        memory.record_choice(turn, recorded)
        fast_desc = describe_order(fast[0].order)
        rec = {
            "game": row["battle_tag"],
            "lost": bool(row["lost"]),
            "decision_index": idx,
            "turn": turn,
            "recorded": recorded,
            "fast_pick": fast_desc,
            "prior_recorded": prior,
            "recorded_repeat_protect": repeat_protect(recorded, prior),
            "fast_repeat_protect": repeat_protect(fast_desc, prior),
            "opp_tailwind": _opp_tailwind(battle),
            "trick_room": _trick_room(battle),
            "arms": {},
        }
        for name in arm_names:
            cfg = replace(
                base_cfg,
                exact_judge_live=True,
                exact_judge_budget_s=args.budget,
                **ARMS[name],
            )
            judge = ExactJudge(cfg, bundle["own_packed_team"])
            started = time.perf_counter()
            judge.rerank(battle, memory, list(fast))
            wall = time.perf_counter() - started
            rep = judge.log[-1]
            rec["arms"][name] = {
                "status": rep["status"],
                "error": rep.get("error"),
                "pick": rep.get("exact_pick") or fast_desc,
                "overturned": rep.get("overturned"),
                "gain": rep.get("gain"),
                "elapsed_ms": round(wall * 1000.0, 1),
                "ranking": [
                    (r["order"], r["value"], r["fast_rank"]) for r in rep.get("ranking", [])
                ],
            }
            if name == arm_names[0] and rep["status"] != "ok":
                break  # nothing judged here (forced switch, few candidates): skip other arms
        records.append(rec)

    def decide(battle):
        try:
            analyse(battle)
        except Exception as exc:  # noqa: BLE001
            records.append(
                {
                    "game": row["battle_tag"],
                    "error": f"{type(exc).__name__}: {exc}",
                    "trace": traceback.format_exc(limit=4),
                }
            )
        return player.choose_random_move(battle)

    player.decide = decide
    await _feed_messages(player, tag, [[str(p) for p in m] for m in bundle["messages"]])
    return {"game": row["battle_tag"], "records": records}


def _run_game(row: dict, args) -> dict:
    try:
        return asyncio.run(replay_game(row, args))
    except Exception as exc:  # noqa: BLE001
        return {
            "game": row["battle_tag"],
            "records": [{"game": row["battle_tag"], "error": f"game failed: {exc!r}"}],
        }


# --- tables -------------------------------------------------------------------------------


def group_labels(rec: dict) -> list[str]:
    labels = []
    if "2695880700" in rec["game"] and 7 <= rec["turn"] <= 11:
        labels.append("a_game17_T7-11")
    if rec["lost"] and (rec["recorded_repeat_protect"] or rec["fast_repeat_protect"]):
        labels.append("b_loss_repeat_protect")
    if (not rec["lost"]) and has_protect(rec["recorded"]):
        labels.append("c_win_protect_control")
    if (not rec["lost"]) and any(
        arm.get("ranking") and any(is_passive(o) for o, _v, _r in arm["ranking"])
        for arm in rec["arms"].values()
    ):
        labels.append("c2_win_passive_candidate")
    if (not rec["lost"]) and has_protect(rec["recorded"]) and (
        rec.get("opp_tailwind") or rec.get("trick_room")
    ):
        labels.append("c3_win_protect_vs_speed_control")
    if (not rec["lost"]) and has_protect(rec["recorded"]) and not is_passive(rec["recorded"]):
        labels.append("c4_win_protect_plus_attack")
    labels.append("all")
    return labels


def same(a: str, b: str) -> bool:
    return a == b


def summarise(records: list[dict]) -> dict:
    good = [r for r in records if "arms" in r and r["arms"]]
    out: dict = {"decisions_with_arms": len(good)}
    names = sorted({n for r in good for n in r["arms"]})
    groups: dict[str, list[dict]] = {}
    for rec in good:
        for label in group_labels(rec):
            groups.setdefault(label, []).append(rec)
    table = {}
    for label, recs in sorted(groups.items()):
        row = {"n": len(recs)}
        judged = [r for r in recs if r["arms"].get("base", {}).get("status") == "ok"]
        row["judged"] = len(judged)
        base_picks_passive = sum(1 for r in judged if is_passive(r["arms"]["base"]["pick"]))
        row["base_pick_is_passive"] = base_picks_passive
        row["base_pick_is_repeat_protect"] = sum(
            1 for r in judged if repeat_protect(r["arms"]["base"]["pick"], r["prior_recorded"])
        )
        for name in names:
            if name == "base":
                continue
            have = [r for r in judged if name in r["arms"] and r["arms"][name]["status"] == "ok"]
            changed = [r for r in have if r["arms"][name]["pick"] != r["arms"]["base"]["pick"]]
            out_of_passive = [
                r
                for r in changed
                if is_passive(r["arms"]["base"]["pick"])
                and not is_passive(r["arms"][name]["pick"])
            ]
            into_passive = [
                r
                for r in changed
                if not is_passive(r["arms"]["base"]["pick"]) and is_passive(r["arms"][name]["pick"])
            ]
            row[name] = {
                "compared": len(have),
                "changed_vs_base": len(changed),
                "passive_to_active": len(out_of_passive),
                "active_to_passive": len(into_passive),
                "repeat_protect_before": sum(
                    1 for r in have if repeat_protect(r["arms"]["base"]["pick"], r["prior_recorded"])
                ),
                "repeat_protect_after": sum(
                    1 for r in have if repeat_protect(r["arms"][name]["pick"], r["prior_recorded"])
                ),
            }
        table[label] = row
    out["groups"] = table
    # Reply questions: agreement with the 16-reply reference.
    for ref, cands in (("wide", ("base", "div")), ("la_wide", ("la", "la_div"))):
        recs = [
            r
            for r in good
            if all(n in r["arms"] and r["arms"][n]["status"] == "ok" for n in (ref, *cands))
        ]
        if not recs:
            continue
        out[f"agreement_with_{ref}"] = {
            "n": len(recs),
            **{
                c: sum(1 for r in recs if r["arms"][c]["pick"] == r["arms"][ref]["pick"])
                for c in cands
            },
        }
    # Latency per arm.
    lat = {}
    for name in names:
        ms = [
            r["arms"][name]["elapsed_ms"]
            for r in good
            if name in r["arms"] and r["arms"][name]["status"] in ("ok", "timeout")
        ]
        stat = [r["arms"][name]["status"] for r in good if name in r["arms"]]
        if ms:
            ms.sort()
            lat[name] = {
                "n": len(ms),
                "p50_ms": round(statistics.median(ms), 1),
                "p90_ms": round(ms[int(0.9 * (len(ms) - 1))], 1),
                "p99_ms": round(ms[int(0.99 * (len(ms) - 1))], 1),
                "max_ms": round(ms[-1], 1),
                "timeouts": stat.count("timeout"),
                "errors": stat.count("error"),
                "statuses": {s: stat.count(s) for s in sorted(set(stat))},
            }
    out["latency"] = lat
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--ladder-dir", type=Path, default=MAIN_CHECKOUT / "runs" / "ladder")
    p.add_argument("--ladder-jsonl", type=Path, default=MAIN_CHECKOUT / "runs" / "ladder.jsonl")
    p.add_argument("--session", default="20261010T003212Z")
    p.add_argument("--only", nargs="*", default=[], help="battle tag substrings to keep")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--arms", type=lambda s: s.split(","), default=list(ARMS))
    p.add_argument("--budget", type=float, default=120.0, help="judge wall-clock cap per call")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", type=Path, required=True, help="JSONL of per-decision records")
    p.add_argument("--summary-only", action="store_true", help="re-tabulate an existing --out")
    args = p.parse_args()
    if args.summary_only:
        records = [json.loads(line) for line in args.out.read_text().splitlines() if line.strip()]
        print(json.dumps(summarise(records), indent=1))
        return 0
    games = load_games(args)
    print(f"{len(games)} games, arms {args.arms}", flush=True)
    records: list[dict] = []
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_run_game, row, args) for row in games]
        for done in as_completed(futures):
            res = done.result()
            records.extend(res["records"])
            print(f"  {res['game'][-24:]}: {len(res['records'])} records "
                  f"({time.time() - started:.0f}s)", flush=True)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text("".join(json.dumps(r, default=str) + "\n" for r in records))
    print(json.dumps(summarise(records), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
