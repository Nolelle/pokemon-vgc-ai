#!/usr/bin/env python
"""Replay saved ladder decisions with `forbid_repeat_protect` off vs on.

Each ladder game's player-view state-replay bundle is replayed message by message (public
information only, like live play). At every move decision where one of our actives Protected
last turn (`protect_counter >= 1`) the live decision path runs twice under the bundle's own
config: rule OFF (legacy) and rule ON. "Decision path" = the fast search followed by the live
exact judge, plus the clock guard's cheap order (`VgcPlayer._cheap_order`). The report counts:

  * decisions where the rule was active (removed at least one candidate) and where it changed
    the final pick, split by game result and by what the legacy pick did (dropped / kept via
    the "forced" exception / kept via the "partner_ko" exception);
  * what the bot plays instead (attack / switch / other status) in the protecting slot;
  * violations: any rule-ON final or cheap pick that is a removed order (must be 0);
  * concrete before/after examples.

Usage: replay_repeat_protect_rule.py --out runs/eval/repeat_protect_replay.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from vgc.actions import describe_order  # noqa: E402
from vgc.battle_state_replay import _feed_messages, _replay_player  # noqa: E402
from vgc.evaluator import (  # noqa: E402
    STALL_ROLL_MOVES,
    apply_repeat_protect_rule,
    build_context,
    score_joint_orders,
)
from vgc.exact_judge import ExactJudge  # noqa: E402

MAIN_CHECKOUT = Path("/Users/edmundyu/code/projects/pokemon-vgc-ai")
FORMAT = "gen9championsvgc2026regmc"
SESSIONS = ("20261008T215234Z", "20261010T003212Z")


def slot_parts(desc: str) -> list[str]:
    return [p.strip() for p in desc.split(" / ")]


def slot_kind(part: str) -> str:
    """attack / switch / protect / other for one slot of a `describe_order` string."""
    if part.startswith("switch->"):
        return "switch"
    move = part.split("@")[0].removesuffix("-mega")
    if move in STALL_ROLL_MOVES:
        return "protect"
    if move in {"wideguard", "quickguard", "matblock", "craftyshield"}:
        return "side-guard"
    from vgc.data import load_moves

    data = load_moves().get(move)
    if data is None:
        return "other"
    return "attack" if data.get("category") != "Status" else "status"


def repeat_slots_of(battle) -> list[int]:
    out = []
    for slot, mon in enumerate(battle.active_pokemon or []):
        if mon is not None and not mon.fainted and int(getattr(mon, "protect_counter", 0)) >= 1:
            out.append(slot)
    return out


def uses_repeat_protect(desc: str, slots: list[int]) -> bool:
    parts = slot_parts(desc)
    return any(slot < len(parts) and slot_kind(parts[slot]) == "protect" for slot in slots)


def board_context(battle) -> str:
    """One line: our actives (hp% / Protect counter) vs foe actives, and the field."""

    def mon(m) -> str:
        if m is None:
            return "-"
        hp = f"{round(100 * m.current_hp_fraction)}%" if m.current_hp_fraction is not None else "?"
        return f"{m.species} {hp}" + (f" (prot{m.protect_counter})" if m.protect_counter else "")

    ours = ", ".join(mon(m) for m in battle.active_pokemon)
    foes = ", ".join(mon(m) for m in battle.opponent_active_pokemon)
    fields = [str(f).split(".")[-1].split(" ")[0].lower() for f in battle.fields]
    weather = [str(w).split(".")[-1].split(" ")[0].lower() for w in battle.weather]
    foe_tw = any("tailwind" in str(c).lower() for c in battle.opponent_side_conditions)
    our_tw = any("tailwind" in str(c).lower() for c in battle.side_conditions)
    extra = [x for x in (fields + weather + (["foe tailwind"] if foe_tw else [])
                         + (["our tailwind"] if our_tw else [])) if x]
    return f"us [{ours}] vs foe [{foes}]; field {extra}"


def load_games(args) -> list[dict]:
    rows = [json.loads(line) for line in args.ladder_jsonl.read_text().splitlines() if line.strip()]
    rows = [
        r
        for r in rows
        if r.get("format") == FORMAT
        and str(r.get("session_id", "")) in args.sessions
        and r.get("lost") in (True, False)
    ]
    if args.only:
        rows = [r for r in rows if any(tag in r["battle_tag"] for tag in args.only)]
    return rows[: args.limit] if args.limit else rows


async def replay_game(row: dict, args) -> dict:
    path = Path(row["state_replay_path"])
    bundle = json.loads(path.read_text())
    metas = bundle["decisions"]
    player, tag = _replay_player(bundle)
    base = replace(player.config, clock_guard_enabled=False)
    cfg_off = replace(
        base, exact_judge_live=True, exact_judge_budget_s=args.budget, forbid_repeat_protect=False
    )
    cfg_on = replace(cfg_off, forbid_repeat_protect=True)
    records: list[dict] = []
    decisions_seen = {"n": 0}

    def judged_pick(config, battle, memory) -> tuple[str, dict]:
        player.config = config
        fast = player._fast_search(battle, memory)
        fast_desc = describe_order(fast[0].order)
        judge = ExactJudge(config, bundle["own_packed_team"])
        result = judge.rerank(battle, memory, list(fast))
        rep = judge.log[-1]
        return describe_order(result[0].order), {
            "fast_pick": fast_desc,
            "judge_status": rep["status"],
            "judge_error": rep.get("error"),
            "overturned": rep.get("overturned"),
        }

    def analyse(battle) -> None:
        recorder = player._decision_replay_recorder
        idx = len(recorder._stream(battle.battle_tag).decisions) - 1
        meta = metas[idx]
        if meta["phase"] == "team_preview" or len(meta.get("legal_actions") or []) <= 1:
            return
        decisions_seen["n"] += 1
        slots = repeat_slots_of(battle)
        if not slots:
            return
        memory = getattr(battle, "_vgc_battle_memory", None)
        turn = int(meta.get("turn") or 0)
        recorded = str(meta.get("chosen_order"))
        if memory is not None:
            memory.record_choice(turn, recorded)

        # What the rule removes from the myopic list, and which exceptions keep a repeat.
        scored_all = score_joint_orders(battle, cfg_off)
        ctx = build_context(battle, cfg_on)
        kept = apply_repeat_protect_rule(list(scored_all), ctx, cfg_on)
        kept_ids = {id(e) for e in kept}
        removed = [e for e in scored_all if id(e) not in kept_ids]
        removed_desc = {describe_order(e.order) for e in removed}
        exception_of = {
            describe_order(e.order): e.breakdown["repeat_protect_exception"]
            for e in kept
            if "repeat_protect_exception" in e.breakdown
        }
        rec = {
            "game": row["battle_tag"],
            "session": row["session_id"],
            "lost": bool(row["lost"]),
            "turn": turn,
            "decision_index": idx,
            "recorded": recorded,
            "repeat_slots": slots,
            "n_myopic": len(scored_all),
            "n_removed": len(removed),
            "recorded_repeat_protect": uses_repeat_protect(recorded, slots),
            "context": board_context(battle),
        }
        if not removed:
            rec["rule_active"] = False
            records.append(rec)
            return
        rec["rule_active"] = True

        off_pick, off_info = judged_pick(cfg_off, battle, memory)
        on_pick, on_info = judged_pick(cfg_on, battle, memory)
        player.config = cfg_off
        cheap_off = describe_order(_cheap(player, battle))
        player.config = cfg_on
        cheap_on = describe_order(_cheap(player, battle))
        off_repeat = uses_repeat_protect(off_pick, slots)
        if not off_repeat:
            path_kind = "legacy_pick_not_a_repeat"
        elif off_pick in removed_desc:
            path_kind = "dropped"
        else:
            path_kind = "kept_" + exception_of.get(off_pick, "unclassified")
        rec.update(
            off_pick=off_pick,
            on_pick=on_pick,
            changed=off_pick != on_pick,
            off_info=off_info,
            on_info=on_info,
            legacy_pick_path=path_kind,
            off_repeat_protect=off_repeat,
            on_repeat_protect=uses_repeat_protect(on_pick, slots),
            on_pick_is_removed=on_pick in removed_desc,
            cheap_off=cheap_off,
            cheap_on=cheap_on,
            cheap_off_was_removed=cheap_off in removed_desc,
            cheap_on_is_removed=cheap_on in removed_desc,
            exceptions_kept={
                "forced": sum(1 for v in exception_of.values() if v == "forced"),
                "partner_ko": sum(1 for v in exception_of.values() if v == "partner_ko"),
            },
            replacement_kinds=[
                slot_kind(slot_parts(on_pick)[s]) for s in slots if s < len(slot_parts(on_pick))
            ],
        )
        records.append(rec)

    def decide(battle):
        try:
            analyse(battle)
        except Exception as exc:  # noqa: BLE001
            records.append(
                {
                    "game": row["battle_tag"],
                    "error": f"{type(exc).__name__}: {exc}",
                    "trace": traceback.format_exc(limit=5),
                }
            )
        return player.choose_random_move(battle)

    player.decide = decide
    await _feed_messages(player, tag, [[str(p) for p in m] for m in bundle["messages"]])
    return {"game": row["battle_tag"], "records": records, "decisions": decisions_seen["n"]}


def _cheap(player, battle):
    return player._cheap_order(battle)


def _run_game(row: dict, args) -> dict:
    try:
        return asyncio.run(replay_game(row, args))
    except Exception as exc:  # noqa: BLE001
        return {
            "game": row["battle_tag"],
            "decisions": 0,
            "records": [{"game": row["battle_tag"], "error": f"game failed: {exc!r}"}],
        }


def summarise(records: list[dict], total_decisions: int) -> dict:
    errors = [r for r in records if "error" in r]
    good = [r for r in records if "error" not in r]
    active = [r for r in good if r.get("rule_active")]
    out: dict = {
        "move_decisions_replayed": total_decisions,
        "decisions_with_a_repeat_slot": len(good),
        "rule_removed_candidates": len(active),
        "errors": len(errors),
    }
    table = {}
    for label, sel in (
        ("loss", [r for r in active if r["lost"]]),
        ("win", [r for r in active if not r["lost"]]),
        ("all", active),
    ):
        changed = [r for r in sel if r["changed"]]
        paths: dict[str, int] = {}
        for r in sel:
            paths[r["legacy_pick_path"]] = paths.get(r["legacy_pick_path"], 0) + 1
        instead: dict[str, int] = {}
        for r in changed:
            for kind in r["replacement_kinds"]:
                instead[kind] = instead.get(kind, 0) + 1
        table[label] = {
            "rule_active": len(sel),
            "final_pick_changed": len(changed),
            "legacy_pick_path": paths,
            "legacy_repeat_protect_picks": sum(r["off_repeat_protect"] for r in sel),
            "rule_on_repeat_protect_picks": sum(r["on_repeat_protect"] for r in sel),
            "replacement_in_protecting_slot": instead,
            "judge_statuses_on": _count(r["on_info"]["judge_status"] for r in sel),
        }
    out["groups"] = table
    out["violations"] = {
        "on_pick_is_removed": sum(r["on_pick_is_removed"] for r in active),
        "cheap_on_is_removed": sum(r["cheap_on_is_removed"] for r in active),
        "cheap_off_was_removed (legacy fallback would have repeated)": sum(
            r["cheap_off_was_removed"] for r in active
        ),
    }
    out["recorded_repeat_protects_in_replayed_games"] = {
        "loss": sum(r["recorded_repeat_protect"] for r in good if r["lost"]),
        "win": sum(r["recorded_repeat_protect"] for r in good if not r["lost"]),
    }
    return out


def _count(items) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in items:
        out[item] = out.get(item, 0) + 1
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--ladder-jsonl", type=Path, default=MAIN_CHECKOUT / "runs" / "ladder.jsonl")
    p.add_argument("--sessions", nargs="*", default=list(SESSIONS))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--only", nargs="*", default=[], help="battle tag substrings to keep")
    p.add_argument("--budget", type=float, default=120.0, help="judge wall-clock cap per call")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--summary-only", action="store_true")
    args = p.parse_args()
    if args.summary_only:
        records = [json.loads(line) for line in args.out.read_text().splitlines() if line.strip()]
        print(json.dumps(summarise(records, 0), indent=1))
        return 0
    games = load_games(args)
    print(f"{len(games)} games", flush=True)
    records: list[dict] = []
    total = 0
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_run_game, row, args) for row in games]
        for done in as_completed(futures):
            res = done.result()
            records.extend(res["records"])
            total += res["decisions"]
            print(
                f"  {res['game'][-24:]}: {len(res['records'])} records ({time.time() - started:.0f}s)",
                flush=True,
            )
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text("".join(json.dumps(r, default=str) + "\n" for r in records))
    print(json.dumps(summarise(records, total), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
