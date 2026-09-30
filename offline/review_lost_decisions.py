#!/usr/bin/env python
"""Engine re-check of the bot's lost public-ladder games.

For every move decision the bot made in a lost ladder game, rebuild exactly what it saw
(the recorded player-view message stream, cut at that decision), then:

* "live" arm -- re-run the shipped decision (`VgcPlayer.decide`, i.e. `vgc.search` with
  the config stored in the replay bundle) and check it reproduces the recorded choice;
* "deep" arm -- re-score every candidate (the bot's actual pick is always forced into the
  searched set) with the official Showdown engine through `LiveExactMirror` +
  `search_joint_orders_exact`, with much wider budgets than live play, averaged over
  hidden-information beliefs by `combine_belief_rankings`.

Regret = deep score of the deep best order minus deep score of the order actually played.
Units are the evaluator's (about 1 point per 1% HP: single digits small, three digits
blunders).

WHAT THIS CAN AND CANNOT FIND.  It finds turns where the bot did not think hard enough:
a better line existed under the SAME scoring, and a wider search would have found it. It
cannot find flaws in the value judgement itself -- both arms rank positions with the same
hand-weighted `_position_value`/myopic terms, so a systematically wrong valuation
(over- or under-rating a position type) is invisible here. Public information only: the
rebuild sees the same messages the bot saw and the opponent's hidden sets are beliefs,
never peeked.
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
from dataclasses import asdict, replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vgc.actions import choice_wire_message, describe_order  # noqa: E402
from vgc.battle_state_replay import (  # noqa: E402
    _feed_messages,
    _replay_player,
)
from vgc.evaluation import clustered_mean  # noqa: E402
from vgc.models import PolicyConfig  # noqa: E402
from vgc.rl.exact_search import (  # noqa: E402
    combine_belief_rankings,
    search_joint_orders_exact,
)
from vgc.rl.live_mirror import LiveExactMirror  # noqa: E402
from vgc.search import _select_search_candidates, search_joint_orders  # noqa: E402

MAIN_CHECKOUT = Path("/Users/edmundyu/code/projects/pokemon-vgc-ai")
FORMAT = "gen9championsvgc2026regmc"
THRESHOLDS = (10.0, 50.0, 100.0)

CAVEAT = (
    "Engine re-check, not a judgement audit. It finds turns where the bot did not search "
    "widely enough: a better line existed under the SAME scoring and a wider, exact-Showdown "
    "search found it. It cannot find flaws in the value judgement itself, because the deep "
    "arm ranks positions with the same hand-weighted evaluator (myopic score + _position_value "
    "+ frozen weights) as live play; a systematically wrong valuation is invisible here. "
    "Opponent hidden sets/spreads/bring are posterior BELIEFS (public information only), "
    "sampled to a handful of branches; regret is measured under those beliefs and under a "
    "handful of random-outcome samples per branch, so small regrets are noise. The opponent "
    "model in the deep arm is its top-N myopic replies, not a full opponent search. "
    "Replay memory: BattleMemory is fed the bot's recorded earlier picks. Very large regrets "
    "(hundreds to thousands) are terminal-position values (a wiped side), not HP points; read "
    "them as 'this line loses on the spot under the beliefs', and prefer the shares, the median "
    "and the capped mean over the raw mean. The exact search looks ONE turn ahead and scores a "
    "wiped side as a huge loss, so with a single Pokemon left, Protect (survive this turn) "
    "beats almost anything and 'regret' there mostly measures one turn of delay, not a proven "
    "better plan; shares_decisions_with_two_own_mons_active excludes those. "
    "23 games / one bot / one opponent "
    "pool: regret here says nothing about win rate directly."
)


def load_games(ladder: Path, limit: int | None) -> list[dict]:
    rows = [json.loads(line) for line in ladder.read_text().splitlines() if line.strip()]
    chosen = [
        row
        for row in rows
        if row.get("format") == FORMAT
        and row.get("lost") is True
        and "local" not in str(row.get("session_id", ""))
    ]
    return chosen[:limit] if limit else chosen


def deep_config(args: argparse.Namespace) -> PolicyConfig:
    return replace(
        PolicyConfig(),
        search_our_candidates=args.deep_candidates,
        search_opp_candidates=args.deep_opp_candidates,
        exact_search_future_samples=args.deep_future_samples,
        exact_search_state_hypotheses=args.deep_state_hypotheses,
        exact_search_spread_hypotheses=args.deep_spread_hypotheses,
        exact_search_set_hypotheses=args.deep_set_hypotheses,
        exact_search_bring_hypotheses=args.deep_bring_hypotheses,
        exact_search_total_hypotheses=args.deep_total_hypotheses,
    )


def deep_rank(mirror: LiveExactMirror, battle, memory, config: PolicyConfig, actual_msg, key):
    """Exact-Showdown ranking averaged over beliefs; the actual order is always searched."""

    def selector(ranked, cfg):
        searched, unsearched = _select_search_candidates(ranked, cfg)
        if all(entry.order.message != actual_msg for entry in searched):
            forced = next((e for e in unsearched if e.order.message == actual_msg), None)
            if forced is not None:
                # Swap the weakest searched entry out so the searched count stays at the
                # configured budget (the selector contract is exactly that many).
                unsearched = [e for e in unsearched if e is not forced]
                searched = list(searched)
                unsearched.insert(0, searched.pop())
                searched.append(forced)
        return searched, unsearched

    beliefs = mirror.hypotheses(battle, memory)
    root = None
    rankings = []
    try:
        for belief in beliefs:
            root = (
                mirror.rebase(root, battle, belief)
                if root is not None
                else mirror.build(battle, belief)
            )
            ranked = search_joint_orders_exact(
                root, "p1", config, candidate_selector=selector, randomness_key=key
            )
            rankings.append((belief.weight, ranked))
    finally:
        if root is not None:
            root.close()
    return combine_belief_rankings(rankings), len(beliefs)


async def review_game(row: dict, args: argparse.Namespace, cfg: PolicyConfig) -> dict:
    path = Path(row["state_replay_path"])
    if not path.exists():
        path = args.ladder_dir / "state-replays" / path.name
    bundle = json.loads(path.read_text())
    decisions_meta = bundle["decisions"]
    player, tag = _replay_player(bundle)
    mirror = LiveExactMirror(bundle["own_packed_team"], cfg)
    records: list[dict] = []
    skips: list[dict] = []
    game_id = row["battle_tag"]

    def skip(idx, turn, reason):
        skips.append({"game": game_id, "decision_index": idx, "turn": turn, "reason": reason})

    def analyse(battle) -> None:
        recorder = player._decision_replay_recorder
        idx = len(recorder._stream(battle.battle_tag).decisions) - 1
        meta = decisions_meta[idx]
        turn = int(meta.get("turn") or 0)
        if meta["phase"] == "team_preview":
            return
        legal = meta.get("legal_actions") or []
        if len(legal) <= 1:
            skip(idx, turn, "single legal option (nothing to choose)")
            return
        recorded = str(meta.get("chosen_order"))
        wire = str(meta.get("chosen_order_wire") or "")
        try:
            started = time.perf_counter()
            live = search_joint_orders(battle, player.config)
            live_s = time.perf_counter() - started
        except Exception as exc:  # noqa: BLE001
            skip(idx, turn, f"live search raised {exc!r}")
            return
        if not live:
            skip(idx, turn, "live search returned no orders")
            return
        live_desc = describe_order(live[0].order)
        reproduced = live_desc == recorded or choice_wire_message(live[0].order) == wire
        # The bot's actual order, located among the rebuilt legal orders.
        actual = next(
            (
                e.order
                for e in live
                if describe_order(e.order) == recorded or choice_wire_message(e.order) == wire
            ),
            None,
        )
        memory = getattr(battle, "_vgc_battle_memory", None)
        memory.record_choice(turn, recorded)  # keep the replay's own-history = the record
        if actual is None:
            skip(idx, turn, "recorded order not legal in the rebuilt battle")
            return
        try:
            started = time.perf_counter()
            ranked, n_beliefs = deep_rank(
                mirror, battle, memory, cfg, actual.message, f"{game_id}:{turn}:{idx}"
            )
            deep_s = time.perf_counter() - started
        except Exception as exc:  # noqa: BLE001
            skip(idx, turn, f"deep search failed: {type(exc).__name__}: {exc}")
            return
        by_msg = {e.order.message: e for e in ranked}
        best = ranked[0]
        act = by_msg[actual.message]
        if not act.breakdown.get("searched"):
            skip(idx, turn, "actual order not searched by deep arm")
            return
        live_entry = by_msg.get(live[0].order.message)
        records.append(
            {
                "game": game_id,
                "decision_index": idx,
                "turn": turn,
                "phase": meta["phase"],
                "bot_order": recorded,
                "deep_best_order": describe_order(best.order),
                "regret": max(0.0, best.score - act.score),
                "deep_best_score": best.score,
                "bot_deep_score": act.score,
                "reproduced_live_choice": bool(reproduced),
                "live_rebuilt_order": live_desc,
                "live_rebuilt_order_regret": (
                    max(0.0, best.score - live_entry.score)
                    if live_entry is not None and live_entry.breakdown.get("searched")
                    else None
                ),
                "single_active_own_mon": "/choose pass" in recorded,
                "legal_orders": len(legal),
                "belief_branches": n_beliefs,
                "live_seconds": round(live_s, 2),
                "deep_seconds": round(deep_s, 2),
            }
        )

    def decide(battle):
        try:
            analyse(battle)
        except Exception as exc:  # noqa: BLE001
            skips.append(
                {
                    "game": game_id,
                    "decision_index": None,
                    "turn": getattr(battle, "turn", None),
                    "reason": f"analysis crashed: {exc!r}",
                    "trace": traceback.format_exc(limit=3),
                }
            )
        return player.choose_random_move(battle)

    player.decide = decide  # instance override: choose_move still records/wraps as live
    try:
        await _feed_messages(player, tag, [[str(p) for p in m] for m in bundle["messages"]])
    finally:
        mirror.close()
    return {"row": row, "records": records, "skips": skips}


def summarise(results: list[dict], args, cfg, elapsed: float) -> dict:
    records = [r for res in results for r in res["records"]]
    skips = [s for res in results for s in res["skips"]]
    per_game = []
    for res in results:
        regs = [r["regret"] for r in res["records"]]
        worst = max(res["records"], key=lambda r: r["regret"], default=None)
        per_game.append(
            {
                "game": res["row"]["battle_tag"],
                "opponent": res["row"].get("opponent"),
                "decisions_reviewed": len(regs),
                "total_regret": sum(regs),
                "max_regret": max(regs, default=0.0),
                "worst_turn": worst["turn"] if worst else None,
            }
        )
    clusters = {}
    for r in records:
        clusters.setdefault(r["game"], []).append(r["regret"])
    cm = clustered_mean(list(clusters.items()))
    shares = {}
    for t in THRESHOLDS:
        ind = [(g, [1.0 if v > t else 0.0 for v in vals]) for g, vals in clusters.items()]
        c = clustered_mean(ind)
        half = 1.96 * c.clustered_se
        shares[f"regret_gt_{int(t)}"] = {
            "count": sum(1 for r in records if r["regret"] > t),
            "share": c.mean,
            "cluster_robust_ci95": [max(0.0, c.mean - half), min(1.0, c.mean + half)],
        }
    two_active = {}
    for r in records:
        if not r["single_active_own_mon"]:
            two_active.setdefault(r["game"], []).append(r["regret"])
    shares_two_active = {}
    for t in THRESHOLDS:
        c = clustered_mean(
            [(g, [1.0 if v > t else 0.0 for v in vals]) for g, vals in two_active.items()]
        )
        half2 = 1.96 * c.clustered_se
        shares_two_active[f"regret_gt_{int(t)}"] = {
            "count": sum(1 for vals in two_active.values() for v in vals if v > t),
            "of": sum(len(v) for v in two_active.values()),
            "share": c.mean,
            "cluster_robust_ci95": [max(0.0, c.mean - half2), min(1.0, c.mean + half2)],
        }
    half = 1.96 * cm.clustered_se
    reasons: dict[str, int] = {}
    for s in skips:
        key = s["reason"].split(":")[0][:80]
        reasons[key] = reasons.get(key, 0) + 1
    repro = sum(1 for r in records if r["reproduced_live_choice"])
    return {
        "overall": {
            "games": len(results),
            "decisions_reviewed": len(records),
            "decisions_skipped": len(skips),
            "skip_reasons": reasons,
            "live_reproduction_rate": repro / len(records) if records else None,
            "live_reproduced": repro,
            "mean_regret": cm.mean,
            "median_regret": statistics.median(r["regret"] for r in records) if records else None,
            "mean_regret_capped_at_300": (
                sum(min(r["regret"], 300.0) for r in records) / len(records) if records else None
            ),
            "mean_regret_cluster_ci95": [max(0.0, cm.mean - half), cm.mean + half],
            "team_effect_sd_tau": cm.tau,
            "shares": shares,
            "shares_decisions_with_two_own_mons_active": shares_two_active,
            "elapsed_seconds": round(elapsed, 1),
        },
        "budget": {
            "deep": {
                k: getattr(cfg, k)
                for k in (
                    "search_our_candidates",
                    "search_opp_candidates",
                    "exact_search_future_samples",
                    "exact_search_state_hypotheses",
                    "exact_search_spread_hypotheses",
                    "exact_search_set_hypotheses",
                    "exact_search_bring_hypotheses",
                    "exact_search_total_hypotheses",
                )
            },
            "live": {
                k: getattr(PolicyConfig(), k)
                for k in (
                    "search_our_candidates",
                    "search_opp_candidates",
                    "exact_search_future_samples",
                    "use_rolling_horizon",
                    "rolling_horizon_turns",
                )
            },
        },
        "caveat": CAVEAT,
        "per_game": per_game,
        "decisions": records,
        "skipped": skips,
        "args": {k: str(v) for k, v in vars(args).items()},
        "deep_config": asdict(cfg),
    }


def _review_one(row: dict, args: argparse.Namespace, cfg: PolicyConfig) -> dict:
    try:
        return asyncio.run(review_game(row, args, cfg))
    except Exception as exc:  # noqa: BLE001
        return {
            "row": row,
            "records": [],
            "skips": [
                {
                    "game": row["battle_tag"],
                    "decision_index": None,
                    "turn": None,
                    "reason": f"game failed: {type(exc).__name__}: {exc}",
                }
            ],
        }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ladder-dir", type=Path, default=MAIN_CHECKOUT / "runs" / "ladder")
    p.add_argument("--ladder-jsonl", type=Path, default=MAIN_CHECKOUT / "runs" / "ladder.jsonl")
    p.add_argument(
        "--output", type=Path, default=MAIN_CHECKOUT / "runs" / "eval" / "lost_decision_review.json"
    )
    p.add_argument("--games", type=int, default=None, help="only the first N lost games")
    p.add_argument("--deep-candidates", type=int, default=16)
    p.add_argument("--deep-opp-candidates", type=int, default=12)
    p.add_argument("--deep-future-samples", type=int, default=6)
    p.add_argument("--deep-state-hypotheses", type=int, default=4)
    p.add_argument("--deep-spread-hypotheses", type=int, default=3)
    p.add_argument("--deep-set-hypotheses", type=int, default=2)
    p.add_argument("--deep-bring-hypotheses", type=int, default=2)
    p.add_argument("--deep-total-hypotheses", type=int, default=4)
    p.add_argument("--workers", type=int, default=6, help="games reviewed in parallel")
    args = p.parse_args()

    rows = load_games(args.ladder_jsonl, args.games)
    cfg = deep_config(args)
    started = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(_review_one, row, args, cfg): row for row in rows}
        for n, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            row = res["row"]
            results.append(res)
            regs = [r["regret"] for r in res["records"]]
            print(
                f"[{n}/{len(rows)}] {row['battle_tag']}: {len(regs)} decisions, "
                f"{len(res['skips'])} skipped, max regret {max(regs, default=0):.1f}",
                flush=True,
            )
            out = summarise(results, args, cfg, time.perf_counter() - started)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(out, indent=2, default=str) + "\n")

    out = summarise(results, args, cfg, time.perf_counter() - started)
    o = out["overall"]
    print(
        f"\nreviewed {o['decisions_reviewed']} decisions in {o['games']} games "
        f"(skipped {o['decisions_skipped']}: {o['skip_reasons']})"
    )
    if o["decisions_reviewed"]:
        print(
            f"live reproduction {o['live_reproduced']}/{o['decisions_reviewed']} "
            f"= {o['live_reproduction_rate']:.1%}; mean regret {o['mean_regret']:.2f} "
            f"(median {o['median_regret']:.2f}, capped-at-300 mean "
            f"{o['mean_regret_capped_at_300']:.2f}) "
            f"CI95 {o['mean_regret_cluster_ci95']}"
        )
        for k, v in o["shares"].items():
            print(f"  {k}: {v['count']} ({v['share']:.1%}) CI95 {v['cluster_robust_ci95']}")
    print("\n10 worst decisions (regret | game turn | bot played -> deep preferred)")
    worst = sorted(out["decisions"], key=lambda r: -r["regret"])[:10]
    for r in worst:
        print(
            f"{r['regret']:7.1f} | {r['game'][-12:]} T{r['turn']} | "
            f"{r['bot_order']}  ->  {r['deep_best_order']}"
        )
    print(f"\nCaveat: {CAVEAT}\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
