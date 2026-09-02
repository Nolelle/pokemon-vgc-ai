"""Rung 3a diagnostic: does signing the effect term change what the exact search picks?

`vgc.rl.exact_search._side_position` scored volatiles and side conditions with `len(...)`,
so a Leech Seed on our own Pokemon and a Substitute we set up were both worth
+`exact_search_effect_weight`, and Stealth Rock on our side scored the same as Tailwind on
our side. `PolicyConfig.exact_search_signed_effects` (default True) spends that same frozen
weight with a sign derived from Champions move targets (`vgc.position_effects`).

This measures how often that changes the search's answer on real public-mirror roots. It is
NOT a strength claim -- it does not play the moves out, and a decision change is not by
itself an improvement. It answers the narrower question 3a has to answer before anything
downstream is worth running: is the fix reaching live decisions at a rate worth gating, or
is it cosmetic?

Games are driven exactly like `offline/evaluate_belief_shortlist_recall.py` -- same pool,
same cheap per-decision budget, same checkpoints, same public information boundary -- so
the two diagnostics are directly comparable. Rates are clustered by our own team file,
because decisions within one team are not independent.

    .venv/bin/python offline/measure_effect_polarity_impact.py --pairs 30
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

from vgc.actions import describe_order
from vgc.battle_memory import BattleMemory
from vgc.config import REPO_ROOT, RUNS_DIR
from vgc.evaluation import clustered_interval
from vgc.evaluator import score_joint_orders
from vgc.gate_evidence import git_head
from vgc.mechanics_state import snapshot_battle
from vgc.position_effects import effect_polarity, side_condition_polarity
from vgc.rl.env import DEFAULT_SHOWDOWN_REPO, DirectBattle, SimWorker, choice_string
from vgc.rl.exact_search import search_joint_orders_exact
from vgc.rl.live_mirror import LiveExactMirror

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_belief_shortlist_recall import (  # noqa: E402
    DIAGNOSTIC_CONFIG,
    _feed_memory,
    _load_team_files,
    _pick_myopic,
    _rank_of,
)

DEFAULT_MANIFEST = REPO_ROOT / "data" / "selfplay" / "archetype_pool_150" / "manifest.json"
DEFAULT_OUTPUT = RUNS_DIR / "eval" / "effect_polarity_impact.json"
# Later than the recall gate's checkpoints on purpose. Turns 2/4/6 are almost always a
# bare board -- no hazards up, no Leech Seed, no screens -- so they cannot exercise the
# effect term at all. Sampling where effects actually exist is what makes the rate mean
# something.
DEFAULT_CHECKPOINTS = (5, 8, 11, 14)


def _signable_effects(state) -> dict[str, int]:
    """Count effects on the board the sign map has an opinion about.

    Reported for context only. It is NOT the denominator for "could the fix have acted
    here": `_position_value` scores each branch as `after - before` with the same `before`
    for every candidate, so an effect already present at the root cancels out and cannot
    move the ranking. Only effects GAINED or LOST inside the searched turn can, and the
    honest test for that is whether the two configurations produce different exchange
    values at all (`exchange_values_differ` below).
    """

    counts = {"helps": 0, "hurts": 0, "unsigned": 0}
    for side in (state.our_side, state.opponent_side):
        for mon in side.pokemon:
            for effect in mon.effects:
                polarity = effect_polarity(effect)
                counts["helps" if polarity > 0 else "hurts" if polarity < 0 else "unsigned"] += 1
        for effect in side.side_conditions:
            polarity = side_condition_polarity(effect)
            counts["helps" if polarity > 0 else "hurts" if polarity < 0 else "unsigned"] += 1
    return counts


def _exchange_values(scored: list) -> dict[str, float]:
    return {
        describe_order(entry.order): entry.breakdown["exchange_value"]
        for entry in scored
        if entry.breakdown.get("searched")
    }


def _evaluate_checkpoint(
    root: DirectBattle,
    memory: BattleMemory,
    own_team: str,
    own_team_file: str,
    opp_team_file: str,
    turn: int,
    config,
) -> tuple[dict[str, Any] | None, str | None]:
    if "p1" not in root.sides_to_move():
        return None, "p1_not_to_move"

    memory.observe_battle(root.battles["p1"])
    setattr(root.battles["p1"], "_vgc_battle_memory", memory)

    mirror = LiveExactMirror(own_team, config)
    try:
        mirror_root = mirror.build(root.battles["p1"])
        try:
            decision_battle = getattr(mirror_root, "_decision_battles", {}).get(
                "p1", mirror_root.battles["p1"]
            )
            decision_battle._vgc_battle_memory = memory
            if not score_joint_orders(decision_battle, config):
                return None, "no_legal_orders"

            board = _signable_effects(snapshot_battle(decision_battle))

            signed = search_joint_orders_exact(
                mirror_root, "p1", replace(config, exact_search_signed_effects=True)
            )
            legacy = search_joint_orders_exact(
                mirror_root, "p1", replace(config, exact_search_signed_effects=False)
            )
            if not signed or not legacy:
                return None, "empty_search"

            signed_pick = describe_order(signed[0].order)
            legacy_pick = describe_order(legacy[0].order)
            signed_values = _exchange_values(signed)
            legacy_values = _exchange_values(legacy)
            deltas = {
                order: signed_values[order] - legacy_values[order]
                for order in signed_values
                if order in legacy_values
            }
            changed_values = {order: d for order, d in deltas.items() if abs(d) > 1e-9}
            return {
                "own_team": own_team_file,
                "opp_team": opp_team_file,
                "turn": turn,
                "signed_pick": signed_pick,
                "legacy_pick": legacy_pick,
                "pick_changed": signed_pick != legacy_pick,
                # Where the signed answer sat in the legacy ranking: rank 2 is a near-tie
                # flip, a deep rank means the sign changed the search's mind entirely.
                "signed_pick_rank_under_legacy": _rank_of(signed_pick, legacy),
                # The real "could this decision have changed" test: did signing the term
                # move any searched candidate's exchange value at all?
                "exchange_values_differ": bool(changed_values),
                "candidates_scored": len(deltas),
                "candidates_whose_value_moved": len(changed_values),
                "max_abs_value_delta": max((abs(d) for d in deltas.values()), default=0.0),
                "board_effects": board,
                "board_has_signed_effect": (board["helps"] + board["hurts"]) > 0,
            }, None
        finally:
            mirror_root.close()
    finally:
        mirror.close()


def run(
    *, manifest_path: Path, pairs: int, checkpoints: tuple[int, ...], seed: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    team_files = _load_team_files(manifest_path, pairs * 2, seed)
    records: list[dict[str, Any]] = []
    skips: dict[str, int] = defaultdict(int)
    config = DIAGNOSTIC_CONFIG
    with SimWorker(DEFAULT_SHOWDOWN_REPO) as worker:
        for pair_index in range(pairs):
            own_path = team_files[2 * pair_index]
            opp_path = team_files[2 * pair_index + 1]
            own_team = own_path.read_text().strip()
            opp_team = opp_path.read_text().strip()
            label = f"{own_path.parent.name}_vs_{opp_path.parent.name}"
            rng = random.Random(seed + pair_index)

            root = DirectBattle.start(
                worker,
                f"effect-polarity-{pair_index}",
                own_team,
                opp_team,
                seed=[rng.randint(1, 2**31 - 1) for _ in range(4)],
            )
            memory = BattleMemory(root.battle_id, our_role="p1")
            try:
                _feed_memory(memory, root.battles["p1"], root.last_lines.get("p1") or [])
                preview = root.step({"p1": "team 1234", "p2": "team 1234"})
                _feed_memory(memory, root.battles["p1"], preview.lines.get("p1") or [])
                next_checkpoint = 0
                for turn in range(1, max(checkpoints) + 1):
                    if root.ended:
                        break
                    if next_checkpoint < len(checkpoints) and turn == checkpoints[next_checkpoint]:
                        try:
                            record, skip_reason = _evaluate_checkpoint(
                                root, memory, own_team, str(own_path), str(opp_path), turn, config
                            )
                        except Exception as exc:  # noqa: BLE001 -- one bad checkpoint
                            # should not abort a long run; the skip is reported.
                            print(f"[{label} turn {turn}] SKIPPED ({exc})", flush=True)
                            skips[f"exception:{type(exc).__name__}"] += 1
                            record, skip_reason = None, None
                        if skip_reason is not None:
                            skips[skip_reason] += 1
                            print(f"[{label} turn {turn}] SKIPPED ({skip_reason})", flush=True)
                        elif record is not None:
                            records.append(record)
                            board = record["board_effects"]
                            print(
                                f"[{label} turn {turn}] "
                                f"{'CHANGED' if record['pick_changed'] else 'same'} "
                                f"signed_rank_under_legacy={record['signed_pick_rank_under_legacy']} "
                                f"board=+{board['helps']}/-{board['hurts']}"
                                f"/?{board['unsigned']}",
                                flush=True,
                            )
                        next_checkpoint += 1
                    choices = {}
                    for side in root.sides_to_move():
                        order = _pick_myopic(root.battles[side], config, rng)
                        if order is not None:
                            choices[side] = choice_string(order)
                    if not choices:
                        break
                    stepped = root.step(choices)
                    _feed_memory(memory, root.battles["p1"], stepped.lines.get("p1") or [])
            finally:
                root.close()
    return records, dict(skips)


def _clustered_rate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Pooled change rate with a cluster-robust interval, clustering by our own team.

    `clustered_interval` wants one `(wins, games)` per cluster -- decisions inside one
    team share a team file and are not independent, so they must not be pooled as if they
    were (the same rule the multi-team gates follow).
    """

    if not records:
        return {"decisions": 0, "teams": 0, "rate": None, "interval": [None, None]}
    per_team: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for record in records:
        bucket = per_team[record["own_team"]]
        bucket[0] += 1 if record["pick_changed"] else 0
        bucket[1] += 1
    clusters = [(wins, games) for wins, games in per_team.values()]
    low, high = clustered_interval(clusters)
    total_changed = sum(wins for wins, _games in clusters)
    total = sum(games for _wins, games in clusters)
    return {
        "decisions": total,
        "teams": len(clusters),
        "rate": total_changed / total,
        "interval": [low, high],
    }


def summarize(records: list[dict[str, Any]], skips: dict[str, int]) -> dict[str, Any]:
    # The honest denominator. An effect already on the board at the root cancels out of
    # every candidate's `after - before`, so it cannot change the ranking; what matters is
    # whether signing the term moved any searched candidate's value at all.
    actionable = [record for record in records if record["exchange_values_differ"]]
    return {
        "git_head": git_head(),
        "decisions": len(records),
        "teams": len({record["own_team"] for record in records}),
        "skips": skips,
        "overall": _clustered_rate(records),
        "when_the_sign_moved_a_candidate_value": _clustered_rate(actionable),
        "decisions_where_a_value_moved": len(actionable),
        "max_value_delta_seen": max(
            (record["max_abs_value_delta"] for record in records), default=0.0
        ),
        "decisions_with_a_signed_effect_at_root": sum(
            1 for record in records if record["board_has_signed_effect"]
        ),
        "unsigned_effect_observations": sum(r["board_effects"]["unsigned"] for r in records),
        "signed_effect_observations": sum(
            r["board_effects"]["helps"] + r["board_effects"]["hurts"] for r in records
        ),
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--pairs", type=int, default=30, help="number of team-pair games")
    parser.add_argument(
        "--checkpoints",
        type=int,
        nargs="+",
        default=list(DEFAULT_CHECKPOINTS),
        help="turns at which to compare the two configurations",
    )
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    records, skips = run(
        manifest_path=args.manifest,
        pairs=args.pairs,
        checkpoints=tuple(sorted(args.checkpoints)),
        seed=args.seed,
    )
    report = summarize(records, skips)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    print()
    print(f"decisions: {report['decisions']} over {report['teams']} teams")
    print(f"skips: {report['skips'] or 'none'}")
    print(
        f"signed effects seen: {report['signed_effect_observations']} "
        f"(unsigned, scored 0: {report['unsigned_effect_observations']})"
    )
    for label, key in (
        ("all decisions", "overall"),
        (
            "decisions where signing moved a candidate value",
            "when_the_sign_moved_a_candidate_value",
        ),
    ):
        block = report[key]
        if block["rate"] is None:
            print(f"top-pick change, {label}: no decisions")
            continue
        low, high = block["interval"]
        print(
            f"top-pick change, {label}: {block['rate']:.3f} [{low:.3f}, {high:.3f}] "
            f"(n={block['decisions']}, teams={block['teams']})"
        )
    print(
        f"decisions where signing moved any candidate value: "
        f"{report['decisions_where_a_value_moved']} / {report['decisions']} "
        f"(largest value shift seen: {report['max_value_delta_seen']:.2f} points)"
    )
    print(f"report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
