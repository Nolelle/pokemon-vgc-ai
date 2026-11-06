#!/usr/bin/env python
"""Part-3 measurement for the own-team rebuild (`vgc.bc.own_team_rebuild`).

Answer-key: units are (game, side, base species) for each Pokemon that appeared on
field in a game where that side has a `|showteam|` line. Truth = that sheet's
item/ability/4 moves. Each unit is rebuilt as if that side's sheet did not exist
(Layer-1 from the game's own events, Layer-2 donors under the production time rule
excluding this game, Layer-3 priors), evaluated at the start-of-game rebuilt set
(pre-mega base ability).

Coverage: own-active item/ability/move-slot sources BEFORE (observed only) vs AFTER
the rebuild over every turn/forced_switch record of train/validation/test games in
the v3 manifest, plus the train 1600+ acting-player subset.

    .venv/bin/python tools/measure_own_team_rebuild.py \
        --data data/bc/decisions_regmc_s8_20261104.jsonl.gz \
        --manifest runs/pretraining_selection_20261102_v3/manifest.jsonl \
        --facts runs/own_team_rebuild_20261104/game_facts.jsonl.gz \
        --out-dir runs/own_team_rebuild_20261104
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vgc.bc.own_team_rebuild import (  # noqa: E402
    COMMON_SET,
    DATA,
    OBSERVED,
    SAME_GAME,
    SAME_PLAYER_GAMES,
    SAME_PLAYER_SHEET,
    RebuildContext,
    UNKNOWN,
    game_facts_from_json,
    game_flagged,
    illusion_affected,
    game_prior_contribution,
    load_manifest,
    resolve_fields,
    subtract_prior,
)

BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 20261104


def _norm(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def answer_key_units(manifest: dict, facts: dict) -> list[dict]:
    units = []
    for game_id, game in manifest.items():
        gf = facts.get(game_id)
        if gf is None or game_flagged(game) or illusion_affected(gf):
            continue
        for side in ("p1", "p2"):
            sheets = gf.sheets.get(side) or {}
            if not sheets:
                continue
            appeared = {species for (fact_side, species) in gf.mons if fact_side == side}
            for species in sorted(appeared & set(sheets)):
                units.append(
                    {"game_id": game_id, "side": side, "species": species,
                     "truth": sheets[species]}
                )
    return units


def evaluate_units(units: list[dict], manifest: dict, facts: dict, ctx: RebuildContext,
                   prior_version: str = "v2"):
    """Rebuild every unit (start-of-game set, no own sheet, leave-one-game-out
    priors, production time rule) and return per-unit rows plus aggregate counters."""
    rows = []
    l1 = Counter()  # fact precision: Layer-1
    l2 = Counter()  # fact precision: Layer-2
    l3 = Counter()  # guess accuracy: Layer-3
    source_mix = {"item": Counter(), "ability": Counter(), "moves": Counter()}
    source_acc = {"item": {}, "ability": {}, "moves": {}}
    for f in ("item", "ability", "moves"):
        for s in (SAME_GAME, DATA, SAME_PLAYER_SHEET, SAME_PLAYER_GAMES, COMMON_SET):
            source_acc[f][s] = Counter()
    share_bins = {"item": Counter(), "ability": Counter()}
    share_bins_correct = {"item": Counter(), "ability": Counter()}
    mismatches = []
    contrib_cache: dict[str, dict] = {}
    for unit in units:
        game_id, side, species = unit["game_id"], unit["side"], unit["species"]
        game = manifest[game_id]
        gf = facts[game_id]
        truth = unit["truth"]
        truth_item = _norm(truth.get("item"))
        truth_ability = _norm(truth.get("ability"))
        truth_moves = {m for m in (truth.get("moves") or []) if m}
        mon = gf.mons.get((side, species))
        account = (game.get("accounts") or {}).get(side) or gf.players.get(side, "")
        lineup = (game.get("rosters") or {}).get(side) or gf.preview.get(side) or []
        target = (game.get("split") or "", game.get("time") or 0, game_id)
        donors = ctx.eligible_donors(account, lineup, species, target)
        has_other_sheet = any(
            donor.sheet and donor.game_id != game_id for donor in donors
        )
        # Answer-key mode: rebuild as if THIS game's own sheet never existed. The
        # game's own Layer-1 facts still count (they're the game's own events); the
        # game is excluded as a donor by `eligible_donors`; the priors are
        # leave-one-game-out -- this game's contribution is subtracted.
        if game_id not in contrib_cache:
            contrib_cache[game_id] = game_prior_contribution(gf)
        prior = ctx.priors.get(species)
        if prior is not None:
            prior = subtract_prior(prior, contrib_cache[game_id].get(species, {}))
        # Teammate items (Item Clause): the other Pokémon on this side's Layer-1
        # start-of-game items (the whole side's sheet is withheld in answer-key mode).
        teammate_items = {
            other.item
            for (s, _sp), other in gf.mons.items()
            if s == side and _sp != species and other.item and not other.item_blocked
        }
        resolved = resolve_fields(
            mon, None, donors, prior,
            observed_moves=[], teammate_items=teammate_items,
            prior_version=prior_version,
        )

        l1_item = None if mon is None or mon.item_blocked else mon.item
        l1_ability = mon.ability if mon else None  # evidence AND single-legal data facts
        l1_moves = [] if mon is None or mon.move_anomaly else list(mon.moves)
        if l1_item:
            l1["item_total"] += 1
            l1["item_correct"] += l1_item == truth_item
            if l1_item != truth_item:
                mismatches.append((game_id, side, species, "item_l1", l1_item, truth_item))
        if l1_ability:
            l1["ability_total"] += 1
            l1["ability_correct"] += l1_ability == truth_ability
            if l1_ability != truth_ability:
                mismatches.append((game_id, side, species, "ability_l1", l1_ability, truth_ability))
        for move_id in l1_moves:
            l1["moves_total"] += 1
            l1["moves_correct"] += move_id in truth_moves
            if move_id not in truth_moves:
                mismatches.append((game_id, side, species, "move_l1", move_id, sorted(truth_moves)))
        # Layer-2 precision: fields a donor layer actually supplied.
        if resolved.item_source in ("same_player_sheet", "same_player_games"):
            l2["item_total"] += 1
            l2["item_correct"] += resolved.item == truth_item
            if resolved.item != truth_item:
                mismatches.append((game_id, side, species, "item_l2", resolved.item, truth_item))
        if resolved.ability_source in ("same_player_sheet", "same_player_games"):
            l2["ability_total"] += 1
            l2["ability_correct"] += resolved.ability == truth_ability
            if resolved.ability != truth_ability:
                mismatches.append((game_id, side, species, "ability_l2", resolved.ability, truth_ability))
        move_srcs = resolved.move_sources
        for move_id, src in resolved.move_source_ids.items():
            if src in ("same_player_sheet", "same_player_games"):
                l2["moves_total"] += 1
                l2["moves_correct"] += move_id in truth_moves
                if move_id not in truth_moves:
                    mismatches.append((game_id, side, species, "move_l2", move_id, sorted(truth_moves)))
            elif src == COMMON_SET:
                l3["moves_total"] += 1
                l3["moves_correct"] += move_id in truth_moves
                if move_id not in truth_moves:
                    mismatches.append((game_id, side, species, "move_l3", move_id, sorted(truth_moves)))
        if resolved.ability_source == COMMON_SET:
            l3["ability_total"] += 1
            l3["ability_correct"] += resolved.ability == truth_ability
        if resolved.item_source == COMMON_SET:
            l3["item_total"] += 1
            l3["item_correct"] += resolved.item == truth_item
        # 5a: per-source accuracy for each field (which layer's value was right).
        if resolved.item_source and resolved.item_source != UNKNOWN:
            source_acc["item"].setdefault(resolved.item_source, Counter())["total"] += 1
            source_acc["item"][resolved.item_source]["correct"] += resolved.item == truth_item
        if resolved.ability_source and resolved.ability_source != UNKNOWN:
            source_acc["ability"].setdefault(resolved.ability_source, Counter())["total"] += 1
            source_acc["ability"][resolved.ability_source]["correct"] += resolved.ability == truth_ability
        for move_id, src in resolved.move_source_ids.items():
            if src in source_acc["moves"]:
                source_acc["moves"][src]["total"] += 1
                source_acc["moves"][src]["correct"] += move_id in truth_moves
        # 5b: Layer-3 item/ability accuracy binned by the chosen value's prior share.
        for field, share, correct in (
            ("item", resolved.item_prior_share, resolved.item == truth_item),
            ("ability", resolved.ability_prior_share, resolved.ability == truth_ability),
        ):
            if (field == "item" and resolved.item_source == COMMON_SET) or (
                field == "ability" and resolved.ability_source == COMMON_SET
            ):
                b = "<0.3" if share < 0.3 else "0.3-0.5" if share < 0.5 else "0.5-0.7" if share < 0.7 else ">=0.7"
                share_bins[field][b] += 1
                share_bins_correct[field][b] += correct

        rebuilt_moves = set(resolved.moves)
        row = {
            "game_id": game_id, "side": side, "species": species,
            "split": game.get("split"), "has_other_sheet_donor": has_other_sheet,
            "item_source": resolved.item_source, "ability_source": resolved.ability_source,
            "move_sources": dict(move_srcs),
            "item_correct": resolved.item == truth_item,
            "ability_correct": resolved.ability == truth_ability,
            "move_overlap": len(rebuilt_moves & truth_moves),
            "all4_exact": rebuilt_moves == truth_moves,
            "fully_exact": (
                resolved.item == truth_item
                and resolved.ability == truth_ability
                and rebuilt_moves == truth_moves
            ),
        }
        source_mix["item"][resolved.item_source] += 1
        source_mix["ability"][resolved.ability_source] += 1
        for src, count in move_srcs.items():
            source_mix["moves"][src] += count
        source_mix["moves"][UNKNOWN] += max(0, 4 - len(rebuilt_moves))
        rows.append(row)
    return rows, l1, l2, l3, source_mix, source_acc, share_bins, share_bins_correct, mismatches


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def _unit_stats(rows: list[dict]) -> dict:
    n = len(rows)
    return {
        "units": n,
        "item_correct": _rate(sum(r["item_correct"] for r in rows), n),
        "ability_correct": _rate(sum(r["ability_correct"] for r in rows), n),
        "mean_move_overlap_over_4": _rate(sum(r["move_overlap"] for r in rows), 4 * n),
        "all4_exact": _rate(sum(r["all4_exact"] for r in rows), n),
        "fully_exact": _rate(sum(r["fully_exact"] for r in rows), n),
    }


def bootstrap_ci(rows: list[dict], resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED) -> dict:
    """95% CIs for the end-to-end metrics over game-clustered resamples."""
    by_game: dict[str, list[dict]] = {}
    for row in rows:
        by_game.setdefault(row["game_id"], []).append(row)
    games = sorted(by_game)
    rng = random.Random(seed)
    metrics = ("item_correct", "ability_correct", "mean_move_overlap_over_4", "all4_exact", "fully_exact")
    draws = {m: [] for m in metrics}
    for _ in range(resamples):
        sample = [row for gid in rng.choices(games, k=len(games)) for row in by_game[gid]]
        stats = _unit_stats(sample)
        for m in metrics:
            draws[m].append(stats[m])
    ci = {}
    for m in metrics:
        values = sorted(draws[m])
        ci[m] = {"lo": round(values[int(0.025 * resamples)], 4),
                 "hi": round(values[int(0.975 * resamples) - 1], 4)}
    return ci


def _new_bucket() -> dict:
    return {
        "records": 0,
        "before_observed": {"item": Counter(), "ability": Counter(), "moves": Counter()},
        "after_sources": {"item": Counter(), "ability": Counter(), "moves": Counter()},
        "own_fully_confirmed": Counter(),
        "bench_complete": Counter(),
        "bench_entries_before": 0,
        "bench_entries_after": 0,
    }


def coverage_pass(data_path: Path, manifest: dict, facts: dict, ctx: RebuildContext) -> dict:
    from vgc.bc.own_team_rebuild import rebuild_record

    opener = gzip.open if data_path.suffix == ".gz" else open
    kept_splits = {"train", "validation", "test"}
    buckets = {split: _new_bucket() for split in ("train", "validation", "test", "train1600")}
    anomalies = Counter()
    blocked = 0
    with opener(data_path, "rt") as src:
        for line in src:
            record = json.loads(line)
            if record.get("decision_kind") not in ("turn", "forced_switch"):
                continue
            game = manifest.get(record["replay_id"])
            if game is None or game["split"] not in kept_splits:
                continue
            side = record["player"]
            expert = (game.get("ratings") or {}).get(side)
            target = [buckets[game["split"]]]
            if isinstance(expert, (int, float)) and expert >= 1600 and game["split"] == "train":
                target.append(buckets["train1600"])
            active = record["state"]["our"]["active"]
            for bucket in target:
                bucket["records"] += 1
                bucket["bench_entries_before"] += len(record["state"]["our"].get("bench") or [])
            for mon in active:
                if not mon:
                    continue
                for bucket in target:
                    bucket["before_observed"]["item"][OBSERVED if mon.get("item") else UNKNOWN] += 1
                    bucket["before_observed"]["ability"][OBSERVED if mon.get("ability") else UNKNOWN] += 1
                    n_moves = len(mon.get("revealed_moves") or [])
                    bucket["before_observed"]["moves"][OBSERVED] += min(n_moves, 4)
                    bucket["before_observed"]["moves"][UNKNOWN] += max(0, 4 - n_moves)
            gf = facts.get(record["replay_id"])
            if gf is None:
                continue
            rebuild_record(record, gf, ctx)
            for bucket in target:
                bucket["bench_entries_after"] += len(record["state"]["our"].get("bench") or [])
                bucket["own_fully_confirmed"][record.get("own_fully_confirmed")] += 1
                bucket["bench_complete"][record["own_info"].get("bench_complete")] += 1
            for info in record["own_info"]["active"]:
                if not info:
                    continue
                for bucket in target:
                    bucket["after_sources"]["item"][info.get("item_source", UNKNOWN)] += 1
                    bucket["after_sources"]["ability"][info.get("ability_source", UNKNOWN)] += 1
                    for src, n in (info.get("move_sources") or {}).items():
                        bucket["after_sources"]["moves"][src] += n
                    filled = sum((info.get("move_sources") or {}).values())
                    bucket["after_sources"]["moves"][UNKNOWN] += max(0, 4 - filled)
    for gf in facts.values():
        for mon in gf.mons.values():
            if mon.move_anomaly:
                anomalies["move_anomaly_mons"] += 1
            if mon.item_blocked:
                blocked += 1
    return {
        "splits": {
            name: {
                "records": b["records"],
                "before_observed": {k: dict(v) for k, v in b["before_observed"].items()},
                "after_sources": {k: dict(v) for k, v in b["after_sources"].items()},
                "own_fully_confirmed": dict(b["own_fully_confirmed"]),
                "bench_complete": dict(b["bench_complete"]),
                "bench_entries_mean_before": round(
                    b["bench_entries_before"] / b["records"], 3
                ) if b["records"] else 0,
                "bench_entries_mean_after": round(
                    b["bench_entries_after"] / b["records"], 3
                ) if b["records"] else 0,
            }
            for name, b in buckets.items()
        },
        "anomalies": dict(anomalies),
        "transfer_blocked_item_facts": blocked,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--facts", type=Path, required=True, help="game_facts.jsonl(.gz) from build_corpus_facts")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--prior-versions", nargs="+", default=["v1", "v2"],
                   choices=["v1", "v2"], help="prior versions to evaluate side by side")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(args.manifest)
    opener = gzip.open if args.facts.suffix == ".gz" else open
    facts = {}
    with opener(args.facts, "rt") as handle:
        for line in handle:
            gf = game_facts_from_json(json.loads(line))
            facts[gf.replay_id] = gf
    ctx = RebuildContext(manifest, facts)

    units = answer_key_units(manifest, facts)
    keys = {}
    all_mismatches = {}
    for version in args.prior_versions:
        rows, l1, l2, l3, source_mix, source_acc, share_bins, share_bins_correct, mismatches = evaluate_units(
            units, manifest, facts, ctx, prior_version=version
        )
        all_mismatches[version] = mismatches
        by_donor = {"with_sheet_donor": [r for r in rows if r["has_other_sheet_donor"]],
                    "without_sheet_donor": [r for r in rows if not r["has_other_sheet_donor"]]}
        keys[version] = {
            "units": len(rows),
            "fact_precision": {
                "layer1": {
                    "item": f"{l1['item_correct']}/{l1['item_total']}",
                    "ability": f"{l1['ability_correct']}/{l1['ability_total']}",
                    "moves_in_sheet": f"{l1['moves_correct']}/{l1['moves_total']}",
                },
                "layer2": {
                    "item": f"{l2['item_correct']}/{l2['item_total']}",
                    "ability": f"{l2['ability_correct']}/{l2['ability_total']}",
                    "moves_in_sheet": f"{l2['moves_correct']}/{l2['moves_total']}",
                },
                "layer3_guess_accuracy": {
                    "item": f"{l3['item_correct']}/{l3['item_total']}",
                    "ability": f"{l3['ability_correct']}/{l3['ability_total']}",
                    "guessed_move_precision": f"{l3['moves_correct']}/{l3['moves_total']}",
                },
            },
            "per_source_accuracy": {
                f: {s: f"{c['correct']}/{c['total']}" for s, c in source_acc[f].items()}
                for f in ("item", "ability", "moves")
            },
            "layer3_prior_share_bins": {
                f: {
                    b: f"{share_bins_correct[f][b]}/{share_bins[f][b]}"
                    for b in ("<0.3", "0.3-0.5", "0.5-0.7", ">=0.7")
                }
                for f in ("item", "ability")
            },
            "end_to_end": _unit_stats(rows),
            "end_to_end_ci95_game_clustered": bootstrap_ci(rows),
            "by_other_sheet_donor": {k: _unit_stats(v) for k, v in by_donor.items()},
            "source_mix": {k: dict(v) for k, v in source_mix.items()},
        }
    with (args.out_dir / "answer_key_fact_mismatches.jsonl").open("w") as out:
        for rid, side, sp, field, ours, sheet in all_mismatches[
            "v2" if "v2" in all_mismatches else args.prior_versions[0]
        ]:
            out.write(json.dumps({"replay_id": rid, "side": side, "species": sp,
                                  "field": field, "ours": ours, "sheet": sheet}) + "\n")
    coverage = coverage_pass(args.data, manifest, facts, ctx)
    report = {"answer_key": keys, "coverage": coverage,
              "commands": {
                  "facts": "build_corpus_facts(manifest, raw_dir) -> game_facts.jsonl.gz",
                  "measure": " ".join(sys.argv)}}
    (args.out_dir / "measurement.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    for version in args.prior_versions:
        print(version, json.dumps(report["answer_key"][version]["end_to_end"]))
    print(f"units: {len(units)}  coverage records: "
          f"{sum(s['records'] for s in coverage['splits'].values())}")
    print(f"wrote {args.out_dir / 'measurement.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
