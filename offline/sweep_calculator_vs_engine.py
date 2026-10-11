"""Sweep our fast calculators against the real Showdown engine, entry by entry.

For every legal move, ability and item (Reg M-C legal lists from the mechanics catalog) this
builds controlled doubles scenarios (`tools/calc_probe.mjs`), runs them through the built Showdown
sim, and compares the engine with `vgc.damage.damage_range`, `vgc.stats`,
`vgc.accuracy.hit_probability` and the Speed/priority ordering `vgc.evaluator`/`vgc.search` use.
See `vgc.calc_sweep` for the classification (EXACT / APPROXIMATE / UNSUPPORTED-SAFE / WRONG).

    .venv/bin/python offline/sweep_calculator_vs_engine.py --profile full \
        --output data/champions/calculator_coverage.json
    .venv/bin/python offline/sweep_calculator_vs_engine.py --profile quick --families moves

Writes the per-entry coverage JSON and prints the coverage tables (entry counts and
usage-weighted share) per category. `--subset-step N` keeps ~1/N of every entry's scenarios (the
standing integration test uses it). Needs Node 22 and the built Showdown checkout; no server.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from vgc import calc_sweep as cs
from vgc import calc_sweep_cases as cc
from vgc.config import REPO_ROOT, SHOWDOWN_REPO
from vgc.data import load_moves

FAMILIES = ("moves", "abilities", "items", "mega", "order", "accuracy", "stats")


def build_cases(
    families: set[str], profile: str, subset_step: int, only: set[str] | None = None
) -> list[cs.Case]:
    """All scenarios for the requested families (optionally thinned to ~1/subset_step each)."""

    cases: list[cs.Case] = []
    moves = cs.legal_moves()
    md = load_moves()

    def keep(entry_cases: list[cs.Case]) -> None:
        if subset_step <= 1:
            cases.extend(entry_cases)
            return
        picked = [c for c in entry_cases if cs.stable_hash(c.cid) % subset_step == 0]
        if not picked and entry_cases:
            picked = [entry_cases[0]]
        cases.extend(picked)

    def wanted(entry: str) -> bool:
        return entry not in cs.UNTESTABLE and (only is None or entry in only)

    if "moves" in families:
        for move_id in moves:
            if md[move_id]["category"] != "Status" and wanted(move_id):
                keep(cs.move_variants(move_id))
    if "abilities" in families:
        for ability in cs.legal_abilities():
            if wanted(ability):
                keep(cc.holder_cases("ability", ability, profile=profile))
    if "items" in families:
        for item in cs.legal_items():
            if wanted(item):
                keep(cc.holder_cases("item", item, profile=profile))
    if "mega" in families:
        from vgc.data import load_items

        for item, data in load_items().items():
            if data.get("megaStone") and wanted(item):
                keep(cc.mega_cases(item))
    if "order" in families:
        for move_id in moves:
            if (md[move_id].get("priority", 0) != 0 or move_id == "grassyglide") and wanted(move_id):
                keep(cc.move_priority_cases(move_id))
        for ability in cs.legal_abilities():
            if wanted(ability):
                keep(cc.speed_cases("ability", ability) + cc.priority_cases("ability", ability))
        for item in cs.legal_items():
            if wanted(item):
                keep(cc.speed_cases("item", item) + cc.priority_cases("item", item))
        if wanted("baseline"):
            keep(cc.speed_cases("field", "baseline"))
    if "accuracy" in families:
        for move_id in moves:
            if wanted(move_id):
                keep(cc.move_accuracy_cases(move_id))
        for ability in cs.legal_abilities():
            if wanted(ability):
                keep(cc.holder_accuracy_cases("ability", ability))
        for item in cs.legal_items():
            if wanted(item):
                keep(cc.holder_accuracy_cases("item", item))
    if "stats" in families:
        for species in cs.legal_species():
            if wanted(species):
                keep(cc.stat_cases(species))
    return cases


COMPARATORS = {
    "damage": cs.compare_damage,
    "order": cc.compare_order,
    "accuracy": cc.compare_accuracy,
    "stats": cc.compare_stats,
}


def _request_key(case: cs.Case) -> str:
    return json.dumps(case.request(), sort_keys=True)


def evaluate(cases: list[cs.Case], workers: int) -> tuple[dict, dict[str, int]]:
    """Run ``cases`` (and their control twins, deduplicated) on the engine and compare.

    Returns (entry results, row-status counts). Memory stays bounded: cases are processed in
    chunks and only compact rows are kept.
    """

    table: dict[tuple[str, str, str], cs.EntryResult] = {}
    totals: Counter[str] = Counter()
    chunk = 12000
    for start in range(0, len(cases), chunk):
        part = cases[start : start + chunk]
        controls: dict[str, cs.Case] = {}
        control_key: list[str | None] = []
        for case in part:
            if case.control is not None:
                key = _request_key(case.control)
                controls.setdefault(key, case.control)
                control_key.append(key)
            else:
                control_key.append(None)
        keys = list(controls)
        results = cs.run_cases([part[i] for i in range(len(part))] + [controls[k] for k in keys], workers=workers)
        probes = results[: len(part)]
        control_probes = dict(zip(keys, results[len(part) :], strict=True))
        for case, probe, ckey in zip(part, probes, control_key, strict=True):
            if case.kind == "damage" and case.control is not None:
                row = cs.compare_damage(case, probe, case.control, control_probes[ckey])
            else:
                row = COMPARATORS[case.kind](case, probe)
            key = (case.kind, case.entry_kind, case.entry)
            if key not in table:
                table[key] = cs.EntryResult(case.kind, case.entry_kind, case.entry)
            table[key].add(row)
            totals[row["status"]] += 1
        del results, probes, control_probes
    for entry in cs.UNTESTABLE:
        for kind in ("damage", "order", "accuracy"):
            entry_kind = "ability"
            result = cs.EntryResult(kind, entry_kind, entry, n=1, n_unexercised=1)
            table[(kind, entry_kind, entry)] = result
    return table, dict(totals)


def showdown_commit() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(SHOWDOWN_REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def build_report(table: dict, profile: str, totals: dict[str, int], elapsed: float) -> dict[str, Any]:
    weights = cs.usage_weights()
    entries: dict[str, dict[str, dict[str, Any]]] = {}
    for (kind, entry_kind, entry), result in sorted(table.items()):
        record = {
            "status": result.status,
            "cases": result.n,
            "unexercised": result.n_unexercised,
            "mismatches": result.n_mismatch,
            "usage": round(weights.get(entry_kind, {}).get(entry, 0.0), 6),
        }
        if result.n_mismatch:
            by_variant: Counter[str] = Counter()
            worst = 0.0
            for row in result.mismatches:
                by_variant[row["variant"].split("|")[-1] if "|" in row["variant"] else row["variant"]] += 1
                worst = max(worst, float(row.get("rel_err", 0.0)))
            record["max_rel_err"] = round(worst, 4)
            record["mismatch_variants"] = dict(by_variant.most_common(6))
            record["example"] = {
                key: result.mismatches[0].get(key)
                for key in ("variant", "reason", "ours_range", "eng_range", "ours", "engine", "detail")
                if result.mismatches[0].get(key) is not None
            }
        entries.setdefault(kind, {}).setdefault(entry_kind, {})[entry] = record
    coverage = cs.usage_coverage(table.values(), {})
    return {
        "meta": {
            "profile": profile,
            "showdown_commit": showdown_commit(),
            "approx_bound": cs.APPROX_BOUND,
            "case_rows": totals,
            "seconds": round(elapsed, 1),
            "usage_sources": {
                "move": "data/usage/set_priors.json (M-C corpus appearances)",
                "ability": "Smogon M-C chaos stats 2026-09",
                "item": "Smogon M-C chaos stats 2026-09",
            },
        },
        "coverage": coverage,
        "entries": entries,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    """Human-readable list of everything that is not EXACT, with the reason."""

    lines = [
        "# Fast-calculator coverage vs the Showdown engine",
        "",
        "Generated by `offline/sweep_calculator_vs_engine.py` -- do not edit by hand.",
        f"Profile `{report['meta']['profile']}`, Showdown `{report['meta']['showdown_commit'][:9]}`.",
        "EXACT entries are not listed; see `data/champions/calculator_coverage.json` for them.",
        "",
    ]
    for category, kinds in report["entries"].items():
        for entry_kind, entries in kinds.items():
            rows = [(r["usage"], name, r) for name, r in entries.items() if r["status"] != "EXACT"]
            if not rows:
                continue
            lines += [f"## {category} / {entry_kind} ({len(rows)} not exact)", ""]
            lines += ["| entry | status | usage | reason |", "|---|---|---|---|"]
            for usage, name, record in sorted(rows, reverse=True):
                reason = cs.KNOWN_REASONS.get(name) or cs.UNTESTABLE.get(name)
                if reason is None:
                    example = record.get("example") or {}
                    reason = f"{example.get('reason', record['status'])} e.g. {example.get('variant', '')}"
                lines.append(f"| {name} | {record['status']} | {usage:.4f} | {reason} |")
            lines.append("")
    path.write_text("\n".join(lines))


def print_tables(report: dict[str, Any]) -> None:
    statuses = ("EXACT", "APPROXIMATE", "UNSUPPORTED-SAFE", "WRONG", "NOT_EXERCISED")
    print(f"{'category':10s} {'kind':8s} {'n':>5s}  " + "  ".join(f"{s[:10]:>16s}" for s in statuses))
    for category, kinds in report["coverage"].items():
        for kind, data in kinds.items():
            cells = []
            for status in statuses:
                count = data["counts"].get(status, 0)
                share = data["usage_weighted"].get(status, 0.0)
                cells.append(f"{count:5d} ({share * 100:5.1f}%)")
            print(f"{category:10s} {kind:8s} {data['entries']:5d}  " + "  ".join(f"{c:>16s}" for c in cells))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", choices=("quick", "full"), default="full")
    parser.add_argument("--families", default=",".join(FAMILIES))
    parser.add_argument("--only", default="", help="comma-separated entry ids to restrict to")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--subset-step", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None, help="write the not-exact list here")
    parser.add_argument("--worst", type=int, default=0, help="print the N worst WRONG entries per category")
    args = parser.parse_args()

    families = {f for f in args.families.split(",") if f}
    only = {e for e in args.only.split(",") if e} or None
    started = time.time()
    cases = build_cases(families, args.profile, args.subset_step, only)
    print(f"{len(cases)} scenarios built in {time.time() - started:.1f}s", flush=True)
    table, totals = evaluate(cases, args.workers)
    elapsed = time.time() - started
    print(f"engine+compare done in {elapsed:.1f}s: {totals}", flush=True)
    report = build_report(table, args.profile, totals, elapsed)
    print_tables(report)
    if args.worst:
        for kind, kinds in report["entries"].items():
            for entry_kind, entries in kinds.items():
                wrong = [(v["usage"], k, v) for k, v in entries.items() if v["status"] == "WRONG"]
                wrong.sort(reverse=True)
                print(f"\n{kind}/{entry_kind}: top WRONG by usage")
                for usage, name, record in wrong[: args.worst]:
                    print(
                        f"  {name:24s} usage={usage:.4f} bad={record['mismatches']}/{record['cases']}"
                        f" err={record.get('max_rel_err')} {record.get('mismatch_variants')} {record.get('example')}"
                    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as file:
            json.dump(report, file, indent=1, sort_keys=True)
        print(f"wrote {args.output}")
    if args.markdown:
        write_markdown(report, args.markdown)
        print(f"wrote {args.markdown}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT / "src"))
    raise SystemExit(main())
