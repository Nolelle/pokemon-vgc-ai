"""Summarise the @pkmn/client shadow trial: three arms, triage of pkmn's unexplained groups,
and which poke-env defects pkmn does not have.

Inputs are the JSON files `audit_battle_parsing.py --reader both` writes
(`runs/eval/pkmn_shadow_trial_{repaired,raw}_<seed>.json`, plus the scenario files):

    .venv/bin/python offline/pkmn_shadow_report.py --dir runs/eval --seeds 99 20261009 20261010

TRIAGE categories (each unexplained pkmn episode gets exactly one; rules are applied to the
episode's SIGNATURE -- the kinds of protocol line that touched the Pokemon on the step the
mismatch first appeared -- and the battle context, in `TRIAGE_RULES` order, first match wins):

    a  adapter bug (should be empty: every adapter mapping is documented in pkmn_shadow.py)
    b  genuine @pkmn/client defect or gap (a repro for most is in pkmn_shadow_repros.py)
    c  hidden information / silent in the protocol (reader-independent)
    d  the comparator's notion of "revealed" is poke-env-shaped (the client reads a legitimate
       reveal channel the independent protocol parser does not model)
    u  not individually verified (episode counts only)
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# (id, category, description, group regex, predicate(sig_tags, flags) -> bool)
Rule = tuple[str, str, str, str, Any]


def _has(sig: set[str], *needles: str) -> bool:
    return any(n in tag for n in needles for tag in sig)


TRIAGE_RULES: list[Rule] = [
    ("B3", "b", "plain -heal (no [from]) zeroes toxic/sleep counters", r"status_counter/mismatch/(slp|tox)",
     lambda sig, fl: "-heal" in sig),
    ("B4", "b", "-heal [from] item: assigns the item to the healed Pokemon (Bug Bite/Fling/Pluck)",
     r"item/foe_(wrong_known|shown_but_unknown|marked_consumed|known_but_unrevealed)/.*",
     lambda sig, fl: _has(sig, "-heal [from] item:") and _has(sig, "Bug Bite", "Fling", "Pluck", "-enditem")),
    ("B5", "b", "Symbiosis item pass not tracked", r"item/.*|ability/.*",
     lambda sig, fl: _has(sig, "ability: Symbiosis")),
    ("B6", "b", "Gastro Acid records no volatile and blanks `ability` (-endability only)",
     r"volatile/missing/gastroacid|ability/foe_revealed_but_unknown/.*",
     lambda sig, fl: _has(sig, "Gastro Acid", "-endability") or not sig or sig == {"move"}),
    ("B8", "b", "ability named by `[from] ability:` on -setboost / -curestatus is ignored (Anger Point, "
     "Natural Cure)", r"ability/foe_revealed_but_unknown/.*",
     lambda sig, fl: _has(sig, "-setboost [from] ability", "-curestatus:slp [from] ability", "-curestatus")),
    ("B9", "b", "own/known extender items (Light Clay, Damp Rock, ...) not applied to durations",
     r"duration/mismatch/.*", lambda sig, fl: True),
    ("B7", "b", "Transform copies the target's species types, not its current types",
     r"types/mismatch/.*", lambda sig, fl: _has(sig, "-transform") and "I" not in fl),
    ("B10", "b", "`yawn` volatile dropped on any later status (engine keeps the inert volatile)",
     r"volatile/missing/yawn", lambda sig, fl: True),
    ("B11", "b", "Psych Up does not clear Dragon Cheer (the client's copied-volatile list is older)",
     r"volatile/stale/dragoncheer", lambda sig, fl: True),
    ("C1", "c", "`trapped` volatile ends silently when the trapper leaves", r"volatile/stale/trapped",
     lambda sig, fl: True),
    ("D1", "d", "move revealed through a cant-line / Focus Punch / Forewarn / own post-Transform "
     "request, which the independent protocol parser does not count",
     r"foe_moves/(known_but_unrevealed|not_in_truth)/.*",
     lambda sig, fl: _has(sig, "cant:", "Focus Punch", "Forewarn", "Transform", "-transform") or "T" in fl),
    ("D2", "d", "Poltergeist names the target's item in the protocol", r"item/foe_known_but_unrevealed/.*",
     lambda sig, fl: _has(sig, "Poltergeist")),
    ("D3", "d", "ability named by Entrainment / an ability-block cant-line",
     r"ability/foe_known_but_unrevealed/.*", lambda sig, fl: _has(sig, "Entrainment", "cant:ability:", "-ability")),
    ("D4", "d", "sleep counter off by one on both readers (Early Bird / sleep-turn accounting in the truth "
     "formula)", r"status_counter/mismatch/slp", lambda sig, fl: "cant:slp" in sig),
    ("B2", "b", "temporary ability (Worry Seed/Simple Beam/Skill Swap/...) recorded as the base ability",
     r"ability/foe_wrong/.*", lambda sig, fl: "I" not in fl),
    ("I1", "d", "Illusion: the comparator's disguise model is shaped around poke-env's ident-keyed team",
     r".*", lambda sig, fl: "I" in fl),
    ("T1", "d", "Transform/Imposter in the battle (copied moves/types/boosts come from two Pokemon)",
     r".*", lambda sig, fl: "T" in fl),
    ("B1", "b", "a returning Pokemon is rebuilt blank (lastPokemon rule), dropping moves/item/ability",
     r"(foe_moves/shown_but_unknown|item/foe_(gone_but_unknown|shown_but_unknown|gone_but_known)|"
     r"ability/foe_revealed_but_unknown|hp/foe_percent|active_flag|status/mismatch|boosts/mismatch)/?.*",
     lambda sig, fl: "switch" in sig or "D" in fl or _has(sig, "drag", "replace")),
]


def classify(group: str, sig_key: str) -> tuple[str, str, str]:
    """(rule id, category, description) for one unexplained episode signature."""
    sig_text, _, flags = sig_key.rpartition("#")
    sig = set(sig_text.split(";")) if sig_text else set()
    bare = re.sub(r"^\[[a-z0-9_]+\] ", "", group)
    for rid, cat, desc, pattern, pred in TRIAGE_RULES:
        if re.fullmatch(pattern, bare) and pred(sig, flags):
            return rid, cat, desc
    return "U", "u", "not individually verified"


def load(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.exists() else None


def arm(result: dict[str, Any] | None, reader: str) -> dict[str, Any] | None:
    return (result or {}).get("readers", {}).get(reader)


def fmt_row(label: str, r: dict[str, Any] | None) -> str:
    if r is None:
        return f"  {label:34s} (missing)"
    s = r["summary"]
    return (
        f"  {label:34s} compares={s['compares']:7d} episodes={s['episodes']:6d} "
        f"unexplained: {s['unexplained_groups']:4d} groups / {s['unexplained_episodes']:5d} episodes   "
        f"exempt: {s['exempt_groups']:4d} groups / {s['exempt_episodes']:5d} episodes"
    )


def family(group: str) -> str:
    bare = re.sub(r"^\[[a-z0-9_]+\] ", "", group)
    return "/".join(bare.split("/")[:2])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dir", type=Path, default=Path("runs/eval"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[99, 20261009, 20261010])
    parser.add_argument("--residual", type=int, default=25, help="how many untriaged groups to list")
    parser.add_argument("--fixes", type=Path, default=None, help="directory of per-repair sweep JSONs")
    args = parser.parse_args()
    if args.fixes:
        print_fixes(args.fixes)
        return 0

    print("== THREE ARMS (2400 battles per seed; same seeds; pkmn graded on both trajectory sets)")
    pk_eps: Counter = Counter()
    raw_po: Counter = Counter()
    rep_po: Counter = Counter()
    raw_pk: Counter = Counter()
    for seed in args.seeds:
        rep = load(args.dir / f"pkmn_shadow_trial_repaired_{seed}.json")
        raw = load(args.dir / f"pkmn_shadow_trial_raw_{seed}.json")
        print(f"\nseed {seed}")
        print(fmt_row("raw poke-env (--disable-fixes all)", arm(raw, "pokeenv")))
        print(fmt_row("repaired poke-env", arm(rep, "pokeenv")))
        print(fmt_row("@pkmn/client (repaired-driven games)", arm(rep, "pkmn")))
        print(fmt_row("@pkmn/client (raw-driven games)", arm(raw, "pkmn")))
        for res, reader, bucket in (
            (raw, "pokeenv", raw_po), (rep, "pokeenv", rep_po), (raw, "pkmn", raw_pk), (rep, "pkmn", pk_eps),
        ):  # fmt: skip
            r = arm(res, reader)
            if r:
                for g, n in r["episodes"].items():
                    if r["classification"][g] == "unexplained":
                        bucket[family(g)] += n

    # ---- triage of pkmn's unexplained episodes (repaired-driven games, all seeds) ----
    by_rule: Counter = Counter()
    by_rule_groups: defaultdict[str, set] = defaultdict(set)
    desc: dict[str, tuple[str, str]] = {}
    residual: Counter = Counter()
    total = 0
    groups_seen: set[str] = set()
    for seed in args.seeds:
        r = arm(load(args.dir / f"pkmn_shadow_trial_repaired_{seed}.json"), "pkmn")
        if not r:
            continue
        for g, sigs in r["ep_sigs"].items():
            if r["classification"][g] != "unexplained":
                continue
            groups_seen.add(g)
            for sig_key, n in sigs.items():
                rid, cat, d = classify(g, sig_key)
                by_rule[rid] += n
                by_rule_groups[rid].add(g)
                desc[rid] = (cat, d)
                total += n
                if rid == "U":
                    residual[g] += n
    print(f"\n== TRIAGE of pkmn unexplained episodes (repaired-driven games, {len(args.seeds)} seeds): "
          f"{total} episodes in {len(groups_seen)} groups")
    cats: Counter = Counter()
    for rid, n in by_rule.most_common():
        cat, d = desc[rid]
        cats[cat] += n
        print(f"  {rid:3s} [{cat}] {n:5d} episodes {len(by_rule_groups[rid]):4d} groups  {d}")
    print("  by category:", dict(cats))
    if residual:
        print("\n  untriaged (rule U) top groups:")
        for g, n in residual.most_common(args.residual):
            print(f"    {n:4d} {g}")

    # ---- defects raw poke-env has that pkmn does not ----
    print("\n== UNEXPLAINED EPISODES BY FAMILY, summed over seeds (raw-driven games for the first two "
          "columns: same trajectories)")
    print(f"  {'family':44s} {'raw poke-env':>12s} {'pkmn(raw)':>10s} {'repaired':>9s} {'pkmn(rep)':>10s}")
    fams = set(raw_po) | set(rep_po) | set(raw_pk) | set(pk_eps)
    for fam in sorted(fams, key=lambda f: -(raw_po[f] + pk_eps[f])):
        print(f"  {fam:44s} {raw_po[fam]:12d} {raw_pk[fam]:10d} {rep_po[fam]:9d} {pk_eps[fam]:10d}")

    # ---- named poke-env defects: exact groups ----
    print("\n== NAMED poke-env DEFECTS (raw-driven games, all seeds): poke-env episodes vs pkmn episodes")
    named = {
        "-copyboost reversed (Psych Up/Costar) / boost drift": r"boosts/mismatch/.*",
        "Champions Mega forme/ability not recognised": r"(species/mismatch/.*|ability/(own|foe_wrong)/.*)",
        "typechange/typeadd marker volatiles": r"volatile/stale/(typechange|typeadd|futuresight)",
        "single-turn effects & activation markers never end": r"volatile/stale/.*",
        "layered hazards (Spikes/Toxic Spikes layers)": r"side_condition/layers/.*",
        "own preview-only Pokemon left in roster/active": r"roster/unknown_mon/own",
        "foe item reveals missed ([from] item:, Mega stones)": r"item/foe_shown_but_unknown/.*",
        "foe ability reveals missed ([from] ability:)": r"ability/foe_revealed_but_unknown/.*",
        "foe consumed item read as unknown": r"item/foe_gone_but_unknown",
        "sleep/toxic counters": r"status_counter/mismatch/.*",
        "partial trap / perish song volatile names": r"volatile/(missing|stale)/(partiallytrapped|perishsong|perish.*|wrap|bind|firespin)",
        "types after type-change effects": r"types/mismatch/.*",
    }
    agg: dict[str, list[int]] = {k: [0, 0] for k in named}
    for seed in args.seeds:
        raw = load(args.dir / f"pkmn_shadow_trial_raw_{seed}.json")
        for idx, reader in enumerate(("pokeenv", "pkmn")):
            r = arm(raw, reader)
            if not r:
                continue
            for g, n in r["episodes"].items():
                if r["classification"][g] != "unexplained":
                    continue
                bare = re.sub(r"^\[[a-z0-9_]+\] ", "", g)
                for label, pattern in named.items():
                    if re.fullmatch(pattern, bare):
                        agg[label][idx] += n
                        break
    for label, (po, pk) in agg.items():
        print(f"  {label:58s} poke-env {po:6d}   pkmn {pk:5d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ---------------------------------------------------------------------------------------
# Per-repair view: which of poke-env's `vgc.poke_env_compat.ALL_FIXES` would pkmn make moot?
#
#     for fix in ALL_FIXES: audit_battle_parsing.py --disable-fixes <fix> --reader both ...
#     .venv/bin/python offline/pkmn_shadow_report.py --fixes runs/eval/fixes
#
# A repair's footprint = the poke-env mismatch groups whose episode count rises when ONLY that
# repair is off (vs the all-repairs-on baseline run on the same seed). The same groups, graded
# on the pkmn client in the SAME run, say whether pkmn shares the defect. A repair the audit
# never exercises (footprint < MIN_FOOTPRINT episodes) is reported as "not exercised", not
# guessed.
# ---------------------------------------------------------------------------------------

MIN_FOOTPRINT = 3


def _bare(group: str) -> str:
    return re.sub(r"^\[[a-z0-9_]+\] ", "", group)


def _episodes(result: dict[str, Any], reader: str) -> Counter:
    r = arm(result, reader)
    out: Counter = Counter()
    if r:
        for g, n in r["episodes"].items():
            out[_bare(g)] += n
    return out


def fixes_table(directory: Path) -> list[dict[str, Any]]:
    base = load(directory / "_baseline.json")
    if base is None:
        raise SystemExit(f"no {directory}/_baseline.json")
    base_po = _episodes(base, "pokeenv")
    rows = []
    for path in sorted(directory.glob("*.json")):
        if path.name.startswith("_"):
            continue
        res = load(path)
        po = _episodes(res, "pokeenv")
        pk = _episodes(res, "pkmn")
        delta = {g: n - base_po.get(g, 0) for g, n in po.items() if n - base_po.get(g, 0) > 0}
        footprint = sum(delta.values())
        pk_same = sum(min(pk.get(g, 0), n) for g, n in delta.items())
        pk_family = sum(
            n for g, n in pk.items() if "/".join(g.split("/")[:2]) in {"/".join(d.split("/")[:2]) for d in delta}
        )
        if footprint < MIN_FOOTPRINT:
            verdict = "not exercised"
        elif pk_same <= 0.10 * footprint:
            verdict = "pkmn avoids"
        elif pk_same >= 0.30 * footprint:
            verdict = "pkmn shares"
        else:
            verdict = "partly"
        rows.append(
            {
                "fix": path.stem, "footprint": footprint, "pkmn_same_groups": pk_same,
                "pkmn_family": pk_family, "verdict": verdict,
                "top": sorted(delta.items(), key=lambda kv: -kv[1])[:3],
            }
        )  # fmt: skip
    return rows


def print_fixes(directory: Path) -> None:
    rows = fixes_table(directory)
    print(f"== PER-REPAIR FOOTPRINT ({len(rows)} repairs, one switched off at a time; poke-env episodes "
          "added vs all-on, and pkmn episodes in those same groups, same battles)")
    for r in sorted(rows, key=lambda r: (r["verdict"], -r["footprint"])):
        top = ", ".join(f"{g}:{n}" for g, n in r["top"])
        print(f"  {r['verdict']:13s} {r['fix']:34s} footprint={r['footprint']:5d} "
              f"pkmn_same={r['pkmn_same_groups']:4d}  {top}")
    counts = Counter(r["verdict"] for r in rows)
    print("  totals:", dict(counts))
