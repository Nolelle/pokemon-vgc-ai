#!/usr/bin/env python
"""Why do speed-control / weather / terrain setup moves lose the exact judge's ranking?

Plays direct `ExactSearchPlayer` mirror games (narrow width, field control ON), captures
every exact ranking, and for each decision with a SEARCHED setup candidate compares the
chosen (top) candidate with the best setup candidate, then re-ranks offline under:
(a) search_myopic_weight 0, (b) worst-case weight 0, (c) both, (d) field_delta counted twice.

    PYTHONPATH=$PWD/src .venv/bin/python offline/diagnose_setup_ranking.py --games 6

Caveat: `combine_belief_rankings` copies breakdown fields other than exchange_value /
myopic_score from the modal belief, so response_* and field_delta come from that belief.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from functools import partial
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from poke_env.battle.move import Move  # noqa: E402

import vgc.rl.exact_player as exact_player  # noqa: E402
from vgc.config import FORMAT_ID  # noqa: E402
from vgc.models import PolicyConfig  # noqa: E402
from vgc.rl.agents import make_direct_agent  # noqa: E402
from vgc.rl.env import SimWorker  # noqa: E402
from vgc.rl.match import run_series  # noqa: E402

MANIFEST = REPO_ROOT / "data" / "selfplay" / "mc_sheet_pool_v2" / "train_manifest.json"
SETUP_MOVES = {
    "tailwind", "trickroom", "icywind", "electroweb", "thunderwave", "sunnyday", "raindance",
    "sandstorm", "snowscape", "electricterrain", "grassyterrain", "psychicterrain",
    "mistyterrain",
}
TEAM_PATTERN = re.compile(
    "trickroom|tailwind|drought|drizzle|sandstream|snowwarning|surge|sunnyday|raindance"
)
RECORDS: list[dict] = []


def setup_moves_in(order) -> list[str]:
    found = []
    for single in (order.first_order, order.second_order):
        target = getattr(single, "order", None)
        if isinstance(target, Move) and target.id in SETUP_MOVES:
            found.append(target.id)
    return found


def cand_row(entry, config: PolicyConfig) -> dict:
    b = entry.breakdown
    values = b.get("response_values") or []
    weights = b.get("response_weights") or []
    orders = b.get("response_orders") or []
    worst_i = min(range(len(values)), key=values.__getitem__) if values else None
    return {
        "order": entry.order.message,
        "desc": b.get("desc"),
        "setup": setup_moves_in(entry.order),
        "final": entry.score,
        "myopic": b.get("myopic_score"),
        "exchange": b.get("exchange_value"),
        "field_delta": b.get("field_delta", 0.0),
        "values": values,
        "weights": weights,
        "worst": min(values) if values else None,
        "expectation": sum(w * v for w, v in zip(weights, values)) if values else None,
        "worst_response": orders[worst_i] if worst_i is not None else None,
        "worst_response_weight": weights[worst_i] if worst_i is not None else None,
    }


def install_capture(config: PolicyConfig) -> None:
    original = exact_player.public_information_exact_search

    def wrapped(*args, **kwargs):
        scored = original(*args, **kwargs)
        searched = [e for e in scored if e.breakdown.get("searched")]
        if any(setup_moves_in(e.order) for e in searched):
            RECORDS.append(
                {"top": cand_row(scored[0], config), "searched": [cand_row(e, config) for e in searched]}
            )
        else:
            RECORDS.append({"top": None, "n": len(searched)})
        return scored

    exact_player.public_information_exact_search = wrapped


def quartiles(xs: list[float]) -> dict:
    if not xs:
        return {}
    q = statistics.quantiles(xs, n=4) if len(xs) >= 2 else [xs[0]] * 3
    return {"q1": q[0], "median": q[1], "q3": q[2], "mean": statistics.fmean(xs), "n": len(xs)}


def analyse(config: PolicyConfig) -> dict:
    wm, wp, w = config.search_myopic_weight, config.search_position_weight, config.search_worst_case_weight
    decs = [r for r in RECORDS if r["top"] is not None]
    n_total = len(RECORDS)

    def score(c, *, myo=wm, worst_w=w, field_mult=1.0):
        if not c["values"]:
            return c["final"]
        x = worst_w * c["worst"] + (1 - worst_w) * c["expectation"] + (field_mult - 1.0) * c["field_delta"]
        return myo * c["myopic"] + wp * x

    variants = {
        "baseline": {},
        "a_myopic0": {"myo": 0.0},
        "b_worst0": {"worst_w": 0.0},
        "c_both": {"myo": 0.0, "worst_w": 0.0},
        "d_field_x2": {"field_mult": 2.0},
        "e_field_x2_myopic0_worst0": {"field_mult": 2.0, "myo": 0.0, "worst_w": 0.0},
    }
    flips = Counter()
    gaps, g_myo, g_exch, g_field, g_exch_nofield = [], [], [], [], []
    chosen_setup = 0
    worst_resp = Counter()
    for r in decs:
        cands = r["searched"]
        top = r["top"]
        setups = [c for c in cands if c["setup"]]
        best_setup = max(setups, key=lambda c: c["final"])
        if top["setup"]:
            chosen_setup += 1
            continue
        gap = top["final"] - best_setup["final"]
        gaps.append(gap)
        dm = wm * (top["myopic"] - best_setup["myopic"])
        dx = wp * (top["exchange"] - best_setup["exchange"])
        df = wp * (top["field_delta"] - best_setup["field_delta"])
        g_myo.append(dm)
        g_exch.append(dx)
        g_field.append(df)
        g_exch_nofield.append(dx - df)
        if best_setup["worst_response"]:
            worst_resp[best_setup["worst_response"]] += 1
        for name, kw in variants.items():
            best_other = max(cands, key=lambda c: score(c, **kw))
            best_s = max(setups, key=lambda c: score(c, **kw))
            if score(best_s, **kw) >= score(best_other, **kw) and best_s is best_other:
                flips[name] += 1
    n_eval = len(gaps)
    return {
        "decisions_total": n_total,
        "decisions_with_searched_setup": len(decs),
        "setup_chosen": chosen_setup,
        "decisions_setup_not_chosen": n_eval,
        "setup_top_rate_by_variant (of decisions with a searched setup candidate, incl. already chosen)": {
            k: (flips[k] + chosen_setup) / max(1, len(decs)) if k == "baseline" else flips[k] / max(1, n_eval)
            for k in variants
        },
        "flip_counts_among_not_chosen": dict(flips),
        "gap_top_minus_best_setup": quartiles(gaps),
        "gap_part_myopic": quartiles(g_myo),
        "gap_part_exchange": quartiles(g_exch),
        "gap_part_field_inside_exchange": quartiles(g_field),
        "gap_part_exchange_excl_field": quartiles(g_exch_nofield),
        "worst_response_vs_best_setup": worst_resp.most_common(15),
        "setup_move_counts_in_best_setup": Counter(
            m for r in decs for m in max((c for c in r["searched"] if c["setup"]), key=lambda c: c["final"])["setup"]
        ).most_common(),
        "example_decisions": [
            {"top": r["top"], "best_setup": max((c for c in r["searched"] if c["setup"]), key=lambda c: c["final"])}
            for r in decs[:5]
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--games", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20261006)
    ap.add_argument("--team-offset", type=int, default=0)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--output", type=Path, default=REPO_ROOT / "runs/eval/setup_ranking_diagnostic.json")
    args = ap.parse_args()

    config = PolicyConfig(
        format_id=FORMAT_ID,
        search_our_candidates=4,
        search_opp_candidates=4,
        exact_search_future_samples=1,
        exact_search_field_control=True,
        exact_search_speed_order_weight=24.0,
        exact_search_field_fit_weight=0.75,
    )
    pool = json.loads(args.manifest.read_text())
    teams = []
    for entry in pool:
        text = (args.manifest.parent / entry["file"]).read_text().strip()
        if TEAM_PATTERN.search(text.lower().replace(" ", "")):
            teams.append(text)
    print(f"{len(teams)} setup-capable teams of {len(pool)}")
    install_capture(config)
    outcomes = []
    with SimWorker() as worker:
        for game in range(args.games):
            ta = teams[(args.team_offset + 2 * game) % len(teams)]
            tb = teams[(args.team_offset + 2 * game + 1) % len(teams)]
            built = []

            def factory(team):
                agent = make_direct_agent("vgc_exact", team, config=config)
                built.append(agent)
                return agent

            try:
                outcomes.extend(
                    run_series(
                        worker,
                        {"a": partial(factory, ta), "b": partial(factory, tb)},
                        {"a": ta, "b": tb},
                        1,
                        seed=args.seed + game,
                    )
                )
            finally:
                for agent in built:
                    close = getattr(agent.player, "close_public_mirror", None)
                    if close:
                        close()
            report = analyse(config)
            print(f"game {game + 1}: decisions={report['decisions_total']} "
                  f"with_setup={report['decisions_with_searched_setup']}", flush=True)
    report = analyse(config)
    report["games"] = len(outcomes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: v for k, v in report.items() if k != "example_decisions"}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
