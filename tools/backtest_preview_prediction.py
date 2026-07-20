#!/usr/bin/env python
"""Backtest `vgc.preview_predict.predict_preview_choice` against real teampreview
ground truth (iteration 6 plan, step 2 -- `docs/preview_prediction_plan.md`).

Ground truth: every `decision_kind == "teampreview"` record in `data/bc/decisions.jsonl`
(one per (replay, player)) -- `action["picked"]`/`action["lead_order"]` are
`vgc.replay_parse`'s best-effort DERIVATION of what that player actually brought/led
(species that actually appeared in the game, `"derived": true`), not a raw `/team XXXX`
capture (replay logs don't carry that). A record whose derived ground truth has fewer
than 4 picked or 2 led species (the game ended before every pick appeared) can't be
scored and is skipped, counted separately.

Metrics, for the predictor AND two baselines:
  - **bring-4 top-1/top-3 accuracy**: does the ground-truth 4-of-6 subset match the
    highest-probability (or one of the top-3) predicted subset, via
    `vgc.preview_predict.bring4_distribution`.
  - **lead top-1/top-3 accuracy**: same idea for the (order-independent) lead pair, via
    `lead_distribution`.

Baselines:
  (a) **uniform random legal**: every one of the 90 legal (pick, lead) combinations
      equally likely. Computed ANALYTICALLY (1/15 top-1, 3/15 top-3 for BOTH metrics --
      15 = both C(6,4) and C(6,2), and by symmetry every subset/pair is equally likely
      under uniform-over-90) rather than run through `bring4_distribution`/
      `lead_distribution`: a uniform distribution makes every candidate exactly tied,
      and that aggregation's tie-break (stable sort, so ties fall back to insertion/
      enumeration order) would silently turn "top-1" into "always guess the first
      `combinations()`-generated subset" -- a hidden, non-random baseline that happens
      to correlate with real picks whenever team-file listing order itself correlates
      with what's commonly brought. The analytical formula has no such artifact.
  (b) **their 4 most-used species by corpus usage**: species ranked by
      `vgc.sets.load_set_priors()`'s global `"appearances"` count (independent of team
      pairing) -- deterministic top-4 for bring-4, deterministic top-2 (of the full
      previewed 6, not conditioned on the bring-4 choice -- kept as an independent
      metric the same way the predictor's own bring4/lead distributions are independent
      of each other) for leads. "Top-3" for this baseline ranks ALL 15 four-subsets (or
      15 species-pairs) by summed usage and checks whether ground truth is among the
      top 3 by that ranking.

GATE (per the plan): if the predictor does not beat baseline (b) meaningfully, do NOT
integrate into `vgc.team_preview.build_team_order` -- that's a separate, later,
explicitly-approved follow-up regardless of what this backtest finds.

Usage:
    .venv/bin/python tools/backtest_preview_prediction.py
    .venv/bin/python tools/backtest_preview_prediction.py --limit 200
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vgc.config import DATA_DIR  # noqa: E402
from vgc.models import PolicyConfig  # noqa: E402
from vgc.preview_predict import bring4_distribution, lead_distribution, predict_preview_choice  # noqa: E402
from vgc.data import load_species  # noqa: E402
from vgc.sets import load_set_priors  # noqa: E402

DEFAULT_DATA_PATH = DATA_DIR.parent / "bc" / "decisions.jsonl"
_PICK_COUNT = 4
_LEADS_COUNT = 2
_TEAM_SIZE = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument(
        "--limit", type=int, default=None, help="score at most this many teampreview records"
    )
    return parser.parse_args()


def _usage_ranked_bring4(preview: list[str], priors: dict) -> list[tuple[int, ...]]:
    species_usage = priors.get("species") or {}
    usage = [species_usage.get(s, {}).get("appearances", 0) for s in preview]
    subsets = list(combinations(range(len(preview)), _PICK_COUNT))
    return sorted(subsets, key=lambda subset: (-sum(usage[i] for i in subset), subset))


def _usage_ranked_leads(preview: list[str], priors: dict) -> list[tuple[int, ...]]:
    species_usage = priors.get("species") or {}
    usage = [species_usage.get(s, {}).get("appearances", 0) for s in preview]
    pairs = list(combinations(range(len(preview)), _LEADS_COUNT))
    return sorted(pairs, key=lambda pair: (-sum(usage[i] for i in pair), pair))


class _Accumulator:
    def __init__(self) -> None:
        self.n = 0
        self.bring4_top1 = 0
        self.bring4_top3 = 0
        self.lead_top1 = 0
        self.lead_top3 = 0

    def record(
        self,
        bring4_ranked: list[tuple[int, ...]],
        lead_ranked: list[tuple[int, ...]],
        gt_pick: tuple[int, ...],
        gt_leads: tuple[int, ...],
    ) -> None:
        self.n += 1
        if bring4_ranked and bring4_ranked[0] == gt_pick:
            self.bring4_top1 += 1
        if gt_pick in bring4_ranked[:3]:
            self.bring4_top3 += 1
        if lead_ranked and lead_ranked[0] == gt_leads:
            self.lead_top1 += 1
        if gt_leads in lead_ranked[:3]:
            self.lead_top3 += 1

    def summary(self) -> dict[str, float]:
        if self.n == 0:
            return {
                "n": 0,
                "bring4_top1": 0.0,
                "bring4_top3": 0.0,
                "lead_top1": 0.0,
                "lead_top3": 0.0,
            }
        return {
            "n": self.n,
            "bring4_top1": self.bring4_top1 / self.n,
            "bring4_top3": self.bring4_top3 / self.n,
            "lead_top1": self.lead_top1 / self.n,
            "lead_top3": self.lead_top3 / self.n,
        }


def main() -> int:
    args = parse_args()
    if not args.data.exists():
        print(f"{args.data} does not exist -- run tools/parse_replays.py first", file=sys.stderr)
        return 1

    config = PolicyConfig()
    priors = load_set_priors()
    known_species = set(load_species().keys())

    predictor_acc = _Accumulator()
    baseline_b_acc = _Accumulator()

    n_seen = 0
    n_scored = 0
    n_skipped_incomplete = 0
    n_skipped_shape = 0
    n_skipped_unknown_species = 0
    start = time.time()

    with args.data.open() as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("decision_kind") != "teampreview":
                continue
            n_seen += 1
            if args.limit is not None and n_scored >= args.limit:
                break

            own_preview = (record.get("state", {}).get("our", {}) or {}).get("preview") or []
            opp_preview = (record.get("state", {}).get("opp", {}) or {}).get("preview") or []
            action = record.get("action") or {}
            picked = action.get("picked") or []
            lead_order = action.get("lead_order") or []

            if len(own_preview) != _TEAM_SIZE or len(opp_preview) != _TEAM_SIZE:
                n_skipped_shape += 1
                continue
            if len(picked) < _PICK_COUNT or len(lead_order) < _LEADS_COUNT:
                n_skipped_incomplete += 1
                continue
            try:
                gt_pick = tuple(sorted(own_preview.index(s) for s in picked[:_PICK_COUNT]))
                gt_leads = tuple(sorted(own_preview.index(s) for s in lead_order[:_LEADS_COUNT]))
            except ValueError:
                n_skipped_shape += 1
                continue
            if len(set(gt_pick)) != _PICK_COUNT or len(set(gt_leads)) != _LEADS_COUNT:
                n_skipped_shape += 1
                continue
            if any(s not in known_species for s in own_preview + opp_preview):
                # A handful of replay-parsed species ids are cosmetic-forme variants
                # (e.g. "florgesblue") that vgc.replay_parse doesn't collapse back to
                # the base species id -- data/champions/species.json only exports the
                # base forme (cosmetic color variants share identical stats/movepool,
                # so there's genuinely nothing distinct to export). Out of scope to fix
                # here; skip rather than crash the whole backtest run over it.
                n_skipped_unknown_species += 1
                continue

            candidates = predict_preview_choice(own_preview, opp_preview, config)
            if not candidates:
                n_skipped_shape += 1
                continue
            bring4_ranked = [subset for subset, _p in bring4_distribution(candidates)]
            lead_ranked = [pair for pair, _p in lead_distribution(candidates)]
            predictor_acc.record(bring4_ranked, lead_ranked, gt_pick, gt_leads)

            baseline_b_acc.record(
                _usage_ranked_bring4(own_preview, priors),
                _usage_ranked_leads(own_preview, priors),
                gt_pick,
                gt_leads,
            )

            n_scored += 1
            if n_scored % 500 == 0:
                elapsed = time.time() - start
                print(
                    f"  ...{n_scored} scored ({elapsed:.1f}s elapsed, "
                    f"{elapsed / n_scored * 1000:.1f}ms/record)",
                    file=sys.stderr,
                )

    elapsed = time.time() - start
    # Analytical, not simulated -- see module docstring's baseline (a) note for why a
    # tie-broken simulation through bring4_distribution/lead_distribution would be a
    # hidden non-random baseline instead of the genuine uniform-random one.
    baseline_a_summary = {
        "n": n_scored,
        "bring4_top1": 1.0 / 15.0,
        "bring4_top3": 3.0 / 15.0,
        "lead_top1": 1.0 / 15.0,
        "lead_top3": 3.0 / 15.0,
    }

    print("backtest_preview_prediction summary:")
    print(f"  teampreview records seen:      {n_seen}")
    print(f"  scored:                        {n_scored}")
    print(f"  skipped (incomplete ground truth): {n_skipped_incomplete}")
    print(f"  skipped (shape/lookup mismatch):   {n_skipped_shape}")
    print(f"  skipped (unrecognized species):    {n_skipped_unknown_species}")
    print(
        f"  elapsed:                        {elapsed:.1f}s "
        f"({elapsed / max(1, n_scored) * 1000:.1f}ms/record)"
    )
    print()
    for name, summary in (
        ("predictor", predictor_acc.summary()),
        ("baseline_a (uniform random legal, analytical)", baseline_a_summary),
        ("baseline_b (top-4 by corpus usage)", baseline_b_acc.summary()),
    ):
        print(f"  {name}:")
        print(f"    bring4 top-1: {summary['bring4_top1']:.4f}")
        print(f"    bring4 top-3: {summary['bring4_top3']:.4f}")
        print(f"    lead   top-1: {summary['lead_top1']:.4f}")
        print(f"    lead   top-3: {summary['lead_top3']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
