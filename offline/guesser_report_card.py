#!/usr/bin/env python
"""Guesser report card: grade hidden-set guesses against exact team files.

Truth comes from packed team files (``teams/*.packed.txt`` -- exact moves,
item, ability, Stat Points and nature). Guesses come from the normal fog-safe
prior path (``build_opponent_beliefs`` on a preview-only view: species known,
everything else hidden). Truth is used only for grading, never fed to the
guesser.

Usage:
    .venv/bin/python offline/guesser_report_card.py teams/mc_ladder_01.packed.txt
    .venv/bin/python offline/guesser_report_card.py teams/*.packed.txt --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vgc.battle_memory import BattleMemory  # noqa: E402
from vgc.guesser_eval import score_beliefs  # noqa: E402
from vgc.opponent_belief import build_opponent_beliefs  # noqa: E402
from vgc.plan_value import parse_packed_team  # noqa: E402
from vgc.stats import STAT_IDS  # noqa: E402


def _preview_battle(species_ids: list[str]):
    preview = [SimpleNamespace(species=s) for s in species_ids]
    return SimpleNamespace(
        teampreview_opponent_team=preview,
        opponent_team={},
        opponent_active_pokemon=[],
        battle_tag="report-card",
        turn=0,
    )


def _merge_reports(reports: list[dict]) -> dict:
    total = sum(r["species_scored"] for r in reports)
    merged: dict = {"species_scored": total}
    for field_name in ("item", "ability"):
        merged[field_name] = {}
        for key in ("top1", "top3", "logloss", "brier"):
            vals = [(r[field_name][key], r["species_scored"]) for r in reports if r[field_name][key] is not None]
            merged[field_name][key] = (
                sum(v * w for v, w in vals) / sum(w for _v, w in vals) if vals else None
            )
    merged["moves"] = {}
    for key in ("recall", "top4_recall"):
        vals = [(r["moves"][key], r["species_scored"]) for r in reports if r["moves"][key] is not None]
        merged["moves"][key] = sum(v * w for v, w in vals) / sum(w for _v, w in vals) if vals else None
    spread_vals = [(r["spread_in_hypotheses"], r["species_scored"]) for r in reports if r["spread_in_hypotheses"] is not None]
    merged["spread_in_hypotheses"] = (
        sum(v * w for v, w in spread_vals) / sum(w for _v, w in spread_vals) if spread_vals else None
    )
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("teams", type=Path, nargs="+", help="packed team files used as ground truth")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    # Grade each team against its OWN true sets: the same species can run
    # different sets on different teams, so a shared truth dict would grade
    # earlier teams against the last team's answers.
    per_team = []
    for path in args.teams:
        sets = parse_packed_team(path.read_text())
        species_ids = []
        truth: dict[str, dict] = {}
        for pset in sets:
            species_id = pset.species_id
            species_ids.append(species_id)
            truth[species_id] = {
                "moves": list(pset.moves),
                "item": pset.item_id or None,
                "ability": pset.ability_id or None,
                "sp": dict(zip(STAT_IDS, pset.evs or [0] * 6)),
                "nature": str(pset.nature or "serious").lower(),
            }
        battle = _preview_battle(species_ids)
        beliefs = build_opponent_beliefs(battle, BattleMemory("report-card"))
        per_team.append(score_beliefs(beliefs, truth))

    report = _merge_reports(per_team)
    report["team_files"] = [str(p) for p in args.teams]
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
