"""Prepare masked replay examples and fixed-split balanced sampling; no training."""
import argparse
from pathlib import Path

from vgc.bc.replay_pretraining import prepare


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--selection-summary", type=Path, required=True)
    p.add_argument("--selection-sha256", required=True, help="Pinned fingerprint of the reviewed split manifest")
    p.add_argument("--raw-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--broad-min-pre-rating", type=float, default=None,
                   help="optional pre-game rating floor for broad-phase sampling eligibility")
    p.add_argument("--broad-outcome-min-pre-rating", type=float, default=None,
                   help="optional pre-game rating floor for broad OUTCOME units "
                        "(pass -inf for 'any known finite rating')")
    p.add_argument("--rebuild-own-team", action="store_true",
                   help="Fill the acting player's own active item/ability/revealed_moves "
                        "through vgc.bc.own_team_rebuild Layers 1-3 before export "
                        "(requires a schema-8 --data file and the raw-dir replays)")
    p.add_argument("--facts-cache", type=Path, default=None,
                   help="cached analyze_log facts for --rebuild-own-team "
                        "(reused across runs; created if absent)")
    args = p.parse_args()
    ctx = None
    if args.rebuild_own_team:
        from vgc.bc.own_team_rebuild import RebuildContext, build_corpus_facts, load_manifest

        games = load_manifest(args.selection)
        facts = build_corpus_facts(games, args.raw_dir, args.facts_cache)
        ctx = RebuildContext(games, facts)
    report = prepare(args.data, args.selection, args.selection_summary, args.raw_dir, args.out,
                     expected_selection_sha256=args.selection_sha256, own_team_ctx=ctx,
                     broad_min_pre_rating=args.broad_min_pre_rating,
                     broad_outcome_min_pre_rating=args.broad_outcome_min_pre_rating)
    print(f"Prepared {sum(s['records'] for s in report['splits'].values()):,} records. No model training performed.")


if __name__ == "__main__":
    main()
