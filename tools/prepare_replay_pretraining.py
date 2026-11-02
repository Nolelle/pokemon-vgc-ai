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
    args = p.parse_args()
    report = prepare(args.data, args.selection, args.selection_summary, args.raw_dir, args.out,
                     expected_selection_sha256=args.selection_sha256)
    print(f"Prepared {sum(s['records'] for s in report['splits'].values()):,} records. No model training performed.")


if __name__ == "__main__":
    main()
