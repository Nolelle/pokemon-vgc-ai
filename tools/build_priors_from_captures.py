#!/usr/bin/env python
"""Fold unknown-species capture logs into set priors.

Reads the JSONL written by live play with ``PolicyConfig.capture_unknown_species``
on (``data/usage/unknown_species.jsonl`` by default), aggregates it with the
same once-per-battle counting rules as ``tools/build_set_priors.py``, and
either writes the supplement alone or merges it into an existing
``set_priors.json``. The live prior file is never overwritten unless it is
explicitly named as ``--out``.

Usage:
    .venv/bin/python tools/build_priors_from_captures.py
    .venv/bin/python tools/build_priors_from_captures.py --merge data/usage/set_priors.json --out data/usage/set_priors.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vgc.sets import USAGE_DIR  # noqa: E402
from vgc.unknown_species import aggregate_captures, load_capture_records, merge_priors  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--captures", type=Path, default=USAGE_DIR / "unknown_species.jsonl")
    parser.add_argument("--merge", type=Path, default=None, help="existing set_priors.json to merge into")
    parser.add_argument("--out", type=Path, default=USAGE_DIR / "set_priors_from_captures.json")
    args = parser.parse_args()

    records = load_capture_records(args.captures)
    supplement = aggregate_captures(records)
    output = supplement
    if args.merge is not None:
        base = json.loads(args.merge.read_text())
        output = merge_priors(base, supplement)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(
        f"captures={len(records)} species={len(supplement['species'])} "
        f"battles={supplement['meta']['captured_battles']} wrote {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
