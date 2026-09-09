#!/usr/bin/env python
"""Parse a downloaded replay corpus (`tools/download_replays.py`'s output,
`data/replays/<format>/<id>.json`) into a behavior-cloning decision dataset
(`data/bc/decisions.jsonl`, one JSON decision record per line -- see
`vgc.replay_parse`'s module docstring for the record schema and the full parsing
design, including a critical finding about how rare `|showteam|` reveals actually are
in this ladder format).

Usage:
    .venv/bin/python tools/parse_replays.py
    .venv/bin/python tools/parse_replays.py --replays-dir data/replays/other --limit 50

The default `--replays-dir` is `data/replays/<FORMAT_ID>/` (currently the small M-C
tree). For the historical warm-start corpus, pass
`--replays-dir data/replays/gen9championsvgc2026regmb`. Schema 5 labels are
format-agnostic; record `format_id` if you mix trees. See
`docs/replay_label_contract.md`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vgc.config import DATA_DIR, FORMAT_ID  # noqa: E402
from vgc.replay_parse import parse_replay  # noqa: E402

DEFAULT_REPLAYS_DIR = REPO_ROOT / "data" / "replays" / FORMAT_ID
DEFAULT_OUT_PATH = DATA_DIR.parent / "bc" / "decisions.jsonl"

# Below this parse-success rate, the CLI prints the top failure reasons prominently
# instead of just a one-line summary -- see this module's docstring/CLAUDE.md's
# validation contract for why 90% is the bar.
SUCCESS_RATE_INVESTIGATE_THRESHOLD = 0.90


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--replays-dir",
        type=Path,
        default=DEFAULT_REPLAYS_DIR,
        help="directory of downloaded <id>.json replays (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_PATH,
        help="output JSONL path for decision records (default: %(default)s)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="parse at most this many replay files (default: all)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.replays_dir.exists():
        print(f"{args.replays_dir} does not exist", file=sys.stderr)
        return 1

    files = sorted(args.replays_dir.glob("*.json"))
    if args.limit is not None:
        files = files[: args.limit]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = args.out.with_suffix(args.out.suffix + ".tmp")

    replays_seen = 0
    replays_parsed = 0
    replays_failed = 0
    fail_reasons: Counter = Counter()
    records_by_kind: Counter = Counter()
    skipped_by_reason: Counter = Counter()
    showteam_player_slots = 0
    replays_with_resolved_winner = 0
    outcomes: Counter = Counter()

    with tmp_path.open("w") as out_file:
        for path in files:
            replays_seen += 1
            try:
                payload = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError) as exc:
                replays_failed += 1
                fail_reasons[f"unreadable_file: {type(exc).__name__}"] += 1
                continue

            replay_id = payload.get("id", path.stem)
            rating = payload.get("rating")
            log = payload.get("log")
            if not isinstance(log, str) or not log:
                replays_failed += 1
                fail_reasons["missing_log_field"] += 1
                continue

            result = parse_replay(replay_id, rating, log)
            if not result.ok:
                replays_failed += 1
                fail_reasons[result.fail_reason or "unknown"] += 1
                continue

            replays_parsed += 1
            showteam_player_slots += len(result.showteam_players)
            if result.winner is not None:
                replays_with_resolved_winner += 1
            for key, count in result.skipped.items():
                skipped_by_reason[key] += count
            for record in result.records:
                records_by_kind[record["decision_kind"]] += 1
                outcomes[record.get("outcome", "unresolved")] += 1
                out_file.write(json.dumps(record, sort_keys=True) + "\n")

    tmp_path.replace(args.out)

    total_decision_records = sum(records_by_kind.values())
    success_rate = replays_parsed / replays_seen if replays_seen else 0.0
    showteam_rate = showteam_player_slots / (2 * replays_seen) if replays_seen else 0.0

    print("parse_replays summary:")
    print(f"  replays seen:      {replays_seen}")
    print(f"  replays parsed:    {replays_parsed} ({success_rate:.1%})")
    print(f"  replays failed:    {replays_failed}")
    print(
        f"  showteam coverage: {showteam_player_slots}/{2 * replays_seen} player-slots ({showteam_rate:.2%})"
    )
    print(f"  decision records:  {total_decision_records}")
    for kind, count in sorted(records_by_kind.items()):
        print(f"    {kind}: {count}")
    winner_rate = replays_with_resolved_winner / replays_parsed if replays_parsed else 0.0
    print(
        f"  resolved winner:   {replays_with_resolved_winner}/{replays_parsed} replays "
        f"({winner_rate:.1%})"
    )
    print("  outcome records:")
    for outcome, count in sorted(outcomes.items()):
        print(f"    {outcome}: {count}")
    if skipped_by_reason:
        print("  skipped records by reason:")
        for reason, count in skipped_by_reason.most_common():
            print(f"    {reason}: {count}")
    else:
        print("  skipped records: 0")
    print(f"  output: {args.out}")

    if replays_seen and success_rate < SUCCESS_RATE_INVESTIGATE_THRESHOLD:
        print(
            f"\nWARNING: parse success rate {success_rate:.1%} is below the "
            f"{SUCCESS_RATE_INVESTIGATE_THRESHOLD:.0%} bar -- top failure reasons:",
            file=sys.stderr,
        )
        for reason, count in fail_reasons.most_common(10):
            print(f"    {reason}: {count}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
