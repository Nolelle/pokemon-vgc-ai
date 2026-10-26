#!/usr/bin/env python
"""Pick a "ladder mirror" subset of a real-team pool whose species mix matches what is
being played on the public ladder right now.

Target: for every public replay in the corpus rated >= `--min-rating` and uploaded in
the last `--days` days (players in `--exclude-player` skipped, e.g. our own account),
both players' team-preview species. A species' target share is the fraction of those
team previews that include it -- i.e. how often an opponent at that rating brings it to
preview. Team preview is public, so this uses only public information.

Selection: greedy forward selection on each side of the pool's train/holdout split
separately (so the mirror keeps the split team-disjoint). At each step it adds the team
that most reduces the squared error between the subset's species shares and the target,
up to `--fraction` of that side. Deterministic (ties broken by file name).

Writes `<pool>/ladder_mirror_train_manifest.json`, `ladder_mirror_holdout_manifest.json`
(same entry format as the input manifests, so `--team-manifest` accepts them) and
`ladder_mirror.json` (target shares, achieved shares, distances, corpus window).

Usage:
    .venv/bin/python tools/build_ladder_mirror.py --pool data/selfplay/mc_sheet_pool_v3 \\
        --exclude-player laplacestheorems
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vgc.config import FORMAT_ID  # noqa: E402
from vgc.damage import to_id  # noqa: E402

DEFAULT_REPLAYS_DIR = REPO_ROOT / "data" / "replays" / FORMAT_ID


def preview_species(details: str) -> str:
    """Species id from a `|poke|` details field ("Indeedee-F, L50, F" -> "indeedeef")."""
    return to_id(details.split(",")[0].strip())


def ladder_target(
    replays_dir: Path, min_rating: int, since: float, excluded: frozenset[str]
) -> tuple[dict[str, float], int, int]:
    """(species -> share of team previews containing it, previews counted, games counted)."""
    counts: Counter[str] = Counter()
    previews = 0
    games = 0
    for path in replays_dir.glob("*-*.json"):
        replay = json.loads(path.read_text())
        rating = replay.get("rating")
        if rating is None or rating < min_rating or (replay.get("uploadtime") or 0) < since:
            continue
        if any(to_id(name) in excluded for name in replay.get("players") or ()):
            continue
        sides: dict[str, set[str]] = {}
        for line in replay.get("log", "").split("\n"):
            if line.startswith("|poke|"):
                parts = line.split("|")
                sides.setdefault(parts[2], set()).add(preview_species(parts[3]))
        full = [species for species in sides.values() if len(species) == 6]
        if not full:
            continue
        games += 1
        for species in full:
            previews += 1
            counts.update(species)
    return ({s: c / previews for s, c in counts.items()} if previews else {}), previews, games


def shares(entries: list[dict]) -> dict[str, float]:
    counts = Counter(s for entry in entries for s in set(entry["species"]))
    return {s: c / len(entries) for s, c in counts.items()} if entries else {}


def distance(achieved: dict[str, float], target: dict[str, float]) -> float:
    """Sum of absolute share differences over every species in either mix (L1)."""
    keys = set(achieved) | set(target)
    return sum(abs(achieved.get(k, 0.0) - target.get(k, 0.0)) for k in keys)


def greedy_select(entries: list[dict], target: dict[str, float], k: int) -> list[dict]:
    keys = sorted(set(target) | {s for e in entries for s in e["species"]})
    index = {s: i for i, s in enumerate(keys)}
    goal = [target.get(s, 0.0) for s in keys]
    vectors = []
    for entry in entries:
        vec = [0] * len(keys)
        for s in set(entry["species"]):
            vec[index[s]] = 1
        vectors.append(vec)
    chosen: list[int] = []
    sums = [0] * len(keys)
    remaining = sorted(range(len(entries)), key=lambda i: entries[i]["file"])
    while remaining and len(chosen) < k:
        n = len(chosen) + 1
        best, best_err = None, None
        for i in remaining:
            vec = vectors[i]
            err = sum(((sums[j] + vec[j]) / n - goal[j]) ** 2 for j in range(len(keys)))
            if best_err is None or err < best_err - 1e-12:
                best, best_err = i, err
        chosen.append(best)
        remaining.remove(best)
        sums = [a + b for a, b in zip(sums, vectors[best])]
    return [entries[i] for i in chosen]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--replays-dir", type=Path, default=DEFAULT_REPLAYS_DIR)
    parser.add_argument("--min-rating", type=int, default=1200)
    parser.add_argument("--days", type=float, default=14.0)
    parser.add_argument("--fraction", type=float, default=0.6)
    parser.add_argument("--exclude-player", action="append", default=[], metavar="NAME")
    args = parser.parse_args()

    now = time.time()
    since = now - args.days * 86400
    excluded = frozenset(to_id(name) for name in args.exclude_player)
    target, previews, games = ladder_target(args.replays_dir, args.min_rating, since, excluded)
    if not previews:
        print("no team previews in the window; nothing to mirror", file=sys.stderr)
        return 1

    report: dict = {
        "window_days": args.days,
        "since_unix": int(since),
        "built_unix": int(now),
        "min_rating": args.min_rating,
        "excluded_players": sorted(excluded),
        "games": games,
        "team_previews": previews,
        "fraction": args.fraction,
        "target_shares": dict(sorted(target.items(), key=lambda kv: -kv[1])),
        "sides": {},
    }
    print(f"ladder window: {games} games, {previews} team previews (last {args.days:g} days, >= {args.min_rating})")
    for side in ("train", "holdout"):
        manifest_path = args.pool / f"{side}_manifest.json"
        entries = json.loads(manifest_path.read_text())
        k = max(1, round(len(entries) * args.fraction))
        picked = greedy_select(entries, target, k)
        before, after = shares(entries), shares(picked)
        out = args.pool / f"ladder_mirror_{side}_manifest.json"
        out.write_text(json.dumps(picked, indent=1) + "\n")
        gaps = sorted(
            ((after.get(s, 0.0) - t, s) for s, t in target.items() if t >= 0.03),
            key=lambda x: -abs(x[0]),
        )[:8]
        report["sides"][side] = {
            "source_manifest": manifest_path.name,
            "source_teams": len(entries),
            "mirror_teams": len(picked),
            "l1_before": round(distance(before, target), 4),
            "l1_after": round(distance(after, target), 4),
            "largest_remaining_gaps": {s: round(g, 4) for g, s in gaps},
            "achieved_shares": dict(sorted(after.items(), key=lambda kv: -kv[1])),
        }
        print(
            f"{side}: {len(picked)}/{len(entries)} teams, species-mix distance "
            f"{distance(before, target):.2f} -> {distance(after, target):.2f} ({out})"
        )
        print("  largest remaining gaps: " + ", ".join(f"{s} {g:+.1%}" for g, s in gaps))
    (args.pool / "ladder_mirror.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"report: {args.pool / 'ladder_mirror.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
