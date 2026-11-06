#!/usr/bin/env python
"""Who-won check: does the higher PRE-game-rated player win more often than not?

Reads a selection manifest (tools/inventory_replay_pretraining.py output). Uses each
game once (not per perspective). Counts a game only when both pre-game ratings are
known and differ, and the winner is known (played out, forfeit, or timer). Bands by
the LOWER player's pre-game rating, because the old post-game >=1200 download filter
mainly distorted games whose weaker player started just under 1200.

    .venv/bin/python tools/rating_skew_check.py MANIFEST [MANIFEST ...]
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

BANDS = [(0, 1000), (1000, 1100), (1100, 1200), (1200, 1234), (1234, 1300),
         (1300, 1400), (1400, 1500), (1500, 1600), (1600, 10_000)]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def known(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def check(path: Path) -> None:
    tallies = defaultdict(lambda: [0, 0])  # (scope, band) -> [stronger_wins, games]
    skipped = defaultdict(int)
    for line in path.open():
        game = json.loads(line)
        if game.get("split") == "excluded" and "after_window" in (game.get("exclusions") or []):
            skipped["after_window"] += 1
            continue
        ratings = game.get("ratings") or {}
        r1, r2 = ratings.get("p1"), ratings.get("p2")
        winner = game.get("winner")
        if not (known(r1) and known(r2)):
            skipped["rating_unknown"] += 1
            continue
        if r1 == r2:
            skipped["equal_ratings"] += 1
            continue
        if winner not in ("p1", "p2"):
            skipped["no_winner"] += 1
            continue
        stronger = "p1" if r1 > r2 else "p2"
        low = min(r1, r2)
        band = next(f"{a}-{b - 1}" if b < 10_000 else f"{a}+" for a, b in BANDS if a <= low < b)
        for scope in ("all", game.get("split") or "unknown"):
            tallies[scope, band][0] += winner == stronger
            tallies[scope, band][1] += 1
    print(f"\n== {path}")
    print("skipped:", dict(skipped))
    for scope in ("all", "train", "validation", "test"):
        rows = [(band, tallies[scope, band]) for band in
                [f"{a}-{b - 1}" if b < 10_000 else f"{a}+" for a, b in BANDS]
                if tallies[scope, band][1]]
        if not rows:
            continue
        print(f"-- scope={scope}  (band = lower player's pre-game rating)")
        for band, (k, n) in rows:
            lo, hi = wilson(k, n)
            print(f"  {band:>10}  games={n:6d}  stronger wins={k / n:6.1%}  95% CI [{lo:.1%}, {hi:.1%}]")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        check(Path(arg))
