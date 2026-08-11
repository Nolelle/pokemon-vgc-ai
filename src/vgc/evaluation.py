"""Shared evaluation statistics.

`wilson_interval` lived in `offline/run_matches.py` and is still importable from there
(that module re-exports it, so `offline.run_gates`' import keeps working). It moved here
so `vgc.rl.match` can use it without a `src/vgc/` package importing the top-level
`offline/` scripts, which would invert the dependency direction.
"""

from __future__ import annotations

import math


def wilson_interval(wins: int, games: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval -- same formula as pokemon-tcg-ai's offline/run_matches.py."""
    if games <= 0:
        return 0.0, 0.0
    p = wins / games
    z2 = z * z
    denom = 1.0 + z2 / games
    center = (p + z2 / (2.0 * games)) / denom
    margin = (z / denom) * math.sqrt((p * (1.0 - p) / games) + (z2 / (4.0 * games * games)))
    return max(0.0, center - margin), min(1.0, center + margin)
