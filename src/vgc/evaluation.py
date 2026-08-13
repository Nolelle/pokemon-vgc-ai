"""Shared evaluation statistics.

`wilson_interval` lived in `offline/run_matches.py` and is still importable from there
(that module re-exports it, so `offline.run_gates`' import keeps working). It moved here
so `vgc.rl.match` can use it without a `src/vgc/` package importing the top-level
`offline/` scripts, which would invert the dependency direction.

## Why `wilson_interval` is not enough for team-pool gates

`wilson_interval` assumes every game is an independent draw from one win rate. That
holds for a fixed-mirror series (one team, one matchup) and does NOT hold for a gate
that plays many DIFFERENT teams: games are clustered inside teams, and teams genuinely
differ in how much the change helps them. Measured on both the 58-team and 160-team
archetype pools, the true team-to-team SD is ~0.11 win rate, which inflates the variance
of the pooled win rate by ~1.8-1.9x -- so a naive Wilson interval on 1044 pooled games
is ~30% too narrow and its lower bound can clear 0.50 on a change that is really a wash.

`clustered_interval` is the fix: a cluster-robust (sandwich) interval that treats each
team as the sampling unit. `variance_components` explains WHERE the uncertainty lives,
and it carries the design consequence that matters most when budgeting a gate: because
the team effect does not shrink with more games per team, a pool of K teams has an
irreducible standard-error floor of `sqrt(tau^2 / K)`. Past that point the only way to
tighten the interval is MORE TEAMS, not more games.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


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


Cluster = tuple[int, int]
"""One sampling unit of a clustered gate: `(wins, games)` for a single team."""


def clustered_interval(clusters: Sequence[Cluster], z: float = 1.96) -> tuple[float, float]:
    """Cluster-robust confidence interval for a pooled win rate, clustering by team.

    The pooled rate is a ratio estimator `sum(wins) / sum(games)`, so this is its
    standard linearized (sandwich) variance, which makes no assumption that games within
    a team are independent and handles unequal team sizes:

        Var(p) = K / ((K - 1) * N^2) * sum_i (wins_i - p * games_i)^2

    Falls back to `wilson_interval` below two clusters, where the between-cluster
    variance is undefined and pooling is the only thing available.
    """

    usable = [(int(wins), int(games)) for wins, games in clusters if int(games) > 0]
    total_games = sum(games for _, games in usable)
    total_wins = sum(wins for wins, _ in usable)
    if total_games <= 0:
        return 0.0, 0.0
    if len(usable) < 2:
        return wilson_interval(total_wins, total_games, z)
    p = total_wins / total_games
    k = len(usable)
    residual = sum((wins - p * games) ** 2 for wins, games in usable)
    variance = k / ((k - 1) * total_games**2) * residual
    margin = z * math.sqrt(variance)
    return max(0.0, p - margin), min(1.0, p + margin)


@dataclass(frozen=True)
class VarianceComponents:
    """Where a clustered gate's uncertainty comes from, and what that costs to shrink."""

    clusters: int
    games: int
    win_rate: float
    tau: float
    """Estimated true between-team SD of the win rate. Real signal, not sampling noise."""
    naive_se: float
    """SE the pooled Wilson interval implicitly assumes (all games independent)."""
    clustered_se: float
    """SE that accounts for clustering by team. This is the honest one."""
    design_effect: float
    """`clustered_se^2 / naive_se^2` -- how much clustering inflates the variance."""
    se_floor: float
    """`sqrt(tau^2 / clusters)`: the SE left if every team played INFINITE games."""

    @property
    def minimum_detectable_effect(self) -> float:
        """Smallest true edge this run could certify (95% lower bound above 0.50)."""

        return 1.96 * self.clustered_se

    @property
    def floor_detectable_effect(self) -> float:
        """Smallest edge THIS TEAM POOL could ever certify, at any number of games.

        Nothing below this is reachable by buying more games; it needs more teams.
        """

        return 1.96 * self.se_floor

    def teams_needed(self, effect: float, games_per_team: int) -> int:
        """Teams required to certify `effect` at `games_per_team` games each."""

        target_se = effect / 1.96
        if target_se <= 0:
            return 0
        per_team = self.tau**2 + 0.25 / max(games_per_team, 1)
        return math.ceil(per_team / target_se**2)


def variance_components(clusters: Sequence[Cluster]) -> VarianceComponents:
    """Split a clustered gate's variance into sampling noise and real team-to-team spread.

    `tau^2` uses the standard one-way random-effects moment (ANOVA) estimator, which
    tolerates unequal team sizes and is truncated at zero when the between-team mean
    square falls below the within-team one (i.e. no detectable team effect).

    Read `tau` (not `tau^2`) with care at small team counts: the truncation at zero plus
    the square root make it biased UPWARD, so a pool of ~58 teams at 18 games each reads
    tau ~= 0.02-0.06 about a tenth of the time with no real team effect at all. Treat a
    small `tau` as "no evidence of a team effect", not as a measured one. Both the
    58-team A/B (0.106) and the 160-team A/B (0.114) sit far outside that null range;
    the A/A null test on the same 58-team pool measured 0.029, consistent with zero.
    """

    usable = [(int(wins), int(games)) for wins, games in clusters if int(games) > 0]
    total_games = sum(games for _, games in usable)
    total_wins = sum(wins for wins, _ in usable)
    k = len(usable)
    if total_games <= 0 or k == 0:
        return VarianceComponents(0, 0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    p = total_wins / total_games
    naive_se = math.sqrt(p * (1.0 - p) / total_games)
    low, high = clustered_interval(usable)
    clustered_se = (high - low) / (2 * 1.96)
    if k < 2:
        return VarianceComponents(k, total_games, p, 0.0, naive_se, naive_se, 1.0, 0.0)

    rates = [wins / games for wins, games in usable]
    between = sum(games * (rate - p) ** 2 for (_, games), rate in zip(usable, rates)) / (k - 1)
    within_df = total_games - k
    within = (
        sum(games * rate * (1.0 - rate) for (_, games), rate in zip(usable, rates)) / within_df
        if within_df > 0
        else 0.0
    )
    # Effective cluster size for unequal team sizes; equals the common size when they match.
    scale = (total_games - sum(games**2 for _, games in usable) / total_games) / (k - 1)
    tau2 = max((between - within) / scale, 0.0) if scale > 0 else 0.0
    return VarianceComponents(
        clusters=k,
        games=total_games,
        win_rate=p,
        tau=math.sqrt(tau2),
        naive_se=naive_se,
        clustered_se=clustered_se,
        design_effect=(clustered_se / naive_se) ** 2 if naive_se > 0 else 1.0,
        se_floor=math.sqrt(tau2 / k),
    )


def normal_cdf(x: float) -> float:
    """Standard normal CDF."""

    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Lentz's algorithm)."""

    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 201):
        m2 = 2 * m
        numerator = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        numerator = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-16:
            break
    return h


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """`I_x(a, b)`. Standard Numerical-Recipes formulation; no SciPy in this project."""

    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def student_t_cdf(t: float, df: float) -> float:
    """Student's t CDF, used where an SE is ESTIMATED from few clusters.

    The archetype guardrail tests ~9-11 teams per archetype, and at 9 clusters the normal
    approximation's one-sided 5% point is 1.645 against the t's 1.86 -- i.e. treating an
    estimated SE as if it were known makes the very false alarms this guardrail exists to
    avoid about a third more likely. Converges to `normal_cdf` as `df` grows.
    """

    if df <= 0:
        return float("nan")
    if t == 0.0:
        return 0.5
    x = df / (df + t * t)
    tail = 0.5 * _regularized_incomplete_beta(df / 2.0, 0.5, x)
    return 1.0 - tail if t > 0 else tail


def holm_rejections(pvalues: Sequence[float], alpha: float = 0.05) -> list[bool]:
    """Holm-Bonferroni step-down: which of `pvalues` survive at family-wise `alpha`.

    Needed wherever a gate applies the same test to several subgroups at once. Six
    independent uncorrected 95% subgroup checks raise the chance of at least one false
    alarm to ~14% under a true null -- which is exactly how this pool gate flagged an
    archetype at 40.7% that measured 54.9% on the very next seed. Holm keeps the
    family-wise error rate at `alpha` while staying uniformly more powerful than a plain
    Bonferroni split.
    """

    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    rejected = [False] * len(pvalues)
    remaining = len(pvalues)
    for rank, index in enumerate(order):
        if pvalues[index] <= alpha / (remaining - rank):
            rejected[index] = True
        else:
            break
    return rejected
