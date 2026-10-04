# LLM-harness test protocol (pre-registered 2026-10-03)

Written before any LLM-arm result exists. Do not edit the decision rules after seeing data;
add a dated amendment instead.

## Question

Does "engine + LLM proposer" beat "engine given the same extra time" in head-to-head play?
The control gets the same wall-clock budget so any gain is from the LLM's candidates, not
from extra thinking time.

## Design

- **Arms:** candidate = engine + LLM proposer; incumbent = engine with the same extra time.
- **Our teams:** `psyspam_sand` and `salamence_tw` (`teams/owner/`).
- **Opponents:** the 88 held-out real M-C teams in
  `data/selfplay/mc_sheet_pool_v2/holdout_manifest.json`. Never tune on these.
- **Harness:** `offline/evaluate_own_spread_pool.py --our-teams ... --manifest <holdout>`.
  Each (our team, opponent team) pair is played in both orientations (candidate pilots our
  team / candidate pilots the opponent's), both seats, so team strength cancels and the
  A/A rate is 50% by construction.
- **Clustering:** by opponent team (pooled over our teams). Our two teams are fixed, so
  they are a fixed effect (reported separately), not a sample.

## Order of operations

1. **A/A first** (`--null-test`, both arms identical). The 95% cluster-robust CI must contain
   50%. If not, the harness is biased: stop and fix before any A/B.
2. Estimate the opponent-team SD `tau` from the A/A (`vgc.evaluation.variance_components`),
   and recompute the power table below. This only sets the budget; it does not change the
   success rule.
3. Run the A/B with a fixed endpoint of N games (decide N from the power table and the $20
   cap before starting).

## Decision rule (success only at the fixed endpoint)

Success iff, at the fixed end of the run, **both**:

- (a) the team-clustered 95% CI lower bound is **> 50%**, and
- (b) the observed gain is **>= +2 points** (win rate >= 52%; run with `--min-gain 0.02`).

The target gain we care about is +3 points; the +2 floor is a sanity check set below the
target. (Requiring the observed gain to be at least 3 would fail about half the time even
when the true gain is exactly 3, so it is not used as the cut-off.) Anything else is not
success. Head-to-head win rate against the incumbent is **not** the same as ladder rating
gain: a 52-53% head-to-head edge is evidence the LLM helps, not a measured ladder
improvement, and it is measured against held-out sheets, not the live ladder population.

## Futility stop (allowed), success stop (not allowed)

Checkpoints every **500 games** (`--checkpoint-games 500`). At a checkpoint the run may stop
early **only** for futility: CI upper bound < 50% + 3 points (`--futility-min-effect 0.03`, the +3 target),
and only once there are at least 10 opponent-team clusters. Never stop early for success:
looking every 500 games and declaring a win the first time the lower bound clears 50% is a
repeated test and multiplies the false-positive rate. Futility stops cannot add false
positives; they only cost a little power.

## Power (chance of declaring success when the LLM really helps)

Limited by the number of opponent teams K = 88, not just by games N. With `tau` the
between-opponent-team SD of the win rate (normal approximation):

    SE = sqrt(tau^2 / K + 0.25 / N)
    success cut-off = max(1.96 * SE, 0.02)          # conditions (a) and (b) together
    power = P(observed gain >= cut-off) = 1 - Phi((cut-off - true gain) / SE)
    irreducible floor (N -> infinity): SE = tau / sqrt(K)

`tau` 0.114 is the largest value measured in this repo (A/B gates on a different design);
the A/A value is expected near 0-0.03. K = 88, both conditions jointly:

| tau | N games | SE | chance of success if true gain is +3 | +4 | +5 |
| --- | --- | --- | --- | --- | --- |
| 0.00 | 2000 | 1.1 pts | 77% | 95% | 99% |
| 0.00 | 4000 | 0.8 pts | 90% | 99% | 100% |
| 0.00 | 6000 | 0.7 pts | 94% | 100% | 100% |
| 0.06 | 2000 | 1.3 pts | 64% | 87% | 97% |
| 0.06 | 4000 | 1.0 pts | 84% | 98% | 100% |
| 0.06 | 6000 | 0.9 pts | 86% | 99% | 100% |
| 0.114 | 2000 | 1.7 pts | 44% | 68% | 86% |
| 0.114 | 4000 | 1.5 pts | 54% | 79% | 93% |
| 0.114 | 6000 | 1.4 pts | 59% | 83% | 95% |

Plainly: if the opponent-team spread is small, 4,000 games finds a true +3 point gain
about 9 times in 10. If it is as large as the worst case seen, even 6,000 games finds +3
only about 6 times in 10, because more games on the same 88 teams cannot shrink the floor;
only more independent opponent teams can. Re-run this table with the real `tau` from the
A/A before fixing N.

## Budget and escalation

- **Hard cap: $20** total spend on the LLM arm (including the A/A if it calls the LLM).
- **Escalation:** ask the owner for ~$10 more only if the result is promising (about +2
  points) but inconclusive (lower bound <= 50%) **and** more independent opponent teams are
  available. More games on the same 88 teams is not a valid reason: the floor above does not
  move.

## Offline saved-position grading

Grading LLM proposals on saved positions (does the proposal beat the engine's pick under
exact search?) is a **filter only**: use it to kill bad prompts cheaply before spending game
budget. It is not evidence of strength; only the head-to-head result above counts.
