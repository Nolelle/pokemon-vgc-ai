# Problem D: search and planning contract

This document defines what the thinking engine must do before committing to a
move, and how we prove that thinking harder helps. The current checkout
partially passes this contract: the machinery is built and unit-tested, but the
strength and latency evidence is stale and must be re-measured here.

## Takeaway

The unit we care about is **one decision with a thinking budget**. Given the
legal choices (Problem C) and the current board plus guesses about hidden facts
(Problem B), the engine must try its own promising actions against plausible
opponent replies, score where each line ends up, and pick the best one -- all
before the battle timer runs out.

Think of it like chess: list your candidate moves, imagine the opponent's best
answers, judge the resulting boards, and move. A fast gut feeling (the myopic
evaluator) ranks the candidates; the search checks the top ones for real.

There are two search paths, and they answer different questions:

- **Shipped live search** (`vgc.search.search_joint_orders`): a Python
  simulation of one turn plus a two-turn damage-race forecast. Fast enough for
  the ladder. Its transitions are approximations; Showdown remains the truth.
- **Exact offline search** (`vgc.rl.exact_search.search_joint_orders_exact`):
  every candidate/reply pair executed by real Showdown clones across several
  randomness samples. Too slow for live play at full width; used for teachers
  and training labels.

## 1. Terms and authority

- **Myopic score:** the one-turn heuristic judgment of an order
  (`vgc.evaluator.score_joint_orders`). No opponent reply is simulated.
- **Candidate:** one of our legal joint orders worth spending simulation on.
- **Shortlist:** the top `search_our_candidates` (default 10) of the ranked
  candidates. Everything outside it keeps a capped tail score and can never
  outrank a searched order (ranking-safety fix, 2026-07).
- **Opponent response:** one plausible joint reply from the opponent's side,
  built from known damaging moves, Protect, utility/control moves, and safe
  bench switches, capped at `search_opp_candidates` (default 12).
- **Exchange:** one simulated turn: our switches resolve first, then all moves
  in priority/speed order against current (possibly already-damaged) states.
- **Position value:** the judgment of an unfinished board (HP left, faints,
  status, boosts, effects, win-condition plan). Strategy, not mechanics.
- **Rolling horizon:** two extra projected joint-attack turns after the
  searched exchange, valued below the immediate turn.
- **Exact branch:** one candidate/reply pair executed by a real Showdown clone
  under one fixed randomness sample.
- **Authority:** whichever component makes the final pick. It must be a search
  until a learned shortcut passes its own promotion gates.

The myopic evaluator is the authority for the initial ranking. The exchange
simulation (Python live, Showdown exact offline) is the authority for what a
line leads to. The aggregated final score is the authority for the pick.

## 2. The shipped live pipeline

For each decision (`VgcPlayer.decide` with `use_two_ply_search`, the default):

1. Rank every legal joint order with the myopic evaluator.
2. Re-sort the ranking by the spread-belief mixture when enabled (ships at 1,
   identity).
3. Keep the top 10 as the searched block; the rest become the capped tail.
4. Enumerate up to 12 plausible opponent responses.
5. Simulate each candidate x response exchange in Python (switches first,
   then priority/speed order, Trick-Room-aware, Protect-aware, with geometric
   repeat-Protect odds).
6. Aggregate each candidate's response values as worst-case blend plus
   likelihood-weighted expectation (worst-case weight 0.4).
7. Project two forecast turns per response when the horizon is on; blend at
   weight 0.45.
8. Final score = myopic + exchange + forecast. Pick the best. Tail stays
   below every searched order by construction.

Every weight lives on `PolicyConfig` with its own comment. No bare strategic
literal may appear in the decision code.

## 3. The exact offline pipeline

For each teaching decision (`search_joint_orders_exact` on a `DirectBattle`):

1. Same myopic ranking and shortlist as live.
2. Opponent replies from the other side's own scored orders (top 12),
   weighted by softmax over their scores.
3. Every candidate x reply x future-seed (default 4) branch executed by a real
   Showdown clone; accuracy, crits, rolls, sleep, ties all come from the
   engine.
4. Position deltas averaged over seeds, blended worst-case plus expectation
   with the same weights as live.
5. Same tail-below-searched guarantee. Every searched branch records
   `mechanics_source: official_showdown_clone`.

## 4. What an unfinished-position score must consider

The position value (both paths) must account for:

- Remaining Pokemon and remaining health on both sides.
- Faint differentials, weighted above single-turn damage.
- Major status (sleep/freeze above burn/poison/paralysis).
- Stat stages.
- Active effects with a correct sign (helpful vs harmful to the holder --
  Rung 3a; unsigned effects score zero, never a bonus).
- Speed-control and board resources the forecast tracks (safe pivots, traps,
  win-condition plan).

The exact weights are hand-chosen and uncalibrated. That is declared policy,
not a mechanics gap: Showdown decides which position is reached; the value
only says how much it is liked. Calibrating it is later work, not this gate.

## 5. Latency budget

A decision must fit comfortably inside the live battle timer with headroom for
network wobble:

- Live Python search: seconds-scale worst case, millisecond-scale typical.
  Every decision records `elapsed_ms` in its breakdown; the audit below sets
  the bound from measured data, not from a guess.
- Exact search at production width is tens of seconds per decision and runs
  off the connection loop (worker thread) or offline only. It must never
  block heartbeats.

## 6. Authority rule

The learned model suggests; search decides. Concretely:

- The behavior-cloning policy blend (`use_bc_policy`) and the outcome value
  head (`use_value_head`) ship **off**. Either needs its own promotion gate.
- Hybrid shortlist guidance admits a non-default candidate as the pick only
  when it beats the best default-searched candidate by `guided_upset_margin`
  (10 points). Pure network argmax is never the decider.
- No learned Q, opponent-response model, or value blending in the playing
  agent (closed direction; see CLAUDE.md).

## 7. Completion gates

Problem D is complete only when all gates pass in the same clean checkout:

1. **Superiority:** the shipped search beats the myopic-only player
   (`use_two_ply_search=False`) in a same-session A/B on the varied team
   pool, with a cluster-robust interval clearing zero. One-team mirrors do
   not count (overfit rule).
2. **Scaling:** more thinking beats less thinking on the same positions:
   wider shortlist and horizon-on must not lose to narrower/horizon-off in a
   paired comparison, and exact-search agreement must rise (or at least not
   fall) with width.
3. **Latency:** the decision-time distribution measured on live local games
   in this checkout sits inside the budget with a stated p50/p99, zero timer
   losses, and zero fallbacks.
4. **Authority:** every playing entry point resolves its pick through a
   search; learned components are provably advisory (flags off by default,
   arbitration margin in hybrid, no learned Q in the agent).

Strength evidence expires when the weights change. Any `PolicyConfig` search
weight change after a passing A/B re-opens gates 1-2 until re-measured in the
same session methodology.

## 8. Current checkout audit (2026-09-03)

### Exact (built, tested, wiring verified)

- Live pipeline end to end, including tail-below-searched ranking safety
  (`tests/test_search.py`, 40+ tests: exchange ordering, Protect math,
  switches, redirection, response weights, aggregation, horizon, pruning).
- Exact offline pipeline with signed effect values and belief-combined
  rankings (`tests/test_exact_search.py`, `tests/test_position_effects.py`,
  `tests/test_live_mirror_branches.py`).
- Shortlist budget, response caps, and all weights on commented
  `PolicyConfig` fields; myopic ranking shared verbatim by both paths.
- Authority wiring: BC policy and value head default off; hybrid upset
  margin shipped at 10; no learned Q in the agent; exception-safe fallback
  counted (`fallback_count`).
- Per-decision `elapsed_ms` recorded in every search breakdown.

### Needs evidence (stale or never measured here)

- **Superiority:** the last clean number predates current weights
  (2026-07-22: 98/100 vs the weak heuristic proxy, 99/100 vs random; the
  search-vs-myopic direct comparison on that era's proxy read 73.3% vs
  81.7% against an opponent that never Protects). Since then the weights
  moved (worst-case 0.25 -> 0.4, repeat-Protect odds, utility/switch modeling)
  with only same-session delta A/Bs, never a fresh full superiority run.
- **Scaling:** shortlist width and horizon on/off have isolation unit tests
  (zero-weight controls) but no paired strength comparison in this checkout.
- **Latency:** point samples exist (6.5ms/decision in July; 0.48s vs 0.78s
  state-handling audit in September) but no decision-time distribution on
  the current default config.

## 9. Implementation order

1. Freeze this contract (this file). No behavior changes.
2. Latency audit: decision-time distribution for the shipped default on live
   local games; publish p50/p99 and the budget.
3. Superiority A/B: shipped search vs myopic-only, same session, varied team
   pool with cluster-robust interval (`offline/evaluate_own_spread_pool.py`
   `--candidate`/`--incumbent` overrides).
4. Scaling probe: shortlist width and horizon on/off paired comparisons on
   the same positions.
5. Record the numbers here; Problem D passes when gates 1-4 are green
   together. Any later search-weight change re-opens gates 1-2.
