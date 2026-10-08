# Pre-registration: engine-measured plan value, holdout confirmation (2026-10-06)

Written before the run. Do not edit after the run starts.

- Change: `exact_search_field_control=true`, `exact_search_field_measured_plan=true`,
  `exact_search_field_fit_weight=0.5` vs `exact_search_field_control=false`.
  Player `vgc_exact`, narrow width (`search_our_candidates=4`,
  `exact_search_future_samples=1`, `search_opp_candidates=4`), commit 2d8e1d5.
- Data: owner teams vs `data/selfplay/mc_sheet_pool_v2/holdout_manifest.json` (88 teams,
  never used for this change), asymmetric mode, both orientations and seats, seed 20261040.
- **Primary** (the post-hoc signal from train): `terrain_pulse_blastoise`, 24 games per
  opponent pair (2112 games). PASS iff the cluster-robust 95% lower bound > 0.50.
- **Secondary**: all six owner teams pooled, 8 games per pair (4224 games). Reported, not a
  ship gate.
- Shipping requires the primary PASS; a fail means the train signal was noise.

## Result (appended after the run)

- Primary, terrain_pulse_blastoise vs 88 holdout teams: **1114/2112 = 52.7%, cluster-robust
  [0.503, 0.552] -- PASS** (`runs/eval/prereg_blastoise.json`).

## Caveat found after the run (appended)

- A Codex review then found that `vgc.plan_value`'s registry was process-wide and keyed by
  species+moves, so in these runs (both arms in one process) a player could read the other
  team's measured entry for a same-species, same-moves Pokemon (160 colliding identities over
  320 test teams). Fixed in 609a9a0 (`vgc.team_scope`). The PASS above ran on the
  contaminated code. The same pre-registered test (same teams, games, criterion) will be
  re-run once the remaining review fixes land; that rerun decides shipping.
- The secondary all-six-teams run was stopped before finishing for the same reason.

## Rerun on fixed code (appended, 2026-10-07; commit 278943f, same teams/games/seed/criterion)

- Primary, terrain_pulse_blastoise vs 88 holdout teams: **1107/2112 = 52.4%, cluster-robust
  [0.499, 0.549] -- FAIL** (lower bound 0.499 is not > 0.50). Not shipped.
- The point estimate is stable across runs (train 54.6%, contaminated holdout 52.7%, clean
  holdout 52.4%): a consistent ~+2.5-point lean the 88-team holdout cannot certify.
