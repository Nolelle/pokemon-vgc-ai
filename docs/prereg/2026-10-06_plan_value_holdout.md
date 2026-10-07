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
