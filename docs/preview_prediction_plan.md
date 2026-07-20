# Iteration 6 plan: symmetric opponent preview prediction

User-approved direction (2026-07-20): the biggest remaining lever is picking the right
4-of-6 against THEIR 4-of-6. Hypothesis (user's): good opponents follow the same
first-principles rules we do (win con, engine, roles, answers), so our own preview
machinery pointed backwards predicts their selection.

## Steps

1. **Predictor** (`vgc/preview_predict.py`): enumerate the OPPONENT's 90 pick-4 x lead
   options over their previewed 6 (sets filled from `data/usage/set_priors.json` +
   spreads), score with the same `team_preview` scoring internals
   (`_build_matchup_matrix`/`_score_choice`) from THEIR side — but score them against
   OUR **species-only** view filled from priors (they can't see our sets; no x-ray
   vision either direction). Softmax scores -> a distribution over (bring4, leads),
   temperature knob on PolicyConfig.
2. **Backtest FIRST** (`tools/backtest_preview_prediction.py`): ground truth = the 5,878
   teampreview records in `data/bc/decisions.jsonl` (what each real player actually
   brought/led; parser marks best-effort derivations). Metrics: top-1/top-3 bring-4
   accuracy and lead-pair accuracy, vs two baselines: (a) uniform random legal, (b)
   "their 4 most-used species by corpus usage". GATE: if the predictor does not beat
   baseline (b) meaningfully, do NOT integrate — the hypothesis fails cheap.
3. **Integration**: `build_team_order` scores OUR 90 candidates against the predicted
   opponent-selection distribution (likelihood-weighted + the usual worst-case hedge
   consistent with `search_worst_case_weight`'s philosophy) instead of the flat all-6
   assumption; predicted leads seed turn-1 threat context. New PolicyConfig knobs,
   default ON only after the backtest gate passes.
4. **Validation**: same-session A/B (note: SimpleHeuristicsPlayer's preview is naive, so
   the A/B may undersell the gain — the backtest and the ladder block are the real
   evidence); then a 25-game ladder block; postmortem watch: `wrong_four`/`wrong_lead`
   categories.

## Context for whoever implements

- Campaign state: peaks 1163/1161/1242/1199/1188; true strength ~1150-1200; goal 1300.
- Block-4 finding motivating this: 9/10 "severely losing response" turns were already
  forced — games are lost BEFORE the midgame, i.e. at preview/matchup trajectory.
- Methodology rules in CLAUDE.md apply (same-session A/Bs only; cd into repo in every
  shell call).
