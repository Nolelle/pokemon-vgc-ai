# Pre-registration: does replay pretraining help RL on many teams? (draft, 2026-11-13)

Status: PILOT LOCKED 2026-11-13 (settings below fixed before any pilot result). Numbers in
"Budget" come from 2026-11-13 smoke runs (`runs/rl_multiteam_design/`,
`runs/rl_multiteam/tooling_check/`).

## Why this experiment

Experiment 1 (`runs/rl_warmstart_exp1/`, fixed `phase2_mirror` team, fixed leads) found
warm = fresh, but freeze ablations showed the trained bots never used the state trunk --
with tactical features (108 -> 109 wins frozen) or without them (97 -> 100, 86 -> 81). On
one fixed mirror there is nothing on the board worth reading, so that setup cannot test
pretraining. Here the learner's and opponent's teams change every game, so reading the
board should matter.

## Question

Does starting RL from the replay-pretrained state encoder
(`runs/long_s15_desthead_20261114/turn_net.pt`, 8 tensors) make the policy stronger on
UNSEEN teams than starting from random weights, at the same number of practice games?

## Setup (identical in both arms except the start)

- Trainer: `selfplay/train_ppo.py` in pool mode.
  - `--archetype-pool data/selfplay/mc_sheet_pool_v2/train_manifest.json --holdout-teams-per-archetype 0`
    (226 train teams; learner AND opponent team drawn per worker).
  - Opponents: trainer defaults -- 25% heuristic anchor, 75% rotating self-play snapshots
    (`--snapshot-pool-size 8`), 25% mirror-team workers.
  - `--teacher-anchor-weight 0` (the default 0.05 calls the exact search teacher every
    decision: 8.5 min and 7/8 worker timeouts per 32 games in the smoke run; off it is 40 s,
    0 errors).
  - `--information-features --tactical-features` (the shipped input set).
  - Team preview: the heuristic's `build_team_order` (varies by matchup; not learned).
- Arms: `fresh` vs `warm` (`--bc-checkpoint runs/long_s15_desthead_20261114/turn_net.pt`).
- Seeds: each seed is shared by both arms (same team draws/schedule as far as the trainer
  allows).
- Held-out teams: `data/selfplay/mc_sheet_pool_v2/holdout_manifest.json` (88 teams, split by
  six-species set, never seen in training).

## Evaluation

At 0, 1024, 2048, 4096 and 8192 training games, per run:
- 512 games on the 88 HOLDOUT teams vs `heuristic` and 512 vs `vgc` (fast search), learner
  and opponent teams both drawn from the holdout pool; per-game rows with team labels.
- 256 games on TRAIN teams vs `heuristic` (generalization gap, secondary).
- Freeze check (validity): the same 512 holdout-vs-heuristic games with the state trunk
  frozen to a constant (`trunkc` in `runs/pretrain_review/ablate_eval.py`, adapted to pool
  mode).

## Pilot first (decides whether the main run happens)

2 seeds x 2 arms to 2048 games (about 2 hours on 12 cores).

- **Validity gate:** at 2048 games, freezing the trunk must cost the FRESH arm at least
  5 points on holdout vs heuristic (pooled over its 2 seeds). If not, the bot again plays
  without reading the board -> stop; this setup cannot test pretraining either. Report and
  rethink (e.g. a learned team preview, or removing per-move identity shortcuts).
- **Noise estimate:** seed-to-seed SD of the holdout win rate sets the main run's seed
  count: the smallest count (3 to 6) whose expected 95% interval on the pooled difference
  is at most +/-8 points. If 6 seeds cannot reach +/-8, run 6 and report the wider interval.
- Pilot runs are reused as the first seeds of the main run (same settings), not discarded.

## Main run

N seeds (from the pilot) x 2 arms to 8192 games.

- **Primary (fixed now):** pretrained minus fresh, holdout win rate vs `heuristic`
  averaged over the 1024/2048/4096/8192 checkpoints (learning speed -- the main thing a
  warm start can buy). Per seed and pooled; interval from seed-to-seed variation (t with
  N-1 df), with team-clustered eval noise reported alongside.
- **Secondary:** the same at 8192 only; the same vs `vgc`; train-minus-holdout gap; the
  freeze-check cost per arm.
- **Decision rule:** warm start is adopted for future RL only if the pooled primary
  difference is > 0 with the interval excluding 0. Otherwise use fresh (simpler).
- No extension or early stop because of results; only the pilot gate can stop it.

## Not in scope

- Beating search. At 8192 games the policy is expected far below `vgc`; the `vgc` numbers
  are tracked to see the trend, not as a pass/fail.
- Teacher anchor, learned team preview, reward shaping -- separate questions.

## Budget (from smoke runs)

- Training: ~0.8 games/s with 8 jobs (32 games in 40 s incl. server start). 8192 games ~3 h
  per run at 8 jobs; 12 cores fit about 2 runs at a time.
- Evaluation vs `vgc`: ~10 games in ~20 s single-process; about 1,800 eval games per
  checkpoint per run with the freeze check.
- Pilot ~2 h. Main with 4 seeds ~12-14 h wall clock (4 extra runs to 8192 plus extending
  the 4 pilot runs from 2048).

## Locked run settings (pilot and main)

- Seeds: pilot 41101, 41102 (main adds 41103.. as needed). Run dirs
  `runs/rl_multiteam/{pilot,main}_{fresh,warm}_s<seed>/`.
- Trainer: `--games-per-iteration 32 --jobs 3 --snapshot-pool-size 8
  --save-checkpoints-at-games 0,1024,2048,4096,8192 --skip-server-start` (one shared
  Showdown server for concurrent runs); fresh arm `--bc-checkpoint /nonexistent` (the
  trainer otherwise warm-starts from a default BC file), warm arm must log 8/8 tensors.
- Eval tool: `offline/evaluate_pool_checkpoint.py` (in-process simulator; learner and
  opponent team drawn independently per game from the manifest; seeded STOCHASTIC policy;
  heuristic team preview both sides; per-game rows; game schedule depends only on
  (eval seed, game index), so every checkpoint and arm plays the same games -> paired).
  Fixed eval seeds: holdout vs heuristic 70001, holdout vs vgc 70002, train vs heuristic
  70003; the freeze check (`--ablate trunkc`) reuses 70001 (paired with the unablated
  games). Freeze constant = first trunk output of eval game 0.
- Pilot freeze check: at the 2048 checkpoint for all 4 runs (gate uses the fresh runs).
- Validity gate metric: pooled over fresh seeds, (unablated wins - trunkc wins) / games on
  holdout vs heuristic, 512 games per run; gate passes if >= 0.05.
- Errored games are excluded and reported; a run with > 2% errored training or eval games
  is flagged, not silently pooled.

## Work needed before the pilot (done 2026-11-13: items 1-3)

1. Eval wrapper: score a checkpoint on the train and holdout manifests separately (not
   `evaluate_learned_policy.py`'s archetype re-split) vs heuristic and vgc, writing per-game
   rows with team labels for cluster-robust intervals (`vgc.evaluation.clustered_interval`).
2. Pool-mode freeze check (adapt `ablate_eval.py`; must reproduce unablated counts).
3. Confirm `--resume` extends a 2048-game pilot run to 8192 with identical settings.
4. Launch script for the arms/seeds; readiness gates on a clean tree at launch.
