# Full VGC learning pipeline

## Current status (2026-09-07)

The current learned models are **experimental**. A3/B1 models load, but the audit
found data-split and offline/live search defects, and B1 still fails recorded retention
requirements. See `docs/audit_implementation_plan_2026-09-07.md` for the repair sequence
and `data/models/registry.json` for saved model identities. No new model is approved.

Think of the stages as teaching a driver. The simulator supplies the laws of physics and
the car's exact controls. Strong replays demonstrate useful habits. A search teacher then
labels complete decisions in situations where the learner has the same information it
will have in a real game. Finally, reinforcement learning lets it practice for the only
score that ultimately matters: winning.

## Information boundary

The learner always knows its own six Pokemon, including their moves, item, ability,
nature, Stat Points, and final stats. It never reads the opponent's private simulator
state.

For each opposing Pokemon it instead maintains a probability distribution over plausible
sets. The starting probabilities come from same-format replay and usage data. Public
reveals make facts certain, while move order and observed damage shift the remaining
hidden Stat Point and nature probabilities. This is the same gradual discovery available
on the real ladder.

Public Showdown replays usually do not contain the acting player's private full team
sheet. Therefore replay training is used to warm-start the shared state representation;
it is not treated as a complete joint-action teacher. Complete action labels are collected
afterward inside the fog-safe simulator, where our own team is known exactly.

## Historical pipeline commands — not a current execution plan

The following commands document the earlier pipeline. Do not execute them as a new
training or deployment recipe: current data recovery, audits, and release requirements
take precedence. The failed outcome-based scaling recipe is closed pending a new hypothesis.

Install the training dependency first:

```bash
uv sync --extra dev --extra train
```

### 1. Refresh same-format high-level replays

```bash
.venv/bin/python tools/download_replays.py --min-rating 1300
.venv/bin/python tools/parse_replays.py
.venv/bin/python tools/build_set_priors.py
```

The downloader is incremental. The parser keeps complete battles together so later
training and validation do not share turns from one battle.

### 2. Warm-start from strong public decisions

```bash
.venv/bin/python tools/train_bc.py --epochs 12 --min-rating 1300 --device cpu \
  --out runs/full_pipeline/bc_high_rating
```

Behavior cloning means learning to copy demonstrated decisions. This first stage learns
useful battle patterns but predicts slot-level replay actions, so it is only a warm start.

### 3. Learn complete joint actions from the search teacher

```bash
.venv/bin/python selfplay/train_imitation.py --games 2000 \
  --team-manifest data/selfplay/archetype_pool_150/manifest.json \
  --bc-checkpoint runs/full_pipeline/bc_high_rating/bc_policy.pt \
  --split-by team --epochs 30 --out-dir runs/full_pipeline/imitation
```

This stage records the full legal doubles order, all six of our sets, fog-safe opponent
beliefs, and first-principles tactical facts. Whole teams are held out during validation.

### 4. Improve through reinforcement learning

```bash
.venv/bin/python selfplay/train_full_pipeline.py --iterations 100 \
  --games-per-iteration 256 --eval-games 1000 --eval-every 5 \
  --init runs/full_pipeline/imitation/best.pt \
  --out-dir runs/full_pipeline/rl
```

Reinforcement learning means updating decisions from game outcomes rather than being
shown a correct answer. Training mixes fixed opponents, prior saved models, mirror and
non-mirror matchups, and many learner teams. A small teacher anchor discourages sudden
forgetting. Rewards remain only win, loss, or draw.

`latest.pt` is the newest saved model. `best.pt` changes only when a frozen evaluation
improves the worst opponent result first and the average result second. A checkpoint is
a saved model file.

### 5. Run the promotion gate

```bash
.venv/bin/python offline/evaluate_learned_policy.py \
  --checkpoint runs/full_pipeline/rl/best.pt \
  --games-per-split 1000 --minimum-games 1000 \
  --output runs/eval/learned_policy_gate.json
```

The gate fails closed unless all of these are true:

- the model uses complete own-team, opponent-belief, and tactical inputs;
- it has at least 10,000 reinforcement-learning games and is linked to a frozen evaluation;
- its unseen-team confidence interval is above 50%;
- its confidence interval against the shipped `vgc` policy is above 50%;
- no opponent rung falls below a 45% point estimate;
- the training-team to unseen-team gap is at most five percentage points; and
- no decision crashed into the random safety fallback.

Team-pool uncertainty is clustered by team instead of pretending every game is an
independent coin flip. The report also stores the model file's SHA-256 fingerprint, so
the result always identifies the exact bytes that were tested.

### 6. Deploy only an explicitly named checkpoint

Start with a local smoke against a running local Showdown server:

```bash
.venv/bin/python ladder/run_ladder.py --local-smoke --n 2 --opponent heuristic \
  --policy-checkpoint runs/full_pipeline/rl/best.pt
```

After the promotion gate passes and the local smoke has zero fallbacks:

```bash
.venv/bin/python ladder/run_ladder.py --n 1 \
  --policy-checkpoint runs/full_pipeline/rl/best.pt
```

Current public play requires hybrid mode, a compatible named checkpoint, and readiness
checks; a model file existing is not approval. The historical command above is insufficient
for current deployment. Release enforcement is part of the repair plan. Local diagnostic
play remains separate from public approval.

## Neural-guided search experiment

The learned model does not control the shipped bot's search shortlist. First run it in
shadow mode: normal search still chooses every action, while the model's ranking,
search-winner rank, known score regret, timing, work counts, and exact checkpoint hash
are written to JSONL.

```bash
.venv/bin/python offline/evaluate_neural_search.py --mode shadow --games 1000 \
  --checkpoint runs/full_pipeline/rl_promotion_10k/best.pt \
  --records runs/eval/neural_search_shadow.jsonl \
  --dataset runs/eval/neural_search_shadow.pt \
  --output runs/eval/neural_search_shadow.json
```

Only after the shadow report's team-clustered recall gate passes should Hybrid-10 be
tested. Safety candidates consume slots *inside* the fixed K budget, so the experiment
cannot hide extra search work. Hybrid mode automatically pairs unchanged full-search
and hybrid games on the same learner/opponent teams, simulator seeds, policies, and
seats. The schedule mixes team mirrors with rotated non-mirror opponents.

```bash
.venv/bin/python offline/evaluate_neural_search.py --mode hybrid --games 1000 \
  --search-k 10 --checkpoint runs/full_pipeline/rl_promotion_10k/best.pt \
  --output runs/eval/neural_search_hybrid10.json
```

The fixed progression is Shadow -> Hybrid-10 -> Hybrid-8 -> Hybrid-5. The feature is
default-off and is not registered as a ladder baseline. Hybrid-8 requires the passed
Hybrid-10 report through `--prior-gate`; Hybrid-5 likewise requires the passed Hybrid-8
report, and the checkpoint fingerprint must match. The JSONL stores real search
scores only for actions that were actually searched; capped tail scores are never
exported as teacher labels. The JSONL is the human-readable audit; the `.pt` file keeps
the encoded observation and candidate arrays needed for later training. The RL value
output is logged but is not connected to
hypothetical search positions in this milestone.

## Historical measured evidence (2026-08-15)

The 2026-08-15 pipeline build produced these development results:

- High-rating replay warm start: move top-1 accuracy rose from 19.2% to 28.7%; top-3 was
  53.6%.
- Complete joint-action imitation on held-out teams: exact teacher agreement rose from
  14.6% to 40.4%; top-3 agreement reached 71.5%.
- The first promotion-scale run completed 10,240 mixed-team reinforcement-learning
  battles. On the exact same 1,000 unseen-team games, the frozen policy improved from
  51.4% to 63.5%. Every opponent rung improved: random 94.4% -> 96.3%, max-power
  70.6% -> 91.2%, heuristic 61.9% -> 73.8%, myopic `vgc` 35.9% -> 50.0%, shallow
  `vgc` 26.4% -> 39.1%, and full `vgc` 21.3% -> 32.3%.
- A separate 2,000-game promotion gate measured 63.5% on training teams and 62.5% on
  unseen teams, only a 1.0-point generalization gap. The unseen-team cluster-aware range
  was 57.5% to 67.5%, and all 2,000 games completed with zero fallbacks.
- The model was **not promoted**. It still scored only 30.5% against full `vgc` and
  39.3% against shallow search in the fresh gate, so it failed both the incumbent and
  weakest-opponent requirements. `best.pt` remains an experimental learned checkpoint.
- A two-game local learned-policy ladder smoke against the heuristic finished 2-0 with
  zero fallbacks, replays, decision traces, and the exact checkpoint fingerprint.

These numbers prove that data flows through every stage and that the safeguards work.
They do not prove competitive strength. Only the full gate above can authorize promotion.
