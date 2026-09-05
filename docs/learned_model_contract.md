# Problem E: learned-models contract

This document defines what the learned components may do, what they must never
do, and how we prove they help. The current checkout substantially passes this
contract: the action-ranking model and its hybrid deployment are built, gated,
and strength-approved, with August evidence on file. Two items are open: the
recall bar must be re-confirmed under the current code, and the deployed
checkpoint is not durably stored.

## Takeaway

The unit we care about is **one decision with learned help**. A network that
has seen many battles suggests which candidates deserve the search's limited
attention. The search still runs those candidates and still makes the pick.
Think of it as an intern who triages the inbox: useful for speed, never the
signatory.

A *policy* here means the model's action preference ("these moves look
promising"). *Behavior cloning* means learning from examples produced by a
stronger teacher (our search), not from winning and losing. A *checkpoint* is
one saved model file. *Recall@K* means: how often the teacher's winning action
survives inside the model's top-K suggestions. *Shadow* means measuring the
guided player on states it reaches itself, rather than states the teacher
visited.

## 1. Terms and authority

- **Action-ranking model:** `vgc.bc.model.BcPolicyNet`, trained by
  `selfplay/train_imitation.py` on search-teacher demonstrations. Input: the
  whole mechanics state as tokens (`vgc.bc.encoding`, layout v4). Output: per
  move/target log-probabilities per slot, plus an optional outcome value head.
- **BC rerank blend** (`use_bc_policy`, ships **off**): re-scores the heuristic
  top-K with network log-probabilities. Advisory only.
- **Guided shortlist** (`vgc.rl.guided_selection`, the deployed path): up to
  half the search budget goes to heuristic safety slots, the rest to neural
  discoveries. A non-default candidate becomes the pick only by clearing
  `guided_upset_margin` (10 points) over the best default-searched candidate.
- **Value head** (`use_value_head`, ships **off**): predicts who is favored.
  Exists, unpromoted.
- **Teacher:** exact-search demonstrations (`public_mirror_exact_showdown_teacher_v2`
  only; private-root history rejected -- see Problem B).
- **Holdout:** teams excluded from training by packed-content match, so
  memorized teams cannot inflate scores.

The project rule stands: the learned model suggests, search decides. Granting
a learned component more authority needs its own powered gate, not an argument.

## 2. Component status

| Component | State | Evidence |
|---|---|---|
| Action ranking (BC net, v4 encoding) | Built, gated | Guided recall §5 |
| Guided shortlist + upset arbitration | Deployed path, gated | Powered re-gate §5 |
| Position value head | Built, **off**, unpromoted | Needs its own gate before any authority |
| Opponent-response model | **Closed** | +0.0011 ± 0.0013 headroom; do not rebuild without a new hypothesis (CLAUDE.md) |
| Learned Q / value blending in the agent | **Closed** | 9 configs flat vs search; pure-RL 100k scale failed promotion |
| Hidden-info estimator | Usage priors + belief builder (Problem B), not learned | No learned hidden-info model exists or is planned |

## 3. Training progression (where we stand)

1. ~~Imitate strong search decisions (behavior cloning).~~ Done: 76,801 v2.x
   teacher decisions, recall-selected checkpoint + hard-example weight 2.0 +
   action-count bin balancing.
2. ~~Self-play practice data.~~ Done: v4-selfplay corpus and checkpoint exist;
   the recipe, not any single file, is the asset.
3. ~~Improve search speed/strength while keeping search in charge.~~ Done via
   hybrid shortlist guidance (§5).
4. **More authority only after conservative evaluations pass.** Not granted and
   not requested: the search remains the decider. This is the correct
   end-state for Problem E, not unfinished business.

## 4. Completion gates

Problem E is complete only when all gates hold in the current checkout:

1. **Recall:** guided recall@10 lower bound ≥ 0.98 on the clean expanded
   holdout, plus live-shadow confirmation. (`offline/evaluate_shortlist_recall.py`,
   `offline/evaluate_neural_search.py`.)
2. **Strength:** hybrid guided search non-inferior to full search in the
   powered paired gate (margin -0.02), with zero fallbacks.
3. **Unseen teams:** all measurements on holdout teams with byte-identical
   training teams excluded by content match.
4. **Speed:** hybrid decisions meaningfully faster than full search at the
   same width, with no strength loss.
5. **Durability:** the deployed checkpoint, its training dataset reference,
   and its recipe are stored outside gitignored scratch space so the result
   survives a `runs/` cleanup. **Currently OPEN -- see §6.**

## 5. August evidence (on file, pre-registered methodology)

- Retrain recipe: recall-selected checkpoint + hard-example weight 2.0 +
  action-count bin balancing, lr 3e-4, on 76,801 v2.x decisions.
- Clean 150-team expanded holdout (18,209 decisions, 10 byte-identical teams
  excluded): pure recall@10 97.7% (LCB 0.974); **guided@10 LCB 0.984 = PASS**.
- Live shadow: **guided@10 99.1% / LCB 0.985 = PASSED**.
- Powered paired strength gate (1,500 pairs): first run FAILED narrowly on
  upset flips; fix shipped (`guided_upset_margin=10`); **re-gate PASSED**
  (hybrid +0.4pts vs full search, delta [-0.009, +0.017]; arbitration fired
  on 0.77% of decisions). Verdict: `runs/eval/neural_search_5x_regate.json`.
- Hybrid latency at gate width: total mean 62ms, p95 150ms; compute reduction
  recorded in the verdict file.
- Public smoke: 6-4 with zero fallbacks; hybrid is the live ladder policy
  (`ladder/run_ladder.py --policy-checkpoint <ckpt> --policy-mode hybrid`).

## 6. Current checkout audit (2026-09-03)

### Verified unchanged since the August gates

- `vgc.search` (the hybrid path's engine): no behavior change since the gates
  -- Rung 2 ships at shortlist identity (1) and Rung 3a touched exact-search
  values only, which the hybrid path never calls.
- Encoder layout v4 and `BcPolicyNet` untouched; all five `data/models/*.pt`
  (committed) still load under the current code.
- Authority wiring untouched: BC blend and value head still default off,
  upset margin still 10, no learned Q in the agent.

### Collection blockers: found and fixed (2026-09-03, commits `68540e1`, `52747c2`)

The first costing probe failed closed on turn 1 of every p2-seat game
(`KeyError: 'p1: ...'`, plus bring-four overflow errors on pool teams).
Root cause: `patch_public_state` installed the teacher's battle object as the
perspective side's parser base without checking seats. From the p2 seat that
put p2-keyed team dicts under p1 request lines, so the first clone step raised
and the game recorded nothing. Pinned by
`test_public_teacher_labels_from_p2_seat_without_keyerror` (new
`live_mirror_seat_parity` mechanics family).

The second probe then crashed live play outright: the teacher submitted a
move the real battle had disabled (`InvalidChoice: ... Dragon Pulse is
disabled`). Root cause: the snapshot never marked request-withheld moves
disabled -- Choice lock, Encore-out moves, and anything else the request
merely withholds never appeared in `MoveSnapshot.disabled` -- so the patched
mirror root offered the full moveset and exact search ranked a forbidden
move. Fixed in `snapshot_pokemon`: a known move the live request does not
offer is now snapshotted `disabled` with reason `"request"`, which the worker
patch already carried into the sim's move slots. Pinned by unit test plus
`test_mirror_reproduces_choice_lock_from_snapshot` (new
`mirror_move_availability` mechanics family).

### Collection cost (measured 2026-09-03, fixed tree)

10 pool-team games vs the heuristic proxy, default production teacher config:
72 decisions in 156.6s = **2.17 s/decision**, zero fail-closed errors.
Reference scales (single-threaded): ~12k decisions ≈ 7h; ~77k (August scale)
≈ 46h. August sharded across ~8 collections, so plan on comparable
parallelism: 12k ≈ 1h at 8-way, 77k ≈ 6h at 8-way. Training time is on top
(CPU torch; August recipe: lr 3e-4, hard-example weight 2.0, bin balancing,
recall-selected checkpoint).

### Pilot run (2026-09-04, `runs/pilot_v4`, this tree)

60 pool games, default 6-opponent mix, production teacher config, 1 CPU epoch:
**457 v3-public-information samples across 52 teams (369 train / 88
validation), both split skip-rate gates green, checkpoint saved as
`candidate-policy-value-v4-mechanics`**. Validation recall@10 37.5% -> 52.3%
and teacher-rank median 21.5 -> 8.0 from random init -- the pipeline learns
end to end. Wall time ~20 min all-in. Two more defect classes appeared on the
way here and are now covered: clone `lastMove` deserialization (Encore-branch
sim crash; fixed in the worker clone path) and end-without-start on
short-transcript parsers (tolerated in ingest with a counter). One
`more choices than unfainted` branch rejection remains OPEN (seen once);
it is skip-counted, not silent.

### Option A collection (2026-09-04, `runs/collect_a3_*`, this tree)

8 shards x 200 games on `archetype_pool_150`, `--split-by battle
--collect-only --max-skip-rate 0.08`: **11,759 v3 samples across 160 teams**,
merged via `offline/merge_collection_shards.py` (same-pool invariant check;
the stock merger correctly refuses partitioned manifests). Skip rates per
shard 0-0.6%, all green. One shard's save died silently mid-write and was
recollected; per-shard datasets verified loadable before merge.
Fresh v3 holdout: 220 games on `expanded_holdout` (content-disjoint from all
train pools) -> **1,637 decisions, 0 skips**, `runs/collect_a3_holdout`.
First train (`runs/train_a3`, August recipe, 12 epochs): validation recall@10
34.8% -> **93.5%**; holdout guided@10 LCB **0.948** -- short of the 0.98 bar,
close to August's 11.5k operating point (96.3%/0.954). The 24-epoch rerun
(`runs/train_a3_24ep`) plateaued flat at ~93% from epoch 5, and soft targets
at this scale (`runs/train_a3_soft`) read 94.0% val / **0.951** holdout LCB --
indistinguishable from hard labels. Three recipes agree near ~0.95: data
scale binds, not training choices. Full Option B (~77k, August's clearing
scale) is the remaining lever; it needs ~9 GB of disk (currently ~2 GB free)
plus the deferred deletions, and ~6h at 8-way sharding.

### Open items

1. **The gated hybrid checkpoint does not load (BLOCKING).** The August
   strength verdict's checkpoint, `runs/full_pipeline/teacher_5x_model/best.pt`
   (sha256 `968c4730...`), is tagged `candidate-policy-value-v3-meta`. The
   current `load_snapshot` (v4-only since commit `33d3a0e`) refuses it --
   verified by running the shadow gate's loader against it. A sweep of every
   `best.pt` in `runs/` and `data/` found **zero** v4-mechanics checkpoints:
   all 20+ are v3-meta. Consequences:
   - `offline/evaluate_neural_search.py` (shadow + hybrid gates) cannot run at
     all: it loads the checkpoint up front and fails for every file that
     exists. Gates 1 (recall, shadow flavor) and 2 (strength) are unrunnable,
     not merely stale.
   - Public ladder play is unrunnable: it requires `--policy-mode hybrid`
     plus a checkpoint, and no loadable checkpoint exists.
   - Copying the v3 file into `data/models/` would NOT fix this -- the bytes
     would still be refused. The fix is a fresh train that writes the v4
     architecture (today's `train_imitation` already stamps
     `RL_ARCHITECTURE_VERSION` at save time), followed by the recall +
     strength gates on the new file.
2. **The offline recall screen is blocked on data, correctly.** The 18,209-
   decision expanded holdout exists only as
   `runs/full_pipeline/teacher_holdout_expanded.pt` in v1 demonstration
   format, which the loader now refuses (private-root teacher -- see Problem
   B). Re-running recall on it would violate the information boundary, so the
   refusal is the gate working, not a tooling bug. Fresh v3 (public-mirror)
   demonstrations are needed for any offline recall number.
3. **What still works.** The BC-rerank path (`score_orders` + `data/models/`
   `*.pt`, default-off) loads under the current code -- verified `v3sp`
   (legacy v2 layout) and `v4` (current) both load, with 93 policy tests
   green. Only the `NeuralSearchPlayer`/snapshot path (shadow, hybrid,
   ladder-hybrid) is stranded.
4. **Strength re-gate: blocked behind items 1-2.** It needs a loadable
   checkpoint first. No retrain is proposed here; that is the first real
   training decision of Problem E/F and needs its own plan (teacher
   collection cost is ~10x per decision at reduced width).

## 7. Implementation order

1. Freeze this contract (this file). No behavior changes.
2. **Scale-up collection (Option A, running 2026-09-04):** 8 shards x 200
   games on `archetype_pool_150` (byte-disjoint from the holdout below),
   `--split-by battle --collect-only --max-skip-rate 0.08`, seeds
   31001-31008, out `runs/collect_a3_s{1..8}` (~12k decisions expected).
   Shard manifests: `data/selfplay/collect_a3/shard_{1..8}.json` (20 teams
   each, regenerated from the pool manifest; all scratch, gitignored).
3. Fresh v3 holdout demonstrations on `data/selfplay/expanded_holdout`
   (150 teams, disjoint by content hash) for the recall verdict.
4. Merge (`offline/merge_demonstrations.py`) + train with the August recipe
   (lr 3e-4, `--hard-example-weight 2.0`, `--balance-action-count-bins`,
   recall-selected) to a v4 checkpoint in durable storage.
5. Recall gate on the fresh holdout, then the powered strength gate.
   Record the numbers here; Problem E passes when gates 1-5 are green.
