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

### Open items

1. **Recall re-confirmation (cheap, this checkout).** The August recall numbers
   predate the current tree. Nothing in the intervening diffs touches the
   recall path, but the contract requires the number to be re-measured, not
   inherited. Run: `offline/evaluate_shortlist_recall.py` on the expanded
   holdout with the deployed checkpoint.
2. **Checkpoint durability (the real gap).** The gated hybrid checkpoint is
   `runs/full_pipeline/teacher_5x_model/best.pt` (sha256 `968c4730...`) --
   gitignored scratch space. `data/models/*.pt` are committed but none of
   them IS that file, and the default `bc_checkpoint_path` still points at
   `bc_policy_v3sp.pt`. If `runs/` is cleaned, the strength-approved model is
   gone while its verdict file claims it exists. Fix: copy the gated file
   into `data/models/` under a descriptive name, point the ladder smoke docs
   at it, and record the sha here. Do NOT retrain; just preserve the file.
3. **Strength re-gate: on demand, not on schedule.** The 1,500-pair gate is
   the most expensive check in this contract. Rerun it only when search
   weights, the encoder, or the checkpoint change -- none have. A passing
   recall re-confirmation plus the unchanged-code audit above is sufficient
   to carry the August strength verdict forward.

## 7. Implementation order

1. Freeze this contract (this file). No behavior changes.
2. Recall re-confirmation on the expanded holdout with current code.
3. Durability fix: preserve the gated checkpoint + recipe reference.
4. Record the numbers here; Problem E passes when gates 1-5 are green.
   Strength re-gate only on search/encoder/checkpoint change.
