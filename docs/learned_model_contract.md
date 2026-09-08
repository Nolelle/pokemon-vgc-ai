# Problem E: learned-models contract

The current learned player is **experimental, not approved for public deployment**.
New v4 models load, but their recorded retention checks fail and current strength
approval is absent. August approval belongs to a different, incompatible model.
The current inventory is `data/models/registry.json`; implementation and validation
progress is tracked in `docs/audit_implementation_plan_2026-09-07.md` and the worklog.

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

- **Action-ranking model:** `vgc.rl.model.CandidatePolicyValueNet`, trained by
  `selfplay/train_imitation.py` to rank complete legal doubles orders. It uses
  `BcPolicyNet` as its shared starting representation, plus own-team, public-opponent,
  history, tactical, and mechanics inputs. Mechanics tokens are byte-sized pieces of
  the serialized public state. Its winning-chance output is not trained by imitation.
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
| Action ranking (candidate model, v4 mechanics) | Built, experimental | Current B1 retention fails |
| Guided shortlist + upset arbitration | Built, unapproved currently | August evidence is historical |
| Position value head | Built, **off**, unpromoted | Needs its own gate before any authority |
| Opponent-response model | **Closed** | +0.0011 ± 0.0013 headroom; do not rebuild without a new hypothesis (CLAUDE.md) |
| Learned Q / value blending in the agent | **Closed** | 9 configs flat vs search; pure-RL 100k scale failed promotion |
| Hidden-info estimator | Usage priors + belief builder (Problem B), not learned | No learned hidden-info model exists or is planned |

## 3. Training progression

The historical experiments below established useful methods, not current approval.
Current work repairs offline/live information parity, both-role team separation,
mandatory auditing, and evidence identity before another training experiment.
No conclusion that more data alone will solve current retention failures is established.

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
- Public smoke: 6-4 with zero fallbacks; hybrid was used in that historical ladder session
  (`ladder/run_ladder.py --policy-checkpoint <ckpt> --policy-mode hybrid`).

## 6. Historical development notes (2026-09-03 through 2026-09-04)

These notes describe past runs. They do not certify current behavior. In particular,
the current mechanics-enabled hybrid calls exact search, and the September 7 audit
found an offline private-root shortcut that must be removed. Old model compatibility
and strength claims do not transfer to the current candidate architecture.

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
close to the hard-label result. This does not isolate data volume as the bottleneck.
The later B1 run contains 41,543 examples and still fails the external retention bar.
The audit found both-role team overlap in its internal validation split; retain those
numbers only as historical development evidence.

### Current status (2026-09-07)

- The historical August model remains incompatible with the current loader.
- A3 and B1 v4-mechanics checkpoints exist and load. B1's recorded guided retention
  is 96.46%, with a conservative lower bound of 95.49%, below the required 98%.
- B1 and soft B1 contain no battle-strength evaluation in their training reports.
- The external A3 holdout has separate team content but has been used for recipe
  comparison. It is development validation, not an untouched final test.
- The old private-root demonstration data remains ineligible for new promotion.
- No current model has release approval. The model register preserves current file
  fingerprints and identifies which older reports lack exact model binding.

## 7. Implementation order

Follow `docs/audit_implementation_plan_2026-09-07.md`: truthful status, shared public
search, mandatory both-role data auditing, target/evidence repair, small diagnostic
experiments, justified training, and enforced release approval. Do not launch the old
large-run recipe merely because it appears in historical instructions.
