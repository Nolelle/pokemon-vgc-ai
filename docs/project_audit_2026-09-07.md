# Project audit — 2026-09-07

The foundations contain useful work, but the evidence does not currently justify calling the learned player approved or treating more training data as the established next solution. Fix experimental validity before another large collection or training run.

Audited checkout: `46d256076986201be9f05c1c8423a8c9cfbedccb`. The working tree was clean at the start. `.agents/worklog.md` is absent. This review inspected current code, dataset records, saved models, evaluation reports, and reran checks. It did not train models, play public ladder games, or change strategy code. Historical results below remain historical, not fresh strength measurements.

## First principles

The goal is to win real hidden-information doubles battles within the decision timer. Five different questions must not be conflated:

1. Are the rules and legal choices correct?
2. Does the player use only information available at that moment?
3. Does its search make good decisions with that information?
4. Does learned assistance improve speed or winning without losing important choices?
5. Does that improvement survive new teams, opponents, and real deployment?

An accurate simulator is like an accurate chessboard: it establishes what moves do, not which move wins. Copying a teacher is useful for reducing search work, but copying accuracy does not independently establish playing strength.

## Confirmed findings, in priority order

### 1. Offline hybrid search and live search have different information access

`src/vgc/rl/match.py:107` attaches the real `DirectBattle` root to offline observations. `src/vgc/rl/search_guidance.py:518` uses that root whenever present; it constructs `LiveExactMirror` and hidden-state possibilities only when the root is absent. `offline/evaluate_neural_search.py:158` runs through this offline match path.

`src/vgc/rl/exact_search.py:137` enumerates responses from the other player's observation, and branch simulation clones the supplied root. With the real root, those transitions use actual hidden sets. Live ladder battles do not supply this root and instead use public reconstructions. Public-only neural inputs do not repair private information used downstream by search.

This is a code-confirmed evaluation/deployment mismatch; this audit did not quantify how often it changes the winner. An offline hybrid strength pass through this path would not establish the live player's strength. The current teacher collector separately uses `public_information_exact_search`, which always builds a public mirror; do not mislabel its new data as private-root data because of this separate defect.

Required correction: use the same public reconstruction for teacher collection, offline hybrid/reference evaluation, and deployment. Add a test that changes unrevealed private sets while holding the visible observation fixed and asserts identical decision inputs and search behavior, with randomness fixed.

### 2. The current B1 validation split violates the project's team-separation contract

Current corpus: 41,543 decisions, 5,600 namespaced battles, 160 own teams. Training uses 33,393 decisions and validation uses 8,150. The manifest assigns 128 own-team fingerprints to training and 32 to validation.

However, `split_samples_grouped` in `src/vgc/rl/demonstrations.py:286` groups only by our team. Counting both our and the opponent's teams gives 160 distinct teams in training, 155 in validation, and **155 shared identities**. These are real, freshly counted overlaps. Therefore the internal validation result is not an unseen-team result under `docs/data_requirements.md`.

The fresh external A3 holdout is different: its 140 team fingerprints across both roles have zero overlap with the 160-team B1 corpus. Its stored recall results remain useful development evidence. The split defect does not make every external result invalid.

Required correction: partition teams before scheduling either side of a battle. For existing data, keep only within-partition matchups or group connected teams together; never silently claim both-role separation from an own-team split. Preserve whole battles.

### 3. The audit and trainer disagree, and training does not require audit success

Fresh command:

```bash
.venv/bin/python offline/audit_training_data.py \
  --dataset runs/collect_b_s{1,2,3,4,5,6,7,8}/demonstrations.pt \
  --split-manifest runs/train_b1/split_manifest.json
```

Result: FAIL, zero malformed examples and zero duplicate decision keys; errors report 582 overlapping battle identifiers and 155 overlapping team fingerprints.

The battle error is an audit defect: `load_shard_datasets` prefixes battle identifiers with the collection directory, but the auditor concatenates raw samples. A separate count using the trainer's prefixes found 5,600 distinct battles and **zero actual battle overlap**. The raw audit reports only 700 battle identifiers. The team-overlap error is real.

The trainer writes a split manifest and proceeds to fitting without invoking the full split audit. Individual dataset loading does validate samples, but that is not equivalent to validating separation. The multi-file auditor also validates only the first file's metadata; the training loader separately checks cross-file invariants.

Required correction: share dataset identity and split-validation logic across training and audit; make failure stop training. Add negative tests with opponent-team overlap and multiple collection files. Do not repair this by suppressing the real overlap check.

### 4. The current model status is experimental, not approved

A checkpoint is a saved model file. Direct loader verification found:

| File under `runs/` | Current loader | Evidence and status |
|---|---|---|
| `full_pipeline/teacher_5x_model/best.pt` | Rejected: v3-meta | Historical August strength pass; incompatible with current loader |
| `train_a3/best.pt` | Loads: v4-mechanics | 11,759-example corpus; external guided retention 95.85%, lower bound 94.83%; FAIL |
| `train_a3_soft/best.pt` | Loads: v4-mechanics | External guided retention 96.09%, lower bound 95.12%; FAIL |
| `train_b1/best.pt` | Loads: v4-mechanics | 41,543-example corpus; external guided retention 96.46%, lower bound 95.49%; FAIL |
| `train_b1_soft/best.pt` | Loads: v4-mechanics | Internal validation retention 94.75%; no separate external recall report found |

Guided retention means the teacher's preferred move survives in ten suggestions that mix model choices with rule-based safety choices. The required lower bound is 98%; the lower bound accounts for uncertainty across teams. These are stored evaluation results, not rerun recall measurements. The recall reports name paths but do not store checkpoint content hashes, limiting exact retrospective attribution.

All four inspected new models have 715,094 learned numbers. Both B1 metrics files have `game_evaluation: {}` and saved arguments `eval_games: 0`. No current B1 strength approval was found. The old approved file cannot transfer its approval to a different model or search path.

Verified B1 SHA-256: `47fe6e701b37adc043b6795752f55c650e577f496d4dfb10b96e0aeb49860cbd`.
Verified B1 soft SHA-256: `d9b62dd38b4b1d1f323a21f3f2d944eba9a982b30d7374ff81a4d36542413f70`.
Verified old August SHA-256: `968c47307db1be840fe550abba48844d55a0d75d5bd649151a11478994e89c48`.

The public ladder runner checks mechanics/state/action readiness and the model's mechanics-feature flag, but does not require a passing strength report tied to the supplied model. Approval is presently a procedural rule rather than an enforced model-release boundary.

### 5. “Data scale binds, not recipe” is not established

The contract reaches this conclusion from similar small-data recipes. Similar outcomes cannot rule out limitations in representation, teacher quality, coverage, or the training target. Increasing the corpus from 11,759 to 41,543 examples moved stored external guided retention from 95.85% to 96.46%: about 0.61 percentage points, still below the requirement. This is useful progress, not proof that a particular larger size will pass. The internal A/B comparisons also need the split caveat above.

The old 76,801-example result used a different architecture and teacher history. Its passing size is not a transferable sample requirement. Diagnose the missed decisions and run controlled comparisons on nested subsets of the corrected corpus before buying another scale increase.

### 6. Soft training targets include artificial scores for unsearched actions

`exact_search.py:211` assigns the unsearched tail scores below the worst searched score according to tail order. These numbers encode exclusion/ranking, not simulated outcome evidence. `distill.py:915` converts the full legal score vector into target probabilities without restricting it to `searched_mask`.

Thus the soft recipe learns partly from synthetic ranking conventions. That may be a usable engineering objective, but it is not a distribution of measured action quality. Correct the terminology and test a searched-only or explicitly censored target before drawing conclusions from the soft-target experiment.

## Data coverage and model design

The source-of-truth game exports, complete own-team data, explicit hidden information, legal-action recording, teacher-source checks, and content fingerprints are sound design choices. All eight inspected B1 collection records name clean source commit `c1cad213e1b6b8b77ac3efb56a37b4c8a6734826`.

However, the generated teams share a small family of curated cores and generation rules. Different packed bytes establish different teams, not independent strategies. Add evaluations on genuinely different cores and fresh real-team sources, rather than interpreting 160 generated teams as complete ladder coverage. The public set-prior file records 2,939 replays with a minimum-rating filter of 1,000; it is public usage evidence, not an expert-strategy corpus. Its metadata lacks a collection-date range. Whether those probabilities are accurate on current opponents remains unestablished here.

The candidate model combines summarized battle features, history, exact own-team/public-opponent features, tactical action features, and a serialized full mechanics state. It is `CandidatePolicyValueNet`, not merely `BcPolicyNet` as the current model contract says. `BcPolicyNet` supplies its shared starting representation.

The complete mechanics state is converted into byte-sized input pieces (tokens). A local pattern reader spanning seven bytes, followed by global average/maximum pooling, compresses it. Keeping every byte at input does not prove the model can associate a status or number with the right Pokemon after compression. This is an architectural concern, not a measured cause of the plateau. Test targeted state changes and remove-one-component comparisons before prescribing a replacement architecture.

The imitation procedure trains action preferences (a policy) by copying the teacher (behavior cloning). It ignores the model's winning-chance output during fitting. Consequently these new files contain an untrained value-output layer; the filename “policy-value” does not make it a useful win predictor. Keeping learned value out of decisions is correct.

The teacher searches a limited set of our actions and opponent responses for one turn, averages sampled outcomes, and combines the result with hand-written scores. Its position assessment is not calibrated to winning chances. Exact turn execution does not make these labels optimal or expert-level. Teacher quality needs independent tactical cases and public-information battle evidence.

## Historical outcome-based training

Reinforcement learning means updating choices from wins and losses. The current implementation uses terminal outcomes, limited updates, and optional pressure to retain teacher choices. Those are coherent ingredients; they do not establish that the complete experiment is effective.

The three saved 100k promotion reports all still record `passed: false`. That supports keeping those models unpromoted, not declaring all outcome-based learning impossible. The documented private-teacher and architecture changes prevent treating old and new experiments as one clean learning curve. Do not resume the same large recipe without a new, testable explanation for why it should improve.

## Evaluation and project management

Keep the existing conservative thresholds and team-aware uncertainty. Also report difficult action sets separately: 533 of the 1,637 external holdout decisions have at most ten legal actions, so retention in ten is automatic on roughly one third of examples. Measure the consequence of misses, not just their count, and compare against a rule-only shortlist at the same time budget.

The same external holdout has been used to compare A3, soft A3, and B1. Treat it as development validation from now on; reserve a new untouched final exam before release. A passing comparison against the incumbent establishes only that comparison, not a high ladder rating.

`docs/learned_model_contract.md` simultaneously says the project substantially passes, describes the old model as unusable, and still says no current model exists. Its claim that hybrid never calls exact search is contradicted by current code. `docs/full_learning_pipeline.md` also contains historical deployment instructions. Replace these conflicting status claims with a model register identifying exact bytes, architecture, source data/splits, source commits, passed tests, and approval status. Preserve source datasets and approved models outside disposable `runs/` storage.

## Fresh verification

- Mechanics readiness: PASS, 128 tests passed.
- Battle-state readiness: PASS, 57 tests passed.
- Action readiness: all 19 tests passed, but the command refused to write an approval artifact because this new, uncommitted audit document made the working tree dirty. This is not a failed action test.
- B1 data audit: FAIL as detailed above. Independent counting separated real team overlap from the auditor's battle-namespace defect.
- Four current candidate checkpoints loaded successfully; the historical approved checkpoint was rejected by the current loader.

The first restricted-server test attempts encountered server startup errors; reruns with local server access succeeded. Readiness checks used `--no-fetch`, so this audit does not claim a fresh comparison with upstream Showdown. No new strength or recall matches were run. Raw check outputs are in `runs/audit_20260907/` (scratch); the findings and model identities are preserved here.

## Recommended order

1. Repair offline/live information parity and the split/audit enforcement. Preserve existing evidence as historical or development-only.
2. Freeze a both-role team split and a separate untouched final test. Record coverage by team core, opponent, turn phase, and legal-action count.
3. Establish the current public-information teacher's tactical correctness, strength, and actual decision time. Compare model assistance to a rule-only shortlist under equal budgets.
4. Diagnose teacher misses, synthetic score targets, and information representation with small controlled experiments and more than one initialization. Separate copying improvement from winning improvement.
5. Only then choose between more diverse data, a different representation, a better teacher, or a new outcome-based training experiment.
6. Release only a model with durable identity and matching public-path retention, strength, timing, and local-smoke evidence.

This is a repairable research pipeline. The main priority is making the evidence trustworthy enough to tell which next experiment is worth running.
