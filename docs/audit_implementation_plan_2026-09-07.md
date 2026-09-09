# Audit implementation plan — 2026-09-07

**Priority confirmed by the user:** focus on data, training, and demonstrated battle
strength. The goal is 1700+ Elo on the public Champions **Reg M-C** ladder. Defer
coaching app work until that playing goal is demonstrated. Offline checks qualify a
candidate for controlled ladder testing; they cannot substitute for ladder results.
Record the exact model/settings, starting and ending rating, games, failures, and
replays for each session. This document remains the repair plan, not a claim that the
steps are complete.

**Progress (2026-09-09):** stages 1–4 are partly done (model registry, public-search
parity, B1/A3 recovery audit PASS, searched-only soft teaching default-off). See
`.agents/worklog.md`. Champions data was re-exported for M-C; usage priors and
spreads remain M-B until the M-C replay corpus is large enough. Do not rebuild
`set_priors.json` from the current ~44 M-C replays.

Build trustworthy training and evaluation first, then improve the model. This plan implements the findings in `docs/project_audit_2026-09-07.md` against checkout `46d2560`. It is a plan only: no training or strategy changes are included in this document.

Success means the same public-information player is trained, tested, and deployed; training cannot proceed on an invalid split; every model result identifies its exact source and saved bytes; and a model receives release approval only after current evidence passes the existing requirements.

The first four stages repair confirmed problems. The fifth measures the teacher and diagnoses learning. The sixth trains a candidate only when that diagnosis supports it. The seventh grants release approval. Complete these as small, reviewable changes in order; do not bundle a model redesign into correctness repairs.

## 1. Establish truthful status and preserve evidence

**Change:** Add a tracked model register, proposed `data/models/registry.json`, with entries for the old August model, A3 variants, B1 variants, and historical outcome-trained models. A checkpoint means a saved model file. Record its content fingerprint, architecture version, source datasets and split, training recipe, source commits, compatible loader, evaluation references, trained outputs, and status: experimental, historical, rejected, or approved. Missing information must be explicit; never invent a historical fingerprint from a current file path.

Correct the current-status sections of `docs/learned_model_contract.md` and `docs/full_learning_pipeline.md`; retain dated historical results. Remove the claims that no current model loads, that current hybrid never calls exact search, and that scaling is the proven remaining lever. Record that the winning-chance output is untrained in current imitation models. The project handoff log is `.agents/worklog.md`.

Preserve source datasets and models unchanged. Do not delete, rename, or convert old artifacts to make them appear approved. Keep small registers/reports in version control. Before release, place the selected model and its dataset/recipe references in durable storage with a verified backup and a restore check; a second copy in disposable `runs/` is insufficient. Choose the concrete storage destination before copying large files, after checking available space. No cloud service is required by this plan.

**Done when:** Every named model has one unambiguous status; current candidates are experimental; historical results cannot be mistaken for current approval. Model registration itself grants no release authority.

## 2. Make offline and live decisions obey one information boundary

**Primary files:** `src/vgc/rl/search_guidance.py`, `src/vgc/rl/distill.py`, `src/vgc/rl/live_mirror.py`, `src/vgc/rl/match.py`, `offline/evaluate_neural_search.py`.

Extract a shared public-search entry point, proposed `src/vgc/rl/public_search.py`, from the collector's public reconstruction path. It accepts the visible observation, exact own team, observed history, search settings, and an optional candidate selector. It creates possible hidden states, searches each, combines results, and closes its resources. Teacher collection, offline reference search, offline guided search, and live guided search must use it.

Remove the hybrid player's shortcut to the real private simulator root. Private simulator access may remain in explicitly isolated mechanics diagnostics, but must never be selected automatically merely because an observation has a private-root attribute. Preserve exact own-team information and both player seats. Derive/fix search randomness consistently so arbitrary battle naming cannot invalidate equivalent-observation comparisons.

Audit all callers, including optional teacher assistance during outcome-based training. Record the information-contract version and search configuration in every decision/evaluation report. A private diagnostic must be ineligible for promotion.

**Tests:** Same visible state and own team, different unrevealed opponent moves/items/spreads/hidden timers: identical public hypotheses, model inputs, candidate scores, and final action under fixed randomness. Publicly revealing a changed fact must update the reconstruction appropriately. Cover both seats, replacement switches, disabled moves, and resource cleanup. Ensure the real root is not invoked, and branches actually advance and differ where expected. Retain existing mechanics/state/action tests.

**Done when:** Offline and live reference/guided decisions use the same public path and pass these tests. Fixing the boundary establishes comparability, not playing strength.

## 3. Unify dataset identity, splitting, and mandatory auditing

**Primary files:** `src/vgc/rl/demonstrations.py`, `offline/merge_collection_shards.py`, `offline/audit_training_data.py`, `selfplay/train_imitation.py`, `tests/test_demonstrations.py`, `tests/test_training_data_audit.py`.

Introduce one shared multi-file loader and validator. Use source identity plus original battle/decision identity consistently in merged and direct-file training. Source identity must survive a file move and must not depend only on its parent directory name. For existing files, derive identity from recorded source/file fingerprints; for new collections, persist an explicit collection identity. Reject repeated copies of the same source rather than counting them as new evidence. Preserve original identities through merging.

Validate every file's metadata and cross-file compatibility, sample validity, fingerprints, counts, duplicate identities, assignments, whole-battle separation, and both teams' membership. Distinguish valid repeated observations in different battles from duplicate recordings of the same decision.

The trainer must call the shared full audit after resolving the split and before model fitting. Failure must prevent training and checkpoint creation. Smoke data may use a separately marked development path but cannot become a promotion corpus. Store the audit identity with the saved model. Do not add a bypass for promotion-bound runs.

For new collections, assign teams to training/validation/test before scheduling either player. Both teams in a game must belong to that partition. Record exclusions, content identities, generator/source versions, and counts by team family. A byte-different team is not automatically a different strategy; track family/core overlap separately.

**Existing-data recovery, preferred:** Assign the full valid B1 corpus to training, and use the already-inspected external A3 holdout as development validation through explicit separate dataset arguments. Verify both-role team separation, shared-battle separation, collection compatibility, and warm-start model provenance before accepting this pairing. Do not pool the two partitions during loading. The former B1 internal validation numbers become historical development measurements, not unseen-team evidence.

If this pairing fails source checks, produce a retention report for an alternative fixed team split, keeping only battles whose two teams belong to the same partition. Report dropped games and coverage changes. Do not force an arbitrary split if connected teams leave too little independent validation data.

Reserve a new final test with no team-content overlap and a separate new-team-family component. Account for available provenance of replay warm starts and public priors; mark unknown historical exposure rather than claiming it was excluded. Freeze this test before comparing new recipes and do not inspect its outcomes during development.

**Tests:** Relocation preserves identity; merged and multi-file input agree; repeated source files fail; a bad second file fails; opponent-only overlap fails; a battle split across roles fails; valid disjoint inputs pass; audit failure stops fitting. The current raw B1 split must still fail for real team overlap, while the false battle-collision report disappears.

**Done when:** Recovered data passes the same audit used by training, and every partition has explicit, verified semantics. No recollection merely to repair metadata or identifiers.

## 4. Correct teaching-target semantics and bind reports to actual models

**Primary files:** `src/vgc/rl/distill.py`, `src/vgc/rl/exact_search.py`, `selfplay/train_imitation.py`, `offline/evaluate_shortlist_recall.py`, `offline/evaluate_neural_search.py`, `tests/test_distill_objective.py`.

Keep hard teacher-choice copying as the unchanged reference recipe. Separate measured searched scores from artificial unsearched ranking scores in the data contract. Retain old arrays for compatibility, but do not interpret an unsearched tail score as measured action quality.

Add an experimental searched-only soft objective: compare the relative probabilities of searched candidates, normalizing both teaching probabilities and model probabilities over the searched subset. This prevents treating unknown actions as measured bad actions. Rows with fewer than two measured actions carry no relative-ranking signal; report them explicitly. If combined with hard-choice copying, keep the terms separate, fix their weight before comparison, and state that the hard term intentionally learns the teacher's choice. Keep this experiment default-off.

Test that changing only unsearched synthetic scores does not change this objective, invalid masks fail, and the hard reference remains unchanged. Report measured-action coverage and tie frequency. Do not change live search weights or tail ordering in this repair.

Add model, dataset, split, warm-start, source-code, vocabulary/encoding, prior-data, and search-configuration fingerprints to reports and checkpoint metadata. Record trained outputs so an untrained winning-chance output cannot be enabled by a filename or architecture flag. Save complete per-file collection provenance, not only the first file's seed/game count. Use atomic writes for checkpoints and large new datasets to avoid half-written files appearing complete.

**Done when:** Evaluation reports identify the exact tested bytes and decision settings; unknown action quality is not mislabeled as measured quality; no improvement claim is attached to the new objective before comparison.

## 5. Validate the teacher and run small diagnostic experiments

This stage resolves hypotheses; it does not assume a larger model or more data is needed.

Build a fixed tactical suite with independently justified expected outcomes: hidden-information pairs, forced switches, Protect, speed changes, target coordination, weather, setup, and harmful effects. Include cases where multiple actions are acceptable. Distinguish guaranteed outcomes from uncertain strategic preferences. Use actual Showdown outcomes for mechanics assertions, not the teacher's own score as its correctness proof.

Measure the corrected public teacher against frozen opponents, including the existing reference, and compare rule-only versus learned suggestions under fixed candidate budgets and separately under equal decision-time budgets. Report median, 95th-percentile, and worst decision times on recorded hardware. Choose the sample size and minimum meaningful effect before running, using variation across teams; short pilots estimate cost/variance, not promotion. When the evaluation harness changes, run identical-player comparisons and check that the uncertainty interval includes 50% and seat bias is absent—do not require the point estimate to equal exactly 50%.

On development data, inspect missed teacher choices by team family, turn phase, legal-action count, teacher margin, and searched coverage. Separate cases with ten or fewer actions, where retention in ten is automatic. Measure the cost of missed choices using comparable public-search evidence and eventual paired battles; teacher scores alone are not wins.

Test representation concerns through targeted fact changes and remove-one-input-branch comparisons. Compare fresh initialization with the existing replay warm start. Then compare hard versus searched-only teaching using the same data, split, budget, and three fixed random initializations. Change one factor at a time. Examine both training and validation performance to distinguish failure to fit examples from failure to generalize.

**Decision rule:** Teacher tactical failure -> repair the specific teacher defect first. Large training/validation gap -> investigate diversity and memorization. Failure even on training examples -> investigate objective/representation/optimization. Failures concentrated in missing situations -> collect those situations. No reproducible advantage -> retain the simpler reference and do not scale that intervention.

**Done when:** A short diagnosis names the demonstrated bottleneck, evidence against alternatives, next intervention, and a cost cap. Report inconclusive findings as inconclusive.

## 6. Train one justified candidate and evaluate without using the final exam for tuning

Freeze the selected recipe, data versions, comparison, stopping rule, and candidate-selection metric. If testing data volume, use nested whole-battle subsets of the same corrected training pool and keep validation fixed; do not compare different teachers/architectures as a single scaling curve.

Measure raw model and guided retention in ten, including difficult states and states reached by the player itself. Preserve the existing guided lower-bound requirement of 98%. A lower bound is the conservative edge of the estimated range after accounting for differences across teams. Preserve the strength margin of at most two percentage points worse than the reference, zero safety fallbacks, and meaningful speed improvement. Choose sufficient paired battles/teams before running; do not reduce requirements after seeing failure.

Reserve the final test for the frozen candidate. If it fails and informs another revision, it becomes development evidence; a later final claim needs a fresh test. Outcome-based training remains a separate future experiment with its own hypothesis; do not restart the failed 100k recipe as part of these repairs.

**Done when:** The candidate has current public-path development evidence supporting a final test. Failure leaves it experimental and produces a diagnosis, not an automatic larger run.

## 7. Enforce release approval and verify restoration

**Primary files:** `ladder/run_ladder.py`, model register, evaluation report writers, `tests/test_ladder.py`; proposed shared release validator under `src/vgc/`.

Make public ladder play require an approval record linking the exact checkpoint, relevant code/data/configuration fingerprints, public-information contract, and passing final retention/strength/timing reports. Validate the report's thresholds and required test size; a report made with weaker command-line thresholds cannot authorize release. Local evaluation of experimental models remains available with explicit experimental status.

Reject missing reports, swapped model bytes, incompatible encodings, untrained output usage, private-path reports, stale relevant settings, and failed/incomplete gates. Approval should not expire solely because unrelated documentation changed; identify the actual decision dependencies. Recheck mechanics/state/action readiness and upstream Showdown parity before release.

Restore the release bundle to a separate location, verify fingerprints and loading, and run a local smoke test with replays and zero fallbacks. Only then is the exact model/configuration pair eligible for public play. Do not mark an older incompatible model as the rollback target; if no approved version exists, stop public deployment.

**Done when:** A release can be reproduced and the ladder runner rejects an unapproved or altered candidate. Strong ladder-rating claims still require actual ladder evidence.

## Testing scope clarification (2026-09-07)

The user prioritizes iteration over exhaustive unit-test coverage. Test lists in this
plan identify critical risks, not a requirement to test every helper or permutation.
Use existing checks and small real runs wherever they provide sufficient evidence.
Add a focused test for a critical run failure or silent evidence error; debug lower-impact
issues as they arise. Do not let incidental test expansion block setting up the bot.

## Work sequence and reporting

Suggested review units: (1) status/register; (2) public-search parity; (3) shared dataset identity/audit; (4) both-role splitting and recovery manifest; (5) target semantics and evidence fingerprints; (6) diagnostic harness/report; (7) justified training and final evaluations; (8) enforced release/restore check. Stage 7 enforcement code can be built immediately after evidence schemas stabilize, before any model qualifies.

After each unit, update the handoff log with changed behavior, commands/results, remaining uncertainty, and next unit. Run focused tests for each change, then the affected readiness checks on a clean implementation checkout. No speculative duration or training-volume promise: the corrected public-search timing pilot sets compute estimates and the experiment cost cap before substantial work.
