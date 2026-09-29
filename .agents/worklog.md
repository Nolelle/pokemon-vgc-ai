# Project handoff log

## 2026-09-09 — documentation pass for Regulation M-C

- Operator docs, contracts, and tool docstrings now state the live format is Reg M-C
  (`gen9championsvgc2026regmc`), with the 2026-09-09 data export counts and Showdown pin
  `efe494857`.
- Dual replay corpus is documented: keep M-B (~2940) as historical warm-start/priors;
  download new rated M-C games incrementally; do not rebuild `set_priors.json` or
  `spreads.json` from the small M-C snapshot.
- `AGENTS.md` is now a pointer to `CLAUDE.md`. Dated snapshots
  (`project_audit_2026-09-07.md`, `rl_roadmap.md`) are marked historical. Preview
  prediction is marked shipped. No replay files, priors, or models were changed.

## 2026-09-09 — replay action and outcome label repair

- User approved separating unknown actions, observed actions, and no-action-required slots; match wins remain the objective, not avoiding every faint.
- Replay schema 5 retains observed Mega Evolution when a move never appears. Missing/unnamed blocked moves no longer become teachable pass labels.
- Explicit win/loss/draw/unresolved outcomes replace ambiguous boolean-only supervision. Legacy pass and outcome labels are excluded from the relevant teaching; raw logs must be reparsed for outcome training. Parsed records retain outcomes even when action slots are skipped.
- Sol made initial changes, then reached its usage limit. Root reviewed, fixed blocked-Mega/status handling and legacy wins-only bias, and completed checks.
- Validation: 142 focused parser/BC/recording/policy tests passed; Ruff and git diff --check passed. Fixed 20-replay comparison preserved all 355 records/states and winners; 102 old pass slots became 75 unknown and 27 no-action-required slots, all excluded from action/target teaching. Remaining 382 action labels unchanged. Two-replay CLI smoke passed.
- Contract: docs/replay_label_contract.md. Sample IDs/hashes and checks: runs/replay_label_audit_20260909/report.json.
- No historical dataset overwritten, model trained, ladder played, or commit made. Own private sets, exact submitted targets, and full joint-choice reconstruction remain separate limitations; this is not a complete replay-to-exact-training conversion.

## 2026-09-07 — audit repair implementation in progress

- Starting commit: 46d2560. Existing audit and implementation-plan documents were untracked.
- User requested root orchestration with a Sol medium sub-agent implementing the plan.
- Root owns status/register documentation and independent review; Sol owns public-search parity first.
- Added data/models/registry.json: 14 current/historical files, no public release approval.
- Corrected current model status and marked older pipeline instructions as historical.
- No model training, public ladder games, deletions, or promotion performed.
- Audit/plan: docs/project_audit_2026-09-07.md; docs/audit_implementation_plan_2026-09-07.md.
- Next: review shared public search and its tests, then enforce consistent dataset identity and both-role split audits before training.

## 2026-09-07 — implementation status before project-level review

- Sol hit a usage limit after public-search edits. Root reviewed and continued them. A retry was requested; no active sub-agent remained at status check.
- Public search now uses one reconstruction entry point; private-set invariance passed from both seats (2 integration tests).
- Shared source identities now survive relocation/merging. The trainer calls the full split audit before fitting and supports explicit separate validation files plus --audit-only.
- Recovery audit PASS: runs/audit_recovery_20260907/training_audit.json; 41,543 B1 train / 1,637 A3 development validation, zero battle or both-role team overlap. Source code equivalence across differing collection commits was verified from tracked source trees.
- Experimental soft teaching now adds a searched-only relative term; hard copying stays the default. Unsearched synthetic scores have no influence on the relative term.
- Added content evidence helpers and public --model-release validation with negative tests. No model has approval. Evidence producers/bundle workflow still need completion and review.
- Unit suite: 1,061 passed, 149 integration tests deselected. Focused release/ladder tests: 46 passed. Full integration/readiness validation after all changes remains outstanding.
- All implementation changes remain uncommitted. No new model trained, public ladder game played, or old data deleted.
- Remaining: additional audit-enforcement edge cases, metadata/report producer integration, full integration checks, diagnostic experiments, frozen final test, justified candidate training, release bundle restoration. Do not call the implementation plan complete.
- User asked to step back for an app/project status review.

## 2026-09-07 — user-requested commit and scope confirmation

- User requested committing the current work and focusing on data/training before coaching app work.
- Goal remains 1700+ Elo on the public Champions Reg M-B ladder. Actual ladder results must establish it; teacher retention and offline wins are intermediate evidence only.
- README, audit implementation plan, and historical RL roadmap now state this priority. No coaching UI work is planned before the playing goal is demonstrated.
- Pre-commit validation: 1,064 unit tests passed; 149 integration tests deselected. Changed Python files passed Ruff; git diff --check passed. The earlier two public-search seat integration tests passed.
- This commit preserves the foundation repairs in progress. Full integration/readiness checks, additional audit edge cases, evidence report integration, diagnostics, justified candidate training, and final release proof remain outstanding.
- No training run, public ladder session, model promotion, or artifact deletion performed.

## 2026-09-09 — Champions Regulation M-C database update and alignment

- Updated format identifier in `src/vgc/config.py` and `src/vgc/rl/env.py` to `gen9championsvgc2026regmc`.
- Exported Champions Reg M-C data from local Showdown checkout (`tools/export_champions_data.py`): 390 legal species (+35), 166 legal items (+18), 509 legal moves (+13), 222 legal abilities (+14).
- Updated `tools/export_mechanics_catalog.mjs` to target `gen9championsvgc2026regmc`; re-exported `data/champions/mechanics_catalog.json` pinned to Showdown HEAD (`efe494857`).
- Updated `catalog_sha256` in `data/champions/mechanics_coverage.json`.
- Updated `tests/test_data_export.py` (Rocky Helmet is now legal) and `tests/test_position_effects.py` (Octolock is now legal with Grapploct).
- Verified parity and static gates: `check_showdown_parity.py` (PASS), `check_mechanics_readiness.py` (PASS), `check_battle_state_readiness.py` (PASS), `check_action_readiness.py` (PASS), and `test_mechanics_catalog_ground_truth.py` (PASS).
- Verified replay downloading: fetched initial batch of 43 rated Reg M-C replays to `data/replays/gen9championsvgc2026regmc/`.


## 2026-09-29 — branch consolidation and Showdown re-pin

- PRs #5 (M-C ladder teams) and #6 (wandb imitation + re-pin) had merged into `docs/align-regulation-mc`, not `main`; opened a PR from that branch to `main`.
- Rebased unpushed local commit (aa6d5f085 data refresh) onto origin; most of the old uncommitted tree was byte-identical to #5/#6 and was dropped. A stash "backup before consolidation 2026-09-29" holds the pre-consolidation tree.
- Committed leftovers: wandb hooks in train_fixed_mirror/train_full_pipeline/train_ppo + tests/test_wandb_run.py; tools/visualize_training_data.py (lint-cleaned, `plot` verified).
- Brought over from the mc-hybrid worktree: disk-space preflight (e4bb2d1) and the clean pin-drift acceptance in `validate_source_compatibility` (+ test). Its aa17ca0fa re-pin was superseded.
- Showdown moved from detached aa17ca0fa to master a5df8274e (Curse/Emergency Exit Champions fixes); re-exported data + catalog, re-pinned sha. All four readiness gates PASS.
- Unit suite: 1079 passed, 1 failed — `test_distillation_improves_teacher_agreement_on_held_out_games` (untrained model already scores 1.0; fails on main too, pre-existing).
- The mc-hybrid worktree is now fully superseded by this branch. No training or ladder play performed.
- Merging into main exposed two parallel W&B layers (main #4 `vgc.wandb_logging` vs branch #6 `vgc.rl.wandb_run`, rebuilt because the branch lacked #4). Kept main's; deleted `vgc.rl.wandb_run` + test; imitation now streams per-epoch metrics via distill's `on_epoch` hook. The earlier PPO wandb-hook commit on this branch is superseded by main's wiring.

## 2026-09-29 — M-B checkpoints marked incompatible with M-C; vocabulary check

- Loaded every `data/models/registry.json` checkpoint via `load_snapshot`: train_a3, train_a3_24ep, train_a3_soft, train_b1, train_b1_soft fail (item embedding 150 rows vs 168 after the 2026-09-09 M-C export). Registry: now `loader.status: incompatible_vocabulary` with reason, date, `vocabulary_rows`; status experimental -> historical.
- Nothing read the registry before, and row counts alone miss a reordered vocabulary (Sol review). New `vgc.model_vocabulary` (torch-free): RL savers (opponents.save_snapshot, train_ppo, train_full_pipeline, train_fixed_mirror x2, train_imitation) write the ordered species/item/ability/move lists as `data_vocabulary`; `load_snapshot`, train_ppo resume, train_fixed_mirror resume + `--init-from`, and `model_release` refuse mismatched OR missing lists.
- BC: `load_bc_policy` built the net from saved sizes but encoded with today's indices, so bc_policy_v3sp/v4/v4_selfplay (1519 species, 150 items, 320 abilities) misread inputs if `use_bc_policy` was on. It now disables them (returns None, warns). `warm_start_state_encoder` skips embedding tables whose source vocabulary differs (moves still copy; they match) and prints which.
- `tests/test_model_registry.py`: compatible entries need `loader.vocabulary_sha256 == vocabulary_fingerprint(current_vocabulary())`; with files present, also file sha256 + real load. Plus reordered/missing-vocabulary/stale-BC cases. Fixtures in test_rl/test_ladder/test_model_release now record the vocabulary.
- Sol second pass: original findings resolved; fixed its three leftovers -- Q-model loaders (`offline/evaluate_q_vs_search_powered.py`, `offline/sweep_opponent_response_weight.py`) now require the vocabulary and `train_counterfactual_q` records it; synthetic registry tests skip cleanly without torch (9 passed / 13 skipped with torch blocked); the real-file registry test checks the file hash for every entry and matches the refusal reason.
- Full unit suite: 1103 passed, 2 failed (test_counterfactual_q needs gitignored data/selfplay/archetype_pool_150, absent in this worktree). With runs/ linked in, all registry tests pass. No retraining.
