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

## 2026-09-29 — TypeSafe/Jev review (PRs #7, #8) and decision-stage timing

- Reviewed PR #7 (Jev usage rules) and PR #8 (Jev "System One" hook in `score_joint_orders`), with a Codex/Sol second opinion.
- PR #7 merged with added project rules: no Jev in the per-turn scoring loop, public information only, pin/validate before it touches play, and a named saving for any speed claim.
- PR #8 closed. The hook fired several times per decision (both sides, per hypothesis, during distillation), fired even with tracing off, blocked up to 2s per call, marked `unknown_item` as known, and asked a classifier for strategy.
- Added `offline/measure_decision_stages.py`: exclusive per-stage timing of `choose_move`/`teampreview`. Default config vs heuristic, meta1: ~41-48 ms p50 per move. Rolling-horizon forecast ~58%, myopic evaluator ~22%, `resolve_exchange` ~9%. Team preview ~26 ms.
- Conclusion: our compute per turn is tens of ms, so a network classifier cannot speed per-turn decisions on this path.
- Hybrid path not timed: all registry "compatible" checkpoints fail to load (item vocab 150 vs 168 for M-C). They are M-B-era and need retraining.

## 2026-09-29 — format_id stamp on parsed replays (fail-closed)

- `tools/parse_replays.py` stamps `format_id` on every record (replay `formatid`, cross-checked vs dir name); exits 2 without writing `--out` on missing/disagreeing format or a mixed-format tree.
- `vgc.bc.dataset`: tracks `format_ids`, raises on partially/blank-labelled files; `check_format_mix` (called in `vgc.bc.train` when `extra_data` is used) refuses any unlabelled dataset in a mix, warns on labelled cross-format mixes. Self-play recorder stamps `Player.format`.
- Real M-C tree: 468/468 parsed, 8650 records all `gen9championsvgc2026regmc`. Planted M-B replay in an M-C dir -> exit 2, no output. Existing `data/bc/decisions.jsonl` predates the stamp: re-parse before mixing.
- M-C replay count corrected in CLAUDE.md/contract doc (468, not ~44). Codex review: fixed blank-label and non-object-JSON gaps. Tests: 185 passed (BC/replay/selfplay + new `tests/test_format_id_contract.py`). Not committed.

## 2026-09-29 — Jev parked; loss study without human labels

- Built `tools/build_loss_review.py` (plain-text hand-labelling sheet for the 23 M-C ladder losses, holdout of 8, bot guesses in a separate file, `--collect` validation). The owner is not a VGC expert, and AI-written labels can't serve as an answer key, so the Jev loss-tagging trial is parked. The status note is in CLAUDE.md/AGENTS.md.
- Added `tools/loss_patterns.py`: checkable replay facts, losses vs wins (46 M-C ladder games, 23/23).
  - The raw per-game "one of ours KO'd before acting" (22/23 vs 14/23) is inflated, because every loss has 4 faints.
  - Per knockout, the share is 70% in losses vs 53% in wins (p≈0.10, optimistic).
  - Directional hints only, none conclusive: speed pressure, opponent Tailwind (8 vs 3), Salamence in their four (9/23 vs 3/23), and our misses (13 vs 6).
  - Needs more games before any finding.
- Engine re-check of lost decisions: in progress (offline/review_lost_decisions.py).
