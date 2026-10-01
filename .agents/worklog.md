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

## 2026-09-29 — M-B checkpoints marked incompatible with M-C; vocabulary check

- Loaded every `data/models/registry.json` checkpoint via `load_snapshot`: train_a3, train_a3_24ep, train_a3_soft, train_b1, train_b1_soft fail (item embedding 150 rows vs 168 after the 2026-09-09 M-C export). Registry: now `loader.status: incompatible_vocabulary` with reason, date, `vocabulary_rows`; status experimental -> historical.
- Nothing read the registry before, and row counts alone miss a reordered vocabulary (Sol review). New `vgc.model_vocabulary` (torch-free): RL savers (opponents.save_snapshot, train_ppo, train_full_pipeline, train_fixed_mirror x2, train_imitation) write the ordered species/item/ability/move lists as `data_vocabulary`; `load_snapshot`, train_ppo resume, train_fixed_mirror resume + `--init-from`, and `model_release` refuse mismatched OR missing lists.
- BC: `load_bc_policy` built the net from saved sizes but encoded with today's indices, so bc_policy_v3sp/v4/v4_selfplay (1519 species, 150 items, 320 abilities) misread inputs if `use_bc_policy` was on. It now disables them (returns None, warns). `warm_start_state_encoder` skips embedding tables whose source vocabulary differs (moves still copy; they match) and prints which.
- `tests/test_model_registry.py`: compatible entries need `loader.vocabulary_sha256 == vocabulary_fingerprint(current_vocabulary())`; with files present, also file sha256 + real load. Plus reordered/missing-vocabulary/stale-BC cases. Fixtures in test_rl/test_ladder/test_model_release now record the vocabulary.
- Sol second pass: original findings resolved; fixed its three leftovers -- Q-model loaders (`offline/evaluate_q_vs_search_powered.py`, `offline/sweep_opponent_response_weight.py`) now require the vocabulary and `train_counterfactual_q` records it; synthetic registry tests skip cleanly without torch (9 passed / 13 skipped with torch blocked); the real-file registry test checks the file hash for every entry and matches the refusal reason.
- Full unit suite: 1103 passed, 2 failed (test_counterfactual_q needs gitignored data/selfplay/archetype_pool_150, absent in this worktree). With runs/ linked in, all registry tests pass. No retraining.

## 2026-09-29 — search no longer scores harmful status on our own ally as a gain

- Ladder 2678505187 T5 (`sleeppowder@-2 / flareblitz@2`, Venusaur slept our Incineroar) reproduced from the state replay. Root cause: `vgc.search._apply_action`'s utility branch credited `our_utility_value` for ANY sleep hit, whichever side it landed on (+18.75 = the whole margin over `sludgebomb@2`). The myopic `_score_parting_shot` was target-blind too: Parting Shot hit our own partner or an empty ally slot on 16/372 rebuilt ladder decisions (~40 recorded in traces).
- Fix: `vgc.principles.harms_ally_target` (foe-directed single-target status kinds; ally-ability exceptions incl. Guts/Flare Boost, Volt Absorb/Motor Drive, Contrary). Search resolves the final target at execution (foe Follow Me redirect, ally fainted mid-turn = fail) and signs the credit. Evaluator: `-PolicyConfig.ally_harmful_status_penalty` (35.0 = sleep_powder_weight) for harmful status on an alive ally, 0 for a fainted-ally slot. Enumeration unchanged (ally targets stay legal).
- Evidence: all 372 saved decisions rebuilt before/after: exactly the 16 ally-harmful picks change, nothing else. New `tests/test_ally_target_harm.py` (13; 9 of the original 10 fail on old code). Unit suite 1110 passed, 2 pre-existing counterfactual_q data failures. Action gate 19/19 tests; verdict BLOCKED only for the dirty tree. Codex review: 4 findings (redirect, fainted mid-turn, Contrary, fainted-slot redirect), all fixed + tested.
- Separate defect found (not fixed): opponent responses offered switches into fainted/unbrought mons, and Rillaboom has no M-B set priors, so every exchange was 0 that turn. Spawned as its own task. Not committed.

## 2026-09-29 — Jev parked; loss study without human labels

- Built `tools/build_loss_review.py` (plain-text hand-labelling sheet for the 23 M-C ladder losses, holdout of 8, bot guesses in a separate file, `--collect` validation). The owner is not a VGC expert, and AI-written labels can't serve as an answer key, so the Jev loss-tagging trial is parked. The status note is in CLAUDE.md/AGENTS.md.
- Added `tools/loss_patterns.py`: checkable replay facts, losses vs wins (46 M-C ladder games, 23/23).
  - The raw per-game "one of ours KO'd before acting" (22/23 vs 14/23) is inflated, because every loss has 4 faints.
  - Per knockout, the share is 70% in losses vs 53% in wins (p≈0.10, optimistic).
  - Directional hints only, none conclusive: speed pressure, opponent Tailwind (8 vs 3), Salamence in their four (9/23 vs 3/23), and our misses (13 vs 6).
  - Needs more games before any finding.
- Added `offline/review_lost_decisions.py`: exact-Showdown re-check of every lost-game decision at a wider budget. 195 decisions, live choice reproduced 99%, median regret 10, 21% over 100 points. The biggest regrets are mostly "Protect with one Pokemon left", a one-turn-horizon artifact. Found a real bug: game 2678505187 T5, our Venusaur used Sleep Powder on our own Incineroar (the search overturned the myopic pick). 9 decisions were skipped with `KeyError: THREEQUESTIONMARKS` ("???" type) in the mirror.

## 2026-09-29 — search opponent model: fainted/unbrought switches, no-prior species

- Game 2678505187 turn 5: foe had only Rillaboom left (bring-4 fully revealed). The search's only foe responses were "switch->incineroar" (fainted) and "switch->charizard" (unbrought). Rillaboom had no moves: no revealed moves, no M-B prior. It picked Sleep Powder on its OWN Incineroar (+18.75 = 25 utility x 0.75 accuracy credited as our gain).
- `vgc.search._opp_switch_pool` + `search_public_bench_filter` (on): faint status from opponent_team, bring-four once 4 are revealed, base-species match, revealed object preferred. `vgc.sets.learnset_fallback_move_ids` + `set_prior_learnset_fallback` (OFF, see CLAUDE.md for why).
- Turn 5 rebuilt: legacy reproduces the recorded self-sleep; shipped config picks Sludge Bomb + Flare Blitz into Rillaboom. The foe is still modeled as "pass", because the fallback is off.
- A/Bs: pool160 49.4% [0.474,0.515]; mc6 search-only 50.9%; mc6 A/A 50.2%; mc6 all-on coverage-fallback 40.0% (fallback-only 41.4%, mc_ladder_04 5.5%); STAB-only fallback 50.6% (-29..+19/team).
- Merged main (701a66c already fixed the self-sleep credit more fully: signed cost, redirects): dropped this branch's `search_sleep_credit_foes_only` knob and its test in favor of main's version. The A/B numbers above included this branch's sleep fix.
- Codex review: fixed accuracy:true scored as 1% (bool is int) and conditional moves (Steel Roller/Belch/Last Resort) in the fallback; also excluded self-KO moves.
- Tests: 3 regression tests in test_search.py; set_priors tests pin the fallback-off path. Unit suite 1100 passed, 2 known data-missing failures. Readiness gates blocked in this worktree: data/selfplay was absent and the tree was dirty. Not committed.

## 2026-09-30 — status-utility moves read set priors (built, A/B wash, OFF)

- `vgc.evaluator._opp_utility_move_ids` + `PolicyConfig.status_utility_uses_set_priors` (default False): Taunt/Encore (status-move count), Will-O-Wisp (physical share) and Wide Guard/Quick Guard/Mat Block (foe spread-move count) can read revealed + set-prior moves via `opponent_move_ids` instead of revealed-only.
- The `infer_hidden_opponent_sets` flag is uncommitted work in the team-preview-tagging worktree and is not on this branch, so this uses its own flag.
- 160-team pool A/B (36 games/team): 2862/5760 = 49.7%, cluster-robust [0.484, 0.509], FAIL -> ships False. Post hoc: 57 teams carrying the moves 51.6% [0.494, 0.538]; 103 unaffected teams 48.6% [0.471, 0.501] (effectively A/A, shows the noise).
- Unit suite: 1112 passed, 2 known data-missing failures (counterfactual_q split files absent in the worktree). The worktree `.venv` is a symlink to the main checkout's venv, so gate-script tests can spawn it.

## 2026-09-30 — M-C replay corpus survey (read-only, nothing downloaded)

- On disk: 468 M-C replays, all uploaded on 2026-09-09 (launch day); 438 rated >=1100, 71 >=1200, 9 >=1300.
- Public listing (search.json paged to the end, 1231 pages): 62,730 M-C replays 2026-09-09..09-30, ~2,500-3,700/day, none private. Rated >=1100: 33,351; >=1200: 18,469; >=1300: 9,556; >=1400: 2,928; >=1500: 680.
- Bot-heavy accounts: pcrlbot12d159c39a (2322 games >=1100), Scorecard-Pokemon (1931), SC-SME (1659), SC-Control (1522), plus our own laplacestheorems. Excluding them: >=1200 13,995; >=1300 7,309.
- `tools/download_replays.py` default `--max-pages 100` reaches only ~1.5 days back at this volume; a full backfill needs ~1300 pages and has no player/bot filter.

## 2026-09-30 — M-C corpus backfill + set priors rebuilt (uncommitted)

- `tools/download_replays.py --exclude-player` (repeatable, matched by Showdown user id). Downloaded 13,977 new M-C replays rated >=1200 back to launch, 0 failures; tree now 14,445 (113 MB). Skipped 11,645 bot-account listings.
- Parsed to `data/bc/decisions_regmc.jsonl`: 14,445/14,445 replays, 277,636 records (199,658 turn / 49,088 forced switch / 28,890 preview), all format_id-stamped. M-B `decisions.jsonl` untouched.
- `data/usage/set_priors.json` rebuilt from M-C (min rating 1200, 14,048 games, 272 species). Preview-slot coverage at set_prior_min_games=5: M-B file 77.5% -> M-C 99.9%.
- Not run: the unit suite (auto-mode classifier denied it with PYTHONPATH=src; the symlinked main venv imports main's src without it). The new priors change live play (search opponent moves) and need a same-session A/B before merge. The 09-29 learnset-fallback A/Bs moved win rate, so the pool harness does exercise priors (unlike the mutual-OTS run_gates gates).

## 2026-10-01 — M-C priors A/B, real-team pool, two DirectBattle crash fixes

- `PolicyConfig.set_priors_file` + `vgc.sets.set_priors_for` (all bot-side prior loads); M-B file kept as `data/usage/set_priors_regmb.json`. Battle-state gate `set_priors_sha256` re-pinned. Rain preview test pinned to M-B data (M-C bring frequencies change its prediction).
- A/B 160-team pool: 2794/5760 = 48.5% [0.455, 0.515], tau 0.177; A/A 49.7%.
- `tools/build_ladder_team_pool.py` -> `data/selfplay/mc_sheet_pool` (298 real M-C sheets, 0 validation failures; Stat Points filled from M-B spreads.json).
- A/B real pool: 2782/5364 = 51.9% [0.490, 0.547], tau 0.229; A/A 48.9%. Post hoc by count of M-B-missing species 0/1/2+: 50.1/51.8/52.6%, overlapping. Wash; M-C stays default as a data refresh.
- Real pool crashed DirectBattle twice: (1) hidden-trap `[Unavailable choice]` (Mega Gengar Shadow Tag) -> recover in live games, fail closed in `evaluate_exact_branches`, clones copy `_waiting` (Codex found the last two); (2) poke-env KeyError on the Round-chain `[from] move: Round` line -> `vgc.poke_env_compat.normalize_for_poke_env` in DirectBattle and VgcPlayer.
- Tests: unit 1117 passed (2 known counterfactual_q data failures). Integration test_rl_env: new trap test passes; 2 pre-existing failures (old regmb format id in the pump test; worker batch error) spun off as a separate task.
- Worktree now has its own `.venv` (uv sync --extra dev --extra train); the symlinked main venv imported main's src.

## 2026-09-30 — Showdown re-pinned to 89905975e; automatic sync before ladder play

- Parity was BLOCKED: 7 upstream commits behind a5df8274e, 3 of them Champions rule fixes (Encore action override moved into onStart, Curse + Follow Me targeting, Sheer Force no longer suppressing Berserk/Pickpocket/Eject Button, Mega Sol + Electro Shot). Pulled, force-built, re-exported. Only catalog change: `condition.onOverrideAction: encore` removed (Encore already covered by the move-execution family). Re-pinned; all three gates PASS (128/57/19 tests).
- New `vgc.showdown_sync` + `tools/sync_showdown.py`: parity check -> ff-merge the exact ref parity fetched -> `node build --force` -> export -> stop-and-revert if the catalog changed beyond generated_from commit/path (`--accept-catalog-changes` after review) -> re-pin -> commit -> three gates. Restores data/champions on any failure; reruns gates if parity holds but a certificate is stale. `ladder/run_ladder.py` runs it before public play and re-execs itself after a sync (`--no-sync-showdown` opts out).
- Verified end to end on throwaway copies (Showdown clone reset to a5df8274e + project worktree at 52c3d98): first run blocked on the Encore change with a clean tree, second run with accept committed identical data. Codex review: fixed partial-data-on-failure, pulling from `origin` instead of the ref parity checked, and retry-after-gate-failure reporting success.
- Training entry points still only check the saved gate certificates, not live parity.
- Second Codex (Sol) review of the final branch: Encore catalog acceptance confirmed safe (4 upstream Encore/Prankster cases + bot Encore checks pass); gate-retry fix confirmed. Fixed: local master ahead of public master could be pinned (now requires HEAD == fetched upstream ref; verified on a throwaway clone), and rollback after a failed commit left staged files (now restores from HEAD; verified with a forced pre-commit failure). Open, owner's call: the catalog diff only sees handler names, so behaviour changes inside existing handlers (e.g. the Sheer Force fix) are auto-accepted once the gates pass.
- First real automatic sync: upstream bebf328c6 (Statmon format, config/formats.ts only) -> synced, catalog content unchanged, three gates PASS, committed 9b77271.

## 2026-10-01 — GitHub Actions CI + weekly Showdown drift watcher

- Added `.github/workflows/ci.yml` (unit: ruff + pytest; engine: pinned Showdown build, seeded pool rebuild, static gates, `-m integration` with a zero-skip guard) and `showdown-watch.yml` (weekly parity check -> one drift issue).
- `VGC_SHOWDOWN_REPO` env override in `vgc.config`; `vgc.rl.env.DEFAULT_SHOWDOWN_REPO` now reads it (it was a second hardcoded copy -- Codex caught that the first rehearsal had silently used the drifted local checkout through it).
- Stale tests fixed: rl_env batch test (worker now always returns `error: null`), parse-equivalence test (hardcoded `regmb` battle tag), catalog ground-truth (ignore machine path, still compare commit). Skip-if-missing for gitignored pools in counterfactual_q and the teacher-label contract test.
- Local Showdown checkout is 10 commits past the pin (`a5df8274e` -> `bebf328c6`), parity BLOCKED; against it 11/149 integration tests failed. Against a clean pinned build: 149/149 pass, 0 skipped, unit 1112 passed. Upstream has 3 Champions rule fixes since the pin (Mega Sol/Electro Shot/Encore, Sheer Force, Curse+Follow Me) -- re-pin is a separate task.
- Verified: full CI sequence rehearsed locally with the home Showdown path disabled; actionlint clean; watcher drift + network-failure paths rehearsed with stubbed `gh`; Linux CPU-torch resolution dry-run via `uv pip compile`. Not yet run on GitHub itself.
- Merged main (Showdown re-pinned to bebf328c6 + `tools/sync_showdown.py`). Re-rehearsed CI on the merged tree against bebf328c6 with the home path disabled: unit 1116 passed, gates PASS, integration 149 passed / 0 skipped. Public master is 11 commits past the new pin (none in data/mods/champions), so the watcher's first run will open a drift issue; its text now points at `tools/sync_showdown.py`.

## 2026-10-01 — test_rl_env follow-up (merged after the CI work)

- Main's CI PR already fixed the same two stale tests. Kept main's fixes; this branch adds only a `requestState == "move"` assertion to the batch test (proves the good battle advanced past team preview), explanatory comments, and a note on the always-present `error` field in `tools/sim_worker.mjs`'s header.

## 2026-10-01 — Rulebook audit + relevance-aware Showdown parity

- Audit: data/champions is a byte-identical re-export of the pinned Showdown (bebf328c6); static gates PASS; parity BLOCKED by 5 upstream commits from 2026-10-01.
- `vgc.showdown_relevance` + `tools/parse_showdown_entries.mjs`: a missing upstream commit now blocks only if it cannot be proven irrelevant to Reg M-C. Cleared: config/formats.ts edits to other formats only, and data/aliases.ts nickname edits not touching legal species/moves/items/abilities, rules or our format. Both files are parsed with the checkout's own TypeScript (comments, escapes, templates cannot disguise changes). Changed entries with load-time code, spread/accessor keys, duplicate names, missing mods block. Dex-table blocks name the changed entries and which are in Reg M-C. Everything else (sim, lib, mod, rulesets, tags, dex tables) always blocks.
- Watch list gained data/tags.ts (Mythical/Restricted Legendary bans), data/aliases.ts, data/formats-data.ts, lib, config/custom-formats.ts -- tags.ts was a real gap in the old checker.
- On 2026-10-01's upstream: 3 commits cleared (October rotation, NDBH, Deltamon removal), 2 block (Supreme Overlord onEnd; Baneful Bunker status source).
- Two Codex (Sol) review rounds: 6 false negatives in the first (line-based) version, 5 more adversarial ones in the second; all closed with regression tests (`tests/test_showdown_parity.py`, integration-marked, need a Showdown checkout with node_modules/typescript).

## 2026-10-01 — Showdown parity: formats check runs Showdown's own loader

- Codex found changed neighbour formats that parse fine but make Showdown throw while loading the shared format list (name `"!!!"`, `mod: null`, `maxLevel: 100`, a bare undefined identifier) were cleared as irrelevant.
- Fix: after the static checks pass, `vgc.showdown_relevance` compiles the commit's `config/formats.ts` + `data/aliases.ts` with the checkout's esbuild/tsconfig, swaps them into the pinned BUILT checkout's `require.cache`, and runs `Dex.formats.all()` plus the server's `formatListText` getter (taken verbatim from `dist/server/rooms.js`). Any error blocks. Load step runs under Node `--permission` with an empty env (damage limitation only; network open).
- Second Codex round on the first version found 5 more: section object stringification, stale (pinned) aliases, tsc-vs-esbuild class-field semantics, hidden formats blocking spuriously, inherited env credentials. All fixed; regression tests for each in `tests/test_showdown_parity.py` (34 pass). Real upstream rotation commits (6 most recent) still clear, ~1.4s each.
- Third Codex round: mod existence came from the pinned build, not the commit. Loader now uses the commit's data/mods folders (deleted mod under an unchanged format blocks; a new mod + its format clears, its rule table checked against base gen9 data). 36 tests pass.
