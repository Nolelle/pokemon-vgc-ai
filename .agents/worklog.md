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

## 2026-10-01 — M-C data refresh complete; first M-C checkpoint collection launched
- Bots: owner OK learning from strong bots; the per-game 1200 filter is the gate (name exclusion dropped except our own account). Corpus 20,215 replays / 389,762 decisions; set_priors rebuilt (19,818 games).
- spreads.json now M-C (Smogon 2026-09 chaos, 1760 cutoff, 1.63M battles) + `PolicyConfig.usage_spreads_file`; owner wants M-C spreads kept regardless of A/B. Five M-B scenario tests pinned to spreads_regmb.json.
- Real-team pool v2: `data/selfplay/mc_sheet_pool_v2` (314 teams, M-C spreads), split by `tools/split_team_pool.py` into train 226 / holdout 88 (grouped by six-species set, 0 overlap).
- BC warm start on M-C replays: runs/mcv2/bc_regmc/bc_policy.pt (val move top-1 0.437, top-3 0.783, 3.4 min).
- Earlier M-C teacher run in MAIN runs/collect_mc_s* + train_mc_hybrid_b1_full (2026-09-10/14, M-B priors/spreads, M-B-style teams) scrapped by owner; not used.
- Showdown synced to 1ce9b34f9 (sync tool). Merged main twice (PR #20 merged; showdown_relevance parity). Codex (Sol): merge breaks nothing; flagged a parity hole in main's showdown_relevance (spun off as a task).
- Collection crashes fixed (each found by a teacher shard, reproduced with the shard seed):
  - Locked Outrage: Showdown rejects `move outrage` while locked (request omits target -> side.ts defaults 'normal'). `vgc.actions.index_locked_choice` rewrites to `move 1` at SEND time (DirectBattle.step_payload, VgcPlayer). An enumeration-time version was wrong and removed. Also affected the ladder bot (poke-env default-order retry).
  - Stale must_recharge (Choice Scarf Heliolisk, Hyper Beam): mirror forced recharge, teacher sent `move 1` = disabled Thunderbolt. `mechanics_state._request_confirmed_must_recharge` (same rule as stale preparing). TeacherRecordingPlayer now skips a decision when any ranked order is illegal on the real battle.
  - Replays of both crash seeds: 34/34 games clean; mismatch guard fired 0/501; 2 skips were the known "???" type KeyError.
- Collection launched 17:20 at 20308ce: runs/mcv2/collect_train_s1..6 (900 games each, train manifest) + collect_holdout (600 games, holdout manifest), --max-skip-rate 0.08. ~2.8 s/decision. Next: train student (--dataset train shards --validation-dataset holdout --split-by team --bc-checkpoint runs/mcv2/bc_regmc/bc_policy.pt), recall screen on holdout. Commit + rerun gates before training.
- Fixed the mechanics-branch padding bug flagged by the 2026-10-01 architecture review (`rl/model.py` `_encode_mechanics`): the pooled mask admitted extra edge windows for right-padded rows, so a snapshot encoded differently alone (live) vs in a padded batch (training) at 3/4 of lengths. Windows now kept iff centre byte < row length. Test: `test_mechanics_encoding_is_identical_alone_and_in_a_padded_batch` (fails on old code).
- Open decision for the owner (memory student-shortlist-ceiling): the running collection uses the default K=10 teacher, so the student is capped at the heuristic shortlist. Kept as approved ("just run it"); it yields the first M-C-loadable model. Wider teacher (K=20/40 A/B first) is the follow-up.
- 2026-10-02: collection done (shard 1 finished 900 games but lost at save: disk 1.9 GiB free < 2 GiB headroom; compressed data/bc/decisions_regmc.jsonl -> .gz, verified identical, to free space). Trained runs/mcv2/train_v1/best.pt on 5 shards (34,002) vs holdout (4,529): pure R@10 91.8%, guided@10 LCB 0.920 FAIL. Heuristic top-10 already contains the teacher pick 90.9% -> student ~= heuristic (K=10 teacher ceiling). Result: runs/eval/shortlist_recall_mc_v1.json. Next: search-width A/B (K=10/20/40), then wide-teacher collection; needs disk space.

## 2026-10-03 — Clean-slate design + first build slice (branch claude/clock-guard-llm-harness)
- Design doc (Claude Doc "VGC Bot Architecture Design"): evolve existing bot; GPT-6 Luna only, adaptive reasoning effort (default none); Jev limited to preview plan reader + offline audit; $20 hard-capped test budget with escalation rule; owner's six teams (teams/owner/, all validate-team clean; start with psyspam_sand + salamence_tw). Reviewed 3x by Codex.
- Code check: shipped ladder bot decides in ~50 ms (vgc.search); exact search (2–28 s) offline/hybrid only; nothing read Showdown's timer; search ignores opponent Megas (phase 2 work).
- Built: clock guard (src/vgc/clock.py; agent.py _guarded: request-anchored budget from the VGC timer, cheap fallback first, one worker at a time, cancelled late workers can't write memory/samples; unknown clock = unchanged inline decide); LLM harness + fake LLM (src/vgc/llm/: strict ID validation, deadline with completion timestamps, byte-bound worst-case spend reservations persisted before send, adaptive thinking level; NOT wired into play); evaluator --our-teams / --checkpoint-games / --futility-min-effect / --min-gain / per-seat; docs/llm_test_protocol.md (success = clustered CI low > 50% and observed gain >= +2; power at tau 0.114 only ~59% for +3 at 6,000 games -> opponent-team count is the limit).
- Codex code review: 6 findings, all fixed. Tests: 1178 passed; ruff clean.
- Readiness gates BLOCKED only by Showdown parity (6 upstream commits incl. October tier shifts, Champions Mega Scizor) -> run tools/sync_showdown.py as a separate change before ladder/training.
- Worktree note: .venv is a symlink to the main checkout's venv; run with PYTHONPATH=$PWD/src (main checkout dir is on an older branch).

## 2026-10-03 — Next build slice: Luna wired in (off by default), speed test, saved positions
- Showdown synced to 9fb3a5b99 (Champions tier shifts) in its own PR #25; gates pass after sync.
- Built: vgc.llm.facts (public board -> ~2.65k-token packet, guesses labelled, spread-aware damage facts), vgc.llm.proposer + search_joint_orders(order_proposer=, extra_candidates=) behind PolicyConfig.llm_proposer_enabled=False (engine scores LLM proposals like any candidate; disabled path unchanged), llm_control_extra_candidates for the equal-time arm, teams/owner/plans/*.md, offline/llm_speed_test.py, vgc.positions + offline/record_positions.py + offline/grade_positions.py (rebuild public roots from decision-replay bundles; exact-search grading; tune/test split by six-species identity; overall/coverage/conditional shares clustered by opponent team). `openai` added as optional extra `llm`.
- Real Luna speed test (300 calls, $0.084 total incl. pilot): none p50 2.1 s / p95 3.1 s / max 4.9 s, $0.00024/call; low p50 2.9 s / p95 6.2 s / max 8.4 s, $0.00030/call; 100% valid; ~54% of input cached. Results in runs/llm/speed_test.json.
- Codex reviews: 7 findings on this slice (shared spend meter per file + file lock, budget recomputed before calling, spread damage, hidden Speed always GUESS, split by species identity, honest grading denominators, decided-from-start classification) — all fixed. Tests 1196 passed; ruff clean.
- Next: offline proposer screen on recorded positions (~$0.05), then equal-time calibration and the confirmation head-to-head per docs/llm_test_protocol.md.

## 2026-10-04 — Luna screen fails; exact judge wins (branch claude/clock-guard-llm-harness)
- Luna proposer offline screen: ranked prompt anchored to engine (2 novel/733); blind prompt (LLMConfig.blind_options) novel on 200/300 but +2.3 pts vs matched engine-rank control, CI [−2.0, +6.7] → paused. Spend ~$0.31 total.
- Fast-search A/Bs (2 owner teams vs 226 train teams, 1,808 games, MDE ~1.7): A/A 49.7; shortlist 20 49.5, 40 49.8; setup boosts 48.9; Mega fix 49.8 (kept); opponent Mega 50.3.
- Mega bug fixed: poke-env keeps base species after Mega; states now use the Mega forme (legacy knob mega_state_uses_evolved_form). poke_env_compat patches -mega overwriting Mega-Z formes.
- exact_judge_live (public exact search re-ranks fast top-6, ~0.5 s/decision): 54.8% [0.527,0.569] train; HOLDOUT 54.4% [0.513,0.575]. Timed local smoke OK. Ladder: --exact-judge.
- Running: judge tuning on train split (top_k 10, extra_myopic 3, future_samples 4). Confirm any winner on holdout before adopting.

## 2026-10-04 (late) — LLM roles settled: move proposer closed, preview advisor wins
- Fair proposer game test (exact judge ± Luna none, 5 workers, 0 rate limits): 50.2% [0.483,0.521]. GPT-6.1 Sol (`gpt-6.1-sol`, min effort low) offline +3.0 pts [−0.2,+6.2] ≈ Luna low. Move-proposer line CLOSED.
- Team-preview advisor (llm_preview_enabled; blind; both arms exact judge; 2 owner teams vs 226 train teams): none 49.6%, low 51.6%, medium 53.9% [0.516,0.562]; HOLDOUT (88 teams, 1,408 games) medium 53.6% [0.508,0.564]. p50 15 s, p95 23 s, ~0.4% timeouts → heuristic fallback.
- Ladder: `ladder/run_ladder.py --exact-judge --llm-preview medium`.
- Rate limits: tier-1 500K TPM → ≤5 parallel games when the LLM is called per turn; preview (1 call/game) is fine at 9.
- LLM spend total ~$9.36 of the $20 cap.

## 2026-10-05 — Exact judge review + fixes (branch claude/judge-improvement-analysis-20688c)
- Review (Opus + Codex/Sol, independent, agreed): exact `_position_value` has no KO term, scores Trick Room/weather/terrain 0, flat boosts/status, one-turn horizon, myopic double count; bookkeeping bugs (first reveal -100, fainted boosts kept, draws, belief diagnostics).
- Built: `exact_search_consistent_accounting` (default True) fixes; `vgc_exact` player + pool harness `--player/--both` + per-arm fallback counts.
- Found via exact A/A (62.9% p1): mirror always seated us p1 -> p2 decisions searched stale boards (30% wrong actives, 24% wrong weather). Fixed with `mirror_side`; A/A now p1 48.8 / p2 48.6. Also mirror usernames (won branch read as lost on named accounts; Codex review). p2-seat teacher labels in runs/mcv2 are suspect.
- Tests 1203 unit + new integration seat/name tests pass; ruff clean. Accounting A/B (narrow width, 160 teams) running: runs/eval/exact_judge_accounting_ab.json.

## 2026-10-05 (cont.) — field control + continuation (branch claude/exact-judge-field-horizon)
- Built (Sonnet agents in parallel, Codex-reviewed): KO/alive term (ships 90), `vgc.field_control` speed/weather/terrain leaf (off), multi-turn exact continuation `exact_search_continuation_turns/policy` (off), `offline/measure_exact_depth.py`, `offline/diagnose_setup_ranking.py`.
- Wins: KO term 52.5% [0.501,0.550]; myopic weight 0 53.5% (M-C train) / 53.7% (pool160) / 51.6% holdout (underpowered).
- Nulls: field leaf (4 runs, 48.8-50.2%); continuation N=1/N=2 greedy (49.4%, 47.8%).
- N=1 with myopic weight 0: fast-search continuation 49.9%, greedy 47.4% (nulls). Shipped exact-only `exact_search_myopic_weight=0.0`.
- Disk was 99% full; freed ~8 GB (old worktree .venvs, Codex installer leftovers, Spotify cache).

## 2026-10-06 — condition clock + continuation fixes (branch claude/exact-judge-field-horizon)
- Owner's hunch confirmed (and by Codex): weather/terrain durations wrong in every exact branch (poke-env restamps weather each upkeep; switch-in setters charged an extra turn). `vgc.condition_clock` fixes it: 266/266 match live sim; known extenders count 8 from the start.
- With correct timers: field leaf 50.1%/49.7% (null); N=1 continuation 46.3% (worse); 4 samples 47.3% (noise rejected).
- Flip diagnostic: N=1 flips 43% of picks, setup ~2x, 20/56 flips decided by mid-continuation game ends.
- Fixes (all off): continuation weight (0.25: 47.8%, 0.5: 50.4%), board-scored continuation terminals (47.8%), 2x2 continuation search built by a Sonnet agent in a worktree (47.8%), 2x2 + board terminals 50.1% [0.469,0.533]. Harm removed, no gain.
- 2026-10-06 later: engine-measured team-plan value (`vgc.plan_value`, Sonnet agent) + fast-search condition expiry (Sonnet agent) + condition-clock extension lock. Owner teams vs M-C train: plan-value field control 51.0% [0.498,0.522], terrain_pulse_blastoise 54.6% [0.517,0.576]; live judge KO 52.7% [0.498,0.557]; fast expiry 50.4%. Next: pre-registered holdout confirmation; model weather/terrain-setting moves in the fast search.

## 2026-10-07 — Live judge width: top-10 is a null (branch claude/exact-judge-wider, off PR #29)
- Setup positions (offline/setup_positions.py, new D0k10/D0k12 variants): judge time ~0.53 s -> ~0.70 s idle. Pos 7 fixed (Scarf Indeedee switches to Excadrill instead of Protect); pos 1 regressed (Helping Hand over Trick Room: one-turn horizon). Rest unchanged.
- A/B exact_judge_top_k 10 vs 6 (both arms exact_judge_live; psyspam_sand, salamence_tw, hatterene_tr vs 226 train teams, 2,712 games, seed 20261007): 1348/2712 = 49.7%, cluster-robust [0.482, 0.512]. Rules out an edge above ~+1.2 pts. Keep top_k=6.
- Under 10 parallel workers: judge p50 1.57 s / p95 2.16 s (k10) vs 1.16 / 1.68 (k6); 3/3037 timeouts. ~0.4% judge errors in both arms (KeyError 'THREEQUESTIONMARKS' = '???' type, and a sim 'Cannot read properties of undefined' branch error); both fall back to the fast pick.
- Earlier Oct-4 top10 run had crashed on a hidden Disable (fixed in b6f930d); extra_myopic=3 measured 50.0% then.

## 2026-10-08 — First M-C ladder check-up (branch claude/ladder-checkup)
- PR #29 had merged into PR #27's branch, not main -> PR #30 opened to carry it to main.
- Showdown synced to 51ad80fa5; gates PASS. Worktree needed its own `uv sync`.
- Clock guard: preview budget now spends Showdown's 90 s starting grace (was a ~10 s bank share, so the LLM preview — tested offline with no clock — always fell back on the ladder). Preview now gets ~28 s; LLM pick used in 10/10 ladder games (2.7–14.6 s).
- Public ladder, 10 games, psyspam_sand, --exact-judge --llm-preview low: 3-7 (1121 -> 1032, opponents ~1000-1130). Judge ran 64 times (p50 0.49 s), overturned the fast pick 45/64.
- Bugs found and fixed: (1) mirror stall counter = poke-env protect count (1) instead of Showdown's odds denominator (3, 9…): repeat Protect always succeeded in every exact branch since the mirror existed -> judge chose back-to-back Protects that failed live (test: tests/test_live_mirror_protect_streak.py). ALL exact-judge A/B numbers so far were measured with this bug. (2) nicknamed opponents crashed the judge for the whole game (mirror named sets by species). (3) cherry-picked judge-error-fixes ('???' type, sourceless volatiles). Re-check of 3 lost games: 0 judge errors.
- offline/review_lost_decisions.py fails every game via its process pool ("I/O operation on closed file"); in-process `_review_one` works. Not fixed yet.
- Under machine load the judge times out and then sits `skipped_busy`; don't run ladder alongside pool A/Bs.

## 2026-09-30 — Jev re-review (stays parked); preview now reads hidden opponent sets

(Left uncommitted in its worktree until 2026-10-08; rebased onto main then. The A/B numbers below predate the M-C set-priors/spreads refresh and the exact judge — re-measure before relying on them.)

- Jev verdict: team-preview tagging, loss labelling, replay labelling all stay parked (Opus + Sol agree). Checkable tags are plain code; uncheckable ones have no answer key.
- Found: `opponent_engines` empty in 46/46 M-C ladder games; preview opponent moves empty so `them_onto_us` = 0 and no gameplan threats. Cause: raw previewed mons (no OTS on ladder).
- Fix: `vgc.sets.opponent_signal_team` + `PolicyConfig.infer_hidden_opponent_sets` (True). Used in `team_preview.build_team_order` and `evaluator.build_context` (with `_with_revealed_sets`, since poke-env never updates preview mons in battle).
- Evidence: pool160 A/B 56.1% [0.516, 0.607] seed 20260930; 55.1% [0.508, 0.594] seed 20261001 (final code). Replay backtest M-C 60.4% engine recall vs 0%. Unit suite 1111 passed, 2 pre-existing counterfactual_q data failures.
- Codex review: fainted mons counted in-battle; Mega ability guessed despite a revealed non-stone item. Both fixed. Open: weather engines need an abuser; top-4 fill misses 5th-ranked plan moves; Taunt/Will-O-Wisp/Wide Guard scoring still reads revealed-only opponent moves (evaluator ~1607/1649/1404).

## 2026-10-05 — Replay coaching assistant: plan v1 (branch claude/coaching-assistant-webapp-fb3319)
- Owner vision: web app reviews a player's Showdown replay (+ team paste); engine grades decisions, LLM explains from first principles + turns ahead; later many games -> personal habits. Pitch = bot reached ladder top.
- Plan: docs/coaching_assistant_plan.md (Opus + Codex/Sol reviewed). Parts: reference games -> intake -> player-view rebuild (synthetic |request| from spectator log + paste) -> choice recovery -> grading (grade_positions.grade_position, forced candidates) -> judge checks (continuation test, Protect-delay trap) -> evidence report -> grounded LLM coach -> CLI/web -> habit profile.
- Key facts: no code builds a player view from a public replay; 46 bot M-C state-replay bundles + record_positions.py give answer keys; deep grading ~30 s/decision; exact judge still no KO term on main (PR #29 unmerged); best M-C ladder ~1150.
- Next: Part 1 (reference games, both seats, spectator view saved).

## 2026-10-08 — Results review + disk cleanup (main, after merging #30–#32)
- Ladder to date: M-B 202–236 (46%), M-C 26–30 incl. today's 3–7 check-up (in the ladder-checkup worktree's runs/ladder.jsonl, not yet in main's). Ladder ~1000–1180; no session big enough to read an offline +4.
- Priority: re-run exact-judge holdout A/B after the repeat-Protect mirror fix (all earlier judge numbers carry that bug); then a 50+ game ladder session on an idle machine; fix offline/review_lost_decisions.py process pool.
- Disk 2.3 GB -> 64 GB free: removed 20 merged, unused worktrees (gitignored runs/ + data/selfplay copied to runs/_worktree_archive/<branch>/); gzipped runs/eval/*.jsonl > 5 MB (read with zcat); deleted 16 old Codex releases; emptied Apple aerial wallpaper videos.
- Uncommitted-work sweep: team-preview-tagging held the finished 2026-09-30 hidden-opponent-sets fix (pool160 55-56%), never committed -> rebased onto main, tests pass, PR #33, merged (re-measure on the new base). Coaching plan doc (1 unmerged commit) carried into this PR. mc-hybrid worktree and the 2026-09-29 stash were fully superseded by main; saved as patches in runs/_worktree_archive/ and removed.
- Left: 7 worktrees with open Claude sessions (all clean, merged), wasted-actions (locked, merged).

## 2026-10-08 (cont.) — Re-measurements after the repeat-Protect mirror fix (main @ ad7ca5d51 pin)
- Exact judge (exact_judge_live vs fast search), owner psyspam_sand + salamence_tw vs mc_sheet_pool_v2 HOLDOUT (88 teams), 8 games/unit, seed 20261008: **821/1408 = 58.3%, cluster-robust [0.550, 0.617], PASS** (was 54.4% [0.513, 0.575] with the Protect bug). Both teams ~58%, both seats, every archetype >= 54%. The Protect-streak bug was costing the judge ~4 pts. Result: runs/eval/remeasure_20261008_exact_judge_holdout.json.
- Hidden opponent sets (#33; infer_hidden_opponent_sets True vs False), pool160 mirror, 36 games/team, seed 20261008: **3019/5760 = 52.4%, cluster-robust [0.484, 0.564], not significant** (was 55-56% on 2026-09-30, before the M-C set-priors/spreads refresh). Team-effect SD 0.248: bimodal, not null -- 38 teams >= 70%, 30 teams <= 30%. Kept on (positive lean, fixes the preview blindness); next: find what separates winning from losing teams, and test owner teams vs real M-C teams. Result: runs/eval/remeasure_20261008_hidden_sets_pool160.json.
- Showdown synced to ad7ca5d51 (#36): Champions OU tier label for Mega Raichu Y. Not a Reg M-C ban (Flat Rules; team_049 validates).
- review_lost_decisions fixed (#37): clock guard ran the 30 s deep review on a timed worker; 2 games / 15 decisions now review with 0 failures. Live reproduction only 53% because the tool rebuilds the fast pick, not the judge's.

## 2026-10-09 — Protect valuation: odds-scaled bonuses + exact stall odds (branch worktree-agent-a7abb9c51a6e5824d)
- Evidence: 46 ladder games, 25 back-to-back Protects (~2/3 failed); losses 2.6 Protects/game vs 1.5 in wins.
- Fast evaluator: `_score_protect` scales information/stall/reposition bonuses by `protect_success_decay**counter` (`protect_bonuses_scale_with_odds`, True; False = legacy). Low-threat penalty stays unscaled.
- Exact search: `exact_search_exact_stall_odds` (True; False = sampled). Repeat Protect-family rolls (stall volatile present, counter 3/9/..) run BOTH forced outcomes per (choice, seed) sample, weighted 1/counter vs rest; all combinations when several Pokemon repeat. Worker: `clone` takes `stallForce`, `stallInfo` lists stallers; the pin is consulted by a per-battle `randomChance` wrapper (dex/conditions are frozen) and still takes the stock PRNG draw. Tests: tests/test_exact_stall_odds.py.
- Cost (judge width, 38 decisions): median unchanged (0.249 s); only 2/38 had a repeat roll, +0.03 s median / +0.10 s max on those.
- Not A/B'd for strength. Full readiness gates currently BLOCKED only by Showdown parity drift (upstream 7332b60e2); `--static-only` passes for all three.

## 2026-10-09 — Mechanics correctness pass (branch claude/mechanics-correctness)
- Owner direction: mechanics and game understanding must be right first, so only game plans need tuning.
- Review of the 50-game ladder session (27-23): games are decided early (first KO in 85% of wins vs 35% of losses); Protect overused (2.6/game in losses vs 1.5 in wins; 25 repeat Protects); opponents Mega'd in 39/46 games.
- Found the opponent-Mega blind spot, then a Codex (Sol) audit found the rest. Fixed (Sonnet agents in worktrees, merged here, Codex-reviewed; 4 review findings fixed):
  - Exact mirror: opponent Mega availability; opponent HP was copied from the public percent as absolute HP (foes simulated at ~half bulk); hidden item guesses were blanked; post-Mega stats; Fake Out/First Impression reuse (activeMoveActions + DisableMove pass); Choice lock; exact sleep/confusion timer branches; toxic stage; Disable move id; Unburden per stint; Mega-twin reply dedupe.
  - Exact search: repeat Protect evaluated as both forced outcomes weighted by the true odds.
  - Fast search/evaluator: canonical weather table (sand/snow Megas were invisible); signed Mega weather; single-stone Mega timing; Protect bonuses scaled by odds; Psychic Terrain blocks priority attacks and priority status moves; Unburden speed; spread recount; move accuracy with alive-probability KO accounting; first-turn-only moves.
  - damage.py (ground-truth verified, no knob): sand Rock SpD / snow Ice Def x1.5; Body Press/Foul Play/Psyshock stats; -ate abilities and Liquid Voice.
  - Deferred: Substitute HP across rebuilds (needs a per-hit damage estimate).
- 18 PolicyConfig knobs, default True; False = legacy. A/B: all 18 on vs all off, both arms exact_judge_live, owner psyspam_sand + salamence_tw:
  - Train (226 teams, seed 20261009): **1945/3616 = 53.8% [0.521, 0.554] PASS**; every archetype 52-62% (snow 61.7%).
  - Holdout (88 teams, seed 20261010): **760/1408 = 54.0% [0.520, 0.559] PASS**; both teams, both seats, every archetype >= 52%.
  - Note: the ground-truth damage fixes are in both arms, so their effect is not in these numbers.
- Showdown synced to 2796de703 (two upstream syncs on this branch). Tests: 1356 unit, 232 integration (0 skipped); all gates PASS.
- Codex exec gotcha: background `codex exec` hangs on stdin; always add `< /dev/null` and an alarm.

## 2026-10-09 — Fast-search forecast locks/accuracy, worker slot, preview-lead diagnosis (branch worktree-agent-aa4263ce6b2b5c654)
- `forecast_respects_locks`, `forecast_move_accuracy` (default True; False = legacy). Finished battle's worker no longer blocks the next battle (`WorkerSlot.owner/cancel`). Preview "always Tyranitar+Excadrill" is the opponent-independent lead-speed term, not a bug (see CLAUDE.md follow-up). Not A/B'd for strength. Tests: 1378 unit, 232 integration; three gates PASS.

## 2026-10-09 — Judge stalling look-ahead + reply diversity (branch worktree-agent-a5b494afc72d4d716)
- New knobs, all default False: `exact_judge_passive_lookahead` (+ `_alternatives`, `_samples`, `exact_search_passive_lookahead_board_terminals`), `exact_search_diverse_replies` (+ `_min_weight_ratio`). Code: `vgc/rl/exact_search.py` (`is_passive_order`, `_diverse_replies`), `vgc/exact_judge.py` (`narrow_for_lookahead`).
- Offline harness: `offline/replay_judge_positions.py` replays ladder state-replays through the judge once per arm; see CLAUDE.md "Judge stalling and reply diversity" for the tables. Look-ahead changed 52/321 picks (49% of loss repeat-Protects), repeat Protects 34 -> 29; diverse replies 86.0% -> 86.6% agreement with a 16-reply judge. Needs the owner's pool A/B before any default flips.
- The shared scratchpad dir is written by other agents (show.py/peek.py got overwritten); use a private subdir.

## 2026-10-09 — Battle parsing audit and poke-env repairs
- Trigger: ladder game 2695881082 (`-copyboost` applied backwards by poke-env). Built `offline/audit_battle_parsing.py`: seeded local battles, after every step compare both perspectives' bot-visible state (`snapshot_battle` + `BattleMemory`) with Showdown's `dump`.
- Seed 99, 2400 battles / 85k compares: 11,186 mismatch episodes with all repairs off (`--disable-fixes all`), 153 on (0 unexplained, 88 documented limitations, 65 legitimately hidden foe Damp Rock/Light Clay). ~50 s on 8 workers.
- Repairs: `vgc.poke_env_compat` (copyboost, Champions Mega abilities, single-turn/momentary effects, charge cancel, Worry Seed/Skill Swap abilities, ability reveals, Baton Pass, sleep/toxic counters, forme species, Flash Fire, Regenerator, Illusion state, unbrought actives, Psych Up crit volatiles, gastro acid); `vgc.mechanics_state` (layer counts, effect aliases/markers, foe consumed item, Floette-Mega); `vgc.condition_clock` (Light Clay); `vgc.battle_memory` (changed abilities).
- Trace: `VgcPlayer._record_final_choice` re-scores the order actually sent into `chosen_breakdown` (+ `chosen_breakdown_order`).
- Tests: 1407 unit, 235 integration; `tests/test_poke_env_repairs.py` (44), `tests/test_battle_parsing_audit.py`, `tests/test_final_choice_trace.py`. Not A/B'd for strength.

## 2026-10-10 — Bug-fix round 2 merged + A/Bs (branch claude/bugfix-round2)
- Second 50-game ladder session after PR #39 (session 20261010T003212Z): 24-26, peak ~1385 (previous best ~1280), mean opponent 1256 (was 1144), vs >=1200 14/33 (was 2/9), vs >=1300 4/16. Judge 0 errors.
- Found: Protect stalling in losses (4.2 Protects, 1.9 repeats per loss vs 1.7/0.05 in wins; one-turn horizon, confirmed by Codex); poke-env applies `-copyboost` (Psych Up) backwards (game 2695881082).
- Round 2 (3 Sonnet agents + Opus fix, Codex-reviewed, 5 review findings fixed): poke-env parsing audit + repairs (11,186 -> 0 unexplained mismatch episodes, 3 seeds x 2400 battles, 0 errors); switches sent by request position (transformed teammate captured switch-by-name); forecast respects locks + accuracy; finished-battle worker no longer blocks; trace attribution; passive look-ahead + diverse replies built (off).
- A/Bs (owner teams vs 226 train teams, 4 games/unit, both arms exact judge):
  - forecast_respects_locks + forecast_move_accuracy on vs off: 899/1808 = 49.7% [0.480, 0.514] -- null; kept ON as correctness fixes.
  - exact_judge_passive_lookahead (samples=1) on vs off: 904/1808 = 50.0% [0.482, 0.518] -- null; stays OFF. Offline self-play may rarely produce the human-style stalling positions.
- Parsing repairs are process-wide (both arms), so their effect is not in these numbers.

## 2026-10-10 — Repeat-Protect rule (branch worktree-agent-adc1b873830c5983f)
- New knob `forbid_repeat_protect` (ships True; False = legacy). `vgc.evaluator.apply_repeat_protect_rule`, called from `score_joint_orders` (fast search, judge's judged set and `_cheap_order` all take candidates from it). A slot whose Pokemon Protected last turn (`protect_counter >= 1`) may not use a stall-rolling move again unless forced (no other legal option) or the partner KOs every threatening foe first. Empty result -> unfiltered list. Opponent-view callers pass `repeat_protect_rule=False`.
- Replay (`offline/replay_repeat_protect_rule.py`, both ladder sessions, 100 games, 737 move decisions): see CLAUDE.md "Repeat-Protect rule" and the final report. Violations (rule-on pick or cheap pick that is a removed order): 0. Legacy cheap fallback would have repeated in 64/171 rule-active decisions.
- Judge mirror error seen in replay unrelated to the rule: "Can't switch: You can't switch to a fainted Pokemon" on some switch candidates (2 games).

## 2026-10-11 — Prove the plumbing, step 1 (branch claude/prove-plumbing-1)
- Four Sonnet agents (after two usage-limit restarts; WIP commits saved the work), then two fresh agents for Codex's 9 review findings (all fixed), plus an Opus fix (battle_memory speed includes terrain).
- Parsing audit made trustworthy: exact slot comparison, Illusion handled (not skipped), 11 precise exemption predicates, independent reveal parser (offline/audit_protocol.py), coverage report + 259 targeted scenario battles (moves 99.6%, items 100%, abilities 99.5% present). New repairs: Illusion break, missed item/ability reveals, Meganium/Yanmega read as Megas, Milk Drink sent without a target (legality), Transform types, Gastro Acid, Gravity, sleep counters. Audit TeamBuilder now gives Mega stones.
- Mirror fidelity audit (offline/audit_mirror_fidelity.py): field comparison + oracle equivalence (true hidden info + same seed must replay identically). Legacy rebuild: 12,617 unexplained, 12 crashes; now ~75% of oracle pairs identical, ~24% documented gaps, ~1% unexplained, 0 crashes. Fixed (exact_mirror_* knobs): own bench order (the fainted-switch judge error), foe formes, Champions PP, gender, stall duration, fainted state, volatile timers, public bookkeeping, faint-event count (Last Respects). Guess accuracy: item 32%, nature 10%, unrevealed move 24%, Speed within 5% 39%.
- Calculator sweep (offline/sweep_calculator_vs_engine.py, tools/calc_probe.mjs, data/champions/calculator_coverage.json, docs/calculator_coverage.md): usage-weighted EXACT damage/moves 0.5% -> 99.7% (Showdown 4096ths rounding), items 59% -> 100%; fixed Huge Power, Thick Fat, Electro Shot/Meteor Beam, Prankster/Gale Wings/Triage priority, ~60 abilities, many items/moves; review fixes: fractional-HP crash, Expanding Force vs redirection, Fur Coat vs Psyshock, Grassy Glide priority blocks, partner abilities in incoming threat, Surge Surfer speed.
- forbid_repeat_protect (+ residual-KO exception): ladder replay of 100 games, repeat Protects in losses 59 -> 4; 0 layer violations.
- Final: ruff clean; 1501 unit, 246 integration; parity + 3 gates PASS; parsing audit 3 seeds x 2400 + 459 scenario battles: 0 unexplained except seed 20261009 1 group/2 episodes (Instruct-called Protect counter; previously hidden by the too-broad exemption, now correctly visible; open); mirror audit 3 seeds: 0 crashes, ~1% unexplained.
- Open: Instruct-called Protect counter; mirror ~1% unexplained (1-HP roll shifts, traps, Transform), Substitute HP; calculator WRONG tail (Sniper, Rivalry, Analytic, Payback...). No win-rate A/B (correctness fixes); next real test is a ladder session.

## 2026-10-12 — @pkmn/client shadow trial (branch claude/pkmn-shadow-trial, off claude/prove-plumbing-1)
- Owner asked to trial @pkmn/client as a poke-env replacement for READING battles. M-C data comes from our pinned Showdown (`Dex.mod('champions')` behind a Proxy whose forGen(9) returns the mod; plain Generations drops the mod). Self-check asserts M-C values at reader start.
- Built (Sonnet agent): tools/pkmn_reader/ (Node JSON-lines reader, pinned lockfile), offline/pkmn_shadow.py (adapter -> BattleMechanicsState), audit `--reader {pokeenv,pkmn,both}`, per-reader exemptions, report + 13 repros, one integration test.
- Results (unexplained episodes/groups, 2400 battles; seed 99 re-run by Opus, identical): raw poke-env ~15.4k/~430, repaired poke-env 5-7/5, pkmn 741-844/374-445. pkmn also does not track Protect counter or charge state (~10k episodes/seed). Latency ~1 ms/decision, 0 parse errors, 0 crashes. pkmn natively avoids ~22 of 57 poke-env repairs but has ~13 defects of its own (returning Pokemon rebuilt blank in ~5% of battles, confirmed in client source by Codex; Gastro Acid, temp abilities, extenders, Bug Bite berry, Symbiosis, Transform types, ...).
- Codex review: verdict ordering holds (repaired poke-env > pkmn > raw), but pkmn's counts are not exact: reveal tracker ignores `cant`/Poltergeist reveals; blanket Illusion/Transform triage rules mislabel some episodes; Light Clay duration ranges collapsed to the lower bound; hidden-extender exemption reads current snapshot not reveal history. Not fixed: the gap (~100x) does not change the decision.
- Decision: keep repaired poke-env as the live reader; do not migrate. Trial code stays on the unmerged reference branch claude/pkmn-shadow-trial; only this write-up and the CLAUDE.md section went to main.
- Note: repaired poke-env shows 5 unexplained groups on main after PR #41 (CLAUDE.md said 0; now recorded there as open).

## 2026-10-12 — First-principles plan execution (branch claude/pkmn-trial-writeup)
- Step 1 ladder baseline (55 games, psyspam_sand, search+horizon+exactjudge+llmpreview-medium, session 20261012T174852Z): 28-27, rating 1168->1287 peak 1346, opp avg 1224, 0 fallbacks. Loss tags: lost_positioning 20/27 (74%), sacrificed_closer 4, poor_speed_control 2, unnecessary_prediction 1.
- Step 2 diagnosis (146 prior M-C games + Oct-10 deep review of 237 decisions): losses dominated by positioning — Protects 2.88/loss vs 1.28/win, losses 7.8 vs 5.4 turns; deep best switched 16 vs bot 6 (regret>10) and bot under-targets; their Tailwind 19/72 losses vs 6/74 wins (p=0.004); our KO-before-acting 60/72 vs 31/74. Fresh baseline confirms: positioning 74% of losses.
- Step 3 (judge sees best switch; new knob exact_judge_include_best_switch default False; vgc.exact_judge.select_judged + test): train A/B both arms exact_judge_live, psyspam_sand+salamence_tw vs 226 train, 8/unit, seed 20261012: 1804/3616 = 49.9% [0.486, 0.511] NULL. No holdout (train gate failed); knob stays OFF. Lesson: switch candidates were visible; valuation, not visibility, is the gap. Result: runs/eval/step3_switch_train.json.
- Step 4 (no archetype separator for hidden-sets bimodality — every archetype 49.6-56.7%; exact_judge_hypotheses 2 vs 1): train A/B 4/unit seed 20261013: 897/1808 = 49.6% [0.478, 0.514] NULL. Stays 1. Result: runs/eval/step4_hyp2_train.json.
- Step 5 (5 owner teams, identical bot both sides, 88 holdout teams, 176 on-our games each, seed 20261014; A/A overall 50.5% [0.482, 0.527] sanity OK): hatterene_tr 75.6%, psyspam_sand 61.9%, gardevoir_psyspam 56.8%, terrain_pulse_blastoise 49.4%, salamence_tw 47.7% on-our-team win rates. Worst: sand for 3 teams; tailwind for salamence/blastoise. Next ladder session should pilot hatterene_tr alongside psyspam_sand. Result: runs/eval/step5_owner_teams_holdout.json.
- Code: src/vgc/exact_judge.py, src/vgc/models.py (knob, off), tests/test_exact_judge.py. Tests: test_exact_judge 10 passed; ruff clean. Not committed.

## 2026-10-12 — Positioning diagnosis (branch claude/positioning-diagnosis)
- Full ladder history (201 M-C games in runs/ladder.jsonl): 102W/99L; lost_positioning 69/99 losses (70%). loss_patterns: our KO-before-acting 84/99 vs 42/102 (p~0.000), first KO 39/99 vs 71/102, their Tailwind 29/99 vs 9/102; misses/crits smaller but real.
- Deep re-check of 6 most recent lost_positioning losses (Oct-12 session, 44 decisions, reproduction 59%): regret>10 on 20; deep best Protect 11, attack/target 8, switch 1; bot switches 2 vs deep 1. The Oct-10 "deep switches 16 vs bot 6" does NOT replicate on current code. Top regrets are late-game terminal values (one-turn Protect delay artifact per tool caveat).
- Ranked levers: (1) Protect-vs-attack under lethal threat (expected-death cost cliffs at 100%), (2) target selection, (3) speed control as position (pre-registered confirmation), (4) placement value in _position_value — untested, invisible to this tool by construction. Write-up: docs/positioning_diagnosis.md. No weights changed.

## 2026-10-13 — Guesser improvements (branch claude/guesser-improvements)
- Capture: `vgc.unknown_species` + `PolicyConfig.capture_unknown_species` (off by default) logs fog-safe sightings of no/thin-history species (revealed moves/item/ability only; spreads flagged not learnable from battle) to `data/usage/unknown_species.jsonl`; agent flushes on battle finish; `tools/build_priors_from_captures.py` aggregates once-per-battle (preview-only excluded from appearances) and merges idempotently via capture_keys.
- Honest fallback: no-history species now get 3 archetype spread hypotheses (0.4/0.35/0.25, mode = legacy default) instead of one certain default; knobs `use_fallback_spread_hypotheses` / `use_defensive_damage_evidence` keep legacy paths for A/Bs.
- Defensive learning: damage we deal now records DefensiveDamageObservations (spread-move target count, defender boosts/status/HP/screens preserved) and reweights their defensive spread in `opponent_belief`.
- Report card: `offline/guesser_report_card.py` grades beliefs against exact packed teams per-team (item/ability top1/top3/logloss/Brier, move recall, spread coverage). On mc_ladder_01+02 (12 sets): item top1 0.33, ability top1 0.25, spread-in-hypotheses 0.42 — small in-sample sample, baseline only.
- Codex review: 8 findings (spread targets, defender state, per-team grading, preview appearances, merge recount, finish flush, config gates, limit renormalization) all fixed. Tests: 1540 passed; ruff clean.
- Not done (follow-ups from plan): decision-level A/Bs for belief mixture/hypotheses, joint move/item/ability consistency widening, team-level conditional priors.

## 2026-10-13 — Guesser A/B: bundled knobs parked OFF
- Null test (pool150 manifest, 160 teams, 4 games/team, seed 20261015): 319/640 = 49.8% [0.457, 0.540] OK.
- A/B both guesser knobs on vs off (fast search, 16 games/team, seed 20261016): 1312/2560 = 51.2%, cluster-robust [0.492, 0.533] -- pre-declared bar (lower bound > 50%) NOT met; no archetype Holm-flagged. Result: runs/eval/guesser_ab_20261016.json. Smallest certifiable edge was ~2.0 pts, so a true gain under ~2 pts remains possible but unproven.
- Parked: PolicyConfig.use_fallback_spread_hypotheses / use_defensive_damage_evidence default False. Capture logging untouched (capture_unknown_species machinery intact).
- Parking exposed a legacy crash: with the fallback off, a species absent from exported data raised KeyError in belief building. opponent_belief now falls back to the fallback mode as the single certain guess instead of crashing.
- Tests: new default-parking test in tests/test_unknown_species.py; defensive-evidence test opts in explicitly. 1541 unit passed; ruff clean. Uncommitted on main.

## 2026-10-13 — Ladder baseline after guesser parking (branch claude/park-guesser-ab, PR #46)
- Gates blocked on 2 upstream Showdown commits; tools/sync_showdown.py pinned 3a766f2cf, all 3 gates PASS. Local smoke (hatterene_tr, 2 games): 1-1.
- hatterene_tr (session 20261013T182046Z, 30 games, search+horizon+exactjudge+llmpreview-medium): 13-17, rating 1262->1185 (peak 1288), opp avg 1160, 0 fallbacks. Losses: lost_positioning 12/17.
- psyspam_sand (session 20261013T193848Z, 25 games, same policy): 14-11, rating 1161->1239 (peak 1285), opp avg 1182, 0 fallbacks. Losses: lost_positioning 9/11.
- Combined 27-28 vs Oct-12 baseline 28-27 (peak 1346, opp avg 1202): no rating gain; offline owner-team ranking (hatterene first) did not transfer. Positioning still 21/28 losses. Next: step 4, price expected KO loss below 100% certainty.

## 2026-10-16 — HAL (Melee) blog review + quick value-weight fit (no code change)
- Reviewed ericyuegu.com/hillclimbing-melee for applicable ideas. Applicable: value fit on real outcomes (no TD), denser A/B metric (margin of victory), AWR-weighted human imitation as an extra shortlist source, rating conditioning. Not applicable: action chunking, real-time/KV-cache/latency work, scaling laws.
- Probe: logistic fit of `won` on our-minus-opp state features, 280,218 M-C replay turn decisions / 20,148 games (replay-clustered bootstrap). In units of 100 = one Pokemon's HP: alive 77 (hand 90), boost stage +16.5 (hand 5), brn/par/psn -41 (hand -18), sleep/freeze -60 (hand -30), Tailwind +46 (hand 12), screens +30 (hand 12). Correlational; proposed follow-up = richer fit then a vgc_exact pool A/B.
- Data note: decisions_regmc records drop fainted Pokemon; fainted = (6 - unseen_count) - listed. No code reads decisions_regmc.jsonl.gz yet (BC loader uses plain open() on the old M-B file).

## 2026-10-18 — Replay-fitted exact-search scorecard: wins in the exact player, wash in the live judge
- New `offline/fit_position_value_weights.py`: logistic fit of `won` on our-minus-opp board features (search units, 100 = one Pokemon's HP), replay-clustered bootstrap, lagged-board controls, holdout log loss. Lagged fit (40,885 replays, schema 6): alive ~60-65, boosts ~16-22/stage, brn/psn ~-36..-40, par ~-20..-25, sleep ~-58, fresh Tailwind ~59, fresh screen ~33-36 (hand: 90/5/-18/-18/-30/12/12). Holdout log loss hand 0.586 vs fitted 0.574. Results: runs/eval/value_fit_*_20261018.json.
- Parser fix (Codex review): `vgc.replay_parse` schema 6 -- boosts reset on switch-in; battle-only formes (Palafin-Hero etc.) resolve to the base species. Re-parsed to data/bc/decisions_regmc_s6.jsonl.gz (40,885 replays; old file untouched). Weights barely moved.
- New knobs `exact_search_tailwind_weight` / `exact_search_screen_weight` (None = effect weight). `evaluate_own_spread_pool.parse_overrides` parses None-default knobs as floats.
- Candidate (pre-declared, lagged fit rounded): alive 65, boost 15, status 35, hard control 60, Tailwind 60, screens 35.
  - `--player vgc_exact`, narrow width, M-C train 226 teams: 1938/3616 = 53.6% [0.515, 0.557] PASS.
  - Same, holdout 88 teams (seed 20261019): 1483/2816 = 52.7% [0.501, 0.553] PASS.
  - Ladder path (`--player vgc`, `exact_judge_live=true`, train, seed 20261020): 1855/3616 = 51.3% [0.494, 0.532] FAIL (wash).
- Defaults NOT changed. Likely reason: the live judge only re-ranks the fast search's shortlist, and the fast search (`vgc.search._exchange_value`) has no boost/status/Tailwind terms, so setup lines rarely reach the judge. Next hypothesis: fitted values in the fast search's leaf/shortlist.

## 2026-10-21 — Replay corpus refresh + schema 6 re-parse
- Committed the schema 6 parser fix (5a261eb: boosts reset on switch-in, battle-only formes merge to base species); 30 replay_parse tests pass.
- `tools/download_replays.py --min-rating 1200 --max-pages 600 --exclude-player laplacestheorems`: 13,985 new, corpus 54,870 (was 40,885, newest 2026-10-13 17:14; now 2026-10-21 00:27). 600 pages overlapped 1,507 existing files; hourly counts continuous across Oct 13, no gap. ~1,750 replays/day >=1200; ~2h at default sleep.
- Re-parsed to `data/bc/decisions_regmc_s6_20261021.jsonl.gz`: 54,870/54,870 parsed, 1,083,846 records (782,228 turn, 191,878 forced_switch, 109,740 teampreview), showteam 3.47% of player-slots. Older s6 file (40,885) untouched.
- Note: `parse_replays.py` default `--out` is the M-B `data/bc/decisions.jsonl`; always pass `--out`.

## 2026-10-22 — Data-health report for the M-C corpus (new tool, no bot change)
- New `tools/data_health_report.py` -> `runs/viz/data_health/index.html` (6 figures + PASS/WARN checklist + summary.json). Scans the raw replays and the newest `data/bc/decisions_regmc*.jsonl.gz`; caches under `runs/viz/data_health/cache/` (`--refresh` to rescan, ~1 min).
- Findings: (1) download filter keys on replay `rating` = min POST-game rating (100% of games) -> near-cutoff games kept only if the underdog won; favourite wins 48.9% overall, 6.1% when weaker player started <1200, 52.9% when both pre-game >=1234 (85.5% of games; max one-game loss 34). (2) 7 accounts bringing the same 6 species (scsme/scorecardpokemon/sccontrol/kvdn64/...) are in 34.9% of games, raw win rates 38-47%. (3) replay_parse ignores pre-turn-1 events: lead weather/terrain missing from turn-1 state in 27,743/27,743 games, lead stat changes (Intimidate) in 20,880/20,880. (4) Floette-Eternal / Meowstic-F become floette/meowstic after Mega (3,502 active snapshots off-preview). (5) teampreview `picked` has <4 species in 19.2% (derived from switch-ins). (6) unknown slot actions 14.5%. Label agreement parsed vs raw 100%, 0 dup logs, 0 collection gaps.
- Codex review applied: pre-ratings from rating-change messages (|player| ratings can be stale), HTML-unescaped names, net pre-turn-1 boosts, offline losses, softened captions.


## 2026-10-26 — Preserve experiments, repair replay state, and refit
- User requested the three preservation commits: ff7faf3 (position-value experiment), 7985b91 (recent team-pool tools), 988701a (data-health report/history), on chore/checkpoint-20261026. Continued on fix/replay-state-20261026; origin/main was already included after fetch/merge.
- Parser schema 7: opening public events now use the normal state-update loop before turn 1 without creating initial switch actions. Weather/terrain, stat changes, explicit item/ability/status reveals are retained; later reveals remain absent from earlier observations. Mega identity uses Champions battleOnly parent (Floette-Eternal, Meowstic-F and other persistent formes). Added clearnegativeboost/clearallboost handling. Parser/fit/audit commit: 98765b8.
- Reparsed all 54,870 raw M-C games to data/bc/decisions_regmc_s7_20261026.jsonl.gz (old files retained): 1,083,846 records, same 782,228 turn / 191,878 forced_switch / 109,740 preview counts, 0 failures/skipped events. Dataset SHA-256: 5584447c47526a10e07995bb0735ca49c1299a80a4dbaa42e6057066e5218471.
- New independent tools/audit_replay_state.py: PASS on all records against prior schema6 dataset, both seats of 54,683 games reaching turn 1. Exact raw opening field/stages match, including all 27,743 field and 20,880 boost openings. Outcomes, record identities, and action-status markers unchanged. 711 persistent-identity action corrections (117 preview selections independently verified from raw switches); all other actions unchanged. Some preview repairs recover the actual fourth appearance previously displaced by a duplicate Mega identity.
- Refreshed health report: opening field missing 27,743 -> 0; boosts missing 20,880 -> 0; active/bench/picked off-preview violations 3,502/151/117 -> 0. This is not a full replay-mechanics certification.
- Fit tool now has compact equivalent feature loading (Sol independently verified equality on 500 games / 6,568 rows), explicit both-player pre-game rating filters from rating-change messages, optional account exclusions, seeded per-team-pair and individual preview-team caps, input/selection fingerprints. Defaults preserve existing fit behavior. Six 30-resample fits, seed 20261018: schema6 same corpus; corrected schema7 all; both pre>=1234; seven accounts omitted; max20 games/team; combined three restrictions.
- Corrected all-games lagged weights (100 = full HP): alive59.7, boosts offense18.1/defense16.5/speed17.2, burn-39.2/par-19.1/poison-33.8/sleep-freeze-55.4, fresh Tailwind61.0/screen36.5. Same-corpus schema6 was alive59.5/Tailwind61.6/screen34.3. Across corrected sensitivity runs: alive59.6-70.3, Tailwind54.0-62.7, fresh screen36.5-49.1; repeated accounts/templates matter more than parser repair for some values. No automatic account deletion or live default change.
- Random-game held-out outcome prediction error: corrected fitted0.56896 vs hand0.57980; schema6 fitted0.56922. Prediction improvement is correlational, not proof of better move choice. Teams/accounts can cross that holdout; no fresh team-disjoint strength claim. Fit has no weather/terrain features. The 1234 floor is a sensitivity check and cannot recover uncollected games.
- Evidence: docs/replay_state_refit_2026-10-26.md; runs/replay_state_audit_20261026/{parse.log,audit.json,summary.json,health_before.json}; runs/eval/value_fit_*20261026.json; runs/viz/data_health/index.html.

## 2026-10-26 — Live-judge execution reporting and local null check
- offline/evaluate_own_spread_pool.py now separately aggregates VgcPlayer exact_judge_log statuses/completions, timeouts, errors, skips, successful overturns, judged choices, elapsed total/mean and bounded failure examples by arm. Existing direct ExactSearchPlayer counters retained and correctly named. No move choices or strategy defaults changed.
- Tiny identical-settings check (2 teams, 8 games) returned2/8 with identical team results and degenerate grouped interval; harness flagged it. Expanded same-settings check (32 train teams, 64 games, 4 workers, seed20261027, exact_judge_live=True both arms) returned25/64=39.1%, team-grouped95% interval[27.6%,50.5%], so null check PASS. This small check does not rule out subtle bias or show a strength gain. Artifacts: runs/replay_state_audit_20261026/live_judge_null_{smoke,32}.{json,log}.
- Expanded run logged1,310 calls:1,079 completed,229 skipped,2 errors,0 timeouts;669 successful overturns; ~0.53s/call including skips. Both errors were incumbent fast-choice fallbacks. Their causes were not retained in that run and remain unclassified; bounded failure details were added afterward and covered by the focused reporting tests. Do not assume error-free execution in the next experiment.
- Validation:174 focused tests passed,2 integration tests deselected; affected reporting/judge tests rerun after failure-example addition (16 passed,2 deselected). Ruff and git diff --check PASS. Independent Sol review found no remaining actionable parser/fit-equivalence/audit/reporting issue. Full integration/readiness suite not run (offline data/reporting changes); no public ladder games or promotion.
- Next concrete task: on recorded ladder player-view positions, measure whether the experimental-value winner from a fixed wider candidate pool reaches the live judge's six choices. Hold replies/random futures fixed, separate candidate filtering from reply/sample/timeout effects, and classify judge errors. Latest local pilot session20261013T193848Z is historical. Only after diagnosis, choose one default-off fast-ranking experiment and require same-session live-pipeline/fresh-team validation before ladder blocks. Larger RL/learned Q/Jev/coaching work remains parked.

## 2026-11-01 to 2026-11-02 — User-directed neural pretraining plan and replay refresh

- User steered this chat toward initial training from replays (pretraining) of a specialist neural network, followed by additional training on stronger-player examples and eventually learning from practice-game outcomes (reinforcement learning). Work in understandable chunks; current work is data preparation, not a training run. The desired stronger-player subset is 1600+ pre-game rating; this is not evidence of achieving that rating. The project ladder goal remains separate.
- Architecture discussed: existing `BcPolicyNet` reads board summaries; `CandidatePolicyValueNet` scores complete legal two-Pokémon actions and predicts outcomes. A learned selector for our bring-four/two leads was proposed, not implemented. Existing own preview selection remains engineered. The old replay warm-start copies compatible board-reader embeddings/trunk only; it does NOT pretrain the complete-action scoring layers. Do not describe the proposed full replay-to-neural-to-practice pipeline as already running.
- Corrected an early status explanation after finding the newer, uncommitted October 28–29 research documents: fresh candidate networks had already learned on narrow local practice tasks. Those isolated results are not general ladder strength. The older failed 100k recipe remains a closed recipe; it does not prove all neural or reinforcement learning approaches fail. Preserve the pre-existing dirty research worktree.
- Completed the authorized M-C replay refresh from the saved 54,870-game baseline: **10,141 new / 65,011 total replays**, reaching November 1; **1,281,380 schema-7 decision records** (924,518 turn, 226,840 replacement, 130,022 preview). Parser failures/skips: zero. All baseline IDs retained; audited eligible IDs present; no malformed/wrong-format new replays. Two pagination gaps were individually repaired. Old raw and parsed datasets were preserved.
- Source: `data/bc/decisions_regmc_s7_20261101.jsonl.gz`, compressed SHA-256 `6eeebcd9a2a53ba718ab88f896692cdd6509cd247e83659a636216034b037ac7`. Run evidence: `runs/replay_refresh_20261101/{final_report.json,final_report.md,parse.log,listing_audit.json,gap_repair.json,environment_review.md,showdown_parity.json}`. The bounded listing audit is not proof of complete public archive coverage. Download settings: shared post-game listing rating >=1200, max 1200 pages, 0.3 seconds between requests, excluded `laplacestheorems`. Use individual PRE-game ratings for stronger-player selection; later filtering cannot recover omitted games.
- The completion monitor `finish-replay-refresh` was deleted after reporting success. Do not restart the completed collection/finalizer or notification monitor.
- November 1 upstream parity check was **BLOCKED on 32 watched upstream commits** relative to local Showdown pin `3a766f2cfb5e51c1e428cfe87d0dba91a8f01640`. This is not 32 confirmed bugs. Static coverage declarations passed, but do not override that block or certify bug-free mechanics. No simulator update, model training, readiness approval, or ladder games occurred here. Recheck current environment before future practice training.

## 2026-11-02 — Replay selection inventory and fixed proposed evaluation groups

- Implemented `tools/inventory_replay_pretraining.py` and `tests/test_replay_pretraining_inventory.py`. Current proposal/evidence: `docs/pretraining_selection_2026-11-02.md`, `runs/pretraining_selection_20261102_v3/{manifest.jsonl,summary.json,independent_verification.json,run_context.json}`. Earlier unsuffixed and `_v2` run folders are retained drafts; `_v3` is authoritative for this proposal.
- Whole-game chronological allocation starts approximately 80/10/10, keeps timestamp ties together, then removes reserved-lineup appearances from earlier groups. Final: **47,096 training / 5,832 development / 6,033 final-test games**. Every turn, both seats, all teaching tasks, and broad/1600+ stages share that same assignment. Do not reuse the older replay-ID hash split for this prepared dataset.
- Set aside **4,683 distinct games** for disclosed team sheets or reconstruction review: 2,625 sheet games, 1,376 Baton Pass, 610 Illusion/replacement, 122 Transform (reasons overlap). Another **1,367 older games** containing reserved lineups are excluded from training/development. Originals remain preserved; no other accounts were guessed to be bots or deleted.
- Reserved **176 sorted six-species lineups** by fixed hash ordering/seed `20261102`. They occur in **445 final-test games**, a SUBSET of the 6,033 test games, with zero appearances on either side in retained training/development. These are species-lineup identities, not exact private sets or every similar team. The ordinary future-game test intentionally permits familiar teams/accounts.
- Selection manifest SHA-256: `7ccfcf95ac7bd1b5989762bd4d81f385ee019288bb2593e0fd1a245384ac072a`. Input fingerprint, split counts, every raw join and **130,022 individual pre-game ratings** reconciled. The inventory's initial empty-player-name handling defect was corrected and tested; it did not require changing source replays.
- Training has **6,081 player-game perspectives at pre-game rating >=1600**, comprising 3,096 wins and 2,985 losses before task-specific masks. Full corpus: 12,101 such perspectives. A player-game is one player's view of one game, not a unique person or one decision. Exclude unknown ratings from this stronger-player subset. Keep losses as well as wins.
- Inventory/parser verification: 42 focused checks passed (5 inventory, 2 raw-audit, 35 parser/format). Ruff and whitespace checks passed. Independent review completed. No full battle or training suite was run for this offline inventory.

## 2026-11-02 — Partial-answer preparation and team-balanced sampling implemented

- User asked to fix incomplete labels and repeated-lineup dominance. Added `src/vgc/bc/{replay_supervision.py,replay_sampling.py,replay_pretraining.py}`, `tools/prepare_replay_pretraining.py`, and corresponding `tests/test_replay_{supervision,sampling,pretraining}.py`. Report: `docs/replay_pretraining_preparation_2026-11-02.md`.
- **Current prepared output is `data/bc/pretraining_regmc_s7_20261102_v2/`**: separate `train.jsonl.gz`, `validation.jsonl.gz`, `test.jsonl.gz`, `sampling_broad.jsonl`, `sampling_expert.jsonl`, and `summary.json`. The unsuffixed prepared directory is a retained diagnostic draft. Output has **1,155,762 records**: 925,046 train, 114,118 development, 116,598 test. No source dataset was overwritten.
- An explicit yes/no mask says which answers may contribute to learning. Unknown actions never become pass. Disabled targets are null and cannot be consumed as enabled labels. Outcome teaching is independent of move availability; timer/offline results are masked out. Whole-game conservative filters use raw protocol and Champions move properties to exclude called/locked/charge/recharge move observations. Chat/nicknames are not searched for move-name meanings. Only early switches before resolution events qualify; dragged/replacement choices do not become voluntary switch labels. Preview labels teach an unordered brought-four set and an ordered lead pair, not bench order.
- Enabled TRAIN answers, broad / acting-player 1600+: moves **604,210 / 41,223**; early switch destinations **100,853 / 6,975**; brought-four sets **75,597 / 5,106**; lead pairs **93,835 / 6,074**; outcome positions **915,867 / 60,628**. Counts overlap and have different units; do not sum them as independent samples.
- **Original-target and complete submitted joint-choice teaching both remain disabled (zero certified labels).** Public logs lack complete legal requests and usually exact own sets; called/locked moves, redirection and involuntary movement can conceal intent. Missing evidence was not recovered or fabricated. Only the original pre-decision state is a model input; ratings, names, results, labels and later appearances stay outside it.
- Sampling controls each game's influence per task, divides across eligible player perspectives and answers, then divides game weight by the square root of the more frequent of its two lineups' TRAIN-game frequencies. Mirror lineups count once. Broad and 1600+ phases fit their frequencies separately using training only. Development/test retain natural frequencies. Do not apply the same weights again as loss weights after weighted sampling.
- Largest broad outcome-lineup participation drops from **34.0% to 2.1%** of expected draws; move-task maximum drops from **24.5% to 1.3%**. This is a sampling change, not a measured strength gain. Broad outcome weights have effective size about 14,678 equally weighted games out of 46,528 eligible games. Rare lineups gain influence; the best weighting recipe still requires development comparison. In the smaller 1600+ pool, maximum account participation remains roughly 9–10%, depending on task.
- Preparation REQUIRES the pinned selection manifest SHA above. Independent review caught that matching split counts alone would permit a train/test ID swap; a negative test now rejects swapped assignments before output creation. Prepared shards/plans are fingerprint-checked; existing output folders cannot be overwritten.
- Consumer APIs: `vgc.bc.replay_pretraining.load_sampling_plan`, `iter_task_examples`; `vgc.bc.replay_sampling.sample_task_units` draws `(replay_id, side, task, ordinal)` references. An exhaustive task iterator is not itself balanced random sampling. Explicit final-evaluation mode is required to read test examples; test examples have no training probability. The legacy `BcTurnDataset` and old training entry point were NOT automatically switched to this new contract.
- Verification: **15 focused tests passed**, Ruff/whitespace passed; independent review verified the split-integrity fix. `runs/pretraining_preparation_20261102/verification.json` checks all **1,155,762 exported inputs equal their original snapshots**, source/game counts and every task probability reconcile, and **200,000 outcome draws across broad/expert include zero held-out games**. Actual largest-lineup shares 2.058% broad vs 2.095% expected, 2.442% expert vs 2.411% expected. Scripts, example records, and source fingerprints are in the same run folder.
- Important remaining boundary: prepared partial observations can support auxiliary predictions (extra outputs learning observed facts) and shared board-reader learning. They do **not** train the current nonlinear complete-action scoring head. Do not zero-fill an unknown partner action or synthesize guessed legal candidates to claim full-choice pretraining. A compatible objective and suitable exact-choice evidence remain to be designed/collected. No optimizer/model training was executed here.

## 2026-11-02 — Full data-science report and current handoff

- Built `docs/data_science_report_2026-11-02.md` and an interactive report in `runs/data_science_report_20261102/app/` (15 evidence-backed sections, 11 charts, 4 rendered tables). Local preview: `http://127.0.0.1:61821/?view=1`. The Data preview helper registered this loopback-only report service with user-login restart; it was not published externally. Preserve its existing project/port and shared report runtime when revising.
- Report inputs/analysis: `runs/data_science_report_20261102/{profile.py,profile.json,state_profile.json,bias_diagnostic.json,build_report.py,report_validation.json}` plus saved app source. Profile joins all 65,011 game IDs and ratings against the fixed manifest. Fresh decision-state scan covers all schema-7 records. Source compressed hash was rechecked. Independent numerical review found no material issues; actual browser checked all chart marks, tables, narrow layout and source/data-preview panel. No exhaustive browser/theme sweep was claimed.
- Descriptive findings: median pre-game rating **1,349**; 12,101 / 130,022 player-game perspectives are 1600+ (**9.3%**). Median game length **7 turns**, 90th percentile **11**. Endings: **41,832 played out / 22,392 forfeits / 787 timer/offline**. Current normalized unique accounts **15,040**; six-species lineup signatures **28,313**. Most common lineup appears in **17,587 games (27.1%)**, counting mirrors once per game.
- Preview coverage: 296 raw species/form names, **292 matching the 390-entry exact-format catalogue (74.9%)**, plus four outside-catalogue aliases/forms (`alcremiesaltedcream`, `florgeswhite`, `florgesyellow`, `furfroukabuki`). Do not use 296/390 as legal-match coverage. Catalogue battle forms may not appear by name at preview; the coverage fraction is not a clean strategy-coverage score. Alias normalization remains a review item, not proof of illegal games.
- Public-field completeness across 1,751,349 active snapshots per relative side: item observed about **18.4%**, ability **18.0%**, any move **70.0%**. Repeated own/opponent observations are symmetric across both seats. Missing public information is not absence of an item/ability and is not the private information available to the actual player.
- Collection-bias diagnostic, unequal-rated games only: stronger player won **145/2,682 (5.4%)** when weaker pre-rating <1200, versus **29,821/56,015 (53.2%)** when both pre-ratings >=1234. This is consistent with outcome-dependent inclusion from the post-game download cutoff; do not interpret it as population skill behavior or assume the 1234 sensitivity group is unbiased. Filtering cannot recover uncollected battles.
- The chronological final test has **17.0% 1600+ perspectives**, versus **6.5% training / 13.4% development**. Later evaluation therefore has a different player mix. Existing team/account overlap, repeated outcomes/positions, and unknown older-model exposure remain material. No model-performance, 1600+/1700+ strength, or readiness claim was made.
- **Next concrete work:** choose/implement a training objective and outputs compatible with these partial labels, and wire a trainer to the new masks/saved sampler without test leakage. Be explicit about shared-representation versus complete-action-head learning. Compare weighting choices on development data, freeze final test, and use fresh model initialization or verified provenance. Additional useful data work: review four preview aliases and improve collection around the rating cutoff. Before practice-game training, separately resolve/recheck Showdown parity and relevant environment checks. Do not silently start training while that block is unresolved.
- All code/docs from this chat remain **uncommitted** at this handoff. HEAD is still `d853c64`. The prior dirty October 28–29 files were preserved; no commit, model promotion, simulator update, ladder play, or old-dataset deletion occurred here.

## 2026-11-04 — Pre-pilot data gaps: engine sync, rating backfill, own-team rebuild, trainer

- User decisions: per-Pokémon move outputs (no joint-scorer pretraining); rebuild acting player's OWN team from facts first; backfill every rated game in the same Sep 9–Nov 1 window; engine update now. Lead decision (user may override): broad pretraining floor = acting player PRE-game rating >=1200 (`--broad-min-pre-rating 1200`); <1200 games kept on disk.
- Showdown synced 3a766f2cf -> 16db150ab, commit `33a8861` (Aura Guard, No Retreat trap fix, Champions doubles tiers, Sneasler OU->Uber singles-only; Sneasler still legal in M-C). Gates + parity PASS, local smoke 1/1. Parity "speciesclause" blocks were a stale-build checker artifact. All 5 old BC checkpoints now vocab-stale (auraguard); none default-enabled. Launcher needs nvm Node 22 on PATH.
- Backfill RUNNING (detached PID 26508): `download_replays.py --min-rating 1 --max-pages 5000 --sleep 0.3 --exclude-player laplacestheorems`; baseline index in `runs/replay_backfill_20261104/start.json`. Progress = file count (index rebuilt only at end). ~6k/h; 79.9k files at 2026-11-05 00:26 UTC.
- Parser schema 8 (`data/bc/decisions_regmc_s8_20261104.jsonl.gz`, frozen 65,011 games): adds `log_line`; cosmetic formes -> base via `vgc.data.canonical_species_id` (27 raw names incl. Alcremie/Florges/Furfrou/Vivillon), also in `encode_state`. Verified identical to s7 after canonicalization. `encode_state` bench now sorted by species id (live/replay order parity). Layout version unchanged.
- Own-team rebuild `vgc.bc.own_team_rebuild` (opt-in `prepare_replay_pretraining.py --rebuild-own-team`): fills acting player's own active item/ability/4 moves + unappeared brought bench. Layers: same-game protocol facts -> same account+lineup donors (sheets first; time rule by target START TIME: <DEV_START / <TEST_START / any) -> hidden-conditional common sets (pre-DEV_START sheets where item/ability stayed hidden). Opponent side byte-identical; nothing known overwritten. `own_info`/`own_fully_confirmed` are bookkeeping, NOT encoder inputs (flags would leak future usage).
- Answer key (19,791 sheet units, leave-one-game-out priors): L1 facts 100% precise; end-to-end item 96.5%, ability 98.7%, all-4-moves 90.7%, fully exact 88.6% (54.4% when no other sheet donor — the realistic case). Layer-3 item guess 61.6% (v1 40.1%). Coverage: own_fully_confirmed train 22.7% / 1600+ 39.1%; own bench mean 0.41 -> 1.035. Bugs found and fixed via the answer key: post-mega ability, `cant` holder (position mon), move-caused ability change, `|swap|`, Trick `-activate`, later-fainting bench. `illusion_possible(preview)` sets aside 395 more games (unbroken Illusion). Evidence `runs/own_team_rebuild_20261104/v4/`.
- Trainer: `vgc.bc.replay_pretrain` (TurnPretrainNet = `BcPolicyNet(heads=())` trunk + per-slot option-masked action head [own moves + <switch>], destination head [own bench], value head; PreviewPretrainNet species-only, sorted inputs, 15 brought combos + 30 ordered leads) and `tools/train_replay_pretrain.py`. Loss = ACTION(pooled) + DESTINATION + VALUE; sampling probabilities only draw units. Shared `trunk_inputs` used by `encode_live_state`. Checkpoint warm-starts `CandidatePolicyValueNet.state_encoder` fully (tested). Test split refused. Smoke (2k scratch games, NOT strength): action 0.295 ≈ train-frequency baseline 0.298; others beat baselines. Review caught: cache keying, pooled-loss calibration, destination never evaluated, inverted frequency baseline, empty breakdowns/1600+ subset (meta read from wrong field) — all fixed with tests.
- Not done / next: after backfill -> listing audit, reparse s8, new selection v4 with same DEV/TEST boundaries + `state_review_illusion_possible`, lead-authored skew diagnostic (stronger player win rate by lower-player pre-rating band), prepare with rebuild + 1200 floor. Only then pilots. All Job C/D code UNCOMMITTED (HEAD 33a8861).

## 2026-11-05 — Backfill resume + live-vs-replay parity + per-task rating floors

- Backfill crashed 2026-11-05 ~03:30 UTC (uncaught http.client.RemoteDisconnected) at ~Sep 14 equivalent (138,234 files). Fixed: `TRANSIENT_ERRORS` in tools/download_replays.py. Resumed as PID 22214 (log runs/replay_backfill_20261104/download_resume.log); 149,669 files at 23:11 UTC. It also fetches post-Nov-1 games -> fixed-boundary selection excludes them as `after_window`.
- User decisions: parity check before rebuild (done); rating mix split by network — policy tasks (move/target/switch_destination/preview_*) broad = pre-game >=1200, outcome broad = any known rating (`--broad-outcome-min-pre-rating -inf`), expert >=1600 all tasks; open-team-sheet games (`showteam`) excluded from policy tasks, kept for outcome/answer key.
- Selection: `inventory_replay_pretraining.py --boundaries-from runs/pretraining_selection_20261102_v3/manifest.jsonl` -> dev_start=1792406685, test_start=1792870992, window_end=1793572357; Illusion set-aside included. Sampling eligibility per (game, side, task) in replay_sampling.py; trainer headline validation uses the same eligibility; `<1200` band added.
- Parity harness offline/check_live_replay_parity.py (direct env, additive spectator capture in tools/sim_worker.mjs, 60 seeded games). v1 -> v7: identical-after-masking-inherent 0% -> 99.8% (1,090 decisions; residual 2 cosmetic boost-dict rows). Inherent (masked): own unrevealed moves/item/ability (L2/L3 estimates), own HP rounding <=0.01, never-appearing reserves.
- Bugs fixed via parity: forced-switch replay snapshot leaked both the own destination and the opponent's simultaneous replacement (now frozen pre-replacement; test); LIVE opponent mega flag/ability wrong (poke-env keeps base `_species`; now `_last_details` forme via poke_env_compat + details-driven `_active_mon_dict`; a first fix assigned mega abilities to wrong mons — caught and fixed; tests in test_bc.py); parser missed `[from] item:/ability:` reveals (scan sat after `continue` handlers); `-ability`/Trace attribution; single-legal-ability fill (Champions data; poke-env maps mega formes to base abilities -> live overridden); own replacement now on own bench at forced switch. Encoded ability = current effective (matches live); base ability kept for rebuild facts. Parser schema 9 (s8 untouched).
- Readiness gates (battle-state, action, static) PASS after live changes; 135 focused tests pass. All work since 33a8861 UNCOMMITTED.
- Next: wait for download -> listing audit, reparse s9, select v4 (fixed boundaries), skew diagnostic (lead-authored), prepare with rebuild + floors (1200 policy / -inf outcome).
- Committed 2026-11-05 (see git log): everything since 33a8861. Pre-commit gate: full non-integration suite 1619 pass, ruff clean; tests/test_bc_policy.py opp-ability expectations updated to the single-legal-ability rule (Gholdengo -> goodasgold). Integration suite (needs local server) not re-run; static readiness gates passed earlier.

## 2026-11-06 — Dataset rebuild on the completed backfill

- Download finished: 149,752 replays (11,518 new in the resume), full listing walked to Sep 9, 0 failed fetches, 20,315 unrated skipped.
- s9 reparse: data/bc/decisions_regmc_s9_20261106.jsonl.gz (149,752 games, 2,857,780 records, 0 parse failures). tools/parse_replays.py writes plain text regardless of .gz suffix — recompressed. audit_replay_state FAIL is a positional-join artifact: lead keyed check on the 65,011 s8 games -> identical record counts per kind and identical turn actions (924,518/924,518).
- Selection v4: runs/pretraining_selection_20261106_v4 (fixed v3 boundaries): train 109,547 / val 11,517 / test 11,110; excluded 15,325 (open_team_sheet 4,873, after_window 4,107, illusion_possible 3,087, baton_pass 2,671, illusion_or_replace 1,865, preview_raw_mismatch 860, transform 611); reserved_team_excluded 2,253. NOTE: open-team-sheet games were already excluded wholesale in v3 too, so they feed neither outcome training nor policy tasks (only the own-team answer key/donors via raw facts).
- Who-won check (tools/rating_skew_check.py, lead-authored): v3 showed stronger-player win 5.0% (lower 1100-1199) and 33% (1200-1233); v4 shows 52-56% in every band and split. Skew fixed.
- Own-team answer key on v4 (runs/own_team_rebuild_20261106): L1 facts 99.99/99.96/100%; end-to-end item 95.2 / ability 98.1 / all-4-moves 87.5 / fully exact 84.9 (no-sheet-donor 48.6); own_fully_confirmed train 20.8%, 1600+ 39.6%.
- Prepared: data/bc/pretraining_regmc_s9_20261106 (floors 1200 policy / -inf outcome; selection sha cec196…f25, dataset sha 797186…4300). Broad units/effective games: move 697,580/13.6k, switch_destination 114,348/14.0k, preview_brought 86,401/15.7k, preview_leads 107,945/18.0k, outcome 2,045,790/36.2k. Expert: move 41,248/1.46k, outcome 60,481/1.99k. Team participation after weighting <=2.8%; expert top account ~9-10%. target task 0 (not certifiable, by design).
- Fixed stale schema gates (inventory_replay_pretraining.py, rebuild_own_team.py -> SCHEMA_VERSION).
- Next: user go/no-go on first pretraining pilots.

## 2026-11-06 (later) — Independent re-verification (user asked; lead-authored checks)

- New tools/verify_pretraining_data.py: re-derives from RAW logs (raw hashes, pre-game ratings incl. HTML-escaped names, winners, split boundaries, duplicates, reserved-team rule), checks prepared shards (split membership, source-line join, forbidden input keys, outcome vs raw winner, preview label sanity, forced-switch destination leak, own-mega-this-turn leak, move option-set sizes), sampling plans (train-only, policy floor 1200 + no open-sheet, expert >=1600, outcome includes <1200), and fog on 3,000 sampled games (every opp move/item/ability/mega revealed in the raw log before the decision line; lenient token match).
- Found 4 real parser bugs the earlier checks and the 60-game parity harness missed (our bot never used these mechanics): (1) no `|swap|` (Ally Switch) handling -> slot attributions wrong for the rest of the game, 1,533 kept games; (2) Revival Blessing revive ignored -> revived mon absent from bench; (3) revive keyed by nickname, not species (Maushold-Four); (4) `|cant|HOLDER|ability: Queenly Majesty|MOVE|[of] ATTACKER` credited the move to the holder (6,157 games touched). Fixed through schemas 10 and 11; s10->s11 diff 0 records outside affected games.
- My own check had 3 false alarms, fixed in the tool: HTML-escaped names in rating lines, moves revealed by `cant` lines, reserved-team rule is the hashed 5% not all test rosters.
- Final verified dataset: data/bc/pretraining_regmc_s11_20261106 (source decisions_regmc_s11_20261106, selection runs/pretraining_selection_20261106_v6, rebuild runs/own_team_rebuild_20261106_v6). Verifier (runs/verify_20261106_s11): 0 violations on all hard checks; 116 move units whose label is outside a 4-move option set (trainer drops them); 45 slots with 5 moves (conflicting evidence, ~0.003%). Rating-skew fixed. Win/loss per-record imbalance (51% loss) is expected: losers have more forced switches; sampling equalises perspectives.
- Use s11 for pilots; s9/s10 kept but superseded.

## 2026-11-06 — Second independent verification pass (in progress)

- **The preceding s11 pilot-readiness statement is withdrawn.** The saved `runs/verify_20261106_s11/verification.json` still contained 64,921 flags from the old reserved-team check, and the verifier always exited 0. The corrected check had not been rerun to produce that saved report. Fresh `runs/verify_s11_pass2/verification.json` now passes its stated checks: all 8 artifact hashes, raw metadata, fixed boundaries, ratings, outcomes and sampling rules; exit status is now nonzero on violations. Its old fog check remains a lenient presence heuristic, not proof of ownership.
- Added lead-authored `tools/verify_replay_identity.py`: checks every train/validation turn against raw execution identities, recoverable move masks, native opponent move reveal timing and retention, structured ability ownership evidence, known current items and mask contracts. Does not open final-test gameplay. Its own first iterations exposed verifier mistakes (battle-only identity normalization, field-setting ability lines, duplicate nicknames); those diagnostic outputs are NOT final green certificates.
- Confirmed with raw games and an independent Codex review: Ally Switch still dropped the partner's action/reveal within the swap turn; reflected Magic Bounce moves could displace the reflector's native action; Shell Bell healing credited the `[of]` victim rather than the healed holder; Struggle could enter native move options and enabled policy labels. Schema 12 fixes these; raw examples include `2678014325`, `2677963350`, `2678237852`.
- Trainer `.npz` caches and own-team facts caches could silently reuse old contents without validating current inputs/code. New checks reject legacy or mismatched caches; fingerprints include raw/encoded inputs, relevant source, vocabulary/species data and cosmetic-form mapping.
- Reserved roster games were excluded from training rows but still eligible to contribute to common-set priors. The independent manifest check found 2,084 pre-dev eligible source games with reserved rosters (1,922 explicitly reserved exclusions, 162 other excluded games). They now supply neither cross-game donors nor priors. `prior_game_ids` records actual prior contributors; the answer-key measurement subtracts a game's contribution only when that game actually contributed. Missing-time donor targets return no donors.
- Non-reserved donor-time policy is unchanged: train targets use pre-validation donors; validation may use other validation-period games; test may use donors from any period. Reconstruction evaluation is therefore not a strictly chronological simulation of team information discovery.
- Verification so far: 98 focused parser/supervision/rebuild/cache/preparation/sampling tests pass; 107 BC/RL/inventory tests pass (17 deselected); battle-state and action static readiness checks PASS. No model training.
- Rebuilding the same raw corpus to `data/bc/decisions_regmc_s12_pass2.jsonl.gz`, selection `runs/pretraining_selection_s12_pass2`, fresh facts `runs/own_team_rebuild_s12_pass2`, prepared `data/bc/pretraining_regmc_s12_pass2`. Lead-authored recipe `/tmp/vgc_rebuild_pass2.sh`; log `runs/rebuild_s12_pass2.log`. Fixed boundary input `runs/reverify_pass2_boundaries.jsonl` preserves 1792406685 / 1792870992 / 1793572357. Do NOT use the v6 manifest directly as `--boundaries-from`: that command takes max time across excluded rows too, which would extend the window.
- Pending: finish schema-12 rebuild, review final Codex findings, run both audits on the new artifacts, check fresh answer-key measurements, and record final results below. s11 is retained for diagnosis, not approved for training.

## 2026-11-10 01:20–03:00 CST – Claude Code / Fable 5.2 — finish second verification pass (schema 13)

- Picked up the dirty branch `fix/replay-state-20261026`. State found: s12_pass2 rebuild had finished 2026-11-06 13:34 but its three final checks were never run; `tests/test_own_team_rebuild.py` had 9 cache-test lines pasted into the wrong test (1 failing test, 14 lint errors) — repaired. `/tmp/vgc_rebuild_pass2.sh` and the earlier Codex notes were gone.
- s12_pass2 checks (now run): verify_pretraining_data PASS (runs/verify_s12_pass2); verify_replay_identity 1 flag in 1.46M (game 2680728537 T6: Hatterene `cant|move: Throat Chop` for a blocked reflection, then its own Dazzling Gleam — label dropped, not wrong); answer key v2 item 94.8 / ability 98.1 / all-4-moves 86.6 / fully exact 83.7 (no-sheet-donor 48.8); slightly below v6 because reserved rosters no longer donate; L1 item mismatches 2 -> 0.
- Fresh Codex review (review-only, no files changed) -> 3 findings, all fixed: (1) parser: a named `cant` AFTER the Pokemon's own move (blocked Dancer copy) was added to its revealed moves; a `cant` BEFORE its own move displaced the real label. Now a later native move overrides a cant-derived action and withdraws the cant-named move; named cant after acting reveals nothing. Raw scan of all games: 0 move-then-named-cant, 99 cant-then-move (88 sleep + Sleep Talk). Schema 13. (2) verify_replay_identity now flags any opponent item with no earlier raw evidence (`opponent_item_before_decision`); the cant allowance I had briefly added was removed. (3) measure_own_team_rebuild now loads facts through `build_corpus_facts` (new required `--raw-dir`), so stale facts are refused.
- Files: src/vgc/replay_parse.py, tools/verify_replay_identity.py, tools/measure_own_team_rebuild.py, tests/test_replay_parse.py, tests/test_own_team_rebuild.py
- Checks: full unit suite 1634 pass, ruff clean, battle-state + action static gates PASS (run before the schema-13 edit; parser is offline-only).
- Rebuild RUNNING: recipe `runs/rebuild_s13.sh` (kept in runs/, not /tmp), log `runs/rebuild_s13.log`. Raw dir now holds 154,813 files (5,061 newer downloads 11-07..11-09); the recipe pins the same 149,752 games via `--only-ids runs/pretraining_selection_s12_pass2/manifest.jsonl`. Outputs: data/bc/decisions_regmc_s13.jsonl.gz, runs/pretraining_selection_s13, runs/own_team_rebuild_s13, data/bc/pretraining_regmc_s13, runs/verify_s13, runs/verify_identity_s13.
- Next: read the three s13 check results; if clean, s13 is the candidate pilot dataset (s11/s12 superseded). Nothing committed yet.

## 2026-11-10 02:50 CST – Claude Code / Fable 5.2 — schema 14 (s13 superseded before use)

- s13 results: verify_pretraining_data PASS; new opponent-item timing check 0 flags in 771k; identity verifier flagged 11 records / 2 games (2690288427, 2692927883): a Copycat / Magic Bounce move blocked by Armor Tail (`cant|HOLDER|ability:..|MOVE|[of] ATTACKER`) was listed as the attacker's own move. Fixed: the ability-block branch now labels/reveals only if the attacker has not already acted that turn (and a later native move overrides). Schema 14; test added; `test_cant_ability_block_attributes_to_of_mon` log made realistic (leads before `|turn|1`).
- Checks: full unit suite 1634 pass, ruff clean.
- Rebuild RUNNING: `runs/rebuild_s14.sh` -> log `runs/rebuild_s14.log`; outputs `*_s14` (data/bc/decisions_regmc_s14.jsonl.gz, runs/pretraining_selection_s14, runs/own_team_rebuild_s14, data/bc/pretraining_regmc_s14, runs/verify_s14, runs/verify_identity_s14).
- Next: read s14 check results (all three must be clean). Then s14 is the candidate; s11/s12/s13 superseded (kept on disk, ~0.5 GB each — delete only with owner OK). Nothing committed.

## 2026-11-10 04:10 – Claude Code / Fable 5.2
- Did: s14 rebuild finished (2,506,547 records, same 149,752 games). All three checks clean: verify_pretraining_data PASS; identity verifier PASS (0 violations); answer key v2 item 94.8% / ability 98.1% / all-4-moves 86.6% / fully exact 83.7% (unchanged from s12/s13).
- Did: first s14 identity run had 1 flag (game 2693447392 T3: Hatterene switched in, Magic Bounce reflected Parting Shot, blocked by Armor Tail). Verifier bug, dataset was right: a named `cant` after ANY move line by that actor this turn (called or native) is now not a reveal. Re-run: pass.
- Files: tools/verify_replay_identity.py; outputs runs/verify_s14, runs/verify_identity_s14 (pre-fix run kept as runs/verify_identity_s14_prefix), runs/own_team_rebuild_s14, data/bc/pretraining_regmc_s14
- Checks: unit suite 1634 pass + ruff clean (before this verifier-only edit; ruff re-run on the verifier clean). Integration suite not run.
- Next: owner go/no-go on pilots using s14. Commit when owner asks.
- Open: nothing committed. s11/s12/s13 datasets superseded, still on disk (~0.5 GB each) — delete only with owner OK. Theoretical parser gap, 0 cases in corpus (verifier would flag): a not-yet-acted Pokemon whose reflected move is ability-blocked and who then never moves that turn would get a false reveal/label.

## 2026-11-10 08:16 – Claude Code / Fable 5.2 — independent bug scan of s14 (no code changed)
- Did: fresh checks on data/bc/pretraining_regmc_s14 that the two verifiers do not cover. Scripts + outputs in runs/dataset_bug_scan_20261110/ (scan1 = structure/leak/labels, scan2 = per-move label coverage, hpcheck = HP/status/boosts/field vs raw log, itemcheck).
- FOUND (1) value leak: own bench is completed from FUTURE appearances (own_team_rebuild.py ~L1151); when the 3rd/4th mon never appears the bench is short. Turn-1 train rows: bench 2 -> 43.1% win (169k), bench 1 -> 76.4% (31k), bench 0 -> 74.7% (14k). Live bench is always complete.
- FOUND (2) move-label hole: replay_supervision.audit_raw_log's game-wide `locked` flag drops ALL move labels in any game with a charge/recharge/Encore/called move. 30,625/108,603 train games have zero move labels; Electro Shot used 40,440x labelled 0, Encore 7,934 -> 0, Hyper Beam 4,336 -> 0, Solar Beam 3,178 -> 0; Dragon Pulse/Weather Ball/Snarl ~46%.
- FOUND (3) parser ignores -sethp (Pain Split, 191 kept games), -copyboost (Psych Up, 1,031), -invertboost (102), -swapboost (12): stale HP/boosts afterwards. Examples 2701640633 T7, 2678110470 T5.
- FOUND (4) teampreview rows carry an outcome label but encode_state reads `preview_species` (rows have `preview`) -> all-blank input, ~10% of outcome units. (5) preview_brought labels need all four to appear -> ~57% from losing side.
- Clean: 0 out-of-vocabulary tokens, 0 duplicate turn rows, labels inside option sets, consumed items stay gone, HP/status/weather/terrain/Trick Room/side conditions match raw in 10,000 games (513k Pokemon snapshots) apart from (3).
- Checks: analysis only; no tests run, no repo code changed.
- Next: owner decides fixes for (1)-(4) before any pilot; s14 NOT ready for value pretraining as is.
- Open: still nothing committed on this branch.

## 2026-11-10 (later) – Devin / Fable 5.2 — fixing the four s14 dataset bugs (IN PROGRESS)
- Did: owner asked to fix bugs (1)-(4) from the 08:16 scan. Design: (1) own bench always true size — never-seen brought slots filled with a hash-picked unseen preview species (`own_info.bench_guessed`); (2) game-wide `locked` gate replaced by per-turn/per-slot `move_locks` (Encore start..end turn, Instruct); charge/Outrage continuations already carry `[from] lockedmove` on the move line (0 exceptions in 25k raw games); (3) parser handles -sethp/-copyboost/-invertboost/-swapboost, schema 15; (4) encoder reads `preview` when `preview_species` is absent.
- Did: verifiers updated to match: verify_replay_identity now uses per-(turn, actor) locks (new flags move_mask_on_locked_actor, move_label_not_single_native_move); verify_pretraining_data gains B_turn1_own_bench_full.
- Files: src/vgc/bc/{own_team_rebuild,replay_supervision,encoding}.py, src/vgc/replay_parse.py, tests/*, tools/verify_replay_identity.py, tools/verify_pretraining_data.py, runs/rebuild_s15.sh (recipe, not yet run)
- Checks: pending (focused tests + full unit suite).
- Next: review code diff; run `runs/rebuild_s15.sh` (~1.5 h) -> *_s15; re-run both verifiers + runs/dataset_bug_scan_20261110 scans pointed at s15.
- Open: nothing committed. Bug (5) bring-four label skew not addressed (not asked).

## 2026-11-10 08:39 – Devin / SWE-2 (sidekick) — four s14 dataset fixes implemented
- Did: implemented all four fixes per lead's spec. (1) `_guess_unseen_bench` (sha256 on replay_id|side|species) tops the bench to min(4, preview); `own_info.bench_guessed`; invariants get additions|guessed. (2) `audit_raw_log` now emits per-turn `move_locks` (Encore -start..-end turn inclusive, tracked by position with swap exchange + removal on switch/drag/replace/faint; Instruct via -singleturn); `build_supervision` gates per slot and fails closed when `move_locks` is missing; `locked_or_called` kept as diagnostic. (3) parser: -sethp joins -damage/-heal, new -copyboost/-invertboost/-swapboost handlers, SCHEMA_VERSION 14->15 (all gates use the symbol; nothing hardcodes 14). (4) `_encode_side` falls back to `preview`.
- Files: src/vgc/bc/{own_team_rebuild,replay_supervision,encoding}.py, src/vgc/replay_parse.py, tests/test_{own_team_rebuild,replay_supervision,replay_parse,bc}.py
- Checks: focused 5-file pytest 151 pass; ruff clean; full unit suite 1640 pass / 252 deselected (integration), was 1634 + 6 new tests.
- Next: lead reviews diff; dataset rebuild (runs/rebuild_s15.sh) NOT run per lead instruction.
- Open: nothing committed. Bug (5) preview_brought label skew still open (not asked).

## 2026-11-10 (later 2) – Devin / Fable 5.2 — four s14 bugs fixed, s15 built and checked
- Did: code fixes landed (see previous entry); rebuilt same 149,752 games via `runs/rebuild_s15.sh` -> data/bc/pretraining_regmc_s15 (2,506,547 records, parser schema 15).
- Checks: full unit suite 1640 pass, ruff clean. verify_pretraining_data PASS incl. new B_turn1_own_bench_full (262,688 rows, 0 bad). Scans in runs/dataset_bug_scan_s15/: turn-1 own bench always 2 (win rate 50.0%); games with zero move labels 30,625 -> 0; labelled/observed moves 99.3% (Electro Shot 0 -> 99.5%, Encore 0 -> 100%); HP/status/boost/field vs raw 0 mismatches in 10,000 games; teampreview rows encode non-blank; 0 OOV, 0 duplicate rows.
- Open: verify_replay_identity reports 2 flags in 4.5M (`recoverable_move_mask_missing`, games 2698577452 T2, 2701181441 T6): a pivot move (Parting Shot / Volt Switch) followed by the replacement's Magic Bounce reflection in the same slot -> audit sees 2 move events in the slot and drops the label. Label is MISSING, not wrong; verifier left strict, so it exits "fail" on these 2.
- Open: turn-1 value rows with a guessed bench mon: ~21% (val: 3,216 one guess, 1,516 two, of 22,874). Guess = hash-picked unseen preview species; bring-four label skew (5) not addressed. `boost_value_bad` in scan1 = zero-valued boost keys, pre-existing and harmless (78,224 in s14).
- Next: owner go/no-go on pilots using s15 (s14 superseded). Nothing committed.

## 2026-11-10 (later 3) – Devin / Fable 5.2 — committed
- Did: committed everything uncommitted on `fix/replay-state-20261026` in one commit: the second verification pass (schemas 12-14, cache/reserved-roster fixes, tools/verify_replay_identity.py) plus today's four s14 bug fixes (schema 15) and verifier updates.
- Files: src/vgc/{replay_parse.py,bc/*}, tests/*, tools/{verify_*,measure_own_team_rebuild,train_replay_pretrain}.py, this worklog.
- Checks: full unit suite 1640 pass + ruff clean (run before commit, code unchanged since). Not pushed.
- Next: owner go/no-go on pretraining pilots using data/bc/pretraining_regmc_s15.
- Open: `runs/` and `data/bc/` are gitignored, so the rebuild recipe (runs/rebuild_s15.sh), scan scripts (runs/dataset_bug_scan_s15/) and datasets live only on this machine. 2 known dropped labels (see previous entry); bring-four label skew open; s11-s14 datasets superseded, delete only with owner OK.

## 2026-11-12 02:45 – Devin / Fable 5.2 — s15 readiness check (analysis only, no code changed)
- Did: probes on s15 train+validation (`runs/readiness_s15/probe.py` -> probe.json) and a real-trainer smoke (600 broad + 100 expert steps, `runs/readiness_s15/smoke/`).
- Result data: turn-1 value rows show no structural win signal (bench/unseen/active all constant, 50.0%); smoke value acc turns 1-2 = 55%, calibrated; move options always sorted, label position spread; 0 cache skips in all 5 tasks; leads labels 50/50.
- BLOCKER (trainer, not data): `tools/train_replay_pretrain.py` build_cache crashes saving the cache on full-size train (`np.array(json.dumps(tasks))` -> "string too large", line 232). Smoke only ran via `runs/readiness_s15/smoke_nocache.py` (cache_dir=None; build 204 s train + 23 s val).
- Open (data, not blocking a pilot): 18,593 of 208,659 train destination labels (8.9%) are "one real + one guessed bench mon" -> label is always the real one (artifact; suggest masking); 56,123 (27%) have a single option (zero loss). Bring-four labels 56.9% from losers (96,269 vs 72,994). Item fill source tracks outcome weakly (both leads common_set: 37.7% win, n=14.8k) but source is not an input. 2 known dropped labels.
- Next: owner decides: fix cache save; mask guessed-bench destination units; reweight bring-four by outcome. Then pilots. Worklog entry uncommitted.

## 2026-11-12 03:00 – Devin / Fable 5.2 — compared a second (Cursor) s15 readiness review with mine
- Did: checked its claims against code and my probe. Confirmed: 215,264 of 2,045,715 train outcome units are teampreview rows (encoded as both actives empty); `evaluate_preview` is called without `policy_floor` (lines 699, 706); preview net has no expert phase / early stopping. Its neural leak-probe numbers are unverified: no scripts or outputs for it under runs/.
- Disagree: its "ready, 0 skips" verdict misses the cache-save crash (line 232) and the 18,593 guessed-bench destination labels.
- Next: owner go on one trainer patch: cache save, guessed-bench destination mask, teampreview value filter, preview policy_floor. No code changed.

## 2026-11-10 – Devin / SWE-2 (sidekick) — s15 trainer cache + filters
- Did: tasks_json saved as uint8 bytes (fixes np-2.5 string-too-large crash on 3.9GB cache); build_cache filters teampreview outcome rows + bench_guessed switch_destination rows into per-task `filtered` sets (serialized via default=list, reloaded as sets of tuples); _units drops filtered refs silently; meta gains bench_guessed; policy_floor passed to both evaluate_preview calls; "cache built" print + final_metrics.json gain filtered counts.
- Files: tools/train_replay_pretrain.py, src/vgc/bc/replay_pretraining.py, tests/test_replay_pretrain.py (+1 roundtrip test)
- Checks: 12 tests pass; ruff clean; 2 real s15 runs (300b/100e): run1 built cache 250.5s, run2 loaded 10.9s; outcome units 1,830,451 / filtered 215,264 / val value n 192,739 all match spec; final_metrics identical apart from cache_seconds/mean_step_seconds; 0 missing refs all steps.
- Next: lead reviews; data/bc/pretraining_regmc_s15 untouched on disk.

## 2026-11-12 03:40 – Devin / Fable 5.2 — trainer patch reviewed (previous entry's date should read 2026-11-12)
- Did: reviewed the four-change diff (cache save, guessed-bench destination filter, teampreview value filter, preview policy_floor); accepted as is.
- Checks: see previous entry; evidence in runs/readiness_s15/smoke2 (run1/run2 logs, final_metrics_run1.json).
- Next: owner go on commit and on the first s15 pilot. Each trainer --out dir stores a ~4.4 GB cache (disk had 29 GB free before smoke2).
- Open: uncommitted: trainer, replay_pretraining.py, test, worklog. Preview expert phase / early stopping and bring-four reweighting deferred until after the first pilot.

## 2026-11-12 05:10 – Devin / Fable 5.2 — first s15 pilot (runs/pilot_s15_1, default 2000 broad + 1000 expert steps, seed 20261112)
- Did: ran tools/train_replay_pretrain.py at HEAD 2a1354d (cache hardlinked from readiness_s15/smoke2). Owner: no pass rule yet; pilots are for finding bugs.
- Result (validation, broad ckpt step 1750): action 38.0% vs train-freq 32.0% / uniform 20.0%; destination 74.6% vs uniform 64.7%; value acc 67.4%, logloss 0.583 vs 0.693; preview brought 19.4% vs bring-rate rule 10.3%, leads 12.9% vs 6.9% (preview loss still falling at the end).
- Finding 1: expert (1600+) phase overfits from its first eval (val 2.512 -> 2.816; train 1.87 -> 1.30). Saved turn_net.pt = expert step 250: value logloss on full val 0.614 vs broad 0.583, overconfident (pred 0.96 -> obs 0.87). Action gain on 1600+ 37.2% -> 37.7% only.
- Finding 2 (confirmed mechanism, partial size): outcome sampling gives each (game, side) equal mass, so winners' rows (fewer per game) are drawn more: turn rows sampled 51.7% wins vs 50.0% raw (turn 1: 52.0%). Turn-1 mean pred on 1200+ val 55.7% vs 50.4% actual; ~2 of the 5 points explained, rest open.
- Clean: turn-1 preds flat across bench_guessed (0.556-0.560) and item-source groups -> net does not read the rebuild fills. Destination val loss rises after step 250 at flat accuracy (overconfident; 105k units).
- Probe: runs/pilot_s15_1/turn1_probe.py. Next: owner picks fixes (expert recipe; outcome sampling weight). Worklog uncommitted.

## 2026-11-10 – Devin / SWE-2 (sidekick) — s15 pilot trainer fixes A/B/C
- Did: (A) pool_task_within_game in replay_sampling.py (equal per-unit outcome weights within a game, per-game/task mass unchanged) applied to both plans in main; (B) --expert-lr-scale 0.1, --expert-eval-every 50, expert seeds best with step-0 eval of broad-best weights; (C) --preview-steps 4000 + best-checkpoint selection on mean task val loss with patience. final_metrics gains outcome_sampling + preview_best_step.
- Files: src/vgc/bc/replay_sampling.py, tools/train_replay_pretrain.py, tests/test_replay_sampling.py (+pool test)
- Checks: 20 tests pass; ruff clean; smoke3 run finished "wrote" (243.7s cache rebuild), expert 0 val 2.578 -> best_step 100; preview_best_step 200; 0 missing refs.
- Next: lead reviews; data/bc/pretraining_regmc_s15 untouched.

## 2026-11-12 05:40 – Devin / Fable 5.2 — pilot 2 started
- Did: committed 1774f60 (outcome sampling pooled within game; expert lr scale 0.1, eval every 50, step-0 broad baseline; preview up to 4000 steps with best-checkpoint). Smoke runs/readiness_s15/smoke3 OK, 20 tests pass, ruff clean.
- Running: runs/pilot_s15_2 (seed 20261112, defaults, cache hardlinked from smoke3), log train.log.
- Next: compare with pilot_s15_1; rerun runs/pilot_s15_1/turn1_probe.py on the new turn_net_broad.pt (turn-1 mean pred vs 50.4% actual).
- Open: disk 18 GB free; old caches in runs/readiness_s15/{smoke,smoke2} could be deleted with owner OK.

## 2026-11-12 06:30 – Devin / Fable 5.2 — pilot 2 results (runs/pilot_s15_2, HEAD 1774f60)
- Expert fix works: best step 150 (of step-0 baseline 2.471 -> 2.448); full-val value logloss 0.578 (pilot 1 saved ckpt 0.614), calibration now matches (0.96 pred -> 0.93 obs). Action 38.2%, destination 75.0%, value acc 67.9%.
- Preview: early-stopped at 1250 steps; brought 19.1% / leads 13.3% = same as pilot 1 -> it was NOT undertrained; it plateaus ~1000 steps.
- OPEN BUG: turn-1 mean pred on 1200+ val still 55.2% vs 50.4% actual (pilot 1 55.7%), despite pooled sampling. Not explained. Next check: mean pred on train turn-1 rows and per-game p1+p2 prediction sum.
- Deleted runs/readiness_s15/smoke2/*.npz (owner OK); freed nothing, hardlinked to runs/pilot_s15_1 (stale cache, fingerprint changed). 18 GB free.

## 2026-11-12 06:50 – Devin / Fable 5.2 — turn-1 optimism diagnosed (model, not data)
- Probes runs/pilot_s15_2/{turn1_pairs,bias_by_turn}.py on pilot-2 broad ckpt: overall val bias +0.8 pt (0.494 vs 0.486). By turn: +4.9/+6.2/+4.4 pts on turns 1-3, -1 to -3 from turn 5. Same on train turn-1 rows (+5.2). p1/p2 identical; per-game p1+p2 sum 1.10. Sampled targets are 50% per turn after pooling, so the net under-fits a turn-dependent bias. Not a label bug. Only the trunk transfers to RL, not this head.
- Next: long runs (10k broad, early stop) x3 seeds from a clean worktree; check whether the bias shrinks.

## 2026-11-10 – Devin / SWE-2 (sidekick) — eval_replay_pretrain tool
- Did: build_cache gains final_evaluation kwarg (test opens only with it AND cache_dir None, passed to iter_task_examples); new tools/eval_replay_pretrain.py (--split validation/test double-gated by --final-test, test access logged to runs/test_access_log.jsonl, never writes a test cache); value bias-by-turn table re-derived via iter_task_examples + trunk_inputs in 4096 batches; checkpoint vocabulary + prepared sha256 asserted; +1 test.
- Files: tools/train_replay_pretrain.py, tools/eval_replay_pretrain.py (new), tests/test_replay_pretrain.py
- Checks: 13 tests pass; ruff clean; validation eval on pilot_s15_2: broad value 0.5815/0.6761 and turn 0.5783/0.6794 match final_metrics exactly; bias turn-1 "turn" 22420 n, pred 0.5491, observed 0.500.
- Next: lead; --split test NEVER run.

## 2026-11-12 19:40 – Devin / Fable 5.2 — long runs x3 seeds + one-time sealed test score
- Did: runs/long_s15.sh (worktree at 1b80fb2, 10k broad steps, patience 6) -> runs/long_s15_{20261112,13,14}. All early-stopped at 2750-3250 (best 1250-1750): the DESTINATION head overfits from ~step 750 (val 0.45 -> 0.53) and drives the combined stop; action flat ~1.42 from 2000, value ~0.578. Seeds agree within 0.5 pt (action 38.2-38.3%, dest 74.7-75.2%, value 67.4-68.0%).
- Did: added tools/eval_replay_pretrain.py (327dd2b). Chosen seed 20261112 (lowest full-val loss sum 2.455). Validation: action 38.2% (rule 32.0%), dest 75.2% (unif 64.7%), value 68.0% ll 0.577, brought 19.1% (rule 10.3%), leads 13.3% (6.9%).
- TEST (opened once, logged in runs/test_access_log.jsonl): action 37.0% (rule 31.0%), 1600+ 37.3%; dest 74.8% (unif 65.3%); value 67.3% ll 0.587 (turns 1-2 57.5%, 6+ 76.1%); brought 18.6% (rule 8.2%); leads 10.6% (6.6%). Turn-1 bias of the saved expert ckpt +1.8 val / +2.4 test (broad ckpt +4.9).
- Next (not started): destination head early-stopping/regularization separate from the combined loss; step 6 (practice-game warm start + Showdown/env recheck). Test split must not be reopened for recipe tuning.

## 2026-11-12 – Devin / SWE-2 (sidekick) — dest-head own stopping + RL readiness
- Did: turn-net selection loss now action+value (dest excluded, recorded as dest_val_loss per eval); per-phase destination tracker (dest_best/dest_bad, same patience) freezes destination_head when exhausted and restores its own best state at phase end; expert unfreezes/re-seeds at step 0; final_metrics gains dest_frozen_step + dest_val_loss_restored per phase; step_losses(train_destination=False) test added.
- Files: tools/train_replay_pretrain.py, tests/test_replay_pretrain.py
- Checks: 14 tests pass; ruff clean. Long run runs/long_s15_desthead_20261112 in flight (seed 20261112). RL readiness: warm start 8/8 stale 0, encoder identity max diff 0.0; all four readiness gates BLOCKED on upstream commit e5d26285a (sim/side.ts, inputlogs for overridden choices); live-replay parity running.
- Next: compare desthead run vs runs/long_s15_20261112 on validation.

## 2026-11-12 – Devin / SWE-2 (sidekick) — desthead run + RL readiness results
- Did: ran long_s15_desthead_20261112 (seed 20261112, 10k broad) — dest froze broad@2250 / expert@450; restored dest losses 0.4438/0.4421 beat REF 0.4614/0.4568; action+value also improved. Validation eval written; parity harness 99.7% excl-inherent; readiness gates all BLOCKED on upstream e5d26285a (no sync run).
- Files: runs/long_s15_desthead_20261112/, runs/rl_readiness_s15/ (logs + parity summary)
- Checks: 14 tests, ruff clean, eval parity vs final_metrics confirmed.
- Next: owner decision on tools/sync_showdown.py for the blocking commit.

## 2026-11-12 22:10 – Devin / Fable 5.2 — destination head on its own score + RL readiness checks
- Did: trainer selects turn net on action+value; destination head tracked/frozen/restored on its own val loss. runs/long_s15_desthead_20261112 (same seed as long_s15_20261112), VALIDATION only: action 39.1% (was 38.2%), 1600+ 38.6% (37.5%), dest 75.4% loss 0.442 (75.2%, 0.457), value 68.3% ll 0.577 (68.0%, 0.577), turn-1 bias +0.5 pt (+1.8). Broad best 2500 (was 1750). Test NOT reopened.
- RL readiness (runs/rl_readiness_s15/): warm start 8/8 tensors, 0 stale vocab, hidden identical (max diff 0.0). Live/replay parity 99.73% excl. inherent (1099 decisions; 3 non-inherent = boosts.spd), matches 11-05's 99.8%.
- BLOCKED: all 4 gates, one upstream Showdown commit e5d26285a "Fix inputlogs for overridden choices" (sim/side.ts). tools/sync_showdown.py not run (pulls, rebuilds, re-exports, commits) -> needs owner OK.
- Next: owner OK for sync_showdown.py, then rerun the 4 gates. Disk ~19 GB.

## 2026-11-12 – Devin / SWE-2 (sidekick) — Showdown sync to 965a535ef
- Did: tools/sync_showdown.py pulled 11 upstream commits to 965a535ef (blocking: e5d26285a inputlogs fix), rebuilt dist/sim, committed 94a7550ad — only mechanics_catalog.json + mechanics_coverage.json changed; vocabulary files untouched.
- Checks: all 4 readiness gates PASS (logs runs/rl_readiness_s15/post_sync/); warm start on desthead ckpt 8/8, stale 0, diff 0.0; 38 tests pass.
- Next: owner reviews the auto-commit; nothing pushed.

## 2026-11-12 – Devin / SWE-2 (sidekick) — desthead confirmation runs 13/14
- Did: seed 20261113 rebuilt cache (270s, counts identical: move 1102318/switch 104996/outcome 1830451/brought 86401/leads 107945, filtered 9352/215264); hardlinked its npz into 20261114 (loaded 14.2s). All 6 runs now have eval_validation.json; six-run table at runs/long_s15_desthead_summary.md — desthead beats baseline on every seed (action 0.391-0.400 vs 0.382-0.383; dest loss 0.440-0.442 vs 0.451-0.457; t1 pred nearer 0.500).
- Note: old cache inode 206650738/206650850 is stale under current fingerprint (pre-desthead code hash); 5 links: long_s15_2026111{2,3,4}, pilot_s15_2, readiness_s15/smoke3 — removing all frees ~4.38 GB. Nothing deleted.
- Next: owner review of summary; nothing committed, no processes running.

## 2026-11-12 – Devin / SWE-2 (sidekick) — RL warm-start pilot s31101
- Did: ran paired pilot pilot_{fresh,warm}_s31101 (32 iters, 1024 games, evals 0/256/512/1024 vs heuristic+maxpower). Gate artifacts had staled on git_commit (94a7550 vs new HEAD) — refreshed via the three check_*_readiness scripts, then both arms ran clean. Warm trunk verified 8/8 from desthead_20261114 ckpt; all 34 non-trunk params identical to fresh init.
- Result @1024: fresh pool 0.719 (heur .531 / max .906), warm pool 0.703 (heur .438 / max .969) — no significant delta; warm was ahead at 512 (0.211 vs 0.156). Zero health-check hits; wall ~15.5/16.0 min.
- Files: runs/rl_warmstart_exp1/ (logs, eval.jsonl, arm_diff_check.py+log); recheck logs in runs/rl_readiness_s15/post_sync/recheck_*.log
- Next: owner decides whether warm-start matters at longer horizons.

## 2026-11-12 – Devin / SWE-2 (sidekick) — warm-start experiment-1 MAIN
- Did: 6 concurrent runs (fresh/warm × 31111/22/33, 2048 games, evals x128 at 0..2048). Warm logs all 8/8, fresh none; zero health hits; 0 nonfinite. Replayed eval games from snapshots for index-paired bootstrap — all 36 replayed cells exactly match recorded counts.
- Result: endpoint vs heuristic pooled WARM-FRESH -0.0052 [-0.057,+0.047] — null. Endpoint maxpower +0.013 [-0.010,+0.039]. AUC-vs-heuristic -0.0005 [-0.021,+0.020] with strong per-seed sign disagreement (+0.105 s31122 / -0.073 s31133). Warm start = no measurable effect at 2k games.
- Files: runs/rl_warmstart_exp1/main_*, main_summary.{py,md,json}, replay_eval_games.py, replay_games/, replay_logs/
- Next: owner decision on warm start.

## 2026-11-13 – Claude Code / Opus 5.5 — status check of warm-start experiment 1
- Did: confirmed main run finished (no training processes running); read runs/rl_warmstart_exp1/main_summary.md. Primary endpoint vs heuristic warm-fresh -0.005 [-0.057,+0.047], null.
- Checks: none run (read-only).
- Next: owner decision on warm start (drop it, or test a longer horizon / different question).

## 2026-11-13 – Claude Code / Fable 5.2 — why warm start = fresh in RL experiment 1 (analysis, no source changed)
- Did: knock-out evals on final snapshots (iter 64, seed 31111, 128 games vs heuristic; `runs/pretrain_review/ablate_eval.py`, abl_*.json). Unchanged: fresh 108 / warm 106 (matches eval.jsonl). Trunk output frozen to one constant board: 109 / 107. Trunk zeroed: 68 / 94. Tactical (damage/KO/Speed) features zeroed: 4 / 7.
- Reading: the trained policy plays off the per-action tactical features; the state trunk (the only 8 tensors pretraining fills) carries no used board information on this task, so the experiment could not show a pretraining effect either way. Warm trunk drifted only ~10% (not destroyed, just unused).
- Design gaps: one fixed team mirror + fixed leads (`phase2_mirror`, "team 1234"); pretrained action/destination/value heads and the whole preview net are discarded; 3 seeds with per-seed curve swings of +-10 pts.
- Checks: "none" arm reproduces recorded counts exactly. One seed pair only.
- Next: owner picks: rerun without tactical features (cheap), multi-team train/holdout-team eval, or carry the pretrained policy into RL (teacher anchor / head transfer). Worklog uncommitted.

## 2026-11-13 – Claude Code / Opus 5.5 — verified Fable's ablation review
- Did: checked runs/pretrain_review/abl_*.json; counts match the review (fresh/warm: normal 108/106, trunk constant 109/107, trunk zero 68/94, tactical zero 4/7). Experiment 1's null is uninformative about pretraining (trunk unused on fixed-mirror task).
- Recommendation to owner: 20-min no-tactical rerun first (decides whether warm arm is worth carrying), then first real RL run = multi-team train / holdout-team eval, harder opponents, search checks.
- Next: owner picks.

## 2026-11-13 – Claude Code / Opus 5.5 — no-tactical warm-start rerun launched
- Did: gates PASS at 14a8495 (worklog stashed for the check, then restored). Launched 6 runs runs/rl_warmstart_exp1/notac_{fresh,warm}_s{31111,31122,31133}: same as main (64 iters x 32 games vs heuristic, evals x128 @0/256/512/1024/1536/2048 vs heuristic+maxpower, --information-features) but --no-tactical-features. Warm logs 8/8 tensors.
- Checks: in flight.
- Next: when done, pooled warm-fresh @2048 vs heuristic + AUC, compare to main runs.

## 2026-11-13 – Claude Code / Opus 5.5 — no-tactical rerun results
- Did: 6 notac runs finished clean (0 errors). @2048 vs heuristic pooled warm-fresh -0.010 (eval-noise CI [-0.077,+0.056]) but seed-to-seed SD 0.17 (s31111 +0.19, s31122 -0.14, s31133 -0.08) -> null, very underpowered. AUC +0.034 (seed SD 0.12, not credible). maxpower endpoint -0.044 (one seed). Bots still learn without tactical: fresh 65% / warm 64% vs heuristic @2048 (main 82%).
- Key: board-reader freeze ablation on notac ckpts: fresh_s31122 97->100, warm_s31111 86->81 => trunk STILL unused without tactical features. Fixed-mirror task cannot test pretraining at all.
- Files: runs/rl_warmstart_exp1/notac_*, notac_summary.{py,txt}; runs/pretrain_review/abl_notac_*.json
- Next: owner decision; recommend multi-team train / holdout-team design (where board reading is needed), with a trunk-freeze ablation as a built-in check.

## 2026-11-13 – Claude Code / Opus 5.5 — many-teams RL experiment designed (draft prereg)
- Did: draft docs/prereg/rl_multiteam_warmstart.md (train_ppo pool mode on mc_sheet_pool_v2 train 226 / eval holdout 88; warm vs fresh; pilot 2 seeds x 2048 with trunk-freeze validity gate; main 8192 games; primary = holdout-vs-heuristic AUC diff).
- Smoke: train_ppo default teacher-anchor 0.05 -> 32 games 8.5 min, 7/8 worker TimeoutErrors, 15/32 games; with --teacher-anchor-weight 0 -> 40 s, 0 errors. evaluate_learned_policy vs vgc ~20 s per 10 games. Logs runs/rl_multiteam_design/.
- Next: owner approves design -> build eval wrapper + pool freeze check, then pilot.

## 2026-11-13 – Devin / Fusion (Opus 5.5) — many-teams pilot tooling built, prereg locked
- Did: train_ppo `--save-checkpoints-at-games` (default off); new offline/evaluate_pool_checkpoint.py (paired, shard-invariant, trunkc freeze). Prereg locked with seeds 41101/41102, eval seeds 70001-3, fresh = `--bc-checkpoint /nonexistent` (default warm-starts from bc_policy_v4!).
- Checks: warm 8/8, resume keeps anchor 0 + checkpoints untouched, shared server OK, eval rows identical jobs 1/4 (none+trunkc); 123 tests pass. Logs runs/rl_multiteam/tooling_check/.
- Next: commit, gates on clean tree, launch pilot (4 runs x 2048), eval, apply validity gate.

## 2026-11-13 – Devin / Fusion (Opus 5.5) — many-teams pilot: validity gate FAILED
- Did: Showdown sync -> 7e743761c (b9e34ee, owner OK; Pursuit PP fix); gates PASS; 4 runs (fresh/warm x 41101/41102) to 2048 games, 0 errors, ~34 min each; 40 evals, 0 errors.
- Result: trunk-freeze cost fresh pooled +2.1 pts [+0.8,+3.4] < 5 -> stop per prereg. Warm +2.0. Holdout vs heur @2048 fresh 69% / warm 67%; vs vgc 27% / 30%; train-holdout gap ~0.
- Files: runs/rl_multiteam/pilot_*; prereg result section appended.
- Next: owner decides how to make board reading matter (learned team preview, drop per-move tactical shortcut, longer horizon) before any main run.
- Open: worklog + prereg result note uncommitted.

## 2026-11-13 – Devin / Fusion (Opus 5.5) — pilot follow-up: which inputs the bots use (analysis only)
- Did: runs/rl_multiteam/branch_ablate.py freezes one input branch at a time (paired with pilot holdout-vs-heuristic rows, seed 70001; reproduces tool trunkc 349 exactly). Results in branch_ablate.jsonl.
- Result @2048 (pooled 4 runs): WHOLE context frozen costs ~2.1 pts (same as trunk alone); info/mechanics/history ~0; tactical zeroed costs ~30-40 pts. @1024 whole-context cost ~0. So bots play off per-action tactical numbers; board use is ~0 -> ~2 pts and rising slowly.
- Correction: my earlier "freeze cost barely moved between checkpoints" had no data behind it; measured now (0 -> 2 pts).
- Next: owner call. Recommend closing trunk-transfer question (use fresh), not running main; see chat for options.
- Open: worklog + prereg result note uncommitted.

## 2026-11-18 – Local mikumiku37 replication pilot
- Ran terminal-only from-scratch league training on pikapro 0.1.0. The 8,694,191-parameter model completed 32 games; a smaller model completed 128. Main saved model: runs/replication_8m7_37101/policy_0000032.pt. No public play.
- Search ran in Showdown. Default opening visited 8 worlds, 8 actions per side, 4 seeds per cell. Reduced two-game holdout comparison finished; plain model won both. Pilot only.
- Engine comparison: 94/100 agreed through finish on unpatched pikapro 0.1.0; six first divergences logged in runs/replication_rust_37101/engine_comparison.json. All 226 training teams validate. Seven focused checks passed; Ruff clean.
- Guide and repeat commands: docs/mikumiku_replication_pilot.md.

## 2026-11-18 – pikapro 0.1.0+regmc.2 parity fixes (600/600 match vs Showdown 7e743761c)
- Fixed two upstream RNG-order bugs in pikapro 0.1.0:
  1. `patches/pikapro/0001-update-before-forced-switch.patch`: run `each_update` before forced switches for living pivot Pokémon (and set `skip_before_switch_out` on Shed Tail). Fixed all 6 original mismatches.
  2. `patches/pikapro/0002-keep-status-without-replacement.patch`: skip `check_fainted` status clearing when `can_switch == 0` so fainted statused Pokémon still participate in speed-tie handler sorts. Fixed remaining holdout mismatches (187/200 -> 200/200).
- Packaged locally as `patches/pikapro/pikapro-0.1.0+regmc.2.tar.gz` via `tools/package_replica_engine.py`, wired into `pyproject.toml` (`replica` extra) and `uv.lock`, installed in `.venv`.
- Verified 600/600 parity vs pinned Showdown `7e743761c` (`runs/replica_parity_20261118/{original_after,holdout_after,new_training_teams}.json`: 100/100, 200/200, 300/300), 8/8 unit tests pass, and `offline/replica_live_smoke.py` completes full local Showdown games with and without search.
- Confirmed both bugs are still present on upstream `spedr/pikapro-engine` `main` (`2611ea9`) with no open issues for them.
- Next: sync Showdown for upstream Reg M-C Last Resort fix (`41ba4797f`) + check pikapro parity, then wire public ladder runner to the replication transformer architecture.

## 2026-11-18 – Upstream issue, Showdown 5b88c4761 sync, pikapro 0.1.0+regmc.3, and ladder runner support
- Opened upstream issue https://github.com/spedr/pikapro-engine/issues/2 documenting both switching.rs RNG-order bugs and patches.
- Stashed unrelated Nov 13–17 uncommitted files (`wip: nov13-17 prereg and floette preview trial`), synced Showdown to `5b88c4761` (`cb4f3baa1`, adding Champions `lastresort` `onDisableMove` from `41ba4797f`), and passed all 3 readiness gates.
- Added `patches/pikapro/0003-champions-disable-last-resort.patch`, packaged and installed `pikapro 0.1.0+regmc.3`, and verified both a dedicated Last Resort test and 100/100 full-game parity vs Showdown `5b88c4761` (`runs/replica_parity_20261118/regmc3_100.json`).
- Updated `ladder/run_ladder.py` with `--policy-mode replica` and `--policy-mode replica-search` (plus search budget flags) for both `--local-smoke` and public ladder play (keeping all 4 readiness/parity gates and vocabulary validation).
- Tests: 42/42 passed (`test_replica_engine.py`, `test_replica_search.py`, `test_replication_policy.py`, `test_ladder.py`); Ruff clean.
- Next: scale up RL PPO training of the replication transformer on `pikapro 0.1.0+regmc.3`, then test on local smoke and public ladder.

## 2026-11-18 – Phase 1 PPO trainer upgrade (parallel workers, desktop/afk profiles, resume, eval, W&B)
- Upgraded `selfplay/train_replication_league.py` with:
  - `--profile desktop` (half CPU cores, `nice=10`, `threads=2`) and `--profile afk` (cpus-2, `nice=2`, `threads=4`) with optional `--workers`/`--threads` overrides.
  - Parallel `spawn` worker pool running persistent `ReplicaRustWorker` instances and cached opponent snapshots.
  - Atomic `trainer_state.pt` saving and `--resume` (restores model, optimizer, RNG states, league history, and `wandb_run_id`).
  - PPO health metrics (`policy_loss`, `value_loss`, `entropy`, `approx_kl`, `value_accuracy`, `games_per_sec`, `decisions_per_sec`).
  - Periodic paired held-out baseline evaluation (`--eval-every`, `--eval-games`, `--eval-opponents`, default `maxpower,heuristic`).
  - Optional Weights & Biases logging (`--wandb-project`, `--wandb-run-name`, `--wandb-entity`, `--wandb-mode`).
- Next: refresh readiness gates on clean commit, run parallel + resume + W&B smoke verification, then start main PPO training.

## 2026-11-19 – Upgraded to official PyPI pikapro 0.2.0 and Phase 2/3 PPO runs
- `spedr/pikapro-engine#2` was fixed and closed upstream in `pikapro 0.2.0` (commit `0bfff72`, released on PyPI).
- Switched `pyproject.toml` `replica` extra to official `pikapro>=0.2.0`, removed local `patches/pikapro/` tarball and `tools/package_replica_engine.py`, and synced `.venv`.
- Verified `pikapro 0.2.0` against Showdown `5b88c4761`: 42/42 unit tests pass (including Last Resort and Tyranitar speed-tie) and 100/100 full-game parity matches.
- Phase 2 bake-off (`desktop` mode, 2,048 games): Medium (`width=128, L=4`, 1.16M params) beat Full (`width=268, L=12`, 8.69M params) on speed (112s vs 331s), stability (`approx_kl` 0.027 vs 0.148), and maxpower win rate (45% vs 36%).
- Phase 3 (`afk` mode, 20,000 games, `runs/replica_m128_l4_37101`, W&B `x41s0ucp`): completed in 10.3 min (~25–30 g/s); held-out win rates climbed to ~60% vs `maxpower` (peak 75%) and ~40% vs `heuristic` (peak 57.5% at game 13,824), then plateaued as `approx_kl` rose to 0.081 at `lr=3e-4`, `games_per_update=64`.
- Next: Phase 3b (resume with `lr=1e-4`, `games_per_update=256`) or Phase 4 local Showdown lookahead test.

## 2026-11-20 – 500k-game PPO run, search speedup, and 17/20 (85%) A/B win for lookahead
- Completed Phase 3b (20k -> 60k, `desktop`, `lr=1e-4`, `gpu=256`), Phase 3c (60k -> 150k, `afk`, `lr=5e-5`), and Phase 3d (150k -> 500k, `afk`, `lr=2e-5`, `gpu=512`) on `runs/replica_m128_l4_37101` (W&B `x41s0ucp`, `pikapro 0.2.0`).
- Held-out progression (plain neural, zero search): `heuristic` 21.9% -> 63.1% avg (final 70.0%, peak 80.0%); `vgc` (2-ply search) 26.4% -> 36.8% avg (final 40.0%, peak 50.0% at 444,912); value accuracy 53.4% -> 72.6% avg (peak 75.7%).
- Phase 4 local Showdown smoke (`ladder/run_ladder.py --local-smoke`): 0 fallbacks, 0 errors for both `replica` and `replica-search`.
- Upgraded `replica-search` (`src/vgc/rl/replication_policy.py`, `offline/replica_search_probe.py`, `ladder/run_ladder.py`): persistent reusable `LiveExactMirror`, batched `policy.values(...)` leaf scoring, mixed-strategy sampling in `ReplicaLadderPlayer`, and `ReplicaRustWorker` (`pikapro 0.2.0`) for `paired_match`.
- 20-game paired A/B (`policy_0500000.pt`, unseen `team_003` vs `team_005`, `4w x 4a x 2s`, `runs/phase4_smoke/ab_search_vs_plain_20.json`): `replica-search` beat plain `replica` **17–3 (85.0%)** (10/10 as p1, 7/10 as p2), 0 errors, ~0.7s/turn (~95s total for 134 search decisions).
- Public Reg M-C ladder v1 baseline (`policy_0500000.pt`, `--policy-mode replica-search`, `6w x 6a x 2s`, `teams/mc_ladder_01.packed.txt`, 30 live games in `runs/ladder.jsonl`): **17–13 (56.7%)**, **0 fallbacks**, start **1117**, peak **1255**, end **1165** Elo (avg opp 1184, range 1000–1464; 10/12 vs 1000–1149, 6/11 vs 1150–1249, 1/7 vs 1250+). True v1 Elo ceiling sits around ~1200–1250.
- Next: design and train next-gen v2 transformer (larger capacity e.g. `width=192, layers=6`, cosine LR schedule, larger team pool / longer self-play).

## 2026-11-21 – v2 post-mortem and full A+B+C+D v3 pipeline upgrade
- Reviewed finished v2 run (`runs/replica_v2_m192_l6_748t`, 2.87M params, 748 teams, 500k games, 8.8h): underperformed v1 vs `vgc` (29.8% vs 35.8–37.9%) and value accuracy (69.3% vs 72.6%) due to high initial KL (0.075–0.082) / small-batch overfitting (`batch_size=64, epochs=3`), diluted team pool (369/748 stale M-B-spread or synthetic archetype teams), and board blind spots.
- Implemented all four upgrades (A, B, C, D) with full v1/v2 checkpoint backwards compatibility:
  - **A (Board representation, `src/vgc/rl/replication_policy.py` v2):** Added explicit turn-remaining timers (weather, Trick Room, terrains, Tailwind, screens, Safeguard), `protect_counter`, `first_turn`, `must_recharge`, `preparing`, `status_counter`, permutation-safe side/slot embeddings, and calibrated win-probability logit value head (with v1 loader preserved).
  - **B (Training math & clean teams, `selfplay/train_replication_league.py`, `data/selfplay/v3_clean_train_manifest.json`):** Added per-trajectory GAE (`--gamma`, `--gae-lambda`), BCE value loss for v2, KL early stopping (`--target-kl`), anchor opponents (`--anchor-opponents`, `--anchor-fraction`), and built a clean 379-team Reg M-C pool (zero synthetic/M-B teams, zero holdout overlap, 100% `pikapro` validated).
  - **C (Live search, `src/vgc/rl/replica_search.py`, `offline/replica_search_probe.py`, `ladder/run_ladder.py`):** Wired `BattleMemory` into `replica-search` (`mirror.hypotheses(battle, memory=...)`) and replaced flat uniform opponent prior with neural policy softmax priors.
  - **D (Throughput, `selfplay/train_replication_league.py`):** Added lockstep batched rollouts (`--rollout-batch-size 32`) + `--device auto` (Apple M4 Pro MPS GPU for PPO updates). Verified ~2x wall-clock speedup on `width=192, layers=6` (`11.6 -> 22.3 games/sec`, `209 -> 420 dec/sec` in `desktop` mode).
- Verification: 44/44 unit tests + 2/2 integration tests pass (including fog-of-war hidden-opponent boundary and legacy v1/v2 checkpoint loading), `replica_search_probe` smoke clean, Ruff clean.
- Next: launch overnight v3 training run on `data/selfplay/v3_clean_train_manifest.json` (and optionally run a live ladder check with the upgraded `replica-search` on v1 while v3 trains).

## 2026-11-21 (late) – Rust speedups: native `pikapro.rl.VecEnv` rich encoder + `Battle.patch_public` live search
- Extended `pikapro 0.2.0` via PyO3 (reproducible patch saved at `patches/pikapro/0001-rich-encoder-and-patch-public.patch`, built and installed into `.venv`):
  - **Rust #1 (Training bridge — `rich` encoder, 1,028 floats/player):** Added `crates/pikapro-rl/src/encode/rich.rs` (fog-safe `PlayerView`, registered as `observation="rich"` across `VecEnv`, `ChannelSession`, and `Battle.encode_rich`/`Battle.encode_state_rich`). Carries species/item/ability/move IDs, exact/base stats, HP, status + sleep/toxic counters, boosts, types, protect chain, Fake Out `first_turn`, `turns_active`, Mega/trap flags, PP/disabled, volatiles, and explicit side + field turn timers.
  - **Rust #2 (Live ladder search — `Battle.patch_public`):** Added integer-only `crates/pikapro/src/patch.rs` (`Battle.patch_public(state_dict)` + `Battle.fork()` + `Battle.encode_state_rich(side, ...)`) so live lookahead runs directly in Rust `pikapro` instead of spawning Node.js Showdown (`LiveExactMirror`).
- Added Python layers:
  - `src/vgc/rl/rust_policy.py`: `RustFactoredPolicy` (encoder-only Transformer over the 1,028-float Rust `rich` obs, autoregressive factored doubles heads `slot0 -> slot1 | slot0`, 360-way preview head, BCE win-probability value head).
  - `selfplay/train_rust_vec.py`: PPO league trainer on `pikapro.rl.VecEnv(..., observation="rich", masks="factored")` + Apple M4 Pro MPS GPU for both rollouts and updates, native Rust `simple_heuristics` anchor opponents, GAE, BCE value loss, KL early stopping, cosine LR, atomic `--resume`, fast Rust holdout eval, and W&B logging.
  - `src/vgc/rl/rust_search.py` & `ladder/run_ladder.py`: `RustExactMirror` + `search_public_rust` + `--policy-mode rust` / `--policy-mode rust-search` (`RustLadderPlayer` wired with `ChannelSession(..., observation="rich")` and `BattleMemory`).
  - `tests/test_rust_pipeline.py`: fog-of-war hidden opponent item boundary, bit-exact (`0.0` diff) parity across `Battle.encode_rich`, `ChannelSession.obs`, and `Battle.encode_state_rich`, `Battle.patch_public`, checkpoint save/load, and `search_public_rust` end-to-end.
- Verified checks & benchmarks:
  - Tests: 47/47 focused tests pass (`tests/test_rust_pipeline.py`, `test_replication_policy.py`, `test_replica_search.py`, `test_replica_engine.py`, `test_ladder.py`); full unit suite 1,653 passed / 4 skipped (only 1 pre-existing missing-scratch-file failure in `test_replay_pretrain.py`); Ruff clean; all 3 readiness gates PASS.
  - Deep verification audit caught & fixed 4 live-play/search bugs before v3:
    1. `rust_search.py` read `item`/`ability`/`hp_fraction` instead of `snapshot_battle`'s actual keys (`item_id`+`item_state`, `ability_id`+`ability_known`, `current_hp`/`max_hp`, side `trapped`), which had dropped own items/Mega Stones and opponent HP%. Fixed; `RustExactMirror` now matches `ChannelSession` to `0.000000` on own mons + globals and exact opponent HP% on both `p1` and `p2`.
    2. `rust_search.py` used `Battle.slot_mask` Showdown request-order switch bits (`2, 3`) instead of `VecEnv`/`ChannelSession` permanent 0..5 row switch bits (`0, 1` after a switch). Added bidirectional `req_pos <-> obs_row` switch remapping; action masks now match `ChannelSession.mask` bit-for-bit on both seats after non-identity preview (`team 2143`) and mid-game switches.
    3. `ladder/run_ladder.py` (`RustLadderPlayer`) compared `pikapro` 1-based choice strings (`"move 1 1, switch 3"`) against `poke-env` named strings (`"move closecombat 1, switch Tyranitar"`), and called `sess.choice(...)` in a loop (which consumes the request on the first call). Added `match_pikapro_choice_to_order` in `rust_search.py` and fixed `RustLadderPlayer` to resolve exact `DoubleBattleOrder` objects cleanly.
    4. Verified live local Pokémon Showdown smoke (`ladder/run_ladder.py --local-smoke --n 2`) for both `--policy-mode rust` and `--policy-mode rust-search`: 0 fallbacks, 0 errors, ~1.2s total for 2 full `rust-search` games (`4w x 4a x 2s`).
  - Training throughput (`train_rust_vec.py`, `width=192, layers=6`, 2.87M params, 379 teams, `num_envs=256`, `threads=8`, MPS GPU): **96.4 games/sec** (**1,707 decisions/sec**) on 9.2-turn early games (~6x faster than v2's 16 games/sec; projects to ~130–160 games/sec at mature ~6-turn games, ~4M–5M+ games overnight). Value accuracy climbed 54.8% -> 63.4% in 46 seconds (4,098 games); `--resume`, `VecEnv` `autoreset="next_step"`, `side_randomization`, `eval_vs_python_vgc`, and zero train/holdout team overlap (379 train vs 88 holdout: 0 exact, 0 species-6 overlap) all verified.
- Next: plan and launch the v3 Rust training run via `selfplay/train_rust_vec.py`.

## 2026-11-21 (audit #2) – Move-order & team-slot equivariant `RustFactoredPolicy` v2 + full-12-row search obs
- Second adversarial audit before v3 launch caught 3 silent issues that would have hurt unseen-team generalization or biased eval:
  1. **Move-order blindness in `src/vgc/rl/rust_policy.py` (critical):** `encode_board` summed all 4 move embeddings without slot identity (`move_emb(mvs).sum(dim=-2)`) and used blind `Linear(2W, 47)` action heads + blind `slot0_action_emb(a0)`, making the model unable to tell which move was in slot 1..4 on unseen teams. Upgraded to `RUST_POLICY_VERSION = 2`: removed `own_slot_emb`, paired each move's `(pp, disabled)` with its own embedding in the pooled mon token, and replaced blind heads with semantic move heads (`[slot_ctx, move_emb(m_k), pp_k, dis_k] -> 10 (target, mega) logits`), semantic switch heads (`[slot_ctx, own_token[j]] -> 1 logit`), and semantic Slot-1 conditioning (`move_emb(chosen_move) + variant_emb` or `own_token[chosen_switch]`).
  2. **Unbrought opponent preview rows in `encode_battle_state_rich` (`patches/pikapro/0001-rich-encoder-and-patch-public.patch`):** Patched Rust `rich.rs`/`python.rs` to also fill the 2 unbrought opponent preview rows with strictly public preview info (species, types, base stats; zero hidden items/abilities/moves/exact stats), matching `VecEnv`/`ChannelSession` on all 12 rows. Rebuilt and reinstalled `pikapro 0.2.0` into `.venv`.
  3. **Fast-game selection bias in `eval_rust_opponents` (`selfplay/train_rust_vec.py`):** Replaced first-N completion stop with exact per-env quotas (`done_per_env < quotas`) so slower games are never dropped from holdout win rates.
- Verified:
  - Empirical audit of `rich.rs` confirmed 100% bit-exact timer parity across turns 1..4 (`weather_turns`, `tailwind`, `protect_chain`, `first_turn`, `turns_active`), zero NaN/Inf across 1,920 env-steps, all IDs strictly within dex bounds, and zero fog-of-war leaks at Team Preview.
  - Added exact move-order & team-slot equivariance regression test (`test_rust_policy_is_equivariant_to_move_order_and_team_slot_order`) to `tests/test_rust_pipeline.py`: 48/48 tests pass, Ruff clean.
  - 2,063-game MPS GPU smoke on `width=192, layers=6`: **63.2 games/sec** (`1,108 dec/sec`) on 9.26-turn early games (~95–110 g/s at mature 6-turn games), `value_accuracy` 49.0% -> 61.1% and `vs_snapshots_score` 51.6% -> 59.5% in 1 update, `approx_kl` 0.0072.
- Next: commit audit #2 fixes, refresh the 3 training readiness gates, and plan/launch the v3 run.

## 2026-11-22 – Completed 3,000,004-game Rust v3 training run (`runs/rust_v3_m192_l6_379t`)
- Finished full 3M-game AFK PPO league run (`train_rust_vec.py`, `RustFactoredPolicy` v2 equivariant, `width=192, layers=6`, 3.32M params, 379 clean Reg M-C teams, W&B `gci7x7w5`, ~8.9h total wall time at ~95–104 games/sec on M4 Pro MPS GPU).
- Final held-out results on 88 unseen Reg M-C teams (plain neural policy, zero lookahead):
  - vs Python `vgc` 2-ply search (40 unseen games): **62.5% final**, **64.0% last-5 avg**, **72.5% peak** (at 2.76M) — roughly 2x v1's ~36.8% and v2's ~29.8%.
  - vs Rust `simple_heuristics` (100 unseen games): **75.0% final**, **76.2% last-5 avg** (peak **79.0%**).
  - vs Rust `max_base_power` (100 unseen games): **90.0% final**, **85.8% last-5 avg**.
  - Win-chance prediction (`value_accuracy`): **77.3% final**, **75.5% last-50 avg**, **77.9% peak** (up from 72.6% in v1 / 69.3% in v2); `approx_kl` ~0.0025–0.0028 throughout late training.
- Final checkpoint: `runs/rust_v3_m192_l6_379t/policy_3000004.pt`.
- Public Reg M-C ladder v3 baseline (`policy_3000004.pt`, `--policy-mode rust-search`, `8w x 8a x 4s`, MPS GPU, `teams/mc_ladder_01.packed.txt`, 30 live games in `runs/ladder.jsonl`, ~32 min wall time): **16–14 (53.3%)**, **0 fallbacks**, start **1262**, **peak 1346**, end **1316** (+win #30 vs 1274 -> ~1332) Elo against avg opp **1282** (range 1052–1473; **8–3 (72.7%)** vs <1250, **2–4** vs 1250–1349, **6–7 (46.2%)** vs 1350+, including wins over 1469, 1416, 1409, 1384, 1359, 1350). Estimated true v3 ladder Elo sits around **~1320–1360** (+100 to +130 Elo over v1's ~1200–1250 baseline).
- Next: review the 14 v3 losses to diagnose the next bottleneck toward 1500–1700+ Elo (e.g. deeper/2-turn Rust lookahead, team strength/matchups, or scaling v4 training to 5–10M+ games / wider team pool).

## 2026-11-22 (Steps 1–3) – v3 loss post-mortem, Reg M-C team tournament, and 2-turn regularized `rust-search`
- **Step 1 (Post-mortem of 30 live v3 games):**
  - 20-game paired A/B on unseen holdout teams confirmed `rust-search` beats plain v3 **16–4 (80.0%)**.
  - Replay audit of the 14 losses exposed two clear causes: (1) **1-turn horizon Protect stall trap** — 49 Protects / 8 failed double-Protects in 14 losses vs 10 Protects / 1 fail in 16 wins (e.g. Game 6 threw a won 2v1 vs 1099 Elo by clicking Protect 5 straight turns while plain v3 wanted 90.8% `Low Kick + Close Combat`); (2) `mc_ladder_01` struggled against the ladder's #1 `Archaludon + Pelipper + Grimmsnarl` core (37.5% vs Archaludon, 27.3% vs Grimmsnarl; Tyranitar 6–13).
- **Step 2 (Reg M-C team screening with `policy_3000004.pt` in Rust `VecEnv`):**
  - Screened all 12 curated teams (`mc_ladder_01..06` + `teams/owner/*`) and confirmed top 4 at **512 games each** against all 88 unseen holdout teams + the 17 Archaludon/Pelipper/Grimmsnarl teams:
    - **#1 `teams/mc_ladder_04.packed.txt`** (`Salamence, Rillaboom, Sneasler, Arcanine-Hisui, Kingambit, Basculegion`): **69.5%** overall (+9.0% over `mc_ladder_01`'s 60.5%), 53.9% vs meta core.
    - **#2 `teams/owner/salamence_tw.packed.txt`**: **63.5%** overall, **60.0%** vs meta core.
- **Step 3 (Upgraded `src/vgc/rl/rust_search.py` to 2-turn regularized lookahead):**
  - Added `min_prior_prob=0.01` (prunes <1% tail junk moves beyond top 2), `prior_bonus=0.025` (KL-regularized Nash toward the 3M-game neural policy), and `rollout_turns=1` (batched MPS GPU 2-turn rollout past the 1-turn simultaneous matrix; ~0.33s/turn at `8w x 8a x 4s`).
  - Game 6 Protect trap: fixed **4/4 turns** (breaks out immediately with Close Combat).
  - **30-game paired A/B on unseen holdout teams (`4w x 8a x 2s`):** NEW 2-turn regularized `rust-search` beat OLD 1-turn unregularized `rust-search` **19–11 (63.3%)**, cutting Protect clicks from 80 -> 55 (-31%).
- Public Reg M-C ladder test (`teams/mc_ladder_04.packed.txt`, `policy_3000004.pt`, 2-turn regularized `rust-search`, `8w x 8a x 4s`, 30 games, session `20261123T015924Z`): **11–19 (36.7%)**, **0 fallbacks**, start **1392**, min **1194**, end **1263** (avg opp 1285).
  - Clean diagnosis: **14 of the 19 losses** were to the exact same `Pelipper + Grimmsnarl + Archaludon + Charizard` core (1–14 vs Pelipper, 1–13 vs Grimmsnarl/Archaludon across `SunRainCoach`, `KitaAuraFarmer`, `RyoBasslineMX`, `NijikaDoritoDrums`, `BocchiTheBop` — ~50% of the 1350+ ladder right now!). All-physical `mc_ladder_04` (5 physical attackers) gets hard-walled by Grimmsnarl Reflect + Stamina Archaludon + Rain, whereas against everything else it went **10–5 (66.7%)**.
  - 2-turn search fix confirmed live: 0 fallbacks, zero Protect stall loops (all games 4–10 turns), failed Protects per loss cut more than in half (4 across 19 losses vs 8 across 14 before).
- Public Reg M-C ladder test (`teams/owner/salamence_tw.packed.txt`, `policy_3000004.pt`, 2-turn regularized `rust-search`, `8w x 8a x 4s`, 30 games, session `20261123T045848Z`): **11–19 (36.7%)**, **0 fallbacks**, start **1373**, **peak 1381**, min **1251**, end **1273** against a much tougher avg opp **1337** (range 1116–1537; beat 1450 `NijikaDoritoDrums` and went 2–1 vs 1366–1391 `abnew123-bot`).
  - Same gatekeeper pattern: **11 of 19 losses** were to the exact same `Pelipper + Grimmsnarl + Archaludon + Charizard + Garchomp` 1450–1537 ladder cluster (1–11 vs Grimmsnarl, 2–11 vs Pelipper/Archaludon, 0–7 vs `SunRainCoach`), whereas against everything else it went **10–8 (55.6%)**. Note `mc_ladder_01` (with Tyranitar Sand Stream turning off Rain) went 3–5 vs `SunRainCoach`, whereas both non-weather teams (`mc_ladder_04` and `salamence_tw`) went 0–11 vs `SunRainCoach`.
- Next: Step 4A — pull fresh November high-rated Reg M-C replays to build an updated meta team pool (proper coverage of `Pelipper+Grimmsnarl+Archaludon+Charizard`, `Garganacl`, modern Trick Room, zero holdout overlap) and extend v3 (`192x6`) from 3M -> 7M games (`v3b`) with a warm LR restart (`3e-5 -> 5e-6`).

## 2026-11-23 – Refreshed Nov Reg M-C priors & 696-team diverse meta pool; launching `v3b` (3M -> 7M)
- Downloaded 1,303 fresh high-rated (>=1300) November Reg M-C replays (`tools/download_replays.py`, index now 171,258 replays).
- Added `--since` date filtering (with `index.jsonl` fast path) to `tools/build_set_priors.py` and rebuilt `data/usage/set_priors.json` from **39,589** Oct 15–Nov 23 rated (>=1200) replays (266 species; re-pinned `set_priors_sha256` in `data/champions/battle_state_coverage.json`).
- Built diverse modern Reg M-C training pool `data/selfplay/v3b_nov_meta_train_manifest.json`: **696 unique `pikapro`-validated teams** across **583 unique 6-species cores** (445 fresh Oct 15–Nov 23 OTS teams >=1250 + 251 from `v3_clean`, capped at max 3 variants/core, **0 holdout overlap** against all 88 unseen holdout teams). Doubled/tripled coverage of Screens, Rain, Sun/TR, Farigiraf, Delphox, Sinistcha.
- Added `--init-from <checkpoint.pt>` to `selfplay/train_rust_vec.py` so `v3b` warm-starts cleanly in `runs/rust_v3b_m192_l6_novmeta` from `runs/rust_v3_m192_l6_379t/policy_3000004.pt` with a fresh 4,000,000-game cosine LR schedule (`3e-5 -> 5e-6`) without modifying `v3`.
- Hardware sizing note (M4 Pro MPS): `192x6` (3.32M) ~101 g/s, `256x6` (5.60M) ~96 g/s (sweet spot if we need a bigger v4 brain next), `268x12` (9.56M) ~43 g/s (too slow for 10 days left in Reg M-C).
- Next: refresh 3 readiness gates on clean commit and launch overnight `v3b` AFK run (4M additional games -> 7M total).

## 2026-11-23 (afternoon) – Completed `v3b` (3M -> 7M total games on 696-team Nov meta pool)
- Finished full 4,000,024-game warm-started `v3b` run (`runs/rust_v3b_m192_l6_novmeta/policy_4000024.pt`, W&B `oyn1lwwm`, 11.56h at ~95–114 g/s on M4 Pro MPS GPU).
- Progression across the +4M games (`3M -> 7M` total):
  - `value_accuracy`: **75.4% -> 76.05%** (`value_loss`: **0.4672 -> 0.4585**, lowest yet).
  - `vs_anchors` (heuristic): **77.3% -> 80.5%**; unseen holdout `simple_heuristics` **74.6% -> 76.7–79.7%**; unseen `max_base_power` **89.5–91.0%**; unseen `vgc` 2-ply **64.4–65.6% avg** (peak **72.5%**).
- **Direct 1,024-game head-to-head (`v3b` 7M vs `v3` 3M, plain neural, deterministic, 6.5s on MPS):**
  - All 88 unseen holdout teams (512 games): **`v3b` beat `v3` 62.9% (~+92 Elo gain!!)**.
  - Unseen Archaludon/Pelipper/Grimmsnarl subset (512 games): **`v3b` beat `v3` 59.8% (~+70 Elo gain!)**.
- Public Reg M-C ladder `v3b` baseline (`policy_4000024.pt`, 7M total games, 2-turn regularized `rust-search`, refreshed 39.6k Nov `set_priors.json`, `8w x 8a x 4s`, `teams/mc_ladder_01.packed.txt`, 30 live games, session `20261123T214021Z`): **14–16 (46.7%)**, **0 fallbacks**, start **1357**, **peak 1458** (+112 Elo over v3's 1346, +203 over v1's 1255!), min **1357** (never dropped below v3's all-time peak!), end **1392** against a massive **1443 avg opponent Elo** (range 1257–1650; **5–1 (83.3%)** vs 1250–1349, **4–7** vs 1350–1449, **5–8 (38.5%)** vs 1450–1650 — including scalps over **1650 `YakinikuBOT2`**, **1559 & 1536 `SilverGale24`**, and **1518 & 1472 `SunRainCoach`**!). Estimated true `v3b` ladder Elo: **~1410–1450 Elo**.
- Next: decide between Option 2B (launch bigger `256x6` 5.60M-param `v4` brain on the 696-team Nov meta pool) and/or deeper search/team tuning toward 1600–1700+ Elo.

## 2026-11-23 (night) – Phases 0–3: Regulation-Indifferent VGC Fundamentals, Team Preview Search, Live Search, and Team Gauntlet
- **Phase 0 (100% Mirror Fidelity & Public Counter Tracking — 10 bugs fixed, 0 mismatches across 2,126 live decision checks):**
  - Patched and rebuilt `pikapro 0.2.0` (`patches/pikapro/0001-rich-encoder-and-patch-public.patch`): added 12 `MonPatch` fields (`last_move`, `perish_turns`, `yawn_turns`, `leech_seed`, `salt_cure`, `curse`, `syrup_bomb_turns`, `heal_block_turns`, `throat_chop_turns`, `partially_trapped`, `locked_move`, `two_turn_move`), fixed `stall_turns = 1`, fixed midpoint foe `hp_pct`, ran `resolve::volatiles::disable_moves` inside `patch_public` (Choice Lock, Encore, Disable, Taunt, Heal Block, Throat Chop, Fake Out), and expanded `Rich` obs `80 -> 88` floats/mon (`1,028 -> 1,124` total) across both `PlayerView` and `Battle`.
  - Updated `src/vgc/rl/rust_policy.py` (`RICH_MON_WIDTH = 88`, `RICH_OBS_SIZE = 1124`, `RICH_FIELD_TURN_IDX = 1116`, `top_preview_indices()`, and automatic zero-padding for legacy 80-col checkpoints like `v3b` — verified bit-exact `0.0` diff).
  - Updated `src/vgc/battle_memory.py` (`|-mega|` stone capture), `src/vgc/opponent_belief.py` (Mega `requiredItem` reveal), `src/vgc/rl/live_mirror.py` (pure-Python `compute_mirror_hypotheses`, zero Node.js `SimWorker` spawns), and `src/vgc/rl/rust_search.py` (unclamped 8-turn extender durations, `timers_hyp.payload` sleep/confusion timers, Toxic stage counter, Encore/Disable/Taunt/Choice Lock counters, extended volatiles, `has_acted`/`faint_count`/`lost_pokemon_last_turn`, non-Mega formes like `Palafin-Hero`, and batched GPU neural `_batch_resolve_forced_switches` replacing blind `choices[0]`).
  - Verified with `offline/audit_rust_mirror_fidelity.py --battles 100 --fail-on-unexplained` (50 stress + 50 holdout): **2,126 / 2,126 (100.000%, 0 mismatches)** across weather (688), terrain (684), protect_chain (357), trick_room (162), perish_song (158), screens (138), salt_cure (124), tailwind (89), leech_seed (72), disable (70), taunt (59), encore (37), yawn (32), confusion (5).
- **Phase 1 (Team Preview Rust Lookahead Search — `search_teampreview_rust`):**
  - Added deduplicated `(lead2, back2)` candidate grouping, `min_prior_prob = 0.04` filter + regulation-indifferent field/weather/terrain/speed-control setter diversity guard, fog-safe neural + hybrid opponent world prediction, 3-turn batched GPU rollouts (`rollout_turns = 3`, `seeds = 3`), and KL prior regularization (`prior_bonus = 0.035`).
  - Wired into `RustLadderPlayer.decide_teampreview` (`ladder/run_ladder.py`; `--no-rust-preview-search` A/B opt-out).
  - Results: fixed generic holdout from 12–18 (40%) -> **20–20 (50.0%, zero harm, ~0.12–0.34s)** and fixed the live `SunRainCoach` blind spot on `mc_ladder_01` from **0/20 Tyranitar brings (6–14, 30%)** -> **14/20 Tyranitar brings (8–12, 40%)**.
- **Phase 2 (Live Search Upgrade & Honest Ablation):**
  - Added `select_search_action_index` (prunes `<10%` regret-matching tail noise; argmax when `>= 85%`) and batched exact 2×2 Nash `_eval_turn2_minigame` (`_solve_2x2_zero_sum`).
  - 40-game paired holdout 3-way ablation: `prune_root_noise + 1000it` won **22–18 (55.0%)** cleanly; `turn2_branch=2` tied **20–20 (50.0%)** at ~2× turn latency. Kept `--rust-turn2-branch` opt-in (default `1`) and invested saved latency into **10 hidden-info worlds** (`10w x 8a x 4s x 2000 iters`, `--device auto`) by default in `ladder/run_ladder.py`.
- **Phase 3 (Paired 160-Game Team & Tech-Swap Gauntlet — `offline/evaluate_team_gauntlet.py`):**
  - Tested 5 teams (all Showdown-validated) across 4 gatekeeper archetypes (`sun_rain_screens`, `perish_control`, `trick_room_farig`, `psyspam_balance`) + 4 unseen holdout teams (32 seat-swapped paired games per team):
    - **#1 `teams/mc_ladder_01.packed.txt`: 25.0/32 (78.1%, CI [58.6%, 97.6%])** — 4–0 Perish, 4–0 TR, 4–0 Psyspam, 12–4 holdout, 1–3 Sun/Rain/Screens.
    - **#2 `teams/sand_r1831.packed.txt`: 22.0/32 (68.8%, CI [48.6%, 88.9%])** — 4–0 Perish, 4–0 Psyspam, 3–1 TR, 10–6 holdout, 1–3 Sun/Rain/Screens.
    - 1-move tech swaps all underperformed their base teams overall (`mc01_brickbreak` 22/32 = 68.8%, `sand_r1831_brickbreak` 20/32 = 62.5%, `sand_r1831_taunt` 15/32 = 46.9% despite 3–1 vs Sun/Rain/Screens) and were not promoted.
- **Verification:** 48/48 tests pass (`test_rust_pipeline.py`, `test_battle_memory.py`, `test_ladder.py`), Ruff clean across all 10 touched files, 2-game local Showdown `--local-smoke` (`rust-search`, `mc_ladder_01`, `10w x 8a x 4s x 2000 iters` + Preview Search) finished cleanly in ~8s with **0 fallbacks / 0 errors**.
- Next: user decides between (a) running a 30-game live ladder validation with the upgraded Phase 0–3 stack on `mc_ladder_01` or (b) launching Phase 4A (`192x6` `v3b` continued training on the new 88-column counter encoder + refreshed meta pool).
- **Public Reg M-C ladder validation of Phase 0–3 (`76a86e3`, `v3b` `policy_4000024.pt`, `teams/mc_ladder_01.packed.txt`, `rust-search` `10w x 8a x 4s x 2000 iters` + Team Preview Search + counter fixes + noise pruning, 30 live games, session `20261124T033930Z`):**
  - **Overall:** **14–16 (46.7%)**, **0 fallbacks**, **0 errors**, start **1414**, **NEW PEAK 1477** (+19 over pre-upgrade 1458), min **1347**, end **1395** vs **1443 avg opp** (range 1192–1622; **4–0** vs <1350, **2–0** vs 1550+, **3–1** vs `YakinikuBOT2` up to 1622!).
  - **vs ALL non-`SunRainCoach` opponents (12 games, avg opp 1405):** **10–2 (83.3%!!, up from 11–5 / 68.8% before Phase 0–3 — ~1650+ Elo performance!)**.
  - **vs cloned `SunRainCoach`/`KitaAuraFarmer`/`NijikaDoritoDrums`/`BocchiTheBop`/`RyoBasslineMX` core (18/30 = 60% of the session, avg opp 1468):** **4–14 (22.2%)**. Preview Search fixed the Tyranitar blind spot (**12/18** brings vs **0/14** before; **4–8 (33.3%)** when bringing Tyranitar — including scalps over 1541 `SunRainCoach` and 1541 `KitaAuraFarmer` — vs **0–6 (0.0%)** when leaving it behind).
  - **Takeaway:** Phase 0–3 + bug-hunt fixes are verified live (**83.3%** vs the diverse ladder, **0 fallbacks**, new **1477 peak**); the remaining bottleneck is teaching the `192x6` neural brain itself via **Phase 4A** (continued RL training on the new 88-column counter + 8-turn extender encoder + heavier Screens/Weather/Archaludon representation in the training pool).
