# Project handoff log

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
