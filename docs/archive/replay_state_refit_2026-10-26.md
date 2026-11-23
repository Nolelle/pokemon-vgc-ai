# Replay state repair and corrected fits — 2026-10-26

The replay reader is repaired, and the corrected dataset has been regenerated and checked against the raw logs. The fitted position scores remain broadly similar. These are estimates from past games; they do not establish a stronger live bot.

## Preserved work

The requested preservation commits are `ff7faf3`, `7985b91`, and `988701a`, on `chore/checkpoint-20261026`. Repair work continues on `fix/replay-state-20261026`, which already included `origin/main` when work began.

## Data repair and evidence

- Schema 7 applies publicly logged opening effects before the turn-1 observation. Later reveals remain absent from earlier observations; team-preview observations remain preview only.
- Mega forms use their persistent team identity from `data/champions/species.json`, including Floette-Eternal and Meowstic-F.
- Opening, turn, and replacement-switch events share one state-update loop. Opening switches do not create action records.
- All 54,870 raw replays parsed, retaining 1,083,846 records: 109,740 preview, 782,228 turn, and 191,878 replacement-switch records. All outcomes and record identities were preserved.
- The independent raw-log check covers both perspectives of all 54,683 games that reach turn 1. Weather/terrain match in all 27,743 affected openings; net stat changes match in all 20,880 affected openings. The other 187 replays have no turn-1 decision.
- 711 action records have corrected persistent identities, including 117 derived preview selections checked against raw switch history. Some recover the fourth brought Pokemon that a duplicate Mega identity previously displaced. All other action labels and action-availability markers are unchanged.
- The refreshed health report reports zero preview-membership violations, versus the earlier 3,502 active snapshots with identity mismatches. The audit does not certify every replay mechanic.

New dataset: `data/bc/decisions_regmc_s7_20261026.jsonl.gz`. Older datasets are retained. Its SHA-256 content fingerprint is `5584447c47526a10e07995bb0735ca49c1299a80a4dbaa42e6057066e5218471`.

Evidence: `runs/replay_state_audit_20261026/audit.json`, `summary.json`, and `parse.log`; refreshed visual report: `runs/viz/data_health/index.html`.

## Corrected fit and sampling sensitivity

The table uses the fit that also accounts for the previous turn. Values are scaled so 100 means one Pokemon's full health. For example, +60 for a living Pokemon is an extra survival bonus beyond its remaining health. Negative numbers penalize harmful conditions. Each run used seed 20261018 and 30 game-level resampling rounds for approximate uncertainty bands.

| Dataset / sensitivity | Games with turn records | Alive bonus | Attack boosts | Defense boosts | Speed boosts | Burn | Paralysis | Poison | Sleep/freeze | Fresh Tailwind | Fresh screen |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Schema 6, same games | 54,683 | 59.5 | 18.5 | 16.2 | 17.3 | -39.5 | -18.9 | -33.8 | -55.8 | 61.6 | 34.3 |
| Schema 7, all games | 54,683 | 59.7 | 18.1 | 16.5 | 17.2 | -39.2 | -19.1 | -33.8 | -55.4 | 61.0 | 36.5 |
| Both pre-game ratings >=1234 | 47,011 | 59.6 | 18.0 | 16.6 | 17.1 | -41.2 | -19.8 | -35.5 | -54.7 | 60.2 | 36.5 |
| Seven repeated accounts omitted | 35,609 | 68.4 | 18.8 | 17.6 | 17.8 | -39.9 | -28.4 | -31.6 | -55.3 | 62.7 | 49.1 |
| At most 20 games per preview team | 16,274 | 68.6 | 18.0 | 17.5 | 15.4 | -40.6 | -25.9 | -35.1 | -47.6 | 59.6 | 36.6 |
| All three sensitivity restrictions | 13,967 | 70.3 | 18.0 | 18.6 | 15.1 | -39.7 | -27.0 | -32.2 | -47.7 | 54.0 | 46.4 |

The same-games parser correction barely moves most fitted values. Removing the repeated accounts changes the survival and screen estimates more than the parser repair does. Limiting repeated teams also raises the survival estimate. Attack/defense/speed boosts and Tailwind retain positive values across all checks. The evidence supports testing this family of scores, while keeping the precise values experimental.

The seven-account restriction is a sensitivity check, not an automatic deletion rule. Accounts: `scsme`, `scorecardpokemon`, `sccontrol`, `kvdn64`, `kvdn53`, `regmcaipilot13`, `swordsdancervgcbot`. Team caps use the unordered six-species preview roster and count both players, with seeded selection independent of the outcome. Full metadata and retained-game fingerprints are in each fit report.

The all-games corrected fit has prediction error 0.5690, versus 0.5798 for the original scorecard, on randomly held-out games (lower is better). This comparison tests predicting outcomes, not choosing good actions. Repeated teams and accounts can occur on both sides of that split. It is not a test on unseen teams.

Remaining limits: eventual outcomes are associations, not proof that a move causes a win; the previous-turn adjustment does not solve this. The rating restriction reduces exposure to the collection cutoff but cannot recover games never downloaded. Thirty resamples give rough uncertainty bands. The value fit has no weather/terrain features, and “fresh” means newly observed since the prior decision rather than exact remaining duration. No fitted values were enabled in the live bot.

## Reproduce

```sh
.venv/bin/python tools/parse_replays.py \
  --replays-dir data/replays/gen9championsvgc2026regmc \
  --out data/bc/decisions_regmc_s7_20261026.jsonl
gzip -n data/bc/decisions_regmc_s7_20261026.jsonl

.venv/bin/python tools/audit_replay_state.py \
  --data data/bc/decisions_regmc_s7_20261026.jsonl.gz \
  --previous data/bc/decisions_regmc_s6_20261021.jsonl.gz \
  --raw-replays data/replays/gen9championsvgc2026regmc \
  --output runs/replay_state_audit_20261026/audit.json

.venv/bin/python offline/fit_position_value_weights.py \
  --data data/bc/decisions_regmc_s7_20261026.jsonl.gz \
  --boots 30 --seed 20261018 \
  --output runs/eval/value_fit_s7_20261026.json
```

Use a new output filename to retain an existing dataset. The parser writes plain JSON (structured text) lines; gzip compression is a separate step. To reproduce the sensitivity fits, add `--raw-replays data/replays/gen9championsvgc2026regmc` and the corresponding `--min-pre-rating 1234`, repeated `--exclude-account NAME`, and/or `--max-games-per-team 20` switches. Each fit report stores its exact settings.

## Evaluation reporting and next experiment

The pool report now separately counts the live judge's completed calls, timeouts, errors, skipped calls, changed choices, and elapsed time for each arm. It keeps the direct exact-player counters separate. Reading these logs changes no choices.

Next, measure how often a wider search's preferred move under the experimental scores appears in the six choices offered to the live judge. Use recorded player-view ladder positions, fixed possible opponent replies and random futures, and report missing choices separately from reply-width effects and judge failures. This diagnosis must precede a change to fast-search ranking. A same-session comparison and fresh opponents with no training-team overlap remain required before promotion; public ladder blocks remain the final strength check.

## Verification

174 focused tests passed; two integration tests were deselected. Ruff and `git diff --check` passed. A separate Sol review found no remaining actionable parser, fit-equivalence, audit, or reporting issue. No public ladder games, model promotion, or live configuration change occurred. The full readiness checks and broad integration suite were not run for these offline changes.

The first identical-settings check used only two teams / eight games. It returned 2/8 and a degenerate between-team interval because both team results were identical; the harness flagged it. No strength conclusion used it.

The expanded check used 32 teams / 64 games with identical live-judge settings on both sides, seed 20261027 and four workers. It returned 25/64 (39.1%), with a team-grouped 95% uncertainty interval of 27.6–50.5%. The harness null check passed because 50% lies inside that interval. This small check can detect only large differences; it does not rule out subtle imbalance or establish a strategy gain.

Across both arms: 1,310 judge calls, 1,079 successful completions, 229 skips, 2 errors and 0 timeouts; 669 successful calls changed the fast-search choice. Average time was about 0.53 seconds per call, including skipped calls. Skips were replacement-switch decisions or fewer than two available candidates. The two errors were in the incumbent arm and retained the fast-search choice. Their causes were not retained by that run; the report now additionally keeps bounded failure examples with team, seed, turn and error text for future investigation. These errors remain unclassified, so a later strength experiment must report them rather than assume perfect judge execution.

Artifacts: `runs/replay_state_audit_20261026/live_judge_null_smoke.json` (tiny flagged check) and `live_judge_null_32.json` / `.log` (expanded check). These are local simulated games, not public ladder results.
