# Replay selection proposal — November 2, 2026

The inventory is complete. The proposed split has **47,096 training games, 5,832 development games, and 6,033 final-test games**. Training contains **6,081 player-game examples from players rated at least 1600 before the battle**. A player-game means one player's perspective on one game, not a unique person or a single move.

These are reviewed selection counts, not approval to start training. Observed moves are still partial evidence of the submitted choices. No model was trained, simulator updated, or ladder game played.

![Proposed allocation of all 65,011 games](../runs/pretraining_selection_20261102_v3/split_counts.png)

## What was selected

The fixed source is `data/bc/decisions_regmc_s7_20261101.jsonl.gz`: 65,011 Regulation M-C games, 1,281,380 records, schema 7. Its compressed SHA-256 fingerprint is `6eeebcd9a2a53ba718ab88f896692cdd6509cd247e83659a636216034b037ac7`.

The proposal requires the correct format, matching raw and parsed outcomes, six distinct preview members per side, one preview record per player, valid recorded health/stat ranges, and replay identities that reconcile with the raw logs. It uses rated games, omits the known project account, and separates exact duplicate logs. No other account was classified as a bot or deleted. Those basic checks found no failures in the final run and no exact duplicate logs.

The following games are set aside, with the original files preserved:

| Reason | Games |
|---|---:|
| Disclosed team sheets: different opponent information from our intended setting | 2,625 |
| Baton Pass: passed stat changes are not fully reconstructed | 1,376 |
| Illusion/replacement: apparent Pokémon identity requires further review | 610 |
| Transform: copied form/state requires further review | 122 |
| **Distinct games with one or more of the above reasons** | **4,683** |
| Additional older games containing a reserved test lineup | 1,367 |

The first four reasons overlap. They total 4,683 distinct games, not their arithmetic sum. This is a conservative first selection, not an exhaustive detector of replay-mechanics problems. Even a mechanically correct replay does not necessarily reveal the submitted action.

## Proposed split

Before reserving unfamiliar teams, the 60,328 remaining games split approximately 80/10/10 by battle start time: 48,262 / 6,033 / 6,033. Games sharing a start timestamp stay together. Every decision and both players in a game receive the same assignment. No timestamp fallback or timestamp anomaly was needed in this corpus.

The final counts below additionally remove earlier appearances of reserved test lineups from training and development. Dates are in America/Edmonton.

| Group | Games | All decision records | 1600+ player-game examples | Battle date range |
|---|---:|---:|---:|---|
| Training: examples the network may learn from | 47,096 | 925,046 | 6,081 | Sep 9, 03:10–Oct 19, 04:43 |
| Development: compare models and training choices | 5,832 | 114,118 | 1,564 | Oct 19, 04:44–Oct 24, 13:39 |
| Final test: assess the selected model | 6,033 | 116,598 | 2,054 | Oct 24, 13:43–Nov 1, 16:30 |

The 1600+ count uses each acting player's pre-game rating from the rating-change message. It does not use the shared replay rating or the possibly stale rating in a player-join message. Both wins and losses remain: the training subset contains 3,096 winning and 2,985 losing player-game perspectives.

Use these same game assignments for every learning task and for later fine-tuning (additional training specializing the model). Fine-tuning draws only from the 1600+ training subset. Development and final-test games must never be reintroduced through another file, training stage, cached representation, or old model initialization.

### Unfamiliar-team check

The final test contains 3,528 distinct sorted six-species lineups. A fixed hash ordering with seed `20261102` reserves 176 of them, approximately 5%. All earlier games containing any reserved lineup on either side are removed from training/development: 1,166 training games and 201 development games.

**445 final-test games involve at least one reserved lineup.** This is a subset of the 6,033 test games, not an additional test set. All 176 reserved lineups have zero appearances in retained training or development games. The other player's lineup can still be familiar. This grouping represents six-species lineups, not exact hidden moves/items/stat choices or every similar team variant.

The ordinary future-game test intentionally allows familiar opponents and lineups: 873 lineups and 2,054 accounts occur in both training and test. That measures performance on later games, not performance only on unknown accounts or teams.

## What the training records can teach

These categories overlap and use different units. Move/target/switch counts are individual active-slot observations; preview counts are player-game selections; outcome counts are decision records.

| Evidence in the training group | Broad training pool | Only 1600+ acting players |
|---|---:|---:|
| Observed moves, with sensible basic state fields | 973,633 | 63,790 |
| Observed moves with a logged target | 971,854 | 63,760 |
| Observed switches during normal turns | 107,245 | 7,383 |
| Four distinct brought Pokémon and two leads, matching raw appearances | 75,605 | 5,106 |
| Decision records with a matched win/loss, excluding timer/offline endings | 915,867 | 60,628 |
| **Certified complete submitted two-Pokémon actions** | **0** | **0** |

There are also 184,226 unknown active-slot actions in training. They must not become pass labels. A known move with an unknown target can potentially teach only the move. A Pokémon that never appeared cannot be invented to complete a bring-four label.

All 130,022 parsed preview actions matched an independent raw-log appearance reader. This validates correspondence to appearances, not the original selection request. Complete bring-four candidates require four distinct preview members and two distinct leads. Bench order is not certified. Later appearances establish an answer to learn; they must not be inserted into the earlier input shown to the network.

The complete action scorer still needs a suitable training contract. Ordinary spectator logs lack full legal-choice requests and usually exact own-team details. Called/locked moves, redirected targets, and involuntary movement can differ from what the player selected. Current move/target/switch counts are therefore **observed candidates**, not a ready-to-use complete-action training file. Replacement-switch records are counted separately and are not automatically certified player choices.

The outcome count includes preview, normal-turn, and replacement records. It estimates outcomes under the recorded players' play, not optimal-play win probabilities. Forfeits are retained; timer/offline results remain recorded but do not teach the proposed outcome task.

## Remaining decisions before training

1. Build and verify the per-task label masks: explicit instructions telling training which known parts of each record it may learn from. Complete-action training requires additional evidence or a method explicitly designed for partial actions.
2. Preserve unknown private information. Public replays do not supply the full information a player had about their own team. Do not fabricate exact moves, items, stat choices, or legal alternatives to make the records look complete.
3. Balance repeated games and teams in the training sampler (the part choosing which examples to show). One lineup appears in **34.0% of training games**, and one account in **11.6%**. These are participation shares, counting a lineup only once in a mirror match. The inventory measures concentration; it does not yet apply sampling weights. Give each game controlled influence so long games do not dominate, and choose any roster-frequency weighting using development results only.
4. Keep final-test identities fixed after this review. None of its games belong to the 54,870-game baseline used in the documented October replay fits. This does not establish universal non-exposure: some older model files have unknown training provenance. Start a fresh model or establish its exact data history before claiming an untouched test.
5. Preserve the collection caveat: the original download threshold was a post-game rating and can bias which outcomes were collected near the cutoff. Pre-game selection does not recover missing games. Coverage is a bounded public replay collection, not the entire ladder.

This selection does not clear the separate Showdown-engine readiness block or establish playing strength. No training or readiness approval is implied.

## Reproduction and checks

Run from the project root, choosing a new output directory; the tool refuses to overwrite an existing one:

```sh
.venv/bin/python tools/inventory_replay_pretraining.py \
  --data data/bc/decisions_regmc_s7_20261101.jsonl.gz \
  --raw-dir data/replays/gen9championsvgc2026regmc \
  --baseline-index runs/replay_refresh_20261101/index_before.jsonl \
  --out runs/pretraining_selection_NEW
```

- [Final per-game manifest](../runs/pretraining_selection_20261102_v3/manifest.jsonl): assignments, ratings, exclusions, source fingerprints, and per-player counts.
- [Machine-readable summary](../runs/pretraining_selection_20261102_v3/summary.json).
- [Independent verification](../runs/pretraining_selection_20261102_v3/independent_verification.json): all 65,011 raw joins and 130,022 pre-game ratings matched the existing independent reader; record/game totals, chronology, reserved rosters, and input fingerprints reconciled.
- [Independent check scripts](../runs/pretraining_selection_review_20261102/verify_inventory.py) and [raw-metadata check](../runs/pretraining_selection_review_20261102/independent_check.py).
- 42 focused tests passed: 5 new inventory checks, 2 existing raw-audit checks, and 35 replay-parser/format checks. Ruff passed for the new code. An independent reviewer checked split isolation and the corrected empty-player-name case. No simulator, training, or broad battle suite ran.

Earlier `pretraining_selection_20261102` and `_v2` run directories are retained diagnostic drafts; `_v3` is the current proposal. The first draft exposed a blank initial player-name line in one replay, fixed without changing the replay or its parsed dataset. The second draft preceded the conservative mechanics-review exclusions. Both drafts retain the source used to produce them.
