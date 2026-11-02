# Partial replay training data and balanced sampling — November 2, 2026

The data-handling fixes are implemented. Prepared records now specify exactly which observed answers may be used, and training examples can be sampled without allowing long games or repeated lineups to dominate. The original replay files and the reviewed game split remain unchanged. No network was trained.

The missing information itself has not been recovered: public spectator replays do not reliably contain the original complete two-Pokémon choice, all legal alternatives, or the player's full private team details. We preserve that limitation instead of inventing answers.

## Prepared data

Current output: `data/bc/pretraining_regmc_s7_20261102_v2/`. The directory contains separate compressed training, development, and final-test files; broad and 1600+ training sampling plans; and a fingerprinted summary.

| Group | Games, unchanged from selection | Prepared records |
|---|---:|---:|
| Training | 47,096 | 925,046 |
| Development: compare learning choices | 5,832 | 114,118 |
| Final test: assess the selected model | 6,033 | 116,598 |
| **Total** | **58,961** | **1,155,762** |

The 125,618 records from the previously excluded/review groups are not exported. Both players and every kind of decision from a game keep the same assignment. No split was recomputed.

## What can contribute to learning

A mask is an explicit yes/no instruction telling training whether a particular answer is available. Disabled answers are stored as null and are not returned by the task reader. Missing actions never become a pass choice. Outcome learning remains possible even when a move is unknown.

| Training task | Broad pool: enabled answers | Only acting players rated 1600+ before play |
|---|---:|---:|
| Ordinary move observations, matched to the raw log | 604,210 | 41,223 |
| Early switch destinations | 100,853 | 6,975 |
| Four brought Pokémon, as an unordered set | 75,597 | 5,106 |
| Ordered lead pair | 93,835 | 6,074 |
| Verified final outcome at a recorded position | 915,867 | 60,628 |
| Originally selected target | **0 — disabled** | **0 — disabled** |
| Complete submitted joint choice | **0 — disabled** | **0 — disabled** |

These counts overlap and use different units. A turn can supply two move observations and one outcome. They are not counts of independent games.

The action checks read structured protocol fields, not chat messages or player names. A game with a called move or a detected forced/repeated move mechanism has move teaching disabled conservatively for the whole game. Charge/recharge flags and move effects come from the committed Champions data. The detector is a conservative filter, not proof that every remaining event was freely chosen.

Switch teaching requires an early raw switch before moves, faints, forced movement, or item/ability resolution. Replacement and dragged-out Pokémon do not become voluntary-switch answers. Target teaching remains disabled because redirection and retargeting can hide the original selection. This intentionally gives up some usable evidence rather than pretending the target is known.

Bring-four and lead answers must agree with both the saved selection evidence and a fresh raw-log reader. Four distinct appearances can teach the brought set; a missing fourth does not prevent learning a separately verified lead pair. Bench ordering is not taught. Slightly fewer preview answers remain than in the initial inventory because the additional raw identity check is conservative.

The only model input stored is the existing pre-decision state. Later reveals, ratings, names, outcomes, and selected actions are stored outside that input. The preparation path does not fill in private sets or create a guessed legal-choice list. Timer/offline results are excluded from outcome teaching; ordinary wins, losses, and forfeits remain.

## How balancing works

For each task independently:

1. Give each eligible training game one starting unit of influence.
2. Divide that influence between eligible player perspectives, then between their known answers. Extra turns or two known slots do not automatically give a game more total influence.
3. Reduce game influence according to the frequency of the more common of its two six-species lineups. Specifically, divide by the square root of that lineup's training-game count. A lineup appearing on both sides counts once when measuring frequency.
4. Normalize the resulting probabilities for that task. The sampler draws a game/perspective group by its probability, then an eligible answer within it.

Both-role weighting matters: a common opponent lineup can dominate observations even when our own lineup is varied. Broad and 1600+ plans are computed separately using training games only. The stronger-player plan checks the acting player's pre-game rating. Unknown ratings cannot qualify. Development and final-test frequencies are unchanged.

![Training lineup concentration before and after balancing](../runs/pretraining_preparation_20261102/balance.png)

For outcome examples, the largest lineup participation share drops from **34.0% to 2.1%** of expected draws. For move examples, after conservative label filtering, it drops from **24.5% to 1.3%**. These are sampling-distribution changes, not measured improvements in battle strength.

The broad outcome distribution has an effective size of approximately **14,678 equally weighted games**, compared with 46,528 eligible games. This describes how unequal weights concentrate statistical influence; no games are deleted by the sampler. Rare lineups receive more relative emphasis, so whether this recipe improves learning must be judged on development data. In the smaller 1600+ pool, the largest account still participates in roughly 9–10% of weighted draws, depending on task; lineup balancing does not guarantee equal account influence.

## Protection against mixing training and testing

Preparation requires the exact reviewed selection fingerprint:

`7ccfcf95ac7bd1b5989762bd4d81f385ee019288bb2593e0fd1a245384ac072a`

A swapped training/test assignment is rejected even when the split counts remain identical. Input, raw-file, prepared-shard, and sampling-plan fingerprints are checked. Preparation refuses to overwrite an existing output directory. The task reader defaults to training and requires explicit final-evaluation mode to read the final test; test examples never receive training probabilities.

The normal task reader emits only enabled answers. It provides their normalized sampling probability and a stable `(replay_id, player, task, ordinal)` reference. The sampler uses those probabilities to draw references. Do not multiply by the same weights again after weighted sampling; that would apply the correction twice. An exhaustive iterator alone does not perform balanced random sampling.

## What this does and does not train

This is a prepared partial-observation dataset and a working sampler, not a launched training run. It can support separate predictions of observed facts and outcomes. Such extra prediction outputs can warm up shared board-reading representations.

**It does not pretrain the current complete-action scorer.** That network scores both Pokémon's actions together; filling an unknown partner action with zero or pass would manufacture a different input. Its action-scoring layers need an explicit compatible learning objective and suitable evidence, such as genuinely recorded legal requests and submitted choices. The existing old replay warm-start copies only the board-reading part of the network. This task has not silently changed that architecture or treated auxiliary prediction as full-choice training.

The collection's post-game rating bias, public-data coverage limits, older-model provenance uncertainty, and the separate Showdown readiness block still apply. These data checks do not establish model strength or engine readiness.

## Reproduction and checks

```sh
.venv/bin/python tools/prepare_replay_pretraining.py \
  --data data/bc/decisions_regmc_s7_20261101.jsonl.gz \
  --selection runs/pretraining_selection_20261102_v3/manifest.jsonl \
  --selection-summary runs/pretraining_selection_20261102_v3/summary.json \
  --selection-sha256 7ccfcf95ac7bd1b5989762bd4d81f385ee019288bb2593e0fd1a245384ac072a \
  --raw-dir data/replays/gen9championsvgc2026regmc \
  --out data/bc/pretraining_NEW
```

Use `vgc.bc.replay_pretraining.iter_task_examples` to consume enabled task answers. Load the saved plan with `vgc.bc.replay_pretraining.load_sampling_plan`, then use `vgc.bc.replay_sampling.sample_task_units` to draw training references. The legacy `BcTurnDataset` and old training entry point are not automatically switched to this contract.

- [Prepared summary and probability diagnostics](../data/bc/pretraining_regmc_s7_20261102_v2/summary.json).
- [Full independent verification](../runs/pretraining_preparation_20261102/verification.json).
- [Inspect a few prepared examples](../runs/pretraining_preparation_20261102/examples.json).
- [Verification script](../runs/pretraining_preparation_20261102/verify_prepared.py).

Fifteen focused tests pass, including unknown-label masking, held-out sampling exclusion, long-game influence, per-player ratings, input preservation, fingerprints, and a training/test swap rejection. The independent full check verified all 1,155,762 exported inputs against their original snapshots, reconciled every sampling probability, and drew 100,000 outcome examples from each plan with zero held-out draws. The largest lineup appeared in 2.058% of broad draws (expected 2.095%) and 2.442% of expert draws (expected 2.411%). Ruff and whitespace checks pass. Independent review found and verified the fix for the split-fingerprint requirement. No network optimization, simulator execution, or ladder games were run.

The first `data/bc/pretraining_regmc_s7_20261102/` export is retained as a diagnostic draft. `_v2` is the current export, produced with the required reviewed split fingerprint.
