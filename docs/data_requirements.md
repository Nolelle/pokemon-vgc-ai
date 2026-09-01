# Training data requirements

This document defines the minimum data needed to train or evaluate the battle agent.
It deliberately does not introduce a database, streaming system, or data warehouse.

## The unit of data

One training example is one decision for one player at one battle state. It must contain:

- a unique battle id, a sequence number for this decision, the current turn, and whether
  the game is requesting moves or replacement switches (one turn can contain both);
- fingerprints of both teams (a fingerprint is a SHA-256 hash of the packed team text);
- only the information that player could know before choosing;
- every legal joint action for the two active Pokemon;
- the teacher's chosen action and its score, stored as an answer rather than an input;
- explicit markers for unknown opponent information;
- the exact mechanics snapshot used by the model; and
- enough source information to reproduce the collection.

The teacher action is valid only when it appears exactly once in the stored legal-action
list. A decision with no legal actions, duplicate legal actions, or mismatched arrays is
invalid and must not be used for training.

## Information boundary

Training and live play must use the same information boundary. At decision time the agent
may use:

- its own complete six-Pokemon team;
- the opponent's six preview species;
- moves, items, abilities, forms, status, and field effects already revealed publicly;
- battle history already observed; and
- probability estimates built from prior public data.

It must not use the opponent's true unrevealed moves, item, ability, Stat Points, nature,
bring choice, privately rolled sleep/confusion duration, future turns, or final result.
Unknown facts stay as probability-weighted possibilities. The official local Pokemon
Showdown engine executes each predicted turn after those possibilities are constructed.

Every traced decision records this boundary under `notes.information_boundary`, using
the contract id `own_exact_opponent_public_beliefs_v1`. Each field is labelled:

- `known` when it comes from our registered team or a public battle message;
- `estimated` when it is a probability-weighted possibility from public prior data; or
- `unknown` when the player cannot know it before choosing.

The ledger includes our exact six sets, the opponent's preview and public reveals,
unknown brought Pokemon and next action, hidden spread possibilities, and future random
outcomes. It is assembled through the same public-observation belief builder used by the
agent and never reads the simulator's private battle root.

The required teacher source is `public_mirror_exact_showdown_teacher_v2`. Older
`exact_showdown_teacher_v1` data used a private simulator root and is historical evidence,
not valid input for a newly promoted model.

## Required dataset source record

Every saved teaching dataset must include:

- creation time in Coordinated Universal Time (UTC);
- repository and local Pokemon Showdown commits, plus whether either checkout had local
  changes;
- battle format;
- collection command or program;
- requested game count and random seed;
- team source and opponent mixture;
- the full policy configuration used by the teacher; and
- the information-contract and teacher-source versions.

Counts alone are not enough to reproduce a dataset.
Large collection requires clean checkouts. The deliberately small `--local-smoke` run may
use an uncommitted worktree, but its metadata records that fact and it is not promoted into
a training corpus.

## Missing and rejected data

Unknown opponent information is normal and must be represented by the existing belief and
availability features. It must never be filled with the simulator's private truth.

A corrupt or incomplete example is different: it must fail the audit. Collection stops
and reports the failed battle if a decision cannot be rebuilt safely; it never silently
calls the requested count complete.

## Training, validation, and test separation

- Training data teaches model parameters.
- Validation data selects settings and the saved model.
- Test data is the untouched final exam.

All decisions from one battle stay in one split. When evaluating generalization to new
teams, the split is made with the packed-team fingerprint, not a filename: renaming an
identical team must not move it across the boundary. Training and validation groups are
written to `split_manifest.json` and must be disjoint.

Final test teams are stored separately and are not used to select settings. Repeatedly
checking the final test and changing the model from its result turns it into validation
data, so a new untouched test is then required.

## Sampling

Collection should cover multiple team archetypes, opponents, early and late turns, and
small and large legal-action sets. No single team or unusually long battle should be
treated as independent evidence for every turn it contributes. Evaluation reports group
uncertainty by team.

Human replays can provide a useful starting representation and public usage estimates,
but this project's short, mostly mid-ladder replay snapshot is not sufficient evidence of
high-level play. Strategic teaching data comes from the public-information exact teacher;
final strength comes from held-out battles against the unchanged full-search benchmark.

## Required audit

Before training, run:

```bash
.venv/bin/python offline/audit_training_data.py \
    --dataset runs/data_foundation/smoke/demonstrations.pt \
    --split-manifest runs/data_foundation/smoke/split_manifest.json
```

The command must report counts and fail on:

- missing required dataset or example fields;
- duplicate examples or duplicate legal actions;
- a teacher action outside the legal-action list;
- malformed or non-finite arrays;
- a dataset fingerprint that does not match the split manifest; or
- battle/team overlap between training and validation.

A new large collection starts only after a small collection passes this audit and a few
examples have been inspected manually.
