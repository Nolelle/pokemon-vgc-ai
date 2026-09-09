# Replay labels and match outcomes

Spectator replays remain useful when a selected move never executes. Missing action
information must not become a label saying that the player chose to pass, and a faint
must not automatically count as a bad decision. Winning the match is the objective;
a sacrifice may help achieve it.

## Action information

- An observed move or switch records what the transcript supports. It is not a
  certificate of a fully recovered submitted joint order.
- An unknown action means a living slot had no recoverable move choice, including a
  faint or an unnamed blocked move. Preserve observed Mega Evolution independently.
- No action required means the slot was empty or fainted at the start of the turn.
- Unknown actions and slots requiring no choice do not teach a move or target class.
  Older replay `pass` labels are ambiguous and must not be taught as genuine choices.
- A known move with an unknown target can teach the move while omitting target teaching.

The parsed records retain match outcomes even when action labels are unusable. The
existing action-copying dataset can skip those slots; it is not a new outcome-only
trainer. This change does not train a model or regenerate historical datasets.

## Outcomes

Keep win, loss, draw, and unresolved separate. An absent or unresolvable winner is not
a loss. Preserve the distinction in stored records and parser summaries. Older records
that collapsed outcomes cannot recover the missing distinction without their raw logs.
The existing binary outcome learner uses explicit wins/losses only. Draws and
unresolved outcomes remain stored but are excluded from that learner. Both positive
and negative legacy boolean-only labels are excluded, avoiding a wins-only dataset;
reparse the raw logs before using old data for outcome training.

## Remaining limits

Spectator records still lack exact own-team sets and complete legal-choice requests.
Executed moves, called or locked moves, and redirected targets may not identify the
original submitted choice. Replacement switches remain distinct from initial joint
choices. Do not promote these records to the exact public-teacher data contract.

Preview selections inferred from appearances are partial when fewer than four team
members appear. Later reveals may establish a label, but must not enter earlier
opponent observations. This repair does not redesign preview or hidden-set recovery.

## Format and replay corpus

Schema 5 is format-agnostic: the same unknown / observed / no-action / outcome rules
apply to every Champions replay. Record `format_id` on parsed datasets. The live
download path is `data/replays/<FORMAT_ID>/` (currently
`data/replays/gen9championsvgc2026regmc/`). Public M-C replays are still scarce
(~44 files). Keep the historical M-B tree
(`data/replays/gen9championsvgc2026regmb/`, ~2940 files) for warm-start and prior
work. Fold new rated M-C games in incrementally. Do not rebuild usage priors from
the small M-C snapshot. Mix formats in training only with an explicit format id.

## Fixed sample check

Use the same 20 cached replay files sampled with Python `random.Random(20260909)`
from the sorted JSON paths under `data/replays/gen9championsvgc2026regmb`. This M-B
sample is pinned for regression stability, not because M-B is the live format. Add
an M-C sample once that corpus is large enough that a 20-file draw is meaningful.
Record their IDs and file hashes in the verification report so subsequent changes to
the cache cannot silently change the sample. Check retained records and outcomes,
excluded unknown-action labels, and the Charizard Mega Evolution followed by a faint
in `gen9championsvgc2026regmb-2649592145`, turn 1, player 1.

Verified on 2026-09-09 against baseline commit `1d439df`: all 20 games parsed,
all 355 records and their state snapshots were retained, and winner identities were
unchanged. The 242 turn records contain 75 unknown actions, 27 slots requiring no
action, 360 observed moves, and 22 observed switches. All 102 former pass slots are
excluded from move and target teaching; the 382 remaining action labels are unchanged.
These counts do not certify that all observed actions equal submitted choices.
Detailed IDs and source hashes are in `runs/replay_label_audit_20260909/report.json`.
