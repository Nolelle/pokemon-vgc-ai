# Controlled full-battle commissioning

Section 5 follows the four-family tactical commissioning pass. This is an offline
learning experiment, not model promotion or evidence of public ladder strength.

## Frozen throughput and retention pilot

- Use the selected tactical seed-77 policy (update 48) as the initial model. Convert
  only the checkpoint container to the fixed-mirror trainer's format; verify every
  tensor is identical. Preserve the source checkpoint hash and vocabulary metadata.
- One fixed legal roster: `teams/phase2_mirror.packed.txt`. Both sides use the
  same roster and fixed `team 1234` selection, including lead and bench order.
- One frozen opponent policy: `random`. Alternate learner seats. No opponent mixture.
- Pilot seed **7707**. **64 complete training games**, two batches of 32.
- Sparse terminal reward, gamma 1, lambda 0.95, no teacher/shaping/dropout,
  four PPO epochs, minibatches of 128, learning rate **0.0003**. The optimizer is
  fresh; this is a declared curriculum transition, not a resumed tactical run.
- Evaluate the initial and endpoint models on the same 64 development games,
  split equally across seats. Primary action selection is sampled.
- Measure tactical retention before and after on 64 fresh futures per task,
  future base **12,100,000**. The model sees the same fogged inputs as before.
- Reject any recovered learner fallback, invalid choice, unfinished trajectory,
  or nonfinite update. Report actual elapsed time and decision counts.
- The pilot is for execution, throughput and identifying immediate forgetting.
  It cannot certify section 5's three-seed learning requirement. Do not extend its
  budget or select a favorable intermediate policy after seeing outcomes.

Use an immutable source snapshot with all three readiness certificates valid for
its own commit. Required skipped tests are now rejected; the ignored archetype
fixture pool is copied and separately fingerprinted. The original worktree remains
uncommitted, and certificates from the snapshot do not certify another source tree.

## Opponent ceiling check

Before the pilot's first update, the tactical seed-77 model already beat random
60/64 times (93.75%; 95% Wilson interval approximately 85.0–97.5%). This leaves
little room for a useful learning comparison against random. The pilot still
finishes at its frozen cap. Independently evaluate all three initial tactical
models against **maxpower**, 64 games each with both seats, using evaluation seeds
7707/8807/9907. No training occurs in this ceiling check; it selects an appropriate
fixed opponent before defining the main full-battle learning experiment.

The pilot completed in 175.1 seconds: 64 training games and 128 evaluation games.
Its matched random-opponent win count changed from **60/64 to 57/64**; this does
not establish improvement. Tactical retention remained above the point target:
64/64 setup, 63/64 replacement, 64/64 target and 64/64 sacrifice, versus 64/64 on
each at initialization. The two training batches took 31.0 and 31.6 seconds.

Initial maxpower scores were **40/64, 57/64, 4/64** for tactical seeds 77/88/99.
Those evaluations took 34.2–35.6 seconds each concurrently. The large spread
shows that solving the controlled tasks does not determine full-battle strength.

## Frozen main learning comparison

- Start again from the three selected tactical policies, not the random-opponent
  pilot's endpoint. Training seeds **7707, 8807, 9907** correspond to tactical
  seeds 77, 88, 99. Each gets a fresh optimizer.
- Fixed mirror roster, fixed preview, alternating seats, frozen **maxpower** opponent.
- **512 complete training games per seed**, 16 updates of 32. Same optimizer,
  reward, feature and sampling settings as the pilot.
- Development evaluations: 64 games at 0/128/256/512 training games. Initialization
  must reproduce the preceding ceiling-check result on the same schedule.
- The **endpoint** is primary. Do not select an earlier snapshot based on either
  battle results or retention. Keep all snapshots and failures.
- Fresh paired confirmation for every seed, regardless of development results:
  **256 games for initialization and 256 for endpoint**, alternating seats,
  evaluation seeds **17607, 18707, 19807**. Record per-game outcomes to compute
  paired differences; same random seeds do not guarantee identical random events
  when action paths diverge.
- Tactical retention: the same 64 development futures per task, base 12,100,000,
  before and after training. No rehearsal or tactical labels are added during
  full-battle training in this comparison.
- Commissioning advance criterion: positive fresh match-score change for each
  training seed, at least five percentage points mean improvement, and at least
  95% tactical success on every task for every endpoint. Report paired uncertainty
  even if the point criteria pass; these are not competitive-strength guarantees.

The budget follows measured throughput. A flat or regressing run ends at 512
games and receives a diagnosis; it is not silently extended. This comparison
does not address varied teams, hidden-set adaptation or learned team preview.

## Main results

| Tactical seed | Initial wins / 256 | Endpoint wins / 256 | Paired score change | Paired bootstrap 95% interval |
| --- | ---: | ---: | ---: | --- |
| 77 | 183 | 231 | +18.75 points | +13.28 to +24.22 |
| 88 | 212 | 221 | +3.52 points | -1.17 to +8.59 |
| 99 | 45 | 226 | +70.70 points | +64.06 to +76.95 |

All runs completed their 512-game cap without an invalid trajectory or learner
fallback. Their elapsed times, including development evaluation, were 436.0,
446.6 and 468.9 seconds while running concurrently. Initialization reproduced the
preceding maxpower ceiling checks exactly. An eight-pair A/A harness check produced
identical per-game outcomes and a zero paired difference.

All endpoint tactical checks met the point target. Seed 77 scored 64/64 setup,
63/64 replacement, 64/64 target and 63/64 sacrifice. Seeds 88 and 99 scored 64/64
throughout. These checks used the frozen development futures, not hidden-team tests.

The predeclared commissioning point criteria pass: every seed's fresh change is
positive, the mean improvement exceeds five points, and tactical skills are retained.
However, seed 88's interval includes zero. The evidence is clear for two seeds and
inconclusive for the third; do not describe every seed as statistically established
improvement. Three models and one fixed matchup do not establish broad robustness.

**Decision:** proceed to section 6's controlled information-use experiment. This
does not authorize a large run or release. The random-opponent trial remains only
a one-seed pilot; another full three-seed opponent comparison remains useful before
claims across opponent behavior. Sections 7 and 8 have not been executed.

Raw source-bound runs: `runs/rl_sections_1_8/full_maxpower_seed{77,88,99}/`.
Per-game paired outcomes and uncertainty: `full_confirm_{77,88,99}.json`.
