# RL retention follow-up

This continues the section-4 diagnostics. It does not approve a model or mark
sections 4–8 complete.

## Runtime and readiness

After the restart, the union of named mechanics, battle-state and action gate
tests passed: **185 cases in 90.36 seconds**, including the required websocket
tests. The raw log is `runs/rl_sections_1_8/restart_readiness_tests.log`; the exact
node list and return code are in `restart_readiness_result.json`. A separate
websocket-module invocation was interrupted after five passing cases and is not
reported as a completed run. Unix socket creation remains denied. These results
are test evidence, not a newly issued clean-commit readiness certificate.

## Update measurements

`offline/diagnose_endgame_updates.py` collected 32 fresh episodes per task from
the original seed-11 endpoint, with future-seed base 4,100,000. It recomputed
full categorical distributions and compared identical data, initial weights,
and saved Adam state under four versus one PPO epochs.

Several empirical task-policy gradients conflict: task 6 has cosine similarity
about -0.59 with task 2 and -0.65 with task 5. Policy-gradient norms ranged from
0.033 to 5.61. With one epoch, mean total-variation changes were roughly 0.09–0.17
on several tasks. Four epochs sometimes reversed part of that movement. Thus
fewer epochs alone is not established as a cure. This is a local measurement
on one checkpoint and one sample, not a proof of the cause of every regression.

Raw evidence: `runs/rl_sections_1_8/update_diagnosis_seed11.json`.

## Frozen larger-batch comparison

Hypothesis: four episodes per task per update yield noisy, conflicting gradients;
more fresh trajectories per update improve retention at a fixed episode budget.

- Change rollout batch from 32 to 128 complete episodes; keep uniform task sampling.
- Keep the original learning rate 0.001, four epochs, minibatches of 64, model,
  sparse rewards, gamma, lambda, entropy, and all information boundaries.
- Keep **2,048 training episodes per seed**: 16 updates instead of 64. Simulator
  and actor RNG seeds are assigned by absolute episode index, matching the old
  schedule. Different updates naturally produce different action trajectories.
- Fresh initializations 11, 22 and 33. No restart from a favorable checkpoint.
- Evaluate the same development futures after 0/512/1024/1536/2048 episodes.
- Use the endpoint, with no best-checkpoint selection. The point target remains
  at least 95% wins on every task for every training seed.
- If all development endpoints meet the target, confirm all three on 128 fresh
  futures per task with seed base 4,900,000. Report every seed. Failure leads to
  diagnosis, not an expanded arbitrary hyperparameter sweep.

The larger batch also changes the sequence of Adam steps and advantage
normalization groups. It tests the practical batch-size intervention, not an
isolated mathematical estimate of gradient variance.

### Larger-batch result

| Seed | Endpoint win rate | Worst task | Per-task development target |
| --- | ---: | ---: | --- |
| 11 | 98.0% | 60/64 | Failed |
| 22 | 100% | 64/64 | Met |
| 33 | 99.0% | 59/64 | Failed |

The hypothesis that this intervention alone resolves retention is unsupported.
Per the frozen rule, these candidates are not sent to fresh confirmation.

## Frozen checkpoint-selection experiment

The original curves show all three seeds reached strong development checkpoints
before some later regressions. This motivates testing selection explicitly,
rather than changing the optimizer repeatedly. This does not fix instability.

- Use the original 32-episode, 64-update, learning-rate-0.001 recipe unchanged.
- Use three new initializations: **44, 55, 66**, not the seeds used to develop
  the selection rule.
- Evaluate at 0/16/32/48/64 updates. Select the checkpoint with the highest
  **minimum task win count** over the eight development tasks; break ties by
  choosing the earliest checkpoint. Keep the final checkpoint separately.
- No repeated reruns, extra budget or per-seed recipe change. All three selected
  checkpoints receive fresh confirmation, including any development failure.
- Confirmation: 128 futures per task, seed base **5,900,000**. The unchanged
  point target is at least 95% wins on every task for every training seed.
- A pass establishes only this narrow endgame milestone with development-based
  selection. It does not finish section 4's broader tactical/oracle requirements.

The full default unit suite now passed with W&B's supported TCP transport and
writable temporary cache/config directories: **1,557 passed, 250 deselected**.
No test was skipped to obtain that result; integration exclusions are the
project's normal default selection. The nine W&B tests also passed separately.

### Checkpoint-selection result

| New seed | Selected update | Fresh total wins | Worst task | Every-task target |
| --- | ---: | ---: | ---: | --- |
| 44 | 32 | 1021/1024 | 125/128 | Met |
| 55 | 16 | 1006/1024 | 120/128 | Failed |
| 66 | 32 | 1021/1024 | 125/128 | Met |

The new selection rule did not satisfy the frozen all-seed requirement. Seed 55's
task 6 won 120/128; task 2 won 122/128. Neither the aggregate nor the other seeds
overrides that failure. The endpoint policies were retained separately and were
not substituted for the selected checkpoints after seeing confirmation outcomes.

Rather than continue changing hyperparameters on the same narrow suite, the next
work builds the broader, independently verified tactical tasks required by the
plan. See [tactical commissioning](rl_tactical_commissioning_2026-10-28.md). The old
eight-task suite remains an open retention/generalization diagnostic.
