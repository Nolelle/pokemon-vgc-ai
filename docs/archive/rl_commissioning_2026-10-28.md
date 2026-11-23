# RL commissioning: first implementation slice

Date: 2026-10-28. Research plan: [rl_research_plan_2026-10-28.md](rl_research_plan_2026-10-28.md).

## Scope and experiment settings

This slice starts stages 1–3: a bounded optimizer diagnostic, real-rollout probability checks, and a sampled observation audit. It does not complete all of these stages or establish Pokémon learning or playing strength.

The diagnostic source fixed the following settings before its first run: CPU, one Torch thread, terminal +1/-1 rewards, gamma 1.0, GAE lambda 0.95, entropy coefficient 0.01, no teacher anchor or shaping, two PPO epochs, minibatches of 32. Fresh models use the current vocabulary. PPO minibatch permutations receive an explicit seeded NumPy generator in this diagnostic; existing callers that omit it retain their previous random-generator behavior.

- Actual battles: four games on `teams/phase2_mirror.packed.txt`, fixed `team 1234` preview on both sides, opponent `maxpower`, alternating learner seat, seed 20261028. All four are collection-only; the subsequent update has learning rate zero. The model includes information, tactical and mechanics features. Head dropout 0.1 intentionally stress-tests the historical mismatch, rather than defining the planned dropout-free learning baseline.
- Synthetic optimizer task: three distinct candidate encodings, one rewarded action, one-step episodes. Same production candidate model and PPO update; fresh seeds 11/22/33, 40 iterations × 64 episodes each, Adam learning rate 0.001, policy dropout zero. The declared target is final probability at least 0.95 on the optimal action for each seed. This task has no Pokémon strategy labels and does not execute game rules.
- Observation audit: existing differential audit, 40 games, seed 20261028, two workers, failure on unexplained findings. This is sampled coverage, not complete mechanics certification.

The original search references and all live policy settings remain unchanged. No old model is promoted or resumed. No full-battle training or public ladder games were started.

## Implementation

`src/vgc/rl/ppo.py` now keeps the model in evaluation mode during gradient updates, matching `PpoVgcPlayer` action collection. This disables dropout and other training-mode behavior without disabling autograd. A frozen policy must not appear changed merely because its model mode changed. Optional seeded minibatch RNG makes the commissioning loop reproducible.

The focused regression collects real policy probabilities with head dropout 0.5, then runs zero-learning-rate updates. It requires approximately zero KL, no PPO clipping, and nonzero gradients. It fails when the old `model.train()` behavior is restored in memory, and passes with the fix. Existing parameter-update and teacher-anchor checks still pass.

`offline/commission_rl.py` supplies a repeatable diagnostic, rather than another general trainer. It records configuration, model/data/source fingerprints, game outcomes, probability consistency and all three learning curves. No trained checkpoint is produced. The real-rollout check intentionally evaluates different batch sizes (1/7/32) with variable candidate and mechanics-input lengths.

## Initial results

| Check | Result |
| --- | --- |
| Static mechanics/state/action declarations | All three PASS; no dynamic release certificates inferred |
| Real complete battles | 4; both seats exercised; 39 learner decisions; 0 fallbacks |
| Largest real-rollout log-probability recomputation error | 4.77e-7, below the declared 1e-5 tolerance |
| Zero-learning-rate PPO clipping fraction | 0 |
| Mean approximate KL at zero learning rate | Approximately -1.12e-8; numerical noise around zero |
| Synthetic optimal-action probability, seed 11 | 0.3315 → approximately 1.0000 |
| Synthetic optimal-action probability, seed 22 | 0.3301 → approximately 1.0000 |
| Synthetic optimal-action probability, seed 33 | 0.3335 → approximately 1.0000 |
| Sampled parsing audit | 817 steps, 1,554 perspective comparisons; 0 unexplained groups; 0 errors |
| RL and fixed-mirror tests | 99 passed, 1 integration test deselected by the default marker |
| Environment/match/encoding checks, including integration | 38 passed |
| Ruff and whitespace checks | PASS |

The first diagnostic took about 4.2 seconds on this machine. This is diagnostic turnaround, not a measured full-training throughput estimate. The four-game record was 1–3; such a small collection-only sample is not a strength comparison.

Local raw evidence: `runs/rl_commission_20261028/report.json` and `parsing_audit.json`. These files are gitignored; the settings, key results and limitations are preserved in this tracked document. The diagnostic script can reproduce fresh evidence, and reports identify the exact source/data files used.

## Reproduce

```sh
.venv/bin/python offline/commission_rl.py --output runs/rl_commission_20261028/recheck.json
.venv/bin/python -m pytest tests/test_rl.py tests/test_train_fixed_mirror.py -q
.venv/bin/python -m pytest tests/test_rl_env.py tests/test_rl_match.py tests/test_mechanics_encoding.py -m 'integration or not integration' -q
.venv/bin/python offline/audit_battle_parsing.py --battles 40 --workers 2 --seed 20261028 --fail-on-unexplained --output runs/rl_commission_20261028/parsing_recheck.json
```

## Remaining before the next milestone

Stage 2 still needs learner-input-specific private-information invariance, live/direct parity and reset/isolation coverage checked against the proposed experiment. Existing tests cover parts, but this slice does not certify every input or reachable mechanic. Static declarations and 40 sampled games cannot do that.

Stage 3 still needs the remaining analytic multi-step return, candidate-permutation and checkpoint/RNG-resume properties collected into the experiment evidence. The new diagnostic verifies one-step terminal targets and probability consistency, rather than every PPO contract.

Stage 4 is not started: build a small real-Showdown endgame with an independently established outcome reference, then demonstrate learning from terminal results across seeds. A synthetic bandit success does not replace that test. Do not launch the full-game scaling recipe on the basis of this report.
