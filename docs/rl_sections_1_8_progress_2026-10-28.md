# RL plan execution: sections 1–8

Date: 2026-10-28. This is a progress report, **not completion of section 8**.

Updated after restart: see [retention follow-up](rl_retention_followup_2026-10-28.md)
and [oracle-backed tactical commissioning](rl_tactical_commissioning_2026-10-28.md).
The required test union now passes (185 cases), and the default unit suite passed
1,557 cases with supported TCP-based W&B logging. Three fresh tactical policies
passed all four task families on fresh future seeds. The earlier retention
failures below are preserved, not reclassified as successes.

## Current position

| Section | Evidence/status |
| --- | --- |
| 1: experiment definition | Commissioning and short-task recipes fixed and run; no full-game experiment started |
| 2: laboratory | Required test union passes; all three clean-snapshot gates pass with no skipped cases; required-skip certification defect fixed |
| 3: update correctness | Probability, multi-step returns, episode separation, candidate permutation and exact optimizer/RNG resume checks pass in the exercised scope |
| 4: small Pokémon tasks | Four oracle-backed tactical families passed fresh confirmation across three new seeds; older eight-task retention issue remains documented |
| 5: full battles | Three 512-game fixed-maxpower runs met frozen point criteria and retained tactical skills; two fresh gains clear, one uncertain |
| 6: hidden-information learning | Not started; invariance tests are correctness evidence, not learned adaptation |
| 7: generalization | Not started |
| 8: learned preview | Not started |

Readiness certificates were issued only for the isolated, tested source snapshot.
No model promotion, live default change or public ladder session was produced.
See [full-battle commissioning](rl_full_battle_commissioning_2026-10-28.md) for the
section-5 protocol, all outcomes, uncertainty and source identity.

## Correctness repairs

### Brought-four resources

poke-env retains the six preview Pokémon in `battle.team` after selecting four. The legacy neural summary treated the two unbrought Pokémon as healthy reserves. For example, it encoded two active Pokémon and four bench Pokémon on the first turn, when the request listed only four total.

`vgc.rl.encoding.encode_live_state` now builds the own bench summary from the identities in the actual private request. Complete six-set metadata remains available through the separate information branch. A missing identity mapping fails rather than silently inventing reserves.

### Preview-selection parity

The snapshot trusted poke-env's `selected_in_teampreview` flags. Live preview logic sets them, while fixed-preview direct agents can bypass that logic. The same actual game therefore produced different mechanics input bytes through the two adapters.

`vgc.mechanics_state` now derives our selected flags from the actual post-preview request. Both sides' observations remain fogged; no opponent request is read. A nonterminal cutoff test now compares summary, information and mechanics inputs through the direct parser and poke-env's live message pump.

### Compatibility and reproducibility

The learner contract is now `candidate-policy-value-v5-brought-resources`. Older checkpoints must not silently receive the changed feature semantics. The model registry preserves the old refusal status as history and records architecture rejection for the affected v4 files. Their bytes were not modified.

The PPO dropout correction from initial commissioning remains in place. Additional checks cover analytic win/loss/draw returns across consecutive episodes, candidate-list permutation, and an exact next-update comparison after restoring model, optimizer and RNG state.

The fixed-mirror trainer now accepts an explicit team and Torch thread count, seeds PPO minibatch permutations, and evaluates checkpoints on the same development seed schedule. These are preparation changes; the full-game CLI was not used for a new training run.

## Real endgame experiments

`offline/research_endgames.py` constructs the tasks by replaying actual Showdown choices from game start. No HP/status patches or guessed mirror roots create these starts. Clones continue the real engine with independent future PRNG seeds. The actor receives the usual player view, exact own team and public history, and chooses from all request-legal joint actions.

The suite has eight fixed 1-v-1 endgames from legal bring-four battles using `teams/phase2_mirror.packed.txt`. The team passed the pinned engine's validator. One task takes two to four actor decisions under the reference policy; the others usually finish quickly. That is evidence for delayed outcome propagation, **not a proof of strategic setup or sacrifice competence**.

Discovery selected states before learning: maxpower won on eight probing futures, while random won at most five of eight. The first 80 generation seeds yielded three eligible tasks; expanding the discovery cap yielded eight by seed 184. This deliberately selects small learnable problems, not a representative benchmark. The maxpower probe is empirical and is **not an optimal policy oracle**. Confirmation later showed that reference can lose on unseen futures.

Inputs and rewards use the corrected contracts. Three fresh model seeds (11/22/33), no teacher anchor, no shaping, gamma 1, entropy coefficient 0.01, four PPO epochs and minibatches of 64 were used. Each multi-task run had a fixed endpoint of 64 updates × 32 episodes = 2,048 episodes. Evaluation at initialization and every 16 updates used 64 futures per task. Every run uses sampled policy actions, not a greedy replacement.

### Development results at the fixed endpoint

| Recipe | Seed 11 | Seed 22 | Seed 33 |
| --- | ---: | ---: | ---: |
| Initial untrained policy | 52.0% | 51.6% | 51.6% |
| Uniform tasks, learning rate 0.001 | 96.3% | 100.0% | 100.0% |
| Uniform tasks, learning rate 0.0003 | 97.9% | 99.2% | 100.0% |
| Hard-task-weighted, learning rate 0.0003 | 98.8% | 91.8% | 100.0% |

These are averages over the eight fixed tasks. They do not override a failed individual task. No seed was dropped, and an early peak was not substituted for the endpoint.

Seed 11's original run initially improved substantially but later regressed on several tasks. Reducing the learning rate did not solve the problem: its multi-decision task finished at 53/64 in development. A separate single-task run learned that same endgame from 36/64 to 63/64 after 512 episodes. On fresh futures it won **126/128**, versus **102/128** for the maxpower reference.

This establishes that the task is learnable by the existing representation and update. It does not distinguish shared-parameter interference from insufficient task-specific experience: the isolated run changes both. A subsequent sampling experiment devoted half the episodes directly to that task and distributed the rest over all eight, holding the 0.0003 recipe otherwise fixed. It improved that task but degraded other tasks in seed 22. That intervention did not solve retention.

### Fresh-future confirmation

Each confirmation used 128 fresh futures per task. The first recipe used future-seed base 1,900,000; after those results informed development, the weighted recipe used a new base of 2,900,000. The point target was at least 95% wins on **every** task. It is a commissioning target, not a confidence-certified generalization claim.

| Candidate | Total wins/games | Worst task | Every-task point target |
| --- | ---: | ---: | --- |
| Original seed 22 | 1020/1024 | 96.9% | Met |
| Original seed 33 | 1019/1024 | 96.1% | Met |
| Smaller-LR seed 11 | 1002/1024 | 86.7% | Not met |
| Weighted seed 11 | 1005/1024 | 93.8% | Not met |
| Weighted seed 22 | 923/1024 | 76.6% | Not met |
| Weighted seed 33 | 1018/1024 | 96.1% | Met |

The original seed-11 endpoint already missed the development target and was not selected for fresh confirmation. These runs establish substantial learning on a narrow suite, not a stable three-seed recipe. They are not held-out-team tests. The starts and hidden past remain fixed; fresh future randomness does not test generalization to different hidden sets or histories.

## Initial validation and runtime blocker (before restart)

- 162 focused learner/environment/state/registry checks passed after the state fixes.
- All 167 executable non-websocket readiness test cases passed. The 17 named websocket nodes remain uncertified; no full readiness PASS was manufactured from the subset.
- A 200-battle differential parsing audit performed 7,334 perspective comparisons: no unexplained groups or errors. It reported 22 episodes under precise documented exemptions, primarily Illusion identity paths and hidden extenders. This does not certify those cases for future expanded training.
- Pinned Showdown parity passed with `--no-fetch`, meaning comparison with the locally available upstream references, not a fresh internet check.
- The full default unit run had 1,556 passes and one W&B offline-process failure. Its error records denied TCP/Unix socket creation and a denied cache-log destination. This is not a green full-suite result.
- All 20 websocket integration cases failed when clients attempted localhost connections. A standalone check confirmed `PermissionError: [Errno 1] Operation not permitted` for 127.0.0.1, localhost and ::1. The environment also denies Unix sockets. These failures cannot certify live transport behavior.
- The test launcher now uses the already built simulator with `--skip-build`, avoiding an unnecessary attempt to write to the read-only sibling checkout. Socket access is still independently blocked.
- Ruff and `git diff --check` pass.

## Evidence and reproducibility

- Task definitions: `data/research/endgames_v1.json` (seed, side and exact choice prefix; team fingerprint included).
- Durable result copies and checkpoint fingerprints: `data/research/rl_sections_1_4_20261028/manifest.json` and its neighboring reports. All attempts, including failed interventions, are retained.
- Raw logs and experimental checkpoints: `runs/rl_sections_1_8/`. These models are unapproved diagnostics and are not registered as deployable policies.
- The runner refuses to overwrite an existing fixture or result destination. It supports uniform-task, isolated-task, weighted-task and separate confirmation runs. There is no automatic progression or promotion on an aggregate win rate.

Example fresh-output reproduction:

```sh
.venv/bin/python offline/research_endgames.py train --seed 11 --iterations 64 --lr 0.001 --output runs/endgame_recheck_seed11
.venv/bin/python offline/research_endgames.py confirm --checkpoint runs/endgame_recheck_seed11/checkpoint.pt --future-seed 3900000 --output runs/endgame_recheck_seed11_confirmation.json
```

Rerunning these commands is another scoped diagnostic, not permission to skip readiness or claim strength. New reports identify the script, fixture and checkpoint fingerprints; historical runs whose metadata predates a field remain historical rather than having missing provenance invented.

## Required next steps before advancing

1. The required websocket checks now pass. Complete clean-snapshot certification with all required fixtures; required skipped cases must not count as evidence.
2. Diagnose retention on the short tasks with trajectory and update measurements. The isolated hard-task control argues against declaring RL incapable. The weighted-task failure argues against assuming more hard-task exposure alone is sufficient. Candidate diagnostics include per-task policy change and value/advantage error before and after an update, and controlled update-size or checkpoint-selection comparisons.
3. Freeze a revised small-task recipe and a selection rule before a fresh confirmation. Do not concatenate the successful parts of different seeds/recipes into one claimed passing agent. Expand independently justified scenario coverage before calling the entire section-4 tactical suite complete.
4. Section 5's bounded three-seed comparison has now run with retention checks.
   Continue to controlled revealed-information learning; sections 6–8 remain open.
