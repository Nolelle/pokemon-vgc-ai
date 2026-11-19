# mikumiku37 local replication pilot

The local bot starts from scratch, picks its own four Pokémon and learns from wins, losses and draws. It plays the current version half the time. The other half, it draws an older version, giving more chances to opponents it has struggled against. A transformer (a model that compares the pieces of a position to one another) scores legal plays and estimates the final outcome. The training rule is proximal policy optimization, which limits how quickly updated move preferences depart from the ones that produced the games.

The project uses official `pikapro >= 0.2.0` from PyPI (which upstreamed our RNG-order fixes from `spedr/pikapro-engine#2` as well as Champions Last Resort handling). One run through the protocol bridge plays games sequentially. The original system collected decisions across 8,192 simultaneous battles. The model sees its side’s observation and the project’s Champions data. Games use 226 training teams from previously revealed team sheets. Human move choices and search are excluded from training. [Simulator](https://github.com/spedr/pikapro-engine) · [Author’s method and follow-up](https://www.smogon.com/forums/threads/mikumiku37-a-self-play-vgc-bot-reached-1-on-the-vgc-reg-m-c-ladder.3789199/)

## Local transformer architecture

The larger pilot model is a dense, encoder-only transformer with **12 layers and 8,694,191 trainable parameters**. The encoder builds a shared representation of the current board for three scoring networks. The design has no transformer decoder or mixture of experts (routing inputs to different specialist networks).

| Component | Specification |
| --- | --- |
| Input | **13 tokens** (12 Pokémon slots and one battlefield record), each represented by **268 numbers** (`d_model = 268`). |
| Attention | **4 heads per layer**, each with **67 numbers per token**. A head is one way to compare the board records. Each token can attend to all 13 tokens. |
| Feed-forward network | **268 → 536 → 268** per layer. It updates each token individually and uses ReLU, which clips negative inputs to zero. |
| Normalization | Pre-LayerNorm, which rescales the inputs before both attention and the feed-forward network. |
| Dropout | **0**. The model does not randomly switch off parts of a layer during training. |
| Parameter count | **6,930,480** in the encoder, about **80%** of the total. The rest represent inputs and score outputs. |
| Numeric precision | **32-bit floating point (FP32)** for model training and scoring on the central processing unit (CPU). The model uses no mixed precision (combining floating-point formats) or quantization (representing weights with fewer bits). |
| Saved model | `runs/replication_8m7_37101/policy_0000032.pt` is **35,148,111 bytes** (35.15 MB). All stored arrays are FP32. The file also contains **62,389 fixed Pokédex values** outside the trainable parameter count. |

### Output heads

These small networks read the encoder's result.

- **Turn choice (804 → 268 → 1).** The head combines the board summary with both action representations and scores each legal doubles pair as a unit. It can score candidate pairs in parallel.
- **Team preview (1,340 → 268 → 1).** The head combines the board summary with four Pokémon representations and scores each ordered selection.
- **Position value (268 → 1).** The head reads the board summary and applies `tanh`, a function that keeps its position score between −1 and +1.

These numbers describe the local implementation. The creator reported roughly 8.7 million parameters for mikumiku37 and described scoring the second Pokémon's action based on the first. The original layer layout remains unpublished in the cited write-up. [Local model code](../src/vgc/rl/replication_policy.py#L236) · [Creator’s write-up](https://www.smogon.com/forums/threads/mikumiku37-a-self-play-vgc-bot-reached-1-on-the-vgc-reg-m-c-ladder.3789199/)

## Repeat the local experiment

Run these commands from the project directory. Each training run needs a new output directory. The project’s three existing readiness checks must have current passing reports. The evaluation commands use the locally built Showdown engine.

```sh
uv sync --locked --extra dev --extra train --extra replica
.venv/bin/python -m selfplay.train_replication_league \
  --out runs/replica_repeat --width 268 --layers 12 --games 32 \
  --games-per-update 8 --snapshot-games 16 --batch-size 16 --epochs 2 --threads 2

.venv/bin/python -m offline.replica_search_probe \
  --checkpoint runs/replica_repeat/policy_0000032.pt

.venv/bin/python -m offline.replica_search_probe \
  --checkpoint runs/replica_repeat/policy_0000032.pt --paired-games 1 \
  --team data/selfplay/mc_sheet_pool_v2/team_003.packed.txt \
  --opponent-team data/selfplay/mc_sheet_pool_v2/team_005.packed.txt \
  --actions 2 --worlds 2 --seeds 2 --iterations 200

.venv/bin/python -m offline.check_replica_engine --games 100
```

## November 18 results

- The larger model had 8,694,191 adjustable numbers. It finished 32 Rust games and trained on 619 choices, including 48 team picks. It saved its final model in `runs/replication_8m7_37101/policy_0000032.pt`. An earlier, smaller model finished 128 Rust games.
- On one opening turn, search considered eight plays per side, eight possible hidden situations and four draws of battle randomness per pairing. The mirror produced eight distinct situations within its limit of sixteen. Search used the trained model to score the positions reached in Showdown.
- The 32-game model then played itself on two teams excluded from training. Search controlled each team once with identical starting randomness. Both games finished. The plain model won both. Two games with reduced search only check that the comparison runs.
- All 226 training teams passed the Rust engine’s validator. Before patching `pikapro 0.1.0`, 94 of 100 fixed-seed comparisons against this project’s pinned Showdown agreed through finish (six diverged, including a knockout discrepancy on seed 12; logged under `runs/replication_rust_37101/engine_comparison.json`). After applying both local patches (`pikapro 0.1.0+regmc.2`), **600 of 600** comparison games matched through finish (see below).

The public post leaves the exact model layout, hidden-world sampler and search blend unspecified. This pilot scores complete ordered move pairs, uses the project’s existing public-state reconstruction with estimated opponent sets, and gives equal weight to the average reply and the worst reply in search. These are reproduction choices. The author’s 1,260 tournament teams, 330 million training games, training speed and ladder strength are outside the evidence from this run.

## Diagnosing and fixing the simulator differences

The original run used the published `pikapro 0.1.0` package (reference Showdown revision `a5df827`), whereas this project pins `7e743761c`. Replaying the mismatches against Showdown exposed two RNG-order bugs in `pikapro 0.1.0`:

1. **Missing pre-switch update (`patches/pikapro/0001-update-before-forced-switch.patch`).** When a living Pokémon is about to leave after a pivot move (such as U-turn), Showdown runs an `Update` pass before issuing the switch request. That pass sorts tied Speeds and consumes a random draw even when no `BeforeSwitchOut` handler runs. Rust skipped the update, leaving its RNG one draw behind so later damage rolls or move orders could diverge. Restoring the update (and setting `skip_before_switch_out` on Shed Tail) fixed all 6 original mismatches.
2. **Clearing status on a fainted Pokémon with no replacement (`patches/pikapro/0002-keep-status-without-replacement.patch`).** Testing 200 unseen holdout-team games exposed a second mismatch (187/200 before the fix): Showdown's `checkFainted` skips sides that have no replacement left (`can_switch == 0`), so a fainted Pokémon keeps its status (e.g. burn) and still participates in handler speed-tie sorts even though its status handler never runs. Rust cleared the status unconditionally, skipping those tie-break draws.

With both patches packaged as `pikapro 0.1.0+regmc.2`, all three parity suites against pinned Showdown `7e743761c` matched **100% (600 / 600)**:

- Original 100 training-team games: **100 / 100** (`runs/replica_parity_20261118/original_after.json`)
- 200 holdout-team games with random leads: **200 / 200** (`runs/replica_parity_20261118/holdout_after.json`)
- 300 extra training-team games with random leads: **300 / 300** (`runs/replica_parity_20261118/new_training_teams.json`)

The player connection also distinguishes two cases that look like “wait.” A player waiting while the opponent replaces a Pokémon can receive updated information about its own team. A player who already submitted an accepted choice while the opponent corrects a rejected choice must keep that choice. The Rust bridge now handles both cases.

The optional `replica` install uses the patched Rust source archive under `patches/pikapro/`. The build needs a Rust compiler. The project records the source archive’s digest in `uv.lock` and the installed simulator version in new training-run settings. To reproduce the archive, pass the original PyPI source archive to `tools/package_replica_engine.py`. The script checks the original digest and applies the readable patches in filename order. [Original source archive](https://pypi.org/project/pikapro/0.1.0/#files)

The comparison checks health, winner, which sides need to act, request phase, rejected-choice counts, and the exact inputs and legal choices the transformer sees at each decision. To exercise both original and unseen situations, use the original 100-game command above and repeat with held-out teams and varied leads.

```sh
.venv/bin/python -m offline.check_replica_engine --games 200 --seed 1001 --random-preview \
  --team-manifest data/selfplay/mc_sheet_pool_v2/holdout_manifest.json
```

The saved 32-game model can play over the Showdown server connection. With the pinned server running locally, this checks team preview, ordinary choices, forced switches, server errors and fallback decisions.

```sh
.venv/bin/python -m offline.replica_live_smoke \
  --checkpoint runs/replication_8m7_37101/policy_0000032.pt \
  --team data/selfplay/mc_sheet_pool_v2/team_003.packed.txt \
  --opponent-team data/selfplay/mc_sheet_pool_v2/team_005.packed.txt
```

Add `--search` for the reduced lookahead check. Both modes completed full local server games.

### Parallel PPO training, resource profiles, resume, and W&B

`selfplay/train_replication_league.py` now supports parallel `pikapro` battle workers, two resource profiles (`--profile desktop` for active computer use at low CPU priority and `--profile afk` for unattended/overnight runs), `--resume`, periodic held-out baseline evaluation (`--eval-every`, `--eval-games`, `--eval-opponents`), and optional Weights & Biases logging (`--wandb-project`):

```sh
# While actively using the computer (desktop profile: half CPU cores, low priority)
.venv/bin/python -m selfplay.train_replication_league \
  --out runs/replica_m128_37101 --profile desktop \
  --width 128 --layers 4 --games 4096 --games-per-update 64 \
  --snapshot-games 256 --eval-every 256 --eval-games 20 \
  --wandb-project pokemon-vgc-ai

# Resume overnight in AFK mode (uses most cores)
.venv/bin/python -m selfplay.train_replication_league \
  --out runs/replica_m128_37101 --resume --profile afk \
  --width 128 --layers 4 --games 16384 --games-per-update 64 \
  --snapshot-games 256 --eval-every 256 --eval-games 20 \
  --wandb-project pokemon-vgc-ai
```

### Upstream Reg M-C Last Resort fix and ladder runner support

- Showdown was synced to `5b88c4761` (including `41ba4797f`, which disables Last Resort in Champions until all other move slots have been used) and all three readiness gates passed.
- `patches/pikapro/0003-champions-disable-last-resort.patch` adds the matching Champions `onDisableMove` rule to `pikapro`, packaged as `pikapro 0.1.0+regmc.3`. Both engines match on a dedicated Last Resort test and 100/100 full-game parity checks (`runs/replica_parity_20261118/regmc3_100.json`).
- `ladder/run_ladder.py` now supports `--policy-mode replica` (plain neural policy + neural preview) and `--policy-mode replica-search` (neural policy + one-turn sampled-world lookahead) for both `--local-smoke` and public ladder play (with all four readiness and Showdown-parity gates still enforced):

```sh
.venv/bin/python ladder/run_ladder.py --local-smoke --n 1 \
  --team data/selfplay/mc_sheet_pool_v2/team_003.packed.txt \
  --policy-checkpoint runs/replication_8m7_37101/policy_0000032.pt \
  --policy-mode replica
```
