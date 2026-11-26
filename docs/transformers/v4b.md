# Transformer v4b (`rust_v4b_m256_l6_distill` -> `rust_v4b_m256_l6_15m`)

> **Takeaway:** `v4b` keeps `v4`'s double-sized brain (`256×6`, `policy_version=3`, **6.40M params**) but replaces the broken cold start with **knowledge distillation** (teaching the bigger student network to copy our 15M-game `v3d` teacher's moves, team previews, and win probabilities). In just **13.9 minutes** (49,422 games), `v4b` reached **full 15M-game parity before PPO Game #1** — beating `v3c` **50.8% (+5.4 Elo)** and sitting **+167 Elo** ahead of the old 13-hour `v4` run. It is now launching into a **15,000,000-game PPO self-play run** to push past the `192×6` capacity ceiling toward 1700+ live Elo.

---

## 1. Quick Specs

| Field | Value |
| --- | --- |
| **Generation** | `v4b` (Distilled `256×6` `policy_version=3` Transformer + 15M PPO Self-Play) |
| **Dates** | 2026-11-25 – 2026-11-26 (active) |
| **Distillation directory** | `runs/rust_v4b_m256_l6_distill` |
| **PPO self-play directory** | `runs/rust_v4b_m256_l6_15m` |
| **Pre-PPO best checkpoint** | `runs/rust_v4b_m256_l6_distill/v4b_from_v3d_best.pt` (24.47 MB) |
| **W&B run (15M PPO)** | `4geeolr9` (`rust_v4b_m256_l6_15m`) |
| **Architecture** | `RustFactoredPolicy` (`policy_version=3`, `width=256`, `layers=6`, `ffn_mult=3`) |
| **Parameters** | **6,403,618** (~6.40M) |
| **Observation** | Rust `rich` encoder (**88 floats/Pokémon, 1,124 total**) + frozen 32-feature static move metadata + GPU Speed/TR rank cues |
| **Hardware / Device** | Apple M4 Pro MPS GPU (desktop `num_envs=128, threads=4` -> AFK `--resume --num-envs 256 --threads 8`) |
| **Team pool** | `data/selfplay/v4_nov_meta_weighted_manifest.json` (696 unique Reg M-C teams, **1,168** weighted entries; `>=1500` Elo share `45.1%`) |
| **Holdout pool** | `data/selfplay/mc_sheet_pool_v2/holdout_manifest.json` (**88** unseen Reg M-C teams) |
| **Seed (15M PPO)** | `37301` (with `seed + completed` resume offset) |

---

## 2. Stage 1: Streaming Distillation (`selfplay/distill_rust_policy.py`)

Instead of wasting 13 hours letting a 6.40M-param network rediscover basic type matchups from scratch, we built `selfplay/distill_rust_policy.py` to stream live Rust self-play positions from our best `192×6` teacher and train the `256×6` student to match its turn moves, 90-group Team Preview probabilities, and win-chance predictions.

We ran **three progressive distillation passes**:

| Pass | Teacher -> Student Init | Waves / Games | Wall Time | LR Schedule | Turn Top-1 Agree | Preview Top-1 Agree | Preview KL | Value MAE | Eval vs Teacher / `v3c` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Pass 1 (`v4b_init`)** | `v3c` (11M) -> scratch | 12 waves / 24,643g | **7.6 min** | `3e-4 -> 3.5e-5` | 77.8% | 52.7% | 1.807 | 0.1653 | **39.8% (-72 Elo)** vs `v3c` (+90 Elo over 13h `v4`!) |
| **Pass 2 (`v4b_polished`)** | `v3c` (11M) -> `v4b_init_best` | 24 waves / 49,317g | **14.9 min** | `1.5e-4 -> 1.5e-5` | **83.8%** | **64.6%** | 1.473 | 0.1153 | **45.7% (-30 Elo)** vs `v3c` (+131 Elo over 13h `v4`) |
| **Pass 3 (`v4b_from_v3d`)** | **`v3d` (15M)** -> `v4b_polished_best` | 24 waves / 49,422g | **13.9 min** | `1.2e-4 -> 1.5e-5` | 82.9% | 59.9% | **0.729** (cut in half!) | **0.1049** | **50.0%** wave eval vs `v3d` |

### Verified Pre-PPO Strength of `v4b_from_v3d_best.pt` (1,024 paired unseen holdout games each)
- **vs `v3c` (11M):** **`v4b` won 50.78% ± 2.85% (+5.4 Elo)** — **+167 Elo stronger than the old 13-hour 4M-game `v4` before playing a single game of PPO!**
- **vs `v3d` (15M):** `v4b` scored **48.93% ± 2.86% (-7.5 Elo)** — virtual tie with its 15M-game teacher.
- **vs `simple_heuristics` (holdout):** **86.0%** (peak **89.0%**).
- **vs `max_base_power` (holdout):** **86.0%**.

---

## 3. Stage 2: 15M-Game PPO Self-Play (`runs/rust_v4b_m256_l6_15m`)

With `v4b_from_v3d_best.pt` already at full 15M-game parity and all 5 post-`v4` trainer bugs fixed, we launched `v4b` into a **15,000,000-game** self-play run (`W&B 4geeolr9`).

### Hyperparameters & Config

| Parameter | Value | Why |
| --- | --- | --- |
| **Init checkpoint** | `runs/rust_v4b_m256_l6_distill/v4b_from_v3d_best.pt` | Starts at 15M-game skill instead of random |
| **Target games** | **15,000,000** | Gives the 2× larger 6.40M brain enough runway to surpass `192×6` |
| **Learning rate** | **`2e-5 -> 3e-6`** (cosine) | Gentle warm-start rate so PPO refines without destabilizing distilled weights |
| **Weight decay** | **`0.01`** (decoupled `AdamW`) | Restored post-`v4` regularization on linear weights |
| **Batch / Update** | `games_per_update=1024`, `batch_size=512`, `epochs=2` | Proven stable update size |
| **GAE / Discount** | `gamma=1.0`, `gae_lambda=0.95` | Undiscounted return matching BCE win-probability value head |
| **KL guardrail** | `target_kl=0.025` (checked per minibatch) | Prevents any `v4`-style trust-region blowup |
| **League & Anchors** | `league_size=32`, `snapshot_games=65,536`, `snapshots_per_wave=2`, `pfsp_ema_games=1000.0`, `anchor_fraction=0.10` | Wider 32-snapshot league prevents cyclical forgetting over 15M games |
| **Eval cadence** | `eval_every=65,536` (100g Rust baselines), `vgc_eval_every=262,144` (40g Python `vgc`) | Tracks steady progress without slowing throughput |

### Launch Status (as of 2026-11-26)
- **Status:** **In progress** (just launched in desktop mode: `num_envs=128, threads=4`, switching to AFK `--resume --num-envs 256 --threads 8` overnight).
- **Initial sanity check (first 2 updates, 2,067 games):**
  - Throughput: `64.0 – 67.1 games/sec` (`951 – 1,010 dec/sec`) in desktop mode.
  - `value_accuracy`: `73.3% -> 76.5%` (`value_loss`: `0.4929 -> 0.4576`).
  - `approx_kl`: **`0.0086 – 0.0106`** (super clean, well below `0.025` cap).
  - `vs_anchors_score`: `76.7% -> 78.5%`; `vs_snapshots_score`: `52.2% -> 52.7%`.
- **Live ladder:** Pending completion of 15M-game PPO milestones and offline head-to-head gating against `v3c` and `v3d`.
