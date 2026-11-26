# Transformer v3d (`rust_v3d_m192_l6_v3upgrade`)

> **Takeaway:** `v3d` upgraded `v3c`'s 11M-game brain in-place (`--upgrade-to-v3`) to `policy_version=3` (adding static move facts, Speed/Trick Room rank cues, and the 4× faster 90-group Team Preview head) and trained **4,000,022 more games** (**15.0M cumulative**) with all post-`v4` trainer fixes enabled. It beat `v3c` cleanly offline (**+19 Elo** on holdout, **+29 Elo** on high-Elo meta teams) and served as the **15M-game teacher** for `v4b`, while its diminishing offline gains (`+92 -> +62 -> +19 Elo` per 4M games) and noisy 60-game live check confirmed the `192×6` (3.32M-param) size is near its capacity ceiling.

---

## 1. Quick Specs

| Field | Value |
| --- | --- |
| **Generation** | `v3d` (Zero-loss `v3` Architecture Upgrade on `192×6`) |
| **Dates** | 2026-11-25 – 2026-11-26 |
| **Run directory** | `runs/rust_v3d_m192_l6_v3upgrade` |
| **Warm-start parent** | `runs/rust_v3c_m192_l6_counters/policy_4000012.pt` (11.0M cumulative games, upgraded via `--upgrade-to-v3`) |
| **Final checkpoint** | `runs/rust_v3d_m192_l6_v3upgrade/policy_4000022.pt` (12.73 MB) |
| **W&B run** | `e3463qh4` (`rust_v3d_m192_l6_v3upgrade`) |
| **Architecture** | `RustFactoredPolicy` (`policy_version=3`, `width=192`, `layers=6`, `ffn_mult=2`) |
| **Parameters** | **3,327,010** (~3.33M; +9,216 params for new `move_meta_proj` & `speed_proj`) |
| **Observation** | Rust `rich` encoder: **88 floats/Pokémon (1,124 floats)** + frozen 32-feature static move table + GPU Speed/TR rank cues |
| **Hardware / Device** | Apple M4 Pro MPS GPU (desktop `num_envs=128, threads=4` for 0->2.31M, then AFK `num_envs=256, threads=8` for 2.31M->4.0M) |
| **Training games** | **4,000,022** new games (**15,000,062** cumulative) |
| **Team pool** | `data/selfplay/v4_nov_meta_weighted_manifest.json` (696 unique Reg M-C teams, **1,168** weighted entries; `>=1500` Elo share raised `22.0% -> 45.1%`; 0 holdout overlap) |
| **Holdout pool** | `data/selfplay/mc_sheet_pool_v2/holdout_manifest.json` (**88** unseen Reg M-C teams) |
| **Seed** | `37301` (with post-`v4` `seed + completed` offset on `--resume`) |

---

## 2. What We Built & Changed from v3c

- **Zero-loss `upgrade_v2_to_v3` transfer (`--upgrade-to-v3`):**
  - Copied all `v3c` weights directly and **zero-initialized** the new `move_meta_proj` (32 static move facts from `data/champions/moves.json`) and `speed_proj` (GPU-computed fog-safe Speed & Trick Room move-order ranks), guaranteeing bit-exact `0.0` Day-0 in-battle parity with `v3c`.
  - Fitted the new **90-group permutation-invariant Team Preview head** in ~3.3s on 8,192 boards (`preview_top1_group_agree=84.57%`, `preview_kl=0.9877`).
- **First full run validating all 5 post-`v4` trainer fixes:**
  1. Fixed `dev.type == "mps"` GPU cache cleanup + 64-row padding + 64-row preview micro-batching -> **zero MPS memory leaks** across the entire 4M-game run.
  2. Moved `target_kl=0.025` early stopping inside the minibatch loop -> **448 KL early-stops** kept `approx_kl` capped at `0.0150` max (`0.0014` late avg).
  3. Restored `weight_decay=0.01` on linear weights (decoupled `AdamW`, `0.0` on embeddings/LayerNorm/biases).
  4. Offset `VecEnv` RNG seed by `args.seed + completed` on `--resume` so resuming never replays old matchups.
  5. Widened PFSP (Prioritized Fictitious Self-Play — picking tougher historical opponents more often) EMA window to `pfsp_ema_games=1000.0` + `--snapshots-per-wave 2` + undiscounted `--gamma 1.0`.
- **Both new `v3` feature layers actively learned:** At 4M games, `move_meta_proj` grew from `0.0` to RMS `0.0075` and `speed_proj` grew to RMS `0.0202`.

---

## 3. Training Hyperparameters & Throughput

| Parameter | Value |
| --- | --- |
| **Learning rate** | `3e-5 -> 5e-6` (cosine over 4M games) |
| **Batch / Update** | `games_per_update=1024`, `batch_size=512`, `epochs=2` |
| **GAE / Discount** | **`gamma=1.0`** (undiscounted, matching win-probability value head), `gae_lambda=0.95` |
| **Regularization** | Decoupled `AdamW`, `weight_decay=0.01`, `entropy_coef=0.01`, `target_kl=0.025` (minibatch-checked) |
| **League & Anchors** | `league_size=24`, `snapshot_games=32,768`, `snapshots_per_wave=2`, `pfsp_ema_games=1000.0`, `anchor_fraction=0.10` |
| **Checkpoints saved** | 122 periodic snapshots (`policy_0000000.pt` .. `policy_4000022.pt`) |
| **Throughput** | Peak `131.1 games/sec` (`1,922 dec/sec`); sustained ~`91–94 games/sec` in AFK mode (0 memory leaks) |

---

## 4. Training, Holdout & Head-to-Head Results (Plain Neural, Zero Search)

| Metric | Start (upgraded `v3c`) | Peak | Last-50 / Last-5 Avg | Final (4M / 15M cum.) |
| --- | --- | --- | --- | --- |
| **Win-chance accuracy (`value_accuracy`)** | 75.9% | **78.6%** | 76.0% | 76.7% |
| **Value loss (BCE)** | 0.463 | 0.424 (min) | 0.4578 | 0.454 |
| **Policy update size (`approx_kl`)** | 0.0144 | 0.0150 (max!) | **0.0014** | 0.0009 |
| **Decision randomness (`entropy`)** | 0.87 | 1.11 | 0.89 | 0.91 |
| **Avg battle length (`mean_turns`)** | 7.57 | 8.30 | 7.70 | 7.45 |
| **vs `anchors` (`simple_heuristics`, train)** | 81.9% | 92.3% | 82.5% | 92.0% |
| **vs `max_base_power` (100g holdout)** | 88.0% | **95.0%** @3.33M | **90.8%** | 91.0% |
| **vs `simple_heuristics` (100g holdout)** | 82.0% | **88.0%** @3.93M | 83.8% | 82.0% |
| **vs Python `vgc` 2-ply (40g holdout)** | 62.5% | **77.5%** @1.05M | 65.5% | 60.0% |

### Paired Deterministic Head-to-Head (Seat-Swapped)
- **vs `v3c` (11M) on all 88 unseen holdout teams (1,024 games):** `v3d` won **52.73% ± 2.95% (+19.0 Elo)** (`s0=54.5%, s1=51.0%`).
- **vs `v3c` (11M) on unseen Weather/Screens holdout (512 games):** `v3d` won **51.76% ± 4.07% (+12.2 Elo)**.
- **vs `v3c` (11M) on High-Elo Meta `w>=3` teams (1,024 games):** `v3d` won **54.20% ± 2.79% (+29.2 Elo)** (`s0=54.1%, s1=54.3%`).
- **vs `v3b` (7M) on all 88 unseen holdout teams (1,024 games):** `v3d` won **58.89% ± 2.89% (+62.4 Elo)**.
- **vs `v3c` in `mc_ladder_01` mirror (512 games):** `v3d` won **66.8%**.
- **vs `v3c` with live `rust-search` (32 holdout games + 12 `sun_rain_screens` games):** `v3d + rust-search` won **17–15 (53.1%)** on holdout and **9–3 (75.0%)** vs `sun_rain_screens` (vs `v3c`'s 6–6).

---

## 5. Live Public Reg M-C Ladder Confirmation (60 live games)

Both sessions used `teams/mc_ladder_01.packed.txt`, Track A `10w` search + Preview Search, rejected Open Team Sheets, and logged **0 fallbacks / 0 errors**:

| Session | Record | Elo (Start -> Peak -> End) | Avg Opp | Split (Non-Clones / Rain Clones) |
| --- | --- | --- | --- | --- |
| `20261126T165341Z` (30g) | 11–19 (36.7%) | 1487 -> 1487 -> 1350 | 1398 | `9–12` non-clones, `2–7` clones |
| `20261126T191928Z` (30g) | 12–18 (40.0%) | 1490 -> 1490 -> 1422 | 1424 | `9–9` non-clones, `3–9` clones |
| **Combined (60g)** | **23–37 (38.3%)** | — | ~1411 | **`18–21 (46.2%)` non-clones, `5–16 (23.8%)` clones** |

---

## 6. Key Lessons & What Drove v4b

1. **Zero-bug verification:** Offline checks ruled out any `policy_version=3` or live-search bug (`v3d` beat `v3c` both plain and with `rust-search`, and turn value calibration matched: 75.6% vs 75.5%).
2. **`192×6` capacity ceiling confirmed:** Each 4M-game block on the 3.32M-parameter model yielded sharply smaller offline gains: **`+92 Elo` (`v3->v3b`) -> `+62 Elo` (`v3b->v3c`) -> `+19 Elo` (`v3c->v3d`)**. Keeping `v3c` as the verified 1593-peak live checkpoint while scaling up to **`v4b` (`256×6`, 6.40M params)** was the clear next move.
3. **Best teacher yet:** `v3d` became the 15M-game teacher that distilled **`v4b`** to full 15M-game strength in just 13.9 minutes.
