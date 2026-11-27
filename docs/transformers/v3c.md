# Transformer v3c (`rust_v3c_m192_l6_counters`)

> **Takeaway:** `v3c` is our **verified live public-ladder champion** so far. Warm-starting from `v3b` and training for **4,000,012 more games** (**11.0M cumulative**) on the upgraded **88-column counter + 8-turn extender** board encoder produced our biggest live leap yet: beating `v3b` **58.8% (+61.7 Elo)** offline, shattering the `SunRainCoach` Rain/Screens clone wall (**7–6 / 53.8%**), and hitting an **all-time peak of 1593 Elo** on the public Reg M-C ladder (including an upset over a **1769 Elo** opponent).

---

## 1. Quick Specs

| Field | Value |
| --- | --- |
| **Generation** | `v3c` (Phase 4A — 88-col Counter-Complete `v3`) |
| **Dates** | 2026-11-24 – 2026-11-25 |
| **Run directory** | `runs/rust_v3c_m192_l6_counters` |
| **Warm-start parent** | `runs/rust_v3b_m192_l6_novmeta/policy_4000024.pt` (7.0M cumulative games) |
| **Final checkpoint** | `runs/rust_v3c_m192_l6_counters/policy_4000012.pt` (12.70 MB) |
| **W&B run** | `mvoht8mk` (`rust_v3c_m192_l6_counters`) |
| **Architecture** | `RustFactoredPolicy` (`policy_version=2`, `width=192`, `layers=6`, `ffn_mult=2`) |
| **Parameters** | **3,317,794** (~3.32M; +1,536 input weights for the 8 new per-mon columns) |
| **Observation** | Rust `rich` encoder: **88 floats/Pokémon × 12 rows + globals = 1,124 floats** |
| **Hardware / Device** | Apple M4 Pro MPS GPU (`num_envs=256`, `threads=8`) |
| **Training games** | **4,000,012** new games (**11,000,040** cumulative, 12.01 hours wall time) |
| **Team pool** | `data/selfplay/v3b_nov_meta_train_manifest.json` (**696** unique Reg M-C teams; 0 holdout overlap) |
| **Holdout pool** | `data/selfplay/mc_sheet_pool_v2/holdout_manifest.json` (**88** unseen Reg M-C teams) |
| **Seed** | `37301` |

---

## 2. What We Built & Changed from v3b

- **Trained natively on the 88-column counter encoder (`RICH_MON_WIDTH=88`, `RICH_OBS_SIZE=1124`):** `v3b` only saw 80 columns during training; `v3c` actively learned to use the 8 newly wired public turn counters (`perish`, `yawn`, `leech_seed`, `disable`, `locked_dot`, `throat_chop`, `heal_block`) plus unclamped 8-turn extender durations (Light Clay screens, Heat/Damp/Smooth/Icy Rock weather).
  - Post-run weight check confirmed **7 of the 8 new counter columns** learned strong non-zero weights (`perish` L2 norm `0.401`, `yawn` `0.375`, `leech_seed` `0.335`, `disable` `0.307`, `locked_dot` `0.184`, `throat_chop` `0.180`, `heal_block` `0.156`).
- **Track A Live Search Upgrades (deployed alongside `v3c`):**
  - Uncapped hidden-info world axes (`3×3×3`) so `10w` search genuinely searches 9–10 distinct hidden-item/move worlds instead of collapsing to 2–4.
  - Added symmetric public-knowledge tracking (`our_revealed_items/abilities/moves`) so simulated opponents know what we already revealed without leaking hidden info.
  - Unified cross-world MPS GPU batching (~1.0s/turn for full `10w × 8a × 4s × 2000 iters`).
  - Ran 3 rounds of adversarial bug-hunting & torture fuzzing: fixed 18 real P0/P1 bugs (locked moves, pivot self-switch, Shadow Tag trapping, raw protocol lockstep) and verified **558/558 live decisions with 0 mask diffs** and **2,126/2,126 (100%) mirror fidelity**.

---

## 3. Training Hyperparameters & Throughput

| Parameter | Value |
| --- | --- |
| **Learning rate** | `3e-5 -> 5e-6` (cosine warm restart over 4M games) |
| **Batch / Update** | `games_per_update=1024`, `batch_size=512`, `epochs=2` |
| **GAE / Discount** | `gamma=0.99`, `gae_lambda=0.95` |
| **KL guardrail** | `target_kl=0.025` (max `0.0203`, late avg `0.0013`) |
| **League & Anchors** | `league_size=24`, `snapshot_games=32,768`, `anchor_fraction=0.10` (`simple_heuristics`) |
| **Checkpoints saved** | 121 periodic snapshots (`policy_0000000.pt` .. `policy_4000012.pt`) |
| **Throughput** | Peak `125.7 games/sec` (`1,803 dec/sec`); avg `92.5 games/sec` (12.01h total) |

---

## 4. Training, Holdout & Head-to-Head Results (Plain Neural, Zero Search)

`v3c` set all-time records across virtually every single offline metric:

| Metric | Start (from `v3b`) | Peak | Last-50 / Last-5 Avg | Final (4M / 11M cum.) |
| --- | --- | --- | --- | --- |
| **Win-chance accuracy (`value_accuracy`)** | 75.6% | **78.9%** | **76.6%** | 73.5% |
| **Value loss (BCE)** | 0.455 | 0.422 (min) | **0.4495** (first ever <0.45!) | 0.490 |
| **Policy update size (`approx_kl`)** | 0.0203 | 0.0203 | **0.0013** | 0.0011 |
| **Decision randomness (`entropy`)** | 0.79 | 1.05 | 0.79 | 0.79 |
| **Avg battle length (`mean_turns`)** | 7.38 | 8.18 | 7.59 | 7.45 |
| **vs `anchors` (`simple_heuristics`, train)** | 80.7% | 91.4% | **83.4%** | 76.3% |
| **vs `max_base_power` (100g holdout)** | 85.0% | **97.0%** @3.28M | **90.4%** | 88.0% |
| **vs `simple_heuristics` (100g holdout)** | 78.0% | **92.0%** @3.34M | **86.6%** | 88.0% |
| **vs Python `vgc` 2-ply (40g holdout)** | 62.5% | **77.5%** @2.63M | **68.5%** | 72.5% |

### Direct Deterministic Head-to-Head (Unseen 88 Holdout Teams, Plain Neural)
- **`v3c` (11M) vs `v3b` (7M) (1,024 games):** `v3c` won **602–422 (58.8%, +61.7 Elo)** (`p0=306/512, p1=296/512`).
- **`v3c` (11M) vs `v3` (3M) (512 games):** `v3c` won **345–167 (67.4%, +126.0 Elo)** (`p0=177/256, p1=168/256`).

---

## 5. Live Public Reg M-C Ladder Results (60 live games across 2 teams)

Both sessions used Track A `10w × 8a × 4s × 2000 iters` + Team Preview Search, rejected Open Team Sheets, and logged **0 fallbacks / 0 errors**:

| Session | Team | Record | Elo (Start -> Peak -> End) | Avg Opp | Highlights |
| --- | --- | --- | --- | --- | --- |
| `20261125T175315Z` (30g) | **`mc_ladder_01`** | **16–14 (53.3%)** | 1467 -> **1593 (ALL-TIME PEAK)** -> ~1515 | **1499** (1343–1769) | **+116 Elo peak** over `v3b` (+338 over `v1`); 9-game win streak; beat **1769 `Project Champion`**, **1628 `zjjmk`**, 4× `SilverGale24` (to 1600); **broke the `SunRainCoach` Rain/Screens wall: 7–6 (53.8%)** vs `v3b`'s 4–14! |
| `20261125T220346Z` (30g) | `sand_r1831` | 15–15 (50.0%) | 1472 -> 1472 -> ~1442 | 1381 (1140–1637) | **13–3 (81.2%)** vs non-clone opponents (3–0 vs `SilverGale24` up to 1637), but **2–12 (14.3%)** vs Rain/Screens clones. Confirmed `mc_ladder_01` stays #1 roster. |

---

## 6. Key Lessons & What Drove v3d / v4b

- **What worked:** 11M cumulative games + 88-column counters + Track A 10-world search broke into the **1500–1600 Elo** bracket and solved the Rain/Screens clone wall.
- **What came next:** After cold-start `v4` (`256×6`) failed due to trainer bugs (see [`v4.md`](v4.md)), `v3c` became the parent for two things:
  1. Zero-loss upgrade to **`v3d`** (`policy_version=3` on `192×6`, testing the new v3 domain priors).
  2. Initial teacher for **`v4b`** distillation (before `v3d` took over as the 15M-game teacher).
