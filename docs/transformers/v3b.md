# Transformer v3b (`rust_v3b_m192_l6_novmeta`)

> **Takeaway:** `v3b` warm-started from `v3`'s 3M-game brain and played **4,000,024 more games** (**7.0M cumulative**) on a fresh **696-team November meta pool**. In direct head-to-head on unseen teams, `v3b` beat `v3` **62.9% (+92 Elo)**. Live on the public ladder, it jumped to a **1458 peak** (and **1477 peak** after our Phase 0–3 search upgrades, going **10–2 / 83.3%** against all non-clone opponents).

---

## 1. Quick Specs

| Field | Value |
| --- | --- |
| **Generation** | `v3b` (Warm-started `v3` on November Meta Pool) |
| **Dates** | 2026-11-23 – 2026-11-24 |
| **Run directory** | `runs/rust_v3b_m192_l6_novmeta` |
| **Warm-start parent** | `runs/rust_v3_m192_l6_379t/policy_3000004.pt` (3.0M games) |
| **Final checkpoint** | `runs/rust_v3b_m192_l6_novmeta/policy_4000024.pt` (12.69 MB) |
| **W&B run** | `oyn1lwwm` (`rust_v3b_m192_l6_novmeta`) |
| **Architecture** | `RustFactoredPolicy` (`policy_version=2`, `width=192`, `layers=6`, `ffn_mult=2`) |
| **Parameters** | **3,316,258** (~3.32M) |
| **Observation** | Rust `rich` encoder: **80 floats/Pokémon (1,028 floats)** during training (forward-compatible with 88-col live search via zero-padding) |
| **Hardware / Device** | Apple M4 Pro MPS GPU (`num_envs=256`, `threads=8`) |
| **Training games** | **4,000,024** new games (**7,000,028** cumulative, 11.56 hours wall time) |
| **Team pool** | `data/selfplay/v3b_nov_meta_train_manifest.json` (**696** unique Reg M-C teams across 583 6-species cores; 0 holdout overlap) |
| **Holdout pool** | `data/selfplay/mc_sheet_pool_v2/holdout_manifest.json` (**88** unseen Reg M-C teams) |
| **Seed** | `37301` |

---

## 2. What We Built & Changed from v3

- **Refreshed opponent set priors:** Downloaded 1,303 fresh November >=1300 replays and rebuilt `data/usage/set_priors.json` from **39,589** Oct 15–Nov 23 rated Reg M-C replays (out of 171,258 indexed).
- **696-team modern November meta pool:** Combined 445 fresh Oct–Nov OTS teams (>=1250 Elo) + 251 teams from `v3_clean`, capped at max 3 variants per 6-species core, doubling/tripling coverage of Rain, Screens, Sun/TR, Farigiraf, Delphox, and Sinistcha.
- **Warm-start support (`--init-from`):** Added clean weight transfer to `selfplay/train_rust_vec.py` so `v3b` inherited all 3M games of `v3` skill with a gentle cosine LR restart (`3e-5 -> 5e-6`).
- **Mid-generation Live Search Upgrades (Phases 0–3, tested on `v3b`):**
  - **Phase 0:** Fixed 10 mirror-fidelity bugs and expanded Rust `rich` obs to **88 floats/Pokémon** (`1,124` total) with full turn counters (Perish, Yawn, Leech Seed, Salt Cure, Disable, Taunt, Encore, Choice Lock, 8-turn extenders) — verified **2,126/2,126 (100%)** mirror fidelity.
  - **Phase 1:** Added **Rust Team Preview Lookahead Search (`search_teampreview_rust`)** with setter-diversity guard and 3-turn batched GPU rollouts (~0.12–0.34s).
  - **Phase 2:** Added root noise pruning (`<10%` tail pruned, argmax `>=85%`, won **22–18 / 55%** A/B) and upgraded live search to **10 worlds** (`10w × 8a × 4s × 2000 iters`).
  - **Phase 3:** 160-game paired Team Gauntlet confirmed `mc_ladder_01` as #1 (**25/32 = 78.1%**), ahead of `sand_r1831` (68.8%).

---

## 3. Training Hyperparameters & Throughput

| Parameter | Value |
| --- | --- |
| **Learning rate** | `3e-5 -> 5e-6` (cosine warm restart over 4M games) |
| **Batch / Update** | `games_per_update=1024`, `batch_size=512`, `epochs=2` |
| **GAE / Discount** | `gamma=0.99`, `gae_lambda=0.95` |
| **KL guardrail** | `target_kl=0.025` (max observed `0.0191` — rock solid!) |
| **League & Anchors** | `league_size=24`, `snapshot_games=32,768`, `anchor_fraction=0.10` (`simple_heuristics`) |
| **Checkpoints saved** | 121 periodic snapshots (`policy_0000000.pt` .. `policy_4000024.pt`) |
| **Throughput** | Peak `124.2 games/sec` (`1,747 dec/sec`); late avg `92.9 games/sec` (11.56h total) |

---

## 4. Training, Holdout & Head-to-Head Results (Plain Neural, Zero Search)

| Metric | Start (from `v3`) | Peak | Last-50 / Last-5 Avg | Final (4M / 7M cum.) |
| --- | --- | --- | --- | --- |
| **Win-chance accuracy (`value_accuracy`)** | 74.4% | **78.5%** | **76.0%** | 76.3% |
| **Value loss (BCE)** | 0.465 | 0.425 (min) | **0.4590** | 0.455 |
| **Policy update size (`approx_kl`)** | 0.0178 | 0.0191 | **0.0012** | 0.0008 |
| **Decision randomness (`entropy`)** | 0.89 | 1.07 | 0.80 | 0.81 |
| **Avg battle length (`mean_turns`)** | 7.05 | 7.81 | 7.45 | 7.64 |
| **vs `anchors` (`simple_heuristics`, train)** | 80.0% | 91.1% | **80.6%** | 77.5% |
| **vs `max_base_power` (100g holdout)** | 87.0% | **95.0%** @2.02M | **90.0%** | 91.0% |
| **vs `simple_heuristics` (100g holdout)** | 75.0% | **85.0%** @3.24M | **78.8%** | 76.0% |
| **vs Python `vgc` 2-ply (40g holdout)** | 60.0% | **72.5%** @3.02M | **63.0%** | 57.5% |

### Direct Deterministic Head-to-Head (`v3b` 7M vs `v3` 3M, plain neural)
- **All 88 unseen holdout teams (512 games):** `v3b` won **62.9% (~+92 Elo)**.
- **Unseen Archaludon/Pelipper/Grimmsnarl subset (512 games):** `v3b` won **59.8% (~+70 Elo)**.

---

## 5. Live Public Reg M-C Ladder Results (60 live games)

Both sessions used `teams/mc_ladder_01.packed.txt`, rejected Open Team Sheets, and logged **0 fallbacks / 0 errors**:

| Session | Live Search Stack | Record | Elo (Start -> Peak -> End) | Avg Opp | Highlights |
| --- | --- | --- | --- | --- | --- |
| `20261123T214021Z` (30g) | 2-turn reg `8w×8a×4s` + Nov priors | 14–16 (46.7%) | 1357 -> **1458** -> 1392 | **1443** | **+112 Elo peak** over `v3`; never dropped below 1357; beat **1650 `YakinikuBOT2`**, 1559/1536 `SilverGale24`, 1518 `SunRainCoach` |
| `20261124T033930Z` (30g) | **Phase 0–3:** `10w×8a×4s×2000it` + Preview Search + noise pruning | 14–16 (46.7%) | 1414 -> **1477** -> 1395 | **1443** | **10–2 (83.3%)** vs all non-clone opponents (~1650+ Elo pace!), **3–1** vs `YakinikuBOT2` up to 1622; Preview Search fixed Tyranitar brings (`0/14` -> `12/18`) vs Rain clones |

---

## 6. Key Lessons & What Drove v3c

- **What worked:** Warm-starting on the 696-team Nov meta pool delivered a massive **+92 Elo** offline jump and **+131 Elo** live peak jump (`1346 -> 1477`), crushing diverse ladder opponents (`10–2`).
- **Remaining bottleneck:** Against the `SunRainCoach` Rain/Screens clone cluster (60% of the second session!), `v3b` still went **4–14 (22.2%)**. Crucially, `v3b` had been trained on the old 80-col encoder (before the Phase 0 88-col counters & 8-turn extender timers existed). Teaching the neural brain itself to read those new 88-col counters became **`v3c`**.
