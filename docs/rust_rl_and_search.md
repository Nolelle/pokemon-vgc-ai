# Rust VecEnv RL & Live Lookahead Search Architecture

*Last updated: 2026-11-24*

This document describes the active playing and training architecture of `pokemon-vgc-ai`
for **Pokemon Showdown Champions VGC 2026 Reg M-C** (`gen9championsvgc2026regmc`).

## Why this architecture

Earlier phases drove Pokemon Showdown's Node.js simulator one turn at a time over IPC,
capping self-play throughput to tens of games per minute. Inspired by MikuMiku-style
self-play + search (see [`mikumiku_replication_pilot.md`](mikumiku_replication_pilot.md)),
we moved self-play and live lookahead to a patched Rust Champions battle simulator
(`pikapro 0.2.0`, patch at [`patches/pikapro/0001-rich-encoder-and-patch-public.patch`](../patches/pikapro/0001-rich-encoder-and-patch-public.patch)):

- **Training throughput:** ~90–100 full doubles games/sec on Apple Silicon (M4 Pro MPS),
  making multi-million-game self-play runs practical overnight.
- **Live lookahead speed:** ~1.0s/turn for full 10-world determinized 2-turn lookahead
  search directly in Rust with unified cross-world MPS GPU batching, plus ~0.12–0.34s
  for Rust Team Preview lookahead search.
- **Verified simulator parity:** 2,126/2,126 (100%) mirror fidelity across weather,
  terrain, Protect chains, Trick Room, Perish Song, screens, Salt Cure, Tailwind,
  Leech Seed, Disable, Taunt, Encore, Yawn, and confusion, plus 558/558 live torture-fuzz
  decisions with 0 action-mask diffs (including Shadow Tag / Arena Trap / Magnet Pull
  trapping, locked moves, and pivot forced switches).

## Components

### 1. Neural policy & value network (`src/vgc/rl/rust_policy.py`)
- **`RUST_POLICY_VERSION = 3`** (`RustFactoredPolicy`, 100% backwards-compatible with
  v2 checkpoints such as `v3c`): a move-order and team-slot equivariant Transformer
  (`256×6`, `ffn_mult=3`, **6.40M parameters** for `v4`; `192×6`, `ffn_mult=2`, **3.32M
  parameters** for `v3`/`v3b`/`v3c`).
- **Inputs (88 floats/mon, 1,124 total):** rich fog-safe per-Pokemon tokens from
  `encode_battle_state_rich` (species, items, abilities, moves, HP, status, boosts,
  volatiles, public turn counters such as Perish/Yawn/Leech Seed/Salt Cure/Heal Block/
  Throat Chop/Disable/Taunt/Encore/Choice Lock, 8-turn extenders, field/side conditions,
  and unbrought opponent preview rows).
- **Built-in v3 domain priors:**
  - Frozen **32-feature static move metadata** table from `data/champions/moves.json`
    (type, category, base power, accuracy, priority, spread targeting, Protect/contact/
    sound flags) so rare moves generalize from Day 1.
  - GPU-computed fog-safe **effective Speed + Trick Room move-order rank cues** over
    living active Pokémon.
- **Outputs:**
  - **90-group permutation-invariant Team Preview head** with `LayerNorm` (collapses
    equivalent lead/back swaps for 4× faster preview and 4× less GPU RAM vs 360-way v2).
  - Semantic move/switch action heads with **Slot 0 -> Slot 1 conditioning** (Slot 1 sees
    the semantic move/switch embedding chosen by Slot 0).
  - Calibrated scalar win-probability value head (BCE loss).

### 2. Self-play trainer (`selfplay/train_rust_vec.py`)
- Vectorized PPO self-play against a **PFSP (Prioritized Fictitious Self-Play)** pool of
  historical checkpoints plus native Rust `simple_heuristics` anchors, with optional
  `--init-from` warm-starts and atomic `--resume`.
- **v4 trainer upgrades:** undiscounted `--gamma 1.0` (consistent with the win-probability
  value head), decoupled `AdamW` weight decay (`0.0` on embeddings/LayerNorm/biases),
  multi-snapshot waves (`--snapshots-per-wave 2`), EMA-capped PFSP win-rate tracking
  (`max_effective_games=200`), merged remainder minibatches, and `--use-manifest-weights`.
- **Rating-weighted meta team pool:** `data/selfplay/v4_nov_meta_weighted_manifest.json`
  — all **696 unique `pikapro`-validated Reg M-C teams** (583 distinct cores, 0 overlap
  with the 88-team holdout split), weighted by ladder rating so `>=1500` Elo teams rise
  from **22.0% -> 45.1%** of sampled games (unweighted pool remains at
  `data/selfplay/v3b_nov_meta_train_manifest.json`).
- **Opponent set priors:** `data/usage/set_priors.json`, rebuilt from **39,589**
  October 15–November 23, 2026 rated (>=1200) Reg M-C replays (out of 171,258 indexed
  replays).

### 3. Live Team Preview + 2-turn regularized search (`src/vgc/rl/rust_search.py`)
- Runs at live decision time (`--policy-mode rust-search`) without reading any private
  opponent state (fog-safe public snapshot only; verified bit-exact invariant to
  hidden opponent mutations on both `p1` and `p2`).
- **Team Preview Search (`search_teampreview_rust`):** deduplicated `(lead2, back2)`
  candidates with setter-diversity guard, fog-safe neural + hybrid opponent world
  predictions, 3-turn batched GPU rollouts, and KL prior regularization (opt-out via
  `--no-rust-preview-search`).
- **In-battle Search (`search_public_rust`):** defaults to **`10w × 8a × 4s × 2000 iters`**
  with uncapped `3×3×3` hypothesis axes so all 10 hidden-info worlds are genuinely
  searched, unified cross-world MPS GPU batching for forced switches/Turn-2 rollouts/leaf
  values, root tail-noise pruning (`<10%`, argmax `>=85%`), and symmetric public-knowledge
  tracking (`BattleMemory` + `RustWorldRoot.encode_obs`) so simulated opponents see our
  revealed items/abilities/moves without leaking private state.
- In paired A/Bs on unseen holdout teams, 2-turn regularized search beat 1-turn
  unregularized search **19–11 (63.3%)** (cutting Protect usage 31%), and root noise
  pruning won **22–18 (55.0%)**.

## Measured progression (live public Reg M-C ladder & unseen holdout)

All live ladder runs use `teams/mc_ladder_01.packed.txt`, reject Open Team Sheets, and
logged **0 fallbacks / 0 protocol errors**:

| Generation | Training | Live Search | Public Ladder (30 games) | Peak Elo | Notes |
| --- | --- | --- | --- | --- | --- |
| `v1` | Initial Rust RL (`192×6`) | 1-turn `rust-search` | — | **1255** | First end-to-end Rust ladder baseline |
| `v3` | 3.0M games (`policy_3000004.pt`), 379 teams | 1-turn `rust-search` | 16–14 (avg opp 1282) | **1346** | +91 Elo peak; post-mortem found 1-turn Protect trap |
| `v3b` | 7.0M total games (`policy_4000024.pt`, +4M on 696 Nov teams) | 2-turn reg `rust-search` (`8w`) + Nov priors | 14–16 (avg opp **1443**) | **1458** | +112 Elo peak over `v3`; beat `v3` **62.9%** (~+92 Elo) on 512 holdout games |
| `v3b` + Phase 0–3 | Same `v3b` (`7.0M`) | `10w` + Preview Search + 88-col counters + noise pruning | 14–16 (avg opp **1443**) | **1477** | **10–2 (83.3%)** vs non-clone ladder; fixed Tyranitar preview blind spot |
| `v3c` | 11.0M total games (`policy_4000012.pt`, +4M on 88-col counters) | Track A `10w` + Preview Search (live run pending) | *Pending* | — | Beat `v3b` **58.8% (+61.7 Elo)** over 1,024 unseen holdout games; **86.6%** vs `simple_heuristics`, **68.5%** avg (**77.5%** peak) vs Python `vgc` 2-ply |

## Common commands

```bash
# 1. Local smoke test (requires local Showdown server on port 8000)
.venv/bin/python ladder/run_ladder.py --local-smoke --n 2 \
  --policy-mode rust-search \
  --rust-checkpoint runs/rust_v3c_m192_l6_counters/policy_4000012.pt \
  --team teams/mc_ladder_01.packed.txt

# 2. Public Reg M-C ladder session
.venv/bin/python ladder/run_ladder.py --n 10 \
  --policy-mode rust-search \
  --rust-checkpoint runs/rust_v3c_m192_l6_counters/policy_4000012.pt \
  --team teams/mc_ladder_01.packed.txt

# 3. Launch v4 (256x6, policy_version=3, 6.40M params) on the rating-weighted Nov meta pool
.venv/bin/python selfplay/train_rust_vec.py \
  --team-manifest data/selfplay/v4_nov_meta_weighted_manifest.json \
  --use-manifest-weights \
  --games 4000000 --width 256 --layers 6 --policy-version 3 \
  --out runs/rust_v4_m256_l6_novmeta
```

## Next steps toward 1700+ Elo

1. **Train `v4` (`256×6`, `policy_version=3`, 6.40M params):** run from scratch on the
   rating-weighted 696-team November meta manifest (~90–96 games/sec on M4 Pro MPS).
2. **Live ladder validation:** benchmark `v3c` and `v4` with the Track A `10w` + Preview
   Search stack on `mc_ladder_01` and top gauntlet rosters (`sand_r1831`, etc.).
3. **Meta generalization & anti-rain/screens:** evaluate whether `v4`'s higher capacity,
   static move/speed priors, and rating-weighted pool close the remaining gap against
   high-Elo `Pelipper + Grimmsnarl + Archaludon` rain/screens cores.
