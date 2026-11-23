# Rust VecEnv RL & Live Lookahead Search Architecture

*Last updated: 2026-11-23*

This document describes the active playing and training architecture of `pokemon-vgc-ai`
for **Pokemon Showdown Champions VGC 2026 Reg M-C** (`gen9championsvgc2026regmc`).

## Why this architecture

Earlier phases drove Pokemon Showdown's Node.js simulator one turn at a time over IPC,
capping self-play throughput to tens of games per minute. Inspired by MikuMiku-style
self-play + search (see [`mikumiku_replication_pilot.md`](mikumiku_replication_pilot.md)),
we moved self-play and live lookahead to a patched Rust Champions battle simulator
(`pikapro 0.2.0`, patch at [`patches/pikapro/0001-rich-encoder-and-patch-public.patch`](../patches/pikapro/0001-rich-encoder-and-patch-public.patch)):

- **Training throughput:** ~100 full doubles games/sec on Apple Silicon (M4 Pro), making
  multi-million-game self-play runs practical overnight.
- **Live lookahead speed:** ~0.33s/turn for determinized 2-turn lookahead search directly
  in Rust, using the trained neural policy to shortlist actions and evaluate leaf states.

## Components

### 1. Neural policy & value network (`src/vgc/rl/rust_policy.py`)
- **`RUST_POLICY_VERSION = 2`** (`RustTransformerPolicy`): a 6-layer, 192-dim (~3.32M
  parameters) move-order and team-slot equivariant Transformer.
- **Inputs:** rich per-Pokemon tokens (species, items, abilities, moves, HP, status,
  boosts, volatiles, field/side conditions, and unbrought opponent preview rows) from
  `encode_battle_state_rich`.
- **Outputs:**
  - Team-preview 4-of-6 + lead selection logits.
  - Semantic move/switch action heads with **Slot 0 -> Slot 1 conditioning** (Slot 1 sees
    the semantic move/switch embedding chosen by Slot 0).
  - Scalar win-probability value head.

### 2. Self-play trainer (`selfplay/train_rust_vec.py`)
- Vectorized PPO self-play against a **PFSP (Prioritized Fictitious Self-Play)** pool of
  historical checkpoints plus heuristic anchors, with optional `--init-from` warm-starts.
- **Meta team pool:** `data/selfplay/v3b_nov_meta_train_manifest.json` — **696 unique
  `pikapro`-validated Reg M-C teams** (583 distinct cores, 0 overlap with the 88-team
  holdout split).
- **Opponent set priors:** `data/usage/set_priors.json`, rebuilt from **39,589**
  October 15–November 23, 2026 rated (>=1200) Reg M-C replays (out of 171,258 indexed
  replays).

### 3. Live 2-turn regularized search (`src/vgc/rl/rust_search.py`)
- Runs at live decision time (`--policy-mode rust-search`) without reading any private
  opponent state (fog-safe public snapshot only).
- Samples `8` plausible hidden opponent set determinizations (*worlds*) from
  `data/usage/set_priors.json`, shortlists top `8` own actions and `4` opponent replies
  using the neural policy, simulates turn 1 plus a greedy 1-turn policy rollout
  (`rollout_turns=1`), and regularizes with `min_prior_prob=0.01` and `prior_bonus=0.025`
  to avoid 1-turn Protect-stall traps.
- In a 30-game paired A/B against 1-turn unregularized search, 2-turn regularized search
  won **19–11 (63.3%)** and cut Protect usage by 31%.

## Measured progression (live public Reg M-C ladder)

All live ladder runs use `teams/mc_ladder_01.packed.txt`, reject Open Team Sheets, and
logged **0 fallbacks / 0 protocol errors**:

| Generation | Training | Live Search | Public Ladder (30 games) | Peak Elo | Notes |
| --- | --- | --- | --- | --- | --- |
| `v1` | Initial Rust RL (`192×6`) | 1-turn `rust-search` | — | **1255** | First end-to-end Rust ladder baseline |
| `v3` | 3.0M games (`policy_3000004.pt`), 379 teams | 1-turn `rust-search` | 16–14 (avg opp 1282) | **1346** | +91 Elo peak; post-mortem found 1-turn Protect trap |
| `v3b` | 7.0M total games (`policy_4000024.pt`, +4M on 696 Nov teams) | 2-turn reg `rust-search` + Nov priors | 14–16 (avg opp **1443**) | **1458** | +112 Elo peak over `v3`; beat **1650**, **1559**, **1536**, **1518** opponents; beat `v3` **62.9%** (~+92 Elo) on 512 unseen holdout games |

## Common commands

```bash
# 1. Local smoke test (requires local Showdown server on port 8000)
.venv/bin/python ladder/run_ladder.py --local-smoke --n 2 \
  --policy-mode rust-search \
  --rust-checkpoint runs/rust_v3b_m192_l6_novmeta/policy_4000024.pt \
  --team teams/mc_ladder_01.packed.txt

# 2. Public Reg M-C ladder session
.venv/bin/python ladder/run_ladder.py --n 10 \
  --policy-mode rust-search \
  --rust-checkpoint runs/rust_v3b_m192_l6_novmeta/policy_4000024.pt \
  --team teams/mc_ladder_01.packed.txt

# 3. Train / continue a Rust VecEnv RL run
.venv/bin/python selfplay/train_rust_vec.py \
  --team-manifest data/selfplay/v3b_nov_meta_train_manifest.json \
  --total-games 4000000 --d-model 256 --n-layers 6 \
  --out runs/rust_v4_m256_l6_novmeta
```

## Next steps toward 1700+ Elo

1. **Scale network capacity (`v4`):** train a `256×6` (~5.60M-parameter) model from
   scratch on the 696-team November meta pool (~96 games/sec on M4 Pro, ~70% more
   capacity than `v3b` at nearly identical speed).
2. **Meta generalization & anti-rain/screens:** widen training coverage and test stronger
   ladder rosters against the current Reg M-C meta (`Pelipper + Grimmsnarl + Archaludon`
   rain/screens cores, psyspam, sand).
3. **Deeper/smarter live search:** evaluate deeper search budgets and opponent-belief
   updates once `v4` raises the policy and value baseline.
