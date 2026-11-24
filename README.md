# pokemon-vgc-ai

A Pokémon Showdown AI bot for **[Gen 9 Champions] VGC 2026 Reg M-C**
(`gen9championsvgc2026regmc`): doubles, bring-6-pick-4, level 50, Mega Evolutions
allowed, best-of-one with **hidden opponent team sheets** (the bot rejects Open Team
Sheets and infers hidden sets from public reveals and recent ladder usage data).

## Goal & Current State (2026-11-24)

- **Target:** **1700+ Elo** on the public Champions Reg M-C ladder.
- **Current best live ladder:** **1477 peak Elo** on the live public Reg M-C ladder
  (30-game Phase 0–3 session with `v3b` on `teams/mc_ladder_01.packed.txt`, averaging
  **1443** opponent Elo, **0 fallbacks**, **10–2 (83.3%)** vs non-clone opponents,
  including wins over **1650**, **1622**, **1559**, **1541**, and **1536** Elo
  opponents).
- **Current best trained checkpoint (`v3c`, 11M self-play games):**
  `runs/rust_v3c_m192_l6_counters/policy_4000012.pt` (`192×6`, 3.32M parameters,
  `RUST_POLICY_VERSION = 2`), trained on the 88-column counter + 8-turn extender rich
  encoder and 696-team November meta pool. Beat `v3b` **58.8% (+61.7 Elo)** over 1,024
  unseen holdout games (live ladder validation pending).
- **Active architecture:** a fast **Rust vectorized self-play reinforcement learning
  (RL)** pipeline paired with **Team Preview search + 2-turn regularized live lookahead
  search**:
  - **Policy & value network:** [`src/vgc/rl/rust_policy.py`](src/vgc/rl/rust_policy.py)
    now supports **`RUST_POLICY_VERSION = 3`** (backwards-compatible with v2 checkpoints
    like `v3c`): a move-order and team-slot equivariant Transformer with 32-feature
    static move metadata, GPU-computed fog-safe Speed/Trick Room rank cues, and a
    90-group permutation-invariant Team Preview head (`256×6`, `ffn_mult=3`, **6.40M**
    parameters for `v4`).
  - **Live lookahead search:** [`src/vgc/rl/rust_search.py`](src/vgc/rl/rust_search.py)
    runs Rust Team Preview lookahead plus in-battle 2-turn regularized search across
    **10 true hidden-info worlds** (`10w × 8a × 4s × 2000 iters`, cross-world MPS GPU
    batching, ~1.0s/turn) using refreshed November Reg M-C priors
    ([`data/usage/set_priors.json`](data/usage/set_priors.json), built from **39,589**
    Oct 15–Nov 23 rated replays) and verified **2,126/2,126 (100%)** mirror fidelity.
  - **Exact Champions rules foundation:** all game data comes from the local Showdown
    `champions` mod export (`data/champions/*.json` — Stat Points, 166 legal items,
    custom Megas, Champions learnsets), backed by a simulator-verified damage engine
    and strict fog-of-war boundaries.
- **Next milestone:** launch the **`v4` (`256×6`, 6.40M-parameter, `policy_version=3`)**
  self-play run on the rating-weighted 696-team November meta manifest
  ([`data/selfplay/v4_nov_meta_weighted_manifest.json`](data/selfplay/v4_nov_meta_weighted_manifest.json))
  and validate `v3c`/`v4` + Track A live search on the public ladder toward 1700+ Elo.
- Coaching UI / product work remains deferred until the 1700+ Elo playing goal is met.

## Documentation

- **Current architecture & results:** [`docs/rust_rl_and_search.md`](docs/rust_rl_and_search.md)
- **Developer handbook & Champions quirks:** [`CLAUDE.md`](CLAUDE.md) (and [`AGENTS.md`](AGENTS.md))
- **Model inventory:** [`data/models/registry.json`](data/models/registry.json)
- **Active technical contracts (enforced by tests & gates):**
  - [`docs/champions_mechanics_catalog.md`](docs/champions_mechanics_catalog.md) & [`docs/mechanics_coverage.md`](docs/mechanics_coverage.md)
  - [`docs/battle_state_contract.md`](docs/battle_state_contract.md) & [`docs/action_generation_contract.md`](docs/action_generation_contract.md)
  - [`docs/calculator_coverage.md`](docs/calculator_coverage.md), [`docs/data_requirements.md`](docs/data_requirements.md), [`docs/learned_model_contract.md`](docs/learned_model_contract.md), [`docs/replay_label_contract.md`](docs/replay_label_contract.md), [`docs/search_contract.md`](docs/search_contract.md)
  - [`docs/mikumiku_replication_pilot.md`](docs/mikumiku_replication_pilot.md)
- **Historical audits, superseded plans, and dated experiment logs:** [`docs/archive/`](docs/archive/README.md)

## Quickstart & Local Verification

Environment uses `uv` (`.venv/bin/python`). Node 22 is auto-detected via `vgc.node.find_node`
(override with `VGC_NODE`).

```bash
# Unit tests (fast, no Showdown server needed)
.venv/bin/python -m pytest -m "not integration"

# Simulator integration tests (needs local Showdown checkout + Node 22)
.venv/bin/python -m pytest -m integration

# Lint
.venv/bin/ruff check .
```

## Running on the Ladder

1. Always verify with a local Showdown smoke run first (start the local server with
   `node pokemon-showdown start --no-security`):

```bash
.venv/bin/python ladder/run_ladder.py --local-smoke --n 2 \
  --policy-mode rust-search \
  --rust-checkpoint runs/rust_v3c_m192_l6_counters/policy_4000012.pt \
  --team teams/mc_ladder_01.packed.txt
```

2. For the public Showdown ladder, supply credentials via environment variables
   (`VGC_SHOWDOWN_USERNAME`, `VGC_SHOWDOWN_PASSWORD`) or a `chmod 600` gitignored
   `.showdown-credentials.json` (`{"username": "...", "password": "..."}`):

```bash
.venv/bin/python ladder/run_ladder.py --n 10 \
  --policy-mode rust-search \
  --rust-checkpoint runs/rust_v3c_m192_l6_counters/policy_4000012.pt \
  --team teams/mc_ladder_01.packed.txt
```

Replays and decision traces are saved under `runs/ladder/`, and one summary record per
game is appended to `runs/ladder.jsonl`. The rule-based heuristic (`--policy-mode heuristic`)
remains available as the zero-checkpoint fallback.
