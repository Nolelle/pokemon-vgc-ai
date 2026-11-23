# pokemon-vgc-ai

A Pokémon Showdown AI bot for **[Gen 9 Champions] VGC 2026 Reg M-C**
(`gen9championsvgc2026regmc`): doubles, bring-6-pick-4, level 50, Mega Evolutions
allowed, best-of-one with **hidden opponent team sheets** (the bot rejects Open Team
Sheets and infers hidden sets from public reveals and recent ladder usage data).

## Goal & Current State (2026-11-23)

- **Target:** **1700+ Elo** on the public Champions Reg M-C ladder.
- **Current best:** **1458 peak Elo** on the live public Reg M-C ladder (30-game session
  on `teams/mc_ladder_01.packed.txt`, averaging **1443** opponent Elo, **0 fallbacks**,
  including wins over **1650**, **1559**, **1536**, and **1518** Elo opponents).
- **Active architecture:** a fast **Rust vectorized self-play reinforcement learning
  (RL)** pipeline paired with **2-turn regularized live lookahead search**:
  - **Policy & value network:** [`src/vgc/rl/rust_policy.py`](src/vgc/rl/rust_policy.py)
    (`RUST_POLICY_VERSION = 2`), a move-order and team-slot equivariant Transformer
    trained for **7M self-play games** (`v3b`, `192×6`, 3.32M parameters) on a
    **696-team** November Reg M-C meta pool.
  - **Live lookahead search:** [`src/vgc/rl/rust_search.py`](src/vgc/rl/rust_search.py)
    samples 8 plausible opponent set worlds from refreshed November Reg M-C priors
    ([`data/usage/set_priors.json`](data/usage/set_priors.json), built from **39,589**
    Oct 15–Nov 23 rated replays) and simulates turn 1 plus a 1-turn policy rollout in
    ~0.33s/turn.
  - **Exact Champions rules foundation:** all game data comes from the local Showdown
    `champions` mod export (`data/champions/*.json` — Stat Points, 166 legal items,
    custom Megas, Champions learnsets), backed by a simulator-verified damage engine
    and strict fog-of-war boundaries.
- **Next milestone:** train a larger **`256×6` (~5.6M-parameter) `v4`** network on the
  696-team November meta pool and test stronger ladder rosters toward 1550–1700+ Elo.
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
  --rust-checkpoint runs/rust_v3b_m192_l6_novmeta/policy_4000024.pt \
  --team teams/mc_ladder_01.packed.txt
```

2. For the public Showdown ladder, supply credentials via environment variables
   (`VGC_SHOWDOWN_USERNAME`, `VGC_SHOWDOWN_PASSWORD`) or a `chmod 600` gitignored
   `.showdown-credentials.json` (`{"username": "...", "password": "..."}`):

```bash
.venv/bin/python ladder/run_ladder.py --n 10 \
  --policy-mode rust-search \
  --rust-checkpoint runs/rust_v3b_m192_l6_novmeta/policy_4000024.pt \
  --team teams/mc_ladder_01.packed.txt
```

Replays and decision traces are saved under `runs/ladder/`, and one summary record per
game is appended to `runs/ladder.jsonl`. The rule-based heuristic (`--policy-mode heuristic`)
remains available as the zero-checkpoint fallback.
