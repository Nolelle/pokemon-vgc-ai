# Transformer Version Reports (`v1` – `v4b`)

*Last updated: 2026-11-26*

This directory holds one complete report for every generation of our self-play VGC Transformer, from our first 500k-game CPU prototype (`v1`) up to our distilled 6.40M-parameter `v4b` entering 15M-game self-play.

Every report documents the **exact architecture specs, hyperparameters, team pools, W&B run IDs, throughput, unseen holdout curves, direct head-to-head results, live public Reg M-C ladder runs, and post-mortem lessons** that shaped the next version.

## Side-by-Side Progression (`v1` -> `v4b`)

| Version | Report | Size (`W×L`, Params) | Policy / Obs | Games (Cum.) | Team Pool | Plain Holdout vs `vgc` (avg / peak) | Direct H2H vs Prior | Live Public Peak Elo | Status / Role |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **`v1`** | [v1.md](v1.md) | `128×4` (1.16M) | Python replica v1 | 500k (500k) | 226 M-C | 36.5% / 50.0% | — | **1255** (17–13) | First live baseline |
| **`v2`** | [v2.md](v2.md) | `192×6` (2.87M) | Python replica v1 | 500k (500k) | 748 (mixed/stale) | 27.5% / 42.5% | Worse than `v1` | *Not deployed* | Post-mortem drove Rust `v3` |
| **`v3`** | [v3.md](v3.md) | `192×6` (3.32M) | Rust `v2` / 80-col | 3.0M (3.0M) | 379 clean M-C | 64.0% / 72.5% | ~2× `v1` vs `vgc` | **1346** (16–14) | Rust `VecEnv` breakthrough |
| **`v3b`** | [v3b.md](v3b.md) | `192×6` (3.32M) | Rust `v2` / 80-col | +4.0M (7.0M) | 696 Nov M-C | 63.0% / 72.5% | **62.9% (+92 Elo)** vs `v3` | **1458 -> 1477** (Phase 0–3) | Nov meta + 2-turn & Preview Search |
| **`v3c`** | [v3c.md](v3c.md) | `192×6` (3.32M) | Rust `v2` / **88-col** | +4.0M (11.0M) | 696 Nov M-C | **68.5% / 77.5%** | **58.8% (+62 Elo)** vs `v3b` | **1593 (ALL-TIME PEAK)** | Verified live ladder champion |
| **`v3d`** | [v3d.md](v3d.md) | `192×6` (3.33M) | Rust **`v3`** / 88-col | +4.0M (15.0M) | 696 weighted | 65.5% / 77.5% | **52.7% (+19 Elo)** vs `v3c` | 1490 (23–37 / 60g) | Proved `192×6` ceiling; `v4b` teacher |
| **`v4`** | [v4.md](v4.md) | `256×6` (6.40M) | Rust `v3` / 88-col | 4.0M (scratch) | 696 weighted | 50.0% / 60.0% | 28.3% (-161 Elo) vs `v3c` | *Not deployed* | Exposed 5 trainer bugs |
| **`v4b`** | [v4b.md](v4b.md) | `256×6` (6.40M) | Rust `v3` / 88-col | Distill + **15M PPO** | 696 weighted | *In progress* | **50.8% (+5.4 Elo)** vs `v3c` pre-PPO | *Pending* | Active 15M-game scaling run |

## Individual Reports

1. **[v1.md](v1.md)** — `runs/replica_m128_l4_37101` (`x41s0ucp`): First 500k-game self-play Transformer & 1255 Elo live baseline.
2. **[v2.md](v2.md)** — `runs/replica_v2_m192_l6_748t` (`24meputr`): Failed CPU scale-up whose post-mortem triggered the Rust rewrite.
3. **[v3.md](v3.md)** — `runs/rust_v3_m192_l6_379t` (`gci7x7w5`): Native Rust `VecEnv` + equivariant `policy_version=2`, 3M games, 1346 peak Elo.
4. **[v3b.md](v3b.md)** — `runs/rust_v3b_m192_l6_novmeta` (`oyn1lwwm`): 7M cumulative games on 696 Nov teams + Phase 0–3 search, 1477 peak Elo.
5. **[v3c.md](v3c.md)** — `runs/rust_v3c_m192_l6_counters` (`mvoht8mk`): 11M cumulative games on 88-col counters + Track A search, **1593 all-time peak Elo**.
6. **[v3d.md](v3d.md)** — `runs/rust_v3d_m192_l6_v3upgrade` (`e3463qh4`): 15M cumulative games via zero-loss `--upgrade-to-v3`, confirmed `192×6` ceiling, distilled `v4b`.
7. **[v4.md](v4.md)** — `runs/rust_v4_m256_l6_novmeta` (`61qxadx5`): Cold-start `256×6` post-mortem isolating 5 trainer bugs.
8. **[v4b.md](v4b.md)** — `runs/rust_v4b_m256_l6_distill` & `runs/rust_v4b_m256_l6_15m` (`4geeolr9`): 13.9-minute distillation to 15M-game parity + active 15M-game PPO self-play run.
