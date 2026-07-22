# pokemon-vgc-ai

A rules-first Pokémon Showdown bot for the Champions VGC 2026 Reg M-B doubles ladder.
It uses the Champions mod's exported data, a simulator-checked damage engine, and an
explicit one-turn evaluator before any learned components are introduced.

## Candidate-aware reinforcement learning

The default ladder bot remains unchanged. An experimental, default-off PPO path under
`vgc.rl` scores every complete legal doubles order, masks padded/illegal candidates,
and learns a policy plus position value from final battle outcomes. Its state encoder
can warm-start from the existing behavior-cloning checkpoint; the joint-action policy
head itself starts new because the old model predicted each active slot independently.

Run a one-battle end-to-end smoke (local Showdown battle, terminal reward, PPO update,
checkpoint write) with:

```bash
.venv/bin/python selfplay/train_ppo.py --iterations 1 --games-per-iteration 1 \
  --out-dir runs/ppo/smoke
```

Longer experiments use the same command with larger iteration/game counts. Checkpoints
and JSONL metrics are written under `runs/ppo/` and are not used by the public ladder
runner until an RL candidate passes separate offline and public-smoke gates.

## Local verification

```bash
.venv/bin/python -m pytest
.venv/bin/python -m pytest -m integration
.venv/bin/python -m ruff check .
```

The project automatically selects an installed Node 22 executable. Set `VGC_NODE` if
Node 22 lives somewhere unusual.

## Offline evaluation

Start the sibling Showdown server, then run the two acceptance gates:

```bash
.venv/bin/python offline/run_gates.py --candidate vgc --incumbent random --n 100 \
  --threshold 0.90 --team teams/meta1.packed.txt
.venv/bin/python offline/run_gates.py --candidate vgc --incumbent heuristic --n 300 \
  --threshold 0.65 --team teams/meta1.packed.txt
```

Both players explicitly accept Open Team Sheets by default. Use
`--no-open-team-sheets` to make both reject for a controlled comparison.

## Curated metagame teams

`data/meta/popular_teams_H8v7TEZcbXo.json` contains the ten teams shown in JoeUX9's
"The Most Popular Teams In Pokemon Champions Explained": 60 complete sets plus the
stated roles and common leads. At team preview, `vgc.meta` recognizes only an exact
six-species match. The evaluator then uses the video's hidden nature for its Speed and
damage estimates and records the archetype in `VGC_TRACE` output.

Live Open Team Sheet information always wins for moves, items, and abilities. The video
did not provide Stat Point spreads, so those still come from `data/usage/spreads.json`.
Validate the curated ids against the Champions export with:

```bash
.venv/bin/python -c 'from vgc.meta import validate_meta_teams; assert not validate_meta_teams()'
```

## Ladder sessions

First exercise the replay, trace, and JSONL logging pipeline locally:

```bash
.venv/bin/python ladder/run_ladder.py --local-smoke --n 2
```

For the public ladder, provide credentials through environment variables:

```bash
export VGC_SHOWDOWN_USERNAME='your-account'
export VGC_SHOWDOWN_PASSWORD='your-password'
.venv/bin/python ladder/run_ladder.py --n 1
```

Alternatively create the gitignored `.showdown-credentials.json`:

```json
{"username": "your-account", "password": "your-password"}
```

Use owner-only permissions (`chmod 600 .showdown-credentials.json`). Replays and
decision traces go under `runs/ladder/`; one outcome record per game is appended to
`runs/ladder.jsonl`. Start with small, respectful sessions.
