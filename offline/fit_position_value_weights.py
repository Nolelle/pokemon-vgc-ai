#!/usr/bin/env python
"""Fit the exact search's scorecard weights from who actually won rated M-C games.

`vgc.rl.exact_search._side_position` scores a board with hand-chosen weights (alive 90,
status 18, sleep 30, 5 per stat stage, 12 per signed effect; 100 = one Pokemon's full
HP). Nobody has checked them against real outcomes. This script fits a logistic
regression of `won` on the SAME our-minus-opponent board features over the human replay
corpus (`data/bc/decisions_regmc.jsonl.gz`, start-of-turn states) and reports every
coefficient in the search's units (coefficient / HP coefficient x 100).

This is the "value supervised on real outcomes, no bootstrapping" idea from the HAL Melee
write-up, applied to a linear scorecard instead of a network. It is correlational:
players set up when they are already ahead in ways the board does not show, which
inflates setup terms. `--lagged` adds the same player's previous-turn board as controls
(asks: given where you were a turn ago, what is ending the turn with X worth?). Only a
pool A/B with the fitted weights says whether they play better.

Replay records drop fainted Pokemon from the lists instead of marking them, so the number
alive is reconstructed from `unseen_count` (see `_side_features`).

Run: .venv/bin/python offline/fit_position_value_weights.py --output runs/eval/value_fit.json
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPO_ROOT / "data/bc/decisions_regmc.jsonl.gz"
BRING_SIZE = 4

SCREENS = ("reflect", "lightscreen", "auroraveil")
HAZARDS = ("stealthrock", "spikes", "toxicspikes", "stickyweb")
FEATURES = (
    "alive",
    "hp",
    "boost_offense",  # Atk + SpA stages on active Pokemon
    "boost_defense",  # Def + SpD
    "boost_speed",
    "boost_acc_eva",  # accuracy + evasion
    "burn",
    "paralysis",
    "poison",  # psn + tox
    "sleep_freeze",
    "tailwind_fresh",  # first start-of-turn decision it is up (just set: most turns left)
    "tailwind_old",
    "screen_fresh",  # count of Reflect / Light Screen / Aurora Veil, first turn up
    "screen_old",
    "hazard",  # count of hazard kinds on this side (bad for the holder)
)
# Hand weights in search units, for the side-by-side table. The search pays one
# exact_search_effect_weight per signed side condition; hazards are signed -1.
HAND = {
    "alive": 90.0,
    "hp": 100.0,
    "boost_offense": 5.0,
    "boost_defense": 5.0,
    "boost_speed": 5.0,
    "boost_acc_eva": 5.0,
    "burn": -18.0,
    "paralysis": -18.0,
    "poison": -18.0,
    "sleep_freeze": -30.0,
    "tailwind_fresh": 12.0,
    "tailwind_old": 12.0,
    "screen_fresh": 12.0,
    "screen_old": 12.0,
    "hazard": -12.0,
}


def _side_features(side: dict, fresh: dict[str, bool]) -> list[float]:
    listed = [m for m in side["active"] if m] + list(side["bench"])
    alive = [m for m in listed if (m.get("hp_fraction") or 0) > 0 and m.get("status") != "fnt"]
    # Fainted Pokemon are dropped from active/bench; seen = 6 - unseen includes them.
    unseen_alive = max(0, BRING_SIZE - (6 - side["unseen_count"]))
    active = [m for m in side["active"] if m and (m.get("hp_fraction") or 0) > 0]

    def stages(*keys: str) -> float:
        return float(sum((m.get("boosts") or {}).get(k, 0) for m in active for k in keys))

    def status(*codes: str) -> float:
        return float(sum(1 for m in alive if m.get("status") in codes))

    conditions = set(side.get("side_conditions") or [])
    screens_up = [s for s in SCREENS if s in conditions]
    return [
        float(len(alive) + unseen_alive),
        sum(float(m["hp_fraction"]) for m in alive) + unseen_alive,
        stages("atk", "spa"),
        stages("def", "spd"),
        stages("spe"),
        stages("accuracy", "evasion"),
        status("brn"),
        status("par"),
        status("psn", "tox"),
        status("slp", "frz"),
        float("tailwind" in conditions and fresh.get("tailwind", False)),
        float("tailwind" in conditions and not fresh.get("tailwind", False)),
        float(sum(1 for s in screens_up if fresh.get(s, False))),
        float(sum(1 for s in screens_up if not fresh.get(s, False))),
        float(sum(1 for h in HAZARDS if h in conditions)),
    ]


def _open(path: Path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else open(path)


def _replay_ids(path: Path) -> set[str]:
    with _open(path) as fh:
        return {json.loads(line)["replay_id"] for line in fh}


def load(path: Path, *, keep: set[str] | None = None) -> dict[str, np.ndarray]:
    by_view: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    with _open(path) as fh:
        for line in fh:
            record = json.loads(line)
            if record["decision_kind"] != "turn" or record["outcome"] not in ("win", "loss"):
                continue
            if keep is not None and record["replay_id"] not in keep:
                continue
            by_view[(record["replay_id"], record["player"])].append(record)

    rows, lagged, has_lag, won, games, total_alive = [], [], [], [], [], []
    for (replay_id, _player), records in by_view.items():
        records.sort(key=lambda r: r["turn"])
        previous_conditions = {"our": set(), "opp": set()}
        previous_row = None
        for record in records:
            state = record["state"]
            feats = {}
            for side_key in ("our", "opp"):
                conditions = set(state[side_key].get("side_conditions") or [])
                fresh = {c: c not in previous_conditions[side_key] for c in conditions}
                feats[side_key] = _side_features(state[side_key], fresh)
                previous_conditions[side_key] = conditions
            row = [a - b for a, b in zip(feats["our"], feats["opp"])]
            rows.append(row)
            lagged.append(previous_row if previous_row is not None else [0.0] * len(row))
            has_lag.append(previous_row is not None)
            previous_row = row
            won.append(float(record["won"]))
            games.append(replay_id)
            total_alive.append(feats["our"][0] + feats["opp"][0])
    return {
        "X": np.asarray(rows),
        "L": np.asarray(lagged),
        "has_lag": np.asarray(has_lag),
        "y": np.asarray(won),
        "games": np.asarray(games),
        "total_alive": np.asarray(total_alive),
    }


def logistic(X: np.ndarray, y: np.ndarray, iters: int = 25) -> np.ndarray:
    """Newton/IRLS logistic regression with an intercept (last coefficient)."""

    Xb = np.c_[X, np.ones(len(X))]
    w = np.zeros(Xb.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xb @ w, -30, 30)))
        H = Xb.T @ (Xb * (p * (1 - p))[:, None]) + 1e-6 * np.eye(len(w))
        step = np.linalg.solve(H, Xb.T @ (y - p))
        w += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w


def log_loss(score: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(1.0 / (1.0 + np.exp(-score)), 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_block(data, mask, *, lagged: bool, boots: int, seed: int) -> dict:
    n = len(FEATURES)
    X = data["X"][mask]
    if lagged:
        X = np.c_[X, data["L"][mask]]
    y = data["y"][mask]
    w = logistic(X, y)
    hp = w[FEATURES.index("hp")]
    point = {name: float(w[i] / hp * 100) for i, name in enumerate(FEATURES)}

    by_game: dict[str, list[int]] = collections.defaultdict(list)
    for row, game in enumerate(data["games"][mask]):
        by_game[game].append(row)
    keys = list(by_game)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(boots):
        idx = np.concatenate([by_game[keys[j]] for j in rng.integers(0, len(keys), len(keys))])
        wb = logistic(X[idx], y[idx], iters=12)
        draws.append(wb[:n] / wb[FEATURES.index("hp")] * 100)
    draws = np.asarray(draws)
    interval = {
        name: [float(np.percentile(draws[:, i], 2.5)), float(np.percentile(draws[:, i], 97.5))]
        for i, name in enumerate(FEATURES)
    }
    return {"decisions": int(mask.sum()), "games": len(keys), "weights": point, "ci95": interval}


def holdout_comparison(data, seed: int) -> dict:
    """Out-of-sample log loss: hand scorecard (one fitted scale) vs fitted scorecard."""

    games = np.unique(data["games"])
    rng = np.random.default_rng(seed)
    test_games = set(rng.choice(games, size=len(games) // 5, replace=False))
    test = np.array([g in test_games for g in data["games"]])
    X, y = data["X"], data["y"]
    hand = np.array([HAND[name] for name in FEATURES])
    hand_score = X @ hand
    scale = logistic(hand_score[~test, None], y[~test])
    fitted = logistic(X[~test], y[~test])
    base = float(y[~test].mean())
    return {
        "test_games": len(test_games),
        "log_loss_constant": log_loss(np.full(test.sum(), np.log(base / (1 - base))), y[test]),
        "log_loss_hand": log_loss(hand_score[test] * scale[0] + scale[1], y[test]),
        "log_loss_fitted": log_loss(np.c_[X[test], np.ones(test.sum())] @ fitted, y[test]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--boots", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument(
        "--same-replays-as",
        type=Path,
        help="only use replays that also appear in this decisions file (like-for-like refits)",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    keep = _replay_ids(args.same_replays_as) if args.same_replays_as else None
    data = load(args.data, keep=keep)
    everyone = np.ones(len(data["y"]), bool)
    blocks = {
        "all": fit_block(data, everyone, lagged=False, boots=args.boots, seed=args.seed),
        "all_lagged": fit_block(
            data, data["has_lag"], lagged=True, boots=args.boots, seed=args.seed
        ),
        "early": fit_block(
            data, data["total_alive"] >= 7, lagged=False, boots=args.boots, seed=args.seed
        ),
        "late": fit_block(
            data, data["total_alive"] <= 4, lagged=False, boots=args.boots, seed=args.seed
        ),
    }
    report = {"features": list(FEATURES), "hand": HAND, "fits": blocks}
    report["holdout"] = holdout_comparison(data, args.seed)

    print("units: 100 = one Pokemon's full HP; our-minus-opponent board, start of turn")
    header = f"{'feature':16s} {'hand':>7s}"
    for name, block in blocks.items():
        header += f" {name:>22s}"
    print(header)
    for feature in FEATURES:
        line = f"{feature:16s} {HAND[feature]:7.1f}"
        for block in blocks.values():
            lo, hi = block["ci95"][feature]
            line += f" {block['weights'][feature]:7.1f} [{lo:5.0f},{hi:5.0f}]"
        print(line)
    for name, block in blocks.items():
        print(f"  {name}: {block['decisions']} decisions, {block['games']} games")
    print("holdout log loss:", json.dumps(report["holdout"], indent=None))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
