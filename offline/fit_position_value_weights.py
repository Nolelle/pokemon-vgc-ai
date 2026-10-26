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
import hashlib
import html
import json
import re
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


def _id(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", html.unescape(value).lower())


def select_replays(ids: set[str], raw_dir: Path, *, min_rating: int | None,
                   exclude_accounts: set[str], team_cap: int | None, seed: int,
                   individual_team_cap: int | None = None):
    """Filter whole games using only the raw log's rating changes and preview."""
    counts = collections.Counter()
    groups: dict[tuple, list[str]] = collections.defaultdict(list)
    selected = set()
    teams_by_game = {}
    raw_digest = hashlib.sha256()
    counts["input_games"] = len(ids)
    for replay_id in sorted(ids):
        path = raw_dir / f"{replay_id}.json"
        if not path.is_file():
            raw_digest.update(f"{replay_id}\tMISSING\n".encode())
            counts["missing_raw"] += 1
            continue
        content = path.read_bytes()
        raw_digest.update(f"{replay_id}\t{hashlib.sha256(content).hexdigest()}\n".encode())
        counts["raw_files_scanned"] += 1
        replay = json.loads(content)
        log = replay.get("log") or ""
        names = {side: _id(name) for side, name in re.findall(
            r"^\|player\|(p[12])\|([^|\n]+)", log, re.M)}
        if len(names) != 2 or not all(names.values()) or len(set(names.values())) != 2:
            counts["missing_or_ambiguous_players"] += 1
            continue
        if exclude_accounts.intersection(names.values()):
            counts["excluded_accounts"] += 1
            continue
        pre = {_id(name): int(before) for name, before in re.findall(
            r"^\|raw\|(.+?)'s rating: (\d+) &rarr; <strong>\d+</strong>", log, re.M)}
        missing_rating = any(name not in pre for name in names.values())
        if missing_rating:
            counts["missing_pre_rating"] += 1
        if min_rating is not None and missing_rating:
            counts["excluded_missing_pre_rating"] += 1
            continue
        if min_rating is not None and any(pre[name] < min_rating for name in names.values()):
            counts["excluded_below_rating"] += 1
            continue
        if team_cap is None and individual_team_cap is None:
            selected.add(replay_id)
            continue
        preview = {side: [] for side in ("p1", "p2")}
        for side, species in re.findall(r"^\|poke\|(p[12])\|([^,|\n]+)", log, re.M):
            preview[side].append(_id(species))
        if any(len(team) != 6 or not all(team) for team in preview.values()):
            counts["excluded_missing_preview"] += 1
            continue
        pair = tuple(sorted(tuple(sorted(team)) for team in preview.values()))
        groups[pair].append(replay_id)
        teams_by_game[replay_id] = pair
    rng = np.random.default_rng(seed)
    for pair in sorted(groups):
        choices = groups[pair]
        if team_cap is not None and len(choices) > team_cap:
            picked = rng.choice(len(choices), team_cap, replace=False)
            selected.update(choices[i] for i in picked)
            counts["excluded_team_cap"] += len(choices) - team_cap
        else:
            selected.update(choices)
    if individual_team_cap is not None:
        team_counts = collections.Counter()
        choices = sorted(selected)
        selected = set()
        for index in rng.permutation(len(choices)):
            replay_id = choices[index]
            # A mirror match is one game for that template, not two games.
            teams = set(teams_by_game[replay_id])
            if any(team_counts[team] >= individual_team_cap for team in teams):
                counts["excluded_individual_team_cap"] += 1
                continue
            selected.add(replay_id)
            team_counts.update(teams)
    counts["retained_games"] = len(selected)
    metadata = dict(counts)
    metadata["raw_inputs_sha256"] = raw_digest.hexdigest()
    metadata["raw_hash_format"] = "sorted replay ID, tab, file SHA-256 or MISSING, newline"
    metadata["counting"] = "exclusion counts are sequential; missing_pre_rating is an overlapping observation"
    return selected, metadata


def load(path: Path, *, keep: set[str] | None = None) -> dict[str, np.ndarray]:
    # Keep only compact features and condition sets; never retain full parsed states.
    by_view = collections.defaultdict(list)
    with _open(path) as fh:
        for line in fh:
            record = json.loads(line)
            if record["decision_kind"] != "turn" or record["outcome"] not in ("win", "loss"):
                continue
            if keep is not None and record["replay_id"] not in keep:
                continue
            sides = []
            for key in ("our", "opp"):
                side = record["state"][key]
                conditions = set(side.get("side_conditions") or [])
                sides.append((_side_features(side, {c: True for c in conditions}), conditions))
            by_view[(record["replay_id"], record["player"])].append(
                (record["turn"], float(record["won"]), sides))

    rows, lagged, has_lag, won, games, total_alive = [], [], [], [], [], []
    for (replay_id, _player), records in by_view.items():
        records.sort(key=lambda r: r[0])
        previous_conditions = [set(), set()]
        previous_row = None
        for _turn, outcome, sides in records:
            feats = []
            for i, (features, conditions) in enumerate(sides):
                features = features.copy()
                for condition in ("tailwind", *SCREENS):
                    if condition in conditions and condition in previous_conditions[i]:
                        fresh, old = (10, 11) if condition == "tailwind" else (12, 13)
                        features[fresh] -= 1
                        features[old] += 1
                feats.append(features)
                previous_conditions[i] = conditions
            row = [a - b for a, b in zip(*feats)]
            rows.append(row)
            lagged.append(previous_row if previous_row is not None else [0.0] * len(row))
            has_lag.append(previous_row is not None)
            previous_row = row
            won.append(outcome)
            games.append(replay_id)
            total_alive.append(feats[0][0] + feats[1][0])
    return {"X": np.asarray(rows), "L": np.asarray(lagged), "has_lag": np.asarray(has_lag),
            "y": np.asarray(won), "games": np.asarray(games),
            "total_alive": np.asarray(total_alive)}


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
    parser.add_argument("--raw-replays", type=Path, help="raw replay JSON directory for filters")
    parser.add_argument("--min-pre-rating", type=int, help="require BOTH pre-game ratings >= this")
    parser.add_argument("--exclude-account", action="append", default=[],
                        help="exclude a normalized account name; repeat for multiple accounts")
    parser.add_argument("--max-games-per-team-pair", type=int,
                        help="seeded cap per unordered pair of six-species preview teams")
    parser.add_argument("--max-games-per-team", type=int,
                        help="seeded cap per individual six-species preview team")
    parser.add_argument("--boots", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument(
        "--same-replays-as",
        type=Path,
        help="only use replays that also appear in this decisions file (like-for-like refits)",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.boots < 1:
        parser.error("--boots must be at least 1")
    if args.max_games_per_team_pair is not None and args.max_games_per_team_pair < 1:
        parser.error("--max-games-per-team-pair must be at least 1")
    if args.max_games_per_team is not None and args.max_games_per_team < 1:
        parser.error("--max-games-per-team must be at least 1")
    excluded = {_id(name) for name in args.exclude_account}
    if "" in excluded:
        parser.error("account names must contain letters or digits")
    filtering = (args.min_pre_rating is not None or excluded or
                 args.max_games_per_team_pair is not None or args.max_games_per_team is not None)
    if filtering and args.raw_replays is None:
        parser.error("--raw-replays is required when using replay filters")
    keep = _replay_ids(args.same_replays_as) if args.same_replays_as else None
    counts = {}
    if args.raw_replays is not None:
        ids = _replay_ids(args.data)
        if keep is not None:
            ids &= keep
        keep, counts = select_replays(ids, args.raw_replays, min_rating=args.min_pre_rating,
                                      exclude_accounts=excluded,
                                      team_cap=args.max_games_per_team_pair, seed=args.seed,
                                      individual_team_cap=args.max_games_per_team)
    selected_filter_ids = keep
    data = load(args.data, keep=keep)
    if len(np.unique(data["games"])) < 5:
        parser.error("at least five retained games with outcomes are required")
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
    def file_digest(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    selected_ids = sorted(set(data["games"]))
    selected_filter_hash = (hashlib.sha256("".join(f"{game}\n" for game in sorted(selected_filter_ids)).encode()).hexdigest()
                            if selected_filter_ids is not None else None)
    selected_hash = hashlib.sha256("".join(f"{game}\n" for game in selected_ids).encode()).hexdigest()
    schemas = set()
    with _open(args.data) as fh:
        for line in fh:
            record = json.loads(line)
            schemas.add(str(record.get("schema", "missing")))
    report = {"features": list(FEATURES), "hand": HAND, "fits": blocks,
              "dataset": {"path": str(args.data.resolve()), "sha256": file_digest(args.data),
                          "schema_versions": sorted(schemas)},
              "selection": {"raw_replays": str(args.raw_replays.resolve()) if args.raw_replays else None,
                            "same_replays_as": str(args.same_replays_as.resolve()) if args.same_replays_as else None,
                            "same_replays_as_sha256": file_digest(args.same_replays_as) if args.same_replays_as else None,
                            "selected_replay_ids_sha256": selected_filter_hash,
                            "fitted_replay_ids_sha256": selected_hash,
                            "fitted_replay_ids_hash_format": "sorted replay IDs, each followed by newline",
                            "min_pre_rating_both": args.min_pre_rating,
                            "exclude_accounts": sorted(excluded),
                            "max_games_per_team_pair": args.max_games_per_team_pair,
                            "max_games_per_team": args.max_games_per_team,
                            "team_cap_order": "pair cap then individual cap; seeded, outcome-independent",
                            "seed": args.seed, "boots": args.boots, "counts": counts,
                            "fitted_games": len(np.unique(data["games"]))}}
    report["holdout"] = holdout_comparison(data, args.seed)
    report["holdout"]["split"] = "random games; accounts and preview teams may repeat across train and test"

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
