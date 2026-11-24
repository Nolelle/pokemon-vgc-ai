#!/usr/bin/env python
"""Fast, regulation-indifferent Rust meta-gauntlet & search A/B harness.

Two modes:
1. **Team Gauntlet** (`--candidate-teams ...`): grades one or more candidate 6-Pokémon
   teams against a structured panel of meta gatekeeper archetypes (Sun/Rain/Screens,
   Trick Room, Psyspam, Balance/Pivot, Perish/Control, plus unseen holdout teams),
   reporting per-archetype win rates, preview bring/lead choices, and turn latency.
2. **Paired Search A/B** (`--ab-holdout`): plays same-seed, side-swapped games on unseen
   holdout teams comparing upgraded `rust-search` (Preview Search + Turn-2 2x2 check +
   noise-pruned root choice) against the legacy baseline (raw preview + 1x1 rollout).

Usage:
    # Grade candidate teams against the gatekeeper gauntlet
    .venv/bin/python offline/evaluate_team_gauntlet.py \
      --checkpoint runs/rust_v3b_m192_l6_novmeta/policy_4000024.pt \
      --candidate-teams teams/mc_ladder_01.packed.txt teams/sand_r1831.packed.txt \
      --games-per-archetype 6

    # Paired A/B of upgraded Preview+Turn-2 search vs legacy v3b search on unseen teams
    .venv/bin/python offline/evaluate_team_gauntlet.py \
      --checkpoint runs/rust_v3b_m192_l6_novmeta/policy_4000024.pt \
      --ab-holdout --ab-pairs 20
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pikapro

from vgc.battle_memory import BattleMemory
from vgc.config import REPO_ROOT
from vgc.evaluation import clustered_interval
from vgc.models import PolicyConfig
from vgc.rl.env import DirectBattle
from vgc.rl.replica_engine import ReplicaRustWorker
from vgc.rl.rust_policy import RustFactoredPolicy, load_rust_policy
from vgc.rl.rust_search import (
    RustExactMirror,
    search_public_rust,
    search_teampreview_rust,
    select_search_action_index,
)

# Exact Reg M-C ladder gatekeeper archetypes (zero private leakage; public archetypes).
GATEKEEPER_TEAMS: dict[str, str] = {
    "sun_rain_screens": (
        "Charizard||CharizarditeY|Blaze|HeatWave,WeatherBall,SolarBeam,Protect|Timid|2,,,32,,32||||50|]"
        "Pelipper||FocusSash|Drizzle|WeatherBall,Hurricane,WideGuard,Protect|Modest|2,,,32,,32||||50|]"
        "Archaludon||Leftovers|Stamina|ElectroShot,DragonPulse,BodyPress,Protect|Modest|20,,12,32,,2||||50|]"
        "Venusaur||LifeOrb|Chlorophyll|SludgeBomb,GigaDrain,SleepPowder,Protect|Modest|2,,,32,,32||||50|]"
        "Garchomp||ClearAmulet|RoughSkin|StompingTantrum,DragonClaw,RockSlide,Protect|Jolly|2,32,,,,32||||50|]"
        "Grimmsnarl||LightClay|Prankster|Reflect,LightScreen,PartingShot,SpiritBreak|Careful|32,,16,,18,||||50|"
    ),
    "perish_control": (
        "Gengar||Gengarite|CursedBody|PerishSong,Protect,ShadowBall,IcyWind|Timid|2,,,32,,32||||50|]"
        "Incineroar||SitrusBerry|Intimidate|FakeOut,PartingShot,FlareBlitz,Protect|Careful|32,2,10,,20,2||||50|]"
        "Rillaboom||MiracleSeed|GrassySurge|FakeOut,GrassyGlide,WoodHammer,Uturn|Adamant|32,32,2,,,||||50|]"
        "Kingambit||BlackGlasses|Defiant|SuckerPunch,KowtowCleave,IronHead,Protect|Adamant|32,32,2,,,||||50|]"
        "Politoed||DampRock|Drizzle|PerishSong,Protect,Scald,HelpingHand|Calm|32,,16,,18,||||50|]"
        "Kommo-o||ThroatSpray|Soundproof|ClangorousSoul,ClangingScales,Flamethrower,Protect|Modest|2,,,32,,32||||50|"
    ),
    "trick_room_farig": (
        "Farigiraf||SitrusBerry|ArmorTail|TrickRoom,HyperVoice,Psychic,Protect|Quiet|32,,16,18,,||||50|]"
        "Torkoal||Charcoal|Drought|Eruption,HeatWave,EarthPower,Protect|Quiet|32,,,32,2,||||50|]"
        "Camerupt||Cameruptite|SolidRock|HeatWave,EarthPower,RockSlide,Protect|Quiet|32,,,32,2,||||50|]"
        "Incineroar||SafetyGoggles|Intimidate|FakeOut,PartingShot,FlareBlitz,KnockOff|Careful|32,2,16,,16,||||50|]"
        "Sylveon||ThroatSpray|Pixilate|HyperVoice,Moonblast,QuickAttack,Protect|Modest|32,,,32,2,||||50|]"
        "Kingambit||LifeOrb|Defiant|SuckerPunch,KowtowCleave,IronHead,Protect|Brave|32,32,2,,,||||50|"
    ),
    "psyspam_balance": (
        "Indeedee-F||RockyHelmet|PsychicSurge|FollowMe,TrickRoom,HelpingHand,Psychic|Calm|32,,32,,2,||||50|]"
        "Gardevoir||Gardevoirite|Telepathy|ExpandingForce,HyperVoice,Moonblast,Protect|Modest|2,,,32,,32||||50|]"
        "Sneasler||PsychicSeed|Unburden|CloseCombat,DireClaw,FakeOut,Protect|Jolly|2,32,,,,32||||50|]"
        "Incineroar||SitrusBerry|Intimidate|FakeOut,PartingShot,FlareBlitz,KnockOff|Careful|32,2,10,,20,2||||50|]"
        "Gholdengo||LifeOrb|GoodasGold|MakeItRain,ShadowBall,NastyPlot,Protect|Modest|2,,,32,,32||||50|]"
        "Whimsicott||FocusSash|Prankster|Tailwind,Moonblast,Encore,Protect|Timid|2,,,32,,32||||50|"
    ),
}


@dataclass(frozen=True)
class PlayerSpec:
    name: str
    use_preview_search: bool = True
    worlds: int = 4
    actions: int = 6
    seeds: int = 2
    iterations: int = 800
    rollout_turns: int = 1
    second_turn_branch: int = 2
    prune_root_noise: bool = True


def _play_direct_game(
    worker: ReplicaRustWorker,
    game_id: str,
    p1_team: str,
    p2_team: str,
    p1_spec: PlayerSpec,
    p2_spec: PlayerSpec,
    policy: RustFactoredPolicy,
    space: pikapro.ActionSpace,
    battle_seed: list[int],
    rng_seed: int,
) -> dict[str, Any]:
    rng = {"p1": np.random.default_rng(rng_seed), "p2": np.random.default_rng(rng_seed + 10007)}
    db = DirectBattle.start(worker, game_id, p1_team, p2_team, seed=battle_seed)
    mem = {"p1": BattleMemory(game_id), "p2": BattleMemory(game_id)}
    mem["p1"].our_role = "p1"
    mem["p2"].our_role = "p2"
    mirrors = {
        "p1": RustExactMirror(p1_team, PolicyConfig(exact_search_total_hypotheses=p1_spec.worlds)),
        "p2": RustExactMirror(p2_team, PolicyConfig(exact_search_total_hypotheses=p2_spec.worlds)),
    }
    specs = {"p1": p1_spec, "p2": p2_spec}
    teams = {"p1": p1_team, "p2": p2_team}

    def _sync() -> None:
        for s in ("p1", "p2"):
            lines = db.last_lines.get(s) or []
            mem[s].observe_protocol([line.split("|") for line in lines if line])
            setattr(db.battles[s], "_vgc_battle_memory", mem[s])

    _sync()
    rust_b = worker._games[game_id].battle

    # Team Preview
    pv_mask = np.ones(space.preview_size, dtype=bool)
    pv_cmds: dict[str, str] = {}
    pv_lat: dict[str, float] = {}
    orders_table = policy.preview_orders.cpu().numpy()
    for s_idx, s_name in enumerate(("p1", "p2")):
        spec = specs[s_name]
        obs = rust_b.encode_rich(s_idx)
        t0 = time.monotonic()
        if spec.use_preview_search:
            cmd, _ = search_teampreview_rust(
                db.battles[s_name],
                teams[s_name],
                policy,
                obs,
                pv_mask,
                candidates=8,
                worlds=6,
                seeds=3,
                seed=int(rng[s_name].integers(0, 2**31)),
            )
            pv_cmds[s_name] = cmd.lstrip("/")
        else:
            pv_idx = policy.choose_preview_index(obs, pv_mask)
            row = orders_table[pv_idx]
            pv_cmds[s_name] = "team " + "".join(str(int(k) + 1) for k in row)
        pv_lat[s_name] = time.monotonic() - t0

    db.step(pv_cmds)
    _sync()

    turn_latencies: dict[str, list[float]] = {"p1": [], "p2": []}
    for _ in range(40):
        if rust_b.over or db.ended:
            break
        choices: dict[str, str] = {}
        for s_idx, s_name in enumerate(("p1", "p2")):
            rk = rust_b.request_kind(s_idx)
            if rk == "wait":
                continue
            spec = specs[s_name]
            t0 = time.monotonic()
            res, _ = search_public_rust(
                db.battles[s_name],
                teams[s_name],
                policy,
                worlds=spec.worlds,
                actions=spec.actions,
                seeds=spec.seeds,
                iterations=spec.iterations,
                seed=int(rng[s_name].integers(0, 2**31)),
                mirror=mirrors[s_name],
                memory=mem[s_name],
                rollout_turns=spec.rollout_turns,
                second_turn_branch=spec.second_turn_branch,
            )
            turn_latencies[s_name].append(time.monotonic() - t0)
            if spec.prune_root_noise:
                sel = select_search_action_index(res, rng[s_name])
            else:
                sel = int(rng[s_name].choice(len(res.actions), p=res.probabilities))
            wire = res.actions[sel]
            legal_set = {str(c) for c in rust_b.legal_choices(s_idx)}
            if wire not in legal_set:
                wire = next((a for a in res.actions if a in legal_set), next(iter(legal_set)))
            choices[s_name] = wire
        if not choices:
            break
        db.step(choices)
        _sync()

    winner = rust_b.winner
    turns = int(rust_b.turn)
    db.close()
    return {
        "winner": winner,  # 0=p1, 1=p2, None=draw
        "turns": turns,
        "p1_preview": pv_cmds["p1"],
        "p2_preview": pv_cmds["p2"],
        "p1_pv_s": pv_lat["p1"],
        "p2_pv_s": pv_lat["p2"],
        "p1_turn_s": turn_latencies["p1"],
        "p2_turn_s": turn_latencies["p2"],
    }


def _load_holdout_teams() -> list[tuple[str, str]]:
    manifest = REPO_ROOT / "data/selfplay/mc_sheet_pool_v2/holdout_manifest.json"
    rows = json.loads(manifest.read_text())
    return [(r["file"], (manifest.parent / r["file"]).read_text().strip()) for r in rows]


def run_ab_holdout(policy: RustFactoredPolicy, pairs: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    holdout = _load_holdout_teams()
    space = pikapro.ActionSpace("championsvgc")

    upgraded = PlayerSpec(
        "upgraded",
        use_preview_search=True,
        worlds=6,
        actions=6,
        seeds=2,
        iterations=1000,
        rollout_turns=1,
        second_turn_branch=1,
        prune_root_noise=True,
    )
    baseline = PlayerSpec(
        "baseline",
        use_preview_search=False,
        worlds=4,
        actions=6,
        seeds=2,
        iterations=400,
        rollout_turns=1,
        second_turn_branch=1,
        prune_root_noise=False,
    )

    per_cluster: dict[str, list[float]] = {}
    up_lat: list[float] = []
    up_pv: list[float] = []
    wins = 0.0
    total = 0

    with ReplicaRustWorker() as worker:
        for pi in range(pairs):
            (f1, t1), (f2, t2) = rng.sample(holdout, 2)
            bseed = [rng.randint(1, 60000) for _ in range(4)]
            # Side-swapped pair with identical teams & battle seed!
            g1 = _play_direct_game(worker, f"ab-{pi}-a", t1, t2, upgraded, baseline, policy, space, bseed, seed + pi * 2)
            s1 = 1.0 if g1["winner"] == 0 else (0.5 if g1["winner"] is None else 0.0)
            g2 = _play_direct_game(worker, f"ab-{pi}-b", t1, t2, baseline, upgraded, policy, space, bseed, seed + pi * 2 + 1)
            s2 = 1.0 if g2["winner"] == 1 else (0.5 if g2["winner"] is None else 0.0)
            per_cluster.setdefault(f"{f1}_vs_{f2}", []).extend([s1, s2])
            wins += s1 + s2
            total += 2
            up_lat.extend(g1["p1_turn_s"] + g2["p2_turn_s"])
            up_pv.extend([g1["p1_pv_s"], g2["p2_pv_s"]])

    ci_low, ci_high = clustered_interval(
        [(int(round(sum(scores))), len(scores)) for scores in per_cluster.values()]
    )
    return {
        "mode": "ab_holdout",
        "games": total,
        "upgraded_points": wins,
        "win_rate": round(wins / max(1, total), 4),
        "clustered_ci95": [round(ci_low, 4), round(ci_high, 4)],
        "upgraded_preview_p50_s": round(statistics.median(up_pv), 3) if up_pv else 0.0,
        "upgraded_turn_p50_s": round(statistics.median(up_lat), 3) if up_lat else 0.0,
        "upgraded_turn_p95_s": round(float(np.percentile(up_lat, 95)), 3) if up_lat else 0.0,
    }


def run_team_gauntlet(
    policy: RustFactoredPolicy,
    candidate_paths: list[Path],
    games_per_archetype: int,
    extra_holdout: int,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    space = pikapro.ActionSpace("championsvgc")
    holdout = _load_holdout_teams()
    opponents: list[tuple[str, str]] = list(GATEKEEPER_TEAMS.items())
    for idx, (fname, tstr) in enumerate(rng.sample(holdout, min(extra_holdout, len(holdout)))):
        opponents.append((f"holdout_{idx}_{Path(fname).stem}", tstr))

    our_spec = PlayerSpec(
        "candidate",
        use_preview_search=True,
        worlds=4,
        actions=6,
        seeds=2,
        iterations=1000,
        rollout_turns=1,
        second_turn_branch=1,
        prune_root_noise=True,
    )
    opp_spec = PlayerSpec(
        "opponent",
        use_preview_search=True,
        worlds=4,
        actions=6,
        seeds=2,
        iterations=1000,
        rollout_turns=1,
        second_turn_branch=1,
        prune_root_noise=True,
    )

    # Pre-sample battle seeds so every candidate team is tested on identical RNG per matchup.
    matchup_seeds: dict[tuple[str, int], list[int]] = {
        (arch_name, gi): [rng.randint(1, 60000) for _ in range(4)]
        for arch_name, _ in opponents
        for gi in range(games_per_archetype)
    }

    results: dict[str, Any] = {}
    with ReplicaRustWorker() as worker:
        for path in candidate_paths:
            cand_team = path.read_text().strip()
            by_arch: dict[str, dict[str, Any]] = {}
            all_clusters: list[tuple[int, int]] = []
            tot_pts = 0.0
            tot_games = 0
            for arch_name, opp_team in opponents:
                pts = 0.0
                n_g = 0
                brings: Counter[str] = Counter()
                arch_scores: list[float] = []
                # Alternate seats per game for exact fairness!
                for gi in range(games_per_archetype):
                    bseed = matchup_seeds[(arch_name, gi)]
                    if gi % 2 == 0:
                        g = _play_direct_game(
                            worker, f"g-{path.stem}-{arch_name}-{gi}", cand_team, opp_team, our_spec, opp_spec, policy, space, bseed, seed + gi * 31
                        )
                        sc = 1.0 if g["winner"] == 0 else (0.5 if g["winner"] is None else 0.0)
                        brings[g["p1_preview"]] += 1
                    else:
                        g = _play_direct_game(
                            worker, f"g-{path.stem}-{arch_name}-{gi}", opp_team, cand_team, opp_spec, our_spec, policy, space, bseed, seed + gi * 31
                        )
                        sc = 1.0 if g["winner"] == 1 else (0.5 if g["winner"] is None else 0.0)
                        brings[g["p2_preview"]] += 1
                    pts += sc
                    n_g += 1
                    arch_scores.append(sc)
                tot_pts += pts
                tot_games += n_g
                all_clusters.append((int(round(sum(arch_scores))), len(arch_scores)))
                by_arch[arch_name] = {
                    "score": f"{pts:.1f}/{n_g}",
                    "win_rate": round(pts / max(1, n_g), 3),
                    "top_brings": brings.most_common(3),
                }
            ci_low, ci_high = clustered_interval(all_clusters)
            results[path.name] = {
                "overall_score": f"{tot_pts:.1f}/{tot_games}",
                "overall_win_rate": round(tot_pts / max(1, tot_games), 4),
                "clustered_ci95": [round(ci_low, 4), round(ci_high, 4)],
                "by_archetype": by_arch,
            }
    return {"mode": "team_gauntlet", "teams": results}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20261123)
    parser.add_argument("--ab-holdout", action="store_true")
    parser.add_argument("--ab-pairs", type=int, default=16)
    parser.add_argument("--candidate-teams", type=Path, nargs="*", default=[])
    parser.add_argument("--games-per-archetype", type=int, default=4)
    parser.add_argument("--extra-holdout", type=int, default=4)
    args = parser.parse_args()

    import torch

    dev = (
        "mps"
        if args.device == "auto" and torch.backends.mps.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    policy = load_rust_policy(args.checkpoint, device=dev)

    if args.ab_holdout:
        out = run_ab_holdout(policy, args.ab_pairs, args.seed)
    elif args.candidate_teams:
        out = run_team_gauntlet(
            policy, args.candidate_teams, args.games_per_archetype, args.extra_holdout, args.seed
        )
    else:
        raise SystemExit("Specify --ab-holdout or --candidate-teams")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
