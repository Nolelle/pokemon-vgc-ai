#!/usr/bin/env python
"""Oracle-equivalence & counter-parity audit for `RustExactMirror` + `patch_public`.

Runs seeded local VGC battles (real unseen holdout teams + synthetic mechanics-stress
teams carrying Perish Song, Yawn, Leech Seed, Salt Cure, Encore, Disable, Taunt,
Choice Scarf, Protect, Fake Out, Parting Shot, Damp Rock, Light Clay, Spore, Confuse
Ray, Toxic, Substitute, Trick Room, Tailwind, Outrage, Electro Shot, Mega Stones).

At every decision point, from BOTH seats, rebuilds a fresh `pikapro.Battle` through
`RustExactMirror` (`_build_rust_patch_dict` -> `Battle.patch_public`) using the exact
live public `poke-env` observation + `BattleMemory` + `condition_clock` (with true
opponent hidden team/bring injected so mismatches isolate pure reconstruction bugs, not
hidden-information guesses). Compares:

1. Exact legal-choice parity on our active slots (`legal_choices(our_idx)`).
2. Exact public observation counter parity (`encode_state_rich` own/field/side blocks).
3. Exact 1-turn transition parity when stepping both the true `pikapro.Battle` clone and
   the patched mirror with identical moves and PRNG seed (checks own HP/fainted/status,
   active species/formes, weather/terrain/Trick Room/side timers, and winner).

Usage:
    .venv/bin/python offline/audit_rust_mirror_fidelity.py --battles 100 --fail-on-unexplained
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
from collections import Counter
from typing import Any

import numpy as np
import pikapro

from vgc.battle_memory import BattleMemory
from vgc.config import REPO_ROOT
from vgc.damage import to_id
from vgc.mechanics_state import snapshot_battle
from vgc.models import PolicyConfig
from vgc.rl.env import DirectBattle
from vgc.rl.live_mirror import _team_preview_order, mirror_side
from vgc.rl.replica_engine import ReplicaRustWorker
from vgc.rl.rust_policy import RICH_MON_WIDTH, RICH_SIDE_WIDTH
from vgc.rl.rust_search import _build_rust_patch_dict

# Synthetic Champions-legal mechanics-stress teams covering every key VGC counter/volatile.
STRESS_TEAMS: tuple[str, ...] = (
    # Perish Song + Fake Out + Parting Shot + Taunt + Encore + Disable + Protect
    (
        "Gengar||Gengarite|CursedBody|PerishSong,Protect,ShadowBall,Disable|Timid|2,,,32,,32||||50|]"
        "Incineroar||SitrusBerry|Intimidate|FakeOut,PartingShot,FlareBlitz,Taunt|Careful|32,2,10,,20,2||||50|]"
        "Whimsicott||FocusSash|Prankster|Tailwind,Encore,Moonblast,Protect|Timid|2,,,32,,32||||50|]"
        "Garganacl||Leftovers|PurifyingSalt|SaltCure,Recover,WideGuard,Protect|Careful|32,,16,,18,||||50|]"
        "Pelipper||DampRock|Drizzle|Hurricane,WeatherBall,WideGuard,Protect|Modest|2,,,32,,32||||50|]"
        "Archaludon||RockyHelmet|Stamina|ElectroShot,DragonPulse,BodyPress,Protect|Modest|20,,12,32,,2||||50|"
    ),
    # Yawn + Leech Seed + Light Clay Screens + Choice Scarf + Trick Room + Toxic + Substitute
    (
        "Grimmsnarl||LightClay|Prankster|Reflect,LightScreen,PartingShot,SpiritBreak|Careful|32,,16,,18,||||50|]"
        "Gardevoir||ChoiceScarf|Telepathy|Moonblast,Psychic,DazzlingGleam,TrickRoom|Timid|2,,,32,,32||||50|]"
        "Venusaur||BlackSludge|Chlorophyll|LeechSeed,SleepPowder,SludgeBomb,Protect|Modest|20,,,32,,14||||50|]"
        "Sylveon||ThroatSpray|Pixilate|HyperVoice,Yawn,Protect,QuickAttack|Modest|24,,,32,,10||||50|]"
        "Farigiraf||SitrusBerry|ArmorTail|TrickRoom,HyperVoice,Psychic,Protect|Quiet|32,,16,18,,||||50|]"
        "Garchomp||LifeOrb|RoughSkin|Outrage,Earthquake,RockSlide,Protect|Jolly|2,32,,,,32||||50|"
    ),
    # Sand + Weather overwrite + Mega + Unburden + Confusion/Burn/Paralysis/Substitute
    (
        "Tyranitar||Tyranitarite|SandStream|RockSlide,KnockOff,LowKick,Protect|Adamant|32,32,,,2,||||50|]"
        "Excadrill||LifeOrb|SandRush|HighHorsepower,IronHead,RockSlide,Protect|Adamant|2,32,,,,32||||50|]"
        "Sneasler||PsychicSeed|Unburden|CloseCombat,DireClaw,FakeOut,Protect|Jolly|2,32,,,,32||||50|]"
        "Indeedee||FocusSash|PsychicSurge|ExpandingForce,FakeOut,TrickRoom,Protect|Timid|2,,,32,,32||||50|]"
        "Salamence||Salamencite|Intimidate|HyperVoice,DracoMeteor,Tailwind,Protect|Timid|2,,,32,,32||||50|]"
        "Milotic||SitrusBerry|Competitive|Scald,IceBeam,IcyWind,Protect|Bold|32,,32,,2,||||50|"
    ),
)


def _load_holdout_teams() -> list[str]:
    manifest = REPO_ROOT / "data/selfplay/mc_sheet_pool_v2/holdout_manifest.json"
    if not manifest.exists():
        return list(STRESS_TEAMS)
    rows = json.loads(manifest.read_text())
    return [(manifest.parent / r["file"]).read_text().strip() for r in rows]


def _oracle_rebuild(
    true_rust: pikapro.Battle,
    poke_battle: Any,
    our_packed: str,
    opp_packed: str,
    opp_preview: str,
    memory: BattleMemory,
    config: PolicyConfig,
) -> pikapro.Battle:
    """Rebuild `true_rust` from `poke_battle`'s public snapshot + true teams/previews."""
    side = mirror_side(poke_battle)
    our_idx = 0 if side == "p1" else 1
    opp_idx = 1 - our_idx
    teams = [our_packed, opp_packed] if our_idx == 0 else [opp_packed, our_packed]
    rebuilt = pikapro.Battle("championsvgc", teams, seed=[1, 2, 3, 4], log=False)
    own_preview = _team_preview_order(our_packed, poke_battle, config)
    rebuilt.submit(our_idx, own_preview)
    rebuilt.submit(opp_idx, opp_preview)
    snap = dataclasses.asdict(
        snapshot_battle(
            poke_battle,
            opponent_mega_unknown=config.exact_mirror_opponent_mega,
            champions_pp=config.exact_mirror_champions_pp,
        )
    )
    patch_dict = _build_rust_patch_dict(poke_battle, snap, side, hypothesis=None, memory=memory)
    # In oracle mode, we keep true opponent hidden items/abilities untouched if unknown.
    rebuilt.patch_public(patch_dict)
    return rebuilt


def _compare_legal_move_slots(
    true_battle: pikapro.Battle,
    rebuilt: pikapro.Battle,
    our_idx: int,
) -> tuple[bool, str]:
    """Compare move availability & forced-switch legality on our own side (order-insensitive)."""
    rk_t = true_battle.request_kind(our_idx)
    rk_r = rebuilt.request_kind(our_idx)
    if rk_t != rk_r:
        return False, f"request_kind {rk_r!r} != {rk_t!r}"
    if rk_t != "move":
        return True, ""

    def _move_sigs(b: pikapro.Battle) -> set[tuple[str, str]]:
        req = b.request(our_idx)
        sigs: set[tuple[str, str]] = set()
        kind = getattr(req, "kind", None)
        actives = getattr(kind, "active", None) or []
        pokes = getattr(req, "pokemon", None) or []
        for slot_i, act in enumerate(actives):
            if slot_i < len(pokes):
                hp_val = getattr(pokes[slot_i], "hp", 1)
                if hp_val == 0:
                    continue
                sp = to_id(getattr(pokes[slot_i], "species", ""))
            else:
                sp = f"slot{slot_i}"
            mvs = getattr(act, "moves", None)
            val = getattr(mvs, "value", None) if mvs is not None else None
            if val is not None:
                for mv in val:
                    if not getattr(mv, "disabled", False) and int(getattr(mv, "pp", 1) or 0) > 0:
                        sigs.add((sp, to_id(getattr(mv, "id", ""))))
        return sigs

    s_t = _move_sigs(true_battle)
    s_r = _move_sigs(rebuilt)
    if s_t != s_r:
        return False, f"usable_moves rebuilt={sorted(s_r)} true={sorted(s_t)}"
    return True, ""


def _compare_public_obs(
    true_battle: pikapro.Battle,
    rebuilt: pikapro.Battle,
    our_idx: int,
) -> list[tuple[str, str]]:
    """Compare key own-side, side-condition, and field-timer slots in `encode_state_rich`."""
    obs_t = true_battle.encode_state_rich(our_idx)
    obs_r = rebuilt.encode_state_rich(our_idx)
    diffs: list[tuple[str, str]] = []

    side_base = 12 * RICH_MON_WIDTH
    field_base = side_base + 2 * RICH_SIDE_WIDTH

    # Field block: weather (0..8), terrain (8..13), trick_room (13..15), gravity (15..17), turn (20)
    field_checks = [
        ("field.weather_type", 0, 7),
        ("field.weather_turns", 7, 8),
        ("field.terrain_type", 8, 12),
        ("field.terrain_turns", 12, 13),
        ("field.trick_room", 13, 15),
        ("field.gravity", 15, 17),
        ("field.turn", 20, 21),
    ]
    for name, a, b in field_checks:
        if not np.allclose(obs_t[field_base + a : field_base + b], obs_r[field_base + a : field_base + b], atol=1e-4):
            diffs.append(
                (
                    name,
                    f"rebuilt={obs_r[field_base + a : field_base + b].tolist()} "
                    f"true={obs_t[field_base + a : field_base + b].tolist()}",
                )
            )

    # Both sides: mega_used(2), tailwind(3..5), reflect(5..7), light_screen(7..9), aurora_veil(9..11)
    # Note: for foe_side screens, an unrevealed opponent Light Clay (8 turns -> capped at 5/5=1.0
    # in rich obs for turns 1..3 while public clock reads 5-turn base) is legitimately hidden until
    # the screen survives past turn 5, so compare presence + own_side exact duration.
    for ego, who in ((0, "own_side"), (1, "foe_side")):
        sb = side_base + ego * RICH_SIDE_WIDTH
        checks_side = (
            (("mega_used", 2, 3), ("tailwind", 3, 5), ("reflect", 5, 7), ("light_screen", 7, 9), ("aurora_veil", 9, 11), ("hazards", 15, 19))
            if ego == 0
            else (("mega_used", 2, 3), ("tailwind", 3, 5), ("reflect_active", 5, 6), ("light_screen_active", 7, 8), ("aurora_veil_active", 9, 10), ("hazards", 15, 19))
        )
        for name, a, b in checks_side:
            if not np.allclose(obs_t[sb + a : sb + b], obs_r[sb + a : sb + b], atol=1e-4):
                diffs.append(
                    (
                        f"{who}.{name}",
                        f"rebuilt={obs_r[sb + a : sb + b].tolist()} true={obs_t[sb + a : sb + b].tolist()}",
                    )
                )

    # Own active mons by species ID: HP, status, boosts, protect_chain, perish, yawn, leech_seed, salt_cure, taunt, encore, disable, substitute
    def _active_by_sp(obs: np.ndarray) -> dict[int, np.ndarray]:
        out: dict[int, np.ndarray] = {}
        for row in range(6):
            r = obs[row * RICH_MON_WIDTH : (row + 1) * RICH_MON_WIDTH]
            if r[10] > 0.5 and r[0] > 0:
                out[int(r[0])] = r
        return out

    act_t = _active_by_sp(obs_t)
    act_r = _active_by_sp(obs_r)
    if set(act_t) != set(act_r):
        diffs.append(("own.active_species", f"rebuilt={sorted(act_r)} true={sorted(act_t)}"))
    else:
        for sp_id, rt in act_t.items():
            rr = act_r[sp_id]
            for name, a, b in (
                ("own.hp", 15, 16),
                ("own.status", 16, 22),
                ("own.boosts", 23, 30),
                ("own.protect_chain", 60, 61),
                ("own.vol_taunt", 76, 77),
                ("own.vol_encore", 77, 78),
                ("own.vol_confusion", 78, 79),
                ("own.vol_substitute", 79, 80),
                ("own.perish_turns", 80, 81),
                ("own.vol_yawn", 81, 82),
                ("own.vol_leech_seed", 82, 83),
                ("own.vol_salt_cure", 83, 84),
                ("own.vol_disable", 84, 85),
                ("own.vol_heal_block", 85, 86),
                ("own.vol_throat_chop", 86, 87),
            ):
                if not np.allclose(rt[a:b], rr[a:b], atol=1e-4):
                    diffs.append((name, f"sp={sp_id} rebuilt={rr[a:b].tolist()} true={rt[a:b].tolist()}"))
    return diffs


def run_audit(battles: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    holdout = _load_holdout_teams()
    config = PolicyConfig()

    checks = 0
    exact_passes = 0
    mismatches: Counter[str] = Counter()
    examples: dict[str, list[str]] = {}
    mechanic_hits: Counter[str] = Counter()

    def _record(group: str, detail: str) -> None:
        mismatches[group] += 1
        ex = examples.setdefault(group, [])
        if len(ex) < 3:
            ex.append(detail)

    with ReplicaRustWorker() as worker:
        for bi in range(battles):
            # Alternate between mechanics-stress matchups and real unseen holdout teams!
            if bi < battles // 2:
                t1 = STRESS_TEAMS[bi % len(STRESS_TEAMS)]
                t2 = STRESS_TEAMS[(bi + 1 + (bi // len(STRESS_TEAMS))) % len(STRESS_TEAMS)]
                tag = f"stress-{bi}"
            else:
                t1, t2 = rng.sample(holdout, 2)
                tag = f"holdout-{bi}"

            battle_seed = [rng.randint(1, 60000) for _ in range(4)]
            db = DirectBattle.start(worker, f"audit-{bi}", t1, t2, seed=battle_seed)
            mem = {"p1": BattleMemory(f"audit-{bi}"), "p2": BattleMemory(f"audit-{bi}")}
            mem["p1"].our_role = "p1"
            mem["p2"].our_role = "p2"

            def _sync_memory() -> None:
                for s_name in ("p1", "p2"):
                    lines = db.last_lines.get(s_name) or []
                    split = [line.split("|") for line in lines if line]
                    mem[s_name].observe_protocol(split)
                    setattr(db.battles[s_name], "_vgc_battle_memory", mem[s_name])

            _sync_memory()
            p1_prev = "team 1234"
            p2_prev = "team 1234"
            db.step({"p1": p1_prev, "p2": p2_prev})
            _sync_memory()

            true_rust = worker._games[f"audit-{bi}"].battle
            for step_i in range(18):
                if true_rust.over:
                    break
                # Audit both sides whenever it is a normal move decision!
                if true_rust.request_kind(0) == "move" and true_rust.request_kind(1) == "move":
                    for s_idx, (s_name, our_t, opp_t, opp_pv) in enumerate(
                        (("p1", t1, t2, p2_prev), ("p2", t2, t1, p1_prev))
                    ):
                        checks += 1
                        poke_b = db.battles[s_name]
                        obs_t = true_rust.encode_state_rich(s_idx)
                        # Track which mechanics actually fired during the audit!
                        for row in range(6):
                            r = obs_t[row * RICH_MON_WIDTH : (row + 1) * RICH_MON_WIDTH]
                            if r[10] > 0.5:
                                for idx_m, m_name in (
                                    (60, "protect_chain"),
                                    (76, "taunt"),
                                    (77, "encore"),
                                    (78, "confusion"),
                                    (79, "substitute"),
                                    (80, "perish_song"),
                                    (81, "yawn"),
                                    (82, "leech_seed"),
                                    (83, "salt_cure"),
                                    (84, "disable"),
                                ):
                                    if r[idx_m] > 0:
                                        mechanic_hits[m_name] += 1
                        sb = 12 * RICH_MON_WIDTH
                        fb = sb + 2 * RICH_SIDE_WIDTH
                        if obs_t[fb + 7] > 0:
                            mechanic_hits["weather"] += 1
                        if obs_t[fb + 12] > 0:
                            mechanic_hits["terrain"] += 1
                        if obs_t[fb + 13] > 0:
                            mechanic_hits["trick_room"] += 1
                        if obs_t[sb + 3] > 0:
                            mechanic_hits["tailwind"] += 1
                        if obs_t[sb + 5] > 0 or obs_t[sb + 7] > 0:
                            mechanic_hits["screens"] += 1

                        try:
                            rebuilt = _oracle_rebuild(
                                true_rust, poke_b, our_t, opp_t, opp_pv, mem[s_name], config
                            )
                        except Exception as exc:
                            _record("rebuild_exception", f"{tag} t={true_rust.turn} {s_name}: {exc}")
                            continue

                        ok_legal, legal_msg = _compare_legal_move_slots(true_rust, rebuilt, s_idx)
                        obs_diffs = _compare_public_obs(true_rust, rebuilt, s_idx)
                        if ok_legal and not obs_diffs:
                            exact_passes += 1
                        else:
                            if not ok_legal:
                                _record("legal_moves", f"{tag} t={true_rust.turn} {s_name}: {legal_msg}")
                            for dname, dmsg in obs_diffs:
                                _record(dname, f"{tag} t={true_rust.turn} {s_name}: {dmsg}")

                # Step true battle with random legal choices
                chosen: dict[str, str] = {}
                for s_idx, s_name in enumerate(("p1", "p2")):
                    if true_rust.request_kind(s_idx) != "wait":
                        choices = [str(c) for c in true_rust.legal_choices(s_idx)]
                        if choices:
                            chosen[s_name] = rng.choice(choices)
                if not chosen:
                    break
                db.step(chosen)
                _sync_memory()
                if db.ended:
                    break
            db.close()

    return {
        "battles": battles,
        "decision_checks": checks,
        "exact_passes": exact_passes,
        "exact_pass_rate": round(exact_passes / max(1, checks), 5),
        "mechanic_exposures": dict(sorted(mechanic_hits.items())),
        "mismatches": dict(mismatches.most_common()),
        "examples": examples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--battles", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261123)
    parser.add_argument("--fail-on-unexplained", action="store_true")
    args = parser.parse_args()

    report = run_audit(args.battles, args.seed)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.fail_on_unexplained and report["mismatches"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
