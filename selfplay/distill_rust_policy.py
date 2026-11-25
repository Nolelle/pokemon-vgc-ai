#!/usr/bin/env python
"""Fast policy + value distillation from a trained Rust teacher (e.g. `v3c` 192x6)
into a larger/newer student `RustFactoredPolicy` (e.g. `v4b` 256x6, `policy_version=3`).

Eliminates cold-start sample waste by transferring 11M+ games of converged VGC policy
priors, Team Preview preferences, and calibrated win-probability evaluation into a
wider `256x6` network before PPO self-play fine-tuning (`train_rust_vec.py --init-from`).

Usage:
    .venv/bin/python selfplay/distill_rust_policy.py \
      --teacher runs/rust_v3c_m192_l6_counters/policy_4000012.pt \
      --out runs/rust_v4b_distill/v4b_init.pt \
      --team-manifest data/selfplay/v4_nov_meta_weighted_manifest.json \
      --use-manifest-weights \
      --width 256 --layers 6 --policy-version 3 \
      --waves 12 --games-per-wave 2048
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pikapro
import pikapro.rl as rl
import torch
import torch.nn.functional as F

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from selfplay.train_rust_vec import (  # noqa: E402
    DEFAULT_HOLDOUT_MANIFEST,
    DEFAULT_MANIFEST,
    _select_batch_actions,
    build_optimizer,
    eval_rust_opponents,
    load_teams,
    resolve_device,
    scheduled_lr,
)
from vgc.action_gate import enforce_action_gate_for_cli  # noqa: E402
from vgc.battle_state_gate import enforce_battle_state_gate_for_cli  # noqa: E402
from vgc.mechanics_gate import enforce_mechanics_gate_for_cli  # noqa: E402
from vgc.rl.rust_policy import (  # noqa: E402
    RustFactoredPolicy,
    load_rust_policy,
    save_rust_policy,
)


def _kl_div_masked(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact `KL(p_teacher || p_student)` over legal actions + top-1 agreement."""
    s_masked = student_logits.masked_fill(~mask, -1e9)
    t_masked = teacher_logits.masked_fill(~mask, -1e9)
    s_lp = F.log_softmax(s_masked, dim=-1)
    t_lp = F.log_softmax(t_masked, dim=-1)
    t_p = t_lp.exp()
    kl = (t_p * (t_lp - s_lp)).masked_fill(~mask, 0.0).sum(dim=-1).mean()
    agree = (s_masked.argmax(dim=-1) == t_masked.argmax(dim=-1)).float().mean()
    return kl, agree


@torch.no_grad()
def _collect_teacher_wave(
    teacher: RustFactoredPolicy,
    env: rl.VecEnv,
    step: object,
    *,
    target_games: int,
    n_self: int,
) -> tuple[dict[str, np.ndarray], object, int]:
    """Collect a wave of diverse teacher self-play + vs-anchor board states."""
    num_envs = env.n
    obs_list: list[np.ndarray] = []
    mask_list: list[np.ndarray] = []
    ms_list: list[np.ndarray] = []
    a0_list: list[np.ndarray] = []
    is_pv_list: list[np.ndarray] = []
    done_games = 0

    while done_games < target_games:
        actions = np.full((num_envs, 2, 2), -1, dtype=np.int64)
        p0_idx = np.flatnonzero(step.needs_action[:, 0])
        p1_idx = np.flatnonzero(step.needs_action[:n_self, 1]) if n_self > 0 else np.empty(0, dtype=np.int64)
        n0 = len(p0_idx)
        n1 = len(p1_idx)
        if n0 + n1 > 0:
            chunks_o, chunks_m, chunks_ms = [], [], []
            if n0 > 0:
                chunks_o.append(step.obs[p0_idx, 0])
                chunks_m.append(step.masks[p0_idx, 0])
                chunks_ms.append(step.min_switches[p0_idx, 0])
            if n1 > 0:
                chunks_o.append(step.obs[p1_idx, 1])
                chunks_m.append(step.masks[p1_idx, 1])
                chunks_ms.append(step.min_switches[p1_idx, 1])
            obs_cat = np.concatenate(chunks_o, axis=0)
            mask_cat = np.concatenate(chunks_m, axis=0)
            ms_cat = np.concatenate(chunks_ms, axis=0)
            a0, a1, _, _, is_pv = _select_batch_actions(
                teacher, env.space, obs_cat, mask_cat, ms_cat, deterministic=False
            )
            if n0 > 0:
                actions[p0_idx, 0, 0] = a0[:n0]
                actions[p0_idx, 0, 1] = a1[:n0]
            if n1 > 0:
                actions[p1_idx, 1, 0] = a0[n0:]
                actions[p1_idx, 1, 1] = a1[n0:]

            valid = (is_pv & (a0 >= 0)) | ((~is_pv) & (a0 >= 0) & (a1 >= 0))
            if valid.any():
                obs_list.append(obs_cat[valid].copy())
                mask_list.append(mask_cat[valid].copy())
                ms_list.append(ms_cat[valid].copy())
                a0_list.append(a0[valid].copy())
                is_pv_list.append(is_pv[valid].copy())

        step = env.step(actions)
        done_games += int(np.count_nonzero(step.done))

    wave = {
        "obs": np.concatenate(obs_list, axis=0),
        "mask": np.concatenate(mask_list, axis=0),
        "min_switches": np.concatenate(ms_list, axis=0),
        "a0": np.concatenate(a0_list, axis=0),
        "is_preview": np.concatenate(is_pv_list, axis=0),
    }
    return wave, step, done_games


@torch.no_grad()
def eval_student_vs_teacher(
    student: RustFactoredPolicy,
    teacher: RustFactoredPolicy,
    holdout_pool: rl.TeamPool,
    *,
    pairs: int,
    threads: int,
    seed: int,
) -> float:
    """Seat-swapped deterministic head-to-head (`2 * pairs` games) on unseen holdout teams."""
    batch_n = min(64, max(1, pairs))
    base_q, rem = divmod(pairs, batch_n)
    quotas = np.asarray([base_q + (1 if i < rem else 0) for i in range(batch_n)], dtype=np.int64)

    def _run_arm(m0: RustFactoredPolicy, m1: RustFactoredPolicy) -> float:
        env = rl.VecEnv(
            "championsvgc",
            n=batch_n,
            teams=holdout_pool,
            threads=threads,
            seed=seed,
            opponent="external",
            ots=0.0,
            preview="learned",
            masks="factored",
            observation="rich",
            side_randomization=True,
        )
        st = env.reset()
        done_per_env = np.zeros(batch_n, dtype=np.int64)
        pts = 0.0
        while int(done_per_env.sum()) < pairs:
            act = np.full((batch_n, 2, 2), -1, dtype=np.int64)
            for slot, m in ((0, m0), (1, m1)):
                idx = np.flatnonzero(st.needs_action[:, slot] & (done_per_env < quotas))
                if len(idx) > 0:
                    a0, a1, _, _, _ = _select_batch_actions(
                        m,
                        env.space,
                        st.obs[idx, slot],
                        st.masks[idx, slot],
                        st.min_switches[idx, slot],
                        deterministic=True,
                    )
                    act[idx, slot, 0] = a0
                    act[idx, slot, 1] = a1
                idle = st.needs_action[:, slot] & (done_per_env >= quotas)
                if idle.any():
                    act[idle, slot, 0] = rl.RANDOM_ACTION
                    act[idle, slot, 1] = rl.RANDOM_ACTION
            st = env.step(act)
            for i in np.flatnonzero(st.done):
                if done_per_env[i] >= quotas[i]:
                    continue
                pts += (float(st.reward[i, 0]) + 1.0) * 0.5
                done_per_env[i] += 1
        return pts / max(pairs, 1)

    wr0 = _run_arm(student, teacher)
    wr1 = 1.0 - _run_arm(teacher, student)
    return round(0.5 * (wr0 + wr1), 4)


def distill_wave(
    student: RustFactoredPolicy,
    teacher: RustFactoredPolicy,
    optimizer: torch.optim.Optimizer,
    space: pikapro.ActionSpace,
    wave: dict[str, np.ndarray],
    rng: random.Random,
    *,
    epochs: int,
    batch_size: int,
    value_weight: float,
) -> dict[str, float]:
    """Train `student` on one wave of states using exact soft KL + value BCE targets."""
    dev = student.device
    s = student.slot_size
    n = len(wave["obs"])
    turn_all = np.flatnonzero(~wave["is_preview"])
    pv_all = np.flatnonzero(wave["is_preview"])

    m1_full = np.zeros((n, s), dtype=bool)
    if len(turn_all) > 0:
        m1_full[turn_all] = space.pair_mask(
            wave["mask"][turn_all], wave["min_switches"][turn_all], wave["a0"][turn_all]
        )

    obs_t = torch.from_numpy(wave["obs"]).to(dev)
    mask_t = torch.from_numpy(wave["mask"]).to(dev)
    m1_t = torch.from_numpy(m1_full).to(dev)
    a0_t = torch.from_numpy(wave["a0"]).to(dev)

    teacher.eval()
    student.train()
    pv_micro = 64 if dev.type == "mps" else batch_size

    loss_sum = 0.0
    turn_kl_sum = 0.0
    turn_updates = 0
    pv_kl_sum = 0.0
    pv_updates = 0
    val_bce_sum = 0.0
    val_mae_sum = 0.0
    turn_agree_sum = 0.0
    pv_agree_sum = 0.0
    updates = 0

    for _ in range(epochs):
        minibatches: list[tuple[bool, list[int]]] = []
        for is_pv_g, idxs in ((False, turn_all.tolist()), (True, pv_all.tolist())):
            rng.shuffle(idxs)
            chunks = [
                idxs[st : st + batch_size]
                for st in range(0, len(idxs), batch_size)
                if idxs[st : st + batch_size]
            ]
            if len(chunks) >= 2 and len(chunks[-1]) < batch_size // 2:
                chunks[-2].extend(chunks.pop())
            for chunk in chunks:
                minibatches.append((is_pv_g, chunk))
        rng.shuffle(minibatches)

        for is_pv_g, rows in minibatches:
            optimizer.zero_grad(set_to_none=True)
            if is_pv_g:
                total_rows = len(rows)
                mb_loss = mb_kl = mb_vbce = mb_vmae = mb_agree = 0.0
                for sub_start in range(0, total_rows, pv_micro):
                    sub = rows[sub_start : sub_start + pv_micro]
                    w = len(sub) / total_rows
                    idx_t = torch.tensor(sub, dtype=torch.long, device=dev)
                    xb = obs_t[idx_t]
                    pvm = mask_t[idx_t, 2 * s :]
                    with torch.no_grad():
                        t_enc = teacher.encode_board(xb)
                        t_val, t_vlog = teacher.value_from_context(t_enc.context)
                        t_pvl = teacher.preview_logits(t_enc)
                        t_winp = torch.sigmoid(t_vlog)
                    s_enc = student.encode_board(xb)
                    s_val, s_vlog = student.value_from_context(s_enc.context)
                    s_pvl = student.preview_logits(s_enc)
                    kl_pv, _ = _kl_div_masked(s_pvl, t_pvl, pvm)
                    vbce = F.binary_cross_entropy_with_logits(s_vlog, t_winp)
                    loss = kl_pv + value_weight * vbce
                    if not torch.isfinite(loss):
                        raise RuntimeError("distillation preview loss was not finite")
                    (loss * w).backward()
                    with torch.no_grad():
                        s_g = student.preview_group_ids[s_pvl.masked_fill(~pvm, -1e9).argmax(dim=-1)]
                        t_g = student.preview_group_ids[t_pvl.masked_fill(~pvm, -1e9).argmax(dim=-1)]
                        mb_agree += float((s_g == t_g).float().mean().item()) * w
                        mb_vmae += float((s_val - t_val).abs().mean().item()) * w
                    mb_loss += float(loss.detach().item()) * w
                    mb_kl += float(kl_pv.detach().item()) * w
                    mb_vbce += float(vbce.detach().item()) * w
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                updates += 1
                pv_updates += 1
                loss_sum += mb_loss
                pv_kl_sum += mb_kl
                val_bce_sum += mb_vbce
                val_mae_sum += mb_vmae
                pv_agree_sum += mb_agree
            else:
                idx_t = torch.tensor(rows, dtype=torch.long, device=dev)
                xb = obs_t[idx_t]
                m0 = mask_t[idx_t, :s]
                m1 = m1_t[idx_t]
                a0_b = a0_t[idx_t]
                with torch.no_grad():
                    t_enc = teacher.encode_board(xb)
                    t_val, t_vlog = teacher.value_from_context(t_enc.context)
                    t_l0 = teacher.slot0_logits(t_enc)
                    t_l1 = teacher.slot1_logits(t_enc, a0_b)
                    t_winp = torch.sigmoid(t_vlog)
                s_enc = student.encode_board(xb)
                s_val, s_vlog = student.value_from_context(s_enc.context)
                s_l0 = student.slot0_logits(s_enc)
                s_l1 = student.slot1_logits(s_enc, a0_b)

                kl0, ag0 = _kl_div_masked(s_l0, t_l0, m0)
                kl1, ag1 = _kl_div_masked(s_l1, t_l1, m1)
                vbce = F.binary_cross_entropy_with_logits(s_vlog, t_winp)
                loss = kl0 + kl1 + value_weight * vbce
                if not torch.isfinite(loss):
                    raise RuntimeError("distillation turn loss was not finite")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()

                updates += 1
                turn_updates += 1
                loss_sum += float(loss.detach().item())
                turn_kl_sum += float((kl0 + kl1).detach().item())
                val_bce_sum += float(vbce.detach().item())
                val_mae_sum += float((s_val - t_val).abs().mean().detach().item())
                turn_agree_sum += 0.5 * float((ag0 + ag1).detach().item())

    student.eval()
    del obs_t, mask_t, m1_t, a0_t
    if dev.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()
        gc.collect()
        torch.mps.empty_cache()

    return {
        "loss": round(loss_sum / max(updates, 1), 5),
        "turn_kl": round(turn_kl_sum / max(turn_updates, 1), 5),
        "preview_kl": round(pv_kl_sum / max(pv_updates, 1), 5),
        "value_bce": round(val_bce_sum / max(updates, 1), 5),
        "value_mae": round(val_mae_sum / max(updates, 1), 5),
        "turn_top1_agree": round(turn_agree_sum / max(turn_updates, 1), 4),
        "preview_top1_agree": round(pv_agree_sum / max(pv_updates, 1), 4),
        "decisions": float(n),
    }


def distill(args: argparse.Namespace) -> Path:
    enforce_mechanics_gate_for_cli("model distillation")
    enforce_battle_state_gate_for_cli("model distillation")
    enforce_action_gate_for_cli("model distillation")

    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    train_teams = load_teams(args.team_manifest, use_weights=args.use_manifest_weights)
    train_pool = rl.TeamPool(train_teams, "championsvgc")
    holdout_teams = load_teams(args.holdout_manifest) if args.eval_every > 0 else []
    holdout_pool = rl.TeamPool(holdout_teams, "championsvgc") if holdout_teams else None

    teacher = load_rust_policy(args.teacher, device=device).eval()
    ffn_mult = int(args.ffn_mult) if int(args.ffn_mult) > 0 else None
    student = (
        RustFactoredPolicy(
            width=int(args.width),
            layers=int(args.layers),
            format_id=teacher.format_id,
            dex_sizes=teacher.dex_sizes,
            policy_version=int(args.policy_version),
            ffn_mult=ffn_mult,
        )
        .to(device)
        .eval()
    )
    optimizer = build_optimizer(student, lr=args.learning_rate, weight_decay=args.weight_decay)

    num_envs = int(args.num_envs)
    n_anchor = int(round(num_envs * args.anchor_fraction)) if args.anchor_opponent else 0
    n_self = num_envs - n_anchor
    env = rl.VecEnv(
        "championsvgc",
        n=num_envs,
        teams=train_pool,
        threads=args.threads,
        seed=args.seed,
        opponent="external",
        ots=0.0,
        preview="learned",
        masks="factored",
        observation="rich",
        autoreset="next_step",
        side_randomization=True,
    )
    if n_anchor > 0:
        env.set_opponent(args.anchor_opponent, which=list(range(n_self, num_envs)))
    step = env.reset()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.out.with_suffix(".jsonl")
    started = time.monotonic()
    total_games = 0
    total_decisions = 0
    best_vs_teacher = -1.0
    best_path = args.out.with_name(args.out.stem + "_best.pt")

    with log_path.open("w") as log_f:
        for wave_idx in range(1, args.waves + 1):
            w_t0 = time.monotonic()
            current_lr = scheduled_lr(
                wave_idx - 1,
                args.waves,
                max_lr=args.learning_rate,
                min_lr=args.min_learning_rate,
                schedule="cosine",
            )
            for group in optimizer.param_groups:
                group["lr"] = current_lr

            wave, step, g_done = _collect_teacher_wave(
                teacher, env, step, target_games=args.games_per_wave, n_self=n_self
            )
            total_games += g_done
            total_decisions += len(wave["obs"])

            metrics = distill_wave(
                student,
                teacher,
                optimizer,
                env.space,
                wave,
                rng,
                epochs=args.epochs_per_wave,
                batch_size=args.batch_size,
                value_weight=args.value_weight,
            )

            eval_metrics: dict[str, float] = {}
            if holdout_pool is not None and (
                wave_idx % args.eval_every == 0 or wave_idx == args.waves
            ):
                eval_metrics.update(
                    eval_rust_opponents(
                        student,
                        holdout_pool,
                        ["max_base_power", "simple_heuristics"],
                        games_per_opponent=args.eval_games,
                        threads=args.threads,
                        seed=args.seed + 900001,
                    )
                )
                if args.h2h_pairs > 0:
                    wr_t = eval_student_vs_teacher(
                        student,
                        teacher,
                        holdout_pool,
                        pairs=args.h2h_pairs,
                        threads=args.threads,
                        seed=args.seed + 920001,
                    )
                    eval_metrics["eval_vs_teacher"] = wr_t
                    if wr_t >= best_vs_teacher:
                        best_vs_teacher = wr_t
                        save_rust_policy(best_path, student)

            row = {
                "wave": wave_idx,
                "games": total_games,
                "total_decisions": total_decisions,
                "lr": round(current_lr, 8),
                "wave_seconds": round(time.monotonic() - w_t0, 2),
                "elapsed_seconds": round(time.monotonic() - started, 2),
                **metrics,
                **eval_metrics,
            }
            line = json.dumps(row, sort_keys=True)
            print(line, flush=True)
            log_f.write(line + "\n")
            log_f.flush()

    save_rust_policy(args.out, student)
    return args.out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--team-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--use-manifest-weights", action="store_true")
    parser.add_argument("--holdout-manifest", type=Path, default=DEFAULT_HOLDOUT_MANIFEST)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--policy-version", type=int, default=3, choices=(2, 3))
    parser.add_argument("--ffn-mult", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--waves", type=int, default=12)
    parser.add_argument("--games-per-wave", type=int, default=2048)
    parser.add_argument("--epochs-per-wave", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--anchor-opponent", default="simple_heuristics")
    parser.add_argument("--anchor-fraction", type=float, default=0.15)
    parser.add_argument("--eval-every", type=int, default=3)
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--h2h-pairs", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=37451)
    args = parser.parse_args()
    distill(args)


if __name__ == "__main__":
    raise SystemExit(main())
