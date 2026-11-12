#!/usr/bin/env python
"""Train the replay-pretraining nets on a prepared dataset.

Loss = exactly three terms, weight 1.0 each:
  ACTION      = mean CE over pooled move + switch units (<switch> frequency comes
                only from the plan's draw ratio)
  DESTINATION = mean CE over switch units with a valid bench destination
  VALUE       = mean BCE over outcome units (logits head)
Selection loss = the three validation means. Preview net trains separately
(512/step, half brought / half leads). The test split is never opened.

    .venv/bin/python tools/train_replay_pretrain.py \
        --prepared runs/own_team_rebuild_20261104/scratch/prepared \
        --out runs/replay_pretrain/smoke --broad-steps 300 --expert-steps 100 \
        --eval-every 100 --seed 20261104
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from vgc.bc.encoding import MOVE_TO_IDX, SPECIES_TO_IDX, trunk_inputs  # noqa: E402
from vgc.bc.replay_pretrain import (  # noqa: E402
    COMBOS_4OF6, LEAD_PAIRS, SWITCH_IDX, PreviewPretrainNet, TurnPretrainNet,
)
from vgc.bc.replay_pretraining import file_hash, iter_task_examples, load_sampling_plan  # noqa: E402
from vgc.bc.replay_sampling import pool_task_within_game, sample_task_units  # noqa: E402
from vgc.data import canonical_species_id  # noqa: E402
from vgc.model_vocabulary import VOCABULARY_KEY, current_vocabulary  # noqa: E402

MAX_ACTION_OPTIONS = 5
MAX_DEST_OPTIONS = 8
BANDS = ("<1200", "1200-1399", "1400-1599", "1600+")
BUCKETS = ("1-2", "3-5", "6+")
UNK_IDX = SPECIES_TO_IDX["<unk>"]


def _band(r) -> str:
    if not isinstance(r, (int, float)) or r < 1200:
        return "<1200"
    return ("1600+" if r >= 1600 else "1400-1599" if r >= 1400 else "1200-1399")


def _bucket(t) -> str:
    t = int(t or 0)
    return "1-2" if t <= 2 else "3-5" if t <= 5 else "6+"


def _cache_fingerprint(prepared: Path, split: str) -> str:
    summary_path = prepared / "summary.json"
    summary = json.loads(summary_path.read_text())
    for name in (f"{split}.jsonl.gz", "sampling_broad.jsonl", "sampling_expert.jsonl"):
        if file_hash(prepared / name) != summary["files_sha256"][name]:
            raise ValueError(f"prepared artifact fingerprint mismatch: {name}")
    paths = [Path(__file__), REPO_ROOT / "src/vgc/bc/encoding.py",
             REPO_ROOT / "src/vgc/bc/replay_pretraining.py",
             REPO_ROOT / "src/vgc/bc/replay_pretrain.py",
             REPO_ROOT / "src/vgc/data.py",
             REPO_ROOT / "data/champions/cosmetic_formes.json"]
    return json.dumps({"prepared": str(prepared.resolve()), "split": split,
                       "summary_sha256": file_hash(summary_path),
                       "vocabulary": current_vocabulary(),
                       "code_sha256": {str(p.relative_to(REPO_ROOT)): file_hash(p) for p in paths}},
                      sort_keys=True)


def build_cache(prepared: Path, split: str, cache_dir: Path | None = None) -> dict:
    """Encode each referenced record once
    per-task contiguous arrays keyed by the
    `iter_task_examples` reference. Labels are validated at cache time: a move
    unit whose label is not among that slot's options and a switch unit whose
    destination is not among the bench options are skipped and counted."""
    if split == "test":
        raise AssertionError("the test split is never opened for pretraining")
    fingerprint = _cache_fingerprint(prepared, split)
    npz_path = (cache_dir / f"cache_{split}.npz") if cache_dir else None
    if npz_path and npz_path.exists():
        z = np.load(npz_path, allow_pickle=True)
        if "provenance_json" not in z or z["provenance_json"].item() != fingerprint:
            z.close()
            raise ValueError("encoded cache provenance mismatch; use a fresh cache directory")
        cached = {"records": {k: z[f"rec_{k}"] for k in (
            "idx", "scalars", "move_opts", "move_mask", "dest_opts", "dest_mask",
            "own6", "opp6")},
            "lineup": z["rec_lineup"].tolist(), "meta": {k: z[f"meta_{k}"] for k in (
            "rating", "turn_bucket", "fully", "lineup", "rating_val")},
            "tasks": {t: {k2: ({tuple(r[:-1]): r[-1] for r in v2}
                             if k2 == "refs" else
                             {tuple(r) for r in v2}
                             if k2 == "filtered" else
                             collections.defaultdict(list, v2)
                             if k2 == "meta" else v2)
                          for k2, v2 in d.items()}
                      for t, d in json.loads(z["tasks_json"].tobytes().decode()).items()},
            "split": split}
        z.close()
        return cached
    rec_rows = {}
    rec = collections.defaultdict(list)
    tasks = collections.defaultdict(
        lambda: {"row": [], "component": [], "label_pos": [], "refs": {},
                 "skipped": 0, "species": [], "filtered": set(),
                 "meta": collections.defaultdict(list)})
    t0 = time.time()
    for task in ("move", "switch_destination", "outcome", "preview_brought", "preview_leads"):
        for ex in iter_task_examples(prepared, task, split=split):
            if ((task == "outcome" and ex["meta"].get("decision_kind") == "teampreview")
                    or (task == "switch_destination" and ex["meta"].get("bench_guessed"))):
                tasks[task]["filtered"].add(ex["reference"])
                continue
            record = ex["input"]
            key = (ex["reference"][0], ex["reference"][1], ex["source_line"])
            if key not in rec_rows:
                rec_rows[key] = len(rec["idx"])
                i_arr, s_arr = trunk_inputs(record)
                rec["idx"].append(i_arr)
                rec["scalars"].append(s_arr)
                active = record["state"]["our"].get("active") or []
                mo = np.zeros((2, MAX_ACTION_OPTIONS), np.int64)
                mm = np.zeros((2, MAX_ACTION_OPTIONS), bool)
                species = [UNK_IDX, UNK_IDX]
                for i, mon in enumerate(active[:2]):
                    opts = [MOVE_TO_IDX[m] for m in (mon or {}).get("revealed_moves") or []
                            if m in MOVE_TO_IDX][:4]
                    opts.append(SWITCH_IDX)
                    mo[i, :len(opts)] = opts
                    mm[i, :len(opts)] = True
                    species[i] = SPECIES_TO_IDX.get(
                        canonical_species_id((mon or {}).get("species") or ""), UNK_IDX)
                rec["move_opts"].append(mo)
                rec["move_mask"].append(mm)
                rec["active_species"].append(species)
                do = np.zeros(MAX_DEST_OPTIONS, np.int64)
                dm = np.zeros(MAX_DEST_OPTIONS, bool)
                opts = [SPECIES_TO_IDX.get(canonical_species_id(e.get("species_id") or ""), UNK_IDX)
                        for e in record["state"]["our"].get("bench") or []][:MAX_DEST_OPTIONS]
                do[:len(opts)] = opts
                dm[:len(opts)] = True
                rec["dest_opts"].append(do)
                rec["dest_mask"].append(dm)
                for sk, out in (("our", "own6"), ("opp", "opp6")):
                    prev = [canonical_species_id(s) for s in (
                        record["state"][sk].get("preview_species")
                        or record["state"][sk].get("preview") or [])]
                    arr = np.full(6, -1, np.int64)
                    if len(prev) == 6:
                        arr[:] = sorted(SPECIES_TO_IDX.get(s, UNK_IDX) for s in prev)
                    rec[out].append(arr)
                rec["lineup"].append(tuple(sorted(prev)))
                meta = ex["meta"]
                rec["rating"].append(meta.get("pre_rating"))
                rec["turn"].append(meta.get("turn"))
                rec["fully"].append(bool(meta.get("own_fully_confirmed")))
                rec["showteam"].append(bool(meta.get("showteam")))
            row = rec_rows[key]
            t = tasks[task]
            comp = ex["component"]
            label, skip = None, False
            if task == "move":
                mi = MOVE_TO_IDX.get(ex["target"])
                opts = rec["move_opts"][row][comp]
                hit = np.nonzero(opts == mi)[0]
                if mi is None or comp is None or not len(hit) or not rec["move_mask"][row][comp][hit[0]]:
                    skip = True
                else:
                    label = int(hit[0])
            elif task == "switch_destination":
                sp = SPECIES_TO_IDX.get(canonical_species_id(ex["target"]))
                dopts = rec["dest_opts"][row]
                hit = np.nonzero((dopts == sp) & rec["dest_mask"][row])[0]
                if sp is None or not len(hit):
                    skip = True
                else:
                    label = int(hit[0])
            elif task == "outcome":
                label = 1.0 if ex["target"] == "win" else 0.0
            elif task in ("preview_brought", "preview_leads"):
                own6 = rec["own6"][row]
                if not (own6 >= 0).all() or len(set(own6.tolist())) != 6:
                    skip = True  # incomplete preview or duplicate (incl. <unk> x2)
                else:
                    try:
                        if task == "preview_brought":
                            pos = tuple(sorted(own6.tolist().index(
                                SPECIES_TO_IDX.get(canonical_species_id(s), UNK_IDX))
                                for s in ex["target"]))
                            label = COMBOS_4OF6.index(pos)
                        else:
                            pair = tuple(own6.tolist().index(
                                SPECIES_TO_IDX.get(canonical_species_id(s), UNK_IDX))
                                for s in ex["target"])
                            label = LEAD_PAIRS.index(pair)
                    except (ValueError, KeyError):
                        skip = True
            if skip:
                t["skipped"] += 1
                continue
            t["refs"][ex["reference"]] = len(t["row"])
            t["row"].append(row)
            t["component"].append(-1 if comp is None else comp)
            t["label_pos"].append(label)
            t["species"].append(rec["active_species"][row][comp] if comp is not None else -1)
            for k, v in (("rating", _band(rec["rating"][row])),
                         ("turn_bucket", _bucket(rec["turn"][row])),
                         ("fully", rec["fully"][row]),
                         ("lineup", rec["lineup"][row]),
                         ("showteam", rec["showteam"][row]),
                         ("rating_val", rec["rating"][row] or 0)):
                t["meta"][k].append(v)
    out = {"records": {k: np.stack(v) for k, v in rec.items() if k != "lineup"},
           "lineup": rec["lineup"], "meta": {
               "rating": np.array([_band(r) for r in rec["rating"]]),
               "turn_bucket": np.array([_bucket(t) for t in rec["turn"]]),
               "fully": np.array(rec["fully"]),
               "lineup": np.array(rec["lineup"]),
               "rating_val": np.array([r or 0 for r in rec["rating"]])},
           "tasks": tasks, "split": split, "built_seconds": round(time.time() - t0, 1)}
    if npz_path:
        np.savez(npz_path,
                 provenance_json=np.array(fingerprint),
                 **{f"rec_{k}": v for k, v in out["records"].items()},
                 rec_lineup=np.array(out["lineup"], dtype=object),
                 **{f"meta_{k}": v for k, v in out["meta"].items()},
                 tasks_json=np.frombuffer(json.dumps(
                     {t: {k2: ([list(rk) + [rv] for rk, rv in v2.items()]
                               if k2 == "refs" else dict(v2)
                               if k2 == "meta" else v2)
                          for k2, v2 in d.items()}
                      for t, d in tasks.items()}, default=list).encode(), dtype=np.uint8))
    return out


def _units(cache, task, refs, missing_counts):
    refs_map = cache["tasks"][task]["refs"]
    filtered = cache["tasks"][task].get("filtered") or ()
    units, miss = [], 0
    for r in refs:
        if r in filtered:
            continue
        if r in refs_map:
            units.append(refs_map[r])
        else:
            miss += 1
    missing_counts[task] += miss
    return units


def _gather(cache, task, units, device):
    t = cache["tasks"][task]
    rows = torch.tensor([t["row"][u] for u in units], device=device)
    comp = torch.tensor([t["component"][u] for u in units], device=device)
    lab = torch.tensor([t["label_pos"][u] for u in units], device=device)
    idx = torch.tensor(cache["records"]["idx"][rows.cpu().numpy()], device=device)
    sc = torch.tensor(cache["records"]["scalars"][rows.cpu().numpy()], device=device)
    return rows, comp, lab, idx, sc


def step_losses(net, cache, move_units, switch_units, outcome_units, device):
    """Return the three weight-1.0 terms: ACTION (pooled), DESTINATION, VALUE."""
    rec = cache["records"]
    terms = {}
    action_units = [("move", u) for u in move_units] + [
        ("switch_destination", u) for u in switch_units]
    if action_units:
        rows_l, comp_l, lab_l = [], [], []
        for task, u in action_units:
            t = cache["tasks"][task]
            rows_l.append(t["row"][u])
            comp_l.append(t["component"][u])
            if task == "move":
                lab_l.append(t["label_pos"][u])
            else:
                row = t["row"][u]
                c = t["component"][u]
                lab_l.append(int(np.nonzero(
                    rec["move_opts"][row][c] == SWITCH_IDX)[0][0]))
        comp = torch.tensor(comp_l, device=device)
        lab = torch.tensor(lab_l, device=device)
        idx = torch.tensor(rec["idx"][np.array(rows_l)], device=device)
        sc = torch.tensor(rec["scalars"][np.array(rows_l)], device=device)
        hidden = net.hidden(idx, sc)
        opts = torch.tensor(rec["move_opts"][np.array(rows_l), np.array(comp_l)],
                            device=device)
        mask = torch.tensor(rec["move_mask"][np.array(rows_l), np.array(comp_l)],
                            device=device)
        logits = net.action_logits(hidden, comp, opts, mask)
        terms["action"] = F.cross_entropy(logits, lab)
    if switch_units:
        t = cache["tasks"]["switch_destination"]
        rows_np = np.array([t["row"][u] for u in switch_units])
        comp_np = np.array([t["component"][u] for u in switch_units])
        lab = torch.tensor([t["label_pos"][u] for u in switch_units], device=device)
        idx = torch.tensor(rec["idx"][rows_np], device=device)
        sc = torch.tensor(rec["scalars"][rows_np], device=device)
        hidden = net.hidden(idx, sc)
        dopts = torch.tensor(rec["dest_opts"][rows_np], device=device)
        dmask = torch.tensor(rec["dest_mask"][rows_np], device=device)
        dlogits = net.destination_logits(
            hidden, torch.tensor(comp_np, device=device), dopts, dmask)
        terms["destination"] = F.cross_entropy(dlogits, lab)
    if outcome_units:
        _, _, lab, idx, sc = _gather(cache, "outcome", outcome_units, device)
        hidden = net.hidden(idx, sc)
        terms["value"] = F.binary_cross_entropy_with_logits(
            net.value_logit(hidden), lab.float())
    return terms


@torch.no_grad()
def evaluate_turn(net, cache, device, baselines, only_1600=False,
                  policy_floor=None, outcome_floor=None) -> dict:
    """Vectorized ACTION/DESTINATION/VALUE metrics + baselines + breakdowns."""
    net.eval()
    rec = cache["records"]
    out = {}
    # ACTION over pooled move+switch units; DESTINATION over switch units.
    # Headline metrics apply TRAINING eligibility: policy tasks need pre-rating >=
    # the policy floor and no open-sheet games; value uses the outcome floor.
    for name, which in (("action", ("move", "switch_destination")),
                        ("destination", ("switch_destination",))):
        rows_l, comp_l, lab_l, meta_i, sp_l, tasks_l = [], [], [], [], [], []
        for task in which:
            t = cache["tasks"][task]
            for u in range(len(t["row"])):
                if only_1600 and t["meta"]["rating_val"][u] < 1600:
                    continue
                if not only_1600:
                    if t["meta"]["showteam"][u]:
                        continue
                    if policy_floor is not None and (
                            t["meta"]["rating_val"][u] or 0) < policy_floor:
                        continue
                rows_l.append(t["row"][u])
                comp_l.append(t["component"][u])
                lab_l.append(t["label_pos"][u] if task == "move" or name == "destination"
                             else int(np.nonzero(
                                 rec["move_opts"][t["row"][u]][t["component"][u]]
                                 == SWITCH_IDX)[0][0]))
                meta_i.append(u)
                sp_l.append(t["species"][u])
                tasks_l.append(task)
        if not rows_l:
            out[name] = {"n": 0}
            continue
        rows_np, comp_np, lab_np = map(np.array, (rows_l, comp_l, lab_l))
        losses, correct, uniform_correct, freq_correct = [], 0, 0, 0
        correct_flags = []
        for lo in range(0, len(rows_np), 2048):
            sl = slice(lo, lo + 2048)
            idx = torch.tensor(rec["idx"][rows_np[sl]], device=device)
            sc = torch.tensor(rec["scalars"][rows_np[sl]], device=device)
            hidden = net.hidden(idx, sc)
            opts = torch.tensor(rec["move_opts"][rows_np[sl], comp_np[sl]], device=device) \
                if name == "action" else torch.tensor(rec["dest_opts"][rows_np[sl]], device=device)
            mask = torch.tensor(rec["move_mask"][rows_np[sl], comp_np[sl]], device=device) \
                if name == "action" else torch.tensor(rec["dest_mask"][rows_np[sl]], device=device)
            logits = (net.action_logits if name == "action" else net.destination_logits)(
                hidden, torch.tensor(comp_np[sl], device=device), opts, mask)
            lab = torch.tensor(lab_np[sl], device=device)
            losses.extend(F.cross_entropy(logits, lab, reduction="none").tolist())
            preds = logits.argmax(1).cpu().numpy()
            correct_flags.extend((preds == lab_np[sl]).tolist())
            correct += int((preds == lab_np[sl]).sum())
            n_opts = mask.sum(1).cpu().numpy()
            uniform_correct += float((1.0 / n_opts).sum())
            if name == "action":
                for j in range(len(rows_np[sl])):
                    freq = baselines["move_freq"].get(sp_l[lo + j], {})
                    cand = [o for o in opts[j][mask[j]].tolist()]
                    best = min(cand, key=lambda o: (-freq.get(o, 0), o))
                    freq_correct += int(opts[j][mask[j]].tolist().index(best) == lab_np[sl][j])
        n = len(rows_np)
        ent = {"n": n, "loss": float(np.mean(losses)),
               "top1": correct / n, "uniform": uniform_correct / n}
        if name == "action":
            ent["train_freq"] = freq_correct / n
        out[name] = ent
        # breakdowns on the same pooled units, in evaluated order
        out[name]["breakdowns"] = _breakdowns(
            cache, [(tasks_l[i], meta_i[i],
                     bool(correct_flags[i]), float(losses[i]))
                    for i in range(n)])
    # VALUE
    t = cache["tasks"]["outcome"]
    keep = [u for u in range(len(t["row"]))
            if (not only_1600 or t["meta"]["rating_val"][u] >= 1600)
            and (only_1600 or outcome_floor is None
                 or (t["meta"]["rating_val"][u] or 0) >= outcome_floor)]
    if keep:
        rows_np = np.array([t["row"][u] for u in keep])
        lab_np = np.array([t["label_pos"][u] for u in keep], dtype=np.float32)
        probs = np.concatenate([
            torch.sigmoid(net.value_logit(net.hidden(
                torch.tensor(rec["idx"][rows_np[sl]], device=device),
                torch.tensor(rec["scalars"][rows_np[sl]], device=device)))).cpu().numpy()
            for sl in (slice(lo, lo + 2048) for lo in range(0, len(rows_np), 2048))])
        n = len(keep)
        ll = float(np.mean(-(lab_np * np.log(probs + 1e-9) + (1 - lab_np) * np.log(1 - probs + 1e-9))))
        p0 = baselines["win_rate"]
        ll05 = float(np.mean(-(lab_np * np.log(0.5) + (1 - lab_np) * np.log(0.5))))
        llwr = float(np.mean(-(lab_np * np.log(p0 + 1e-9) + (1 - lab_np) * np.log(1 - p0 + 1e-9))))
        acc = float(np.mean((probs > 0.5) == lab_np.astype(bool)))
        cal = [{"bin": f"{b / 10:.1f}-{(b + 1) / 10:.1f}", "n": int(m.sum()),
                "mean_pred": float(probs[m].mean()) if m.sum() else None,
                "observed": float(lab_np[m].mean()) if m.sum() else None}
               for b in range(10)
               for m in [(probs >= b / 10) & (probs < (b + 1) / 10 + (b == 9))]]
        out["value"] = {"n": n, "loss": ll, "top1": acc,
                        "brier": float(np.mean((probs - lab_np) ** 2)),
                        "logloss_const_half": ll05, "logloss_train_rate": llwr,
                        "calibration": cal}
        value_flags = (probs > 0.5) == lab_np.astype(bool)
        out["value"]["breakdowns"] = _breakdowns(
            cache, [("outcome", keep[i], bool(value_flags[i]), None) for i in range(n)])
    else:
        out["value"] = {"n": 0}
    return out


def _breakdowns(cache, unit_rows):
    """`unit_rows` = parallel `(task, unit_index, correct, loss)` in evaluated
    order. Per-field n / accuracy / mean loss per value, per task entry."""
    fields = collections.defaultdict(lambda: collections.defaultdict(
        lambda: {"n": 0, "correct": 0, "loss_sum": 0.0, "loss_n": 0}))
    lineups_seen = cache.get("_train_lineups", set())
    for task, u, cm, loss in unit_rows:
        t = cache["tasks"][task]
        for f, v in (("rating", t["meta"]["rating"][u]),
                     ("turn_bucket", t["meta"]["turn_bucket"][u]),
                     ("lineup_seen", tuple(t["meta"]["lineup"][u]) in lineups_seen),
                     ("fully", t["meta"]["fully"][u])):
            d = fields[f][str(v)]
            d["n"] += 1
            d["correct"] += int(cm)
            if loss is not None:
                d["loss_sum"] += loss
                d["loss_n"] += 1
    return {f: {v: {"n": d["n"], "acc": d["correct"] / d["n"],
                    "loss": (d["loss_sum"] / d["loss_n"] if d["loss_n"] else None)}
                for v, d in vals.items()} for f, vals in fields.items()}
    return {f: {v: {"n": n, "acc": c / n} for v, (n, c) in d.items()}
            for f, d in fields.items()}


def build_baselines(train_cache) -> dict:
    """Train-only priors for the validation baselines."""
    move_freq = collections.defaultdict(collections.Counter)
    win = []
    for task in ("move", "switch_destination"):
        t = train_cache["tasks"][task]
        for u in range(len(t["row"])):
            sp = t["species"][u]
            if task == "move":
                move_freq[sp][int(train_cache["records"]["move_opts"]
                                  [t["row"][u]][t["component"][u]][t["label_pos"][u]])] += 1
            else:
                move_freq[sp][SWITCH_IDX] += 1
    previewed = collections.Counter()
    brought = collections.Counter()
    led0, led1 = collections.Counter(), collections.Counter()
    for u in range(len(train_cache["tasks"]["preview_brought"]["row"])):
        row = train_cache["tasks"]["preview_brought"]["row"][u]
        for sp in train_cache["records"]["own6"][row].tolist():
            previewed[int(sp)] += 1
        for pos in COMBOS_4OF6[train_cache["tasks"]["preview_brought"]["label_pos"][u]]:
            brought[int(train_cache["records"]["own6"][row][pos])] += 1
    for u in range(len(train_cache["tasks"]["preview_leads"]["row"])):
        row = train_cache["tasks"]["preview_leads"]["row"][u]
        a, b = LEAD_PAIRS[train_cache["tasks"]["preview_leads"]["label_pos"][u]]
        led0[int(train_cache["records"]["own6"][row][a])] += 1
        led1[int(train_cache["records"]["own6"][row][b])] += 1
    win = [train_cache["tasks"]["outcome"]["label_pos"][u]
           for u in range(len(train_cache["tasks"]["outcome"]["row"]))]
    bring_rate = {sp: brought[sp] / previewed[sp] for sp in previewed}
    lead0_rate = {sp: led0[sp] / previewed[sp] for sp in previewed}
    lead1_rate = {sp: led1[sp] / previewed[sp] for sp in previewed}
    return {"move_freq": {sp: dict(c) for sp, c in move_freq.items()},
            "bring_rate": bring_rate, "lead0_rate": lead0_rate,
            "lead1_rate": lead1_rate,
            "win_rate": float(np.mean(win)) if win else 0.5}


@torch.no_grad()
def evaluate_preview(net, cache, device, baselines, policy_floor=None) -> dict:
    net.eval()
    out = {}
    for task, top_name in (("preview_brought", "top4_bring"), ("preview_leads", "top_lead")):
        t = cache["tasks"][task]
        # preview tasks are policy tasks: same floor + no open-sheet games
        keep = [u for u in range(len(t["row"]))
                if not t["meta"]["showteam"][u]
                and (policy_floor is None
                     or (t["meta"]["rating_val"][u] or 0) >= policy_floor)]
        n = len(keep)
        if not n:
            out[task] = {"n": 0}
            continue
        losses, correct, base_correct = [], 0, 0
        for lo in range(0, n, 2048):
            units = keep[lo:lo + 2048]
            rows_np = np.array([t["row"][u] for u in units])
            own = torch.tensor(cache["records"]["own6"][rows_np], device=device)
            opp = torch.tensor(cache["records"]["opp6"][rows_np], device=device)
            brought, leads = net(own, opp)
            logits = brought if task == "preview_brought" else leads
            lab = torch.tensor([t["label_pos"][u] for u in units], device=device)
            losses.extend(F.cross_entropy(logits, lab, reduction="none").tolist())
            correct += int((logits.argmax(1) == lab).sum().item())
        base_correct = 0
        for u in keep:
            own6 = cache["records"]["own6"][t["row"][u]].tolist()
            if task == "preview_brought":
                pred = frozenset(sorted(
                    own6, key=lambda s: (-baselines["bring_rate"].get(int(s), 0.0), int(s))
                )[:4])
                actual = frozenset(int(own6[p]) for p in COMBOS_4OF6[t["label_pos"][u]])
                base_correct += int(pred == actual)
            else:
                l0, l1 = baselines["lead0_rate"], baselines["lead1_rate"]
                pred_pair = max(
                    ((a, b) for a in own6 for b in own6 if a != b),
                    key=lambda ab: (l0.get(int(ab[0]), 0.0) + l1.get(int(ab[1]), 0.0),
                                    -int(ab[0]), -int(ab[1])))
                a, b = LEAD_PAIRS[t["label_pos"][u]]
                base_correct += int(pred_pair == (own6[a], own6[b]))
        out[task] = {"n": n, "loss": float(np.mean(losses)), "top1": correct / n,
                     "uniform": 1.0 / (15 if task == "preview_brought" else 30),
                     top_name: base_correct / n}
    return out


def _save(net, path, provenance, phase=None):
    torch.save({
        "model_state_dict": net.state_encoder.state_dict(),
        "heads_state_dict": {k: v for k, v in net.state_dict().items()
                             if not k.startswith("state_encoder.")},
        VOCABULARY_KEY: current_vocabulary(),
        "provenance": provenance | ({"phase": phase} if phase else {}),
    }, path)


def _provenance(prepared: Path, seed: int) -> dict:
    return {
        "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "dirty": bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()),
        "prepared_summary_sha256": __import__("hashlib").sha256(
            (prepared / "summary.json").read_bytes()).hexdigest(),
        "seed": seed,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prepared", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--broad-steps", type=int, default=2000)
    ap.add_argument("--expert-steps", type=int, default=1000)
    ap.add_argument("--expert-lr-scale", type=float, default=0.1)
    ap.add_argument("--expert-eval-every", type=int, default=50)
    ap.add_argument("--preview-steps", type=int, default=4000)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20261104)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")

    t0 = time.time()
    train_cache = build_cache(args.prepared, "train", args.out)
    val_cache = build_cache(args.prepared, "validation", args.out)
    train_cache["_train_lineups"] = {tuple(x) for x in train_cache["lineup"]}
    val_cache["_train_lineups"] = train_cache["_train_lineups"]
    cache_seconds = round(time.time() - t0, 1)
    skips = {t: train_cache["tasks"][t]["skipped"] for t in train_cache["tasks"]}
    filtered = {t: len(train_cache["tasks"][t].get("filtered") or ())
                for t in train_cache["tasks"]}
    print("cache built in", cache_seconds, "s; units:",
          {t: len(train_cache["tasks"][t]["row"]) for t in train_cache["tasks"]},
          "skips:", skips, "filtered:", filtered)

    plan_broad = load_sampling_plan(args.prepared, phase="broad")
    plan_expert = load_sampling_plan(args.prepared, phase="expert")
    # Outcome rows pool within each game: the side with more rows (the loser,
    # via forced switches) no longer draws each row less often.
    plan_broad = pool_task_within_game(plan_broad, "outcome")
    plan_expert = pool_task_within_game(plan_expert, "outcome")
    eu = {t: plan_broad.diagnostics[t]["eligible_units"]
          for t in ("move", "switch_destination")}
    move_share = eu["move"] / max(eu["move"] + eu["switch_destination"], 1)
    baselines = build_baselines(train_cache)
    metrics_path = (args.out / "metrics.jsonl").open("w")

    def record(step, phase, **kw):
        metrics_path.write(json.dumps({"step": step, "phase": phase, **kw}) + "\n")
        metrics_path.flush()

    def draw_losses(plan, step, phase_i):
        n_out = 1024 // 3
        n_move = int((1024 - n_out) * move_share)
        n_sw = 1024 - n_out - n_move
        missing = collections.Counter()
        drawn = 0
        units = {}
        for task, n in (("move", n_move), ("switch_destination", n_sw), ("outcome", n_out)):
            refs = sample_task_units(plan, task, n,
                                     seed=args.seed * 1_000_000 + phase_i * 500_000 + step)
            drawn += len(refs)
            units[task] = _units(train_cache, task, refs, missing)
        if drawn and sum(missing.values()) / drawn > 0.05:
            raise RuntimeError(f">5% of drawn refs missing from cache at step {step}: {missing}")
        if missing:
            print(f"  step {step}: {sum(missing.values())}/{drawn} refs skipped (skipped units)")
        t_start = time.time()
        terms = step_losses(net, train_cache, units["move"],
                            units["switch_destination"], units["outcome"], device)
        loss = sum(terms.values()) if terms else None
        return loss, time.time() - t_start

    sampling_summary = json.loads((args.prepared / "summary.json").read_text())["sampling"]
    policy_floor = sampling_summary["broad"].get("broad_min_pre_rating")
    outcome_floor = sampling_summary["broad"].get("broad_outcome_min_pre_rating")

    def phase_val(phase):
        only_1600 = phase == "expert"
        m = evaluate_turn(net, val_cache, device, baselines, only_1600=only_1600,
                          policy_floor=policy_floor, outcome_floor=outcome_floor)
        vals = [m[t].get("loss") for t in ("action", "destination", "value")]
        return m, sum(v for v in vals if v is not None and math.isfinite(v))

    net = TurnPretrainNet().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    provenance = _provenance(args.prepared, args.seed)
    results = {}
    step_times = []
    for phase, plan, steps, lr_scale in (
            ("broad", plan_broad, args.broad_steps, 1.0),
            ("expert", plan_expert, args.expert_steps, args.expert_lr_scale)):
        if phase == "expert":
            for g in opt.param_groups:
                g["lr"] = 1e-3 * lr_scale
        best = {"loss": math.inf, "state": None, "step": -1}
        bad = 0
        eval_every = args.expert_eval_every if phase == "expert" else args.eval_every
        if phase == "expert":
            # Seed with the broad-best weights' evaluation: if no expert step
            # improves on it the phase keeps the broad weights (best_step 0).
            metrics, vl = phase_val(phase)
            record(0, phase, val=metrics, val_loss=vl)
            print(f"{phase} 0 val {vl:.3f}")
            best = {"loss": vl, "step": 0,
                    "state": {k: v.clone() for k, v in net.state_dict().items()},
                    "metrics": metrics}
        for step in range(1, steps + 1):
            loss, dt = draw_losses(plan, step, 0 if phase == "broad" else 1)
            step_times.append(dt)
            if loss is None:
                continue
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % eval_every == 0 or step == steps:
                metrics, vl = phase_val(phase)
                record(step, phase, train_loss=loss.item(), val=metrics, val_loss=vl)
                print(f"{phase} {step} train {loss.item():.3f} val {vl:.3f}")
                if vl < best["loss"]:
                    best = {"loss": vl, "step": step,
                            "state": {k: v.clone() for k, v in net.state_dict().items()},
                            "metrics": metrics}
                    bad = 0
                else:
                    bad += 1
                    if bad >= args.patience:
                        break
        if best["state"]:
            net.load_state_dict(best["state"])
        results[phase] = {"best_step": best["step"], "val_loss": best["loss"],
                          "metrics": best.get("metrics")}
        if phase == "expert":
            results[phase]["val_full"] = evaluate_turn(
                net, val_cache, device, baselines, only_1600=False,
                policy_floor=policy_floor, outcome_floor=outcome_floor)
        _save(net, args.out / ("turn_net.pt" if phase == "expert" else "turn_net_broad.pt"),
              provenance, phase)

    # --- preview net (broad plan, separate) ---
    pnet = PreviewPretrainNet().to(device)
    popt = torch.optim.AdamW(pnet.parameters(), lr=1e-3, weight_decay=1e-4)
    best_preview = {"loss": math.inf, "state": None, "step": -1}
    bad = 0
    for step in range(1, args.preview_steps + 1):
        losses = []
        for task, n in (("preview_brought", 256), ("preview_leads", 256)):
            refs = sample_task_units(plan_broad, task, n, seed=args.seed * 3_000_000 + step)
            units = _units(train_cache, task, refs, collections.Counter())
            if not units:
                continue
            rows_np = np.array([train_cache["tasks"][task]["row"][u] for u in units])
            own = torch.tensor(train_cache["records"]["own6"][rows_np], device=device)
            opp = torch.tensor(train_cache["records"]["opp6"][rows_np], device=device)
            brought, leads = pnet(own, opp)
            lab = torch.tensor([train_cache["tasks"][task]["label_pos"][u] for u in units],
                               device=device)
            losses.append(F.cross_entropy(
                brought if task == "preview_brought" else leads, lab))
        if not losses:
            continue
        total = torch.stack(losses).mean()
        popt.zero_grad()
        total.backward()
        popt.step()
        if step % args.eval_every == 0 or step == args.preview_steps:
            preview_val = evaluate_preview(pnet, val_cache, device, baselines,
                                           policy_floor=policy_floor)
            preview_losses = [m["loss"] for m in preview_val.values()
                              if m.get("loss") is not None]
            preview_vl = float(np.mean(preview_losses)) if preview_losses else math.inf
            record(step, "preview", train_loss=total.item(), val=preview_val,
                   val_loss=preview_vl)
            if preview_vl < best_preview["loss"]:
                best_preview = {"loss": preview_vl, "step": step,
                                "state": {k: v.clone() for k, v in pnet.state_dict().items()}}
                bad = 0
            else:
                bad += 1
                if bad >= args.patience:
                    break
    if best_preview["state"] is not None:
        pnet.load_state_dict(best_preview["state"])
    torch.save({"model_state_dict": pnet.state_dict(),
                VOCABULARY_KEY: current_vocabulary(), "provenance": provenance},
               args.out / "preview_net.pt")
    metrics_path.close()
    (args.out / "final_metrics.json").write_text(json.dumps(
        {"broad": results["broad"], "expert": results["expert"],
         "preview_validation": evaluate_preview(pnet, val_cache, device, baselines,
                                                policy_floor=policy_floor),
         "cache_seconds": cache_seconds,
         "mean_step_seconds": float(np.mean(step_times)) if step_times else None,
         "outcome_sampling": "pooled_within_game",
         "preview_best_step": best_preview["step"],
         "skips": skips, "filtered": filtered}, indent=2))
    (args.out / "config.json").write_text(json.dumps(
        {"args": {k: str(v) for k, v in vars(args).items()},
         "provenance": provenance}, indent=2))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
