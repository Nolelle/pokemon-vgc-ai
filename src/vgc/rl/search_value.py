"""Shadow-only audit of the RL value head on existing search leaves.

The shipped search remains the actor. This module observes the already-searched
candidate/response states, translates their public state into the RL model's input,
and records what small value weights *would* have done. It never returns a score to
``vgc.search`` and therefore cannot change the move that is played.

Only the action actually selected by the baseline search receives the eventual battle
outcome as a valid target. Counterfactual actions are intentionally left unlabeled.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from poke_env.battle.double_battle import DoubleBattle

try:
    import torch
except ImportError as exc:  # pragma: no cover - train extra is optional
    raise ImportError(
        "vgc.rl.search_value requires the train extra -- run `uv sync --extra train`."
    ) from exc

from vgc.actions import describe_order
from vgc.agent import VgcPlayer
from vgc.bc.encoding import (
    MOVE_TO_IDX,
    SPECIES_TO_IDX,
    TARGET_TO_IDX,
    encode_state,
    flatten_state,
)
from vgc.bc.policy import exchange_state_record, load_bc_policy, score_orders
from vgc.decision_trace import record_note
from vgc.replay_parse import _resolve_species
from vgc.rl.encoding import (
    INFORMATION_TEAM_SLOTS,
    META_ACTIVE_SLOTS,
    META_ARCHETYPE_SLOTS,
    META_PRIOR_SCALARS_PER_SLOT,
    OPP_SCALARS_PER_MON,
    OWN_SCALARS_PER_MON,
    TACTICAL_FEATURE_DIM,
    InformationFeatures,
    encode_battle_history,
    encode_information_context,
    encode_live_state,
    encode_meta_context,
)
from vgc.rl.search_guidance import checkpoint_sha256
from vgc.search import (
    SearchShadowBatch,
    _aggregate_exchange_values,
    search_joint_orders,
)

DEFAULT_VALUE_WEIGHTS = (0.05, 0.1, 0.25, 0.5)
_STATUS_IDS = ("none", "brn", "par", "psn", "tox", "slp", "frz")
_STATUS_INDEX = {status: index for index, status in enumerate(_STATUS_IDS)}


def _weight_key(weight: float) -> str:
    return format(float(weight), ".6g")


def _status_vector(status: object) -> np.ndarray:
    value = np.zeros(len(_STATUS_IDS), dtype=np.float32)
    status_id = getattr(status, "name", str(status or "none")).lower()
    value[_STATUS_INDEX.get(status_id, 0)] = 1.0
    return value


def _base_species_id(raw: object) -> str:
    species_id = str(raw or "")
    return _resolve_species(species_id)[0]


def _future_record(leaf, context) -> dict:
    """Build the best public post-exchange record the current simulator supports."""

    record = exchange_state_record(
        leaf.exchange.our_states,
        leaf.exchange.opp_states,
        context,
    )
    state = record["state"]
    field = state["field"]
    field["weather"] = leaf.exchange.weather or field.get("weather")
    field["trick_room"] = bool(leaf.exchange.trick_room)
    field["turn"] = int(field.get("turn") or 0) + 1
    state["our"]["side_conditions"] = sorted(
        set(leaf.exchange.our_screens)
        | ({"tailwind"} if leaf.exchange.our_tailwind else set())
    )
    state["opp"]["side_conditions"] = sorted(
        set(leaf.exchange.opp_screens)
        | ({"tailwind"} if leaf.exchange.opp_tailwind else set())
    )
    return record


def _observed_future_mons(side: dict) -> dict[str, tuple[dict, bool]]:
    observed: dict[str, tuple[dict, bool]] = {}
    for mon in side.get("bench") or []:
        observed[_base_species_id(mon.get("species_id"))] = (mon, False)
    for mon in side.get("active") or []:
        if mon:
            observed[_base_species_id(mon.get("species"))] = (mon, True)
    return observed


def _patch_information(
    base: InformationFeatures,
    record: dict,
) -> InformationFeatures:
    """Patch only factual active/HP/status fields; keep fog-safe beliefs unchanged."""

    indices = np.array(base.indices, copy=True)
    scalars = np.array(base.scalars, copy=True)
    slots = INFORMATION_TEAM_SLOTS
    opp_species_offset = slots * (3 + 4)
    state = record["state"]

    our_observed = _observed_future_mons(state["our"])
    for slot in range(slots):
        species_index = int(indices[slot])
        species_id = next(
            (key for key, value in SPECIES_TO_IDX.items() if value == species_index),
            "",
        )
        observed = our_observed.get(_base_species_id(species_id))
        if observed is None:
            continue
        mon, active = observed
        row = slot * OWN_SCALARS_PER_MON
        hp = float(mon.get("hp_fraction") or 0.0)
        scalars[row + 1] = float(active and hp > 0.0)
        scalars[row + 2] = hp
        scalars[row + 3] = float(hp <= 0.0)
        scalars[row + 4 : row + 11] = _status_vector(mon.get("status"))

    opp_observed = _observed_future_mons(state["opp"])
    opp_scalar_start = slots * OWN_SCALARS_PER_MON
    for slot in range(slots):
        species_index = int(indices[opp_species_offset + slot])
        species_id = next(
            (key for key, value in SPECIES_TO_IDX.items() if value == species_index),
            "",
        )
        observed = opp_observed.get(_base_species_id(species_id))
        if observed is None:
            continue
        mon, active = observed
        row = opp_scalar_start + slot * OPP_SCALARS_PER_MON
        hp = float(mon.get("hp_fraction") or 0.0)
        scalars[row + 1] = 1.0
        scalars[row + 2] = float(active and hp > 0.0)
        scalars[row + 3] = hp
        scalars[row + 4] = float(hp <= 0.0)
    return InformationFeatures(indices=indices, scalars=scalars)


def _patch_meta(base: np.ndarray, record: dict, battle) -> np.ndarray:
    """Preserve team identity and update only the two active-opponent prior slots."""

    result = np.array(base, copy=True)
    scalar_start = META_ARCHETYPE_SLOTS * 2
    current = list(getattr(battle, "opponent_active_pokemon", None) or [])[:META_ACTIVE_SLOTS]
    current += [None] * (META_ACTIVE_SLOTS - len(current))
    future = list(record["state"]["opp"].get("active") or [])[:META_ACTIVE_SLOTS]
    future += [None] * (META_ACTIVE_SLOTS - len(future))
    for slot, (current_mon, future_mon) in enumerate(zip(current, future, strict=True)):
        row = scalar_start + slot * META_PRIOR_SCALARS_PER_SLOT
        if not future_mon or float(future_mon.get("hp_fraction") or 0.0) <= 0.0:
            result[row : row + META_PRIOR_SCALARS_PER_SLOT] = 0.0
            continue
        current_species = _base_species_id(getattr(current_mon, "species", None))
        future_species = _base_species_id(future_mon.get("species"))
        if current_species != future_species:
            # A hypothetical switch-in has revealed no moves in this branch. Presence
            # is certain; the remaining-prior fraction is conservatively unknown/high.
            result[row : row + META_PRIOR_SCALARS_PER_SLOT] = (1.0, 0.0, 1.0)
        else:
            result[row] = 1.0
    return result


def _record_state_arrays(record: dict) -> tuple[np.ndarray, np.ndarray]:
    indices, scalars = flatten_state(encode_state(record))
    return indices, np.concatenate((scalars, np.zeros(2, dtype=np.float32)))


def _batched_values(
    model,
    *,
    state_indices: list[np.ndarray],
    state_scalars: list[np.ndarray],
    history: np.ndarray,
    meta: list[np.ndarray] | None,
    information: list[InformationFeatures] | None,
    device: str,
) -> np.ndarray:
    count = len(state_indices)
    moves = np.full((count, 1, 2), MOVE_TO_IDX["<pass>"], dtype=np.int64)
    targets = np.full((count, 1, 2), TARGET_TO_IDX["<none>"], dtype=np.int64)
    species = np.full((count, 1, 2), SPECIES_TO_IDX["<pad>"], dtype=np.int64)
    flags = np.zeros((count, 1, 2, 4), dtype=np.float32)
    mask = np.ones((count, 1), dtype=np.bool_)
    tactical = np.zeros((count, 1, TACTICAL_FEATURE_DIM), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        _logits, values = model(
            torch.as_tensor(np.stack(state_indices), dtype=torch.long, device=device),
            torch.as_tensor(np.stack(state_scalars), dtype=torch.float32, device=device),
            torch.as_tensor(
                np.repeat(history[None, :], count, axis=0),
                dtype=torch.float32,
                device=device,
            ),
            torch.as_tensor(moves, dtype=torch.long, device=device),
            torch.as_tensor(targets, dtype=torch.long, device=device),
            torch.as_tensor(species, dtype=torch.long, device=device),
            torch.as_tensor(flags, dtype=torch.float32, device=device),
            torch.as_tensor(mask, dtype=torch.bool, device=device),
            meta_scalars=(
                torch.as_tensor(np.stack(meta), dtype=torch.float32, device=device)
                if meta is not None
                else None
            ),
            information_indices=(
                torch.as_tensor(
                    np.stack([item.indices for item in information]),
                    dtype=torch.long,
                    device=device,
                )
                if information is not None
                else None
            ),
            information_scalars=(
                torch.as_tensor(
                    np.stack([item.scalars for item in information]),
                    dtype=torch.float32,
                    device=device,
                )
                if information is not None
                else None
            ),
            tactical_features=(
                torch.as_tensor(tactical, dtype=torch.float32, device=device)
                if model.use_tactical_features
                else None
            ),
        )
    return values.detach().cpu().numpy().astype(np.float64)


def score_shadow_batch(
    model,
    batch: SearchShadowBatch,
    memory,
    config,
    *,
    device: str,
    value_weights: tuple[float, ...] = DEFAULT_VALUE_WEIGHTS,
) -> dict[str, object]:
    """Score all leaves once and report hypothetical blends without changing search."""

    started_at = time.perf_counter()
    battle = batch.context.battle
    history = encode_battle_history(memory)
    base_meta = encode_meta_context(battle, config) if model.use_meta_features else None
    base_information = (
        encode_information_context(battle, memory, config)
        if model.use_information_features
        else None
    )
    current_indices, current_scalars = encode_live_state(battle, config)
    state_indices = [current_indices]
    state_scalars = [current_scalars]
    metas = [base_meta] if base_meta is not None else None
    information = [base_information] if base_information is not None else None
    records: list[dict] = []
    leaves = []
    for action in batch.actions:
        for leaf in action.leaves:
            record = _future_record(leaf, batch.context)
            indices, scalars = _record_state_arrays(record)
            state_indices.append(indices)
            state_scalars.append(scalars)
            if metas is not None:
                metas.append(_patch_meta(base_meta, record, battle))
            if information is not None:
                information.append(_patch_information(base_information, record))
            records.append(record)
            leaves.append(leaf)

    values = _batched_values(
        model,
        state_indices=state_indices,
        state_scalars=state_scalars,
        history=history,
        meta=metas,
        information=information,
        device=device,
    )
    value_before = float(values[0])
    leaf_values = iter(float(value) for value in values[1:])
    action_rows: list[dict[str, object]] = []
    for action in batch.actions:
        afters = [next(leaf_values) for _leaf in action.leaves]
        responses = [leaf.response for leaf in action.leaves]
        deltas = [value - value_before for value in afters]
        aggregated_after = _aggregate_exchange_values(
            afters, responses, batch.context, config
        )
        aggregated_delta = _aggregate_exchange_values(
            deltas, responses, batch.context, config
        )
        hybrid_scores: dict[str, float] = {}
        for weight in value_weights:
            hybrid_leaf_values = [
                leaf.baseline_value + float(weight) * 100.0 * delta
                for leaf, delta in zip(action.leaves, deltas, strict=True)
            ]
            hybrid_exchange = _aggregate_exchange_values(
                hybrid_leaf_values, responses, batch.context, config
            )
            hybrid_scores[_weight_key(weight)] = float(
                config.search_myopic_weight * action.myopic_score
                + config.search_position_weight * hybrid_exchange
            )
        action_rows.append(
            {
                "action": describe_order(action.order),
                "search_score": action.baseline_final_score,
                "search_exchange_value": action.baseline_exchange_value,
                "rl_value_after_aggregate": aggregated_after,
                "rl_value_delta_aggregate": aggregated_delta,
                "hybrid_scores": hybrid_scores,
                "leaves": [
                    {
                        "opponent_response": leaf.response.describe(),
                        "search_leaf_value": leaf.baseline_value,
                        "rl_value_after": after,
                        "rl_value_delta": delta,
                        "outcome_target": None,
                    }
                    for leaf, after, delta in zip(
                        action.leaves, afters, deltas, strict=True
                    )
                ],
            }
        )

    baseline_winner = max(action_rows, key=lambda row: float(row["search_score"]))
    shadow_winners = {
        _weight_key(weight): max(
            action_rows,
            key=lambda row: float(dict(row["hybrid_scores"])[_weight_key(weight)]),
        )["action"]
        for weight in value_weights
    }
    return {
        "value_before": value_before,
        "value_before_win_equivalent": float(np.clip((value_before + 1.0) / 2.0, 0.0, 1.0)),
        "baseline_winner": baseline_winner["action"],
        "shadow_winners": shadow_winners,
        "actions": action_rows,
        "latency_ms": round((time.perf_counter() - started_at) * 1000.0, 3),
        "adapter_contract": {
            "history": "frozen_at_root",
            "beliefs": "fog_safe_root_beliefs_with_public_hp_active_patch",
            "counterfactual_outcomes": "unlabeled",
        },
    }


def _clustered_mean_interval(
    values_by_cluster: dict[str, list[float]], z: float = 1.96
) -> tuple[float, float]:
    usable = [values for values in values_by_cluster.values() if values]
    values = [value for cluster in usable for value in cluster]
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    if len(usable) < 2:
        variance = sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1)
        standard_error = math.sqrt(variance / len(values))
    else:
        residuals = [sum(value - mean for value in cluster) for cluster in usable]
        k = len(usable)
        standard_error = math.sqrt(
            k / ((k - 1) * len(values) ** 2) * sum(value**2 for value in residuals)
        )
    return mean - z * standard_error, mean + z * standard_error


def _fit_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    means = train_x.mean(axis=0)
    scales = train_x.std(axis=0)
    scales[scales < 1e-8] = 1.0
    train = (train_x - means) / scales
    test = (test_x - means) / scales
    train = np.column_stack((np.ones(len(train)), train))
    test = np.column_stack((np.ones(len(test)), test))
    ridge = np.eye(train.shape[1], dtype=np.float64) * 1e-6
    ridge[0, 0] = 0.0
    coefficients = np.linalg.solve(train.T @ train + ridge, train.T @ train_y)
    return np.clip(test @ coefficients, -1.0, 1.0)


def _incremental_predictive_metrics(records: list[dict[str, object]]) -> dict[str, object]:
    rows = [record for record in records if record.get("outcome") is not None]
    battle_ids = sorted({str(record["battle_id"]) for record in rows})
    if len(battle_ids) < 10 or len(rows) < 20:
        return {"available": False, "reason": "requires at least 10 battles and 20 decisions"}
    fold_by_battle = {
        battle_id: int(hashlib.sha256(battle_id.encode()).hexdigest(), 16) % 5
        for battle_id in battle_ids
    }
    outcomes = np.asarray([float(record["outcome"]) for record in rows], dtype=np.float64)
    baseline_features: list[list[float]] = []
    hybrid_features: list[list[float]] = []
    for record in rows:
        winner = str(record["baseline_winner"])
        action = next(item for item in record["actions"] if item["action"] == winner)
        baseline = [float(record["turn"]), float(action["search_score"])]
        baseline_features.append(baseline)
        hybrid_features.append(
            [*baseline, float(action["rl_value_after_aggregate"])]
        )
    baseline_x = np.asarray(baseline_features, dtype=np.float64)
    hybrid_x = np.asarray(hybrid_features, dtype=np.float64)
    baseline_predictions = np.zeros(len(rows), dtype=np.float64)
    hybrid_predictions = np.zeros(len(rows), dtype=np.float64)
    evaluated = np.zeros(len(rows), dtype=np.bool_)
    for fold in range(5):
        test = np.asarray(
            [fold_by_battle[str(record["battle_id"])] == fold for record in rows]
        )
        train = ~test
        if not test.any() or train.sum() < 10:
            continue
        baseline_predictions[test] = _fit_predict(
            baseline_x[train], outcomes[train], baseline_x[test]
        )
        hybrid_predictions[test] = _fit_predict(
            hybrid_x[train], outcomes[train], hybrid_x[test]
        )
        evaluated[test] = True
    if not evaluated.all():
        return {"available": False, "reason": "grouped folds were too small"}
    baseline_error = (baseline_predictions - outcomes) ** 2
    hybrid_error = (hybrid_predictions - outcomes) ** 2
    improvement = baseline_error - hybrid_error
    by_team: dict[str, list[float]] = defaultdict(list)
    for record, value in zip(rows, improvement, strict=True):
        by_team[str(record.get("team_id") or "unknown")].append(float(value))
    return {
        "available": True,
        "method": "five_fold_battle_disjoint_linear_prediction",
        "decisions": len(rows),
        "battles": len(battle_ids),
        "baseline_features": ["turn", "search_score"],
        "hybrid_added_feature": "aggregated_rl_value_after",
        "baseline_expected_return_mse": float(baseline_error.mean()),
        "hybrid_expected_return_mse": float(hybrid_error.mean()),
        "mse_reduction": float(improvement.mean()),
        "brier_reduction": float(improvement.mean() / 4.0),
        "team_clustered_mse_reduction_interval": _clustered_mean_interval(by_team),
        "adds_predictive_information": bool(
            _clustered_mean_interval(by_team)[0] > 0.0
        ),
        "interpretation": "predictive only; paired battles are still required for strength",
    }


def summarize_value_records(
    records: list[dict[str, object]],
    value_weights: tuple[float, ...] = DEFAULT_VALUE_WEIGHTS,
) -> dict[str, object]:
    completed = [record for record in records if record.get("outcome") is not None]
    raw_values = np.asarray(
        [float(record["value_before"]) for record in completed], dtype=np.float64
    )
    outcomes = np.asarray(
        [float(record["outcome"]) for record in completed], dtype=np.float64
    )
    probabilities = np.clip((raw_values + 1.0) / 2.0, 0.0, 1.0)
    wins = (outcomes + 1.0) / 2.0
    bins: list[dict[str, object]] = []
    for low in np.linspace(0.0, 0.8, 5):
        high = low + 0.2
        mask = (probabilities >= low) & (
            probabilities <= high if high >= 1.0 else probabilities < high
        )
        bins.append(
            {
                "range": [float(low), float(high)],
                "count": int(mask.sum()),
                "mean_prediction": float(probabilities[mask].mean()) if mask.any() else None,
                "observed_win_rate": float(wins[mask].mean()) if mask.any() else None,
            }
        )

    choice_changes: dict[str, dict[str, object]] = {}
    for weight in value_weights:
        key = _weight_key(weight)
        changed = 0
        baseline_regrets: list[float] = []
        for record in records:
            baseline = str(record["baseline_winner"])
            shadow = str(dict(record["shadow_winners"])[key])
            if shadow == baseline:
                continue
            changed += 1
            scores = {row["action"]: float(row["search_score"]) for row in record["actions"]}
            baseline_regrets.append(max(0.0, scores[baseline] - scores[shadow]))
        choice_changes[key] = {
            "decisions_changed": changed,
            "change_rate": changed / len(records) if records else 0.0,
            "baseline_search_score_regret_mean": (
                sum(baseline_regrets) / len(baseline_regrets) if baseline_regrets else 0.0
            ),
            "causal_win_claim": False,
        }

    latencies = [float(record.get("value_latency_ms", 0.0)) for record in records]
    return {
        "schema": "vgc-search-value-shadow-summary-v1",
        "decisions": len(records),
        "battles": len({str(record["battle_id"]) for record in records}),
        "teams": len({str(record.get("team_id")) for record in records}),
        "calibration": {
            "target": "expected_terminal_return_minus1_to_plus1",
            "expected_return_mse": (
                float(np.mean((raw_values - outcomes) ** 2)) if len(completed) else None
            ),
            "win_probability_brier": (
                float(np.mean((probabilities - wins) ** 2)) if len(completed) else None
            ),
            "raw_value_min": float(raw_values.min()) if len(completed) else None,
            "raw_value_max": float(raw_values.max()) if len(completed) else None,
            "bins": bins,
        },
        "incremental_predictive_value": _incremental_predictive_metrics(records),
        "shadow_choice_changes": choice_changes,
        "value_latency_ms": {
            "mean": sum(latencies) / len(latencies) if latencies else 0.0,
            "p50": float(np.percentile(latencies, 50)) if latencies else 0.0,
            "p95": float(np.percentile(latencies, 95)) if latencies else 0.0,
        },
        "outcome_label_contract": "only the baseline-played action is outcome-labeled",
    }


def write_value_records(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records))


class NeuralValueShadowPlayer(VgcPlayer):
    """Play unchanged full search while auditing RL values on its future leaves."""

    def __init__(
        self,
        *,
        model,
        checkpoint_path: Path,
        checkpoint_hash: str | None = None,
        device: str = "cpu",
        team_id: str | None = None,
        opponent_team_id: str | None = None,
        value_weights: tuple[float, ...] = DEFAULT_VALUE_WEIGHTS,
        active_value_weight: float | None = None,
        **player_kwargs,
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint_hash = checkpoint_hash or checkpoint_sha256(self.checkpoint_path)
        self.device = device
        self.team_id = team_id
        self.opponent_team_id = opponent_team_id
        self.value_weights = tuple(sorted({float(weight) for weight in value_weights}))
        if not self.value_weights or min(self.value_weights) <= 0.0:
            raise ValueError("value_weights must contain positive values")
        self.active_value_weight = (
            float(active_value_weight) if active_value_weight is not None else None
        )
        if (
            self.active_value_weight is not None
            and self.active_value_weight not in self.value_weights
        ):
            raise ValueError("active_value_weight must be included in value_weights")
        self.decision_records: list[dict[str, object]] = []
        super().__init__(**player_kwargs)
        if self.config.use_value_head:
            raise ValueError("shadow baseline requires PolicyConfig.use_value_head=False")

    def decide(self, battle):
        if not isinstance(battle, DoubleBattle) or not self.config.use_two_ply_search:
            return super().decide(battle)
        memory = self._memory_for(battle)
        shadow: dict[str, object] = {}

        def score_batch(batch: SearchShadowBatch) -> None:
            shadow.update(
                score_shadow_batch(
                    self.model,
                    batch,
                    memory,
                    self.config,
                    device=self.device,
                    value_weights=self.value_weights,
                )
            )

        def adjust(batch: SearchShadowBatch) -> list[list[float]]:
            score_batch(batch)
            return [
                [
                    float(leaf["search_leaf_value"])
                    + self.active_value_weight * 100.0 * float(leaf["rl_value_delta"])
                    for leaf in action["leaves"]
                ]
                for action in shadow["actions"]
            ]

        if self.active_value_weight is None:
            scored = search_joint_orders(battle, self.config, leaf_observer=score_batch)
        else:
            scored = search_joint_orders(
                battle,
                self.config,
                leaf_value_adjuster=adjust,
            )
        if not scored:
            return self.choose_random_move(battle)
        if self.config.use_bc_policy:
            scored = score_orders(
                load_bc_policy(self.config.bc_checkpoint_path), battle, scored, self.config
            )
        winner = describe_order(scored[0].order)
        if self.active_value_weight is None and shadow.get("baseline_winner") != winner:
            raise RuntimeError("value shadow baseline disagrees with the played search action")
        record = {
            "schema": "vgc-search-value-decision-v1",
            "battle_id": battle.battle_tag,
            "team_id": self.team_id,
            "opponent_team_id": self.opponent_team_id,
            "turn": int(getattr(battle, "turn", 0) or 0),
            "observation_id": f"{battle.battle_tag}:turn-{int(getattr(battle, 'turn', 0) or 0)}",
            "checkpoint": str(self.checkpoint_path.resolve()),
            "checkpoint_sha256": self.checkpoint_hash,
            "value_weights": list(self.value_weights),
            "active_value_weight": self.active_value_weight,
            "outcome": None,
            "played_action": winner,
            "outcome_target_action": winner,
            "value_latency_ms": float(shadow.pop("latency_ms", 0.0)),
            **shadow,
        }
        self.decision_records.append(record)
        record_note("neural_value_shadow", record)
        memory.record_choice(int(getattr(battle, "turn", 0) or 0), winner)
        return scored[0].order

    def _battle_finished_callback(self, battle) -> None:
        outcome = 1.0 if battle.won else (-1.0 if battle.lost else 0.0)
        for record in self.decision_records:
            if record["battle_id"] == battle.battle_tag:
                record["outcome"] = outcome


class NeuralValueSearchPlayer(NeuralValueShadowPlayer):
    """Default-off search player with one explicitly enabled learned-value weight."""

    def __init__(self, *, value_weight: float, **kwargs) -> None:
        super().__init__(
            value_weights=(float(value_weight),),
            active_value_weight=float(value_weight),
            **kwargs,
        )
