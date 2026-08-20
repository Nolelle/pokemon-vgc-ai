"""Default-off neural candidate guidance for the existing VGC search.

There are deliberately two modes:

``shadow``
    The network ranks every legal joint action, but the shipped search selects and
    chooses exactly as it did before this module existed.

``hybrid``
    The network changes which fixed number of actions receive expensive search.  A
    small deterministic safety reserve keeps strategically distinct actions available;
    search still chooses the final action.

The module also owns the decision-record schema used to measure recall, known search
regret, and compute before any hybrid is considered for promotion.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from poke_env.battle.double_battle import DoubleBattle

try:
    import torch
except ImportError as exc:  # pragma: no cover - train extra is optional
    raise ImportError(
        "vgc.rl.search_guidance requires the train extra -- run `uv sync --extra train`."
    ) from exc

from vgc.actions import describe_order, enumerate_joint_orders
from vgc.agent import VgcPlayer
from vgc.bc.policy import load_bc_policy, score_orders
from vgc.decision_trace import record_note
from vgc.evaluation import clustered_interval
from vgc.evaluator import ScoredOrder
from vgc.rl.encoding import (
    CandidateFeatures,
    InformationFeatures,
    encode_battle_history,
    encode_candidates,
    encode_information_context,
    encode_live_state,
    encode_meta_context,
    pad_candidate_features,
    pad_candidate_tactical_features,
)
from vgc.search import _order_tags, _select_search_candidates, search_joint_orders

GuidanceMode = Literal["shadow", "hybrid"]
_CONTROL_TAGS = frozenset({"speed_control", "setup", "screen", "action_denial"})


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        while chunk := checkpoint_file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class NeuralRanking:
    """One model pass over every legal action, ordered from most to least preferred."""

    descriptions: tuple[str, ...]
    probabilities: tuple[float, ...]
    logits: tuple[float, ...]
    state_value: float
    elapsed_ms: float
    encoded: EncodedGuidanceInput | None = None

    @property
    def rank_by_description(self) -> dict[str, int]:
        return {description: rank for rank, description in enumerate(self.descriptions, start=1)}

    @property
    def probability_by_description(self) -> dict[str, float]:
        return dict(zip(self.descriptions, self.probabilities, strict=True))

    def rows(self) -> list[dict[str, object]]:
        return [
            {
                "action": description,
                "rank": rank,
                "probability": probability,
                "logit": logit,
            }
            for rank, (description, probability, logit) in enumerate(
                zip(self.descriptions, self.probabilities, self.logits, strict=True), start=1
            )
        ]


@dataclass(frozen=True)
class EncodedGuidanceInput:
    """Model-ready observation and candidates retained for later search distillation."""

    state_indices: np.ndarray
    state_scalars: np.ndarray
    history_scalars: np.ndarray
    candidates: CandidateFeatures
    meta_scalars: np.ndarray | None
    information: InformationFeatures | None
    action_descriptions: tuple[str, ...]
    neural_probabilities: np.ndarray


@dataclass
class SearchGuidanceSample:
    """One teacher example; NaN search scores mean the action was not searched."""

    battle_id: str
    team_id: str | None
    opponent_team_id: str | None
    observation_id: str
    encoded: EncodedGuidanceInput
    searched_mask: np.ndarray
    search_scores: np.ndarray
    search_winner_index: int
    outcome: float | None = None


def write_guidance_samples(path: Path, samples: list[SearchGuidanceSample]) -> None:
    """Save model-ready data without inventing labels for unsearched actions."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "vgc-neural-search-guidance-v1",
            "samples": samples,
            "sample_count": len(samples),
            "battle_count": len({sample.battle_id for sample in samples}),
            "team_count": len({sample.team_id for sample in samples if sample.team_id}),
        },
        path,
    )


def rank_legal_orders(
    model,
    battle: DoubleBattle,
    orders: list,
    memory,
    config,
    *,
    device: str,
) -> NeuralRanking:
    """Rank ``orders`` with the same fog-safe inputs used by ``PpoVgcPlayer``."""

    started_at = time.perf_counter()
    state_indices, state_scalars = encode_live_state(battle, config)
    history_scalars = encode_battle_history(memory)
    candidates = encode_candidates(
        orders,
        battle=battle if model.use_tactical_features else None,
        memory=memory if model.use_tactical_features else None,
        config=config,
    )
    moves, targets, species, flags, mask = pad_candidate_features([candidates])
    tactical = pad_candidate_tactical_features([candidates])
    meta = encode_meta_context(battle, config) if model.use_meta_features else None
    information = (
        encode_information_context(battle, memory, config)
        if model.use_information_features
        else None
    )

    model.eval()
    with torch.no_grad():
        logits, values = model(
            torch.as_tensor(state_indices[None, :], dtype=torch.long, device=device),
            torch.as_tensor(state_scalars[None, :], dtype=torch.float32, device=device),
            torch.as_tensor(history_scalars[None, :], dtype=torch.float32, device=device),
            torch.as_tensor(moves, dtype=torch.long, device=device),
            torch.as_tensor(targets, dtype=torch.long, device=device),
            torch.as_tensor(species, dtype=torch.long, device=device),
            torch.as_tensor(flags, dtype=torch.float32, device=device),
            torch.as_tensor(mask, dtype=torch.bool, device=device),
            meta_scalars=(
                torch.as_tensor(meta[None, :], dtype=torch.float32, device=device)
                if meta is not None
                else None
            ),
            information_indices=(
                torch.as_tensor(
                    information.indices[None, :], dtype=torch.long, device=device
                )
                if information is not None
                else None
            ),
            information_scalars=(
                torch.as_tensor(
                    information.scalars[None, :], dtype=torch.float32, device=device
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
        probabilities = torch.softmax(logits[0], dim=0)
        ranked_indices = torch.argsort(logits[0], descending=True).cpu().tolist()

    return NeuralRanking(
        descriptions=tuple(describe_order(orders[index]) for index in ranked_indices),
        probabilities=tuple(float(probabilities[index].item()) for index in ranked_indices),
        logits=tuple(float(logits[0, index].item()) for index in ranked_indices),
        state_value=float(values[0].item()),
        elapsed_ms=round((time.perf_counter() - started_at) * 1000.0, 3),
        encoded=EncodedGuidanceInput(
            state_indices=np.array(state_indices, copy=True),
            state_scalars=np.array(state_scalars, copy=True),
            history_scalars=np.array(history_scalars, copy=True),
            candidates=candidates,
            meta_scalars=np.array(meta, copy=True) if meta is not None else None,
            information=information,
            action_descriptions=tuple(describe_order(order) for order in orders),
            neural_probabilities=probabilities.cpu().numpy().copy(),
        ),
    )


def _best_distinct_with_tags(
    myopic: list[ScoredOrder],
    selected_ids: set[int],
    wanted: frozenset[str],
) -> ScoredOrder | None:
    return next(
        (
            entry
            for entry in myopic
            if id(entry) not in selected_ids and bool(_order_tags(entry.order) & wanted)
        ),
        None,
    )


def select_neural_guided_candidates(
    myopic: list[ScoredOrder],
    config,
    ranking: NeuralRanking,
    *,
    requested_safety_slots: int = 4,
) -> tuple[list[ScoredOrder], list[ScoredOrder], list[dict[str, str]]]:
    """Return an exact-K neural shortlist with deterministic safety slots inside K.

    At most half of the budget can be reserved for safety.  This keeps Hybrid-5 from
    becoming a disguised all-heuristic selector while still protecting the current
    myopic leader and rare switch/Protect/control/offense alternatives.
    """

    cutoff = min(len(myopic), max(1, int(config.search_our_candidates)))
    safety_budget = min(max(0, requested_safety_slots), cutoff // 2)
    selected: list[ScoredOrder] = []
    selected_ids: set[int] = set()
    safety: list[dict[str, str]] = []

    def add_safety(label: str, entry: ScoredOrder | None) -> None:
        if entry is None or id(entry) in selected_ids or len(safety) >= safety_budget:
            return
        selected.append(entry)
        selected_ids.add(id(entry))
        safety.append({"reason": label, "action": describe_order(entry.order)})

    if safety_budget:
        add_safety("heuristic_top", myopic[0])
        groups = (
            ("switch", frozenset({"switch"})),
            ("protect", frozenset({"protect"})),
            ("control", _CONTROL_TAGS),
            ("offense", frozenset({"double_attack"})),
            ("non_protect", frozenset({"non_protect"})),
        )
        for label, tags in groups:
            if len(safety) >= safety_budget:
                break
            if any(bool(_order_tags(entry.order) & tags) for entry in selected):
                continue
            add_safety(label, _best_distinct_with_tags(myopic, selected_ids, tags))

    entry_by_description = {describe_order(entry.order): entry for entry in myopic}
    for description in ranking.descriptions:
        if len(selected) >= cutoff:
            break
        entry = entry_by_description.get(description)
        if entry is not None and id(entry) not in selected_ids:
            selected.append(entry)
            selected_ids.add(id(entry))
    for entry in myopic:
        if len(selected) >= cutoff:
            break
        if id(entry) not in selected_ids:
            selected.append(entry)
            selected_ids.add(id(entry))

    unsearched = [entry for entry in myopic if id(entry) not in selected_ids]
    return selected, unsearched, safety


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def summarize_decision_records(records: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate shadow/hybrid records without pretending turns or teams are independent."""

    by_team_and_metric: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    regrets: dict[str, list[float]] = defaultdict(list)
    search_ms: list[float] = []
    ranking_ms: list[float] = []
    total_ms: list[float] = []
    exchanges: list[float] = []
    forecasts: list[float] = []
    for record in records:
        team = str(record.get("team_id") or record.get("battle_id") or "unknown")
        retention = dict(record.get("retention", {}))
        for metric, payload in retention.items():
            payload = dict(payload)
            by_team_and_metric[team][metric].append(int(bool(payload.get("retained"))))
            if payload.get("known_regret") is not None:
                regrets[metric].append(float(payload["known_regret"]))
        latency = dict(record.get("latency_ms", {}))
        search_ms.append(float(latency.get("search", 0.0)))
        ranking_ms.append(float(latency.get("neural_ranking", 0.0)))
        total_ms.append(float(latency.get("total", 0.0)))
        work = dict(record.get("search_work", {}))
        exchanges.append(float(work.get("exchange_count", 0.0)))
        forecasts.append(float(work.get("forecast_count", 0.0)))

    metric_names = sorted(
        {metric for per_team in by_team_and_metric.values() for metric in per_team}
    )
    retention_summary: dict[str, dict[str, object]] = {}
    for metric in metric_names:
        clusters = [
            (sum(per_team.get(metric, [])), len(per_team.get(metric, [])))
            for per_team in by_team_and_metric.values()
            if per_team.get(metric)
        ]
        hits = sum(wins for wins, _games in clusters)
        decisions = sum(games for _wins, games in clusters)
        values = regrets.get(metric, [])
        retention_summary[metric] = {
            "decisions": decisions,
            "retained": hits,
            "recall": hits / decisions if decisions else 0.0,
            "team_clustered_interval": clustered_interval(clusters),
            "known_regret_mean": sum(values) / len(values) if values else None,
            "known_regret_p95": _percentile(values, 95.0) if values else None,
            "known_regret_samples": len(values),
        }

    def latency_summary(values: list[float]) -> dict[str, float]:
        return {
            "p50": _percentile(values, 50.0),
            "p95": _percentile(values, 95.0),
            "mean": sum(values) / len(values) if values else 0.0,
        }

    return {
        "schema": "vgc-neural-search-shadow-summary-v1",
        "decisions": len(records),
        "teams": len(by_team_and_metric),
        "retention": retention_summary,
        "latency_ms": {
            "search": latency_summary(search_ms),
            "neural_ranking": latency_summary(ranking_ms),
            "total": latency_summary(total_ms),
        },
        "search_work": {
            "exchange_count_mean": sum(exchanges) / len(exchanges) if exchanges else 0.0,
            "forecast_count_mean": sum(forecasts) / len(forecasts) if forecasts else 0.0,
        },
    }


def write_decision_records(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records))


class NeuralSearchPlayer(VgcPlayer):
    """Rank with a checkpoint, but always let the existing search choose the move."""

    def __init__(
        self,
        *,
        model,
        checkpoint_path: Path,
        checkpoint_hash: str | None = None,
        mode: GuidanceMode = "shadow",
        device: str = "cpu",
        team_id: str | None = None,
        opponent_team_id: str | None = None,
        safety_slots: int = 4,
        metric_ks: tuple[int, ...] = (1, 3, 5, 8, 10),
        **player_kwargs,
    ) -> None:
        if mode not in ("shadow", "hybrid"):
            raise ValueError("mode must be 'shadow' or 'hybrid'")
        self.model = model.to(device)
        self.model.eval()
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint_hash = checkpoint_hash or checkpoint_sha256(self.checkpoint_path)
        self.mode = mode
        self.device = device
        self.team_id = team_id
        self.opponent_team_id = opponent_team_id
        self.safety_slots = max(0, int(safety_slots))
        self.metric_ks = tuple(sorted({int(k) for k in metric_ks if int(k) > 0}))
        self.decision_records: list[dict[str, object]] = []
        self.training_samples: list[SearchGuidanceSample] = []
        super().__init__(**player_kwargs)

    @staticmethod
    def _rebuild_myopic(scored: list[ScoredOrder]) -> list[ScoredOrder]:
        rebuilt = [
            ScoredOrder(
                order=entry.order,
                score=float(entry.breakdown.get("myopic_score", entry.score)),
                breakdown=dict(entry.breakdown),
            )
            for entry in scored
        ]
        rebuilt.sort(key=lambda entry: entry.score, reverse=True)
        return rebuilt

    @staticmethod
    def _retention_payload(
        descriptions: set[str],
        winner: str,
        reference_scores: dict[str, float],
    ) -> dict[str, object]:
        retained = winner in descriptions
        known_scores = [reference_scores[action] for action in descriptions if action in reference_scores]
        known_regret = None
        if not retained and known_scores and winner in reference_scores:
            known_regret = max(0.0, reference_scores[winner] - max(known_scores))
        return {"retained": retained, "known_regret": known_regret}

    def decide(self, battle):
        if not isinstance(battle, DoubleBattle) or not self.config.use_two_ply_search:
            return super().decide(battle)
        total_started_at = time.perf_counter()
        orders = enumerate_joint_orders(battle)
        if not orders:
            return self.choose_random_move(battle)
        memory = self._memory_for(battle)
        ranking = rank_legal_orders(
            self.model, battle, orders, memory, self.config, device=self.device
        )
        if ranking.encoded is None:  # defensive: real rank_legal_orders always supplies it
            raise RuntimeError("neural ranking did not retain its encoded observation")
        encoded = ranking.encoded
        selection_audit: dict[str, object] = {}

        if self.mode == "hybrid":

            def guided_selector(myopic, config):
                default_searched, _default_tail = _select_search_candidates(myopic, config)
                searched, unsearched, safety = select_neural_guided_candidates(
                    myopic,
                    config,
                    ranking,
                    requested_safety_slots=self.safety_slots,
                )
                selection_audit.update(
                    {
                        "heuristic_shortlist": [
                            describe_order(entry.order) for entry in default_searched
                        ],
                        "hybrid_shortlist": [describe_order(entry.order) for entry in searched],
                        "safety_candidates": safety,
                    }
                )
                return searched, unsearched

            scored = search_joint_orders(
                battle, self.config, candidate_selector=guided_selector
            )
        else:
            scored = search_joint_orders(battle, self.config)
            rebuilt = self._rebuild_myopic(scored)
            default_searched, _default_tail = _select_search_candidates(rebuilt, self.config)
            selection_audit["heuristic_shortlist"] = [
                describe_order(entry.order) for entry in default_searched
            ]

        if not scored:
            return self.choose_random_move(battle)
        if self.config.use_bc_policy:
            policy = load_bc_policy(self.config.bc_checkpoint_path)
            scored = score_orders(policy, battle, scored, self.config)

        winner = describe_order(scored[0].order)
        reference_scores = {
            describe_order(entry.order): float(entry.score)
            for entry in scored
            if bool(entry.breakdown.get("searched"))
        }
        retention: dict[str, dict[str, object]] = {}
        pure_ranked = list(ranking.descriptions)
        rebuilt = self._rebuild_myopic(scored)
        for k in self.metric_ks:
            pure = set(pure_ranked[: min(k, len(pure_ranked))])
            retention[f"neural_at_{k}"] = self._retention_payload(
                pure, winner, reference_scores
            )
            metric_config = type("MetricConfig", (), {"search_our_candidates": k})()
            guided, _tail, _safety = select_neural_guided_candidates(
                rebuilt,
                metric_config,
                ranking,
                requested_safety_slots=self.safety_slots,
            )
            guided_descriptions = {describe_order(entry.order) for entry in guided}
            retention[f"guided_at_{k}"] = self._retention_payload(
                guided_descriptions, winner, reference_scores
            )

        search_metrics = dict(scored[0].breakdown.get("search_metrics", {}))
        record = {
            "schema": "vgc-neural-search-decision-v1",
            "mode": self.mode,
            "battle_id": battle.battle_tag,
            "team_id": self.team_id,
            "opponent_team_id": self.opponent_team_id,
            "turn": int(getattr(battle, "turn", 0) or 0),
            "observation_id": f"{battle.battle_tag}:turn-{int(getattr(battle, 'turn', 0) or 0)}",
            "legal_action_count": len(orders),
            "checkpoint": str(self.checkpoint_path.resolve()),
            "checkpoint_sha256": self.checkpoint_hash,
            "state_value": ranking.state_value,
            "neural_actions": ranking.rows(),
            **selection_audit,
            "searched_actions": [
                {
                    "action": describe_order(entry.order),
                    "neural_rank": ranking.rank_by_description.get(
                        describe_order(entry.order)
                    ),
                    "neural_probability": ranking.probability_by_description.get(
                        describe_order(entry.order)
                    ),
                    "search_score": float(entry.score),
                    "myopic_score": float(entry.breakdown.get("myopic_score", 0.0)),
                    "opponent_responses": int(entry.breakdown.get("n_responses", 0)),
                }
                for entry in scored
                if bool(entry.breakdown.get("searched"))
            ],
            "search_winner": winner,
            "search_winner_neural_rank": ranking.rank_by_description.get(winner),
            "retention": retention,
            "search_work": {
                key: search_metrics.get(key, 0)
                for key in (
                    "searched_actions",
                    "opponent_responses",
                    "exchange_count",
                    "forecast_count",
                )
            },
            "latency_ms": {
                "neural_ranking": ranking.elapsed_ms,
                "search": float(search_metrics.get("elapsed_ms", 0.0)),
                "total": round((time.perf_counter() - total_started_at) * 1000.0, 3),
            },
            "outcome": None,
        }
        self.decision_records.append(record)
        original_descriptions = encoded.action_descriptions
        searched_mask = np.asarray(
            [description in reference_scores for description in original_descriptions],
            dtype=np.bool_,
        )
        search_scores = np.asarray(
            [reference_scores.get(description, np.nan) for description in original_descriptions],
            dtype=np.float32,
        )
        self.training_samples.append(
            SearchGuidanceSample(
                battle_id=battle.battle_tag,
                team_id=self.team_id,
                opponent_team_id=self.opponent_team_id,
                observation_id=str(record["observation_id"]),
                encoded=encoded,
                searched_mask=searched_mask,
                search_scores=search_scores,
                search_winner_index=original_descriptions.index(winner),
            )
        )
        record_note("neural_search", record)
        memory.record_choice(int(getattr(battle, "turn", 0) or 0), winner)
        return scored[0].order

    def _battle_finished_callback(self, battle) -> None:
        outcome = 1.0 if battle.won else (-1.0 if battle.lost else 0.0)
        for record in self.decision_records:
            if record["battle_id"] == battle.battle_tag:
                record["outcome"] = outcome
        for sample in self.training_samples:
            if sample.battle_id == battle.battle_tag:
                sample.outcome = outcome
