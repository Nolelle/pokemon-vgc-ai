"""Teach the candidate policy to imitate the existing search policy before PPO.

This is a bootstrap, not the final objective. The teacher supplies a competent legal
joint action for each state; PPO later remains free to improve beyond that teacher from
actual win/loss rewards.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from poke_env.battle.double_battle import DoubleBattle

try:
    import torch
    from torch import nn
    from torch.distributions import Categorical
except ImportError as exc:  # pragma: no cover - train extra is optional
    raise ImportError(
        "vgc.rl.distill requires the 'train' extra (torch) -- run `uv sync --extra train`."
    ) from exc

from vgc.actions import describe_order
from vgc.agent import VgcPlayer
from vgc.evaluator import score_joint_orders
from vgc.models import PolicyConfig
from vgc.rl.guided_selection import (
    SAFETY_COLUMN_TAGS,
    SAFETY_TAG_COLUMNS,
    select_guided_candidate_indices,
)
from vgc.search import _order_tags, search_joint_orders
from vgc.rl.encoding import (
    META_SCALAR_DIM,
    CandidateFeatures,
    InformationFeatures,
    INFORMATION_INDEX_DIM,
    INFORMATION_SCALAR_DIM,
    encode_battle_history,
    encode_candidates,
    encode_live_state,
    encode_information_context,
    encode_meta_context,
    pad_candidate_features,
    pad_candidate_tactical_features,
)
from vgc.rl.mechanics_encoding import (
    MechanicsFeatures,
    encode_mechanics_context,
    pad_mechanics_features,
)
from vgc.rl.exact_search import combine_belief_rankings, search_joint_orders_exact
from vgc.rl.live_mirror import LiveExactMirror

PUBLIC_TEACHER_SOURCE_ID = "public_mirror_exact_showdown_teacher_v2"


@dataclass(frozen=True)
class DistillationConfig:
    epochs: int = 8
    batch_size: int = 128
    max_grad_norm: float = 1.0
    val_fraction: float = 0.2
    seed: int = 0
    # Consecutive epochs with no improvement in validation accuracy (only checked when
    # distill_policy is called with val_samples) before training stops early. 0 (the
    # default) disables early stopping entirely -- distill_policy always runs the full
    # `epochs` and never restores a "best" checkpoint, current behavior byte-for-byte.
    early_stopping_patience: int = 0
    # Which validation metric picks the restored "best" epoch. "loss" keeps the
    # historical behavior (validation cross-entropy primary). "recall_at_k" selects by
    # validation Recall@K instead -- the shortlist gate is set membership at rank K, not
    # distribution sharpness, so an epoch that sharpens rank-1-vs-2 distinctions while
    # pushing one more action past rank K is visibly worse here while cross-entropy
    # barely notices. Val loss breaks ties in both modes.
    checkpoint_metric: str = "loss"
    checkpoint_recall_k: int = 10
    # Upweight training samples whose teacher action sits past rank K under the CURRENT
    # model -- exactly the samples that fail the shortlist gate. Ranks are recomputed at
    # each epoch boundary from a fresh pass over the training set; 0.0 disables. The
    # first epoch's ranks come from a randomly initialized model, so they are noise --
    # harmless, just not yet informative.
    hard_example_weight: float = 0.0
    hard_example_rank: int = 10
    # Rebalance sampling toward large-branching turns: expected draws equalized across
    # the legal-action-count bins the recall screen reports (<=25, 26-100, 101-200,
    # 201+), with samples of unknown count keeping their original overall share. Without
    # this, the rare 201+ bin (~6% of collected decisions) contributes too few gradient
    # updates to move the stratum that fails most. False keeps plain uniform shuffling.
    balance_action_count_bins: bool = False
    # Train against the teacher's full score distribution (softmax of the stored
    # `search_scores` at `soft_target_temperature`) instead of its argmax. Measured tie
    # mass on the 5x collection: ~23% of decisions hold a top-2 gap under 10 points --
    # argmax labels there are close to coin flips, and hard cross-entropy spends real
    # gradient defending arbitrary choices while recall@K cares about the whole shortlist.
    # Samples whose dataset predates stored scores fall back to hard labels per-row.
    soft_targets: bool = False
    soft_target_temperature: float = 16.0


@dataclass(frozen=True)
class DistillationSample:
    battle_id: str
    state_indices: np.ndarray
    state_scalars: np.ndarray
    history_scalars: np.ndarray
    candidates: CandidateFeatures
    teacher_action_index: int
    # A battle turn can contain more than one request: for example, replace a fainted
    # Pokemon and then choose moves without the displayed turn number changing.
    decision_index: int = 0
    request_kind: str | None = None
    # Always recorded (regardless of whether the model being distilled uses meta
    # features) -- cheap and lets --meta-features be combined with --bootstrap-games
    # without needing a separate teacher-recording pass.
    meta_scalars: np.ndarray | None = None
    information: InformationFeatures | None = None
    mechanics: MechanicsFeatures | None = None
    source_id: str = "simulator_teacher"
    team_id: str | None = None
    opponent_team_id: str | None = None
    # Stratification metadata. Recall is not uniform across a battle -- a turn with 300
    # legal joint orders is a much harder ranking problem than one with 20 -- so a
    # headline recall number that does not say WHICH states it came from is not
    # comparable across collections. Defaulted so older saved datasets still load.
    turn: int = 0
    legal_action_count: int = 0
    # Schema-v2 guidance metadata, aligned with the candidate rows. `candidate_ranks[i]`
    # is the candidate's 0-based position in the MYOPIC evaluator's score order (-1 if
    # unmatched) and `candidate_tags[i]` flags the safety categories
    # (`vgc.rl.guided_selection.SAFETY_TAG_COLUMNS`) its joint order satisfies. Together
    # they let an offline evaluator replay the deployed safety-slotted shortlist
    # (`vgc.rl.search_guidance`) instead of approximating it with the raw top-K -- which
    # matters because up to half of the deployed budget is heuristic-reserved. Defaulted
    # so older saved datasets still load (they simply cannot simulate guided selection).
    candidate_myopic_ranks: np.ndarray | None = None
    candidate_tags: np.ndarray | None = None
    # Schema-v2.x guidance metadata, aligned with the candidate rows. `search_scores`
    # is the teacher's COMPLETE comparable score vector -- `search_joint_orders`
    # re-scores searched actions from simulated exchanges and leaves unsearched ones
    # at their (scaled) myopic score, so all values share one currency -- and
    # `searched_mask` marks which rows are exchange-refined. This is what lets later
    # experiments (soft targets over near-tied actions, tie-mass measurement,
    # regret-weighted mining) work on THIS collection instead of requiring yet
    # another one: the argmax label alone throws away exactly the information those
    # losses need. Defaulted so older datasets keep loading untouched.
    search_scores: np.ndarray | None = None
    searched_mask: np.ndarray | None = None
    # Canonical action strings make the saved example independently auditable: the
    # teacher answer must appear exactly once in the legal candidate set. Feature arrays
    # alone cannot prove that membership after the original battle object is gone.
    candidate_descriptions: tuple[str, ...] | None = None
    teacher_action_description: str | None = None
    # Content hashes, not filenames, define team-disjoint splits. Two identical packed
    # teams renamed in different generated pools must still be recognized as overlap.
    team_sha256: str | None = None
    opponent_team_sha256: str | None = None


def public_information_exact_search(
    battle,
    config: PolicyConfig,
    own_packed_team: str,
    *,
    memory=None,
    mirror: LiveExactMirror | None = None,
):
    """Rank from a public reconstruction, never the direct simulator's private root."""

    if not own_packed_team:
        raise ValueError("the public-information teacher requires its packed own team")
    owned_mirror = mirror is None
    live_mirror = mirror or LiveExactMirror(own_packed_team, config)
    root = None
    rankings = []
    try:
        for belief in live_mirror.hypotheses(battle, memory):
            root = (
                live_mirror.rebase(root, battle, belief)
                if root is not None
                else live_mirror.build(battle, belief)
            )
            rankings.append(
                (
                    belief.weight,
                    search_joint_orders_exact(root, "p1", config),
                )
            )
        return combine_belief_rankings(rankings)
    finally:
        if root is not None:
            root.close()
        if owned_mirror:
            live_mirror.close()


def _shuffle_for_storage(scored, battle_id: str, decision_index: int):
    """Deterministically hide the teacher answer's row position in the saved data."""

    stored = list(scored)
    digest = hashlib.sha256(
        f"{battle_id}:{decision_index}:candidate-order-v1".encode()
    ).digest()
    random.Random(int.from_bytes(digest[:8], "big")).shuffle(stored)
    return stored


class TeacherRecordingPlayer(VgcPlayer):
    """Play the existing policy while recording its choice among every legal order."""

    def __init__(self, **player_kwargs) -> None:
        self.distillation_samples: list[DistillationSample] = []
        self.recording_failures: list[str] = []
        self.skipped_fallback_to_search: int = 0
        self.skipped_fallback_to_random: int = 0
        self._recording_decision_index = 0
        supplied_team = player_kwargs.get("team")
        self._exact_own_packed_team = supplied_team if isinstance(supplied_team, str) else None
        self._public_exact_mirror: LiveExactMirror | None = None
        super().__init__(**player_kwargs)

    def skipped_decisions(self) -> int:
        return len(self.recording_failures)

    def attempted_decisions(self) -> int:
        return self._recording_decision_index

    def close_public_mirror(self) -> None:
        if self._public_exact_mirror is not None:
            self._public_exact_mirror.close()
            self._public_exact_mirror = None

    def _battle_finished_callback(self, battle) -> None:
        self.close_public_mirror()
        super()._battle_finished_callback(battle)

    def _recording_failure(self, battle, reason: str):
        """Skip this decision's label but keep playing a sane move.

        A skipped decision records its cause visibly and the game continues on
        the shipped Python search (random only if that also fails), so one
        unlabelable position cannot abort a whole collection run. Skips bias
        the dataset toward labelable positions -- `collect_demonstrations`
        callers must report the skip rate by cause and gate on it instead of
        pretending every position was teachable.
        """

        self.recording_failures.append(
            f"{battle.battle_tag} turn {int(getattr(battle, 'turn', 0) or 0)}: {reason}"
        )
        try:
            searched = (
                search_joint_orders(battle, self.config)
                if isinstance(battle, DoubleBattle)
                else []
            )
        except Exception:
            searched = []
        if searched:
            self.skipped_fallback_to_search += 1
            return searched[0].order
        self.skipped_fallback_to_random += 1
        return self.choose_random_move(battle)

    def decide(self, battle):
        if not isinstance(battle, DoubleBattle):
            return self.choose_random_move(battle)
        self._recording_decision_index += 1
        decision_index = self._recording_decision_index
        request_kind = (
            "switch" if any(getattr(battle, "force_switch", None) or ()) else "move"
        )
        memory = self._memory_for(battle)
        state_indices, state_scalars = encode_live_state(battle, self.config)
        history_scalars = encode_battle_history(memory)
        if self.config.use_two_ply_search:
            if not self._exact_own_packed_team:
                return self._recording_failure(battle, "packed own team is unavailable")
            if self._public_exact_mirror is None:
                self._public_exact_mirror = LiveExactMirror(
                    self._exact_own_packed_team, self.config
                )
            try:
                scored = public_information_exact_search(
                    battle,
                    self.config,
                    self._exact_own_packed_team,
                    memory=memory,
                    mirror=self._public_exact_mirror,
                )
            except Exception as exc:
                self.recording_failures.append(
                    f"{battle.battle_tag} turn "
                    f"{int(getattr(battle, 'turn', 0) or 0)}: exact search raised {exc!r}; "
                    f"force_switch={getattr(battle, 'force_switch', None)!r}, "
                    f"wait={getattr(battle, 'wait', None)!r}, "
                    f"available_moves={[len(slot) for slot in battle.available_moves]!r}, "
                    f"available_switches="
                    f"{[len(slot) for slot in battle.available_switches]!r}"
                )
                raise
            # One extra myopic pass per decision: the exchange search reorders candidates
            # by simulated outcome, but the safety-slot replay needs each candidate's
            # position in the CHEAP evaluator's order. Worth its cost at collection time
            # only.
            myopic_scored = score_joint_orders(battle, self.config) if scored else None
        else:
            # A myopic evaluator is a policy baseline, not a mechanics-exact labeler.
            return self._recording_failure(battle, "exact search is disabled")
        if not scored:
            return self._recording_failure(battle, "exact search returned no legal ranking")

        chosen = scored[0].order
        teacher_description = describe_order(chosen)
        stored_scored = _shuffle_for_storage(
            scored,
            battle.battle_tag,
            decision_index,
        )
        orders = [entry.order for entry in stored_scored]
        descriptions = tuple(describe_order(order) for order in orders)
        teacher_matches = [
            index for index, description in enumerate(descriptions)
            if description == teacher_description
        ]
        if len(teacher_matches) != 1:
            return self._recording_failure(
                battle, "teacher answer did not match exactly one stored legal action"
            )
        guidance_metadata = build_guidance_metadata(orders, myopic_scored)
        ranks, tags = guidance_metadata if guidance_metadata is not None else (None, None)
        score_metadata = build_score_metadata(stored_scored)
        score_vector, searched_flags = (
            score_metadata if score_metadata is not None else (None, None)
        )
        self.distillation_samples.append(
            DistillationSample(
                battle_id=battle.battle_tag,
                state_indices=np.array(state_indices, copy=True),
                state_scalars=np.array(state_scalars, copy=True),
                history_scalars=np.array(history_scalars, copy=True),
                candidates=encode_candidates(
                    orders, battle=battle, memory=memory, config=self.config
                ),
                teacher_action_index=teacher_matches[0],
                decision_index=decision_index,
                request_kind=request_kind,
                meta_scalars=encode_meta_context(battle, self.config),
                information=encode_information_context(battle, memory, self.config),
                mechanics=encode_mechanics_context(battle),
                source_id=PUBLIC_TEACHER_SOURCE_ID,
                turn=int(getattr(battle, "turn", 0) or 0),
                legal_action_count=len(orders),
                candidate_myopic_ranks=ranks,
                candidate_tags=tags,
                search_scores=score_vector,
                searched_mask=searched_flags,
                candidate_descriptions=descriptions,
                teacher_action_description=teacher_description,
            )
        )
        memory.record_choice(int(getattr(battle, "turn", 0) or 0), describe_order(chosen))
        return chosen

    def completed_samples(self) -> list[DistillationSample]:
        completed = {tag for tag, battle in self.battles.items() if battle.finished}
        return [sample for sample in self.distillation_samples if sample.battle_id in completed]


def teacher_action_index(
    battle,
    config,
    orders,
    *,
    own_packed_team: str | None = None,
    memory=None,
) -> int | None:
    """Return the existing policy's preferred action within ``orders``.

    The network's candidate order is intentionally independent of the evaluator's
    score order, so matching uses the complete Showdown order description, including
    targets and transformation flags.
    """

    if config.use_two_ply_search:
        if not own_packed_team:
            return None
        scored = public_information_exact_search(
            battle,
            config,
            own_packed_team,
            memory=memory,
        )
    else:
        return None
    if not scored:
        return None
    teacher_description = describe_order(scored[0].order)
    matches = [
        index for index, order in enumerate(orders) if describe_order(order) == teacher_description
    ]
    return matches[0] if len(matches) == 1 else None


def build_score_metadata(scored) -> tuple[np.ndarray, np.ndarray] | None:
    """Teacher score vector + exchange-searched flags, aligned with ``scored`` order.

    Under a two-ply search teacher, ``searched`` rows carry exchange-refined scores and
    the rest scaled myopic ones; under a myopic-only teacher every score is already
    final, so the mask is all-True (nothing was left unrefined).
    """

    if not scored:
        return None
    scores = np.asarray([float(entry.score) for entry in scored], dtype=np.float32)
    flags = [entry.breakdown.get("searched") for entry in scored]
    if all(flag is None for flag in flags):
        # No 'searched' key anywhere: a myopic-only teacher whose scores are all final.
        searched = np.ones(len(scored), dtype=np.bool_)
    else:
        searched = np.asarray([bool(flag) for flag in flags], dtype=np.bool_)
    return scores, searched


def build_guidance_metadata(orders, myopic_scored) -> tuple[np.ndarray, np.ndarray] | None:
    """Per-candidate myopic position and safety-tag flags aligned with ``orders``.

    ``myopic_scored`` is the cheap evaluator's complete legal set in its own score order
    -- either ``ScoredOrder`` entries (as the search returns them) or bare order objects
    when the teacher itself is myopic. Returns None when no myopic ordering is available,
    so callers can store schema-v1 samples unchanged. Tag COLUMNS are the coarse safety
    labels (`SAFETY_TAG_COLUMNS`); the stored "control" column means "any of
    `CONTROL_TAGS`" and is expanded back to those tags on replay.
    """

    if not orders or myopic_scored is None:
        return None
    position: dict[str, int] = {}
    for rank, entry in enumerate(myopic_scored):
        position.setdefault(describe_order(getattr(entry, "order", entry)), rank)
    ranks = np.full(len(orders), -1, dtype=np.int64)
    tags = np.zeros((len(orders), len(SAFETY_TAG_COLUMNS)), dtype=np.int8)
    for index, order in enumerate(orders):
        ranks[index] = position.get(describe_order(order), -1)
        order_tags = _order_tags(order)
        for column, name in enumerate(SAFETY_TAG_COLUMNS):
            tags[index, column] = int(bool(order_tags & SAFETY_COLUMN_TAGS[name]))
    return ranks, tags


def split_samples_by_battle(
    samples: list[DistillationSample],
    *,
    val_fraction: float,
    seed: int,
) -> tuple[list[DistillationSample], list[DistillationSample]]:
    """Game-disjoint train/validation split so adjacent turns cannot leak across it."""

    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    battle_ids = sorted({sample.battle_id for sample in samples})
    if len(battle_ids) < 2:
        raise ValueError("distillation requires at least two completed battles")
    rng = random.Random(seed)
    rng.shuffle(battle_ids)
    val_count = max(1, min(len(battle_ids) - 1, round(len(battle_ids) * val_fraction)))
    val_ids = set(battle_ids[:val_count])
    train = [sample for sample in samples if sample.battle_id not in val_ids]
    val = [sample for sample in samples if sample.battle_id in val_ids]
    return train, val


def _padded_scores(sample: DistillationSample, width: int) -> np.ndarray:
    row = np.full(width, np.nan, dtype=np.float32)
    if sample.search_scores is not None:
        count = (
            sample.legal_action_count
            if 0 < sample.legal_action_count <= len(sample.search_scores)
            else len(sample.search_scores)
        )
        row[:count] = sample.search_scores[:count]
    return row


def soft_target_distribution(
    scores: torch.Tensor,
    candidate_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Teacher score distribution over LEGAL candidates; zero on padded columns.

    Rows whose scores are entirely missing (schema-v1 samples) produce all-zero rows --
    callers must route those to hard labels via the availability mask rather than use
    this output.
    """

    finite = torch.isfinite(scores)
    masked = torch.where(finite & candidate_mask.bool(), scores, torch.full_like(scores, float("-inf")))
    return torch.softmax(masked / float(temperature), dim=1)


def _tensor_batch(samples: list[DistillationSample], device: str) -> dict[str, torch.Tensor]:
    moves, targets, species, flags, mask = pad_candidate_features(
        [sample.candidates for sample in samples]
    )
    tactical = pad_candidate_tactical_features([sample.candidates for sample in samples])
    batch = {
        "state_indices": torch.as_tensor(
            np.stack([sample.state_indices for sample in samples]),
            dtype=torch.long,
            device=device,
        ),
        "state_scalars": torch.as_tensor(
            np.stack([sample.state_scalars for sample in samples]),
            dtype=torch.float32,
            device=device,
        ),
        "history_scalars": torch.as_tensor(
            np.stack([sample.history_scalars for sample in samples]),
            dtype=torch.float32,
            device=device,
        ),
        "move_indices": torch.as_tensor(moves, dtype=torch.long, device=device),
        "target_indices": torch.as_tensor(targets, dtype=torch.long, device=device),
        "switch_species_indices": torch.as_tensor(species, dtype=torch.long, device=device),
        "flags": torch.as_tensor(flags, dtype=torch.float32, device=device),
        "candidate_mask": torch.as_tensor(mask, dtype=torch.bool, device=device),
        "tactical_features": torch.as_tensor(tactical, dtype=torch.float32, device=device),
        "teacher_actions": torch.as_tensor(
            [sample.teacher_action_index for sample in samples],
            dtype=torch.long,
            device=device,
        ),
        # Raw teacher scores, NaN-padded to the batch width. Rows from schema-v1
        # samples are all-NaN and must fall back to hard labels (see
        # soft_target_distribution).
        "search_scores": torch.as_tensor(
            np.stack(
                [
                    _padded_scores(sample, moves.shape[1])
                    for sample in samples
                ]
            ),
            dtype=torch.float32,
            device=device,
        ),
        # Hand-built samples (e.g. in tests) may not set meta_scalars -- fall back to a
        # well-formed zero vector rather than requiring every caller to populate it.
        "meta_scalars": torch.as_tensor(
            np.stack(
                [
                    sample.meta_scalars
                    if sample.meta_scalars is not None
                    else np.zeros(META_SCALAR_DIM, dtype=np.float32)
                    for sample in samples
                ]
            ),
            dtype=torch.float32,
            device=device,
        ),
    }
    batch["information_indices"] = torch.as_tensor(
        np.stack(
            [
                sample.information.indices
                if sample.information is not None
                else np.zeros(INFORMATION_INDEX_DIM, dtype=np.int64)
                for sample in samples
            ]
        ),
        dtype=torch.long,
        device=device,
    )
    batch["information_scalars"] = torch.as_tensor(
        np.stack(
            [
                sample.information.scalars
                if sample.information is not None
                else np.zeros(INFORMATION_SCALAR_DIM, dtype=np.float32)
                for sample in samples
            ]
        ),
        dtype=torch.float32,
        device=device,
    )
    present_mechanics = [sample.mechanics for sample in samples if sample.mechanics is not None]
    if present_mechanics and len(present_mechanics) != len(samples):
        raise ValueError("a batch cannot mix complete and missing mechanics snapshots")
    if present_mechanics:
        mechanics_tokens, mechanics_mask = pad_mechanics_features(present_mechanics)
    else:
        mechanics_tokens = np.zeros((len(samples), 1), dtype=np.int64)
        mechanics_mask = np.zeros((len(samples), 1), dtype=np.bool_)
    batch["mechanics_tokens"] = torch.as_tensor(
        mechanics_tokens, dtype=torch.long, device=device
    )
    batch["mechanics_mask"] = torch.as_tensor(
        mechanics_mask, dtype=torch.bool, device=device
    )
    return batch


def evaluate_agreement(
    model: nn.Module,
    samples: list[DistillationSample],
    *,
    batch_size: int,
    device: str,
) -> dict[str, float]:
    if not samples:
        raise ValueError("cannot evaluate an empty distillation sample set")
    model.eval()
    correct = 0
    top3_correct = 0
    total = 0
    losses: list[float] = []
    entropies: list[float] = []
    teacher_probabilities: list[float] = []
    teacher_ranks: list[int] = []
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch = _tensor_batch(samples[start : start + batch_size], device)
            logits, _values = model(
                batch["state_indices"],
                batch["state_scalars"],
                batch["history_scalars"],
                batch["move_indices"],
                batch["target_indices"],
                batch["switch_species_indices"],
                batch["flags"],
                batch["candidate_mask"],
                meta_scalars=batch["meta_scalars"] if model.use_meta_features else None,
                information_indices=(
                    batch["information_indices"] if model.use_information_features else None
                ),
                information_scalars=(
                    batch["information_scalars"] if model.use_information_features else None
                ),
                tactical_features=(
                    batch["tactical_features"] if model.use_tactical_features else None
                ),
                mechanics_tokens=(
                    batch["mechanics_tokens"]
                    if getattr(model, "use_mechanics_features", False)
                    else None
                ),
                mechanics_mask=(
                    batch["mechanics_mask"]
                    if getattr(model, "use_mechanics_features", False)
                    else None
                ),
            )
            loss = nn.functional.cross_entropy(logits, batch["teacher_actions"])
            distribution = Categorical(logits=logits)
            probabilities = distribution.probs
            teacher_probabilities.extend(
                probabilities.gather(1, batch["teacher_actions"].unsqueeze(1))
                .squeeze(1)
                .cpu()
                .tolist()
            )
            ranked = logits.topk(min(3, logits.shape[-1]), dim=-1).indices
            correct += int((ranked[:, 0] == batch["teacher_actions"]).sum().item())
            top3_correct += int(
                (ranked == batch["teacher_actions"].unsqueeze(1)).any(dim=1).sum().item()
            )
            # Same rank definition as recall_at_k: legal candidates strictly outscoring
            # the teacher, so validation tracks the shortlist gate during training.
            teacher_logit = logits.gather(1, batch["teacher_actions"].unsqueeze(1))
            better = ((logits > teacher_logit) & batch["candidate_mask"].bool()).sum(dim=1)
            teacher_ranks.extend((better + 1).cpu().tolist())
            total += logits.shape[0]
            losses.append(float(loss))
            entropies.append(float(distribution.entropy().mean()))
    rank_array = np.asarray(teacher_ranks, dtype=np.float64)
    return {
        "accuracy": correct / total,
        "top3_accuracy": top3_correct / total,
        "recall_at_5": float((rank_array <= 5).mean()),
        "recall_at_10": float((rank_array <= 10).mean()),
        "teacher_rank_median": float(np.median(rank_array)),
        "teacher_rank_p95": float(np.percentile(rank_array, 95)),
        "teacher_probability": sum(teacher_probabilities) / len(teacher_probabilities),
        "loss": sum(losses) / len(losses),
        "entropy": sum(entropies) / len(entropies),
        "samples": float(total),
    }


def _action_count_bin(count: int) -> str | None:
    """Same bins the offline recall screen reports; None marks an unknown count."""

    if count <= 0:
        return None
    if count <= 25:
        return "0-25"
    if count <= 100:
        return "26-100"
    if count <= 200:
        return "101-200"
    return "201+"


def _action_count_sampling_probabilities(
    samples: list[DistillationSample],
) -> np.ndarray | None:
    """Per-sample draw probabilities that equalize the four action-count bins.

    Known-count samples share mass uniformly across bins (each bin gets 1/4 of the
    pool's total mass); unknown-count samples keep their original overall share so old
    datasets are not silently distorted. Returns None when there is nothing to rebalance
    (no metadata, or fewer than two distinct bins).
    """

    from collections import Counter

    total = len(samples)
    known_bins = [_action_count_bin(int(s.legal_action_count)) for s in samples]
    known_total = sum(bin_name is not None for bin_name in known_bins)
    if known_total == 0 or known_total == total:
        return None
    counts = Counter(bin_name for bin_name in known_bins if bin_name is not None)
    if len(counts) < 2:
        return None
    known_mass = known_total / total
    probabilities = np.full(total, (1.0 - known_mass) / (total - known_total))
    for index, bin_name in enumerate(known_bins):
        if bin_name is not None:
            probabilities[index] = known_mass / (len(counts) * counts[bin_name])
    return probabilities


def _teacher_ranks(
    model: nn.Module,
    samples: list[DistillationSample],
    *,
    batch_size: int,
    device: str,
) -> np.ndarray:
    """Rank of the teacher's action under the model, one value per sample."""

    model.eval()
    ranks: list[int] = []
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch = _tensor_batch(samples[start : start + batch_size], device)
            logits, _values = model(
                batch["state_indices"],
                batch["state_scalars"],
                batch["history_scalars"],
                batch["move_indices"],
                batch["target_indices"],
                batch["switch_species_indices"],
                batch["flags"],
                batch["candidate_mask"],
                meta_scalars=batch["meta_scalars"] if model.use_meta_features else None,
                information_indices=(
                    batch["information_indices"] if model.use_information_features else None
                ),
                information_scalars=(
                    batch["information_scalars"] if model.use_information_features else None
                ),
                tactical_features=(
                    batch["tactical_features"] if model.use_tactical_features else None
                ),
                mechanics_tokens=(
                    batch["mechanics_tokens"]
                    if getattr(model, "use_mechanics_features", False)
                    else None
                ),
                mechanics_mask=(
                    batch["mechanics_mask"]
                    if getattr(model, "use_mechanics_features", False)
                    else None
                ),
            )
            teacher_logit = logits.gather(1, batch["teacher_actions"].unsqueeze(1))
            better = ((logits > teacher_logit) & batch["candidate_mask"].bool()).sum(dim=1)
            ranks.extend((better + 1).cpu().tolist())
    return np.asarray(ranks, dtype=np.int64)


def distill_policy(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_samples: list[DistillationSample],
    config: DistillationConfig,
    *,
    device: str,
    val_samples: list[DistillationSample] | None = None,
) -> dict[str, float]:
    """Train `model` to imitate the teacher for `config.epochs` epochs.

    When `val_samples` is given, evaluates honest validation agreement after EVERY
    epoch, tracks the best epoch by the configured validation metric
    (`DistillationConfig.checkpoint_metric`), and restores that best state_dict into
    `model` before returning -- this is the early-stopping / best-epoch-selection lever
    described in the module docstring's overfitting rationale.
    `config.early_stopping_patience > 0` additionally stops training once that many
    consecutive epochs pass with no improvement over the best. When `val_samples` is
    None (the default caller shape), behavior is exactly as before this argument
    existed: fixed `config.epochs`, no extra evaluation cost, no restore.

    Training-side retention levers (all default off, so defaults reproduce history):
    `balance_action_count_bins` oversamples rare large-branching turns, and
    `hard_example_weight` upweights samples whose teacher action currently ranks past
    `hard_example_rank` -- the only samples a shortlist gate can actually fail.
    """

    if not train_samples:
        raise ValueError("cannot train from an empty distillation sample set")
    rng = np.random.default_rng(config.seed)
    sampling_probabilities = (
        _action_count_sampling_probabilities(train_samples)
        if config.balance_action_count_bins
        else None
    )
    losses: list[float] = []
    grad_norms: list[float] = []
    best_epoch = 0
    best_val_accuracy = -1.0
    # Checkpoint selection. "loss" mode keeps the historical key (validation
    # cross-entropy primary -- see DistillationConfig.checkpoint_metric for why
    # "recall_at_k" exists as the alternative). Loss breaks ties in both modes.
    best_val_key = (float("-inf"), -1.0, -1.0)
    best_val_teacher_probability = 0.0
    best_val_loss = float("inf")
    best_val_recall: dict[str, float] = {}
    best_state_dict: dict[str, torch.Tensor] | None = None
    epochs_since_improvement = 0
    val_history: list[dict[str, float]] = []
    epochs_run = 0
    for epoch in range(1, config.epochs + 1):
        if sampling_probabilities is None:
            indices = rng.permutation(len(train_samples))
        else:
            # With-replacement draws weighted toward the rare large-branching bins.
            indices = rng.choice(
                len(train_samples), size=len(train_samples), replace=True,
                p=sampling_probabilities,
            )
        if config.hard_example_weight > 0:
            # Ranks under the CURRENT model; epoch 1's are from random weights and are
            # effectively noise -- harmless, and correct from epoch 2 on.
            epoch_ranks = _teacher_ranks(
                model, train_samples, batch_size=config.batch_size, device=device
            )
            sample_weights = np.where(
                epoch_ranks > config.hard_example_rank,
                1.0 + config.hard_example_weight,
                1.0,
            )
        else:
            sample_weights = None
        model.train()
        for start in range(0, len(indices), config.batch_size):
            selected = indices[start : start + config.batch_size]
            batch = _tensor_batch([train_samples[int(index)] for index in selected], device)
            logits, _values = model(
                batch["state_indices"],
                batch["state_scalars"],
                batch["history_scalars"],
                batch["move_indices"],
                batch["target_indices"],
                batch["switch_species_indices"],
                batch["flags"],
                batch["candidate_mask"],
                meta_scalars=batch["meta_scalars"] if model.use_meta_features else None,
                information_indices=(
                    batch["information_indices"] if model.use_information_features else None
                ),
                information_scalars=(
                    batch["information_scalars"] if model.use_information_features else None
                ),
                tactical_features=(
                    batch["tactical_features"] if model.use_tactical_features else None
                ),
                mechanics_tokens=(
                    batch["mechanics_tokens"]
                    if getattr(model, "use_mechanics_features", False)
                    else None
                ),
                mechanics_mask=(
                    batch["mechanics_mask"]
                    if getattr(model, "use_mechanics_features", False)
                    else None
                ),
            )
            if sample_weights is None and not config.soft_targets:
                loss = nn.functional.cross_entropy(logits, batch["teacher_actions"])
            else:
                # Per-sample loss so heterogeneous rows can be combined: hard CE as the
                # base (and the fallback for samples without stored scores), soft CE
                # where schema-v2.x scores exist, then hard-example weights on top.
                per_sample = nn.functional.cross_entropy(
                    logits, batch["teacher_actions"], reduction="none"
                )
                if config.soft_targets:
                    # Subset FIRST: building the distribution over rows without stored
                    # scores would run softmax on all -inf inputs, and even where()
                    # masking cannot stop those NaNs from leaking into gradients.
                    has_scores = torch.isfinite(batch["search_scores"]).any(dim=1)
                    if bool(has_scores.any()):
                        sub_logits = logits[has_scores]
                        sub_mask = batch["candidate_mask"][has_scores].bool()
                        targets = soft_target_distribution(
                            batch["search_scores"][has_scores],
                            sub_mask,
                            config.soft_target_temperature,
                        )
                        log_probs = nn.functional.log_softmax(sub_logits, dim=1)
                        log_probs = log_probs.masked_fill(~sub_mask, 0.0)
                        soft = -(targets * log_probs).sum(dim=1)
                        index = torch.nonzero(has_scores, as_tuple=True)[0]
                        per_sample = per_sample.clone()
                        per_sample[index] = soft
                if sample_weights is not None:
                    # Weighted mean, not plain mean: keeps the loss scale comparable to
                    # the unweighted case while spending proportionally more gradient
                    # on the samples that actually fail the shortlist gate.
                    weights = torch.as_tensor(
                        sample_weights[selected], dtype=torch.float32, device=device
                    )
                    loss = (per_sample * weights).sum() / weights.sum()
                else:
                    loss = per_sample.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            losses.append(float(loss.detach()))
            grad_norms.append(float(grad_norm.detach()))
        epochs_run = epoch

        if val_samples is not None:
            # evaluate_agreement puts the model in eval mode internally; return to
            # train mode afterward so the next epoch's dropout/BN behavior is
            # unaffected by having just evaluated.
            epoch_metrics = evaluate_agreement(
                model, val_samples, batch_size=config.batch_size, device=device
            )
            model.train()
            val_history.append(
                {
                    "epoch": float(epoch),
                    "accuracy": epoch_metrics["accuracy"],
                    "teacher_probability": epoch_metrics["teacher_probability"],
                    "loss": epoch_metrics["loss"],
                    # Optional so legacy evaluate_agreement stand-ins keep working.
                    "recall_at_10": epoch_metrics.get("recall_at_10"),
                    "teacher_rank_p95": epoch_metrics.get("teacher_rank_p95"),
                }
            )
            if config.checkpoint_metric == "recall_at_k":
                metric_name = f"recall_at_{config.checkpoint_recall_k}"
                candidate_key = (
                    epoch_metrics[metric_name],
                    -epoch_metrics["loss"],
                    epoch_metrics["teacher_probability"],
                )
            else:
                candidate_key = (
                    -epoch_metrics["loss"],
                    epoch_metrics["teacher_probability"],
                    epoch_metrics["accuracy"],
                )
            if candidate_key > best_val_key:
                best_val_key = candidate_key
                best_val_accuracy = epoch_metrics["accuracy"]
                best_val_teacher_probability = epoch_metrics["teacher_probability"]
                best_val_loss = epoch_metrics["loss"]
                best_val_recall = {
                    name: value
                    for name, value in epoch_metrics.items()
                    if name.startswith("recall_at_")
                }
                best_epoch = epoch
                best_state_dict = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
                if (
                    config.early_stopping_patience > 0
                    and epochs_since_improvement >= config.early_stopping_patience
                ):
                    break

    result: dict[str, float] = {
        "loss": sum(losses) / len(losses),
        "grad_norm": sum(grad_norms) / len(grad_norms),
        "samples": float(len(train_samples)),
        "epochs": float(config.epochs),
    }
    if val_samples is not None:
        assert best_state_dict is not None  # at least one epoch always runs
        model.load_state_dict(best_state_dict)
        result["best_epoch"] = float(best_epoch)
        result["best_val_accuracy"] = best_val_accuracy
        result["best_val_teacher_probability"] = best_val_teacher_probability
        result["best_val_loss"] = best_val_loss
        for name, value in best_val_recall.items():
            result[f"best_val_{name}"] = value
        if config.checkpoint_metric == "recall_at_k":
            result["checkpoint_metric"] = config.checkpoint_metric
            result["checkpoint_recall_k"] = float(config.checkpoint_recall_k)
        result["epochs_run"] = float(epochs_run)
        result["val_history"] = val_history
    return result


def turn_bucket(turn: int) -> str:
    """Same early/mid/late split the counterfactual collector uses, so they compare."""

    if turn <= 2:
        return "early"
    return "mid" if turn <= 4 else "late"


def recall_at_k(
    model: nn.Module,
    samples: list[DistillationSample],
    *,
    ks: Sequence[int],
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    """Per-decision hit flags for "is the teacher's action inside the model's top K".

    This is the metric that decides whether the network may replace the hand-written
    candidate selector: search only ever sees the shortlist, so an action ranked K+1 is
    gone no matter how good the search that follows is.

    Returned `hits[k]` is a list aligned with `samples`, so a caller can cluster by team
    (`vgc.evaluation.clustered_interval`) instead of pooling decisions that share a
    roster. `trivial` counts decisions with at most K legal actions, where retention is
    automatic -- a collection weighted toward those inflates recall without the model
    getting better, so it is reported rather than hidden.
    """

    if not samples:
        raise ValueError("cannot evaluate an empty distillation sample set")
    if not ks or any(k < 1 for k in ks):
        raise ValueError("ks must be positive")
    model.eval()
    hits: dict[int, list[bool]] = {int(k): [] for k in ks}
    ranks: list[int] = []
    ranked_indices: list[list[int]] = []
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            chunk = samples[start : start + batch_size]
            batch = _tensor_batch(chunk, device)
            logits, _values = model(
                batch["state_indices"],
                batch["state_scalars"],
                batch["history_scalars"],
                batch["move_indices"],
                batch["target_indices"],
                batch["switch_species_indices"],
                batch["flags"],
                batch["candidate_mask"],
                meta_scalars=batch["meta_scalars"] if model.use_meta_features else None,
                information_indices=(
                    batch["information_indices"] if model.use_information_features else None
                ),
                information_scalars=(
                    batch["information_scalars"] if model.use_information_features else None
                ),
                tactical_features=(
                    batch["tactical_features"] if model.use_tactical_features else None
                ),
                mechanics_tokens=(
                    batch["mechanics_tokens"]
                    if getattr(model, "use_mechanics_features", False)
                    else None
                ),
                mechanics_mask=(
                    batch["mechanics_mask"]
                    if getattr(model, "use_mechanics_features", False)
                    else None
                ),
            )
            teacher = batch["teacher_actions"]
            # Rank of the teacher's action = how many legal candidates outscore it.
            # Strict `>` means an exact logit tie counts in the teacher's favor -- the
            # optimistic reading, and unreachable in float32 practice except for
            # genuinely identical candidate rows.
            teacher_logit = logits.gather(1, teacher.unsqueeze(1))
            better = ((logits > teacher_logit) & batch["candidate_mask"].bool()).sum(dim=1)
            ranks.extend((better + 1).cpu().tolist())
            # Legal candidates best-first, for callers that replay the deployed
            # safety-slotted selector (`simulate_guided_hits`). Padded slots are forced
            # to -inf so they sort strictly after every legal action; a model that does
            # not mask its own padded logits therefore still yields a legal ranking.
            masked_logits = logits.masked_fill(~batch["candidate_mask"].bool(), float("-inf"))
            order = torch.argsort(masked_logits, dim=-1, descending=True)
            for row, sample in enumerate(chunk):
                ranked_indices.append(order[row, : len(sample.candidates)].cpu().tolist())
            for k in hits:
                hits[k].extend((better < k).cpu().tolist())
    rank_array = np.asarray(ranks, dtype=np.float64)
    return {
        "samples": len(samples),
        "hits": {str(k): value for k, value in hits.items()},
        "recall": {str(k): float(np.mean(value)) for k, value in hits.items()},
        # Legal candidate indices per sample, best-first (see the masking note above).
        "ranked_indices": ranked_indices,
        # legal_action_count defaults to 0 on datasets collected before it was recorded;
        # that means "unknown", not "trivially retained", so it must not be counted here.
        "trivial": {
            str(k): int(sum(1 for s in samples if 0 < s.legal_action_count <= k)) for k in hits
        },
        "teacher_rank": {
            "mean": float(rank_array.mean()),
            "median": float(np.median(rank_array)),
            "p95": float(np.percentile(rank_array, 95)),
            "max": float(rank_array.max()),
        },
    }


def simulate_guided_hits(
    samples: list[DistillationSample],
    ranked_indices: list[list[int]],
    *,
    ks: Sequence[int],
    safety_slots: int = 4,
) -> dict[int, list[bool | None]]:
    """Replay the deployed safety-slotted shortlist offline, per decision.

    For every sample this runs `vgc.rl.guided_selection`'s selection -- the exact code
    the hybrid player uses at battle time -- against the recorded myopic ranks and tag
    flags, then reports whether the teacher's action survived it. The result is a
    DIFFERENT number than the raw top-K: at K=10 up to four slots belong to heuristic
    safety picks, so a network can rank the teacher's action 8th and still lose it, or
    rank it 40th and keep it via the myopic-leader slot.

    Returns hits aligned with ``samples`` per K; ``None`` marks samples whose dataset
    predates the guidance metadata and therefore cannot be replayed (they must be
    reported as unknown, never silently counted either way).
    """

    if len(samples) != len(ranked_indices):
        raise ValueError("ranked_indices must align with samples")
    hits: dict[int, list[bool | None]] = {int(k): [] for k in ks}
    for sample, ranked in zip(samples, ranked_indices, strict=True):
        n_candidates = len(sample.candidates)
        has_metadata = (
            sample.candidate_myopic_ranks is not None and sample.candidate_tags is not None
        )
        legal_ranked = [index for index in dict.fromkeys(ranked) if 0 <= index < n_candidates]
        for k in hits:
            if k >= n_candidates:
                # Every legal action fits inside the budget; retention is automatic for
                # any correct selector.
                hits[k].append(True)
                continue
            if not has_metadata:
                hits[k].append(None)
                continue
            tags_by_index = {
                index: frozenset(
                    tag
                    for column, flag in zip(SAFETY_TAG_COLUMNS, sample.candidate_tags[index])
                    if flag
                    for tag in SAFETY_COLUMN_TAGS[column]
                )
                for index in range(n_candidates)
            }
            selected, _safety = select_guided_candidate_indices(
                n_candidates,
                ranked_indices=legal_ranked,
                myopic_position={
                    index: int(sample.candidate_myopic_ranks[index])
                    for index in range(n_candidates)
                    if sample.candidate_myopic_ranks[index] >= 0
                },
                tags_by_index=tags_by_index,
                cutoff=k,
                safety_slots=safety_slots,
            )
            hits[k].append(sample.teacher_action_index in selected)
    return hits
