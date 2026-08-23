"""Index-level core of the neural-guided candidate selection, shared by live play and
offline replay.

The deployed hybrid shortlist is NOT the network's raw top-K: up to half of the budget
is reserved for deterministic heuristic "safety" picks (myopic leader, then the best
available switch, Protect, control move, all-out offense, and non-Protect line).
Measuring recall against the raw top-K therefore answers a different question than the
one deployment asks. Extracting the algorithm here lets `vgc.rl.search_guidance` (live,
over ``ScoredOrder`` objects) and `offline.evaluate_shortlist_recall` (offline, over
recorded per-candidate metadata) execute literally the same selection logic.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

# Column order for the compact per-candidate tag matrix stored on distillation samples.
SAFETY_TAG_COLUMNS = ("switch", "protect", "control", "offense", "non_protect")
CONTROL_TAGS = frozenset({"speed_control", "setup", "screen", "action_denial"})
# Raw order tags each column stands for. Two columns are coarse: "control" means any of
# CONTROL_TAGS and "offense" means a joint order attacking with both slots
# ("double_attack"). Build and replay must both go through this table so the stored
# matrix and the live tag semantics can never drift apart.
SAFETY_COLUMN_TAGS: dict[str, frozenset[str]] = {
    "switch": frozenset({"switch"}),
    "protect": frozenset({"protect"}),
    "control": CONTROL_TAGS,
    "offense": frozenset({"double_attack"}),
    "non_protect": frozenset({"non_protect"}),
}
_SAFETY_GROUPS = (
    ("switch", frozenset({"switch"})),
    ("protect", frozenset({"protect"})),
    ("control", CONTROL_TAGS),
    ("offense", frozenset({"double_attack"})),
    ("non_protect", frozenset({"non_protect"})),
)


def select_guided_candidate_indices(
    n_candidates: int,
    *,
    ranked_indices: Sequence[int],
    myopic_position: Mapping[int, int],
    tags_by_index: Mapping[int, frozenset[str]],
    cutoff: int,
    safety_slots: int,
) -> tuple[list[int], list[dict[str, str]]]:
    """Return exactly ``cutoff`` selected indices plus the safety-slot audit trail.

    Semantics mirror the deployed selector:
      1. reserve ``min(safety_slots, cutoff // 2)`` slots for deterministic heuristic
         picks -- the myopic leader first, then the highest-myopic action covering each
         still-missing strategic category (categories already present among the selected
         safety picks are skipped, not re-filled);
      2. fill the remaining slots in the network's rank order;
      3. fill anything left over from the myopic order, so an undersized ranking can
         never shrink the search budget.
    """

    cutoff = min(n_candidates, max(1, int(cutoff)))
    safety_budget = min(max(0, int(safety_slots)), cutoff // 2)

    def _position(index: int) -> int:
        return myopic_position.get(index, n_candidates)

    myopic_order = sorted(range(n_candidates), key=_position)
    selected: list[int] = []
    selected_set: set[int] = set()
    safety: list[dict[str, str]] = []

    def add_safety(index: int | None, label: str) -> None:
        if index is None or index in selected_set or len(safety) >= safety_budget:
            return
        selected.append(index)
        selected_set.add(index)
        safety.append({"reason": label, "index": str(index)})

    def add(index: int) -> bool:
        if index in selected_set:
            return False
        selected.append(index)
        selected_set.add(index)
        return True

    if safety_budget:
        add_safety(myopic_order[0], "heuristic_top")
        for label, wanted in _SAFETY_GROUPS:
            if len(safety) >= safety_budget:
                break
            if any(tags_by_index.get(index, frozenset()) & wanted for index in selected):
                continue
            match = next(
                (
                    index
                    for index in myopic_order
                    if tags_by_index.get(index, frozenset()) & wanted
                ),
                None,
            )
            add_safety(match, label)

    for index in ranked_indices:
        if len(selected) >= cutoff:
            break
        add(int(index))
    for index in myopic_order:
        if len(selected) >= cutoff:
            break
        add(index)

    return selected, safety
