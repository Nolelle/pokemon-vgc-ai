"""One public-information boundary for exact teacher and guided search.

The caller supplies the observation a real player can see and its exact own packed
team.  This module owns reconstruction of possible hidden opponent states, exact
Showdown search over those states, probability-weighted combination, deterministic
search randomness, and cleanup of the temporary simulator resources.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import MutableMapping

from vgc.mechanics_state import snapshot_battle
from vgc.models import PolicyConfig
from vgc.rl.exact_search import ExactCandidateSelector, combine_belief_rankings
from vgc.rl.exact_search import search_joint_orders_exact
from vgc.rl.live_mirror import LiveExactMirror

PUBLIC_SEARCH_CONTRACT_VERSION = "public_search_v1"


def public_search_randomness_key(battle, own_packed_team: str, memory=None) -> str:
    """Return a stable key from decision-visible inputs, excluding battle names.

    Exact search samples future Showdown randomness.  Keying those samples from the
    public position makes equivalent observations comparable even when an evaluation
    harness gives their battles different identifiers.
    """

    memory_summary = memory.summary() if memory is not None else None
    payload = {
        "contract": PUBLIC_SEARCH_CONTRACT_VERSION,
        "observation": asdict(snapshot_battle(battle)),
        "own_packed_team": own_packed_team,
        "history": memory_summary,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()


def public_information_exact_search(
    battle,
    config: PolicyConfig,
    own_packed_team: str,
    *,
    memory=None,
    mirror: LiveExactMirror | None = None,
    candidate_selector: ExactCandidateSelector | None = None,
    audit: MutableMapping[str, object] | None = None,
):
    """Rank actions using only a public reconstruction of ``battle``.

    A supplied mirror may be reused across decisions to avoid repeatedly starting its
    worker.  The temporary battle root is always closed here; a mirror created here is
    also closed here.  Attached private simulator attributes are never inspected.
    """

    if not own_packed_team:
        raise ValueError("public search requires its exact packed own team")
    randomness_key = public_search_randomness_key(battle, own_packed_team, memory)
    owned_mirror = mirror is None
    public_mirror = mirror or LiveExactMirror(own_packed_team, config)
    root = None
    rankings = []
    try:
        beliefs = public_mirror.hypotheses(battle, memory)
        if audit is not None:
            audit.update(public_mirror.last_hypothesis_audit)
            audit.update(
                {
                    "information_contract": PUBLIC_SEARCH_CONTRACT_VERSION,
                    "randomness_key": randomness_key,
                }
            )
        for belief in beliefs:
            root = (
                public_mirror.rebase(root, battle, belief)
                if root is not None
                else public_mirror.build(battle, belief)
            )
            rankings.append(
                (
                    belief.weight,
                    search_joint_orders_exact(
                        root,
                        "p1",
                        config,
                        candidate_selector=candidate_selector,
                        randomness_key=randomness_key,
                    ),
                )
            )
        return combine_belief_rankings(rankings)
    finally:
        if root is not None:
            root.close()
        if owned_mirror:
            public_mirror.close()
