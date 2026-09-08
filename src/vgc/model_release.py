"""Require exact, current evidence before a learned model can play publicly.

This validates an evidence bundle, not a boolean in the model inventory. Local
experiments stay available even when no model qualifies for a public release.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from vgc.artifact_evidence import decision_evidence, file_sha256, read_verified_json

RELEASE_SCHEMA = "vgc-model-release-v1"
MINIMUM_PAIRED_GAMES = 1500
MINIMUM_TEAMS = 20
MINIMUM_RECALL = 0.98
MAXIMUM_STRENGTH_MARGIN = 0.02


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("release measurement must be a finite number")
    result = float(value)
    _require(math.isfinite(result), "release measurement must be finite")
    return result


def validate_model_release(
    release_path: Path,
    checkpoint: Path,
    config,
    *,
    safety_slots: int,
) -> dict:
    """Check bound reports and their actual measurements, never only `passed`.

    All report references use a content hash; relative paths are interpreted from
    the release file's directory. A release protocol must be frozen before final
    evaluation, and all reports identify that exact protocol. Tests of randomness
    and statistical power remain part of protocol review, not inferred here.
    """
    from vgc.rl.opponents import RL_ARCHITECTURE_VERSION

    import torch

    release_path = Path(release_path).resolve()
    base = release_path.parent
    release = json.loads(release_path.read_text())
    _require(release.get("schema") == RELEASE_SCHEMA, "unsupported model release schema")
    checkpoint_hash = file_sha256(checkpoint)
    _require(release.get("checkpoint_sha256") == checkpoint_hash, "release model mismatch")
    current = decision_evidence(config, safety_slots=safety_slots)
    _require(release.get("decision_evidence") == current, "stale decision code/data/settings")
    protocol_ref = release["protocol"]
    protocol = read_verified_json(protocol_ref, base=base)
    _require(protocol.get("schema") == "vgc-final-protocol-v1", "missing final protocol")
    _require(protocol.get("partition_role") == "final_test", "development data cannot approve")
    _require(bool(protocol.get("frozen_commit")), "protocol must identify its frozen commit")
    _require(bool(protocol.get("team_sha256")), "protocol must identify final test teams")
    minimum_pairs = _number(protocol["minimum_paired_games"])
    minimum_teams = _number(protocol["minimum_teams"])
    _require(minimum_pairs >= MINIMUM_PAIRED_GAMES, "protocol weakens paired-game requirement")
    _require(minimum_teams >= MINIMUM_TEAMS, "protocol weakens team requirement")
    _require(_number(protocol["recall_threshold"]) >= MINIMUM_RECALL, "weak recall threshold")
    margin = _number(protocol["noninferiority_margin"])
    _require(0 < margin <= MAXIMUM_STRENGTH_MARGIN, "weak strength margin")
    speed_fraction = _number(protocol["minimum_time_reduction"])
    _require(0 < speed_fraction < 1, "protocol must require a measured time improvement")
    deadline_ms = _number(protocol["maximum_decision_ms"])
    _require(deadline_ms > 0, "protocol must specify its decision-time budget")

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    _require(payload.get("architecture") == RL_ARCHITECTURE_VERSION, "incompatible model")
    _require(payload.get("use_mechanics_features") is True, "model lacks mechanics inputs")
    trained = payload.get("trained_outputs", {})
    _require(trained.get("action_preferences") is True, "action training provenance missing")
    if config.use_value_head:
        _require(trained.get("winning_chance") is True, "winning-chance output is untrained")
    audit_ref = release["training_audit"]
    audit = read_verified_json(audit_ref, base=base)
    _require(audit.get("verdict") == "PASS", "training audit did not pass")
    _require(not audit.get("errors"), "training audit contains errors")
    _require(
        payload.get("training_audit_sha256") == audit_ref.get("sha256"),
        "training audit is not bound to the checkpoint",
    )
    final_teams = set(protocol["team_sha256"])
    exposed_teams = set(payload.get("development_team_sha256", []))
    _require(bool(exposed_teams), "checkpoint development-team provenance missing")
    _require(not final_teams & exposed_teams, "final teams appeared during development")

    reports = {}
    for name in ("recall", "strength", "timing", "local_smoke", "restore"):
        report = read_verified_json(release[name], base=base)
        _require(report.get("checkpoint_sha256") == checkpoint_hash, f"{name}: wrong model")
        _require(report.get("decision_evidence") == current, f"{name}: wrong decision evidence")
        _require(report.get("protocol_sha256") == protocol_ref["sha256"], f"{name}: wrong protocol")
        _require(report.get("partition_role") == "final_test", f"{name}: not final evidence")
        reports[name] = report

    recall = reports["recall"]
    k = str(config.search_our_candidates)
    metric = recall["by_k_guided"][k]
    _require(_number(metric["clustered_lower_bound"]) >= MINIMUM_RECALL, "recall too low")
    _require(_number(metric["teams"]) >= minimum_teams, "too few recall teams")
    _require(metric.get("decisions_without_guidance_metadata") == 0, "incomplete recall")
    _require(recall.get("guided_safety_slots") == safety_slots, "recall safety budget mismatch")

    strength = reports["strength"]
    _require(strength.get("mode") == "hybrid", "strength evidence is not hybrid")
    _require(strength.get("search_k") == config.search_our_candidates, "strength width mismatch")
    _require(strength.get("safety_slots") == safety_slots, "strength safety budget mismatch")
    paired = strength["paired_strength"]
    _require(_number(paired["paired_games"]) >= minimum_pairs, "too few paired games")
    _require(_number(paired["teams"]) >= minimum_teams, "too few strength teams")
    _require(
        _number(paired["team_clustered_difference_interval"][0]) > -margin,
        "strength is not certified within the allowed margin",
    )
    _require(strength.get("fallbacks") == {"shadow": 0, "hybrid": 0}, "strength fallback")
    for arm in ("shadow", "hybrid"):
        retained = strength[arm]["retention"][f"guided_at_{k}"]
        _require(
            _number(retained["team_clustered_interval"][0]) >= MINIMUM_RECALL,
            f"{arm}: player-visited retention too low",
        )
    timing = reports["timing"]
    _require(timing.get("same_hardware") is True, "timing hardware differs")
    _require(timing.get("equal_budget") is True, "timing budgets differ")
    _require(
        _number(timing["time_reduction_lower_bound"]) >= speed_fraction,
        "time improvement is not demonstrated",
    )
    _require(_number(timing["maximum_decision_ms"]) <= deadline_ms, "decision timer exceeded")
    smoke = reports["local_smoke"]
    _require(_number(smoke["games"]) >= 2 and smoke.get("fallbacks") == 0, "local smoke failed")
    _require(smoke.get("completed") is True, "local smoke incomplete")
    restored = reports["restore"]
    _require(restored.get("loaded") is True, "restore did not load")
    _require(restored.get("backup_verified") is True, "backup not verified")
    _require(restored.get("restored_checkpoint_sha256") == checkpoint_hash, "restore differs")
    return release


def enforce_model_release_for_cli(release_path, checkpoint, config, *, safety_slots: int) -> None:
    if release_path is None:
        raise SystemExit("public play requires --model-release with current passing evidence")
    try:
        validate_model_release(release_path, checkpoint, config, safety_slots=safety_slots)
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:
        raise SystemExit(f"model release refused: {exc}") from exc
