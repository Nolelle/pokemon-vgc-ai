"""LLM team-preview advisor: ask for a bring-4 and a lead pair, keep the heuristic on failure.

`choose_preview(battle, config, heuristic_order)` is the only entry point. The caller has
already computed the heuristic order (`vgc.team_preview.build_team_order`), so every
failure here (spend cap, no key, late, refusal, malformed/invalid answer, short clock)
just returns that heuristic order unchanged. It never raises.

Blindness: the prompt never shows the heuristic's own pick. The model sees only public
information: our six (exact, we own them), our written team plan, their six species with
set-prior GUESSES, and the predicted opponent bring/leads (`vgc.preview_predict`) labelled
as estimates.

Answer: `{plan, bring: 4 distinct names from our six, leads: 2 of those 4, why}` under a
strict JSON schema whose names are an enum of our six. The harness runs the validator
below (`invalid_answer` status in the call log), then the answer is mapped to the same
`"/team XXXX"` string `build_team_order` returns: 1-based indices into `battle.team`,
leads first, then the two back Pokemon.
"""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path
from typing import Any

from vgc.clock import cancelled, time_left
from vgc.damage import to_id
from vgc.decision_trace import record_note
from vgc.llm.config import LEVELS
from vgc.llm.facts import (
    PLANS_DIR,
    _mon_name,
    _move_reference_lines,
    _opponent_guess_lines,
    _our_team_lines,
    _safe,
    load_team_plan,
)
from vgc.llm.harness import advise
from vgc.llm.packet import new_request_id
from vgc.llm.types import ContextPacket
from vgc.models import PolicyConfig

PICK_COUNT = 4
LEAD_COUNT = 2
MAX_LIVE_LEVEL = "medium"

PREVIEW_RULES = """\
You advise a Pokemon Showdown bot at team preview in gen9championsvgc2026regmc (Champions \
VGC, Reg M-C): doubles, level 50, best of one. Each side previews six Pokemon, then secretly \
brings FOUR; the first two listed lead (the left and right active slots), the other two \
start in the back.
- Stats use Stat Points (max 32 per stat, 66 total), not EVs. Items and moves differ from \
normal VGC: only the items and moves shown in this packet are real here.
- Each player may Mega Evolve ONE Pokemon per game (holding its Mega Stone).
- Our six are known exactly. Their six species are known; their sets, items, Speed and \
which four they bring are NOT known. Everything marked GUESS or "estimate" comes from ladder \
replay statistics, not from this game.
- Answer with the JSON schema given: "plan" = at most 2 short sentences, "bring" = exactly \
4 distinct names from our six, "leads" = 2 of those 4 in slot order (left, right), "why" = \
one short sentence."""

PREVIEW_CHECKLIST = """\
How to think (the owner's method):
1. Read their likely plan from their six: speed control, weather/terrain, setup, Fake Out, \
and the one or two Pokemon on their side that can beat us.
2. Pick the four of ours that answer those threats and keep OUR game plan (below) working; \
leave behind Pokemon that are weak to their most likely leads.
3. Choose a lead pair that is good against both of their likely lead pairs and sets up the \
back two. Our written plan lists default leads; they are defaults, not orders."""

_MIN_PROB = 0.02  # do not print predicted brings/leads below this probability


def _packed_species(path: Path) -> frozenset[str]:
    """Species ids in a packed team file (nickname|species|item|...]...)."""
    out: set[str] = set()
    try:
        text = path.read_text().strip()
    except OSError:
        return frozenset()
    for mon in text.split("]"):
        fields = mon.split("|")
        if len(fields) > 1:
            out.add(to_id(fields[1] or fields[0]) or "")
    return frozenset(out - {""})


_auto_plans: dict[frozenset[str], str] = {}


def plan_for_team(battle: Any, config: PolicyConfig) -> str:
    """Our team plan: `config.llm_team_plan` if set, else the owner team whose six species
    match `battle.team` (our own team, so this is public information). "" if none."""
    if config.llm_team_plan:
        return str(_safe(lambda: load_team_plan(config.llm_team_plan), "") or "")
    species = frozenset(
        to_id(str(getattr(m, "base_species", None) or m.species))
        for m in (battle.team or {}).values()
    )
    if species in _auto_plans:
        return _auto_plans[species]
    plan = ""
    for packed in sorted((PLANS_DIR.parent).glob("*.packed.txt")):
        if _packed_species(packed) == species:
            plan = str(_safe(lambda: load_team_plan(packed), "") or "")
            break
    _auto_plans[species] = plan
    return plan


def _predicted_lines(battle: Any, config: PolicyConfig) -> list[str]:
    """Predicted opponent bring-4s and lead pairs with probabilities (estimates)."""
    from vgc.preview_predict import (
        bring4_distribution,
        lead_distribution,
        predict_preview_hybrid,
    )

    opp = list(getattr(battle, "teampreview_opponent_team", None) or [])
    ours = list((battle.team or {}).values())
    candidates = predict_preview_hybrid(
        [to_id(m.species) for m in opp], [to_id(m.species) for m in ours], config
    )
    if not candidates:
        return []

    def names(idxs: tuple[int, ...]) -> str:
        return " + ".join(_mon_name(opp[i]) for i in idxs)

    lines = ["Likely brings (ESTIMATE; usage-driven, shares over all 15 possible brings):"]
    for subset, p in bring4_distribution(candidates)[:3]:
        if p >= _MIN_PROB:
            lines.append(f"- {round(100 * p)}%: {names(subset)}")
    lines.append("Likely lead pairs (ESTIMATE):")
    for pair, p in lead_distribution(candidates)[:3]:
        if p >= _MIN_PROB:
            lines.append(f"- {round(100 * p)}%: {names(pair)}")
    return lines


def preview_schema(names: list[str]) -> dict[str, Any]:
    pick = {"type": "array", "items": {"type": "string", "enum": names}}
    return {
        "type": "object",
        "properties": {"plan": {"type": "string"}, "bring": pick, "leads": pick,
                       "why": {"type": "string"}},
        "required": ["plan", "bring", "leads", "why"],
        "additionalProperties": False,
    }


def validate_answer(data: dict[str, Any], names: list[str]) -> str | None:
    """None if `data` is a legal bring/leads answer over `names`, else a short reason."""
    bring, leads = data.get("bring"), data.get("leads")
    if not isinstance(bring, list) or not isinstance(leads, list):
        return "bring/leads not lists"
    if not all(isinstance(x, str) for x in [*bring, *leads]):
        return "non-string name"
    if len(bring) != PICK_COUNT or len(set(bring)) != PICK_COUNT:
        return "bring is not 4 distinct names"
    if len(leads) != LEAD_COUNT or len(set(leads)) != LEAD_COUNT:
        return "leads are not 2 distinct names"
    if any(n not in names for n in bring):
        return "bring has a name outside our six"
    if any(n not in bring for n in leads):
        return "leads not in bring"
    return None


def order_string(names: list[str], bring: list[str], leads: list[str]) -> str:
    """'/team XXXX': 1-based indices into our six, leads first, then the back two."""
    ordered = [*leads, *(n for n in bring if n not in leads)]
    return "/team " + "".join(str(names.index(n) + 1) for n in ordered)


def build_preview_packet(
    battle: Any, config: PolicyConfig, names: list[str], request_id: str
) -> ContextPacket:
    from vgc.sets import set_priors_for

    priors = _safe(lambda: set_priors_for(config), {})
    parts = [PREVIEW_RULES, PREVIEW_CHECKLIST]
    team = _safe(lambda: _our_team_lines(battle), [])
    if team:
        parts.append("## OUR SIX (known exactly)\n" + "\n".join(team))
    plan = plan_for_team(battle, config)
    if plan.strip():
        parts.append("## OUR GAME PLAN\n" + plan.strip())
    ref = _safe(lambda: _move_reference_lines(battle, priors), [])
    if ref:
        parts.append("## MOVE REFERENCE (type, category, base power; engine data)\n"
                     + "\n".join(ref))
    foes = _safe(lambda: _opponent_guess_lines(battle, priors), [])
    turn = [
        "## THEIR SIX (species known; sets are GUESSES from ladder replay frequencies, "
        "% = share of games where seen)\n" + "\n".join(foes)
    ]
    predicted = _safe(lambda: _predicted_lines(battle, config), [])
    if predicted:
        turn.append("## PREDICTED OPPONENT PICKS\n" + "\n".join(predicted))
    turn.append("## TASK\nChoose which 4 of our six to bring and which 2 of those 4 lead. "
                "Our six: " + ", ".join(names) + ".")
    return ContextPacket(
        fixed_text="\n\n".join(parts),
        turn_text="\n\n".join(turn),
        option_ids=tuple(names),
        request_id=request_id,
        turn=0,
        kind="preview",
        schema=preview_schema(names),
        validate=lambda data: validate_answer(data, names),
    )


def apply_order_to_battle(battle: Any, order: str) -> None:
    """Make the battle object agree with the order we are sending (what
    `build_team_order` does for its own pick): selected-in-preview flags and the
    `_vgc_preview_plan` that turn scoring reads. A planned closer or default Mega that
    is not in the new bring is dropped rather than left pointing at a benched Pokemon."""
    team = list(battle.team.values())
    picked = [int(c) - 1 for c in order.removeprefix("/team ").strip()]
    for idx, mon in enumerate(team):
        mon._selected_in_teampreview = idx in picked
    plan = getattr(battle, "_vgc_preview_plan", None)
    if plan is None:
        return
    species = [to_id(team[i].species) for i in picked]

    def keep(sid: str | None) -> str | None:
        return sid if sid in species else None

    try:
        battle._vgc_preview_plan = dataclasses.replace(
            plan,
            our_closer_species=keep(plan.our_closer_species),
            default_mega_species=keep(plan.default_mega_species),
            picked_species=tuple(species),
            lead_species=tuple(species[:LEAD_COUNT]),
        )
    except (AttributeError, TypeError):
        pass


def _call(battle: Any, config: PolicyConfig, note: dict[str, Any]) -> str | None:
    """Return the validated LLM order string, or None (with note['fallback_reason'])."""
    from vgc.llm.proposer import _runtime, llm_config_for

    team = list((battle.team or {}).values())
    if len(team) != 6 or not getattr(battle, "teampreview_opponent_team", None):
        note["fallback_reason"] = "no_full_preview"
        return None
    names = [_mon_name(m) for m in team]
    if len(set(names)) != 6:
        note["fallback_reason"] = "ambiguous_names"
        return None
    level = config.llm_preview_level
    if level not in LEVELS or LEVELS.index(level) > LEVELS.index(MAX_LIVE_LEVEL):
        note["fallback_reason"] = f"bad_level:{level}"
        return None
    llm_cfg = llm_config_for(config)
    left = time_left()
    budget = float(config.llm_preview_budget_s)
    if left is not None:
        budget = min(budget, left - config.llm_safety_margin_s)
    note["budget_s"] = round(budget, 2)
    if budget < llm_cfg.min_call_budget_s:
        note["fallback_reason"] = "short_clock"
        return None
    prep = time.monotonic()
    packet = build_preview_packet(battle, config, names, new_request_id(0))
    note["prompt_chars"] = len(packet.full_text)
    budget -= time.monotonic() - prep
    client, meter = _runtime(config)
    advice = advise(packet, client, level, budget, meter, config.llm_log_path, config=llm_cfg)
    if advice is None or advice.data is None:
        note["fallback_reason"] = "no_advice"
        return None
    data = advice.data
    note.update(bring=list(data["bring"]), leads=list(data["leads"]), plan=advice.plan)
    return order_string(names, list(data["bring"]), list(data["leads"]))


def choose_preview(battle: Any, config: PolicyConfig, heuristic_order: str) -> str:
    """The LLM's order if it gave a valid one in time, else `heuristic_order`."""
    note: dict[str, Any] = {"used": False, "level": config.llm_preview_level,
                            "fallback_reason": ""}
    started = time.monotonic()
    order = heuristic_order
    try:
        llm_order = _call(battle, config, note)
        if llm_order is not None and cancelled():
            note["fallback_reason"] = "cancelled"
        elif llm_order is not None:
            apply_order_to_battle(battle, llm_order)
            order = llm_order
            note["used"] = True
    except Exception as exc:  # noqa: BLE001 - the heuristic order must always win
        note["fallback_reason"] = f"error:{type(exc).__name__}: {exc}"
    if not note["used"]:
        for key in ("bring", "leads", "plan"):
            note.setdefault(key, None)
    note["order"] = order
    note["heuristic_order"] = heuristic_order
    note["elapsed_s"] = round(time.monotonic() - started, 3)
    record_note("llm_preview", note)
    return order
