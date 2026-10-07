#!/usr/bin/env python
"""Hand-built setup positions: does the bot make the calls a good VGC player would, and why?

Each position pits one of the owner's game-plan teams (teams/owner/) against a real ladder
team (data/selfplay/mc_sheet_pool_v2/), plays team preview with explicit orders, optionally
scripts prelude turns (our choices are given; the opponent plays the shipped `vgc` bot from
its own view), and then asks five bot versions for OUR decision, all from our public view:

  A   exact judge, shipped defaults (public mirror + Showdown branches)
  B   A + field control with the measured plan value (fit weight 0.5)
  C   B + measured Tailwind / Trick Room speed payoff
  D0  live ladder bot: VgcPlayer(exact_judge_live=True), defaults otherwise
  D1  D0 + fast-search setter modelling, measured plan, fit weight 0.5, condition expiry

For each version it records the chosen order, the top-3 orders with scores, whether the
expected setup call was searched and where it ranked.  This is a diagnostic for a human
VGC player to read, not a test suite.  Width for A-C: search_our_candidates=10,
search_opp_candidates=8, exact_search_future_samples=2.  D0/D1 keep the live judge's own
width (top-6 of the fast search, 4 opponent replies, 2 futures); only its wall-clock cap is
raised so a slow ask is never silently replaced by the fast pick.

    PYTHONPATH=$PWD/src .venv/bin/python offline/setup_positions.py
    PYTHONPATH=$PWD/src .venv/bin/python offline/setup_positions.py --only 1,8 --versions A,D0
"""

from __future__ import annotations

import argparse
import contextvars
import json
import sys
import time
import traceback
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from poke_env.battle.move import Move  # noqa: E402
from poke_env.battle.pokemon import Pokemon  # noqa: E402

from vgc.actions import _describe_single, describe_order  # noqa: E402
from vgc.condition_clock import CLOCK_ATTRIBUTE, remaining_turns  # noqa: E402
from vgc.config import FORMAT_ID  # noqa: E402
from vgc.models import PolicyConfig  # noqa: E402
from vgc.own_team import apply_own_spreads  # noqa: E402
from vgc.rl.agents import make_direct_agent  # noqa: E402
from vgc.rl.env import DirectBattle, SimWorker  # noqa: E402
from vgc.rl.public_search import public_information_exact_search  # noqa: E402
from vgc.team_scope import bind_own_team  # noqa: E402

OWNER_DIR = REPO_ROOT / "teams" / "owner"
POOL_DIR = REPO_ROOT / "data" / "selfplay" / "mc_sheet_pool_v2"

VERSIONS = ("A", "B", "C", "D0", "D1")


# --- scenario table ----------------------------------------------------------------------


@dataclass
class Position:
    pid: int
    key: str
    own: str  # teams/owner/<own>.packed.txt
    opp: str  # mc_sheet_pool_v2 team id, e.g. "045"
    own_order: str  # "team XXXX": digits are 1-based slots in the packed file, leads first
    opp_order: str
    own_leads: tuple[str, str]  # expected active species after preview (ids, sorted)
    opp_leads: tuple[str, str]
    title: str
    expected: str  # what a good player does, in words
    # Verdict spec: `forbid` moves make the call wrong; `all` moves present = right; if
    # none match the call is arguable when `soft` else wrong.  `probe` = the moves whose
    # best-ranked order we look up for the "setup rank" column (default: `all`).
    all: tuple[str, ...] = ()
    forbid: tuple[str, ...] = ()
    probe: tuple[str, ...] = ()
    soft: bool = False
    partial: tuple[str, ...] = ()  # playing any of these still makes the call arguable
    # `pressure`: species id that a "right" call must also attack (a targeted damaging move or a
    # spread move, ignoring `pressure_excl`); otherwise the call is only arguable.
    pressure: str = ""
    pressure_excl: tuple[str, ...] = ()
    prelude: tuple[str, ...] = ()  # our scripted choice strings; opponent plays the bot
    note: str = ""


POSITIONS: list[Position] = [
    Position(
        1, "hatterene_tr_vs_fast", "hatterene_tr", "045",
        "team 5243", "team 5146",
        ("hatterene", "indeedeef"), ("garchomp", "sneasler"),
        "hatterene_tr leads Indeedee-F + Hatterene vs team_045 (fast offense) Sneasler + Garchomp",
        "Follow Me + Trick Room",
        all=("followme", "trickroom"), partial=("trickroom",),
    ),
    Position(
        2, "hatterene_tr_vs_slow_tr", "hatterene_tr", "034",
        "team 5243", "team 3452",
        ("hatterene", "indeedeef"), ("indeedeef", "torkoal"),
        "hatterene_tr leads Indeedee-F + Hatterene vs team_034 (slow Trick Room / sun) "
        "Indeedee-F + Torkoal",
        "NOT a blind Trick Room (it helps their slow side, and their Indeedee-F can "
        "reverse it); report what it does",
        forbid=("trickroom",), probe=("trickroom",),
    ),
    Position(
        3, "salamence_tw_vs_balance", "salamence_tw", "003",
        "team 1235", "team 5614",
        ("salamence", "sneasler"), ("incineroar", "rillaboom"),
        "salamence_tw leads Mega Salamence + Sneasler vs team_003 (mid-speed balance) "
        "Incineroar + Rillaboom",
        "Fake Out + Tailwind",
        all=("fakeout", "tailwind"), partial=("tailwind",),
    ),
    Position(
        4, "salamence_tw_vs_tr", "salamence_tw", "003",
        "team 1235", "team 2561",
        ("salamence", "sneasler"), ("farigiraf", "incineroar"),
        "salamence_tw leads Salamence + Sneasler vs team_003 Farigiraf + Incineroar "
        "(Trick Room setter)",
        "Pressure the Trick Room setter instead of a Tailwind Trick Room would reverse "
        "(NB Farigiraf's Armor Tail blocks Fake Out on both of them)",
        forbid=("tailwind",), probe=("tailwind",),
        pressure="farigiraf", pressure_excl=("fakeout",),
    ),
    Position(
        5, "gardevoir_psyspam_vs_balance", "gardevoir_psyspam", "000",
        "team 1236", "team 4253",
        ("gardevoir", "indeedeef"), ("incineroar", "metagross"),
        "gardevoir_psyspam leads Indeedee-F + Mega Gardevoir vs team_000 Incineroar + "
        "Metagross",
        "Follow Me + Expanding Force (Psychic Terrain is up from Indeedee's switch-in)",
        all=("followme", "expandingforce"), partial=("expandingforce",),
    ),
    Position(
        6, "blastoise_vs_garchomp_volcarona", "terrain_pulse_blastoise", "000",
        "team 1536", "team 3145",
        ("rillaboom", "blastoise"), ("garchomp", "volcarona"),
        "terrain_pulse_blastoise leads Rillaboom + Mega Blastoise vs team_000 Garchomp + "
        "Volcarona",
        "Fake Out + Shell Smash (or a strong Grassy Terrain Pulse); report",
        all=("fakeout", "shellsmash"), soft=True,
    ),
    Position(
        7, "psyspam_sand_vs_grassy", "psyspam_sand", "019",
        "team 1234", "team 1234",
        ("indeedee", "sneasler"), ("rillaboom", "incineroar"),
        "psyspam_sand leads Indeedee-M + Sneasler vs team_019 Rillaboom + Incineroar",
        "Expanding Force pressure; terrain fight (Rillaboom's Grassy Terrain vs Indeedee's "
        "Psychic Terrain)",
        all=("expandingforce",), soft=True,
    ),
    Position(
        8, "hatterene_tr_turn2", "hatterene_tr", "045",
        "team 5243", "team 5146",
        ("hatterene", "indeedeef"), ("garchomp", "sneasler"),
        "hatterene_tr vs team_045; scripted turn 1 = our Follow Me + Trick Room, opponent "
        "by the vgc bot; evaluate turn 2 with Trick Room up",
        "Do NOT Trick Room again (it would end our own room); attack",
        forbid=("trickroom",), probe=("trickroom",),
        prelude=("move 1, move 2",),
    ),
]


def read_packed(path: Path) -> str:
    return path.read_text().strip()


# --- order helpers -----------------------------------------------------------------------


def move_ids(order) -> list[str]:
    ids = []
    for single in (order.first_order, order.second_order):
        if isinstance(single.order, Move):
            ids.append(single.order.id)
    return ids


def _target_name(obs, target: int | None) -> str:
    if target is None:
        return ""
    opp = list(obs.opponent_active_pokemon or [])
    own = list(obs.active_pokemon or [])
    mon = None
    if target > 0 and target - 1 < len(opp):
        mon = opp[target - 1]
    elif target < 0 and -target - 1 < len(own):
        mon = own[-target - 1]
    return f">{mon.species}" if mon is not None else f">{target}"


def pretty(order, obs) -> str:
    """'indeedeef: followme & hatterene: trickroom' with targets resolved to species."""

    actives = list(obs.active_pokemon or [])
    parts = []
    for slot, single in enumerate((order.first_order, order.second_order)):
        mon = actives[slot] if slot < len(actives) else None
        who = mon.species if isinstance(mon, Pokemon) else f"slot{slot}"
        what = _describe_single(single)
        if "@" in what:
            base, _, tgt = what.partition("@")
            try:
                what = base + _target_name(obs, int(tgt))
            except ValueError:
                pass
        parts.append(f"{who}: {what}")
    return " & ".join(parts)


def pressures(order, obs, species_id: str, excluded: tuple[str, ...]) -> bool:
    """Does the order attack ``species_id`` (targeted damaging move, or a spread move)?"""

    foes = list(obs.opponent_active_pokemon or [])
    for single in (order.first_order, order.second_order):
        move = single.order
        if not isinstance(move, Move) or move.id in excluded:
            continue
        if move.category.name == "STATUS":
            continue
        target = single.move_target
        if target and target > 0 and target - 1 < len(foes) and foes[target - 1] is not None:
            if _sid(foes[target - 1]) == species_id:
                return True
        if getattr(move, "deduced_target", "") in ("allAdjacentFoes", "allAdjacent"):
            return True
    return False


def verdict(pos: Position, ids: list[str], pressured: bool = True) -> str:
    if any(move in ids for move in pos.forbid):
        return "wrong"
    if pos.all:
        if all(move in ids for move in pos.all):
            return "right"
        if pos.soft or any(move in ids for move in pos.partial):
            return "arguable"
        return "wrong"
    if pos.pressure and not pressured:
        return "arguable"
    return "right"


# --- board facts -------------------------------------------------------------------------


def board_facts(obs) -> dict[str, Any]:
    def mons(items):
        out = []
        for mon in items:
            if mon is None:
                out.append(None)
                continue
            out.append(
                {
                    "species": mon.species,
                    "hp_pct": round(100 * (mon.current_hp_fraction or 0), 1),
                    "item": mon.item,
                    "ability": mon.ability or (sorted(mon.possible_abilities)[0:1] or None),
                    "status": mon.status.name if mon.status else None,
                    "boosts": {k: v for k, v in mon.boosts.items() if v},
                    "moves": [m for m in (mon.moves or {})],
                }
            )
        return out

    clock = vars(obs).get(CLOCK_ATTRIBUTE) or {}
    conditions = []
    for key in clock:
        if key[0] not in ("weather", "field", "side"):
            continue
        kind = key[0]
        side = key[1] if kind == "side" else None
        effect = key[-1]
        conditions.append(
            {
                "kind": kind,
                "side": side,
                "effect": effect,
                "turns_left": remaining_turns(obs, kind, effect, side),
            }
        )
    return {
        "turn": obs.turn,
        "weather": {str(k.name): v for k, v in (obs.weather or {}).items()},
        "fields": {str(k.name): v for k, v in (obs.fields or {}).items()},
        "our_side": {str(k.name): v for k, v in (obs.side_conditions or {}).items()},
        "their_side": {str(k.name): v for k, v in (obs.opponent_side_conditions or {}).items()},
        "conditions": conditions,
        "ours": mons(obs.active_pokemon),
        "theirs": mons(obs.opponent_active_pokemon),
    }


# --- building a position -----------------------------------------------------------------


@dataclass
class Built:
    pos: Position
    worker: SimWorker
    battle: DirectBattle
    own_team: str
    opp_team: str
    history: list[tuple[str, list[str]]] = field(default_factory=list)
    prelude_log: list[str] = field(default_factory=list)

    @property
    def obs(self):
        return self.battle.battles["p1"]


def build(worker: SimWorker, pos: Position) -> Built:
    own_team = read_packed(OWNER_DIR / f"{pos.own}.packed.txt")
    opp_team = read_packed(POOL_DIR / f"team_{pos.opp}.packed.txt")
    tag = f"setup{pos.pid}"
    seed = [101 * pos.pid + k for k in (1, 2, 3, 4)]
    battle = DirectBattle.start(
        worker, tag, own_team, opp_team, seed=seed, own_team_spreads=True
    )
    built = Built(pos, worker, battle, own_team, opp_team)
    opp_agent = make_direct_agent("vgc", opp_team)

    def feed(lines_by_side: dict[str, list[str]]) -> None:
        for side in ("p1", "p2"):
            lines = list(lines_by_side.get(side) or [])
            built.history.append((side, lines))
            if side == "p2":
                opp_agent.observe(tag, lines)

    feed(battle.last_lines)
    # Team preview with explicit orders, then verify the leads.
    result = battle.step({"p1": pos.own_order, "p2": pos.opp_order})
    feed(result.lines)
    leads_own = tuple(sorted(_sid(m) for m in battle.battles["p1"].active_pokemon if m))
    leads_opp = tuple(sorted(_sid(m) for m in battle.battles["p1"].opponent_active_pokemon if m))
    want_own = tuple(sorted(_sid_str(s) for s in pos.own_leads))
    want_opp = tuple(sorted(_sid_str(s) for s in pos.opp_leads))
    if leads_own != want_own or leads_opp != want_opp:
        raise RuntimeError(
            f"position {pos.pid}: leads are {leads_own} vs {leads_opp}, wanted {want_own} vs "
            f"{want_opp}; fix the team order digits ({pos.own_order!r} / {pos.opp_order!r})"
        )
    for index, choice in enumerate(pos.prelude, start=1):
        to_move = battle.sides_to_move()
        if "p1" not in to_move:
            raise RuntimeError(f"position {pos.pid}: prelude turn {index} but p1 is not to move")
        battle.battles["p2"]._vgc_direct_root = battle
        battle.battles["p2"]._vgc_direct_side = "p2"
        opp_choice = opp_agent.choose(battle.battles["p2"])
        before = {"turn": battle.battles["p1"].turn}
        result = battle.step({"p1": choice, "p2": opp_choice})
        feed(result.lines)
        built.prelude_log.append(
            f"turn {before['turn']}: we played `{choice}`; the vgc bot played `{opp_choice}`"
        )
        if battle.ended:
            raise RuntimeError(f"position {pos.pid}: battle ended during the prelude")
    if "p1" not in battle.sides_to_move():
        raise RuntimeError(f"position {pos.pid}: p1 is not to move at the evaluation point")
    return built


def _sid(mon) -> str:
    return _sid_str(mon.species)


def _sid_str(text: str) -> str:
    text = "".join(ch for ch in text.lower() if ch.isalnum())
    for suffix in ("mega", "megax", "megay"):
        if text.endswith(suffix) and len(text) > len(suffix):
            text = text[: -len(suffix)]
            break
    return text


# --- asking a version --------------------------------------------------------------------


def base_config() -> PolicyConfig:
    return PolicyConfig(
        format_id=FORMAT_ID,
        search_our_candidates=10,
        search_opp_candidates=8,
        exact_search_future_samples=2,
    )


def version_config(name: str) -> PolicyConfig:
    base = base_config()
    field_cfg = replace(
        base,
        exact_search_field_control=True,
        exact_search_field_measured_plan=True,
        exact_search_field_fit_weight=0.5,
    )
    live = PolicyConfig(
        format_id=FORMAT_ID,
        exact_judge_live=True,
        exact_judge_budget_s=120.0,  # wall-clock cap only: never fall back silently
    )
    return {
        "A": base,
        "B": field_cfg,
        "C": replace(field_cfg, exact_search_field_measured_speed=True),
        "D0": live,
        "D1": replace(
            live,
            search_model_field_setters=True,
            exact_search_field_measured_plan=True,
            exact_search_field_fit_weight=0.5,
            search_condition_expiry=True,
        ),
    }[name]


def fresh_agent(built: Built, config: PolicyConfig):
    """A new own-side agent that has seen exactly the protocol history so far."""

    agent = make_direct_agent("vgc", built.own_team, config=config)
    tag = built.battle.battle_id
    for side, lines in built.history:
        if side == "p1":
            agent.observe(tag, lines)
    return agent


def row_of(entry, obs, rank: int) -> dict[str, Any]:
    b = entry.breakdown
    return {
        "rank": rank,
        "message": entry.order.message,
        "order": pretty(entry.order, obs),
        "moves": move_ids(entry.order),
        "score": round(float(entry.score), 2),
        "exchange": None if b.get("exchange_value") is None else round(float(b["exchange_value"]), 2),
        "myopic": None if b.get("myopic_score") is None else round(float(b["myopic_score"]), 2),
        "field_delta": None if b.get("field_delta") is None else round(float(b["field_delta"]), 2),
        "searched": b.get("searched"),
        "responses": [
            {"reply": str(text), "value": round(float(value), 1), "weight": round(float(weight), 2)}
            for text, value, weight in zip(
                b.get("response_orders") or [],
                b.get("response_values") or [],
                b.get("response_weights") or [],
            )
        ],
    }


def best_with(rows: list[dict[str, Any]], moves: tuple[str, ...]) -> dict[str, Any] | None:
    if not moves:
        return None
    for row in rows:
        if all(move in row["moves"] for move in moves):
            return row
    return None


def probe_messages(rows: list[dict[str, Any]], moves: tuple[str, ...], key: str) -> list[str]:
    """Messages of the 3 best orders (by ``key`` descending) that contain every probe move."""

    if not moves:
        return []
    hits = [row for row in rows if all(move in row["moves"] for move in moves)]
    hits.sort(key=lambda row: -(row.get(key) if row.get(key) is not None else -1e9))
    return [row["message"] for row in hits[:3]]


def forced_probe_exact(built, config, memory, ranked, rows, obs) -> dict[str, Any] | None:
    """Exact-search the probe orders even if the shortlist cut them, same futures as the ask.

    Answers "would the setup call have won had it been searched?"  The shortlist is widened
    by the probe orders only (``search_our_candidates`` grows to match), so every other
    order's value is identical to the main ask.
    """

    moves = built.pos.probe or built.pos.all
    wanted_probe = probe_messages(rows, moves, "myopic")
    if not wanted_probe:
        return None
    wanted = {row["message"] for row in rows if row["searched"]} | set(wanted_probe)
    wide = replace(config, search_our_candidates=len(wanted))

    def selector(ranked_entries, _config):
        searched = [e for e in ranked_entries if e.order.message in wanted]
        return searched, [e for e in ranked_entries if e.order.message not in wanted]

    def run():
        bind_own_team(built.own_team)
        return public_information_exact_search(
            obs, wide, built.own_team, memory=memory, candidate_selector=selector
        )

    again = contextvars.copy_context().run(run)
    again_rows = [row_of(entry, obs, index + 1) for index, entry in enumerate(again)]
    by_msg = {row["message"]: row for row in again_rows}
    if abs(by_msg[rows[0]["message"]]["score"] - rows[0]["score"]) > 1e-6:
        print(
            "  WARNING: the widened probe search did not reproduce the main ask's value "
            f"({by_msg[rows[0]['message']]['score']} vs {rows[0]['score']}); futures differ",
            flush=True,
        )
    return {
        "chosen_score": by_msg[rows[0]["message"]]["score"],
        "chosen_exchange": by_msg[rows[0]["message"]]["exchange"],
        "best_rank_overall": min(by_msg[m]["rank"] for m in wanted_probe),
        "probe": [by_msg[m] for m in wanted_probe],
    }


DICE_REROLLS = 5


def dice_check(built, config, memory, rows, obs, forced) -> dict[str, Any] | None:
    """Re-ask with ``DICE_REROLLS`` other random-future keys: how much is the pick dice?

    Each re-roll keeps the main ask's shortlist (plus the probe orders), only the sampled
    Showdown futures change.  Reports the winner among the ORIGINAL shortlist (what the bot
    would play) per re-roll and how often the best setup order would have beaten it.
    """

    import vgc.rl.public_search as public_search

    if DICE_REROLLS <= 0:
        return None
    original_key = public_search.public_search_randomness_key
    searched = {row["message"] for row in rows if row["searched"]}
    probe = [p["message"] for p in (forced or {}).get("probe", [])]
    wanted = searched | set(probe)
    wide = replace(config, search_our_candidates=len(wanted))

    def selector(ranked_entries, _config):
        picked = [e for e in ranked_entries if e.order.message in wanted]
        return picked, [e for e in ranked_entries if e.order.message not in wanted]

    winners: dict[str, int] = {}
    gaps: list[float] = []
    probe_wins = 0
    try:
        for salt in range(1, DICE_REROLLS + 1):
            public_search.public_search_randomness_key = (
                lambda *a, _s=salt, **k: original_key(*a, **k) + f":dice{_s}"
            )

            def run():
                bind_own_team(built.own_team)
                return public_information_exact_search(
                    obs, wide, built.own_team, memory=memory, candidate_selector=selector
                )

            again = contextvars.copy_context().run(run)
            scored = [(e, row_of(e, obs, i + 1)) for i, e in enumerate(again) if e.breakdown.get("searched")]
            base = [(e, r) for e, r in scored if r["message"] in searched]
            winner = max(base, key=lambda item: item[1]["score"])[1]
            winners[winner["order"]] = winners.get(winner["order"], 0) + 1
            probes = [r for _e, r in scored if r["message"] in probe]
            if probes:
                best_probe = max(probes, key=lambda r: r["score"])
                gaps.append(best_probe["score"] - winner["score"])
                probe_wins += best_probe["score"] > winner["score"]
    finally:
        public_search.public_search_randomness_key = original_key
    return {
        "rerolls": DICE_REROLLS,
        "winners": sorted(winners.items(), key=lambda kv: -kv[1]),
        "probe_beats_winner": probe_wins if probe else None,
        "mean_probe_minus_winner": (sum(gaps) / len(gaps)) if gaps else None,
    }


def ask_exact(built: Built, name: str) -> dict[str, Any]:
    config = version_config(name)
    agent = fresh_agent(built, config)
    obs = built.obs
    apply_own_spreads(obs)
    memory = agent.player._memory_for(obs)

    def warm():
        # The fast search fills BattleMemory's plan fields on a position's first look, and
        # the exact search's random futures are keyed on the memory summary.  Warm it once
        # (as the live bot's fast search does before its judge) so every ask of this position
        # -- A, B, C and the forced probes -- shares one key, i.e. common random numbers.
        bind_own_team(built.own_team)
        agent.player._fast_search(obs, memory)

    contextvars.copy_context().run(warm)
    started = time.perf_counter()

    def run():
        bind_own_team(built.own_team)
        return public_information_exact_search(obs, config, built.own_team, memory=memory)

    ranked = contextvars.copy_context().run(run)
    elapsed = time.perf_counter() - started
    rows = [row_of(entry, obs, index + 1) for index, entry in enumerate(ranked)]
    forced = forced_probe_exact(built, config, memory, ranked, rows, obs)
    dice = dice_check(built, config, memory, rows, obs, forced)
    return {
        "version": name,
        "elapsed_s": round(elapsed, 1),
        "chosen_order_obj": ranked[0].order,
        "forced": forced,
        "dice": dice,
        "chosen": rows[0],
        "top3": rows[:3],
        "rows": rows,
        "n_ranked": len(rows),
        "n_searched": sum(1 for row in rows if row["searched"]),
    }


def forced_probe_live(built, config, memory, obs, fast_list, fast_rows):
    """Exact values (the judge's own width and metric) for the probe orders, judged or not."""

    from vgc.exact_judge import default_ranker, judge_config, select_judged

    moves = built.pos.probe or built.pos.all
    wanted_probe = probe_messages(fast_rows, moves, "score")
    if not wanted_probe:
        return None
    judged = select_judged(fast_list, config.exact_judge_top_k, config.exact_judge_extra_myopic)
    wanted = {e.order.message for e in judged} | set(wanted_probe)
    judge_cfg = judge_config(config, len(wanted))

    def run():
        bind_own_team(built.own_team)
        return default_ranker(
            obs, memory, built.own_team, judge_cfg, frozenset(wanted), config.exact_judge_metric
        )

    values = contextvars.copy_context().run(run)
    by_msg = {row["message"]: row for row in fast_rows}
    return {
        "values": values,
        "probe": [
            {**by_msg[m], "exact_value": values.get(m), "judged": m in {e.order.message for e in judged}}
            for m in wanted_probe
        ],
    }


def ask_live(built: Built, name: str) -> dict[str, Any]:
    config = version_config(name)
    agent = fresh_agent(built, config)
    player = agent.player
    obs = built.obs
    apply_own_spreads(obs)
    started = time.perf_counter()
    memory = player._memory_for(obs)
    # The fast search alone (the live bot's shortlist source), ranked best-first.
    contextvars.copy_context().run(lambda: bind_own_team(built.own_team))

    def fast():
        bind_own_team(built.own_team)
        return player._fast_search(obs, memory)

    fast_list = contextvars.copy_context().run(fast)
    fast_rows = [row_of(entry, obs, index + 1) for index, entry in enumerate(fast_list)]
    by_text = {describe_order(entry.order): row for entry, row in zip(fast_list, fast_rows)}

    # Forced exact values BEFORE the real decision: choose_move records our choice into
    # BattleMemory, which would change the random-futures key the judge itself used.
    forced = forced_probe_live(built, config, memory, obs, fast_list, fast_rows)
    chosen_order = contextvars.copy_context().run(lambda: player.choose_move(obs))
    elapsed = time.perf_counter() - started
    if forced is not None:
        forced["chosen_value"] = forced["values"].get(chosen_order.message)
    log = player.exact_judge_log[-1] if player.exact_judge_log else {}
    if forced is not None:
        for item in log.get("ranking", []):
            match = next((e for e in fast_list if describe_order(e.order) == item["order"]), None)
            mine = None if match is None else forced["values"].get(match.order.message)
            if mine is not None and abs(mine - item["value"]) > 0.01:
                print(
                    f"  WARNING: forced value {mine:.2f} != judge value {item['value']:.2f} "
                    f"for {item['order']}",
                    flush=True,
                )
    ranking = []
    for item in log.get("ranking", []):
        row = dict(by_text.get(item["order"], {"order": item["order"], "moves": []}))
        row["exact_value"] = item["value"]
        row["fast_rank"] = item["fast_rank"]
        ranking.append(row)
    chosen_pretty = pretty(chosen_order, obs)
    chosen = {
        "order": chosen_pretty,
        "moves": move_ids(chosen_order),
        "exact_value": next(
            (r["exact_value"] for r in ranking if r["order"] == chosen_pretty), None
        ),
        "fast_rank": next((r["fast_rank"] for r in ranking if r["order"] == chosen_pretty), None),
    }
    return {
        "version": name,
        "elapsed_s": round(elapsed, 1),
        "chosen_order_obj": chosen_order,
        "forced": forced,
        "chosen": chosen,
        "judge_status": log.get("status"),
        "judge_error": log.get("error"),
        "judged": log.get("judged"),
        "fast_pick": by_text.get(log.get("fast_pick"), {}).get("order", log.get("fast_pick")),
        "overturned": log.get("overturned"),
        "gain": log.get("gain"),
        "top3": ranking[:3],
        "ranking": ranking,
        "fast_rows": fast_rows,
        "n_fast": len(fast_rows),
        "n_fast_searched": sum(1 for r in fast_rows if r["searched"] is not False),
    }


def setup_lookup(pos: Position, result: dict[str, Any]) -> dict[str, Any]:
    moves = pos.probe or pos.all
    if result["version"] in ("D0", "D1"):
        judged = best_with(result["ranking"], moves)
        fast = best_with(result["fast_rows"], moves)
        return {
            "moves": list(moves),
            "judged": judged is not None,
            "exact_rank": None
            if judged is None
            else 1 + next(i for i, r in enumerate(result["ranking"]) if r is judged),
            "exact_value": None if judged is None else judged["exact_value"],
            "fast_rank": None if fast is None else fast["rank"],
            "fast_searched": None if fast is None else fast["searched"],
            "order": (judged or fast or {}).get("order"),
        }
    row = best_with(result["rows"], moves)
    return {
        "moves": list(moves),
        "searched": None if row is None else bool(row["searched"]),
        "rank": None if row is None else row["rank"],
        "score": None if row is None else row["score"],
        "exchange": None if row is None else row["exchange"],
        "field_delta": None if row is None else row["field_delta"],
        "order": None if row is None else row["order"],
        "n_ranked": result["n_ranked"],
    }


# --- report ------------------------------------------------------------------------------


def fmt_top3(result: dict[str, Any]) -> str:
    cells = []
    for index, row in enumerate(result["top3"], start=1):
        if result["version"] in ("D0", "D1"):
            extra = f"exact {row.get('exact_value')}, fast #{row.get('fast_rank')}"
        else:
            extra = f"score {row['score']}, exch {row['exchange']}"
            if row["field_delta"] is not None:
                extra += f", field {row['field_delta']}"
            if row["myopic"] is not None and row["searched"]:
                extra += f", myo {row['myopic']}"
            if not row["searched"]:
                extra += ", UNSEARCHED"
        cells.append(f"{index}. {row['order']} ({extra})")
    return "<br>".join(cells)


def fmt_forced(result: dict[str, Any]) -> str:
    forced = result.get("forced")
    if not forced:
        return ""
    if result["version"] in ("D0", "D1"):
        best = max(
            (p for p in forced["probe"] if p["exact_value"] is not None),
            key=lambda p: p["exact_value"],
            default=None,
        )
        if best is None:
            return ""
        return (
            f"<br>forced exact value of best setup order ({best['order']}): "
            f"{round(best['exact_value'], 1)} vs chosen "
            f"{'n/a' if forced.get('chosen_value') is None else round(forced['chosen_value'], 1)}"
        )
    best = max(forced["probe"], key=lambda p: p["score"])
    extra = f", field {best['field_delta']}" if best["field_delta"] is not None else ""
    text = (
        f"<br>searched anyway: {best['order']} scores {best['score']}{extra} "
        f"(rank {best['rank']}) vs chosen {forced['chosen_score']}"
    )

    def worst(row) -> str:
        replies = row.get("responses") or []
        if not replies:
            return "n/a"
        low = min(replies, key=lambda r: r["value"])
        heavy = max(replies, key=lambda r: r["weight"])
        return (
            f"worst reply [{low['reply']}] {low['value']} (w {low['weight']}); "
            f"most likely reply [{heavy['reply']}] {heavy['value']} (w {heavy['weight']})"
        )

    return text + f"<br>setup order: {worst(best)}<br>chosen: {worst(result['chosen'])}"


def fmt_setup(result: dict[str, Any]) -> str:
    return _fmt_setup(result) + fmt_forced(result)


def _fmt_setup(result: dict[str, Any]) -> str:
    s = result["setup"]
    if not s["moves"]:
        return "-"
    label = "+".join(s["moves"])
    if result["version"] in ("D0", "D1"):
        if s["fast_rank"] is None:
            return f"{label}: not generated by the fast search"
        judged = (
            f"judged, exact rank {s['exact_rank']}, value {s['exact_value']}"
            if s["judged"]
            else "NOT in the judged top-6"
        )
        return f"{label}: fast rank {s['fast_rank']}; {judged}"
    if s["rank"] is None:
        return f"{label}: no legal order contains it"
    if not s["searched"]:
        return (
            f"{label}: rank {s['rank']}/{s['n_ranked']}, UNSEARCHED (cut by the myopic "
            "shortlist; its listed score is a placeholder)"
        )
    extra = f", exch {s['exchange']}" if s["exchange"] is not None else ""
    if s["field_delta"] is not None:
        extra += f", field {s['field_delta']}"
    return f"{label}: rank {s['rank']}/{s['n_ranked']}, searched, score {s['score']}{extra}"


def board_text(facts: dict[str, Any]) -> list[str]:
    def mon_line(m):
        if m is None:
            return "(empty)"
        boosts = f" boosts {m['boosts']}" if m["boosts"] else ""
        status = f" {m['status']}" if m["status"] else ""
        return f"{m['species']} {m['hp_pct']}%{status}{boosts} item={m['item']}"

    ours = ", ".join(facts["our_side"]) or "none"
    theirs = ", ".join(facts["their_side"]) or "none"
    lines = [
        f"- turn {facts['turn']}; weather {facts['weather'] or 'none'}; "
        f"fields {facts['fields'] or 'none'}",
        f"- our side conditions: {ours}; their side conditions: {theirs}",
    ]
    timed = [c for c in facts["conditions"]]
    if timed:
        lines.append(
            "- tracked conditions (turns left incl. this turn): "
            + "; ".join(
                f"{c['effect']}{'/' + c['side'] if c['side'] else ''}={c['turns_left']}"
                for c in timed
            )
        )
    lines.append("- ours: " + " ; ".join(mon_line(m) for m in facts["ours"]))
    lines.append("- theirs: " + " ; ".join(mon_line(m) for m in facts["theirs"]))
    ours_ids = [m["species"] if m else "-" for m in facts["ours"]]
    theirs_ids = [m["species"] if m else "-" for m in facts["theirs"]]
    lines.append(
        f"- reply legend: opponent orders list their slots {theirs_ids} in that order; "
        f"`@1`/`@2` = our {ours_ids}, `@-1`/`@-2` = their own slots"
    )
    return lines


def render(results: list[dict[str, Any]], versions: list[str]) -> str:
    out = [
        "# Setup positions: does the bot make the setup calls a good VGC player would?",
        "",
        "Generated by `offline/setup_positions.py` (2026-10-07). Real Showdown engine, our "
        "side is p1 and decides from its public view only. Scores for A/B/C are the exact "
        "search's final score (myopic weight 0 so score = exact exchange value, 1 point ~ "
        "1% of one Pokemon's max HP, a standing Pokemon = 90 more); D0/D1 rank the fast "
        "search's top 6 by exact exchange value. `field` = the field-control leaf's share "
        "of the exchange value (B, C only).",
        "",
        "Versions: A exact judge defaults; B = A + field control + measured plan (fit 0.5); "
        "C = B + measured speed payoff; D0 live ladder bot (fast search + exact judge top-6); "
        "D1 = D0 + fast-search setter modelling, measured plan, fit 0.5, condition expiry.",
        "",
    ]
    for res in results:
        pos: Position = res["pos"]
        out += [f"## Position {pos.pid}: {pos.title}", ""]
        out += [
            f"- Teams: ours `teams/owner/{pos.own}.packed.txt` (order `{pos.own_order}`), "
            f"opponent `mc_sheet_pool_v2/team_{pos.opp}` (order `{pos.opp_order}`)",
        ]
        if res.get("prelude"):
            out += [f"- Prelude: {'; '.join(res['prelude'])}"]
        out += board_text(res["board"])
        out += [f"- **Expected:** {pos.expected}", ""]
        if res.get("error"):
            out += [f"**Position failed to build:** `{res['error']}`", ""]
            continue
        out += ["| version | chosen | verdict | top 3 | setup call rank |", "|---|---|---|---|---|"]
        for name in versions:
            ver = res["versions"].get(name)
            if ver is None:
                continue
            if ver.get("error"):
                out.append(f"| {name} | ERROR | - | `{ver['error']}` | - |")
                continue
            chosen = ver["chosen"]
            extra = ""
            if name in ("D0", "D1"):
                extra = (
                    f"<br>fast pick: {ver['fast_pick']}; overturned={ver['overturned']}, "
                    f"gain {ver['gain']}, judge {ver['judge_status']}"
                )
            out.append(
                f"| {name} | {chosen['order']}{extra} | {ver['verdict']} | "
                f"{fmt_top3(ver)} | {fmt_setup(ver)} |"
            )
        out.append("")
        for name in versions:
            ver = res["versions"].get(name) or {}
            dice = ver.get("dice")
            if not dice:
                continue
            winners = "; ".join(f"{order} x{count}" for order, count in dice["winners"])
            beats = (
                ""
                if dice["probe_beats_winner"] is None
                else f" Best setup-call order (probe) beats that winner in {dice['probe_beats_winner']}/"
                f"{dice['rerolls']} re-rolls (mean gap {dice['mean_probe_minus_winner']:+.1f})."
            )
            out.append(
                f"- Dice check {name} ({dice['rerolls']} re-rolls of the sampled futures, same "
                f"shortlist): picks {winners}.{beats}"
            )
        out.append("")
    # Summary
    out += ["## Summary", "", "| position | " + " | ".join(versions) + " |",
            "|---|" + "---|" * len(versions)]
    for res in results:
        pos = res["pos"]
        cells = []
        for name in versions:
            ver = res["versions"].get(name) if not res.get("error") else None
            cells.append("-" if ver is None else ("ERR" if ver.get("error") else ver["verdict"]))
        out.append(f"| {pos.pid} {pos.key} | " + " | ".join(cells) + " |")
    out += [""]
    return "\n".join(out)


# --- driver ------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", default="", help="comma-separated position ids (default all)")
    ap.add_argument("--versions", default=",".join(VERSIONS))
    ap.add_argument("--output", type=Path, default=REPO_ROOT / "runs/eval/setup_positions.md")
    ap.add_argument("--json", type=Path, default=REPO_ROOT / "runs/eval/setup_positions.json")
    args = ap.parse_args()
    wanted = {int(x) for x in args.only.split(",") if x}
    versions = [v for v in args.versions.split(",") if v]
    positions = [p for p in POSITIONS if not wanted or p.pid in wanted]
    results: list[dict[str, Any]] = []
    with SimWorker() as worker:
        for pos in positions:
            print(f"== position {pos.pid}: {pos.key}", flush=True)
            res: dict[str, Any] = {"pos": pos, "versions": {}}
            try:
                built = build(worker, pos)
                res["board"] = board_facts(built.obs)
                res["prelude"] = built.prelude_log
                for name in versions:
                    try:
                        if name in ("D0", "D1"):
                            ver = ask_live(built, name)
                        else:
                            ver = ask_exact(built, name)
                        pressured = (
                            pressures(ver["chosen_order_obj"], built.obs, pos.pressure, pos.pressure_excl)
                            if pos.pressure
                            else True
                        )
                        ver["verdict"] = verdict(pos, ver["chosen"]["moves"], pressured)
                        ver["pressured"] = pressured
                        ver["setup"] = setup_lookup(pos, ver)
                    except Exception as exc:  # noqa: BLE001 - reported, never hidden
                        traceback.print_exc()
                        ver = {"version": name, "error": f"{type(exc).__name__}: {exc}"}
                    res["versions"][name] = ver
                    print(
                        f"  {name}: "
                        + (ver.get("error") or f"{ver['verdict']:9s} {ver['chosen']['order']}")
                        + (f"  [{ver.get('elapsed_s')}s]" if not ver.get("error") else ""),
                        flush=True,
                    )
                built.battle.close()
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                res["error"] = f"{type(exc).__name__}: {exc}"
                res.setdefault("board", {"turn": "?", "weather": {}, "fields": {}, "our_side": {},
                                         "their_side": {}, "conditions": [], "ours": [], "theirs": []})
            results.append(res)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(results, versions))
    for res in results:
        for ver in res["versions"].values():
            ver.pop("chosen_order_obj", None)
    serial = [
        {k: (v.__dict__ if k == "pos" else v) for k, v in res.items()} for res in results
    ]
    args.json.write_text(json.dumps(serial, indent=1, default=str))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
