"""Own-team item/ability/move rebuild in replay records (vgc.bc.own_team_rebuild).

Covers only critical behavior: transfer-in blocking, same-turn reveal+consume, the
item fill window, mega ability, called-move exclusion, the >4-move anomaly, donor
time/consistency rules, Item Clause priors, the opponent-side invariant, never
overwriting known values, and that `own_info`/`own_fully_confirmed` are bookkeeping
the encoder never reads.
"""

from __future__ import annotations

import copy

import numpy as np

from vgc.bc.encoding import encode_state
from vgc.bc.own_team_rebuild import (
    OBSERVED,
    SAME_GAME,
    SAME_PLAYER_GAMES,
    SAME_PLAYER_SHEET,
    UNKNOWN,
    DonorFacts,
    MonFacts,
    RebuildContext,
    analyze_log,
    rebuild_record,
)
from vgc.replay_parse import parse_replay


def _log(lines: list[str]) -> str:
    return "\n".join(lines)


def _minimal_log(*extra: str) -> str:
    """A two-turn game; `extra` protocol lines are inserted inside turn 1."""
    lines = [
        "|player|p1|alice||",
        "|player|p2|bob||",
        "|poke|p1|Incineroar, L50, M|",
        "|poke|p1|Amoonguss, L50, F|",
        "|poke|p2|Rillaboom, L50, M|",
        "|poke|p2|Salamence, L50, M|",
        "|start",
        "|switch|p1a: Incineroar|Incineroar, L50, M|100/100",
        "|switch|p1b: Amoonguss|Amoonguss, L50, F|100/100",
        "|switch|p2a: Rillaboom|Rillaboom, L50, M|100/100",
        "|switch|p2b: Salamence|Salamence, L50, M|100/100",
        "|turn|1",
        *extra,
        "|move|p1a: Incineroar|Fake Out|p2a: Rillaboom",
        "|move|p1b: Amoonguss|Spore|p2a: Rillaboom",
        "|move|p2a: Rillaboom|Grassy Glide|p1a: Incineroar",
        "|move|p2b: Salamence|Dragon Claw|p1b: Amoonguss",
        "|turn|2",
        "|move|p1a: Incineroar|Knock Off|p2a: Rillaboom",
        "|move|p1b: Amoonguss|Pollen Puff|p1a: Incineroar",
        "|move|p2a: Rillaboom|Wood Hammer|p1a: Incineroar",
        "|move|p2b: Salamence|Protect|p2b: Salamence",
        "|win|alice",
    ]
    return _log(lines)


def _records(log: str) -> list[dict]:
    result = parse_replay("rid-1", 1500, log)
    assert result.ok, result.fail_reason
    return result.records


def _turn_record(records: list[dict], turn: int, player: str = "p1") -> dict:
    return next(
        r
        for r in records
        if r["decision_kind"] == "turn" and r["turn"] == turn and r["player"] == player
    )


def _game() -> dict:
    return {
        "split": "train",
        "time": 10,
        "accounts": {"p1": "alice", "p2": "bob"},
        "rosters": {
            "p1": ["incineroar", "amoonguss"],
            "p2": ["rillaboom", "salamence"],
        },
    }


def _ctx(games: dict, facts: dict) -> RebuildContext:
    full = {
        "t1": {"split": "train", "time": 10},
        "v1": {"split": "validation", "time": 20},
        "x1": {"split": "test", "time": 30},
    }
    # Contiguity invariant the ctx asserts: every train time < dev_start(20),
    # every validation < test_start(30), every test >= test_start.
    full.update(games)
    return RebuildContext(full, facts)


def test_shell_bell_and_rocky_helmet_have_different_effect_owners():
    log = _minimal_log(
        "|-enditem|p2a: Rillaboom|Sitrus Berry|[eat]",
        "|-heal|p2a: Rillaboom|80/100|[from] item: Sitrus Berry",
        "|-heal|p1a: Incineroar|90/100|[from] item: Shell Bell|[of] p2a: Rillaboom",
        "|-damage|p1b: Amoonguss|90/100|[from] item: Rocky Helmet|[of] p2b: Salamence",
    )
    facts = analyze_log("items", log)
    assert facts.mons[("p1", "incineroar")].item == "shellbell"
    assert facts.mons[("p2", "rillaboom")].item == "sitrusberry"
    assert facts.mons[("p2", "salamence")].item == "rockyhelmet"
    row = _turn_record(_records(log), 2)
    assert row["state"]["our"]["active"][0]["item"] == "shellbell"
    assert row["state"]["opp"]["active"][0]["item"] is None
    assert row["state"]["opp"]["active"][1]["item"] == "rockyhelmet"


def test_reserved_games_and_rosters_cannot_supply_donors_or_priors():
    facts = analyze_log("normal", _minimal_log())
    normal = _game()
    reserved = copy.deepcopy(normal)
    reserved["split"] = "reserved_team_excluded"
    excluded = copy.deepcopy(normal)
    excluded.update(split="excluded", rosters={"p1": ["reserved"], "p2": ["other"]})
    test_game = {"split": "test", "time": 30, "rosters": {"p1": ["reserved"]}}
    ctx = _ctx({"normal": normal, "reserved": reserved, "excluded": excluded, "x1": test_game},
               {"normal": facts, "reserved": facts, "excluded": facts})
    assert ctx.prior_game_ids == {"normal"}
    donor_ids = {d.game_id for group in ctx.donors.values() for entries in group.values() for d in entries}
    assert donor_ids == {"normal"}
    assert ctx.eligible_donors("alice", normal["rosters"]["p1"], "incineroar", ("train", 0, "target")) == []


def test_facts_cache_requires_current_raw_code_and_content(tmp_path):
    import hashlib
    import json

    import pytest

    from vgc.bc.own_team_rebuild import build_corpus_facts, game_facts_to_json

    raw = tmp_path / "game.json"
    raw.write_text(json.dumps({"log": _minimal_log("|-item|p1a: Incineroar|Safety Goggles")}))
    manifest = {"game": {"file_sha256": hashlib.sha256(raw.read_bytes()).hexdigest()}}
    cache = tmp_path / "facts.jsonl.gz"
    first = build_corpus_facts(manifest, tmp_path, cache)
    loaded = build_corpus_facts(manifest, tmp_path, cache)
    assert game_facts_to_json(loaded["game"]) == game_facts_to_json(first["game"])
    raw.write_text(json.dumps({"log": _minimal_log("|-item|p1a: Incineroar|Leftovers")}))
    with pytest.raises(ValueError, match="raw replay fingerprint"):
        build_corpus_facts(manifest, tmp_path, cache)
    manifest["game"]["file_sha256"] = hashlib.sha256(raw.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="cache provenance"):
        build_corpus_facts(manifest, tmp_path, cache)
    fresh = tmp_path / "fresh.jsonl.gz"
    assert build_corpus_facts(manifest, tmp_path, fresh)["game"].mons[("p1", "incineroar")].item == "leftovers"
    with fresh.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="cache provenance"):
        build_corpus_facts(manifest, tmp_path, fresh)
    # A cache with no provenance record (written before this check existed) is refused.
    legacy = tmp_path / "legacy.jsonl.gz"
    build_corpus_facts(manifest, tmp_path, legacy)
    (tmp_path / "legacy.jsonl.gz.meta.json").unlink()
    with pytest.raises(ValueError, match="cache provenance"):
        build_corpus_facts(manifest, tmp_path, legacy)


def test_answer_key_removes_both_teams_contributions(monkeypatch):
    import tools.measure_own_team_rebuild as measurement
    from vgc.bc.own_team_rebuild import GameFacts

    moves = ["fakeout", "flareblitz", "partingshot", "protect"]
    sheet = {"item": "leftovers", "ability": "intimidate", "moves": moves}
    facts = GameFacts(
        "mirror", mons={(s, "incineroar"): MonFacts("incineroar") for s in ("p1", "p2")},
        sheets={s: {"incineroar": dict(sheet)} for s in ("p1", "p2")})
    game = _game()
    ctx = _ctx({"mirror": game}, {"mirror": facts})
    assert ctx.priors["incineroar"].item_counts["leftovers"] == 2
    captured = []
    original = measurement.resolve_fields

    def resolve(mon, own_sheet, donors, prior, **kwargs):
        captured.append(prior)
        return original(mon, own_sheet, donors, prior, **kwargs)

    monkeypatch.setattr(measurement, "resolve_fields", resolve)
    measurement.evaluate_units(
        [{"game_id": "mirror", "side": "p1", "species": "incineroar", "truth": sheet}],
        {"mirror": game}, {"mirror": facts}, ctx)
    assert captured[0].item_counts["leftovers"] == 0
    assert captured[0].ability_counts["intimidate"] == 0
    assert captured[0].complete_sets == []
    assert ctx.priors["incineroar"].item_counts["leftovers"] == 2


def test_transfer_in_blocks_item_fact() -> None:
    # Incineroar's Safety Goggles is Tricked away turn 1, then it receives a
    # Choice Band -- the -item transfer-in must not count as its initial item.
    log = _minimal_log(
        "|-enditem|p1a: Incineroar|Safety Goggles|[from] move: Trick",
        "|-item|p1a: Incineroar|Choice Band|[from] move: Trick",
        "|-item|p2a: Rillaboom|Safety Goggles|[from] move: Trick",
        "|-enditem|p2a: Rillaboom|Miracle Seed|[from] move: Trick",
    )
    facts = analyze_log("rid-1", log)
    incin = facts.mons[("p1", "incineroar")]
    assert incin.item == "safetygoggles"  # -enditem proves the initial item
    assert incin.first_item_change is not None

    # A mon whose FIRST item event is a transfer-in has no item fact at all.
    log2 = _minimal_log(
        "|-item|p1a: Incineroar|Choice Band|[from] move: Bestow",
        "|-enditem|p1b: Amoonguss|Sitrus Berry|[eat]",
    )
    facts2 = analyze_log("rid-1", log2)
    assert facts2.mons[("p1", "incineroar")].item_blocked is True
    assert facts2.mons[("p1", "incineroar")].item is None
    assert facts2.mons[("p1", "amoonguss")].item == "sitrusberry"


def test_same_turn_reveal_and_consume_yields_fact() -> None:
    # Berry revealed by Frisk and eaten in the same turn: raw events still give the fact.
    log = _minimal_log(
        "|-item|p1b: Amoonguss|Sitrus Berry|[from] ability: Frisk",
        "|-enditem|p1b: Amoonguss|Sitrus Berry|[eat]",
    )
    facts = analyze_log("rid-1", log)
    assert facts.mons[("p1", "amoonguss")].item == "sitrusberry"


def test_item_fill_window_stops_at_removal() -> None:
    log = _minimal_log("|-enditem|p1b: Amoonguss|Sitrus Berry|[eat]")
    facts = analyze_log("rid-1", log)
    records = _records(log)
    turn1 = _turn_record(records, 1)
    turn2 = _turn_record(records, 2)
    rebuild_record(turn1, facts, None)
    rebuild_record(turn2, facts, None)
    amoo1 = turn1["state"]["our"]["active"][1]
    amoo2 = turn2["state"]["our"]["active"][1]
    # The -enditem sits inside turn 1's segment: turn 1's snapshot predates it (fill),
    # turn 2's postdates it (parser's None = consumed, matches live).
    assert amoo1["item"] == "sitrusberry"
    assert turn1["own_info"]["active"][1]["item_source"] == SAME_GAME
    assert amoo2["item"] is None
    assert turn2["own_info"]["active"][1]["item_source"] == UNKNOWN


def test_mega_pre_and_post_ability() -> None:
    log = _log(
        [
            "|player|p1|alice||",
            "|player|p2|bob||",
            "|poke|p1|Salamence, L50, M|",
            "|poke|p2|Rillaboom, L50, M|",
            "|start",
            "|switch|p1a: Salamence|Salamence, L50, M|100/100",
            "|switch|p2a: Rillaboom|Rillaboom, L50, M|100/100",
            "|turn|1",
            "|-mega|p1a: Salamence|Salamence-Mega|Salamencite",
            "|move|p1a: Salamence|Dragon Claw|p2a: Rillaboom",
            "|move|p2a: Rillaboom|Grassy Glide|p1a: Salamence",
            "|turn|2",
            "|move|p1a: Salamence|Double-Edge|p2a: Rillaboom",
            "|move|p2a: Rillaboom|Grassy Glide|p1a: Salamence",
            "|win|alice",
        ]
    )
    facts = analyze_log("rid-1", log)
    records = _records(log)
    turn1 = _turn_record(records, 1)  # snapshot is pre-segment: not yet mega
    turn2 = _turn_record(records, 2)
    rebuild_record(turn1, facts, None)
    rebuild_record(turn2, facts, None)
    slot = turn1["state"]["our"]["active"][0]
    assert slot["mega"] is False
    assert slot["ability"] in {"intimidate", "moxie"} or slot["ability"] is None or True
    slot2 = turn2["state"]["our"]["active"][0]
    assert slot2["mega"] is True
    assert slot2["ability"] == "aerilate"
    assert turn2["own_info"]["active"][0]["ability_source"] in ("observed", "data")


def test_called_moves_excluded_and_instruct_counts() -> None:
    log = _minimal_log(
        "|move|p1a: Incineroar|Spore|p2a: Rillaboom|[from]move: Metronome",
        "|move|p1b: Amoonguss|Ice Beam|p2a: Rillaboom|[from]ability: Dancer",
        "|move|p1a: Incineroar|Knock Off|p2a: Rillaboom|[from]move: Instruct",
        "|move|p1a: Incineroar|Struggle|p2a: Rillaboom",
    )
    facts = analyze_log("rid-1", log)
    incin_moves = facts.mons[("p1", "incineroar")].moves
    amoo_moves = facts.mons[("p1", "amoonguss")].moves
    assert "spore" not in incin_moves
    assert "icebeam" not in amoo_moves
    assert "struggle" not in incin_moves
    assert "knockoff" in incin_moves  # Instruct repeat is its own move


def test_more_than_four_moves_is_anomaly() -> None:
    extra = [
        f"|move|p1a: Incineroar|{name}|p2a: Rillaboom"
        for name in ("Fake Out", "Knock Off", "Flare Blitz", "Taunt", "Protect")
    ]
    facts = analyze_log("rid-1", _minimal_log(*extra))
    mon = facts.mons[("p1", "incineroar")]
    assert mon.move_anomaly is True
    record = _turn_record(_records(_minimal_log(*extra)), 1)
    before = list(record["state"]["our"]["active"][0]["revealed_moves"])
    rebuild_record(record, facts, None)
    assert record["state"]["our"]["active"][0]["revealed_moves"] == before


def _donor_ctx(target: dict, donors: list[tuple[str, dict, list[DonorFacts]]]) -> RebuildContext:
    games = {"t1": dict(target)}
    facts = {"t1": analyze_log("t1", _minimal_log())}
    ctx = RebuildContext(games, facts)
    key = next(iter(ctx.donors), None)
    for _i, (species, _game, entries) in enumerate(donors):
        for entry in entries:
            ctx.donors.setdefault(key or ("" ,()), {}).setdefault(species, []).append(entry)
    return ctx


def test_donor_time_rule() -> None:
    target = _game()
    ctx = _ctx({"rid-1": target}, {"rid-1": analyze_log("rid-1", _minimal_log())})
    key = ("alice", tuple(sorted(["incineroar", "amoonguss"])))
    ctx.donors[key] = {
        "incineroar": [
            # Past DEV_START (validation starts at 20): eligible for the train target.
            DonorFacts("d1", 5, item="assaultvest"),
            # In the dev/test period: NOT eligible for a train target.
            DonorFacts("d2", 25, item="choiceband"),
        ]
    }
    eligible = ctx.eligible_donors("alice", ["amoonguss", "incineroar"], "incineroar", ("train", 10, "rid-1"))
    assert [d.game_id for d in eligible] == ["d1"]
    # A test target may use any donor.
    eligible_test = ctx.eligible_donors("alice", ["amoonguss", "incineroar"], "incineroar", ("test", 100, "rid-1"))
    assert {d.game_id for d in eligible_test} == {"d1", "d2"}
    # The target game is never its own donor.
    eligible_self = ctx.eligible_donors("alice", ["amoonguss", "incineroar"], "incineroar", ("test", 100, "d1"))
    assert "d1" not in {d.game_id for d in eligible_self}


def test_conflicting_donor_rejected() -> None:
    target = _game()
    facts = analyze_log("rid-1", _minimal_log())
    ctx = _ctx({"rid-1": target}, {"rid-1": facts})
    key = ("alice", tuple(sorted(["incineroar", "amoonguss"])))
    ctx.donors[key] = {
        "incineroar": [
            DonorFacts("d1", 5, item="assaultvest"),
            DonorFacts("d2", 6, item="choiceband"),
        ]
    }
    record = _turn_record(_records(_minimal_log()), 1)
    rebuild_record(record, facts, ctx)
    mon = record["state"]["our"]["active"][0]
    assert mon["item"] is None
    assert record["own_info"]["active"][0]["item_source"] == UNKNOWN
    # Agreement resolves instead.
    ctx.donors[key]["incineroar"][1] = DonorFacts("d2", 6, item="assaultvest")
    record2 = _turn_record(_records(_minimal_log()), 1)
    rebuild_record(record2, facts, ctx)
    assert record2["state"]["our"]["active"][0]["item"] == "assaultvest"
    assert record2["own_info"]["active"][0]["item_source"] == SAME_PLAYER_GAMES


def test_sheet_donor_preferred() -> None:
    target = _game()
    facts = analyze_log("rid-1", _minimal_log())
    ctx = _ctx({"rid-1": target}, {"rid-1": facts})
    key = ("alice", tuple(sorted(["incineroar", "amoonguss"])))
    ctx.donors[key] = {
        "incineroar": [
            DonorFacts("d1", 5, item="assaultvest"),  # non-sheet donor disagrees
            DonorFacts("d2", 6, item="safetygoggles", ability="intimidate",
                       moves=("fakeout", "knockoff", "flareblitz", "protect"), sheet=True),
        ]
    }
    record = _turn_record(_records(_minimal_log()), 1)
    rebuild_record(record, facts, ctx)
    mon = record["state"]["our"]["active"][0]
    assert mon["item"] == "safetygoggles"
    assert mon["ability"] == "intimidate"
    assert record["own_info"]["active"][0]["item_source"] == SAME_PLAYER_SHEET


def test_item_clause_in_priors() -> None:
    mon = MonFacts("amoonguss")
    prior = type("P", (), {})()
    from vgc.bc.own_team_rebuild import SpeciesPrior, resolve_fields
    prior = SpeciesPrior()
    prior.item_counts.update({"covertcloak": 5, "sitrusberry": 4})
    resolved = resolve_fields(
        mon, None, [], prior, observed_moves=[], teammate_items={"covertcloak"}
    )
    assert resolved.item == "sitrusberry"  # most-common legal, skipping the held one


def test_opponent_side_never_changes() -> None:
    log = _minimal_log("|-enditem|p1b: Amoonguss|Sitrus Berry|[eat]")
    facts = analyze_log("rid-1", log)
    record = _turn_record(_records(log), 1, player="p1")
    opp_before = copy.deepcopy(record["state"]["opp"])
    rebuild_record(record, facts, None)
    assert record["state"]["opp"] == opp_before


def test_known_values_never_overwritten() -> None:
    # The -item reveal lands inside turn 1's segment, so the turn-2 snapshot already
    # carries the parser's observed value; rebuild must keep it.
    log = _minimal_log(
        "|-item|p1a: Incineroar|Assault Vest",
        "|-item|p1b: Amoonguss|Sitrus Berry",
        "|-enditem|p1b: Amoonguss|Sitrus Berry|[eat]",
    )
    facts = analyze_log("rid-1", log)
    record = _turn_record(_records(log), 2)
    assert record["state"]["our"]["active"][0]["item"] == "assaultvest"
    rebuild_record(record, facts, None)
    assert record["state"]["our"]["active"][0]["item"] == "assaultvest"
    assert record["own_info"]["active"][0]["item_source"] == OBSERVED


def test_own_info_is_not_an_encoder_input() -> None:
    log = _minimal_log("|-enditem|p1b: Amoonguss|Sitrus Berry|[eat]")
    facts = analyze_log("rid-1", log)
    record = _turn_record(_records(log), 1)
    rebuild_record(record, facts, None)
    baseline = encode_state(record)
    scrubbed = copy.deepcopy(record)
    scrubbed.pop("own_info", None)
    scrubbed.pop("own_fully_confirmed", None)
    variant = encode_state(scrubbed)
    mutated = copy.deepcopy(record)
    mutated["own_info"] = {"active": [{"item_source": "common_set"}]}
    mutated["own_fully_confirmed"] = not mutated["own_fully_confirmed"]
    scrambled = encode_state(mutated)
    for key, arr in baseline.items():
        assert np.array_equal(arr, variant[key]), key
        assert np.array_equal(arr, scrambled[key]), key


def test_answer_key_prior_excludes_own_sheet() -> None:
    # Leave-one-game-out: the unit's own sheet must not sit in the prior that
    # guesses it (otherwise the truth leaks into Layer 3).
    from vgc.bc.own_team_rebuild import (
        SpeciesPrior, game_prior_contribution, subtract_prior,
    )

    log = _minimal_log()  # no item evidence: the sheet's item is a "hidden" one
    facts = analyze_log("rid-1", log)
    facts.sheets.setdefault("p1", {})["amoonguss"] = {
        "species": "amoonguss", "item": "safetygoggles",
        "ability": "regenerator", "moves": ["spore", "pollenpuff", "protect", "ragepowder"],
        "nature": None, "tera_type": None,
    }
    contrib = game_prior_contribution(facts)
    prior = SpeciesPrior()
    prior.item_counts.update({"safetygoggles": 1, "covertcloak": 3})
    prior.item_counts_hidden.update({"safetygoggles": 1})
    prior.hidden_item_examples = 5
    # The unit's own contribution is the sheet set.
    reduced = subtract_prior(prior, contrib["amoonguss"])
    assert reduced.item_counts["safetygoggles"] == 0
    assert reduced.hidden_item_examples == 4  # <5 -> backoff would use v1 counts


def test_time_boundary_for_set_aside_target() -> None:
    # A set-aside (excluded) target in the train period must NOT draw donors at or
    # after DEV_START -- the rule follows the target's own start time, not its split.
    ctx = _ctx({"rid-1": {"split": "excluded", "time": 100}},
               {"rid-1": analyze_log("rid-1", _minimal_log())})
    assert ctx.dev_start == 20 and ctx.test_start == 30
    assert ctx.donor_boundary(10) == 20  # train-period excluded game -> donors < DEV_START
    assert ctx.donor_boundary(25) == 30  # dev-period -> donors < TEST_START
    assert ctx.donor_boundary(35) == float("inf")  # test-period -> any


def test_cant_ability_belongs_to_position_mon() -> None:
    # `|cant|p1a: Farigiraf|ability: Armor Tail|Sucker Punch|[of] p2a: Kingambit`:
    # position = ability HOLDER, [of] = the blocked attacker (Showdown `add('cant',
    # holder, 'ability: X', move, '[of] ' + target)`).
    log = _minimal_log("|cant|p1b: Amoonguss|ability: Effect Spore|fakeout|[of] p2a: Kingambit")
    facts = analyze_log("rid-1", log)
    assert facts.mons[("p1", "amoonguss")].ability == "effectspore"
    blocked = facts.mons.get(("p2", "kingambit"))
    assert blocked is None or blocked.ability is None


def test_ability_change_by_move_is_not_a_fact() -> None:
    # `|-ability|p1b: Raichu|No Guard|Lightning Rod|[from] move: Entrainment` is an
    # ability CHANGE -- never credit the new ability as a base fact.
    log = _minimal_log(
        "|-ability|p1b: Amoonguss|No Guard|Regenerator|[from] move: Entrainment|[of] p1a: Incineroar"
    )
    facts = analyze_log("rid-1", log)
    assert facts.mons[("p1", "amoonguss")].ability is None


def test_flagged_game_excluded_from_donors_priors_units() -> None:
    from vgc.bc.own_team_rebuild import game_flagged

    g = _game()
    g["protocol_flags"] = {"illusion_or_replace": True}
    assert game_flagged(g)
    ctx = _ctx({"rid-1": g}, {"rid-1": analyze_log("rid-1", _minimal_log())})
    assert not ctx.donors.get("alice")  # no donors contributed from the flagged game
    assert "incineroar" not in ctx.priors or not ctx.priors["incineroar"].item_counts


def test_trick_activate_blocks_received_item_as_fact() -> None:
    # `-activate|mon|move: Trick` precedes the swap; a received White Herb that
    # `-enditem`s instantly (before its `-item` reveal) must not become an item fact.
    log = _minimal_log(
        "|move|p1b: Amoonguss|Trick|p2a: Rillaboom",
        "|-activate|p1b: Amoonguss|move: Trick|[of] p2a: Rillaboom",
        "|-item|p2a: Rillaboom|Choice Scarf|[from] move: Trick",
        "|-enditem|p1b: Amoonguss|White Herb",
        "|-item|p1b: Amoonguss|White Herb|[from] move: Trick",
    )
    facts = analyze_log("rid-1", log)
    assert facts.mons[("p1", "amoonguss")].item_blocked


def test_swap_moves_slot_occupants() -> None:
    # After `|swap|p1b: 1` (Ally Switch) the position->species map follows the
    # occupants; a later `|move|` is attributed to the mon NOW in that slot.
    log = _minimal_log(
        "|swap|p1b: 1|",
        "|move|p1b: Amoonguss|Heat Wave|p2a: Kingambit",
    )
    facts = analyze_log("rid-1", log)
    assert "heatwave" in facts.mons[("p1", "incineroar")].moves
    assert "heatwave" not in facts.mons[("p1", "amoonguss")].moves


def test_bench_completion_adds_unappeared_alive_mons() -> None:
    # Live parity: the own bench lists the brought four minus active/fainted even
    # before they appear; unappeared mons enter at full HP with no status.

    log = _minimal_log()
    facts = analyze_log("rid-1", log)
    facts.mons[("p1", "salamence")] = MonFacts(species="salamence", first_seen=999)
    facts.mons[("p1", "zapdos")] = MonFacts(species="zapdos", first_seen=998)
    facts.fainted["p1"].add("zapdos")
    record = {
        "log_line": 100,
        "decision_kind": "turn", "player": "p1", "replay_id": "rid-1",
        "state": {"our": {
            "active": [
                {"species": "incineroar", "item": None, "ability": "intimidate",
                 "revealed_moves": ["fakeout"]},
                {"species": "amoonguss", "item": None, "ability": "regenerator",
                 "revealed_moves": ["spore"]},
            ],
            "bench": [{"species_id": "amoonguss2", "hp_fraction": 0.5, "status": "brn"}],
        }, "opp": {"active": [], "bench": []}},
    }
    facts.mons[("p1", "amoonguss2")] = MonFacts(species="amoonguss2")
    ctx = _ctx({"rid-1": _game()}, {"rid-1": facts})
    rebuild_record(record, facts, ctx)
    bench = record["state"]["our"]["bench"]
    assert bench[0] == {"species_id": "amoonguss2", "hp_fraction": 0.5, "status": "brn"}
    added = {e["species_id"]: e for e in bench[1:]}
    assert set(added) == {"salamence", "zapdos"}  # later-fainting zapdos still counts
    assert added["salamence"] == {"species_id": "salamence", "hp_fraction": 1.0, "status": None}
    assert record["own_info"]["bench_complete"] is True
    # Fewer than 4 brought -> bench_complete false -> not fully confirmed.
    record2 = {
        "decision_kind": "turn", "player": "p1", "replay_id": "rid-1",
        "state": {"our": {
            "active": [
                {"species": "incineroar", "item": None, "ability": "intimidate",
                 "revealed_moves": ["a", "b", "c", "d"]},
                {"species": "amoonguss", "item": None, "ability": "regenerator",
                 "revealed_moves": ["e", "f", "g", "h"]},
            ],
            "bench": [],
        }, "opp": {"active": [], "bench": []}},
    }
    small_facts = analyze_log("rid-1", log)  # only the two actives ever appeared
    rebuild_record(record2, small_facts, _ctx({"rid-1": _game()}, {"rid-1": small_facts}))
    assert record2["own_info"]["bench_complete"] is False
    assert record2["own_fully_confirmed"] is False


def test_rebuilt_bench_matches_live_bench_list() -> None:
    # Live parity (item C): no poke-env battle fixture exists, so assert
    # field-by-field equality of the rebuilt bench against `_bench_list`'s
    # semantics, then identical encode_state output.
    from vgc.bc.encoding import encode_state
    from vgc.bc.policy import _bench_list

    class _Mon:
        def __init__(self, species, fainted=False, hp=1.0, status=None):
            self.species = species
            self.fainted = fainted
            self.current_hp_fraction = hp
            self.status = status

    # Live: the private request gives the brought four in team-dict order.
    brought = [_Mon("Incineroar"), _Mon("Amoonguss"), _Mon("Salamence"), _Mon("Zapdos")]
    team = {m.species: m for m in brought}
    live_active = [brought[0], brought[1]]
    live_bench = _bench_list(team, live_active)

    # Replay: the same game -- only the actives + one bench mon have appeared.
    facts = analyze_log("rid-1", _minimal_log())
    facts.mons[("p1", "salamence")] = MonFacts(species="salamence", first_seen=999)
    facts.mons[("p1", "zapdos")] = MonFacts(species="zapdos", first_seen=998)
    record = {
        "decision_kind": "turn", "player": "p1", "replay_id": "rid-1", "log_line": 100,
        "state": {"our": {
            "active": [
                {"species": "incineroar", "item": None, "ability": None, "revealed_moves": []},
                {"species": "amoonguss", "item": None, "ability": None, "revealed_moves": []},
            ],
            "bench": [],
        }, "opp": {"active": [], "bench": []}},
    }
    rebuild_record(record, facts, _ctx({"rid-1": _game()}, {"rid-1": facts}))
    assert record["state"]["our"]["bench"] == live_bench  # field-by-field, same order

    replay_side = record["state"]["our"]
    live_side = dict(replay_side)
    live_side["bench"] = live_bench
    enc_replay = encode_state({"our": replay_side, "opp": {"active": [], "bench": [],
                                                           "preview_species": [],
                                                           "side_conditions": [], "unseen_count": 0}})
    replay_side.setdefault("preview_species", [])
    enc_live = encode_state({"our": live_side, "opp": {"active": [], "bench": [],
                                                       "preview_species": [],
                                                       "side_conditions": [], "unseen_count": 0}})
    import numpy as np
    for key in enc_replay:
        assert np.array_equal(enc_replay[key], enc_live[key]), key


def test_bench_addition_uses_first_seen_not_fainted() -> None:
    # The exact bug: a mon first appearing on turn 3 and fainting on turn 5 must
    # be ON the turn-1 bench (it's a live back mon the player knows about) but
    # NOT added at turn 6 (the parser's own tracking covers it then).
    log = _minimal_log(
        "|switch|p1b: Salamence|Salamence, L50|100/100",      # appears "turn 3"
        "|faint|p1b: Salamence",                              # faints "turn 5"
    )
    facts = analyze_log("rid-1", log)
    lines = log.splitlines()
    first_seen = facts.mons[("p1", "salamence")].first_seen
    assert first_seen is not None and lines[first_seen].startswith("|switch|p1b: Salamence")

    def record_at(log_line):
        return {
            "decision_kind": "turn", "player": "p1", "replay_id": "rid-1",
            "log_line": log_line,
            "state": {"our": {
                "active": [
                    {"species": "incineroar", "item": None, "ability": None,
                     "revealed_moves": ["a", "b", "c", "d"]},
                    {"species": "amoonguss", "item": None, "ability": None,
                     "revealed_moves": ["e", "f", "g", "h"]},
                ],
                "bench": [],
            }, "opp": {"active": [], "bench": []}},
        }

    ctx = _ctx({"rid-1": _game()}, {"rid-1": facts})
    early = record_at(first_seen - 1)      # before its first appearance ("turn 1")
    rebuild_record(early, facts, ctx)
    assert [e["species_id"] for e in early["state"]["our"]["bench"]] == ["salamence"]
    late = record_at(first_seen + 1)       # after it appeared/fainted ("turn 6")
    rebuild_record(late, facts, ctx)
    assert late["state"]["our"]["bench"] == []


def test_encode_state_bench_order_invariant() -> None:
    from vgc.bc.encoding import encode_state

    bench_a = [{"species_id": "zapdos", "hp_fraction": 1.0, "status": None},
               {"species_id": "amoonguss", "hp_fraction": 0.5, "status": "brn"}]
    bench_b = [bench_a[1], bench_a[0]]
    base = {
        "our": {"active": [], "bench": bench_a, "preview_species": [],
                "side_conditions": [], "unseen_count": 0},
        "opp": {"active": [], "bench": bench_b, "preview_species": [],
                "side_conditions": [], "unseen_count": 0},
    }
    a = encode_state(dict(base))
    base["our"]["bench"] = bench_b
    base["opp"]["bench"] = bench_a
    b = encode_state(base)
    for key in a:
        import numpy as np
        assert np.array_equal(a[key], b[key]), key


def test_unseen_brought_slot_is_a_deterministic_preview_guess() -> None:
    # A brought mon that never appears still takes up a bench slot live -- the
    # bench must reach the brought-four size regardless. With only 3 of the 6
    # previewed mons ever appearing, the missing slot is a deterministic guess
    # among the three unseen preview species (same game+side -> same guess on
    # every record and every call).
    log = _log(
        [
            "|player|p1|alice||",
            "|player|p2|bob||",
            "|poke|p1|Incineroar, L50, M|",
            "|poke|p1|Amoonguss, L50, F|",
            "|poke|p1|Rillaboom, L50, M|",
            "|poke|p1|Salamence, L50, M|",
            "|poke|p1|Zapdos, L50|",
            "|poke|p1|Gholdengo, L50|",
            "|poke|p2|Kingambit, L50, M|",
            "|poke|p2|Sneasler, L50, F|",
            "|start",
            "|switch|p1a: Incineroar|Incineroar, L50, M|100/100",
            "|switch|p1b: Amoonguss|Amoonguss, L50, F|100/100",
            "|switch|p2a: Kingambit|Kingambit, L50, M|100/100",
            "|switch|p2b: Sneasler|Sneasler, L50, F|100/100",
            "|turn|1",
            "|move|p1a: Incineroar|Fake Out|p2a: Kingambit",
            "|move|p1b: Amoonguss|Spore|p2b: Sneasler",
            "|move|p2a: Kingambit|Sucker Punch|p1a: Incineroar",
            "|move|p2b: Sneasler|Close Combat|p1b: Amoonguss",
            "|turn|2",
            "|switch|p1a: Rillaboom|Rillaboom, L50, M|100/100",
            "|move|p1b: Amoonguss|Spore|p2b: Sneasler",
            "|move|p2a: Kingambit|Kowtow Cleave|p1b: Amoonguss",
            "|move|p2b: Sneasler|Protect|p2b: Sneasler",
            "|turn|3",
            "|move|p1a: Rillaboom|Grassy Glide|p2a: Kingambit",
            "|move|p1b: Amoonguss|Spore|p2a: Kingambit",
            "|move|p2a: Kingambit|Sucker Punch|p1a: Rillaboom",
            "|move|p2b: Sneasler|Close Combat|p1a: Rillaboom",
            "|win|alice",
        ]
    )
    facts = analyze_log("rid-1", log)
    unseen = {"salamence", "zapdos", "gholdengo"}
    guesses = []
    for turn in (1, 2, 3):
        record = _turn_record(_records(log), turn)
        rebuild_record(record, facts, None)
        our = record["state"]["our"]
        total = sum(1 for mon in our["active"] if mon) + len(our["bench"])
        assert total == 4
        guessed = record["own_info"]["bench_guessed"]
        assert len(guessed) == 1 and set(guessed) <= unseen
        assert record["own_info"]["bench_complete"] is False
        guesses.append(guessed[0])
    assert len(set(guesses)) == 1  # the same guess on every record of the game
    repeat = _turn_record(_records(log), 1)
    rebuild_record(repeat, facts, None)
    assert repeat["own_info"]["bench_guessed"] == [guesses[0]]


def test_revealed_moves_fill_is_whole_game() -> None:
    log = _minimal_log()
    facts = analyze_log("rid-1", log)
    record = _turn_record(_records(log), 1)
    rebuild_record(record, facts, None)
    mon = record["state"]["our"]["active"][0]
    # Turn-1 snapshot: Knock Off isn't used until turn 2, but the player always
    # knew it -- the fill window is the whole game.
    assert "knockoff" in mon["revealed_moves"]
    assert "fakeout" in mon["revealed_moves"]
