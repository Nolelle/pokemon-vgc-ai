"""Unit tests for `vgc.replay_parse`, using small hand-built protocol-log snippets plus
one small REAL downloaded replay (`tests/fixtures/replay_sample.json`) for an end-to-end
sanity check. See `vgc.replay_parse`'s module docstring for the parsing design and the
showteam-coverage finding these tests were built alongside.
"""

from __future__ import annotations

import json
from pathlib import Path

from vgc.replay_parse import parse_replay, parse_showteam_line

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "replay_sample.json"


def _log(lines: list[str]) -> str:
    return "\n".join(lines) + "\n"


def _turn_records(result, player: str) -> list[dict]:
    return [r for r in result.records if r["decision_kind"] == "turn" and r["player"] == player]


def _teampreview_record(result, player: str) -> dict:
    return next(
        r for r in result.records if r["decision_kind"] == "teampreview" and r["player"] == player
    )


# --- real fixture: end-to-end sanity check --------------------------------------------


def test_real_fixture_parses_end_to_end() -> None:
    payload = json.loads(FIXTURE_PATH.read_text())
    result = parse_replay(payload["id"], payload.get("rating"), payload["log"])

    assert result.ok is True
    assert result.fail_reason is None

    kinds = [r["decision_kind"] for r in result.records]
    assert kinds.count("teampreview") == 2
    assert kinds.count("turn") >= 2  # both players' turn-1 decisions, at minimum

    p1_turn1 = next(
        r
        for r in result.records
        if r["decision_kind"] == "turn" and r["player"] == "p1" and r["turn"] == 1
    )
    # p1 actually switched Scizor -> Torkoal as their turn-1 slot-1 action (revealing
    # Drought weather) while slot 0 used Fake Out -- confirms both switch-as-turn-action
    # and move-as-turn-action are attributed correctly from the same real log.
    assert p1_turn1["action"]["slot1"] == {"kind": "switch", "switch_species": "torkoal"}
    assert p1_turn1["action"]["slot0"]["kind"] == "move"
    assert p1_turn1["action"]["slot0"]["move_id"] == "fakeout"

    for record in result.records:
        assert record["replay_id"] == payload["id"]
        assert record["rating"] == payload["rating"]


# --- showteam extraction ---------------------------------------------------------------


def test_parse_showteam_line_extracts_sets() -> None:
    line = (
        "|showteam|p1|Garchomp||LifeOrb|RoughSkin|DragonClaw,Earthquake,IronHead,Protect"
        "|Jolly||M|||50|]Klefki||Leftovers|Prankster|SpikyShield,FoulPlay,Reflect,LightScreen"
        "|Bold||F|||50|"
    )
    result = parse_showteam_line(line.split("|"))
    assert result is not None
    side, sets = result
    assert side == "p1"
    assert sets["garchomp"]["item"] == "lifeorb"
    assert sets["garchomp"]["ability"] == "roughskin"
    assert sets["garchomp"]["moves"] == ["dragonclaw", "earthquake", "ironhead", "protect"]
    assert sets["garchomp"]["nature"] == "jolly"
    assert sets["klefki"]["item"] == "leftovers"
    assert sets["klefki"]["moves"][0] == "spikyshield"


def test_showteam_attached_to_first_record_for_that_player() -> None:
    log = _log(
        [
            "|gen|9",
            "|poke|p1|Garchomp, L50, M|",
            "|poke|p1|Klefki, L50, F|",
            "|poke|p2|Charizard, L50, M|",
            "|poke|p2|Incineroar, L50, F|",
            "|teampreview|4",
            "|showteam|p1|Garchomp||LifeOrb|RoughSkin|DragonClaw,Earthquake,IronHead,Protect"
            "|Jolly||M|||50|]Klefki||Leftovers|Prankster|SpikyShield,FoulPlay,Reflect,LightScreen"
            "|Bold||F|||50|",
            "|t:|1000",
            "|start",
            "|switch|p1a: Garchomp|Garchomp, L50, M|100/100",
            "|switch|p1b: Klefki|Klefki, L50, F|100/100",
            "|switch|p2a: Charizard|Charizard, L50, M|100/100",
            "|switch|p2b: Incineroar|Incineroar, L50, F|100/100",
            "|turn|1",
            "|win|test",
        ]
    )
    result = parse_replay("test-showteam", 1300, log)

    assert result.ok is True
    assert result.showteam_players == {"p1"}

    p1_records = [r for r in result.records if r["player"] == "p1"]
    p2_records = [r for r in result.records if r["player"] == "p2"]
    # Only the FIRST p1 record (teampreview, turn 0) carries "sets" -- not every record.
    assert "sets" in p1_records[0]
    assert p1_records[0]["sets"]["garchomp"]["item"] == "lifeorb"
    assert all("sets" not in r for r in p1_records[1:])
    assert all("sets" not in r for r in p2_records)


# --- a simple 2-turn doubles exchange: 2 records/player, correct actions/targets -----


def _two_turn_log() -> str:
    return _log(
        [
            "|gen|9",
            "|poke|p1|Garchomp, L50, M|",
            "|poke|p1|Klefki, L50, F|",
            "|poke|p2|Charizard, L50, M|",
            "|poke|p2|Incineroar, L50, F|",
            "|teampreview|4",
            "|t:|1000",
            "|start",
            "|switch|p1a: Garchomp|Garchomp, L50, M|100/100",
            "|switch|p1b: Klefki|Klefki, L50, F|100/100",
            "|switch|p2a: Charizard|Charizard, L50, M|100/100",
            "|switch|p2b: Incineroar|Incineroar, L50, F|100/100",
            "|turn|1",
            "|t:|1001",
            "|move|p1a: Garchomp|Dragon Claw|p2a: Charizard",
            "|-damage|p2a: Charizard|70/100",
            "|move|p1b: Klefki|Protect|p1b: Klefki",
            "|move|p2a: Charizard|Heat Wave|p1a: Garchomp|[spread] p1a,p1b",
            "|-damage|p1a: Garchomp|80/100",
            "|move|p2b: Incineroar|Fake Out|p1a: Garchomp",
            "|-damage|p1a: Garchomp|70/100",
            "|upkeep",
            "|turn|2",
            "|t:|1002",
            "|move|p1a: Garchomp|Earthquake|p2a: Charizard|[spread] p2a,p2b",
            "|-damage|p2a: Charizard|40/100",
            "|-damage|p2b: Incineroar|90/100",
            "|move|p1b: Klefki|Foul Play|p2b: Incineroar",
            "|-damage|p2b: Incineroar|75/100",
            "|move|p2a: Charizard|Solar Beam|p1a: Garchomp",
            "|-damage|p1a: Garchomp|20/100",
            "|move|p2b: Incineroar|Parting Shot|p1b: Klefki",
            "|-unboost|p1b: Klefki|atk|1",
            "|-unboost|p1b: Klefki|spa|1",
            "|upkeep",
            "|win|test",
        ]
    )


def test_two_turn_exchange_produces_two_records_per_player() -> None:
    result = parse_replay("test-two-turn", 1400, _two_turn_log())
    assert result.ok is True

    p1_turns = _turn_records(result, "p1")
    p2_turns = _turn_records(result, "p2")
    assert len(p1_turns) == 2
    assert len(p2_turns) == 2
    assert [r["turn"] for r in p1_turns] == [1, 2]


def test_two_turn_exchange_actions_and_targets_are_correct() -> None:
    result = parse_replay("test-two-turn", 1400, _two_turn_log())
    p1_turn1, p1_turn2 = _turn_records(result, "p1")

    assert p1_turn1["action"]["slot0"] == {
        "kind": "move",
        "move_id": "dragonclaw",
        "target_slot": "opp0",
        "mega": False,
    }
    assert p1_turn1["action"]["slot1"] == {
        "kind": "move",
        "move_id": "protect",
        "target_slot": "self",
        "mega": False,
    }
    assert p1_turn2["action"]["slot0"]["move_id"] == "earthquake"
    assert p1_turn2["action"]["slot1"] == {
        "kind": "move",
        "move_id": "foulplay",
        "target_slot": "opp1",
        "mega": False,
    }


def test_hp_fraction_tracked_across_turns_via_damage() -> None:
    result = parse_replay("test-two-turn", 1400, _two_turn_log())
    p1_turn2 = _turn_records(result, "p1")[1]
    # Garchomp took 30 (Heat Wave spread) then 10 (Fake Out) damage during turn 1 --
    # turn 2's snapshot (captured before turn 2's own events) must reflect that.
    assert p1_turn2["state"]["our"]["active"][0]["species"] == "garchomp"
    assert p1_turn2["state"]["our"]["active"][0]["hp_fraction"] == 0.70


# --- spread-move target marking ---------------------------------------------------------


def test_spread_move_marks_target_slot_spread() -> None:
    result = parse_replay("test-two-turn", 1400, _two_turn_log())
    p1_turn2 = _turn_records(result, "p1")[1]
    assert p1_turn2["action"]["slot0"]["target_slot"] == "spread"
    p2_turn1 = _turn_records(result, "p2")[0]
    assert p2_turn1["action"]["slot0"]["target_slot"] == "spread"  # Charizard's Heat Wave


# --- forced-switch record between turns (faint mid-turn) ------------------------------


def test_forced_switch_after_faint_produces_its_own_record() -> None:
    log = _log(
        [
            "|gen|9",
            "|poke|p1|Garchomp, L50, M|",
            "|poke|p1|Klefki, L50, F|",
            "|poke|p1|Gholdengo, L50|",
            "|poke|p2|Charizard, L50, M|",
            "|poke|p2|Incineroar, L50, F|",
            "|teampreview|4",
            "|t:|1000",
            "|start",
            "|switch|p1a: Garchomp|Garchomp, L50, M|100/100",
            "|switch|p1b: Klefki|Klefki, L50, F|1/100",
            "|switch|p2a: Charizard|Charizard, L50, M|100/100",
            "|switch|p2b: Incineroar|Incineroar, L50, F|100/100",
            "|turn|1",
            "|t:|1001",
            "|move|p2a: Charizard|Heat Wave|p1b: Klefki",
            "|-damage|p1b: Klefki|0 fnt",
            "|faint|p1b: Klefki",
            "|move|p1a: Garchomp|Dragon Claw|p2a: Charizard",
            "|-damage|p2a: Charizard|60/100",
            "|move|p2b: Incineroar|Fake Out|p1a: Garchomp",
            "|-damage|p1a: Garchomp|90/100",
            "|",
            "|t:|1002",
            "|switch|p1b: Gholdengo|Gholdengo, L50|100/100",
            "|upkeep",
            "|turn|2",
            "|win|test",
        ]
    )
    result = parse_replay("test-forced-switch", 1400, log)
    assert result.ok is True

    forced = [r for r in result.records if r["decision_kind"] == "forced_switch"]
    assert len(forced) == 1
    assert forced[0]["player"] == "p1"
    assert forced[0]["turn"] == 1
    assert forced[0]["action"] == {"slot": 1, "switch_species": "gholdengo"}

    # The fainted slot's turn-1 "turn" record shows "pass" -- Klefki never got a chance
    # to act (it fainted before any |move| line for it), and the Gholdengo switch-in is
    # its OWN forced_switch record, not folded into this one.
    p1_turn1 = _turn_records(result, "p1")[0]
    assert p1_turn1["action"]["slot1"] == {"kind": "pass"}


# --- HP fraction tracking through damage AND heal --------------------------------------


def test_hp_fraction_tracks_damage_and_heal() -> None:
    log = _log(
        [
            "|gen|9",
            "|poke|p1|Garchomp, L50, M|",
            "|poke|p2|Charizard, L50, M|",
            "|teampreview|4",
            "|t:|1000",
            "|start",
            "|switch|p1a: Garchomp|Garchomp, L50, M|100/100",
            "|switch|p2a: Charizard|Charizard, L50, M|100/100",
            "|turn|1",
            "|t:|1001",
            "|move|p2a: Charizard|Heat Wave|p1a: Garchomp",
            "|-damage|p1a: Garchomp|55/100",
            "|-heal|p1a: Garchomp|75/100|[from] item: Sitrus Berry",
            "|move|p1a: Garchomp|Dragon Claw|p2a: Charizard",
            "|upkeep",
            "|turn|2",
            "|win|test",
        ]
    )
    result = parse_replay("test-hp-tracking", 1400, log)
    assert result.ok is True
    p1_turn2 = _turn_records(result, "p1")[1]
    assert p1_turn2["state"]["our"]["active"][0]["hp_fraction"] == 0.75


# --- mega flag attribution -------------------------------------------------------------


def test_mega_flag_attributed_to_the_acting_slots_move() -> None:
    log = _log(
        [
            "|gen|9",
            "|poke|p1|Garchomp, L50, M|",
            "|poke|p2|Charizard, L50, M|",
            "|teampreview|4",
            "|t:|1000",
            "|start",
            "|switch|p1a: Garchomp|Garchomp, L50, M|100/100",
            "|switch|p2a: Charizard|Charizard, L50, M|100/100",
            "|turn|1",
            "|t:|1001",
            "|detailschange|p1a: Garchomp|Garchomp-Mega, L50, M",
            "|-mega|p1a: Garchomp|Garchomp|Garchompite",
            "|move|p1a: Garchomp|Dragon Claw|p2a: Charizard",
            "|move|p2a: Charizard|Flamethrower|p1a: Garchomp",
            "|upkeep",
            "|turn|2",
            "|win|test",
        ]
    )
    result = parse_replay("test-mega", 1400, log)
    assert result.ok is True
    p1_turn1 = _turn_records(result, "p1")[0]
    assert p1_turn1["action"]["slot0"]["mega"] is True
    assert p1_turn1["action"]["slot0"]["move_id"] == "dragonclaw"
    # The PRE-decision snapshot (captured before turn 1's own events) must NOT already
    # show the mega -- it hasn't happened yet at the moment this decision was made.
    assert p1_turn1["state"]["our"]["active"][0]["mega"] is False


# --- skip-not-crash contract -------------------------------------------------------------


def test_malformed_events_are_skipped_not_fatal() -> None:
    log = _log(
        [
            "|gen|9",
            "|poke|p1|Garchomp, L50, M|",
            "|poke|p2|Charizard, L50, M|",
            "|teampreview|4",
            "|t:|1000",
            "|start",
            "|switch|p1a: Garchomp|Garchomp, L50, M|100/100",
            "|switch|p2a: Charizard|Charizard, L50, M|100/100",
            "|turn|1",
            "|t:|1001",
            "|move|p1a: Garchomp",  # malformed: truncated, no move name/target
            "|switch|weird",  # malformed: truncated, no details/HP
            "|move|p2a: Charizard|Flamethrower|p1a: Garchomp",
            "|upkeep",
            "|turn|2",
            "|win|test",
        ]
    )
    result = parse_replay("test-malformed", 1400, log)

    assert result.ok is True
    assert result.fail_reason is None
    assert result.skipped["malformed_move"] >= 1
    assert result.skipped["malformed_switch"] >= 1

    p1_turn1 = _turn_records(result, "p1")[0]
    # The malformed move line never registered a real action for p1's slot 0.
    assert p1_turn1["action"]["slot0"] == {"kind": "pass"}
    p2_turn1 = _turn_records(result, "p2")[0]
    assert p2_turn1["action"]["slot0"]["move_id"] == "flamethrower"


def test_parse_replay_never_raises_on_a_totally_broken_log() -> None:
    result = parse_replay("test-broken", 1400, None)  # type: ignore[arg-type]
    assert result.ok is False
    assert result.fail_reason is not None
    assert result.records == []


# --- schema 2: schema marker, revealed_moves, bench identity --------------------------


def test_schema_version_present_on_every_record() -> None:
    result = parse_replay("test-two-turn", 1400, _two_turn_log())
    assert result.ok is True
    assert result.records  # sanity: there's something to check
    assert all(record["schema"] == 4 for record in result.records)


def test_revealed_moves_accumulate_from_moves_actually_used() -> None:
    result = parse_replay("test-two-turn", 1400, _two_turn_log())
    p1_turn1, p1_turn2 = _turn_records(result, "p1")
    # Turn 1's snapshot is pre-turn-1 -- Garchomp hasn't used anything yet.
    assert p1_turn1["state"]["our"]["active"][0]["revealed_moves"] == []
    # Turn 2's snapshot reflects turn 1's Dragon Claw having been used.
    assert p1_turn2["state"]["our"]["active"][0]["revealed_moves"] == ["dragonclaw"]
    # Klefki used Protect turn 1.
    assert p1_turn2["state"]["our"]["active"][1]["revealed_moves"] == ["protect"]


def test_revealed_moves_includes_a_blocked_cant_attempted_move() -> None:
    log = _log(
        [
            "|gen|9",
            "|poke|p1|Garchomp, L50, M|",
            "|poke|p2|Charizard, L50, M|",
            "|teampreview|4",
            "|t:|1000",
            "|start",
            "|switch|p1a: Garchomp|Garchomp, L50, M|100/100",
            "|switch|p2a: Charizard|Charizard, L50, M|100/100",
            "|turn|1",
            "|t:|1001",
            "|move|p2a: Charizard|Fake Out|p1a: Garchomp",
            "|-damage|p1a: Garchomp|95/100",
            "|cant|p1a: Garchomp|flinch|Dragon Claw",
            "|upkeep",
            "|turn|2",
            "|win|test",
        ]
    )
    result = parse_replay("test-cant-reveal", 1400, log)
    assert result.ok is True
    p1_turn2 = _turn_records(result, "p1")[1]
    # Dragon Claw never connected (flinched) but the player DID choose it -- it's
    # revealed all the same, per the module docstring's "cant" reasoning.
    assert p1_turn2["state"]["our"]["active"][0]["revealed_moves"] == ["dragonclaw"]


def test_bench_entries_have_species_id_hp_fraction_and_status() -> None:
    log = _log(
        [
            "|gen|9",
            "|poke|p1|Garchomp, L50, M|",
            "|poke|p1|Klefki, L50, F|",
            "|poke|p1|Incineroar, L50, F|",
            "|poke|p2|Charizard, L50, M|",
            "|teampreview|4",
            "|t:|1000",
            "|start",
            "|switch|p1a: Garchomp|Garchomp, L50, M|100/100",
            "|switch|p1b: Klefki|Klefki, L50, F|100/100",
            "|switch|p2a: Charizard|Charizard, L50, M|100/100",
            "|turn|1",
            "|t:|1001",
            "|move|p2a: Charizard|Toxic|p1b: Klefki",
            "|-status|p1b: Klefki|tox",
            "|move|p1a: Garchomp|Dragon Claw|p2a: Charizard",
            "|upkeep",
            "|turn|2",
            "|t:|1002",
            "|switch|p1b: Incineroar|Incineroar, L50, F|100/100",  # chosen swap -- Klefki to bench
            "|move|p2a: Charizard|Flamethrower|p1a: Garchomp",
            "|upkeep",
            "|turn|3",
            "|win|test",
        ]
    )
    result = parse_replay("test-bench-identity", 1400, log)
    assert result.ok is True
    p1_turn3 = _turn_records(result, "p1")[2]
    bench = p1_turn3["state"]["our"]["bench"]
    assert bench == [{"species_id": "klefki", "hp_fraction": 1.0, "status": "tox"}]


# --- schema 3: winner resolution + per-record "won" ----------------------------------


def _log_with_players(lines: list[str], *, p1_name: str = "alice", p2_name: str = "bob") -> str:
    return _log(
        [
            "|gen|9",
            f"|player|p1|{p1_name}|101|1500",
            f"|player|p2|{p2_name}|102|1500",
            "|poke|p1|Garchomp, L50, M|",
            "|poke|p2|Charizard, L50, M|",
            "|teampreview|4",
            "|t:|1000",
            "|start",
            "|switch|p1a: Garchomp|Garchomp, L50, M|100/100",
            "|switch|p2a: Charizard|Charizard, L50, M|100/100",
            "|turn|1",
            *lines,
        ]
    )


def test_win_line_resolves_winner_to_the_matching_side() -> None:
    log = _log_with_players(["|win|bob"])
    result = parse_replay("test-win-p2", 1400, log)
    assert result.ok is True
    assert result.winner == "p2"


def test_win_line_marks_only_the_winning_players_records_as_won() -> None:
    log = _log_with_players(["|win|alice"])
    result = parse_replay("test-win-p1", 1400, log)
    assert result.winner == "p1"
    assert all(record["won"] is True for record in result.records if record["player"] == "p1")
    assert all(record["won"] is False for record in result.records if record["player"] == "p2")


def test_tie_line_leaves_winner_none_and_every_record_unwon() -> None:
    log = _log_with_players(["|tie|"])
    result = parse_replay("test-tie", 1400, log)
    assert result.winner is None
    assert all(record["won"] is False for record in result.records)


def test_win_line_with_unresolvable_name_leaves_winner_none() -> None:
    log = _log_with_players(["|win|someone-else-entirely"])
    result = parse_replay("test-win-unresolved", 1400, log)
    assert result.winner is None
    assert result.skipped["unresolved_winner_name"] == 1
    assert all(record["won"] is False for record in result.records)


def test_no_player_lines_at_all_leaves_winner_none_gracefully() -> None:
    # Every OTHER fixture log in this file omits |player| lines entirely -- confirms
    # that's handled gracefully (not just incidentally not-crashing).
    result = parse_replay("test-two-turn", 1400, _two_turn_log())
    assert result.ok is True
    assert result.winner is None
    assert all(record["won"] is False for record in result.records)


def test_every_record_carries_a_won_key_regardless_of_decision_kind() -> None:
    result = parse_replay("test-two-turn", 1400, _two_turn_log())
    assert result.records
    assert all("won" in record for record in result.records)
    kinds = {record["decision_kind"] for record in result.records}
    assert "teampreview" in kinds and "turn" in kinds


def test_real_fixture_winner_matches_the_actual_replay_outcome() -> None:
    payload = json.loads(FIXTURE_PATH.read_text())
    result = parse_replay(payload["id"], payload.get("rating"), payload["log"])
    assert result.ok is True
    # tests/fixtures/replay_sample.json: |player|p1|Marihuano0503|..., |player|p2|
    # pcrlbot0421735f7b|..., |win|pcrlbot0421735f7b -- p2 won.
    assert result.winner == "p2"
    p2_records = [r for r in result.records if r["player"] == "p2"]
    p1_records = [r for r in result.records if r["player"] == "p1"]
    assert p2_records and all(r["won"] is True for r in p2_records)
    assert p1_records and all(r["won"] is False for r in p1_records)


# --- schema 4: preview_species + unseen_count on turn/forced_switch state -------------


def _preview_log(extra_lines: list[str]) -> str:
    return _log(
        [
            "|gen|9",
            "|poke|p1|Garchomp, L50, M|",
            "|poke|p1|Klefki, L50, F|",
            "|poke|p1|Incineroar, L50, F|",
            "|poke|p2|Charizard, L50, M|",
            "|poke|p2|Gholdengo, L50|",
            "|teampreview|4",
            "|t:|1000",
            "|start",
            "|switch|p1a: Garchomp|Garchomp, L50, M|100/100",
            "|switch|p1b: Klefki|Klefki, L50, F|100/100",
            "|switch|p2a: Charizard|Charizard, L50, M|100/100",
            *extra_lines,
        ]
    )


def test_side_dict_includes_preview_species_in_poke_line_order() -> None:
    log = _preview_log(["|turn|1", "|win|test"])
    result = parse_replay("test-preview-species", 1400, log)
    assert result.ok is True
    p1_turn1 = _turn_records(result, "p1")[0]
    assert p1_turn1["state"]["our"]["preview_species"] == ["garchomp", "klefki", "incineroar"]
    assert p1_turn1["state"]["opp"]["preview_species"] == ["charizard", "gholdengo"]


def test_unseen_count_reflects_previewed_but_never_appeared_species() -> None:
    log = _preview_log(["|turn|1", "|win|test"])
    result = parse_replay("test-unseen-count", 1400, log)
    p1_turn1 = _turn_records(result, "p1")[0]
    # Garchomp and Klefki led (appeared); Incineroar never showed up in this short log.
    assert p1_turn1["state"]["our"]["unseen_count"] == 1
    # p2's Gholdengo was previewed but only Charizard led -- 1 still unseen.
    assert p1_turn1["state"]["opp"]["unseen_count"] == 1


def test_unseen_count_drops_once_a_previewed_bench_mon_appears() -> None:
    log = _preview_log(
        [
            "|turn|1",
            "|t:|1001",
            "|switch|p1a: Incineroar|Incineroar, L50, F|100/100",
            "|upkeep",
            "|turn|2",
            "|win|test",
        ]
    )
    result = parse_replay("test-unseen-count-drops", 1400, log)
    p1_turn1, p1_turn2 = _turn_records(result, "p1")
    assert p1_turn1["state"]["our"]["unseen_count"] == 1  # pre-switch: Incineroar unseen
    assert p1_turn2["state"]["our"]["unseen_count"] == 0  # post-switch: all 3 previewed seen


def test_forced_switch_record_also_carries_preview_species_and_unseen_count() -> None:
    log = _preview_log(
        [
            "|switch|p2b: Gholdengo|Gholdengo, L50|100/100",
            "|turn|1",
            "|t:|1001",
            "|move|p2a: Charizard|Heat Wave|p1a: Garchomp",
            "|-damage|p1a: Garchomp|0 fnt",
            "|faint|p1a: Garchomp",
            "|move|p1b: Klefki|Protect|p1b: Klefki",
            "|move|p2b: Gholdengo|Shadow Ball|p1b: Klefki",
            "|",
            "|t:|1002",
            "|switch|p1a: Incineroar|Incineroar, L50, F|100/100",
            "|upkeep",
            "|turn|2",
            "|win|test",
        ]
    )
    result = parse_replay("test-forced-switch-preview", 1400, log)
    forced = next(r for r in result.records if r["decision_kind"] == "forced_switch")
    assert forced["state"]["our"]["preview_species"] == ["garchomp", "klefki", "incineroar"]
    # Snapshot is taken BEFORE the replacement switch is applied -- Incineroar still unseen.
    assert forced["state"]["our"]["unseen_count"] == 1


def test_preview_species_defensively_resolves_mega_formes_via_resolve_species() -> None:
    # Not realistic real-corpus data (Mega Evolution can't happen before turn 1) but the
    # module docstring documents this resolution as defensive -- exercise it directly.
    log = _log(
        [
            "|gen|9",
            "|poke|p1|Garchomp-Mega, L50, M|",
            "|poke|p2|Charizard, L50, M|",
            "|teampreview|4",
            "|t:|1000",
            "|start",
            "|switch|p1a: Garchomp|Garchomp, L50, M|100/100",
            "|switch|p2a: Charizard|Charizard, L50, M|100/100",
            "|turn|1",
            "|win|test",
        ]
    )
    result = parse_replay("test-preview-mega-resolve", 1400, log)
    assert result.ok is True
    p1_turn1 = _turn_records(result, "p1")[0]
    assert p1_turn1["state"]["our"]["preview_species"] == ["garchomp"]
