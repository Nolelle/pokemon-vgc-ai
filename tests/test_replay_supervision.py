from vgc.bc.replay_supervision import audit_raw_log, build_supervision

BASE = """|player|p1|Alice
|player|p2|Bob
|poke|p1|Pikachu, L50
|poke|p1|Charizard, L50
|switch|p1a: Pikachu|Pikachu, L50|100/100
|switch|p1b: Charizard|Charizard, L50|100/100
|turn|1
"""


def record(kind="turn"):
    return {"decision_kind": kind, "player": "p1", "turn": 1,
            "state": {"our": {"preview": ["pikachu", "charizard"]}, "opp": {}, "field": {}},
            "action": {"slot0": {"kind": "move", "move_id": "thunderbolt", "target_slot": 1},
                       "slot1": {"kind": "unknown"}},
            "action_status": {"slot0": "observed", "slot1": "unknown"},
            "outcome": "win", "won": True}


def build(log, row=None, ending="played_out"):
    return build_supervision(row or record(), {"winner": "p1", "ending": ending}, audit_raw_log(log))


def test_observed_move_and_unknown_never_pass():
    log = BASE + "|move|p1a: Pikachu|Thunderbolt|p2a: Foe\n|win|Alice\n"
    result = build(log)
    assert result["targets"]["move"] == ["thunderbolt", None]
    assert result["masks"]["target"] == [False, False]
    assert result["masks"]["outcome"]
    row = record()
    row["action"]["slot1"] = {"kind": "no_action_required"}
    row["action_status"]["slot1"] = "no_action_required"
    assert build(log, row)["targets"]["move"] == result["targets"]["move"]
    row["action"]["slot1"] = {"kind": "unknown", "placeholder": "different"}
    assert build(log, row)["targets"]["move"] == result["targets"]["move"]


def test_called_locked_and_raw_mismatch_mask_move():
    for event in ("|move|p1a: Pikachu|Thunderbolt|p2a: Foe|[from] move: Sleep Talk",
                  "|-start|p1a: Pikachu|Encore\n|move|p1a: Pikachu|Thunderbolt|p2a: Foe",
                  "|move|p1a: Pikachu|Protect|p1a: Pikachu"):
        assert build(BASE + event + "\n|win|Alice\n")["masks"]["move"] == [False, False]


def test_switch_only_before_resolving_actions_and_never_drag():
    row = record()
    row["action"]["slot0"] = {"kind": "switch", "switch_species": "pikachu"}
    switch = "|switch|p1a: Pikachu|Pikachu, L50|100/100\n"
    assert build(BASE + switch + "|win|Alice\n", row)["masks"]["switch_destination"][0]
    for prefix in ("|move|p2a: Foe|Protect|p2a: Foe\n", "|faint|p1a: Pikachu\n",
                   "|drag|p2a: Foe|Charizard, L50|100/100\n",
                   "|-enditem|p1a: Pikachu|Eject Pack\n"):
        assert not build(BASE + prefix + switch + "|win|Alice\n", row)["masks"]["switch_destination"][0]
    assert not build(BASE + switch.replace("|switch|", "|drag|") + "|win|Alice\n", row)["masks"]["switch_destination"][0]
    row["decision_kind"] = "forced_switch"
    assert not build(BASE + switch + "|win|Alice\n", row)["masks"]["switch_destination"][0]


def test_timeout_masks_outcome_but_move_independent():
    log = BASE + "|move|p1a: Pikachu|Thunderbolt|p2a: Foe\n|-message|Bob lost due to inactivity.\n|win|Alice\n"
    result = build(log, ending="timer_offline")
    assert result["masks"]["move"][0]
    assert not result["masks"]["outcome"]


def test_partial_preview_leads_and_no_brought_label():
    row = record("teampreview")
    row["action"] = {"picked": ["pikachu", "charizard"], "lead_order": ["pikachu", "charizard"]}
    audit = audit_raw_log(BASE + "|win|Alice\n")
    metadata = {"winner": "p1", "ending": "played_out", "expected_preview": {"p1": row["action"]}}
    result = build_supervision(row, metadata, audit)
    assert result["targets"]["preview_leads"] == ["pikachu", "charizard"]
    assert result["targets"]["preview_brought"] is None


def test_lock_detection_uses_protocol_fields_and_catalog_not_chat():
    ordinary = BASE + "|c|Fly|encore outrage solar beam\n|move|p1a: Pikachu|Thunderbolt|p2a: Foe\n|win|Alice\n"
    assert build(ordinary)["masks"]["move"][0]
    # An opponent's charge move locks THEIR slot, not ours -- Pikachu's fresh
    # Thunderbolt choice is still labelled.
    charged = BASE + "|move|p2a: Foe|Freeze Shock|p1a: Pikachu\n|move|p1a: Pikachu|Thunderbolt|p2a: Foe\n|win|Alice\n"
    assert build(charged)["masks"]["move"][0]


def test_ally_switch_labels_follow_decision_start_occupants_in_both_directions():
    from vgc.replay_parse import parse_replay

    log = BASE.replace("Pikachu", "Houndstone").replace("Charizard", "Baxcalibur") + (
        "|move|p1a: Houndstone|Ally Switch|p1a: Houndstone\n"
        "|swap|p1a: Houndstone|1|[from] move: Ally Switch\n"
        "|move|p1a: Baxcalibur|Dragon Dance|p1a: Baxcalibur\n"
        "|turn|2\n"
        "|move|p1b: Houndstone|Ally Switch|p1b: Houndstone\n"
        "|swap|p1b: Houndstone|0|[from] move: Ally Switch\n"
        "|move|p1b: Baxcalibur|Dragon Claw|p2a: Foe\n"
        "|win|Alice\n")
    parsed = parse_replay("swap", None, log)
    assert parsed.ok
    rows = [r for r in parsed.records if r["player"] == "p1" and r["decision_kind"] == "turn"]
    for row, expected in zip(rows, (["allyswitch", "dragondance"], ["dragonclaw", "allyswitch"])):
        assert [row["action"][f"slot{i}"]["move_id"] for i in range(2)] == expected
        supervision = build(log, row)
        assert supervision["masks"]["move"] == [True, True]
        assert supervision["targets"]["move"] == expected
    assert rows[1]["state"]["our"]["active"][0]["species"] == "baxcalibur"
    assert "dragondance" in rows[1]["state"]["our"]["active"][0]["revealed_moves"]


def test_struggle_is_not_a_native_move_or_policy_label():
    from vgc.replay_parse import parse_replay

    log = BASE + "|move|p1a: Pikachu|Struggle|p2a: Foe\n|turn|2\n|win|Alice\n"
    rows = [r for r in parse_replay("struggle", None, log).records
            if r["player"] == "p1" and r["decision_kind"] == "turn"]
    supervision = build(log, rows[0])
    assert not supervision["masks"]["move"][0]
    assert supervision["targets"]["move"][0] is None
    assert supervision["reasons"]["move"][0] == "forced_struggle_not_choice"
    assert "struggle" not in rows[1]["state"]["our"]["active"][0]["revealed_moves"]


def test_instruct_repetition_without_from_annotation_is_not_a_choice():
    from vgc.replay_parse import parse_replay

    log = BASE.replace("Pikachu", "Oranguru").replace("Charizard", "Snorlax") + (
        "|move|p1a: Oranguru|Protect|p1a: Oranguru\n"
        "|move|p1b: Snorlax|Curse|p1b: Snorlax\n|turn|2\n"
        "|move|p1a: Oranguru|Instruct|p1b: Snorlax\n"
        "|-singleturn|p1b: Snorlax|move: Instruct|[of] p1a: Oranguru\n"
        "|move|p1b: Snorlax|Curse|p1b: Snorlax\n"
        "|cant|p1b: Snorlax|move: Taunt|Yawn\n|win|Alice\n")
    audit = audit_raw_log(log)
    assert audit["locked_or_called"]
    rows = [r for r in parse_replay("instruct", None, log).records
            if r["player"] == "p1" and r["decision_kind"] == "turn"]
    first, second = rows
    # Turn 1: both moves were fresh choices -> both labelled.
    result = build_supervision(first, {"winner": "p1", "ending": "played_out"}, audit)
    assert result["masks"]["move"] == [True, True]
    assert result["targets"]["move"] == ["protect", "curse"]
    # Turn 2: Oranguru's Instruct is its own choice; Snorlax's Instruct-driven
    # repeat carries no [from] but is locked by the -singleturn volatile.
    result = build_supervision(second, {"winner": "p1", "ending": "played_out"}, audit)
    assert result["masks"]["move"] == [True, False]
    assert result["targets"]["move"] == ["instruct", None]


def test_charge_move_first_turn_labels_second_turn_locks():
    from vgc.replay_parse import parse_replay

    log = BASE + (
        "|move|p1a: Pikachu|Solar Beam||[still]\n"
        "|-prepare|p1a: Pikachu|Solar Beam\n"
        "|move|p1b: Charizard|Flamethrower|p2a: Foe\n|turn|2\n"
        "|move|p1a: Pikachu|Solar Beam|p2a: Foe|[from] lockedmove\n"
        "|move|p1b: Charizard|Dragon Claw|p2a: Foe\n|win|Alice\n")
    audit = audit_raw_log(log)
    rows = [r for r in parse_replay("charge", None, log).records
            if r["player"] == "p1" and r["decision_kind"] == "turn"]
    first, second = rows
    # The charge turn IS a fresh choice -- per-turn locks no longer mask it.
    result = build_supervision(first, {"winner": "p1", "ending": "played_out"}, audit)
    assert result["masks"]["move"] == [True, True]
    assert result["targets"]["move"] == ["solarbeam", "flamethrower"]
    # The firing turn carries [from] lockedmove on its own move line -> masked
    # by the event-level `called` flag; the partner's fresh move still labels.
    result = build_supervision(second, {"winner": "p1", "ending": "played_out"}, audit)
    assert result["masks"]["move"] == [False, True]
    assert result["targets"]["move"] == [None, "dragonclaw"]


def test_encore_locks_from_start_turn_through_end_turn():
    from vgc.replay_parse import parse_replay

    log = BASE + (
        "|move|p1a: Pikachu|Thunderbolt|p2a: Foe\n"
        "|-start|p1a: Pikachu|Encore\n"
        "|move|p1b: Charizard|Flamethrower|p2a: Foe\n|turn|2\n"
        "|move|p1a: Pikachu|Thunderbolt|p2a: Foe\n"
        "|move|p1b: Charizard|Dragon Claw|p2a: Foe\n"
        "|-end|p1a: Pikachu|Encore\n|turn|3\n"
        "|move|p1a: Pikachu|Protect|p1a: Pikachu\n"
        "|move|p1b: Charizard|Dragon Claw|p2a: Foe\n|win|Alice\n")
    audit = audit_raw_log(log)
    rows = [r for r in parse_replay("encore", None, log).records
            if r["player"] == "p1" and r["decision_kind"] == "turn"]
    # Encore masks its slot from the -start turn through the -end turn,
    # inclusive; the partner's fresh moves label on all three turns.
    expected = [(["thunderbolt", "flamethrower"], [False, True]),
                (["thunderbolt", "dragonclaw"], [False, True]),
                (["protect", "dragonclaw"], [True, True])]
    for row, (moves, masks) in zip(rows, expected):
        result = build_supervision(row, {"winner": "p1", "ending": "played_out"}, audit)
        assert result["masks"]["move"] == masks
        assert result["targets"]["move"] == [
            move if mask else None for move, mask in zip(moves, masks)]


def test_audit_without_move_locks_disables_move_labels():
    audit = audit_raw_log(BASE + "|move|p1a: Pikachu|Thunderbolt|p2a: Foe\n|win|Alice\n")
    del audit["move_locks"]
    result = build_supervision(record(), {"winner": "p1", "ending": "played_out"}, audit)
    assert result["masks"]["move"] == [False, False]
