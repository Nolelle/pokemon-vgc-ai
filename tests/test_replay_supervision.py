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
    charged = BASE + "|move|p2a: Foe|Freeze Shock|p1a: Pikachu\n|move|p1a: Pikachu|Thunderbolt|p2a: Foe\n|win|Alice\n"
    assert not build(charged)["masks"]["move"][0]
