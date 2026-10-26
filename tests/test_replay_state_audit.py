"""The data audit must reject real label changes while accepting repaired identities."""

from tools.audit_replay_state import normalized_action, opening, raw_identities


def test_identity_allowance_uses_that_players_preview_and_recovers_fourth_pick():
    log = "\n".join([
        "|poke|p1|Meowstic-F, L50|", "|poke|p1|Incineroar, L50|",
        "|poke|p1|Sinistcha, L50|", "|poke|p1|Indeedee, L50|",
        "|poke|p2|Meowstic, L50|",
        "|switch|p1a: Cat|Meowstic-F, L50|100/100",
        "|switch|p1b: Tiger|Incineroar, L50|100/100", "|turn|1|",
        "|switch|p1a: Tea|Sinistcha, L50|100/100", "|turn|2",
        "|switch|p1a: Cat|Meowstic-F-Mega, L50|100/100", "|turn|3",
        "|switch|p1a: Butler|Indeedee, L50|100/100",
    ])
    species = {"meowsticfmega": {"isMega": True, "baseSpecies": "Meowstic",
                                  "battleOnly": "Meowstic-F"}}
    aliases, actions = raw_identities(log, species)
    assert aliases == {"p1": {"meowstic": "meowsticf"}, "p2": {}}
    assert actions["p1"] == {"derived": True,
                             "picked": ["meowsticf", "incineroar", "sinistcha", "indeedee"],
                             "lead_order": ["meowsticf", "incineroar"]}
    record = {"decision_kind": "forced_switch", "action": {"switch_species": "meowstic"}}
    assert normalized_action(record, aliases["p2"])["switch_species"] == "meowstic"
    assert normalized_action(record, aliases["p1"])["switch_species"] == "meowsticf"


def test_opening_checker_stops_before_turn_one_events():
    field, stages = opening("\n".join([
        "|switch|p1a: Cat|Incineroar, L50|100/100", "|-weather|RainDance",
        "|-unboost|p1a: Cat|atk|1", "|-boost|p1a: Cat|atk|1",
        "|-fieldstart|move: Electric Terrain", "|-fieldend|move: Electric Terrain",
        "|turn|1", "|-weather|SunnyDay", "|-unboost|p1a: Cat|atk|2",
    ]))
    assert field == {"weather": "rain", "terrain": None, "trick_room": False, "turn": 1}
    assert not any(stages["p1a"].values())
