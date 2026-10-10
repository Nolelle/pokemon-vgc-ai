from types import SimpleNamespace

from vgc.actions import index_switch_choice


def _battle(team):
    return SimpleNamespace(last_request={"side": {"pokemon": team}})


# Our Ditto transformed into our own Golurk (audit battle 1111): Showdown matches
# `switch golurk` against the ACTIVE Ditto's current species and rejects the choice.
TEAM = [
    {"ident": "p1: Ditto", "details": "Ditto, L50", "active": True, "condition": "80/125"},
    {"ident": "p1: Farigiraf", "details": "Farigiraf, L50, M", "active": True,
     "condition": "195/195"},
    {"ident": "p1: Golurk", "details": "Golurk, L50", "active": False, "condition": "196/196"},
    {"ident": "p1: Corviknight", "details": "Corviknight, L50, M", "active": False,
     "condition": "0 fnt"},
]


def test_switch_to_benched_teammate_is_sent_by_position():
    message = "/choose move substitute, switch Golurk"
    assert index_switch_choice(_battle(TEAM), message) == "/choose move substitute, switch 3"


def test_bare_choice_without_choose_prefix():
    assert index_switch_choice(_battle(TEAM), "switch Golurk, move protect") == (
        "switch 3, move protect"
    )


def test_moves_and_numeric_switches_are_untouched():
    for message in ("/choose move earthquake, move protect", "/choose switch 3, move protect"):
        assert index_switch_choice(_battle(TEAM), message) == message


def test_no_request_leaves_message_alone():
    assert index_switch_choice(SimpleNamespace(last_request={}), "switch Golurk") == (
        "switch Golurk"
    )
