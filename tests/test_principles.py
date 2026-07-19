from __future__ import annotations

from types import SimpleNamespace

from vgc.principles import classify_mon, detect_team_signals, utility_kind


def _mon(species: str, moves: list[str], ability: str | None = None):
    return SimpleNamespace(species=species, moves=dict.fromkeys(moves), ability=ability)


def test_detects_weather_engine_and_its_setter() -> None:
    team = [
        _mon("pelipper", ["hurricane", "tailwind"], "drizzle"),
        _mon("basculegion", ["wavecrash", "aquajet"], "swiftswim"),
    ]
    signals = detect_team_signals(team)
    assert "rain" in signals.engines
    assert "tailwind" in signals.engines
    assert 0 in signals.engine_enabler_indices


def test_classifies_pressure_denial_redirection_setup_and_pivot_roles() -> None:
    mon = _mon(
        "smeargle",
        ["fakeout", "followme", "swordsdance", "partingshot"],
        "intimidate",
    )
    roles = classify_mon(mon)
    assert roles.action_denial
    assert roles.redirection
    assert roles.setup
    assert roles.pivot


def test_every_first_principles_utility_family_has_an_explicit_kind() -> None:
    expected = {
        "protect": "protect",
        "ragepowder": "redirection",
        "tailwind": "speed_control",
        "taunt": "action_denial",
        "yawn": "action_denial",
        "swordsdance": "setup",
        "reflect": "screen",
        "willowisp": "burn",
        "wideguard": "wide_defense",
        "recover": "recovery",
        "partingshot": "pivot",
    }
    assert {move_id: utility_kind(move_id) for move_id in expected} == expected

