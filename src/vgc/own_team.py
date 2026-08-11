"""Fill in our OWN Stat Points and nature on a parsed battle.

poke-env learns our own spread from exactly one message: the `|showteam|` that Showdown
sends when Open Team Sheets are accepted (`Player._handle_battle_message`'s `showteam`
branch is the only caller of `apply_teambuilder_team`). `AbstractBattle` never registers
our own previewed team either -- `_register_teampreview_pokemon` appends only the
OPPONENT's -- so the teambuilder path it guards can never fire on its own.

On the public ladder that message essentially never arrives: opponents must explicitly
`/acceptopenteamsheets`, and CLAUDE.md records ~0.2% of 2,939 replays containing a
showteam. So without this module, `Pokemon.evs` is `None` for our own team and
`vgc.evaluator._our_pokemon_state` falls back to `vgc.stats.default_opponent_spread` --
the guess meant for UNKNOWN opponents, applied to the team we built ourselves.

That guess is badly wrong for an offensive team, and wrong in a consistent direction.
On `teams/meta1.packed.txt` it misses by up to 35.6% (Incineroar's Attack, 135 -> 183),
and it underestimates Speed on all six Pokemon (-13% to -27%) while overestimating HP on
all six (+12% to +19%): it assumes a bulky, slow build. Speed decides move order, so a
bot that believes it is ~20% slower than it is will systematically over-Protect and
over-switch.

We always know our own team -- it is the file we handed the Teambuilder -- so none of
this needs to depend on what the opponent chooses to reveal.

Deliberately NOT `AbstractBattle.apply_teambuilder_team`: that routes through
`Pokemon._update_from_teambuilder`, which recomputes `_stats` with poke-env's VANILLA
gen-9 EV formula. The champions mod is linear in Stat Points (`HP = base + SP + 75`, see
`vgc/stats.py`), and the simulator's own request already carried mod-correct stats, so
calling it would replace right numbers with wrong ones. Only the three fields the
request genuinely cannot tell us are set here.
"""

from __future__ import annotations

from typing import Any, Iterable

from poke_env.teambuilder.teambuilder import Teambuilder


def spread_index(entries: Iterable[Any]) -> dict[str, Any]:
    """Index `TeambuilderPokemon`s by the name their battle ident carries."""

    return {entry.nickname or entry.species or "": entry for entry in entries}


def index_from_packed(packed_team: str) -> dict[str, Any]:
    """`spread_index` for a packed team string (`teams/*.packed.txt`)."""

    return spread_index(Teambuilder.parse_packed_team(packed_team))


def apply_own_spreads(battle: Any, index: dict[str, Any] | None = None) -> int:
    """Set `evs`/`ivs`/`nature` on our own team, returning how many were filled.

    `index` defaults to the battle's own teambuilder team, which poke-env populates
    per battle in `Player._create_battle`. Pokemon that already know their spread (an
    OTS `showteam` did arrive) are left alone, so this is a no-op in that case.

    Never raises: it runs inside the live battle message path, where a malformed team
    entry must not be able to end a game.
    """

    try:
        if index is None:
            index = spread_index(getattr(battle, "teambuilder_team", None) or [])
        if not index:
            return 0
        filled = 0
        for ident, pokemon in (getattr(battle, "team", None) or {}).items():
            if pokemon.evs is not None:
                continue
            entry = index.get(str(ident).split(": ", 1)[-1])
            if entry is None:
                continue
            pokemon._evs = entry.evs
            pokemon._ivs = entry.ivs
            pokemon._nature = (entry.nature or "serious").lower()
            filled += 1
        return filled
    except Exception:  # noqa: BLE001 - must never break a live battle
        return 0
