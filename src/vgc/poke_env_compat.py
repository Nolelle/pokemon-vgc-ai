"""Repairs for poke-env 0.15 parsing bugs.

Two kinds, both installed by importing this module:

* `normalize_for_poke_env` rewrites a protocol line just before `parse_message`. Callers
  that record or replay the raw stream (`BattleMemory`, decision-replay recorders) keep the
  original line; only the copy handed to poke-env is rewritten.
* Method patches on poke-env classes (`Pokemon.mega_evolve`, `Pokemon._update_from_pokedex`,
  `AbstractBattle.parse_message`, `DoubleBattle.parse_request`) for state poke-env gets wrong
  that a line rewrite cannot express.

`offline/audit_battle_parsing.py` compares the parsed state with Showdown's own and is the
evidence for every repair here. Each repair has a name in `ALL_FIXES`; list names in the
`VGC_DISABLE_COMPAT_FIXES` environment variable (comma separated) or call
`set_disabled_fixes` to switch them off, which the audit uses for its before/after table.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from typing import Any

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.double_battle import DoubleBattle
from poke_env.battle.effect import Effect
from poke_env.battle.pokemon import Pokemon

ALL_FIXES = (
    "copyboost",  # Psych Up / Costar copied the wrong way round
    "champions_mega_ability",  # 16 Megas whose Champions ability poke-env's dex has wrong
    "single_turn_effects",  # -singleturn / -singlemove effects were never removed
    "momentary_activate_effects",  # -activate markers (Struggle, Quick Claw, ...) stuck forever
    "skipped_charge",  # Power Herb / Solar Beam in sun left the foe "preparing" forever
    "move_changed_ability",  # Worry Seed / Simple Beam / Entrainment made a permanent ability
    "ability_reveal",  # [from] ability: lines revealed nothing about the foe's ability
    "confusion_resets_protect",  # a confusion self-hit left the Protect counter alone
    "unbrought_not_active",  # own preview-only Pokemon stayed `active`
    "baton_pass",  # Baton Pass dropped the passed boosts and volatiles
    "sleep_counter",  # Snore / Sleep Talk counted a sleep turn twice
    "interrupted_charge",  # a charge turn cut short by par / flinch left the user "preparing"
    "forme_species",  # Palafin-Hero, Mimikyu-Busted, Aegislash-Blade: species stayed the base
    "snapshot_layers",  # spikes / toxic spikes layer count replaced by a start turn
    "snapshot_foe_item",  # a foe's consumed / knocked-off item read as "unknown"
    "snapshot_effects",  # poke-env marker effects and aliases reached the Showdown mirror
    "screen_clock",  # Light Clay screens were assumed to last 5 turns
    "gastro_acid",  # Gastro Acid left the target's ability active
    "toxic_replacement",  # a replacement sent in after the residual phase got a toxic tick
    "flash_fire_persists",  # Flash Fire was "used up" by the holder's first Fire move
    "psych_up_crit",  # Psych Up silently re-sets the crit-stage volatiles (Focus Energy ...)
    "regenerator_double_heal",  # Regenerator's switch-out heal applied twice
    "swapped_ability_temporary",  # an ability revealed after Skill Swap outlived the switch
    "mega_changes_from",  # Floette-Mega was not recognised (it changes from Floette-Eternal)
    "memory_ability_changes",  # BattleMemory recorded a Worry Seed / Trace result as the base
    "illusion_break_state",  # a broken Illusion dropped the disguise's boosts and volatiles
    "cud_chew_keeps_item",  # Cud Chew's re-eaten berry "consumed" the held item
    "stolen_item_eaten",  # a stolen White Herb is logged eaten BEFORE it is received
    "symbiosis_item_order",  # a Symbiosis item is logged BEFORE the berry it replaces
    "blocked_priority_is_not_cant",  # Armor Tail / Dazzling `cant` lines are the holder's, not a miss
    "shed_tail",  # Shed Tail's Substitute was not handed to the replacement
    "toxic_stage_cap",  # Showdown caps the toxic stage at 15
    "status_change_resets_counter",  # a new status inherited the old status's turn counter
)
_DISABLED: set[str] = {
    name.strip()
    for name in os.environ.get("VGC_DISABLE_COMPAT_FIXES", "").split(",")
    if name.strip()
}


def set_disabled_fixes(names: Iterable[str]) -> None:
    """Switch the named repairs off (the audit's "before" arm). Empty re-enables all."""
    unknown = set(names) - set(ALL_FIXES)
    if unknown:
        raise ValueError(f"unknown compat fixes: {sorted(unknown)}")
    _DISABLED.clear()
    _DISABLED.update(names)


def fix_enabled(name: str) -> bool:
    return name not in _DISABLED


_ROUND_CHAIN_TAGS = ("[from] move: Round", "[from]move: Round")


def normalize_for_poke_env(split: list[str]) -> list[str]:
    """Return `split` (a `|`-split protocol line) with known poke-env traps removed.

    Round chain: when allies both use Round, Showdown moves the second one up and tags
    its line `|move|<mon>|Round|<target>|[from] move: Round`. poke-env treats that tag
    as a borrowed move (like Copycat), skips the reveal, then indexes `mon.moves["round"]`
    -- a `KeyError` for an opponent that has not revealed Round yet. The Pokemon did use
    its own Round, so dropping the tag gives poke-env the correct reading.

    Copy boost: see `_swap_copyboost_roles`.
    """
    if len(split) > 3 and split[1] == "-copyboost" and fix_enabled("copyboost"):
        return _swap_copyboost_roles(split)
    if (
        len(split) > 3
        and split[1] == "move"
        and split[3] == "Round"
        and any(part in _ROUND_CHAIN_TAGS for part in split[4:])
    ):
        return [part for part in split if part not in _ROUND_CHAIN_TAGS]
    return split


def _swap_copyboost_roles(split: list[str]) -> list[str]:
    """`|-copyboost|RECEIVER|SOURCE|...` -> the argument order poke-env expects.

    Showdown's Psych Up (`this.add('-copyboost', source, target, ...)` where `source` is the
    USER, who has just copied the TARGET's boosts) and Costar (`pokemon, ally`) both put the
    receiver first, and the Showdown client reads it the same way. poke-env 0.15 reads
    `-copyboost|SOURCE|TARGET` and does `target.copy_boosts(source)`, i.e. backwards: the
    user kept its own boosts and the foe it copied from lost theirs (ladder game
    2695881082: a +evasion Slowbro looked unboosted while the Psych Up user's boosts were
    wiped). Swapping the two idents makes poke-env's own handler do the right thing.
    """
    return [*split[:2], split[3], split[2], *split[4:]]


# --- Mega Evolution keeps the exact forme -------------------------------------------------
#
# Showdown sends `|detailschange|<mon>|Lucario-Mega-Z, ...` and THEN `|-mega|<mon>|Lucario|...`.
# poke-env's `detailschange` handler loads the exact Mega forme, but its `-mega` handler
# calls `Pokemon.mega_evolve`, which re-derives "<species>mega" from the base species and
# overwrites the stats/types/ability with the plain Mega whenever that id exists -- so Mega
# Lucario Z, Garchomp Z and Absol Z became their plain Megas (~9% of Mega Evolutions in the
# M-C corpus). `mega_evolve` is redundant once the forme change has been applied.

_original_mega_evolve = Pokemon.mega_evolve


def _mega_evolve_keeping_exact_forme(self: Pokemon, stone: str) -> None:
    if self.forme_change_ability is not None:
        self.temporary_ability = None  # the one side effect `mega_evolve` always has
        return
    _original_mega_evolve(self, stone)


if not getattr(Pokemon.mega_evolve, "_vgc_keeps_exact_forme", False):
    _mega_evolve_keeping_exact_forme._vgc_keeps_exact_forme = True  # type: ignore[attr-defined]
    Pokemon.mega_evolve = _mega_evolve_keeping_exact_forme  # type: ignore[method-assign]


# --- Champions Mega abilities -------------------------------------------------------------
#
# poke-env bundles a vanilla gen9 pokedex. For 16 of the 98 Megas the Champions mod changes the
# ability (Garchomp-Mega-Z has Levitate, Golisopod-Mega Tough Claws, Staraptor-Mega Contrary,
# ...); stats and types are identical. poke-env reads a Mega's ability from its own dex entry
# into `forme_change_ability`, so a Mega'd Golisopod looked like it still had Emergency Exit
# (own side too: requests only carry the BASE ability) and `vgc.mechanics_state` could not
# recognise those Megas at all (it matches the Mega forme by its Champions ability). The exported
# `data/champions/species.json` is the ground truth.


def _to_id(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


_original_update_from_pokedex = Pokemon._update_from_pokedex


def _update_from_pokedex_with_champions_abilities(
    self: Pokemon, species: str, store_species: bool = True
) -> None:
    _original_update_from_pokedex(self, species, store_species)
    if not fix_enabled("champions_mega_ability") or self.forme_change_ability is None:
        return
    from vgc.data import load_species  # local: keep this module import-light

    entry = load_species().get(_to_id(species))
    if entry and entry.get("isMega"):
        ability = (entry.get("abilities") or {}).get("0")
        if ability:
            self.forme_change_ability = ability


if not getattr(Pokemon._update_from_pokedex, "_vgc_champions_abilities", False):
    _update_from_pokedex_with_champions_abilities._vgc_champions_abilities = True  # type: ignore[attr-defined]
    Pokemon._update_from_pokedex = _update_from_pokedex_with_champions_abilities  # type: ignore[method-assign]


# --- stateful message repairs -------------------------------------------------------------


def _find_mon(battle: AbstractBattle, ident: str) -> Pokemon | None:
    """The poke-env Pokemon for a protocol ident ("p1a: Name"), without creating one."""
    key = ident if len(ident) < 4 or ident[2] == ":" else ident[:2] + ident[3:]
    for team in (battle.team, battle.opponent_team):
        mon = team.get(key)
        if mon is not None:
            return mon
    return None


def _ephemeral(battle: AbstractBattle) -> dict[str, list[tuple[Pokemon, Effect]]]:
    store = vars(battle).get("_vgc_ephemeral")
    if store is None:
        store = {"turn": [], "move": []}
        battle._vgc_ephemeral = store  # type: ignore[attr-defined]
    return store


_OF = re.compile(r"\[of\] (p[12][ab]?: [^|]+)")
_FROM_ABILITY = re.compile(r"\[from\] ability: ([^|\]]+)")


def _revealed_ability_holder(split: list[str]) -> tuple[str, str] | None:
    """(ident, ability name) when a line unambiguously shows whose ability fired, else None.

    The `[of]` tag means the effect's SOURCE, which is the ability holder for damage, status,
    weather, field and Frisk lines but the ATTACKER for an absorbing `-heal` (Water Absorb),
    so a `-heal` with `[of]` is skipped rather than guessed.
    """
    tag = split[1]
    parts = split[2:]
    if tag in ("-ability", "-transform", "-formechange", "detailschange"):
        return None  # poke-env handles these (Trace / Imposter swap abilities)
    if tag in ("-activate", "-start", "-end") and len(split) > 3:
        if split[3].lower().startswith("ability:"):
            return split[2], split[3].split(":", 1)[1].strip()
    for part in parts:
        match = _FROM_ABILITY.search(part)
        if not match:
            continue
        ability = match.group(1).strip()
        of = next((m.group(1) for p in parts if (m := _OF.search(p))), None)
        if tag in ("-weather", "-fieldstart", "-sidestart"):
            return (of, ability) if of else None
        if tag == "-heal":
            # `[of]` is the attacker whose move was absorbed (Water Absorb, Dry Skin), even
            # when that attacker is the holder's ally. Hospitality is the one ability whose
            # `[of]` is the healer: it heals an ally.
            if of is not None and _to_id(ability) == "hospitality":
                return of, ability
            return split[2], ability
        if tag == "-enditem" and _to_id(ability) in ("pickpocket", "magician"):
            return None  # its `[of]` repeats the victim; the paired `-item` line is reliable
        if tag == "-item" and _to_id(ability) != "frisk":
            # Pickpocket / Magician: the first Pokemon receives the item (holder = thief).
            # Frisk is the opposite: the first Pokemon is the frisked one.
            return split[2], ability
        if of:
            return of, ability
        if split[2][:2] in ("p1", "p2"):
            return split[2], ability
        return None
    return None


# Trap moves announce themselves with `-activate` but really do persist 4-5 turns (Showdown's
# `partiallytrapped`), so they must not be treated as momentary.
_PERSISTENT_ACTIVATE_EFFECTS = frozenset(
    getattr(Effect, name)
    for name in (
        "BIND", "CLAMP", "FIRE_SPIN", "INFESTATION", "MAGMA_STORM", "SAND_TOMB", "SNAP_TRAP",
        "THUNDER_CAGE", "WHIRLPOOL", "WRAP",
    )
    if hasattr(Effect, name)
)  # fmt: skip

# Volatiles Baton Pass hands to the incoming Pokemon (Showdown: conditions without `noCopy`
# that outlive a turn). Boosts always pass.
_BATON_PASS_EFFECTS = frozenset(
    getattr(Effect, name)
    for name in (
        "AQUA_RING", "CONFUSION", "DRAGON_CHEER", "FOCUS_ENERGY", "GASTRO_ACID", "HEAL_BLOCK",
        "INGRAIN", "LEECH_SEED", "MAGNET_RISE", "NO_RETREAT", "OCTOLOCK", "POWER_TRICK",
        "SUBSTITUTE", "TAUNT", "THROAT_CHOP",
    )
    if hasattr(Effect, name)
)  # fmt: skip


_CRIT_VOLATILES = tuple(
    getattr(Effect, name)
    for name in ("DRAGON_CHEER", "FOCUS_ENERGY", "G_MAX_CHI_STRIKE", "LASER_FOCUS")
    if hasattr(Effect, name)
)


def _swapped(battle: AbstractBattle) -> set[int]:
    store = vars(battle).get("_vgc_ability_swapped")
    if store is None:
        store = set()
        battle._vgc_ability_swapped = store  # type: ignore[attr-defined]
    return store


def _drop(mon: Pokemon, effect: Effect) -> bool:
    """Remove `effect` without `end_effect`'s side effects (Skill Swap clears the ability).
    Returns False so it can sit inside a keep-filter."""
    mon._effects.pop(effect, None)
    return False


def _chain(*hooks: Any) -> Any:
    live = [hook for hook in hooks if hook is not None]
    if not live:
        return None
    if len(live) == 1:
        return live[0]

    def run() -> None:
        for hook in live:
            hook()

    return run


def _is_sleeping(mon: Pokemon) -> bool:
    return mon._status is not None and mon._status.name == "SLP"


def _prepare_message(battle: AbstractBattle, split: list[str]) -> Any:
    """Capture what a repair needs from BEFORE poke-env handles `split`; return a closure to
    run after it, or None. One small handler per protocol tag."""
    if len(split) < 2:
        return None
    tag = split[1]
    if tag == "upkeep":
        # Everything switched in from here until `|turn|` arrives after the residual phase.
        battle._vgc_after_upkeep = True  # type: ignore[attr-defined]
        battle._vgc_late_switch_ins = []  # type: ignore[attr-defined]
        return None
    if tag == "turn":
        return _on_turn(battle)
    if len(split) < 3:
        return None
    handler = _HANDLERS.get(tag)
    return handler(battle, split) if handler is not None else None


def _on_turn(battle: AbstractBattle) -> Any:
    late_mons = list(vars(battle).get("_vgc_late_switch_ins") or ())
    battle._vgc_after_upkeep = False  # type: ignore[attr-defined]
    battle._vgc_late_switch_ins = None  # type: ignore[attr-defined]
    battle._vgc_cant_sleep = None  # type: ignore[attr-defined]

    def end_of_turn() -> None:
        if fix_enabled("toxic_replacement"):
            # `Pokemon.end_turn` ticks every active toxic Pokemon, but a replacement sent in
            # after the residual phase was never poisoned by it (Showdown: stage 0).
            for mon in late_mons:
                if mon._status is not None and mon._status.name == "TOX":
                    mon._status_counter = max(0, mon._status_counter - 1)
        if fix_enabled("toxic_stage_cap"):
            for mon in battle.all_active_pokemons:
                if mon is not None and mon._status_counter > 15 and mon._status is not None:
                    if mon._status.name == "TOX":
                        mon._status_counter = 15
        if fix_enabled("single_turn_effects"):
            store = _ephemeral(battle)
            for mon, effect in store["turn"]:
                _drop(mon, effect)
            store["turn"] = []

    return end_of_turn


def _on_move_or_cant(battle: AbstractBattle, split: list[str]) -> Any:
    mon = _find_mon(battle, split[2])
    if mon is None:
        return None
    tag = split[1]
    battle._vgc_last_enditem = None  # type: ignore[attr-defined]
    battle._vgc_symbiosis = None  # type: ignore[attr-defined]
    if fix_enabled("single_turn_effects"):
        # A -singlemove effect (Destiny Bond, Grudge) ends when its holder next acts.
        store = _ephemeral(battle)
        store["move"] = [(m, e) for m, e in store["move"] if m is not mon or _drop(m, e)]
    if tag == "cant":
        reason = split[3] if len(split) > 3 else ""
        if (
            reason.startswith("ability: ")
            and reason != "ability: Truant"
            and fix_enabled("blocked_priority_is_not_cant")
        ):
            # `|cant|HOLDER|ability: Armor Tail|Sucker Punch|[of] ATTACKER`: the holder is the
            # one that blocked the move and did not lose its own action, so its Protect chain
            # and charge are untouched (poke-env resets both on any `cant`).
            counter = mon._protect_counter
            move, target = mon._preparing_move, mon._preparing_target

            def blocked_attacker() -> None:
                mon._protect_counter = counter
                mon._preparing_move, mon._preparing_target = move, target

            return blocked_attacker
        hooks = []
        if fix_enabled("interrupted_charge"):
            # Showdown drops the `twoturnmove` volatile when the second turn is aborted (full
            # paralysis, flinch, sleep, ...); poke-env kept the user "preparing".
            mon._preparing_move = None
            mon._preparing_target = None
        if reason == "slp":
            battle._vgc_cant_sleep = mon  # type: ignore[attr-defined]
        elif fix_enabled("sleep_counter") and _is_sleeping(mon):

            def not_a_sleep_turn() -> None:
                # Only `cant|X|slp` is the sleep timer ticking; poke-env counted a recharge
                # turn (which Showdown resolves before the sleep check) as slept too.
                mon._status_counter = max(0, mon._status_counter - 1)

            hooks.append(not_a_sleep_turn)
        return _chain(*hooks)
    # tag == "move"
    cant_for = vars(battle).get("_vgc_cant_sleep")
    if cant_for is not mon:
        battle._vgc_cant_sleep = None  # type: ignore[attr-defined]
    hooks = [_after_move(battle, mon, split)]
    if cant_for is mon and fix_enabled("sleep_counter") and _is_sleeping(mon):

        def undo_double_count() -> None:
            # `|cant|X|slp` already counted this sleep turn; Snore / Sleep Talk (and the move
            # Sleep Talk calls) then arrive as `|move|` lines that poke-env counts again.
            mon._status_counter = max(0, mon._status_counter - 1)

        hooks.append(undo_double_count)
    return _chain(*hooks)


def _after_move(battle: AbstractBattle, mon: Pokemon, split: list[str]) -> Any:
    """A move line whose `[from]` tag shows the Pokemon did not choose the move this turn
    (a reflected move, a Dancer copy) must not reset its Protect chain."""
    if not fix_enabled("confusion_resets_protect"):
        return None
    reflected = any(
        part
        in ("[from] ability: Magic Bounce", "[from] ability: Dancer", "[from] move: Magic Coat")
        for part in split[4:]
    )
    if not reflected:
        return None
    counter = mon._protect_counter

    def restore() -> None:
        mon._protect_counter = counter

    return restore


def _on_single_effect(battle: AbstractBattle, split: list[str]) -> Any:
    if len(split) < 4 or not fix_enabled("single_turn_effects"):
        return None
    tag = split[1]
    effect_name = split[3].replace("move: ", "")

    def register() -> None:
        mon = _find_mon(battle, split[2])
        if mon is None:
            return
        effect = Effect.from_showdown_message(effect_name)
        _ephemeral(battle)["turn" if tag == "-singleturn" else "move"].append((mon, effect))

    return register


def _on_activate(battle: AbstractBattle, split: list[str]) -> Any:
    if len(split) < 4:
        return None
    mon = _find_mon(battle, split[2])
    if mon is None:
        return None
    before = set(mon.effects)
    if split[3] == "ability: Cud Chew":
        battle._vgc_cud_chew = mon  # type: ignore[attr-defined]
    symbiosis_to = None
    if split[3] == "ability: Symbiosis" and len(split) > 5 and fix_enabled("symbiosis_item_order"):
        symbiosis_to = _find_mon(battle, split[5].replace("[of] ", ""))
    swap_target = next((m.group(1) for p in split[4:] if (m := _OF.search(p))), None)
    skill_swap = split[3].replace("move: ", "") == "Skill Swap"
    # Ally Skill Swap hides both abilities (`|-activate|A|move: Skill Swap|||[of] B`).
    hidden_swap = skill_swap and not any(part for part in split[4:] if not part.startswith("[of]"))
    known_before = (_effective_ability(mon), None)
    other = _find_mon(battle, swap_target) if swap_target else None
    if other is not None:
        known_before = (known_before[0], _effective_ability(other))

    def after() -> None:
        if symbiosis_to is not None and symbiosis_to._item:
            # Showdown logs the ally's item arriving BEFORE the berry it replaces is eaten.
            battle._vgc_symbiosis = (symbiosis_to, symbiosis_to._item)  # type: ignore[attr-defined]
        if skill_swap and fix_enabled("swapped_ability_temporary"):
            # Both Pokemon hold a swapped ability until they switch out, even when the
            # abilities were hidden: anything revealed later is temporary.
            _swapped(battle).add(id(mon))
            if other is not None:
                _swapped(battle).add(id(other))
            if hidden_swap and other is not None:
                _swap_hidden_abilities(battle, mon, other, known_before)
        if fix_enabled("momentary_activate_effects"):
            for effect in set(mon.effects) - before - _PERSISTENT_ACTIVATE_EFFECTS:
                _ephemeral(battle)["turn"].append((mon, effect))

    return after


_LAST_TEMPORARY: dict[int, tuple[Pokemon, str]] = {}
_original_faint = Pokemon.faint


def _faint_remembering_ability(self: Pokemon) -> None:
    if self._temporary_ability:
        _LAST_TEMPORARY[id(self)] = (self, self._temporary_ability)
    _original_faint(self)


if not getattr(Pokemon.faint, "_vgc_remembers", False):
    _faint_remembering_ability._vgc_remembers = True  # type: ignore[attr-defined]
    Pokemon.faint = _faint_remembering_ability  # type: ignore[method-assign]


def _effective_ability(mon: Pokemon) -> str | None:
    """`mon.ability`, or the swapped-in one it held when it fainted: Showdown logs a KO'd
    Pokemon's Skill Swap line after the KO damage, and `faint()` has already wiped it."""
    if mon.fainted and fix_enabled("swapped_ability_temporary"):
        remembered = _LAST_TEMPORARY.get(id(mon))
        if remembered is not None and remembered[0] is mon:
            return remembered[1]
    return mon.ability


def _swap_hidden_abilities(
    battle: AbstractBattle, source: Pokemon, target: Pokemon, before: tuple[Any, Any]
) -> None:
    """Ally Skill Swap: neither ability is announced, but each Pokemon now has the OTHER's
    old one. poke-env swaps only when both were already known; when one was not, that side's
    ability is now unknown (`''`, which every consumer reads as "no ability known") instead
    of staying at the stale old value."""
    old_source, old_target = before
    if old_source is not None and old_target is not None:
        # Both known (always true for our own pair): set the exchange explicitly. poke-env's
        # own swap left the first result in place when the pair swapped back.
        source._temporary_ability = old_target
        target._temporary_ability = old_source
        return
    foe_role = "p2" if battle.player_role == "p1" else "p1"
    if _mon_role(battle, source) != foe_role or _mon_role(battle, target) != foe_role:
        return  # our own abilities come from the request
    source._temporary_ability = old_target if old_target is not None else ""
    target._temporary_ability = old_source if old_source is not None else ""


def _mon_role(battle: AbstractBattle, mon: Pokemon) -> str | None:
    for role, team in (
        (battle.player_role, battle.team),
        ("p2" if battle.player_role == "p1" else "p1", battle.opponent_team),
    ):
        if any(candidate is mon for candidate in team.values()):
            return role
    return None


def _on_copyboost(battle: AbstractBattle, split: list[str]) -> Any:
    if len(split) < 4 or "[from] move: Psych Up" not in split or not fix_enabled("psych_up_crit"):
        return None
    # `split` is the ORIGINAL line here (normalisation happens before parse_message and has
    # already swapped it): split[2] is the SOURCE of the boosts, split[3] the user.
    user = _find_mon(battle, split[3])
    source = _find_mon(battle, split[2])

    def copy_crit_volatiles() -> None:
        if user is None or source is None:
            return
        for effect in _CRIT_VOLATILES:
            user._effects.pop(effect, None)
            if effect in source._effects:
                user._effects[effect] = source._effects[effect]

    return copy_crit_volatiles


def _on_heal(battle: AbstractBattle, split: list[str]) -> Any:
    if len(split) > 4 and "[from] ability: Regenerator" in split[4:]:
        mon = _find_mon(battle, split[2])
        if mon is not None:
            vars(battle).setdefault("_vgc_regen_healed", set()).add(id(mon))
    return None


def _on_damage(battle: AbstractBattle, split: list[str]) -> Any:
    if len(split) < 5 or "[from] confusion" not in split[4:]:
        return None
    mon = _find_mon(battle, split[2])
    if mon is None:
        return None

    def confusion_hit() -> None:
        # The self-hit is the Pokemon's action for the turn. Showdown's `stall` (Protect chain)
        # and a charging move do not survive it, but poke-env only reacts to `|move|` and
        # `|cant|`, which a confusion hit sends neither of.
        if fix_enabled("confusion_resets_protect"):
            mon._protect_counter = 0
        if fix_enabled("interrupted_charge"):
            mon._preparing_move = None
            mon._preparing_target = None
        if fix_enabled("single_turn_effects"):
            store = _ephemeral(battle)
            store["move"] = [(m, e) for m, e in store["move"] if m is not mon or _drop(m, e)]

    return confusion_hit


def _on_status(battle: AbstractBattle, split: list[str]) -> Any:
    if len(split) < 4 or not fix_enabled("status_change_resets_counter"):
        return None
    mon = _find_mon(battle, split[2])
    if mon is None:
        return None
    before_status = mon._status

    def new_status() -> None:
        # Toxic's ticks leaked into Rest's sleep counter (and vice versa).
        if mon._status != before_status:
            mon._status_counter = 0

    return new_status


def _on_start(battle: AbstractBattle, split: list[str]) -> Any:
    if len(split) > 3 and split[3] == "Smack Down" and fix_enabled("interrupted_charge"):
        mon = _find_mon(battle, split[2])

        def grounded() -> None:
            # Smack Down / Thousand Arrows pull a Fly / Bounce user out of the sky and abort
            # the charge.
            if mon is not None:
                mon._preparing_move = None
                mon._preparing_target = None

        return grounded
    return None


def _on_endability(battle: AbstractBattle, split: list[str]) -> Any:
    if not fix_enabled("gastro_acid"):
        return None
    mon = _find_mon(battle, split[2])

    def suppressed() -> None:
        if mon is not None:
            mon._effects.setdefault(Effect.GASTRO_ACID, 0)

    return suppressed


def _on_ability(battle: AbstractBattle, split: list[str]) -> Any:
    """`-ability|mon|NEW|OLD|[from] move: ...` (Worry Seed, Simple Beam, Entrainment, Role
    Play) and `[from] ability: ...` (Trace, Mummy) are CHANGES: temporary, gone on switch-out."""
    if len(split) < 5 or not fix_enabled("move_changed_ability"):
        return None
    changed_by_move = any("[from] move:" in part for part in split[4:])
    if not changed_by_move and not any("[from] ability:" in part for part in split[4:]):
        return None
    mon = _find_mon(battle, split[2])
    if mon is None:
        return None
    base_before = mon._ability

    def to_temporary() -> None:
        _swapped(battle).add(id(mon))
        if changed_by_move and base_before is None and mon._ability is not None:
            # poke-env stored the NEW ability as the base; the line also names the OLD, real one.
            mon._temporary_ability = mon._ability
            old = split[4] if not split[4].startswith("[") else ""
            mon._ability = _to_id(old) or None

    return to_temporary


def _on_anim(battle: AbstractBattle, split: list[str]) -> Any:
    if len(split) < 4 or not fix_enabled("skipped_charge"):
        return None
    mon = _find_mon(battle, split[2])

    def after_anim() -> None:
        move = mon._preparing_move if mon is not None else None
        if mon is not None and move is not None and _to_id(split[3]) == move.id:
            # `-prepare` then `-anim` with no second `|move|`: Power Herb, Solar Beam in sun,
            # Electro Shot in rain etc. fired on the spot.
            mon._preparing_move = None
            mon._preparing_target = None

    return after_anim


def _on_enditem(battle: AbstractBattle, split: list[str]) -> Any:
    if len(split) < 4:
        return None
    mon = _find_mon(battle, split[2])
    if mon is None:
        return None
    if not any(part.startswith("[from]") for part in split[4:]):
        battle._vgc_last_enditem = (mon, _to_id(split[3]))  # type: ignore[attr-defined]
    given = vars(battle).get("_vgc_symbiosis")
    if (
        given is not None
        and given[0] is mon
        and not any(part.startswith("[from]") for part in split[4:])
    ):
        battle._vgc_symbiosis = None  # type: ignore[attr-defined]

        def keep_given() -> None:
            mon._item = given[1]

        return keep_given
    if fix_enabled("cud_chew_keeps_item") and vars(battle).get("_vgc_cud_chew") is mon:
        held = mon._item
        battle._vgc_cud_chew = None  # type: ignore[attr-defined]

        def keep() -> None:
            # Cud Chew re-eats the berry it ate last turn; the Pokemon's current item is
            # whatever it holds now (Shell Bell), which that line says nothing about.
            mon._item = held

        return keep
    return None


def _on_item(battle: AbstractBattle, split: list[str]) -> Any:
    if len(split) < 5 or not fix_enabled("stolen_item_eaten"):
        return None
    mon = _find_mon(battle, split[2])
    last = vars(battle).get("_vgc_last_enditem")
    if (
        mon is None
        or last is None
        or last[0] is not mon
        or last[1] != _to_id(split[3])
        or not any(part.startswith("[from] move:") for part in split[4:])
    ):
        return None
    battle._vgc_last_enditem = None  # type: ignore[attr-defined]

    def eaten_on_arrival() -> None:
        # Thief / Covet logs the stolen berry or White Herb as used up, THEN as received.
        mon._item = None

    return eaten_on_arrival


_HANDLERS = {
    "-enditem": _on_enditem,
    "-item": _on_item,
    "move": _on_move_or_cant,
    "cant": _on_move_or_cant,
    "switch": lambda battle, split: _prepare_switch(battle, split),
    "drag": lambda battle, split: _prepare_switch(battle, split),
    "detailschange": lambda battle, split: _prepare_forme_change(battle, split),
    "-formechange": lambda battle, split: _prepare_forme_change(battle, split),
    "-singleturn": _on_single_effect,
    "-singlemove": _on_single_effect,
    "-activate": _on_activate,
    "-copyboost": _on_copyboost,
    "-heal": _on_heal,
    "-damage": _on_damage,
    "-status": _on_status,
    "-start": _on_start,
    "-endability": _on_endability,
    "-ability": _on_ability,
    "-anim": _on_anim,
}


def _prepare_switch(battle: AbstractBattle, split: list[str]) -> Any:
    """Baton Pass: poke-env cleared the outgoing Pokemon's boosts and volatiles on switch-out
    and the incoming one started clean; Showdown hands them over."""
    battle._vgc_cant_sleep = None  # type: ignore[attr-defined]
    late = vars(battle).get("_vgc_late_switch_ins")
    if late is not None and vars(battle).get("_vgc_after_upkeep"):
        late_key = split[2]

        def note_late() -> None:
            mon = _find_mon(battle, late_key)
            if mon is not None:
                late.append(mon)

        late_hook: Any = note_late
    else:
        late_hook = None
    tail = split[5:] if len(split) > 5 else []
    out_mon = None
    slot = split[2][:3]
    active = (
        battle._active_pokemon
        if slot[:2] == battle._player_role
        else (battle._opponent_active_pokemon)
    )
    _restore_temporary_forme(battle, active.get(slot))
    leaving = active.get(slot)
    healed = vars(battle).get("_vgc_regen_healed")
    hp_after_protocol_heal = None
    if leaving is not None:
        _swapped(battle).discard(id(leaving))
        if healed and id(leaving) in healed and fix_enabled("regenerator_double_heal"):
            # The protocol already carried Regenerator's heal (`-heal ... [silent]`);
            # `Pokemon.switch_out` is about to add its own 1/3 on top.
            hp_after_protocol_heal = leaving._current_hp
            healed.discard(id(leaving))
    if fix_enabled("baton_pass") and any(
        part.replace("move: ", "").endswith("Baton Pass") for part in tail
    ):
        out_mon = active.get(slot)
    if fix_enabled("shed_tail") and any(
        part.replace("move: ", "").endswith("Shed Tail") for part in tail
    ):
        shed_from = active.get(slot)

        def hand_over_substitute() -> None:
            incoming = _find_mon(battle, split[2])
            if incoming is not None and incoming is not shed_from:
                incoming._effects.setdefault(Effect.SUBSTITUTE, 0)

        inner_shed = late_hook

        def shed_and_note() -> None:
            hand_over_substitute()
            if inner_shed is not None:
                inner_shed()

        late_hook = shed_and_note
    if hp_after_protocol_heal is not None and leaving is not None:
        outgoing_hp = hp_after_protocol_heal
        inner_hook = late_hook

        def keep_protocol_hp() -> None:
            leaving._current_hp = outgoing_hp
            if inner_hook is not None:
                inner_hook()

        late_hook = keep_protocol_hp
    if out_mon is None:
        return late_hook
    boosts = dict(out_mon._boosts)
    passed = {e: v for e, v in out_mon._effects.items() if e in _BATON_PASS_EFFECTS}

    def hand_over() -> None:
        incoming = _find_mon(battle, split[2])
        if incoming is None or incoming is out_mon:
            return
        incoming._boosts.update(boosts)
        incoming._effects.update(passed)
        if late_hook is not None:
            late_hook()

    return hand_over


# `detailschange` is a permanent forme change (Palafin-Hero, Mimikyu-Busted, Megas);
# `-formechange` is temporary and reverts when the Pokemon leaves the field (Aegislash-Blade).
# poke-env loads the new forme's stats and types but keeps `species` at the base species, so
# everything keyed by species (our damage calculator, set priors) read Palafin-Hero as
# Palafin-Zero. Megas are left alone: `mechanics_state` already recognises them and several
# poke-env code paths expect the base species there.


def _forme_species_id(details: str) -> str:
    return _to_id(details.split(",")[0])


def _is_mega_forme(species_id: str) -> bool:
    from vgc.data import load_species

    entry = load_species().get(species_id)
    return bool(entry and entry.get("isMega")) or "mega" in species_id


def _prepare_forme_change(battle: AbstractBattle, split: list[str]) -> Any:
    if not fix_enabled("forme_species"):
        return None
    mon = _find_mon(battle, split[2])
    if mon is None:
        return None
    new_species = _forme_species_id(split[3])
    if _is_mega_forme(new_species) or new_species == mon._species:
        return None
    temporary = split[1] == "-formechange"
    original = mon._species

    def after() -> None:
        from poke_env.data import GenData

        if new_species not in GenData.from_gen(mon._gen).pokedex:
            return
        mon._species = new_species
        if temporary:
            temps = vars(battle).setdefault("_vgc_temporary_formes", {})
            temps.setdefault(id(mon), (mon, original))

    return after


def _restore_temporary_forme(battle: AbstractBattle, mon: Pokemon | None) -> Any:
    temps = vars(battle).get("_vgc_temporary_formes")
    if mon is not None and temps:
        entry = temps.pop(id(mon), None)
        if entry is not None:
            mon._species = entry[1]
    return None


def _ability_before(battle: AbstractBattle, split: list[str]) -> Any:
    """What a reveal line concerns and the holder's ability state BEFORE poke-env reads it
    (poke-env itself already stores the ability for a few lines, e.g. Frisk)."""
    found = _revealed_ability_holder(split)
    if found is None:
        return None
    ident, ability = found
    if ident[:2] != ("p2" if battle.player_role == "p1" else "p1"):
        return None  # our own abilities come from the request
    mon = _find_mon(battle, ident)
    ability_id = _to_id(ability)
    if mon is None or not ability_id:
        return None
    return mon, ability_id, mon._ability, mon._temporary_ability


def _reveal_ability(battle: AbstractBattle, before: Any) -> None:
    mon, ability_id, base_before, temporary_before = before
    swapped = id(mon) in _swapped(battle) and fix_enabled("swapped_ability_temporary")
    if swapped and base_before is None:
        # Skill Swap'd (possibly with the abilities hidden): this is the SWAPPED-in ability,
        # gone at switch-out. poke-env may already have stored it as the base.
        mon._ability = None
        mon._temporary_ability = ability_id
    elif mon.ability == ability_id:
        return
    elif swapped:
        mon._temporary_ability = ability_id
    elif mon._ability is None and mon._temporary_ability is None:
        mon._ability = ability_id
    elif mon.forme_change_ability is None:
        # The holder's ability is NOW this one although another was known: it was changed.
        mon._temporary_ability = ability_id


_original_parse_message = AbstractBattle.parse_message


def _parse_message_with_repairs(self: AbstractBattle, split_message: list[str]) -> None:
    after = _prepare_message(self, split_message)
    reveal = (
        _ability_before(self, split_message)
        if len(split_message) > 3 and fix_enabled("ability_reveal")
        else None
    )
    _original_parse_message(self, split_message)
    if after is not None:
        after()
    if reveal is not None:
        _reveal_ability(self, reveal)


if not getattr(AbstractBattle.parse_message, "_vgc_repairs", False):
    _parse_message_with_repairs._vgc_repairs = True  # type: ignore[attr-defined]
    AbstractBattle.parse_message = _parse_message_with_repairs  # type: ignore[method-assign]


# --- requests -----------------------------------------------------------------------------
#
# The team-preview request marks the first two of the six Pokemon `active`. After the `/team`
# choice the request lists only the four brought, and poke-env updates only the ones it lists,
# so the two left home stayed `active` (and looked like they were on the field) all game.

_original_parse_request = DoubleBattle.parse_request


def _parse_request_dropping_unbrought(
    self: DoubleBattle, request: dict[str, Any], strict_battle_tracking: bool = False
) -> None:
    _original_parse_request(self, request, strict_battle_tracking)
    if not fix_enabled("unbrought_not_active") or request.get("teamPreview"):
        return
    listed = {mon["ident"] for mon in request.get("side", {}).get("pokemon", ())}
    if not listed:
        return
    for ident, mon in self.team.items():
        if ident not in listed and mon._active:
            mon._active = False


if not getattr(DoubleBattle.parse_request, "_vgc_repairs", False):
    _parse_request_dropping_unbrought._vgc_repairs = True  # type: ignore[attr-defined]
    DoubleBattle.parse_request = _parse_request_dropping_unbrought  # type: ignore[method-assign]


# --- Flash Fire stays on -------------------------------------------------------------------
#
# `Pokemon.moved` treats Flash Fire like Charge and ends it after the holder's first damaging
# Fire move. In Showdown the `flashfire` volatile is a lasting 1.5x Fire boost that only ends
# when the Pokemon leaves the field.

_original_moved = Pokemon.moved


def _moved_keeping_flash_fire(self: Pokemon, *args: Any, **kwargs: Any) -> None:
    had = Effect.FLASH_FIRE in self._effects
    count = self._effects.get(Effect.FLASH_FIRE, 0)
    _original_moved(self, *args, **kwargs)
    if had and fix_enabled("flash_fire_persists") and Effect.FLASH_FIRE not in self._effects:
        self._effects[Effect.FLASH_FIRE] = count


if not getattr(Pokemon.moved, "_vgc_keeps_flash_fire", False):
    _moved_keeping_flash_fire._vgc_keeps_flash_fire = True  # type: ignore[attr-defined]
    Pokemon.moved = _moved_keeping_flash_fire  # type: ignore[method-assign]


# --- Illusion ending ---------------------------------------------------------------------
#
# While Zoroark's Illusion holds, every boost, status and volatile it earns is announced under
# the disguise's name, so poke-env files them on the disguise's Pokemon object. When the
# Illusion breaks poke-env copies the HP and status to the real Pokemon and then clears the
# disguise (`was_illusioned` -> `switch_out`), dropping the boosts and volatiles on the floor.

_original_end_illusion_on = AbstractBattle._end_illusion_on


def _end_illusion_on_keeping_state(
    self: AbstractBattle, illusionist: str | None, illusioned: Pokemon | None, details: str
) -> Pokemon:
    carried = None
    if illusioned is not None and fix_enabled("illusion_break_state"):
        carried = (
            dict(illusioned._boosts),
            dict(illusioned._effects),
            illusioned._protect_counter,
            illusioned._status_counter,
        )
    real = _original_end_illusion_on(self, illusionist, illusioned, details)
    if carried is not None and real is not illusioned:
        real._boosts = carried[0]
        real._effects.update(carried[1])
        real._protect_counter = carried[2]
        real._status_counter = carried[3]
    return real


if not getattr(AbstractBattle._end_illusion_on, "_vgc_repairs", False):
    _end_illusion_on_keeping_state._vgc_repairs = True  # type: ignore[attr-defined]
    AbstractBattle._end_illusion_on = _end_illusion_on_keeping_state  # type: ignore[method-assign]


# The Illusion can also break through a REQUEST: our own team-preview-style `active` flags in
# the next request name the real Pokemon, and poke-env then clears the disguise the same way.
_original_update_team_from_request = AbstractBattle._update_team_from_request


def _update_team_keeping_illusion_state(
    self: AbstractBattle, side: dict[str, Any], strict_battle_tracking: bool = False
) -> None:
    pairs: list[tuple[Pokemon, tuple[Any, ...]]] = []
    if fix_enabled("illusion_break_state"):
        falsely: list[Pokemon] = []
        truly: list[Pokemon] = []
        for entry in side["pokemon"]:
            mon = self._team.get(entry["ident"])
            if mon is None:
                continue
            if entry["active"] and not mon.active:
                truly.append(mon)
            elif not entry["active"] and mon.active:
                falsely.append(mon)
        # (A team-preview request also leaves bench Pokemon looking active, so only the
        # ones carrying state can be the disguise.)
        carrying = [m for m in falsely if any(m._boosts.values()) or m._effects]
        if carrying and len(carrying) == len(truly):
            pairs = [
                (real, (dict(fake._boosts), dict(fake._effects)))
                for real, fake in zip(truly, carrying, strict=True)
            ]
    _original_update_team_from_request(self, side, strict_battle_tracking)
    for real, (boosts, effects) in pairs:
        real._boosts = boosts
        real._effects.update(effects)


if not getattr(AbstractBattle._update_team_from_request, "_vgc_repairs", False):
    _update_team_keeping_illusion_state._vgc_repairs = True  # type: ignore[attr-defined]
    AbstractBattle._update_team_from_request = _update_team_keeping_illusion_state  # type: ignore[method-assign]
