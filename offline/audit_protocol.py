"""What a client could LEARN from the raw protocol, derived with code that shares nothing
with `vgc.poke_env_compat`, `vgc.battle_memory` or poke-env.

The audit used to reuse `poke_env_compat._revealed_ability_holder` to decide which foe
abilities "had been revealed", so a wrong holder rule in the repaired reader also wrote the
audit's answer key and the two agreed on the mistake. This module re-derives reveals
(abilities, items, moves) and a few history facts straight from the `|`-split protocol lines,
following PROTOCOL.md and the way `sim/` emits each message.

Two things are deliberately separate:

* `ProtocolKnowledge.feed(lines)` records CANDIDATE reveals: "ability A was shown on one of
  {POKE, OF}". Some lines are ambiguous without knowing which side of the `[of]` pair holds
  the effect (a Water Absorb `-heal` carries `[of] attacker`; Rocky Helmet's `-damage` carries
  `[of] holder`).
* `ProtocolKnowledge.resolve(...)` attributes each candidate to the candidate Pokemon whose
  Showdown-dumped truth actually has that ability/item/move. The dump is the arbiter, so a
  rule gap here cannot silently agree with a wrong reader: a candidate that no Pokemon's
  truth supports is returned as UNATTRIBUTED and the audit reports it.

No import from `vgc` is needed here (ids are normalised locally).
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

_ID = re.compile(r"[^a-z0-9]")


def to_id(value: Any) -> str:
    return _ID.sub("", str(value or "").lower())


_PROTECT_FAMILY = frozenset(
    "protect detect kingsshield spikyshield banefulbunker obstruct silktrap burningbulwark "
    "quickguard wideguard matblock craftyguard endure".split()
)
_IDENT = re.compile(r"^(p[1-4])([a-z]?): (.*)$")
_FROM = re.compile(r"^\[from\] ?(ability|item|move): ?(.*)$")
_OF = re.compile(r"^\[of\] (.*)$")


def parse_ident(text: str) -> tuple[str, str, str] | None:
    """('p2', 'a', 'Name') for "p2a: Name" (and 'p2: Name'), else None."""
    match = _IDENT.match(text.strip())
    return (match.group(1), match.group(2), match.group(3)) if match else None


def _kwargs(parts: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in parts:
        if part.startswith("[") and "]" in part:
            key, _, value = part[1:].partition("]")
            out[key.strip()] = value.strip()
    return out


@dataclass
class Candidate:
    kind: str  # "ability" | "item" | "item_gone" | "move"
    value: str  # id
    holders: tuple[tuple[str, str], ...]  # (role, name), in preference order
    tag: str  # the protocol line shape that produced it
    step: int
    resolved: tuple[str, str] | None = None
    # An explicit statement ("X now holds / has ...") needs no arbitration; an EFFECT line
    # (`[from] item: X`, `ability: X`) can be about a thing the Pokemon merely used (a stolen
    # berry eaten with Bug Bite), so it must match the dump.
    explicit: bool = False
    pos: int = 0  # index of the protocol line within its step
    transformed: bool = False  # the mover was transformed when the line arrived


@dataclass
class MonKnowledge:
    abilities: list[tuple[str, str]] = field(default_factory=list)  # (ability id, tag), in order
    item_shown: str | None = None
    item_gone: bool = False
    moves: set[str] = field(default_factory=set)
    events: set[str] = field(default_factory=set)
    forme_ids: list[str] = field(default_factory=list)


class ProtocolKnowledge:
    """One perspective's accumulated public knowledge."""

    def __init__(self, role: str) -> None:
        self.role = role
        self.mons: dict[tuple[str, str], MonKnowledge] = defaultdict(MonKnowledge)
        self.pending: list[Candidate] = []
        self.step = 0
        self.ever_ability: dict[tuple[str, str], set[str]] = defaultdict(set)
        self.ever_item: dict[tuple[str, str], set[str]] = defaultdict(set)
        self.tag_counts: dict[str, int] = defaultdict(int)
        self.step_flags: set[str] = set()  # per-step facts (e.g. "illusion_break:p2")
        self.episode_log: dict[tuple[str, str], list[tuple[str, str, Any]]] = {}
        self.break_pos: dict[str, int] = {}  # role -> first Illusion-break line index this step
        self.pos = 0
        self.cud_chewing: set[tuple[str, str]] = set()
        self.last_step_idents: set[tuple[str, str]] = set()

    # -- ingest ---------------------------------------------------------------------------

    def _mon(self, ident: str) -> tuple[str, str] | None:
        parsed = parse_ident(ident)
        return (parsed[0], parsed[2]) if parsed else None

    def feed(self, lines: list[str]) -> None:
        self.step += 1
        self.step_flags = set()
        self.break_pos = {}
        self.last_step_idents = set()
        for index, raw in enumerate(lines):
            split = raw.split("|")
            if len(split) < 3:
                continue
            self.pos = index
            self._line(split)

    def _holders(self, subject: str | None, of: str | None) -> tuple[tuple[str, str], ...]:
        out = []
        for ident in (subject, of):
            mon = self._mon(ident) if ident else None
            if mon is not None and mon not in out:
                out.append(mon)
        return tuple(out)

    def _add(
        self,
        kind: str,
        value: str,
        holders: tuple[tuple[str, str], ...],
        tag: str,
        explicit: bool = False,
    ) -> None:
        if value and holders:
            first = self.mons.get(holders[0])
            self.pending.append(
                Candidate(
                    kind, to_id(value), holders, tag, self.step, explicit=explicit, pos=self.pos,
                    transformed=bool(first and "transformed" in first.events),
                )
            )

    def _line(self, split: list[str]) -> None:
        tag = split[1]
        kw = _kwargs(split[3:])
        subject = split[2] if split[2][:1] == "p" else None
        of = kw.get("of")
        frm = _FROM.match(f"[from] {kw['from']}") if "from" in kw else None
        from_kind, from_name = (frm.group(1), frm.group(2)) if frm else (None, None)
        for part in split[2:]:
            mon = self._mon(part) if part[:1] == "p" and ": " in part else None
            if mon is not None:
                self.last_step_idents.add(mon)

        # --- abilities ---------------------------------------------------------------
        if tag == "-ability" and len(split) > 3:
            new_ability = split[3]
            self._add("ability", new_ability, self._holders(subject, None), "-ability", True)
            if from_kind == "ability" and of:
                # Trace / Receiver / Power of Alchemy: the OTHER Pokemon has that ability too
                self._add("ability", new_ability, self._holders(of, None), "-ability[of]")
            if from_kind == "move" and of and to_id(from_name) in ("roleplay", "doodle"):
                self._add("ability", new_ability, self._holders(of, None), "-ability[roleplay-of]")
            if from_kind == "ability" and subject:
                self._add("ability", from_name, self._holders(subject, None), "-ability[from]")
            if from_kind == "ability" and to_id(from_name) in ("receiver", "powerofalchemy"):
                self.events_for(subject).add("receiver_announced")
        elif tag in ("-activate", "-start", "-end", "-block", "-immune", "-fail", "-singleturn",
                     "-singlemove", "-sidestart", "-fieldstart", "-hint", "-prepare"):
            # `ability: X` as the effect name: the effect belongs to the subject (or the [of]
            # holder for -block, where the subject is the protected Pokemon)
            effect = split[3] if len(split) > 3 else ""
            if effect.startswith("item:") and "[consumed]" in split:
                pass  # a (stolen / flung) berry eaten: not the eater's own item
            elif effect.startswith("ability:"):
                ability_name = effect.split(":", 1)[1]
                if tag == "-block":
                    # `-block|PROTECTED|ability: X|[of] HOLDER`: the holder is the [of] mon
                    self._add("ability", ability_name, self._holders(of, subject), f"{tag}:ability")
                else:
                    self._add(
                        "ability", ability_name, self._holders(subject, of), f"{tag}:ability"
                    )
                if to_id(ability_name) == "symbiosis" and tag == "-activate" and len(split) > 5:
                    # |-activate|GIVER|ability: Symbiosis|ITEM|[of] RECEIVER
                    self._add("item", split[4], self._holders(of, None), "symbiosis", True)
                elif tag == "-activate" and len(split) > 4 and parse_ident(split[4]):
                    # Mummy / Lingering Aroma: |-activate|HOLDER|ability: Mummy|ATTACKER|[ability] OLD
                    # leaves the attacker with the holder's ability
                    self._add("ability", ability_name, self._holders(split[4], None), "-activate:swap")
            elif effect.startswith("item:"):
                self._add(
                    "item", effect.split(":", 1)[1], self._holders(subject, of), f"{tag}:item"
                )
            elif effect.startswith("move: Skill Swap") or effect == "Skill Swap":
                self._skill_swap(split, subject, of)
            elif tag == "-activate" and effect.startswith("move: "):
                move = effect.split(":", 1)[1]
                if to_id(move) in ("mimic",):
                    self.events_for(subject).add("mimic")
        if from_kind in ("ability", "item") and tag != "-ability":
            # `[of]` is the effect's SOURCE: the holder for damage / status / weather lines
            # (Rough Skin, Rocky Helmet, Drought), but the ATTACKER for an absorbing heal
            # (Water Absorb) and the thief's victim for Pickpocket; the dump arbitrates, with the
            # tag's usual holder tried first.
            of_first = tag in ("-damage", "-weather", "-fieldstart", "-sidestart", "-status", "-start") or (
                tag == "-item" and to_id(from_name) == "frisk"
            )
            order = (of, subject) if of_first else (subject, of)
            self._add(from_kind, from_name, self._holders(*order), f"{tag}[from {from_kind}]")
        if from_kind == "move" and tag == "-heal" and to_id(from_name) == "healingwish":
            pass

        # --- items -------------------------------------------------------------------
        if tag == "-item" and len(split) > 3:
            self._add("item", split[3], self._holders(subject, None), "-item", True)
            mon = self._mon(split[2])
            if mon:
                self.mons[mon].item_gone = False
        elif tag == "-enditem" and len(split) > 3:
            mon = self._mon(split[2])
            if mon and mon in self.cud_chewing:
                self.cud_chewing.discard(mon)  # Cud Chew re-eats an EARLIER berry, not the held item
                mon = None
            if mon:
                self._add("item", split[3], (mon,), "-enditem", True)
                self._add("item_gone", split[3], (mon,), "-enditem", True)
        elif tag == "-mega" and len(split) > 4:
            self._add("item", split[4], self._holders(subject, None), "-mega", True)
        if tag == "-curestatus" or tag == "-status":
            pass

        # --- moves -------------------------------------------------------------------
        if tag == "move" and len(split) > 3:
            mon = self._mon(split[2])
            if mon is not None:
                # Struggle is not one of the Pokemon's moves and no client lists it as one
                called_from_own_set = to_id(from_name) in ("sleeptalk", "round") if from_kind == "move" else False
                if (not kw.get("from") or called_from_own_set) and to_id(split[3]) != "struggle":
                    self._add("move", split[3], (mon,), "move")
                if to_id(split[3]) in ("rest",):
                    self.events_for(split[2]).add("rest")
        if tag == "cant" and len(split) > 4 and split[4]:
            pass

        # --- history facts -----------------------------------------------------------
        if tag == "-end" and len(split) > 3 and split[3] == "Illusion":
            mon = self._mon(split[2])
            if mon:
                self.step_flags.add(f"illusion_break:{mon[0]}")
                self.break_pos.setdefault(mon[0], self.pos)
        if tag == "replace":
            mon = self._mon(split[2])
            if mon:
                self.step_flags.add(f"illusion_break:{mon[0]}")
                self.break_pos.setdefault(mon[0], self.pos)
        if tag == "-start" and len(split) > 3 and split[3].startswith("typeadd"):
            self.events_for(split[2]).add("typeadd")
        if tag == "-start" and len(split) > 3 and split[3] == "move: Yawn":
            self.events_for(split[2]).add("yawned")
        if tag == "-end" and len(split) > 3 and split[3] == "move: Yawn":
            self.events_for(split[2]).discard("yawned")
        if tag == "move" and from_kind == "move" and to_id(from_name) in (
            "copycat", "instruct", "sleeptalk"
        ) and to_id(split[3]) in _PROTECT_FAMILY:
            self.events_for(split[2]).add("called_protect")
        if tag == "-activate" and len(split) > 3 and split[3] == "ability: Cud Chew":
            mon = self._mon(split[2])
            if mon:
                self.cud_chewing.add(mon)
        if tag == "-singleturn" and len(split) > 3 and split[3] == "move: Instruct":
            self.events_for(split[2]).add("called_protect")
        if tag == "-transform" and len(split) > 3:
            self.events_for(split[3]).add("transform_target")
            self.events_for(split[2]).add("transformed")
            holders = self._holders(split[2], split[3])
            if len(holders) == 2:
                self.pending.append(
                    Candidate("transform", "", holders, "-transform", self.step, explicit=True)
                )
        if tag in ("switch", "drag"):
            mon = self._mon(split[2])
            if mon:
                self.mons[mon].events.discard("yawned")
                self.mons[mon].events.discard("transformed")
        self.tag_counts[tag] += 1

    def events_for(self, ident: str | None) -> set[str]:
        mon = self._mon(ident) if ident else None
        return self.mons[mon].events if mon else set()

    def _skill_swap(self, split: list[str], subject: str | None, of: str | None) -> None:
        # |-activate|SRC|move: Skill Swap|SRC_ABILITY|TARGET_ABILITY|[of] TARGET  (across sides,
        # abilities shown) or |-activate|SRC|move: Skill Swap||[of] TARGET (same side, hidden)
        shown = [p for p in split[4:6] if p and not p.startswith("[")]
        if len(shown) == 2:
            # (as announced to the opposing side: source gets target's ability, and vice versa)
            self._add("ability", shown[0], self._holders(subject, None), "skillswap:new", True)
            self._add("ability", shown[1], self._holders(of, None), "skillswap:new-of", True)
        else:
            for ident in (subject, of):
                self.events_for(ident).add("hidden_skillswap")
            holders = self._holders(subject, of)
            if len(holders) == 2:
                self.pending.append(
                    Candidate("swap_abilities", "", holders, "skillswap:hidden", self.step, explicit=True)
                )

    # -- attribution --------------------------------------------------------------------------

    def resolve(
        self,
        truth_mons: dict[tuple[str, str], dict[str, Any]],
        alias_before: dict[tuple[str, str], tuple[str, str]],
        alias_after: dict[tuple[str, str], tuple[str, str]] | None = None,
    ) -> list[Candidate]:
        """Attribute pending candidates using the Showdown dump; return the unattributed ones.

        `truth_mons` maps (role, protocol name) to the dumped Pokemon dict (`_truth_mon`);
        `alias` maps a disguise's (role, name) to the real Illusion holder's key.
        """

        # refresh "ever held" before matching, so a swap undone inside one step still matches
        for key, tm in truth_mons.items():
            self.ever_ability[key] |= {tm["ability"], tm["base_ability"]} - {""}
            self.ever_item[key] |= {tm["item"], tm["last_item"]} - {""}
            if tm["item"]:
                self.ever_item[key].add(tm["item"])
            for m in tm["base_moves"] + tm["moves"]:
                self.ever_item[key].add("move:" + m)
        unresolved: list[Candidate] = []
        alias_after = alias_after if alias_after is not None else alias_before
        for cand in self.pending:
            winner = None
            # A line announced under a disguise's ident belongs to the real holder only while
            # the disguise holds: before the break line in a step that breaks it, the disguise
            # that was up at the START of the step; after it, none of those.
            role = cand.holders[0][0]
            broke_at = self.break_pos.get(role)
            alias = dict(alias_after)
            if broke_at is None or cand.pos < broke_at:
                alias.update(alias_before)
            elif broke_at is not None:
                alias = {k: v for k, v in alias.items() if k not in alias_before}
            if cand.kind == "swap_abilities":
                a, b = (self.mons[h] for h in cand.holders)
                la = a.abilities[-1] if a.abilities else None
                lb = b.abilities[-1] if b.abilities else None
                if la or lb:
                    if lb:
                        a.abilities.append((lb[0], "skillswap:inferred"))
                    if la:
                        b.abilities.append((la[0], "skillswap:inferred"))
                continue
            if cand.explicit:
                winner = cand.holders[0]
            for holder in cand.holders:
                if winner is not None:
                    break
                real = alias.get(holder, holder)
                tm = truth_mons.get(real)
                if tm is None:
                    continue
                if cand.kind == "ability" and cand.value in self.ever_ability[real]:
                    winner = holder
                elif cand.kind in ("item", "item_gone") and (
                    cand.value in self.ever_item[real] or cand.value == tm["item"]
                ):
                    winner = holder
                elif cand.kind == "move" and (
                    ("move:" + cand.value) in self.ever_item[real]
                    or cand.value in tm["moves"]
                    or tm["transformed"]
                ):
                    winner = holder
                if winner:
                    break
            if winner is None and broke_at is not None and cand.pos < broke_at:
                # A Zoroark that switched in disguised and broke in the SAME step is visible as a
                # disguise at neither end of it: try the side's Illusion holders directly.
                for key, tm in truth_mons.items():
                    if key[0] != role or tm["base_ability"] != "illusion":
                        continue
                    if (cand.kind == "item" and cand.value in self.ever_item[key]) or (
                        cand.kind == "move" and ("move:" + cand.value) in self.ever_item[key]
                    ) or (cand.kind == "ability" and cand.value in self.ever_ability[key]):
                        winner = key
                        break
            if winner is None:
                unresolved.append(cand)
                continue
            # What a client records is keyed by the ident it was shown. While a disguise holds
            # that is the disguise's name; the events are logged so that, if the Illusion BREAKS,
            # they can be moved to the real holder (and left where they are if it never does).
            disguised_holder = alias.get(winner)
            cand.resolved = winner
            mk = self.mons[winner]
            if cand.kind == "transform":
                copier, target = cand.holders
                target_ability = (
                    truth_mons[alias.get(target, target)]["ability"]
                    if target[0] == self.role and alias.get(target, target) in truth_mons
                    else self.current_ability(*target)
                )
                if target_ability:
                    mk.abilities.append((target_ability, "transform"))
                continue
            log = self.episode_log.setdefault(winner, []) if disguised_holder else None
            if cand.kind == "ability":
                if log is not None:
                    log.append(("ability", cand.value, cand.tag))
                mk.abilities.append((cand.value, cand.tag))
            elif cand.kind == "item":
                if log is not None:
                    log.append(("item", cand.value, (mk.item_shown, mk.item_gone)))
                mk.item_shown = cand.value
                mk.item_gone = False
            elif cand.kind == "item_gone":
                if log is not None:
                    log.append(("item_gone", cand.value, (mk.item_shown, mk.item_gone)))
                mk.item_gone = True
            elif cand.kind == "move":
                real_tm = truth_mons.get(disguised_holder or winner)
                if cand.value == "transform" or not (
                    cand.transformed or (real_tm and real_tm["transformed"])
                ):
                    if log is not None:
                        log.append(("move", cand.value, cand.value in mk.moves))
                    mk.moves.add(cand.value)  # (a transformed Pokemon uses the TARGET's moves)
        # Disguise episodes that ended this step: an Illusion break hands what was recorded under
        # the disguise's name to its real holder; any other ending leaves it where the client put it.
        for key, holder in alias_before.items():
            if key in alias_after:
                continue
            log = self.episode_log.pop(key, [])
            if self.break_pos.get(key[0]) is None:
                continue
            disguise, real = self.mons[key], self.mons[holder]
            for kind, value, prior in log:
                if kind == "ability":
                    real.abilities.append((value, "illusion-break"))
                    for i in range(len(disguise.abilities) - 1, -1, -1):
                        if disguise.abilities[i][0] == value:
                            del disguise.abilities[i]
                            break
                elif kind in ("item", "item_gone"):
                    if kind == "item":
                        real.item_shown, real.item_gone = value, False
                    else:
                        real.item_gone = True
                    disguise.item_shown, disguise.item_gone = prior
                elif kind == "move":
                    real.moves.add(value)
                    if not prior:
                        disguise.moves.discard(value)
        self.pending = []
        return unresolved

    # -- queries ------------------------------------------------------------------------------

    def current_ability(self, role: str, name: str) -> str | None:
        mk = self.mons.get((role, name))
        return mk.abilities[-1][0] if mk and mk.abilities else None

    def revealed_abilities(self, role: str, name: str) -> set[str]:
        mk = self.mons.get((role, name))
        return {a for a, _ in mk.abilities} if mk else set()

    def revealed_moves(self, role: str, name: str) -> set[str]:
        mk = self.mons.get((role, name))
        return set(mk.moves) if mk else set()

    def has_event(self, role: str, name: str, event: str) -> bool:
        mk = self.mons.get((role, name))
        return bool(mk and event in mk.events)
