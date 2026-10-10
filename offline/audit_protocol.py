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
        self.last_step_idents: set[tuple[str, str]] = set()

    # -- ingest ---------------------------------------------------------------------------

    def _mon(self, ident: str) -> tuple[str, str] | None:
        parsed = parse_ident(ident)
        return (parsed[0], parsed[2]) if parsed else None

    def feed(self, lines: list[str]) -> None:
        self.step += 1
        self.step_flags = set()
        self.last_step_idents = set()
        for raw in lines:
            split = raw.split("|")
            if len(split) < 3:
                continue
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
            self.pending.append(
                Candidate(kind, to_id(value), holders, tag, self.step, explicit=explicit)
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
        elif tag in ("-activate", "-start", "-end", "-block", "-immune", "-fail", "-singleturn",
                     "-singlemove", "-sidestart", "-fieldstart", "-hint", "-prepare"):
            # `ability: X` as the effect name: the effect belongs to the subject (or the [of]
            # holder for -block, where the subject is the protected Pokemon)
            effect = split[3] if len(split) > 3 else ""
            if effect.startswith("ability:"):
                self._add(
                    "ability", effect.split(":", 1)[1], self._holders(subject, of), f"{tag}:ability"
                )
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
        if from_kind == "ability" and tag != "-ability":
            self._add("ability", from_name, self._holders(subject, of), f"{tag}[from ability]")
        if from_kind == "item":
            self._add("item", from_name, self._holders(subject, of), f"{tag}[from item]")
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
                if not kw.get("from"):
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
        if tag == "replace":
            mon = self._mon(split[2])
            if mon:
                self.step_flags.add(f"illusion_break:{mon[0]}")
        if tag == "-start" and len(split) > 3 and split[3].startswith("typeadd"):
            self.events_for(split[2]).add("typeadd")
        if tag == "-start" and len(split) > 3 and split[3] == "move: Yawn":
            self.events_for(split[2]).add("yawned")
        if tag == "-end" and len(split) > 3 and split[3] == "move: Yawn":
            self.events_for(split[2]).discard("yawned")
        if tag == "-transform":
            self.events_for(split[2]).add("transformed")
        if tag in ("switch", "drag"):
            mon = self._mon(split[2])
            if mon:
                self.mons[mon].events.discard("yawned")
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

    # -- attribution --------------------------------------------------------------------------

    def resolve(
        self,
        truth_mons: dict[tuple[str, str], dict[str, Any]],
        alias: dict[tuple[str, str], tuple[str, str]],
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
        for cand in self.pending:
            winner = None
            if cand.explicit or (cand.kind == "move" and cand.value == "struggle"):
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
            if winner is None:
                unresolved.append(cand)
                continue
            cand.resolved = winner
            mk = self.mons[winner]
            if cand.kind == "ability":
                mk.abilities.append((cand.value, cand.tag))
            elif cand.kind == "item":
                mk.item_shown = cand.value
                mk.item_gone = False
            elif cand.kind == "item_gone":
                mk.item_gone = True
            elif cand.kind == "move":
                mk.moves.add(cand.value)
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
