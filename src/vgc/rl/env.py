"""Direct `BattleStream` environment: poke-env's parser, none of its transport.

This is Phase 1 of `docs/rl_roadmap.md`. It drives `tools/sim_worker.mjs` over
JSON-lines and feeds each side's protocol lines into its OWN `DoubleBattle`, so a
battle runs with no Showdown server, no websocket, no account, and no asyncio.

## What this is and is not

It replaces `poke_env.player.Player`'s TRANSPORT and message pump only. Every consumer
of a parsed battle -- `vgc.actions.enumerate_joint_orders`, `vgc.evaluator`, `vgc.sets`,
`vgc.rl.encoding` -- keeps working unchanged, because what they consume is a
`DoubleBattle` and that is exactly what this module produces. Reimplementing the
Showdown protocol parser is explicitly out of scope; `AbstractBattle.parse_message` /
`parse_request` are drivable directly and that is what we do.

## Fogging is structural, not reimplemented

`DirectBattle` keeps one `DoubleBattle` per side and feeds each ONLY the lines the
worker read off that player's `.p1`/`.p2` stream. Those are already the fogged views the
real server sends -- p1 sees its own Charizard as `163/163` and the opponent's as
`100/100` (percentages), and never sees unrevealed moves, items, or bench HP. There is
no filtering step here that could drift out of sync with the server's, because there is
no filtering step at all. The omniscient stream is used only for the terminal result,
which the worker reports as `winner`.

## Choice strings

poke-env builds `/choose move heatwave 1, move weatherball 2`; the sim's `>p1 ...` input
wants that without the client-side `/choose ` prefix. `choice_string` does that one
transformation, so callers can hand it a `DoubleBattleOrder` straight out of
`enumerate_joint_orders`.

## Errors are fatal here, deliberately

`Player` treats `|error|[Invalid choice]` as recoverable and retries with a default
move -- correct against a live ladder opponent, wrong for an RL environment, where an
illegal choice means our legality logic disagrees with the simulator and every
trajectory collected after it is suspect. `DirectBattle.step` raises `InvalidChoice`
instead. Same reasoning for unknown protocol messages: rather than skipping anything we
do not recognize (which would silently drop battle state), only the known-cosmetic tags
in `_COSMETIC_MESSAGES` are skipped and everything else reaches poke-env, which raises
`NotImplementedError` on tags it cannot parse.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from poke_env.battle.double_battle import DoubleBattle
from poke_env.player.battle_order import DoubleBattleOrder
from poke_env.teambuilder.teambuilder import Teambuilder

from vgc.node import find_node

DEFAULT_FORMAT = "gen9championsvgc2026regmb"
DEFAULT_SHOWDOWN_REPO = Path.home() / "code" / "projects" / "pokemon-showdown"
WORKER_SCRIPT = Path(__file__).resolve().parents[3] / "tools" / "sim_worker.mjs"

SIDES: tuple[str, str] = ("p1", "p2")

# Protocol tags the direct sim emits that carry no battle state and that poke-env's
# parse_message would raise NotImplementedError on (it has no default branch). Kept as a
# short EXPLICIT allowlist rather than "skip anything unrecognized" so that a genuinely
# unparsed state-carrying message fails loudly instead of corrupting observations.
#
# The membership here was found empirically, not guessed: a discovery run over 12
# random-vs-random battles (~330 turns) collected every tag poke-env refused, and `t:`
# and `uhtmlchange` were the only two.
#
# `t:` is the sim's wall-clock timestamp line. `uhtmlchange` blanks the Open Team Sheets
# prompt button the sim renders into the chat area (`uhtml` itself is already in
# poke-env's own MESSAGES_TO_IGNORE); OTS is never accepted in this environment, so both
# are pure decoration.
_COSMETIC_MESSAGES = frozenset({"t:", "uhtmlchange"})

_LOGGER = logging.getLogger("vgc.rl.env")


class SimWorkerError(RuntimeError):
    """The Node worker reported an error or died."""


class InvalidChoice(SimWorkerError):
    """The simulator rejected a choice we believed was legal.

    Always a bug in our legality handling (or in the choice string), never something to
    retry around -- see this module's docstring.
    """


class SimWorker:
    """A long-lived `tools/sim_worker.mjs` process addressed over JSON-lines.

    One worker hosts many concurrent battles; `battle_id` namespaces them. Throughput
    comes from running one worker per core (the simulator is CPU-bound), not from
    threads inside one process.
    """

    def __init__(
        self,
        showdown_repo: str | Path = DEFAULT_SHOWDOWN_REPO,
        *,
        node: str | Path | None = None,
        script: str | Path = WORKER_SCRIPT,
    ) -> None:
        self.showdown_repo = Path(showdown_repo)
        self._next_rid = 0
        self._process = subprocess.Popen(
            [str(node or find_node()), str(script), str(self.showdown_repo)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send one command and return its response, raising on a worker-side error."""

        if self._process.poll() is not None:
            raise SimWorkerError(f"worker exited with code {self._process.returncode}")
        self._next_rid += 1
        payload = {**payload, "rid": self._next_rid}
        assert self._process.stdin is not None and self._process.stdout is not None
        self._process.stdin.write(json.dumps(payload) + "\n")
        self._process.stdin.flush()
        line = self._process.stdout.readline()
        if not line:
            stderr = self._process.stderr.read() if self._process.stderr else ""
            raise SimWorkerError(f"worker closed its stdout; stderr:\n{stderr}")
        response = json.loads(line)
        if response.get("rid") != payload["rid"]:
            raise SimWorkerError(
                f"response rid {response.get('rid')} does not match request {payload['rid']}"
            )
        if "error" in response:
            raise SimWorkerError(response["error"])
        return response

    def close(self) -> None:
        if self._process.poll() is None:
            try:
                assert self._process.stdin is not None
                self._process.stdin.close()
                self._process.wait(timeout=5)
            except (subprocess.TimeoutExpired, BrokenPipeError, AssertionError):
                self._process.kill()
                self._process.wait(timeout=5)

    def __enter__(self) -> SimWorker:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def choice_string(order: DoubleBattleOrder | str) -> str:
    """Turn a poke-env order into what the sim's `>p1 ...` input expects.

    poke-env emits client-side commands (`/choose move heatwave 1, ...`, `/team 1234`);
    `BattleStream` wants the bare choice. Plain strings pass through with the same
    leading-slash handling so callers can hand this a raw `"team 1234"` too.
    """

    message = order if isinstance(order, str) else order.message
    message = message.removeprefix("/choose ")
    return message.removeprefix("/")


@dataclass
class StepResult:
    """What changed for each side after one `DirectBattle.step`."""

    request_state: str
    ended: bool
    winner: str | None
    lines: dict[str, list[str]] = field(default_factory=dict)


class DirectBattle:
    """One battle driven through the worker, with a fogged `DoubleBattle` per side.

    Typical use::

        with SimWorker(repo) as worker:
            battle = DirectBattle.start(worker, "b0", team, team, seed=[1, 2, 3, 4])
            while not battle.ended:
                choices = {
                    side: choice_string(pick(battle.battles[side]))
                    for side in battle.sides_to_move()
                }
                battle.step(choices)
            outcome = battle.outcome("p1")   # +1 / -1 / 0
    """

    def __init__(
        self,
        worker: SimWorker,
        battle_id: str,
        *,
        usernames: dict[str, str],
        gen: int = 9,
    ) -> None:
        self.worker = worker
        self.battle_id = battle_id
        self.usernames = usernames
        self.battles: dict[str, DoubleBattle] = {
            side: DoubleBattle(
                battle_tag=battle_id,
                username=usernames[side],
                logger=_LOGGER,
                gen=gen,
            )
            for side in SIDES
        }
        self.request_state: str = ""
        self.ended: bool = False
        self.winner: str | None = None
        self._waiting: dict[str, bool] = {side: True for side in SIDES}
        # side -> {nickname: TeambuilderPokemon}, consumed by _apply_own_spreads.
        self._teambuilder: dict[str, dict[str, Any]] = {side: {} for side in SIDES}

    @classmethod
    def start(
        cls,
        worker: SimWorker,
        battle_id: str,
        p1_team: str,
        p2_team: str,
        *,
        battle_format: str = DEFAULT_FORMAT,
        seed: Sequence[int] | None = None,
        usernames: dict[str, str] | None = None,
    ) -> DirectBattle:
        """Create the battle in the worker and parse both sides up to the first request.

        `p1_team`/`p2_team` are packed team strings (`teams/*.packed.txt`). They are also
        handed to each side's own `DoubleBattle` as its teambuilder team, so our own
        Stat Points and natures are exact rather than estimated -- `vgc.evaluator`
        reads them straight off `Pokemon.evs`/`Pokemon.nature`, and falling back to
        `vgc.stats.default_opponent_spread` for our OWN side would quietly make every
        heuristic opponent weaker here than on the poke-env path.
        """

        usernames = usernames or {"p1": "p1", "p2": "p2"}
        battle = cls(worker, battle_id, usernames=usernames)
        teams = {"p1": p1_team, "p2": p2_team}
        for side in SIDES:
            battle._teambuilder[side] = {
                entry.nickname or entry.species or "": entry
                for entry in Teambuilder.parse_packed_team(teams[side])
            }
        payload: dict[str, Any] = {
            "cmd": "start",
            "id": battle_id,
            "format": battle_format,
            "p1": {"name": usernames["p1"], "team": p1_team},
            "p2": {"name": usernames["p2"], "team": p2_team},
        }
        if seed is not None:
            payload["seed"] = list(seed)
        battle._apply(worker.request(payload))
        return battle

    def sides_to_move(self) -> list[str]:
        """Sides the simulator is currently waiting on a choice from.

        A side that gets a `{"wait": true}` request (the other player has a forced
        switch, say) is not asked to choose -- `step` omits it, and the worker writes
        nothing for it.
        """

        if self.ended:
            return []
        return [side for side in SIDES if not self._waiting[side]]

    def step(self, choices: dict[str, str]) -> StepResult:
        """Submit one choice per side that owes one and advance the battle."""

        expected = set(self.sides_to_move())
        given = {side for side, value in choices.items() if value is not None}
        if given != expected:
            raise ValueError(
                f"battle {self.battle_id} expects choices from {sorted(expected)}, got "
                f"{sorted(given)}"
            )
        payload: dict[str, Any] = {"cmd": "choose", "id": self.battle_id}
        for side in SIDES:
            payload[side] = choices.get(side)
        return self._apply(self.worker.request(payload))

    def close(self) -> None:
        try:
            self.worker.request({"cmd": "close", "id": self.battle_id})
        except SimWorkerError:
            # Battle already gone (ended and reaped, or the worker died) -- nothing to
            # release on our side either way.
            pass

    def outcome(self, side: str) -> float:
        """`+1` win / `-1` loss / `0` draw or unfinished, from `side`'s perspective."""

        if not self.ended or self.winner is None:
            return 0.0
        return 1.0 if self.winner == self.usernames[side] else -1.0

    # --- internals -------------------------------------------------------------------

    def _apply(self, response: dict[str, Any]) -> StepResult:
        self.request_state = response.get("requestState", "")
        self.ended = bool(response.get("ended"))
        self.winner = response.get("winner")
        lines = {side: list(response.get(side) or []) for side in SIDES}
        for side in SIDES:
            self._ingest(side, lines[side])
        if self.ended:
            self._waiting = {side: True for side in SIDES}
        return StepResult(
            request_state=self.request_state,
            ended=self.ended,
            winner=self.winner,
            lines=lines,
        )

    def _apply_own_spreads(self, side: str) -> None:
        """Copy our own Stat Points/nature onto our own team from the packed team.

        Needed because poke-env only learns our spread from an Open Team Sheets
        `|showteam|` message (`Player._handle_battle_message`), and OTS never fires here.
        Without this, `vgc.evaluator._our_pokemon_state` finds `Pokemon.evs is None` and
        falls back to `vgc.stats.default_opponent_spread` -- i.e. every heuristic agent
        would play its OWN team off a guessed spread, and direct-env results would not
        line up with the poke-env gate path.

        Deliberately NOT `AbstractBattle.apply_teambuilder_team`, which routes through
        `Pokemon._update_from_teambuilder` and recomputes `_stats` with poke-env's
        VANILLA gen-9 EV formula. The champions mod is linear in Stat Points
        (`HP = base + SP + 75`; see CLAUDE.md and `vgc/stats.py`), and the sim's own
        request already carried the mod-correct stats, so overwriting them with vanilla
        numbers would replace right answers with wrong ones. Only the three fields the
        request genuinely cannot tell us are set here; item/ability/moves/stats come
        from the request as before.
        """

        teambuilder = self._teambuilder[side]
        if not teambuilder:
            return
        for ident, pokemon in self.battles[side].team.items():
            if pokemon.evs is not None:
                continue
            entry = teambuilder.get(ident.split(": ", 1)[-1])
            if entry is None:
                continue
            pokemon._evs = entry.evs
            pokemon._ivs = entry.ivs
            pokemon._nature = (entry.nature or "serious").lower()

    def _ingest(self, side: str, lines: Iterable[str]) -> None:
        battle = self.battles[side]
        saw_request = False
        for line in lines:
            split = line.split("|")
            if len(split) < 2:
                continue
            tag = split[1]
            if tag == "request":
                # Rejoin rather than take split[2]: the request payload is JSON and may
                # itself contain a "|" inside a string value.
                payload = "|".join(split[2:])
                if not payload:
                    continue
                request = json.loads(payload)
                battle.parse_request(request)
                self._apply_own_spreads(side)
                self._waiting[side] = bool(battle._wait)
                saw_request = True
            elif tag == "win":
                battle.won_by(split[2])
            elif tag == "tie":
                battle.tied()
            elif tag == "error":
                raise InvalidChoice(f"{side} in battle {self.battle_id}: {'|'.join(split[2:])}")
            elif tag in _COSMETIC_MESSAGES:
                continue
            else:
                battle.parse_message(split)
        if not saw_request:
            # No new request for this side this step means the simulator is not waiting
            # on it (it is mid-resolution, or the battle just ended).
            self._waiting[side] = True
