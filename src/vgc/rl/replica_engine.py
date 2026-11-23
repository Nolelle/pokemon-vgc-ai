"""Run the local replication trainer on pikapro through DirectBattle's player views."""

from dataclasses import dataclass

from vgc.config import FORMAT_ID
from vgc.rl.env import SIDES


@dataclass
class _Game:
    battle: object
    event_cursor: int = 0


class ReplicaRustWorker:
    """Implement the three DirectBattle operations used in league training."""

    def __init__(self):
        import pikapro

        self._engine = pikapro
        self._games = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._games.clear()

    def request(self, payload):
        command = payload["cmd"]
        key = payload["id"]
        if command == "close":
            self._games.pop(key, None)
            return {}
        rejections = {}
        if command == "start":
            if payload["format"] != FORMAT_ID or key in self._games:
                raise ValueError("Expected a new Champions Reg M-C battle")
            players = [payload[side]["name"] for side in SIDES]
            teams = [payload[side]["team"] for side in SIDES]
            self._games[key] = _Game(
                self._engine.Battle(
                    "championsvgc", teams, seed=payload.get("seed"), players=players
                )
            )
        elif command == "choose":
            for i, side in enumerate(SIDES):
                if payload.get(side) is not None:
                    try:
                        self._games[key].battle.submit(i, payload[side])
                    except ValueError as exc:
                        if not str(exc).startswith("[Unavailable choice]"):
                            raise
                        rejections[side] = str(exc)
        else:
            raise ValueError(f"Unsupported replication engine command {command!r}")

        game = self._games[key]
        battle = game.battle
        result = {"ended": battle.over, "winner": None, "requestState": ""}
        if battle.over and battle.winner is not None:
            result["winner"] = battle.players[battle.winner]
        for i, side in enumerate(SIDES):
            # The cursor counts engine events. One event may emit multiple protocol lines.
            lines = battle.log(side, start=game.event_cursor)
            if not battle.over:
                kind = battle.request_kind(i)
                if kind != "wait":
                    result["requestState"] = {
                        "team_preview": "teampreview",
                        "move": "move",
                        "force_switch": "switch",
                    }[kind]
                # A wait request still refreshes this player's private team state.
                # If a choice was already accepted, the engine instead holds the
                # old request. Re-sending it would ask the player to choose twice.
                if kind != "wait" or not battle.request(i).answered:
                    if side in rejections:
                        lines.append("|error|" + rejections[side])
                    lines.append("|request|" + battle.request_json(i))
            result[side] = lines
        game.event_cursor = battle.event_count
        return result
