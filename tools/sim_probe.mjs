// Generic Showdown BattleStream driver used as ground truth for src/vgc/stats.py and
// src/vgc/damage.py: it runs a scripted battle (or just a team-preview-only "battle" for
// pure stat dumps) directly against the local Showdown checkout's built sim -- NOT
// through the websocket server -- and prints every protocol line plus any `>eval`
// results as JSON, for a Python caller to parse and assert against.
//
// Usage:
//   node tools/sim_probe.mjs <path-to-pokemon-showdown-repo> <scenario.json>
//
// Scenario JSON shape:
//   {
//     "format": "gen9championscustomgame" | "gen9championsdoublescustomgame" | ...,
//     "seed": [1, 2, 3, 4],                 // optional; RandomBattleSeed-shaped
//     "p1": {"name": "p1", "team": [<PokemonSet-like objects, see below>]},
//     "p2": {"name": "p2", "team": [...]},
//     "commands": ["p1 team 1", "p2 team 1", "p1 move 1", "p2 move 1", "eval ..."]
//   }
//
// Each command in `commands` is written verbatim as `>${command}` to the underlying
// BattleStream (i.e. don't include the leading `>`). Team preview and move/switch
// choices are the caller's responsibility to script in order -- this tool does not
// auto-advance turns.
//
// PokemonSet objects (passed through Teams.pack -- see sim/teams.ts for the full
// interface): {species, name?, item?, ability, moves, nature, evs (Stat Points for this
// mod, NOT vanilla EVs -- see src/vgc/stats.py), ivs, level, gender?, shiny?, teraType?}.
// `evs`/`ivs` are {hp,atk,def,spa,spd,spe} objects; omitted stats default to 0 (evs) / 31
// (ivs, via Teams.pack's own default-omission behavior) which is what this mod requires
// anyway (see src/vgc/stats.py FIXED_IV).
//
// Output (stdout, single JSON object):
//   {"log": ["<protocol line>", ...], "evalResults": ["<<< ...", ...]}
// `log` is every non-empty protocol line across the whole battle in order (e.g.
// `|switch|p1a: Garchomp|Garchomp, L50, M|145/145`, `|-damage|p2a: ...|80/145`).
// `evalResults` is just the `<<< ...` lines from any `eval` commands, split out for
// convenience since those carry arbitrary JSON payloads (e.g. a stats dump).
import { pathToFileURL } from "node:url";
import path from "node:path";
import fs from "node:fs";

const [, , showdownRepo, scenarioPath] = process.argv;
if (!showdownRepo || !scenarioPath) {
	console.error("usage: node sim_probe.mjs <showdown-repo> <scenario.json|->");
	console.error(`  "-" for <scenario.json> reads the scenario JSON from stdin instead.`);
	process.exit(1);
}

const simIndexPath = path.join(showdownRepo, "dist", "sim", "index.js");
if (!fs.existsSync(simIndexPath)) {
	console.error(`${simIndexPath} not found -- run "node build" in the showdown repo first`);
	process.exit(1);
}

const simModule = await import(pathToFileURL(simIndexPath).href);
const Sim = simModule.BattleStream ? simModule : simModule.default;
const { BattleStream, getPlayerStreams, Teams } = Sim;

const scenarioText = scenarioPath === "-" ? fs.readFileSync(0, "utf8") : fs.readFileSync(scenarioPath, "utf8");
const scenario = JSON.parse(scenarioText);

const streams = getPlayerStreams(new BattleStream({ debug: false }));

const log = [];
const evalResults = [];

// Start draining the omniscient stream BEFORE writing input -- BattleStream pushes
// output synchronously as input is processed, so the reader must already be attached.
const drain = (async () => {
	for await (const chunk of streams.omniscient) {
		for (const line of chunk.split("\n")) {
			if (!line) continue;
			log.push(line);
			// `>eval` results are emitted as `battle.add('', '<<< ' + result)`, which
			// serializes to a protocol line with an empty first field: `||<<< ...`.
			if (line.startsWith("||<<< ")) evalResults.push(line.slice(6));
		}
	}
})();

const startSpec = { formatid: scenario.format };
if (scenario.seed) startSpec.seed = scenario.seed;

const p1Team = Teams.pack(scenario.p1.team);
const p2Team = Teams.pack(scenario.p2.team);
if (!p1Team || !p2Team) {
	console.error("Teams.pack produced an empty team -- check scenario p1/p2 team objects");
	process.exit(1);
}

const inputLines = [
	`>start ${JSON.stringify(startSpec)}`,
	`>player p1 ${JSON.stringify({ name: scenario.p1.name || "p1", team: p1Team })}`,
	`>player p2 ${JSON.stringify({ name: scenario.p2.name || "p2", team: p2Team })}`,
	...(scenario.commands || []).map((cmd) => `>${cmd}`),
];

await streams.omniscient.write(inputLines.join("\n"));
await streams.omniscient.writeEnd();
await drain;

console.log(JSON.stringify({ log, evalResults }));
