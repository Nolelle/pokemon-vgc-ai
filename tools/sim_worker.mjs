// Long-lived Showdown `BattleStream` worker for the Phase 1 RL environment
// (`docs/rl_roadmap.md`). Hosts many concurrent battles in one Node process and speaks
// JSON-lines over stdin/stdout, so the Python side can step battles WITHOUT a Showdown
// server, a websocket, accounts, or asyncio.
//
// Relationship to `tools/sim_probe.mjs`: same `dist/sim` + `getPlayerStreams` pattern,
// but sim_probe runs ONE fully-scripted battle to completion and exits (it is ground-truth
// tooling for `src/vgc/damage.py`). This worker is interactive and long-lived -- the
// caller supplies each side's choice one decision at a time and reads back what each
// player saw.
//
// ## Why per-player streams, not the omniscient one
//
// `getPlayerStreams()` gives `.p1`/`.p2` streams carrying exactly the protocol lines the
// real server would send THAT player -- i.e. already fogged (no hidden items, no
// unrevealed moves, no opponent's bench HP). Phase 2 of the roadmap requires fogged
// observations, and taking them from these streams makes the masking correct by
// construction instead of something we reimplement and hope matches. `.omniscient` is
// read ONLY to detect the terminal `|win|`/`|tie|`, which the caller needs as the reward
// and which a player stream also carries, but omniscient is the unambiguous source.
//
// ## Wire protocol
//
// One JSON object per line, in and out. Every request may carry an `rid` which is echoed
// on the response so the caller can match them up.
//
//   -> {"cmd":"start","id":"b0","format":"gen9championsvgc2026regmb",
//       "seed":[1,2,3,4],                         // optional; RandomBattleSeed-shaped
//       "p1":{"name":"p1","team":[<PokemonSet>,...] | "<packed string>"},
//       "p2":{...}}
//   <- {"rid":...,"id":"b0","p1":[...lines],"p2":[...lines],
//       "requestState":"teampreview","ended":false,"winner":null}
//
//   -> {"cmd":"choose","id":"b0","p1":"team 1234","p2":"team 1234"}
//   -> {"cmd":"choose","id":"b0","p1":"move 1 1, move 1 2","p2":"move 2 -1, pass"}
//       Either side may be omitted/null when that side has nothing to choose (the sim
//       sends it a `wait` request). Values are written verbatim as `>p1 <value>`.
//   <- {"rid":...,"id":"b0","p1":[...],"p2":[...],"requestState":"move",
//       "ended":false,"winner":null}
//
//   -> {"cmd":"close","id":"b0"}            <- {"rid":...,"id":"b0","closed":true}
//   -> {"cmd":"ping"}                       <- {"rid":...,"pong":true}
//
//   -> {"cmd":"batch","items":[{"cmd":"choose","id":"b0",...},{"cmd":"choose","id":"b1",...}]}
//   <- {"rid":...,"results":[<one response per item, positionally>]}
//       Advances many INDEPENDENT battles per round trip; see handleBatch for why this
//       is a latency win rather than a serialization one. At most one command per battle
//       id per batch. A failing item yields {"id":...,"error":...} in its slot instead of
//       failing the whole batch.
//
// Errors come back as {"rid":...,"id":...,"error":"<message>"} and never kill the
// process: one malformed battle must not take down a worker hosting dozens of others.
// Successful battle responses also carry `error`: null normally, or the last `|error|`
// line / stream throw seen while settling (alongside `requestState` and the lines), so
// check its value, not its presence.
//
// `p1`/`p2` in a response are the protocol lines that player received SINCE THE PREVIOUS
// response for that battle (not cumulative), each already stripped of empty lines and
// ready to hand to poke-env's `AbstractBattle.parse_message` after splitting on "|".
//
// Usage:
//   node tools/sim_worker.mjs <path-to-pokemon-showdown-repo>
import crypto from "node:crypto";
import { pathToFileURL } from "node:url";
import path from "node:path";
import fs from "node:fs";
import readline from "node:readline";

const [, , showdownRepo] = process.argv;
if (!showdownRepo) {
	console.error("usage: node sim_worker.mjs <showdown-repo>");
	process.exit(1);
}

const simIndexPath = path.join(showdownRepo, "dist", "sim", "index.js");
if (!fs.existsSync(simIndexPath)) {
	console.error(`${simIndexPath} not found -- run "node build" in the showdown repo first`);
	process.exit(1);
}

const simModule = await import(pathToFileURL(simIndexPath).href);
const Sim = simModule.BattleStream ? simModule : simModule.default;
const { Battle, BattleStream, getPlayerStreams, PRNG, Teams } = Sim;

/** @type {Map<string, {stream: any, streams: any, buffers: {p1: string[], p2: string[]}, transcript: {p1: string[], p2: string[]}, winner: string|null, ended: boolean}>} */
const battles = new Map();

// Yield to the event loop so the async-iterator drains below can run. BattleStream
// processes input synchronously into its output queues, but those queues are delivered
// through async generators, so the buffers are only guaranteed populated after the
// microtask+macrotask queues have flushed.
const tick = () => new Promise((resolve) => setImmediate(resolve));

function attachDrain(entry, side) {
	(async () => {
		for await (const chunk of entry.streams[side]) {
			for (const line of chunk.split("\n")) {
				if (line) {
					entry.buffers[side].push(line);
					entry.transcript[side].push(line);
					entry.lineCount++;
					if (line.startsWith("|error|")) {
						entry.lastError = line.slice("|error|".length) || line;
					}
				}
			}
		}
	})().catch((err) => {
		// Stream torn down by `close` is normal. A throw inside `go()` used to vanish
		// here, so a patched battle's next choose returned no lines and no request.
		if (!entry.lastError && err) {
			entry.lastError = err.stack || err.message || String(err);
		}
	});
}

function attachOmniscient(entry) {
	(async () => {
		for await (const chunk of entry.streams.omniscient) {
			for (const line of chunk.split("\n")) {
				if (line) entry.lineCount++;
				if (line.startsWith("|error|")) {
					entry.lastError = line.slice("|error|".length) || line;
				} else if (line.startsWith("|win|")) {
					entry.winner = line.slice(5).trim();
					entry.ended = true;
					// NB: the draw line is a bare `|tie` (or `|tie|`). Do NOT use
					// startsWith("|tie") -- `|tier|[Gen 9 Champions] VGC 2026 Reg M-B`
					// is emitted at battle start and matches that prefix.
				} else if (line === "|tie" || line.startsWith("|tie|")) {
					entry.winner = null;
					entry.ended = true;
				}
			}
		}
	})().catch((err) => {
		if (!entry.lastError && err) {
			entry.lastError = err.stack || err.message || String(err);
		}
	});
}

function makeEntry(stream, transcript = { p1: [], p2: [] }) {
	const streams = getPlayerStreams(stream);
	const entry = {
		stream,
		streams,
		buffers: { p1: [], p2: [] },
		transcript: { p1: [...transcript.p1], p2: [...transcript.p2] },
		winner: null,
		ended: false,
		lineCount: 0,
		lastError: null,
	};
	attachDrain(entry, "p1");
	attachDrain(entry, "p2");
	attachOmniscient(entry);
	return entry;
}

function normalizeState(state) {
	const normalized = structuredClone(state);
	if (Array.isArray(normalized.log)) {
		normalized.log = normalized.log.map((line) => line.startsWith("|t:|") ? "|t:|" : line);
	}
	return normalized;
}

function stateHash(entry) {
	const battle = entry.stream.battle;
	if (!battle) throw new Error("battle has not started");
	return crypto.createHash("sha256")
		.update(JSON.stringify(normalizeState(battle.toJSON())))
		.digest("hex");
}

// Settle: wait until the sim has (a) actually emitted output for the write we just made,
// and (b) come to rest -- either finished or waiting on input again.
//
// Both halves are load-bearing. `battle.requestState` is '' mid-resolution and
// 'teampreview'/'move'/'switch' once the sim wants a choice, but it holds its PREVIOUS
// value for the first few ticks after a write, so checking it alone returns the stale
// "still waiting for input" state and the caller races ahead of the sim (symptom: the
// next `choose` gets `|error|[Invalid choice] Can't do anything: The game is over`).
// Waiting for new output alone is also not enough, because output arrives in several
// chunks across ticks. So: wait for the line count to move, then for it to go quiet while
// the sim reports itself at rest.
async function settle(entry) {
	const before = entry.lineCount;
	for (let i = 0; i < 2000; i++) {
		await tick();
		if (entry.lineCount > before) break;
	}
	let last = -1;
	let quiet = 0;
	for (let i = 0; i < 2000; i++) {
		await tick();
		const battle = entry.stream.battle;
		const atRest = entry.ended || Boolean(battle && (battle.ended || battle.requestState !== ""));
		if (entry.lineCount === last) quiet++;
		else {
			quiet = 0;
			last = entry.lineCount;
		}
		if (atRest && quiet >= 2) break;
	}
}

function drainBuffers(entry) {
	const out = { p1: entry.buffers.p1, p2: entry.buffers.p2 };
	entry.buffers = { p1: [], p2: [] };
	return out;
}

function packTeam(team) {
	if (typeof team === "string") return team;
	const packed = Teams.pack(team);
	if (!packed) throw new Error("Teams.pack produced an empty team");
	return packed;
}

async function handleStart(msg) {
	if (battles.has(msg.id)) throw new Error(`battle id ${msg.id} already exists`);

	const stream = new BattleStream({ debug: false });
	const entry = makeEntry(stream);
	battles.set(msg.id, entry);

	const startSpec = { formatid: msg.format };
	if (msg.seed) startSpec.seed = msg.seed;

	const lines = [
		`>start ${JSON.stringify(startSpec)}`,
		`>player p1 ${JSON.stringify({ name: msg.p1.name || "p1", team: packTeam(msg.p1.team) })}`,
		`>player p2 ${JSON.stringify({ name: msg.p2.name || "p2", team: packTeam(msg.p2.team) })}`,
	];
	void entry.streams.omniscient.write(lines.join("\n"));
	await settle(entry);
	return respond(msg, entry);
}

// ## Forced Protect-family rolls (exact stall odds)
//
// Showdown's `stall` condition (data/conditions.ts) rolls `randomChance(1, counter)` in
// `onStallMove` every time a Pokemon that still holds the `stall` volatile uses a
// protect-family move (Protect, Detect, Spiky Shield, King's Shield, Baneful Bunker,
// Burning Bulwark, Silk Trap, Obstruct, Endure, Max Guard). Wide Guard / Quick Guard
// add the volatile but never roll. A clone may therefore pin that one roll:
// `{"cmd":"clone",...,"stallForce":{"p1a":true,"p2b":false}}`.
//
// The dex (and so the `stall` condition) is frozen and shared by every battle, so the
// pin is applied per battle: it is stored on the holder's `stall` volatile state
// (stamped with the battle turn, so a pin that was never consumed cannot leak into a
// later turn) and consulted by a wrapper installed on THIS battle's `randomChance`. The
// wrapper only acts on the call made while the stall condition's `StallMove` handler is
// running for the pinned volatile; it always makes the stock draw first, so a forced
// success and a forced failure consume the same PRNG value and share the rest of the
// random stream. Every other `randomChance` call, and any stall roll without a pin,
// behaves exactly as stock (the stock handler still deletes the volatile on failure).
//
// Moves whose `onPrepareHit` runs `runEvent('StallMove')` in data/moves.ts are the only
// ones that roll the stall odds; Wide Guard, Quick Guard and Mat Block add the volatile
// without rolling. tests/test_exact_stall_odds.py checks this list against moves.ts.
const STALL_ROLL_MOVES = new Set([
	'protect', 'detect', 'spikyshield', 'kingsshield', 'banefulbunker', 'burningbulwark',
	'silktrap', 'obstruct', 'endure', 'maxguard',
]);

function patchStallForce(battle) {
	if (Object.prototype.hasOwnProperty.call(battle, 'randomChance')) return;
	const stock = battle.randomChance;
	battle.randomChance = function (numerator, denominator) {
		const drawn = stock.call(this, numerator, denominator);
		const state = this.effectState;
		if (this.effect && this.effect.id === 'stall' && this.event && this.event.id === 'StallMove' &&
			state && state.vgcForce) {
			const force = state.vgcForce;
			delete state.vgcForce;
			if (force.turn === this.turn) return force.success;
		}
		return drawn;
	};
}

function applyStallForce(battle, stallForce) {
	for (const [key, success] of Object.entries(stallForce || {})) {
		const match = /^(p[12])([a-c])$/.exec(key);
		if (!match) throw new Error(`bad stallForce key ${JSON.stringify(key)}`);
		const side = battle.sides[Number(match[1][1]) - 1];
		const pokemon = side && side.active[match[2].charCodeAt(0) - 97];
		const stall = pokemon && pokemon.volatiles && pokemon.volatiles.stall;
		if (!stall) throw new Error(`stallForce ${key}: no stall volatile to force`);
		stall.vgcForce = { success: Boolean(success), turn: battle.turn };
	}
}

// Active Pokemon that currently hold a stall volatile (so their next protect-family use
// is a roll), with the odds denominator and each legal move id in request order, so the
// caller can tell from a choice string whether the roll will happen.
function handleStallInfo(msg) {
	const entry = battles.get(msg.id);
	if (!entry) throw new Error(`unknown battle id ${msg.id}`);
	const battle = entry.stream.battle;
	if (!battle) throw new Error(`battle ${msg.id} has not started`);
	const stallers = [];
	for (const side of battle.sides) {
		side.active.forEach((pokemon, position) => {
			const stall = pokemon && !pokemon.fainted && pokemon.volatiles.stall;
			if (!stall) return;
			const request = pokemon.getMoveRequestData();
			stallers.push({
				side: side.id,
				position: String.fromCharCode(97 + position),
				species: pokemon.species.id,
				counter: stall.counter || 1,
				moves: request.moves.map((move) => move.id),
				disabled: request.moves.map((move) => Boolean(move.disabled)),
				rolls: request.moves.map((move) => STALL_ROLL_MOVES.has(move.id)),
			});
		});
	}
	return { id: msg.id, stallers };
}

async function handleClone(msg) {
	if (battles.has(msg.id)) throw new Error(`battle id ${msg.id} already exists`);
	const source = battles.get(msg.source);
	if (!source) throw new Error(`unknown source battle id ${msg.source}`);
	const sourceBattle = source.stream.battle;
	if (!sourceBattle) throw new Error(`source battle ${msg.source} has not started`);

	const stream = new BattleStream({ debug: false });
	const entry = makeEntry(stream, source.transcript);
	// `toJSON()` is serializable but may still contain live nested object references.
	// Round-trip through JSON so sibling clones cannot share mutable Pokemon/set state.
	const serialized = JSON.parse(JSON.stringify(sourceBattle.toJSON()));
	stream.battle = Battle.fromJSON(serialized);
	// Deserialization leaves `lastMove`/`lastMoveUsed` as unresolved "[DataMove:...]"
	// reference strings. Encore's start handler reads `lastMove.flags`, which crashes
	// on a string (`failencore`), so resolve them back to dex objects. Anything
	// unresolvable keeps its serialized form rather than inventing a move.
	for (const side of stream.battle.sides) {
		for (const pokemon of side.pokemon) {
			for (const field of ['lastMove', 'lastMoveUsed', 'lastMoveEncore']) {
				const value = pokemon[field];
				const match = typeof value === 'string' && value.match(/^\[DataMove:(.+)\]$/);
				if (match) {
					const resolved = stream.battle.dex.moves.get(match[1]);
					if (resolved && resolved.exists !== false) pokemon[field] = resolved;
				}
			}
		}
	}
	stream.battle.restart((type, data) => {
		if (Array.isArray(data)) data = data.join("\n");
		stream.pushMessage(type, data);
		if (type === "end" && !stream.keepAlive) stream.pushEnd();
	});
	if (msg.seed) {
		// Change only FUTURE mechanics randomness. Direct assignment avoids adding a
		// user-visible "RNG was reset" protocol line to an otherwise exact clone.
		stream.battle.prng = new PRNG(msg.seed);
	}
	patchStallForce(stream.battle);
	if (msg.stallForce) applyStallForce(stream.battle, msg.stallForce);
	entry.ended = Boolean(stream.battle.ended);
	entry.winner = entry.ended ? source.winner : null;
	battles.set(msg.id, entry);

	return {
		id: msg.id,
		p1: msg.omitTranscript ? [] : [...source.transcript.p1],
		p2: msg.omitTranscript ? [] : [...source.transcript.p2],
		requestState: stream.battle.requestState,
		ended: entry.ended,
		winner: entry.winner,
		stateHash: stateHash(entry),
	};
}

function handleInspect(msg) {
	const entry = battles.get(msg.id);
	if (!entry) throw new Error(`unknown battle id ${msg.id}`);
	const battle = entry.stream.battle;
	return {
		id: msg.id,
		requestState: battle ? battle.requestState : "",
		ended: entry.ended || Boolean(battle && battle.ended),
		winner: entry.winner,
		stateHash: stateHash(entry),
		prngSeed: battle.prng.getSeed(),
		transcriptLines: {
			p1: entry.transcript.p1.length,
			p2: entry.transcript.p2.length,
		},
		error: entry.lastError || null,
	};
}

function handleDump(msg) {
	const entry = battles.get(msg.id);
	if (!entry) throw new Error(`unknown battle id ${msg.id}`);
	const battle = entry.stream.battle;
	if (!battle) throw new Error(`battle ${msg.id} has not started`);
	return { id: msg.id, state: JSON.parse(JSON.stringify(battle.toJSON())) };
}

function effectElapsed(snapshot, battle) {
	if (!snapshot || !Number.isInteger(snapshot.turns)) return null;
	if (snapshot.counter_kind === 'start_turn' || snapshot.counter_kind === 'side_start_turn') {
		// May be negative: vgc.condition_clock reports a known item-extended condition
		// (8 turns) net of its extension, so base duration - elapsed is still right.
		return Number(battle.turn || 0) - snapshot.turns;
	}
	if (snapshot.counter_kind === 'elapsed_actions') return Math.max(0, snapshot.turns);
	return null;
}

function effectState(id, target, snapshot, battle, hidden = {}) {
	const state = { id, target };
	const effect = battle.dex.conditions.get(id);
	const elapsed = effectElapsed(snapshot, battle);
	if (snapshot && snapshot.counter_kind === 'layers' && Number.isInteger(snapshot.turns)) {
		state.layers = snapshot.turns;
	} else if (Number.isInteger(effect.duration) && elapsed !== null) {
		// poke-env exposes when an effect began (field/side/weather) or how many
		// action opportunities elapsed (Pokemon volatile). Showdown stores the
		// remaining duration, so copying the raw number would reverse its meaning.
		state.duration = Math.max(1, effect.duration - elapsed);
	}
	if (id === 'confusion') {
		// Confusion's 2-5-turn timer is sampled at start and is deliberately hidden.
		// The Python belief builder supplies one legal remaining-time hypothesis.
		state.time = Number.isInteger(hidden.confusionTime) ? hidden.confusionTime : 1;
	}
	if (snapshot && snapshot.raw_value !== null && snapshot.raw_value !== undefined) {
		state.publicValue = snapshot.raw_value;
	}
	return state;
}

function dexBaseId(battle, speciesId) {
	if (!speciesId) return '';
	const entry = battle.dex.species.get(speciesId);
	const base = entry && entry.exists !== false && entry.baseSpecies
		? String(entry.baseSpecies).toLowerCase().replace(/[^a-z0-9]/g, '')
		: '';
	return base || String(speciesId).toLowerCase().replace(/[^a-z0-9]/g, '');
}

function findPokemon(side, snapshot, used) {
	const ids = new Set([snapshot.species_id, snapshot.base_species_id].filter(Boolean));
	const battle = side.battle;
	for (const pokemon of side.pokemon) {
		if (used.has(pokemon)) continue;
		if (ids.has(pokemon.species.id) || ids.has(pokemon.baseSpecies.id)) {
			used.add(pokemon);
			return pokemon;
		}
		if (battle) {
			const want = dexBaseId(battle, snapshot.base_species_id || snapshot.species_id);
			if (
				want &&
				(dexBaseId(battle, pokemon.species.id) === want ||
					dexBaseId(battle, pokemon.baseSpecies.id) === want)
			) {
				used.add(pokemon);
				return pokemon;
			}
		}
	}
	return null;
}

// The snapshot ids types with `to_id`, which turns poke-env's typeless `???` (left after
// Burn Up / Double Shock) into "threequestionmarks". The dex has no entry for that, so it
// used to come back as a fake type named "threequestionmarks" that Showdown then echoed in
// `-start|typechange` lines poke-env cannot parse. Unknown types fail loudly instead.
function showdownTypeName(battle, type) {
	if (type === "threequestionmarks" || type === "???") return "???";
	const info = battle.dex.types.get(type);
	if (!info.exists) throw new Error(`patchPublic: unknown type ${JSON.stringify(type)}`);
	return info.name;
}

// True when `itemId` is a Mega stone THIS Pokemon's base species can use. The snapshot has
// no item for a foe whose item is hidden, so patchPokemon would blank the stone the mirror
// was built around (the belief or prior item), and a foe without its stone can never Mega.
function isMegaStoneFor(battle, pokemon, itemId) {
	if (!itemId) return false;
	const item = battle.dex.items.get(itemId);
	return Boolean(item.exists && item.megaStone && item.megaStone[pokemon.baseSpecies.name]);
}

// Showdown (Champions) shows a foe's HP as `floor(100 * hp / maxhp) || 1` percent. Of the
// absolute HP values that display as `percent`, pick the middle one: it is the unbiased
// guess for how much HP is really left. 100% is exact (only hp === maxhp shows 100), and
// anything alive shows at least 1 HP.
function hpForPublicPercent(percent, maxhp) {
	if (percent <= 0) return 0;
	if (percent >= 100) return maxhp;
	const low = Math.max(1, Math.ceil(percent * maxhp / 100));
	const high = Math.max(low, Math.min(maxhp - 1, Math.ceil((percent + 1) * maxhp / 100) - 1));
	return Math.floor((low + high) / 2);
}

function patchPokemon(battle, pokemon, snapshot, hidden = {}, opts = {}) {
	const species = battle.dex.species.get(snapshot.species_id);
	// The set the battle was BUILT from still names the item the Pokemon started with.
	const originalItem = pokemon.set && pokemon.set.item;
	if (species.exists) {
		// A foe that already Mega Evolved arrives as the Mega species with BASE-forme
		// stats: its numeric stats are never public. Recompute them the way Showdown's
		// setSpecies does, from the new forme's base stats and this Pokemon's own set.
		if (opts.megaStats && species.isMega && pokemon.species.id !== species.id &&
			!snapshot.transformed) {
			const formeStats = battle.spreadModify(species.baseStats, pokemon.set);
			for (const stat of ['atk', 'def', 'spa', 'spd', 'spe']) {
				pokemon.storedStats[stat] = formeStats[stat];
				pokemon.baseStoredStats[stat] = formeStats[stat];
			}
		}
		pokemon.species = species;
		pokemon.types = snapshot.types.length ? snapshot.types.map(
			(type) => showdownTypeName(battle, type)
		) : species.types.slice();
	}
	if (opts.hpScale && snapshot.current_hp !== null && snapshot.max_hp) {
		// A foe's HP is a public PERCENT. Keep the mirror set's calculated max HP and
		// convert, instead of declaring a 186-HP Pokemon to have 100 max HP.
		const fraction = snapshot.current_hp / snapshot.max_hp;
		pokemon.hp = snapshot.max_hp === 100
			? hpForPublicPercent(snapshot.current_hp, pokemon.maxhp)
			: Math.min(pokemon.maxhp, Math.max(fraction > 0 ? 1 : 0, Math.round(fraction * pokemon.maxhp)));
	} else {
		pokemon.hp = snapshot.current_hp === null ? pokemon.hp : snapshot.current_hp;
		pokemon.maxhp = snapshot.max_hp === null ? pokemon.maxhp : snapshot.max_hp;
	}
	pokemon.fainted = Boolean(snapshot.fainted);
	const publicStats = Object.fromEntries(snapshot.stats);
	for (const stat of ['atk', 'def', 'spa', 'spd', 'spe']) {
		if (Number.isInteger(publicStats[stat])) {
			pokemon.storedStats[stat] = publicStats[stat];
			pokemon.baseStoredStats[stat] = publicStats[stat];
		}
	}
	pokemon.speed = pokemon.storedStats.spe;
	pokemon.status = snapshot.status || '';
	pokemon.statusState = { id: pokemon.status, target: pokemon };
	if (opts.restoreState && pokemon.status === 'tox') {
		// Showdown's Toxic damage is stage/16 of max HP and it counts residual ticks since
		// switch-in; poke-env counts the same ticks. Without it Toxic did 0 damage.
		pokemon.statusState.stage = Math.min(15, Math.max(0, Number(snapshot.status_counter || 0)));
	}
	if (pokemon.status === 'slp' || pokemon.status === 'frz') {
		const defaultRemaining = Math.max(1, 3 - Number(snapshot.status_counter || 0));
		const remaining = pokemon.status === 'slp' && Number.isInteger(hidden.sleepTime)
			? hidden.sleepTime : defaultRemaining;
		pokemon.statusState.startTime = 3;
		pokemon.statusState.time = remaining;
	}
	pokemon.boosts = Object.fromEntries(snapshot.boosts);
	// An unrevealed foe item is the mirror's belief, not a public fact: keep it. It is
	// cleared only when the snapshot says the item is known, or is publicly gone
	// (`consumed`).
	const keepGuess = snapshot.item_state === 'unknown' && pokemon.item &&
		(opts.keepHiddenItems || (opts.opponentMega && isMegaStoneFor(battle, pokemon, pokemon.item)));
	if (!keepGuess) {
		pokemon.item = snapshot.item_id || '';
	}
	pokemon.itemState = { id: pokemon.item, target: pokemon };
	pokemon.baseAbility = snapshot.base_ability_id || pokemon.baseAbility;
	pokemon.ability = snapshot.temporary_ability_id || snapshot.ability_id || pokemon.ability;
	pokemon.abilityState = { id: pokemon.ability, target: pokemon };
	pokemon.volatiles = {};
	for (const effect of snapshot.effects) {
		pokemon.volatiles[effect.id] = effectState(effect.id, pokemon, effect, battle, hidden);
		if (effect.id === 'substitute' && pokemon.volatiles[effect.id].hp === undefined) {
			pokemon.volatiles[effect.id].hp = Math.max(1, Math.floor(pokemon.maxhp / 4));
		}
	}
	if (opts.restoreState) {
		// Unburden has no `-start` line, so poke-env never lists its volatile: a mon that
		// lost its item (consumed, knocked off) came back without the doubled Speed.
		// Own side: the packed team gave it an item and the request now shows none.
		// Foe: the item is publicly known to be gone.
		const itemGone = snapshot.item_state === 'consumed' ||
			(snapshot.item_state === 'none' && Boolean(originalItem));
		if (pokemon.ability === 'unburden' && !pokemon.item && itemGone &&
			!pokemon.volatiles.unburden) {
			pokemon.volatiles.unburden = { id: 'unburden', target: pokemon };
		}
		// Disable's target move is only in the `-start` line; poke-env drops it. The
		// caller recovers it from the protocol history. Without it the volatile is inert.
		const disabledMove = hidden.disabledMove;
		if (pokemon.volatiles.disable && disabledMove && !pokemon.volatiles.disable.move) {
			pokemon.volatiles.disable.move = disabledMove;
		}
	}
	if (snapshot.must_recharge && !pokemon.volatiles.mustrecharge) {
		pokemon.volatiles.mustrecharge = { id: 'mustrecharge', target: pokemon };
	}
	if (snapshot.preparing && snapshot.preparing_move_id && !pokemon.volatiles.twoturnmove) {
		pokemon.volatiles.twoturnmove = {
			id: 'twoturnmove',
			target: pokemon,
			move: snapshot.preparing_move_id,
		};
	}
	if (snapshot.protect_counter > 0) {
		// poke-env counts consecutive successful Protects; Showdown's stall condition stores
		// the odds denominator instead (3 after one Protect, x3 per repeat, capped at 729).
		const stall = battle.dex.conditions.get('stall');
		pokemon.volatiles.stall = {
			id: 'stall', target: pokemon, duration: 2,
			counter: Math.min(stall.counterMax || 729, 3 ** snapshot.protect_counter),
		};
	}
	pokemon.trapped = false;
	pokemon.maybeTrapped = false;
	pokemon.transformed = Boolean(snapshot.transformed);
	pokemon.activeTurns = snapshot.first_turn ? 0 : Math.max(1, pokemon.activeTurns || 1);
	if (opts.restoreState) {
		// Champions' Fake Out / First Impression are disabled once `activeMoveActions` is
		// non-zero. A rebuilt Pokemon starts at 0, so a foe could Fake Out again on every
		// turn of every branch. poke-env's `first_turn` is exactly "has not acted since
		// switching in": any later turn means at least one move action.
		pokemon.activeMoveActions = snapshot.first_turn ? 0 : Math.max(1, pokemon.activeMoveActions || 1);
	}
	pokemon.weighthg = snapshot.weight === null ? pokemon.weighthg : Math.round(snapshot.weight * 10);
	pokemon.lastMove = snapshot.last_move_id ? battle.dex.moves.get(snapshot.last_move_id) : null;
	pokemon.lastMoveUsed = pokemon.lastMove;
	if (snapshot.terastallized && snapshot.tera_type) {
		pokemon.terastallized = battle.dex.types.get(snapshot.tera_type).name;
	}
	pokemon.moveSlots = pokemon.moveSlots.map((slot) => {
		const publicMove = snapshot.moves.find((move) => move.id === slot.id);
		if (!publicMove) return slot;
		return {
			...slot,
			pp: publicMove.current_pp === null ? slot.pp : publicMove.current_pp,
			maxpp: publicMove.max_pp === null ? slot.maxpp : publicMove.max_pp,
			disabled: Boolean(publicMove.disabled),
			disabledSource: publicMove.disabled_reason || '',
		};
	});
	if (opts.restoreState && snapshot.item_state !== 'unknown' && !snapshot.first_turn &&
		snapshot.last_move_id && pokemon.item && !pokemon.volatiles.choicelock) {
		// Choice lock: a PUBLICLY KNOWN Choice item and a move used since switching in. The
		// lock volatile stores the locked move; the DisableMove pass in handlePatchPublic
		// then disables every other move exactly as Showdown's own turn start does.
		const item = battle.dex.items.get(pokemon.item);
		const locked = pokemon.moveSlots.find((slot) => slot.id === snapshot.last_move_id);
		if (item.isChoice && locked) {
			pokemon.volatiles.choicelock = {
				id: 'choicelock', target: pokemon, move: locked.id, sourceEffect: item,
			};
		}
	}
	if (pokemon.volatiles.encore && !pokemon.volatiles.encore.move) {
		// A patched Encore arrives without the engine's `move` field (the snapshot
		// carries durations, not locked-move ids). Branching with it crashes the
		// engine when it reads the undefined move's flags. Encore always leaves
		// exactly the locked move enabled, so derive it; fall back to lastMove.
		const enabled = pokemon.moveSlots.filter((slot) => !slot.disabled);
		const locked = enabled.length === 1
			? enabled[0]
			: pokemon.moveSlots.find((slot) => pokemon.lastMove && slot.id === pokemon.lastMove.id);
		if (locked) pokemon.volatiles.encore.move = locked.id;
	}
}

function patchSide(battle, side, snapshot, hiddenBySpecies = {}, opts = {}) {
	const opponentMega = Boolean(opts.opponentMega);
	const used = new Set();
	const bySpecies = new Map();
	for (const pokemonSnapshot of snapshot.pokemon) {
		const pokemon = findPokemon(side, pokemonSnapshot, used);
		// A player's parser can retain all six preview Pokemon while Showdown's chosen
		// side contains only the brought four. Unmatched, inactive preview-only entries
		// are knowledge, not members of this concrete bring hypothesis.
		if (!pokemon) continue;
		patchPokemon(
			battle,
			pokemon,
			pokemonSnapshot,
			hiddenBySpecies[pokemonSnapshot.species_id] || {},
			opts,
		);
		bySpecies.set(pokemonSnapshot.species_id, pokemon);
		if (pokemonSnapshot.base_species_id) {
			bySpecies.set(pokemonSnapshot.base_species_id, pokemon);
		}
	}
	for (const pokemon of side.pokemon) pokemon.isActive = false;
	// poke-env replaces a fainted active with `null` while it waits for that slot's
	// forced switch. Showdown must keep the fainted Pokemon in the active slot so its
	// switch request is not mistaken for an already-completed choice. The per-Pokemon
	// public snapshot retains exactly which fainted Pokemon was active.
	const forcedSlotSpecies = snapshot.active_species.slice();
	const usedForcedSpecies = new Set(forcedSlotSpecies.filter(Boolean));
	for (let index = 0; index < forcedSlotSpecies.length; index++) {
		if (forcedSlotSpecies[index]) continue;
		const faintedActive = snapshot.pokemon.find(
			(candidate) => candidate.active && candidate.fainted &&
				!usedForcedSpecies.has(candidate.species_id)
		);
		if (!faintedActive && snapshot.force_switch[index]) {
			throw new Error(`cannot identify the fainted active for forced slot ${index}`);
		}
		if (!faintedActive) continue;
		forcedSlotSpecies[index] = faintedActive.species_id;
		usedForcedSpecies.add(faintedActive.species_id);
	}
	side.active = forcedSlotSpecies.map((speciesId) => {
		if (!speciesId) return null;
		const pokemon = bySpecies.get(speciesId) || side.pokemon.find(
			(candidate) => candidate.species.id === speciesId || candidate.baseSpecies.id === speciesId
		);
		if (!pokemon) {
			throw new Error(`cannot activate ${speciesId} on ${side.id}`);
		}
		pokemon.isActive = true;
		return pokemon;
	});
	const activePokemon = side.active.filter(Boolean);
	side.pokemon = [
		...activePokemon,
		...side.pokemon.filter((pokemon) => !activePokemon.includes(pokemon)),
	];
	for (let index = 0; index < side.pokemon.length; index++) {
		side.pokemon[index].position = index;
	}
	side.slotConditions = side.active.map(() => ({}));
	// A null `can_mega_evolve` slot means the Mega ability is not observable (the foe's
	// request is private). Showdown's constructor computed it from the item the Pokemon
	// was BUILT with, and the patches above may have changed item or species since, so ask
	// the engine again for every Pokemon on this side. A side that has spent its Mega gets
	// null everywhere below.
	if (opponentMega && !snapshot.used_mega_evolution &&
		snapshot.can_mega_evolve.some((value) => value === null || value === undefined)) {
		for (const pokemon of side.pokemon) {
			pokemon.canMegaEvo = battle.actions.canMegaEvo(pokemon);
		}
	}
	for (let index = 0; index < side.active.length; index++) {
		if (!side.active[index]) continue;
		side.active[index].switchFlag = snapshot.force_switch[index] ? true : false;
		side.active[index].forceSwitchFlag = snapshot.force_switch[index] ? true : false;
		side.active[index].trapped = Boolean(snapshot.trapped[index]);
		side.active[index].maybeTrapped = Boolean(snapshot.maybe_trapped[index]);
		if (snapshot.can_mega_evolve[index] === false) {
			side.active[index].canMegaEvo = null;
		}
	}
	side.pokemonLeft = side.pokemon.filter((pokemon) => pokemon.hp > 0 && !pokemon.fainted).length;
	side.sideConditions = {};
	for (const effect of snapshot.side_conditions) {
		side.sideConditions[effect.id] = effectState(effect.id, side, effect, battle);
	}
	side.megaEvoUsed = Boolean(snapshot.used_mega_evolution);
	if (snapshot.used_mega_evolution) {
		for (const pokemon of side.pokemon) pokemon.canMegaEvo = null;
	}
	side.zMoveUsed = Boolean(snapshot.used_z_move);
	side.dynamaxUsed = Boolean(snapshot.used_dynamax);
	side.terastallizeUsed = Boolean(snapshot.used_tera);
}

// Showdown's addVolatile records who caused each volatile (`source`/`sourceSlot`), and
// some conditions dereference it. Imprison reads the holder's moves through it (a missing
// source threw "reading 'hasMove'" and killed the branch); Leech Seed heals into
// `sourceSlot` (missing -> the drain silently never happened). Self-applied volatiles get
// their holder, which is exact. The public snapshot does not say WHICH foe inflicted a
// volatile, so foe-inflicted ones stay sourceless (Showdown guards those reads), except
// Leech Seed, which gets the first active foe's slot: the drain is right, only which foe
// it heals may be wrong. Guessing a source for Attract/Octolock/Syrup Bomb/Lock-On would
// make them end (or aim) on the guessed foe's switch instead of the real one's.
const FOE_TARGETS = new Set([
	'normal', 'any', 'adjacentFoe', 'allAdjacentFoes', 'allAdjacent', 'randomNormal',
]);

function fillVolatileSources(battle) {
	for (const side of battle.sides) {
		const foes = side.foe.active.filter(Boolean);
		const foe = foes.find((pokemon) => !pokemon.fainted) || foes[0] || null;
		for (const pokemon of side.pokemon) {
			for (const [id, state] of Object.entries(pokemon.volatiles)) {
				if (state.source) continue;
				const move = battle.dex.moves.get(id);
				if (!(move.exists && FOE_TARGETS.has(move.target))) {
					state.source = pokemon;
					if (pokemon.isActive) state.sourceSlot = pokemon.getSlot();
				} else if (id === 'leechseed' && foe) {
					state.sourceSlot = foe.getSlot();
				}
			}
		}
	}
}

async function handlePatchPublic(msg) {
	const entry = battles.get(msg.id);
	if (!entry) throw new Error(`unknown battle id ${msg.id}`);
	const battle = entry.stream.battle;
	if (!battle) throw new Error(`battle ${msg.id} has not started`);
	const state = msg.state;
	if (!state || !state.our_side || !state.opponent_side) {
		throw new Error('patchPublic requires a complete public mechanics state');
	}
	battle.turn = Number(state.turn || 0);
	const perspective = msg.perspective || 'p1';
	const ownIndex = perspective === 'p1' ? 0 : 1;
	const hidden = msg.hidden || {};
	const options = msg.options || {};
	// Both sides: state the public snapshot carries but the parser's rebuild used to drop.
	const shared = {
		megaStats: Boolean(options.megaStats),
		restoreState: Boolean(options.restoreState),
	};
	patchSide(battle, battle.sides[ownIndex], state.our_side, hidden.our || {}, shared);
	patchSide(battle, battle.sides[1 - ownIndex], state.opponent_side, hidden.opponent || {}, {
		...shared,
		opponentMega: Boolean(msg.opponentMega),
		hpScale: Boolean(options.hpScale),
		keepHiddenItems: Boolean(options.keepHiddenItems),
	});
	fillVolatileSources(battle);
	if (options.restoreState) {
		// Showdown computes which moves are unavailable (Choice lock, Disable, Fake Out
		// after the first turn, ...) at turn start. The patch restored the state those
		// rules read, so run the same DisableMove pass. It only ADDS disabled moves: the
		// flags our own request already reported are kept.
		for (const side of battle.sides) {
			for (const pokemon of side.active) {
				if (!pokemon || pokemon.fainted || !pokemon.hp) continue;
				battle.runEvent('DisableMove', pokemon);
				for (const moveSlot of pokemon.moveSlots) {
					const activeMove = battle.dex.getActiveMove(moveSlot.id);
					battle.singleEvent('DisableMove', activeMove, null, pokemon);
					if (activeMove.flags['cantusetwice'] && pokemon.lastMove &&
						pokemon.lastMove.id === moveSlot.id) {
						pokemon.disableMove(pokemon.lastMove.id);
					}
				}
			}
		}
	}
	battle.field.weather = state.weather.length ? state.weather[0].id : '';
	battle.field.weatherState = effectState(
		battle.field.weather,
		battle.field,
		state.weather.length ? state.weather[0] : null,
		battle,
	);
	battle.field.terrain = '';
	battle.field.terrainState = { id: '', target: battle.field };
	battle.field.pseudoWeather = {};
	for (const effect of state.fields) {
		if (effect.id.endsWith('terrain')) {
			battle.field.terrain = effect.id;
			battle.field.terrainState = effectState(effect.id, battle.field, effect, battle);
		} else {
			battle.field.pseudoWeather[effect.id] = effectState(
				effect.id, battle.field, effect, battle
			);
		}
	}
	battle.midTurn = false;
	// `queue` is a BattleQueue, not an array. Replacing it with `[]` used to make the
	// next `go()` throw inside the stream (swallowed by the drains), so every choose on
	// a patched battle produced no output and no new request -- the branch never moved.
	battle.queue.clear();
	battle.faintQueue = [];
	battle.clearRequest();
	const forcedSwitch = [state.our_side, state.opponent_side].some(
		(side) => side.force_switch.some(Boolean)
	);
	battle.makeRequest(forcedSwitch ? 'switch' : 'move');
	for (const side of battle.sides) side.emitRequest();
	battle.sentRequests = true;
	for (let index = 0; index < 3; index++) await tick();
	return respond(msg, entry);
}

async function handleChoose(msg) {
	const entry = battles.get(msg.id);
	if (!entry) throw new Error(`unknown battle id ${msg.id}`);
	if (entry.ended) throw new Error(`battle ${msg.id} has already ended`);

	const lines = [];
	if (msg.p1 != null) lines.push(`>p1 ${msg.p1}`);
	if (msg.p2 != null) lines.push(`>p2 ${msg.p2}`);
	if (!lines.length) throw new Error("choose requires at least one of p1/p2");

	void entry.streams.omniscient.write(lines.join("\n"));
	await settle(entry);
	return respond(msg, entry);
}

function respond(msg, entry) {
	const battle = entry.stream.battle;
	const buffers = drainBuffers(entry);
	const error = entry.lastError || null;
	entry.lastError = null;
	return {
		id: msg.id,
		p1: buffers.p1,
		p2: buffers.p2,
		requestState: battle ? battle.requestState : "",
		ended: entry.ended || Boolean(battle && battle.ended),
		winner: entry.winner,
		error,
	};
}

async function handleClose(msg) {
	const entry = battles.get(msg.id);
	if (entry) {
		battles.delete(msg.id);
		try {
			await entry.stream.destroy();
		} catch {
			// Already torn down.
		}
	}
	return { id: msg.id, closed: true };
}

// Advance many INDEPENDENT battles in one round trip.
//
// This is where the throughput is, and the reason is latency, not JSON size: `settle`
// (above) spends many event-loop ticks per step waiting for the sim to come to rest, and
// serially that cost is paid once per battle per decision. Promise.all lets every
// battle's settle overlap on the shared event loop, so N battles cost roughly one
// settle instead of N.
//
// Safety rests on each entry being self-contained -- its own stream, buffers, and
// lineCount -- so concurrent settles cannot observe or drain each other's state. Two
// commands for the SAME battle in one batch would break that, hence the duplicate check.
async function handleBatch(msg) {
	const items = Array.isArray(msg.items) ? msg.items : [];
	const seen = new Set();
	for (const item of items) {
		if (item && item.id !== undefined) {
			if (seen.has(item.id)) {
				throw new Error(`batch contains two commands for battle ${item.id}`);
			}
			seen.add(item.id);
		}
	}
	const results = await Promise.all(
		items.map(async (item) => {
			try {
				return await dispatch(item);
			} catch (err) {
				// One bad battle must not fail the whole batch -- the caller matches
				// results positionally and can handle this entry alone.
				return { id: item && item.id, error: err.message };
			}
		}),
	);
	return { results };
}

async function dispatch(msg) {
	switch (msg.cmd) {
		case "start":
			return handleStart(msg);
		case "clone":
			return handleClone(msg);
		case "choose":
			return handleChoose(msg);
		case "inspect":
			return handleInspect(msg);
		case "dump":
			return handleDump(msg);
		case "stallInfo":
			return handleStallInfo(msg);
		case "patchPublic":
			return handlePatchPublic(msg);
		case "close":
			return handleClose(msg);
		case "batch":
			return handleBatch(msg);
		case "ping":
			return { pong: true, battles: battles.size };
		default:
			throw new Error(`unknown cmd ${JSON.stringify(msg.cmd)}`);
	}
}

const rl = readline.createInterface({ input: process.stdin, terminal: false });

// Commands are serialized: each battle's `settle` awaits the event loop, and interleaving
// two commands for the SAME battle would let one drain the other's buffers. A single
// queue keeps ordering trivially correct; throughput comes from running one worker
// process per core, not from concurrency inside one process (the sim is CPU-bound).
let queue = Promise.resolve();

rl.on("line", (line) => {
	if (!line.trim()) return;
	queue = queue.then(async () => {
		let msg;
		try {
			msg = JSON.parse(line);
		} catch (err) {
			process.stdout.write(`${JSON.stringify({ error: `bad json: ${err.message}` })}\n`);
			return;
		}
		try {
			const result = await dispatch(msg);
			if (msg.rid !== undefined) result.rid = msg.rid;
			process.stdout.write(`${JSON.stringify(result)}\n`);
		} catch (err) {
			const payload = { id: msg.id, error: err.stack || err.message };
			if (msg.rid !== undefined) payload.rid = msg.rid;
			process.stdout.write(`${JSON.stringify(payload)}\n`);
		}
	});
});

rl.on("close", () => {
	queue.then(() => process.exit(0));
});
