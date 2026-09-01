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
				}
			}
		}
	})().catch(() => {
		// Stream torn down by `close` -- nothing to report, the battle is gone.
	});
}

function attachOmniscient(entry) {
	(async () => {
		for await (const chunk of entry.streams.omniscient) {
			for (const line of chunk.split("\n")) {
				if (line) entry.lineCount++;
				if (line.startsWith("|win|")) {
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
	})().catch(() => {});
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
		return Math.max(0, Number(battle.turn || 0) - snapshot.turns);
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

function findPokemon(side, snapshot, used) {
	const ids = new Set([snapshot.species_id, snapshot.base_species_id].filter(Boolean));
	for (const pokemon of side.pokemon) {
		if (used.has(pokemon)) continue;
		if (ids.has(pokemon.species.id) || ids.has(pokemon.baseSpecies.id)) {
			used.add(pokemon);
			return pokemon;
		}
	}
	return null;
}

function patchPokemon(battle, pokemon, snapshot, hidden = {}) {
	const species = battle.dex.species.get(snapshot.species_id);
	if (species.exists) {
		pokemon.species = species;
		pokemon.types = snapshot.types.length ? snapshot.types.map(
			(type) => battle.dex.types.get(type).name
		) : species.types.slice();
	}
	pokemon.hp = snapshot.current_hp === null ? pokemon.hp : snapshot.current_hp;
	pokemon.maxhp = snapshot.max_hp === null ? pokemon.maxhp : snapshot.max_hp;
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
	if (pokemon.status === 'slp' || pokemon.status === 'frz') {
		const defaultRemaining = Math.max(1, 3 - Number(snapshot.status_counter || 0));
		const remaining = pokemon.status === 'slp' && Number.isInteger(hidden.sleepTime)
			? hidden.sleepTime : defaultRemaining;
		pokemon.statusState.startTime = 3;
		pokemon.statusState.time = remaining;
	}
	pokemon.boosts = Object.fromEntries(snapshot.boosts);
	pokemon.item = snapshot.item_id || '';
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
		pokemon.volatiles.stall = {
			id: 'stall', target: pokemon, counter: snapshot.protect_counter,
		};
	}
	pokemon.trapped = false;
	pokemon.maybeTrapped = false;
	pokemon.transformed = Boolean(snapshot.transformed);
	pokemon.activeTurns = snapshot.first_turn ? 0 : Math.max(1, pokemon.activeTurns || 1);
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
}

function patchSide(battle, side, snapshot, hiddenBySpecies = {}) {
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
		if (!pokemon) throw new Error(`cannot activate ${speciesId} on ${side.id}`);
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
	patchSide(battle, battle.sides[ownIndex], state.our_side, hidden.our || {});
	patchSide(battle, battle.sides[1 - ownIndex], state.opponent_side, hidden.opponent || {});
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
	battle.queue = [];
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
	return {
		id: msg.id,
		p1: buffers.p1,
		p2: buffers.p2,
		requestState: battle ? battle.requestState : "",
		ended: entry.ended || Boolean(battle && battle.ended),
		winner: entry.winner,
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
