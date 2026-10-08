// Plan-value probe: how much damage does ONE of our Pokemon sets deal over two consecutive
// turns under each weather x terrain condition, measured by the real Showdown engine.
//
// Why this exists: weather/terrain are worth what they ENABLE (rain lets a charge move
// fire in one turn, makes Thunder/Hurricane 100% accurate, Psychic Terrain makes Expanding
// Force hit both foes, Terrain Pulse / Weather Ball change type...). None of that is in the
// exported move data; it lives in Showdown callbacks. So we do not model it, we run it.
// The Python side (`vgc.plan_value`, `tools/build_plan_value_cache.py`) turns the numbers
// into a committed cache that `vgc.field_control` reads.
//
// Protocol (long-lived, JSON lines on stdin/stdout, one request per line):
//   -> {"rid":1,"packed":"<ONE packed set>","mega":true,"moves":["archaludon move ids..."],
//       "conditions":[["none","none"],["raindance","psychicterrain"],...],
//       "seeds":[1,2,3,4]}
//   <- {"rid":1,"results":{"<moveid>":[[[turn1,turn2] per profile] per condition]},...}
//      i.e. results[move][conditionIndex][profileIndex] = mean over seeds of the
//      two-turn TOTAL, in % of the reference target's max HP (summed over foes).
//   -> {"cmd":"ping"}  <- {"pong":true}
//
// Profiles (foe sides; every foe is a '???'-typed -- type-neutral -- Incineroar-stat body so
// the number is not a type-chart accident):
//   0  one grounded foe
//   1  two grounded foes (spread moves and Expanding Force hit both; the engine applies
//      the spread reduction itself; % is summed over both foes)
//   2  one airborne foe (Levitate): captures grounded-only terrain effects
//
// How the measurement is held steady (all of this is engine state, not a model):
//  * Weather/terrain are applied with the engine's own `field.setWeather` / `setTerrain`
//    AFTER the switch-in, with our Pokemon as the source. Any weather/terrain a switch-in
//    ability already created (Drought, Psychic Surge...) is cleared first so the 'none'
//    baseline is truly bare, and so a setter is measured like everyone else. Because the
//    set goes through the real code path, abilities/items that react to a change
//    (Protosynthesis, Psychic Seed, Surge-based Seeds) fire. Duration is the engine's own
//    5 turns, which covers both measured turns.
//  * Damage is read from the protocol's `|-damage|` lines on the foes that carry no
//    `[from]` tag: weather chip, Leech Seed etc. are excluded, recoil is on our side.
//  * Randomness we do not want to measure is removed rather than sampled: the damage roll
//    is the exact mean (92.5%), crits cannot happen (Lucky Chant on the foes' side), every
//    accuracy check is forced to pass but weighted by the accuracy the engine computed
//    (so Thunder 70% -> 100% in rain is exact), and 2-5-hit moves are run at 5 hits and
//    weighted by the engine's own hit-count distribution. Whatever randomness is left
//    (secondary effects, Magnitude) is averaged over `seeds`.
//  * Foe HP is restored between the two turns so a big first hit cannot KO the reference.
//  * The foes use Tackle each turn so conditional moves (Sucker Punch) are not unfairly
//    zeroed; our side is the single Pokemon under test, and it Mega Evolves on turn 1 if
//    it holds its stone and the request says `"mega":true` (default; `false` = base form).
//
// Usage: node tools/plan_value_probe.mjs <path-to-pokemon-showdown-repo>
import { pathToFileURL } from "node:url";
import path from "node:path";
import fs from "node:fs";
import readline from "node:readline";

const [, , showdownRepo] = process.argv;
if (!showdownRepo) {
	console.error("usage: node plan_value_probe.mjs <showdown-repo>");
	process.exit(1);
}
const simIndexPath = path.join(showdownRepo, "dist", "sim", "index.js");
if (!fs.existsSync(simIndexPath)) {
	console.error(`${simIndexPath} not found -- run "node build" in the showdown repo first`);
	process.exit(1);
}
const simModule = await import(pathToFileURL(simIndexPath).href);
const Sim = simModule.Battle ? simModule : simModule.default;
const { Battle, Teams } = Sim;

const FORMAT = "gen9championsdoublescustomgame";
const SPREAD_ROLL = 92.5; // mean of Showdown's 85..100 damage roll
const FIVE_HIT_SURVIVAL = [1, 1, 13 / 20, 6 / 20, 3 / 20]; // P(hit k lands) for 2-5-hit moves
const TURNS = 2;
// A big hit must not be clipped by the reference foe's HP (a sun Eruption that would "KO" is
// worth more than one that exactly KOs), so the foes carry HP_SCALE times their real HP and
// damage is still reported as % of the REAL max HP. The few moves whose damage is a fraction
// of the foe's current HP (or compares HP) keep the real HP so their % stays honest.
const HP_SCALE = 10;
const ALLY_TARGETS = new Set(["adjacentAlly", "adjacentAllyOrSelf"]);
const REAL_HP_MOVES = new Set([
	"superfang", "naturesmadness", "ruination", "guardianofalola", "endeavor", "finalgambit",
]);

// Doubles needs two actives per side. Where a profile has one real Pokemon the other slot is
// this dummy, fainted right after switch-in -- exactly what a side with one Pokemon left
// looks like in a real game (the slot stays, fainted, and is skipped).
const DUMMY_SET = {
	species: "Incineroar",
	ability: "noability",
	item: "",
	moves: ["splash"],
	nature: "Serious",
	evs: { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 },
	level: 50,
};
const FOE_SET = {
	species: "Incineroar",
	ability: "noability",
	item: "",
	moves: ["tackle"],
	nature: "Serious",
	evs: { hp: 32, atk: 0, def: 16, spa: 0, spd: 16, spe: 0 },
	level: 50,
};
const FOE_AIRBORNE_SET = { ...FOE_SET, ability: "levitate" };
const PROFILES = [
	{ foes: [FOE_SET] },
	{ foes: [FOE_SET, FOE_SET] },
	{ foes: [FOE_AIRBORNE_SET] },
];

const WEATHER_SOURCE = {
	sunnyday: "drought",
	raindance: "drizzle",
	sandstorm: "sandstream",
	snowscape: "snowwarning",
};
const TERRAIN_SOURCE = {
	electricterrain: "electricsurge",
	grassyterrain: "grassysurge",
	psychicterrain: "psychicsurge",
	mistyterrain: "mistysurge",
};

const HP_LINE = /^\|-damage\|(p2[ab]): [^|]*\|(?:(\d+)\/(\d+)(?: [a-z]+)?|0 fnt)(?:\||$)/;
const HEAL_LINE = /^\|-heal\|(p2[ab]): [^|]*\|(\d+)\/(\d+)/;

function instrument(battle, record) {
	// Accuracy: record what the engine computed, then let the check pass.
	const originalRunEvent = battle.runEvent.bind(battle);
	let pending = null;
	battle.runEvent = (eventid, target, source, effect, relayVar, ...rest) => {
		const result = originalRunEvent(eventid, target, source, effect, relayVar, ...rest);
		if (eventid === "Accuracy" && target && typeof target.getSlot === "function") {
			const slot = target.getSlot();
			if (result === true || result === undefined) {
				record.accuracy(slot, 1);
				pending = null;
			} else if (typeof result === "number") {
				pending = { slot, value: result };
			}
		}
		return result;
	};
	const originalRandomChance = battle.randomChance.bind(battle);
	battle.randomChance = (numerator, denominator) => {
		if (pending && denominator === 100 && numerator === pending.value) {
			const { slot, value } = pending;
			pending = null;
			record.accuracy(slot, Math.max(0, Math.min(1, value / 100)));
			return true;
		}
		pending = null;
		// Everything else that rolls a chance (crits, secondary effects, full paralysis,
		// flinches...) never fires: this probe measures what the field enables, not luck.
		return false;
	};
	void originalRandomChance;
	// Secondary-effect rolls are battle.random(100) < chance; 99 keeps every <100% chance off.
	const originalRandom = battle.random.bind(battle);
	battle.random = (m, n) => (n === undefined && m === 100 ? 99 : originalRandom(m, n));
	// Exact mean damage roll.
	battle.randomizer = (baseDamage) => battle.trunc(battle.trunc(baseDamage * SPREAD_ROLL) / 100);
	// 2-5-hit moves: run the maximum and weight by the engine's own distribution.
	const originalSample = battle.sample.bind(battle);
	battle.sample = (items) => {
		if (Array.isArray(items) && items.length === 20 && items.every((n) => n >= 2 && n <= 5)) {
			record.fiveHit = true;
			return 5;
		}
		return originalSample(items);
	};
}

function newRecorder() {
	return {
		fiveHit: false,
		accuracyBySlot: {},
		accuracy(slot, p) {
			(this.accuracyBySlot[slot] ||= []).push(p);
		},
		reset() {
			this.fiveHit = false;
			this.accuracyBySlot = {};
		},
	};
}

function parseLog(lines, from, state, recorder) {
	// Walk new log lines, returning expected damage dealt to the foes this turn, in raw HP.
	const hitsBySlot = {};
	for (let i = from; i < lines.length; i++) {
		const line = lines[i];
		const damage = HP_LINE.exec(line);
		if (damage) {
			const slot = damage[1];
			const hp = damage[2] === undefined ? 0 : Number(damage[2]);
			const previous = state[slot] ?? Number(damage[3]);
			state[slot] = hp;
			if (!line.includes("|[from]")) {
				(hitsBySlot[slot] ||= []).push(Math.max(0, previous - hp));
			}
			continue;
		}
		const heal = HEAL_LINE.exec(line);
		if (heal) state[heal[1]] = Number(heal[2]);
	}
	let total = 0;
	for (const [slot, hits] of Object.entries(hitsBySlot)) {
		const accuracy = recorder.accuracyBySlot[slot] || [];
		let cumulative = 1;
		hits.forEach((dealt, index) => {
			if (index < accuracy.length) cumulative *= accuracy[index];
			let weight = cumulative;
			if (recorder.fiveHit && hits.length > 1) {
				weight *= FIVE_HIT_SURVIVAL[Math.min(index, 4)];
			}
			total += dealt * weight;
		});
	}
	return total;
}

function applyCondition(battle, ours, weather, terrain) {
	battle.field.clearWeather();
	battle.field.clearTerrain();
	if (weather !== "none") {
		const source = battle.dex.abilities.get(WEATHER_SOURCE[weather]);
		battle.field.setWeather(weather, ours, source);
	}
	if (terrain !== "none") {
		const source = battle.dex.abilities.get(TERRAIN_SOURCE[terrain]);
		battle.field.setTerrain(terrain, ours, source);
	}
}

function measure(setPacked, moveIndex, condition, profile, seed, mega) {
	const [weather, terrain] = condition;
	// The real format forces level 50 (Adjust Level); a packed set with no level would be 100.
	const ourTeam = [...Teams.unpack(setPacked).map((set) => ({ ...set, level: 50 })), { ...DUMMY_SET }];
	const foeTeam = profile.foes.map((foe) => ({ ...foe }));
	while (foeTeam.length < 2) foeTeam.push({ ...DUMMY_SET });
	const battle = new Battle({
		formatid: FORMAT,
		seed: [seed, seed + 1, seed + 2, seed + 3],
		p1: { name: "p1", team: Teams.pack(ourTeam) },
		p2: { name: "p2", team: Teams.pack(foeTeam) },
	});
	const recorder = newRecorder();
	instrument(battle, recorder);
	battle.makeChoices("team 12", "team 12");
	const ours = battle.sides[0].pokemon[0];
	const foes = battle.sides[1].pokemon.slice(0, profile.foes.length);
	for (const side of battle.sides) {
		const dummy = side.pokemon[1];
		if (side === battle.sides[0] || profile.foes.length === 1) {
			dummy.faint();
		}
	}
	battle.faintMessages();
	battle.clearRequest();
	battle.makeRequest("move");
	const scale = REAL_HP_MOVES.has(ours.moveSlots[moveIndex]?.id) ? 1 : HP_SCALE;
	for (const foe of foes) {
		foe.types = ["???"];
		foe.baseMaxhp = foe.maxhp * scale;
		foe.maxhp = foe.baseMaxhp;
		foe.hp = foe.maxhp;
	}
	battle.sides[1].addSideCondition("luckychant", foes[0]);
	// The dummy ally's faint must not count (Last Respects, Retaliate read these).
	battle.sides[0].totalFainted = 0;
	battle.sides[0].faintedThisTurn = null;
	battle.sides[0].faintedLastTurn = null;
	// Mega Evolve BEFORE the condition is applied: a Mega whose new ability sets weather
	// (Charizard-Y's Drought) must not leak that weather into the "bare field" baseline.
	if (mega && ours.canMegaEvo) battle.actions.runMegaEvo(ours);
	applyCondition(battle, ours, weather, terrain);

	const foeState = {};
	foes.forEach((foe, index) => {
		foeState[`p2${"ab"[index]}`] = foe.maxhp;
	});
	const referenceMax = foes[0].maxhp / scale;
	const turns = [];
	for (let turn = 0; turn < TURNS; turn++) {
		const logStart = battle.log.length;
		recorder.reset();
		const foeChoice = foes.map(() => "move 1 1").join(", ");
		// Asked fresh each turn: Expanding Force turns into a spread move in Psychic Terrain,
		// and a charge/recharge/lock-in turn offers one move and refuses a target.
		const request = ours.getMoveRequestData();
		const requested = request.moves || [];
		const locked = requested.length === 1 && request.trapped;
		// A move the engine will not let us repeat (e.g. Blood Moon) deals nothing that turn.
		const unavailable = !locked && requested[moveIndex]?.disabled;
		let slot = moveIndex;
		if (unavailable) {
			// A disabled move (Fake Out after turn 1) deals nothing, but the choice must still be
			// legal: with the ally slot fainted an ally-targeting move (Helping Hand) has no legal
			// target, so prefer a move aimed at a foe or at nobody.
			slot = requested.findIndex((entry) => !entry.disabled && !ALLY_TARGETS.has(entry.target));
			if (slot < 0) slot = Math.max(0, requested.findIndex((entry) => !entry.disabled));
		}
		const targetText = battle.actions.targetTypeChoices(requested[slot]?.target) ? " 1" : "";
		const ourChoice = locked ? "move 1" : `move ${slot + 1}${targetText}`;
		try {
			battle.makeChoices(ourChoice, foeChoice);
		} catch (err) {
			return { error: `turn ${turn} ${condition} move ${moveIndex}: ${err.message} (${ourChoice})` };
		}
		const raw = parseLog(battle.log, logStart, foeState, recorder);
		if (process.env.PLAN_VALUE_DEBUG) {
			console.error(`seed ${seed} turn ${turn} raw ${raw}\n  ` + battle.log.slice(logStart).join("\n  "));
		}
		turns.push(unavailable ? 0 : (100 * raw) / referenceMax);
		// Both sides start every measured turn at full HP (HP-dependent moves such as Eruption
		// must not be penalised for the foes' Tackle).
		ours.hp = ours.maxhp;
		foes.forEach((foe, index) => {
			foe.hp = foe.maxhp;
			foeState[`p2${"ab"[index]}`] = foe.maxhp;
		});
		if (battle.ended) break;
	}
	while (turns.length < TURNS) turns.push(0);
	return { total: turns[0] + turns[1] };
}

function measureAverage(packed, index, condition, profile, seeds, mega) {
	let sum = 0;
	for (const seed of seeds) {
		const outcome = measure(packed, index, condition, profile, seed, mega);
		if (outcome.error) return outcome;
		sum += outcome.total;
	}
	return { total: sum / seeds.length };
}

function handleRequest(msg) {
	const packed = msg.packed;
	// `mega: true` Mega Evolves a stone holder on turn 1; false measures the base form (the
	// Python side asks for both and registers each under its own species id).
	const mega = msg.mega !== false;
	const team = Teams.unpack(packed);
	const moveNames = team[0].moves.map((name) => String(name).toLowerCase().replace(/[^a-z0-9]/g, ""));
	const results = {};
	const errors = [];
	const seeds = msg.seeds && msg.seeds.length ? msg.seeds : [1];
	for (const moveId of msg.moves) {
		const index = moveNames.indexOf(moveId);
		if (index < 0) continue;
		// With luck removed the measurement is deterministic for almost every move; check that
		// with a second seed on the bare field and only pay for more seeds when it is not
		// (Magnitude, Present, Loaded Dice...).
		let moveSeeds = seeds.slice(0, 1);
		if (seeds.length > 1) {
			const bare = msg.conditions[0];
			const same = PROFILES.every((profile) => {
				const a = measure(packed, index, bare, profile, seeds[0], mega);
				const b = measure(packed, index, bare, profile, seeds[1], mega);
				return a.error || b.error || Math.abs(a.total - b.total) < 1e-9;
			});
			if (!same) moveSeeds = seeds;
		}
		results[moveId] = msg.conditions.map((condition) =>
			PROFILES.map((profile) => {
				const outcome = measureAverage(packed, index, condition, profile, moveSeeds, mega);
				if (outcome.error) {
					errors.push(`${moveId}: ${outcome.error}`);
					return null;
				}
				return outcome.total;
			}),
		);
	}
	return { results, errors };
}

const rl = readline.createInterface({ input: process.stdin, terminal: false });
rl.on("line", (line) => {
	if (!line.trim()) return;
	let msg;
	try {
		msg = JSON.parse(line);
	} catch (err) {
		process.stdout.write(`${JSON.stringify({ error: `bad json: ${err.message}` })}\n`);
		return;
	}
	try {
		if (msg.cmd === "ping") {
			process.stdout.write(`${JSON.stringify({ rid: msg.rid, pong: true })}\n`);
			return;
		}
		const out = handleRequest(msg);
		out.rid = msg.rid;
		process.stdout.write(`${JSON.stringify(out)}\n`);
	} catch (err) {
		process.stdout.write(`${JSON.stringify({ rid: msg.rid, error: err.stack || String(err) })}\n`);
	}
});
