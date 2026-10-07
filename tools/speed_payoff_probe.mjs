// Speed-payoff probe: what is it worth to ONE of our sets to be under Tailwind / Trick Room,
// measured by the real Showdown engine against a panel of real opposing sets.
//
// Why: moving first matters through KOs before the foe acts, so what Tailwind or Trick Room
// is worth depends on WHAT the team does with the turn (a fast attacker gets its hit off, a
// slow bulky attacker survives to hit under Trick Room). `vgc.field_control`'s generic
// "who is faster" term cannot see that. This probe measures it; `vgc.speed_payoff` and
// `tools/build_speed_payoff_cache.py` turn it into a cache that `vgc.field_control` reads.
//
// Protocol (long-lived, JSON lines on stdin/stdout, like tools/plan_value_probe.mjs):
//   -> {"rid":1,"subject":"<ONE packed set>","foes":["<packed set>", ...],"seeds":[1]}
//   <- {"rid":1,"results":[{"ourMove":"hyper voice","foeMove":"...","ourSpeed":..,"foeSpeed":..,
//        "base":[dealt,taken],"ourtw":[..],"foetw":[..],"tr":[..],"koFoe":{..},"koUs":{..}}, ...],
//       "errors":[...]}
//   dealt / taken are the 2-turn TOTAL expected %HP the subject deals to / takes from the foe,
//   each as a % of the damaged Pokemon's own REAL max HP (a KO caps it at what was left).
//   -> {"cmd":"ping"}  <- {"pong":true}
//
// The duel (one subject vs one reference foe, doubles format with the second slot fainted,
// exactly what a side with one Pokemon left looks like):
//  * MOVE RULE, fixed and symmetric: each side uses, both turns, its single strongest damaging
//    move INTO THE OTHER, where "strongest" is the 2-turn expected %HP that move deals to the
//    other's real max HP when the other just uses Tackle and BOTH sides carry 10x HP (so a big
//    hit is not clipped by the target's HP and nobody is KO'd during selection). Accuracy
//    counts (expected damage = damage x accuracy). The rule is the same with or without
//    Tailwind/Trick Room, so the payoff isolates what moving first does to THESE two moves.
//    A side with no damaging move uses Splash. No Protect, no switching, no status moves.
//  * SCENARIOS, same moves each: "base" (bare field), "ourtw" (Tailwind on OUR side),
//    "foetw" (Tailwind on the foe's side), "tr" (Trick Room). Added after Mega Evolution with
//    the subject as source, through the engine's own addSideCondition/addPseudoWeather, so
//    abilities that react (Wind Rider, Wind Power, Ice Face...) fire. Both turns are inside
//    the condition's duration (Tailwind 4, Trick Room 5).
//  * REAL HP, no resets: the second turn starts from whatever turn 1 left, so a Pokemon that is
//    KO'd on turn 1 deals nothing on turn 2 and a foe KO'd before it moves never hits back.
//    Expected damage is NOT clipped by anything but the foe's remaining HP.
//  * LUCK REMOVED like plan_value_probe: mean damage roll, no crits (Lucky Chant), secondary
//    effects/paralysis/flinch never fire, accuracy forced to pass but each hit weighted by the
//    accuracy the engine computed, 2-5-hit moves run at 5 hits weighted by the engine's own hit
//    distribution. Known approximation: a forced hit also decides who is KO'd; the accuracy
//    weight is applied to the damage only, not to the KO branch.
//  * Recoil, Life Orb, Rocky Helmet and weather chip (`[from]` damage) are real HP lost and count
//    unweighted for whoever lost it (so Double-Edge's recoil is a cost, a foe's recoil a gain).
//
// Usage: node tools/speed_payoff_probe.mjs <path-to-pokemon-showdown-repo>
import { pathToFileURL } from "node:url";
import path from "node:path";
import fs from "node:fs";
import readline from "node:readline";

const [, , showdownRepo] = process.argv;
if (!showdownRepo) {
	console.error("usage: node speed_payoff_probe.mjs <showdown-repo>");
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
const Dex = Sim.Dex.mod("champions");

const FORMAT = "gen9championsdoublescustomgame";
const SPREAD_ROLL = 92.5; // mean of Showdown's 85..100 damage roll
const FIVE_HIT_SURVIVAL = [1, 1, 13 / 20, 6 / 20, 3 / 20];
const TURNS = 2;
const HP_SCALE = 10; // selection only: both sides carry 10x HP
const SCENARIOS = ["base", "ourtw", "foetw", "tr"];
// Moves whose damage is a fraction of current HP or compares HP keep real HP in selection.
const REAL_HP_MOVES = new Set([
	"superfang", "naturesmadness", "ruination", "guardianofalola", "endeavor", "finalgambit",
]);

const DUMMY_SET = {
	species: "Incineroar",
	ability: "noability",
	item: "",
	moves: ["splash"],
	nature: "Serious",
	evs: { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 },
	level: 50,
};

const HP_LINE = /^\|-damage\|(p[12][ab]): [^|]*\|(?:(\d+)\/(\d+)(?: [a-z]+)?|0 fnt)(?:\||$)/;
const HEAL_LINE = /^\|-heal\|(p[12][ab]): [^|]*\|(\d+)\/(\d+)/;

const toId = (name) => String(name).toLowerCase().replace(/[^a-z0-9]/g, "");

// --- luck removal (same mechanics as plan_value_probe.mjs; kept self-contained) -------------

function instrument(battle, record) {
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
	battle.randomChance = (numerator, denominator) => {
		if (pending && denominator === 100 && numerator === pending.value) {
			const { slot, value } = pending;
			pending = null;
			record.accuracy(slot, Math.max(0, Math.min(1, value / 100)));
			return true;
		}
		pending = null;
		return false;
	};
	const originalRandom = battle.random.bind(battle);
	battle.random = (m, n) => (n === undefined && m === 100 ? 99 : originalRandom(m, n));
	battle.randomizer = (baseDamage) => battle.trunc(battle.trunc(baseDamage * SPREAD_ROLL) / 100);
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

// Expected damage this turn per slot, in raw HP, from the new log lines. Hits are accuracy-
// weighted; `[from]` damage (recoil, Life Orb, Rocky Helmet, weather chip) is real HP lost and
// is added unweighted to the Pokemon that lost it.
function parseLog(lines, from, state, recorder) {
	const hitsBySlot = {};
	const extraBySlot = {};
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
			} else {
				extraBySlot[slot] = (extraBySlot[slot] || 0) + Math.max(0, previous - hp);
			}
			continue;
		}
		const heal = HEAL_LINE.exec(line);
		if (heal) state[heal[1]] = Number(heal[2]);
	}
	const bySlot = {};
	for (const [slot, hits] of Object.entries(hitsBySlot)) {
		const accuracy = recorder.accuracyBySlot[slot] || [];
		let cumulative = 1;
		let total = 0;
		hits.forEach((dealt, index) => {
			if (index < accuracy.length) cumulative *= accuracy[index];
			let weight = cumulative;
			if (recorder.fiveHit && hits.length > 1) {
				weight *= FIVE_HIT_SURVIVAL[Math.min(index, 4)];
			}
			total += dealt * weight;
		});
		bySlot[slot] = total;
	}
	for (const [slot, extra] of Object.entries(extraBySlot)) bySlot[slot] = (bySlot[slot] || 0) + extra;
	return bySlot;
}

// --- battle construction ---------------------------------------------------------------------

function restrictedSet(set, moveIds) {
	return { ...set, moves: moveIds, level: 50 };
}

function damagingMoveIds(set) {
	return set.moves
		.map(toId)
		.filter((id) => {
			const move = Dex.moves.get(id);
			return move.exists && move.category !== "Status";
		});
}

// A 1v1 duel: slot 'a' of each side holds the real Pokemon, slot 'b' is a fainted dummy.
function makeBattle(oursSet, foeSet, seed, scale) {
	const battle = new Battle({
		formatid: FORMAT,
		seed: [seed, seed + 1, seed + 2, seed + 3],
		p1: { name: "p1", team: Teams.pack([oursSet, { ...DUMMY_SET }]) },
		p2: { name: "p2", team: Teams.pack([foeSet, { ...DUMMY_SET }]) },
	});
	const recorder = newRecorder();
	instrument(battle, recorder);
	battle.makeChoices("team 12", "team 12");
	for (const side of battle.sides) side.pokemon[1].faint();
	battle.faintMessages();
	battle.clearRequest();
	battle.makeRequest("move");
	const ours = battle.sides[0].pokemon[0];
	const foe = battle.sides[1].pokemon[0];
	const realMax = { ours: ours.maxhp, foe: foe.maxhp };
	if (scale !== 1) {
		for (const mon of [ours, foe]) {
			mon.baseMaxhp = mon.maxhp * scale;
			mon.maxhp = mon.baseMaxhp;
			mon.hp = mon.maxhp;
		}
	}
	for (const side of battle.sides) {
		side.addSideCondition("luckychant", side.pokemon[0]);
		side.totalFainted = 0;
		side.faintedThisTurn = null;
		side.faintedLastTurn = null;
	}
	// Mega Evolve BEFORE any condition (a Mega's new ability can set weather / change speed).
	if (ours.canMegaEvo) battle.actions.runMegaEvo(ours);
	if (foe.canMegaEvo) battle.actions.runMegaEvo(foe);
	return { battle, recorder, ours, foe, realMax };
}

function applyScenario(battle, ours, scenario) {
	if (scenario === "ourtw") battle.sides[0].addSideCondition("tailwind", ours);
	else if (scenario === "foetw") battle.sides[1].addSideCondition("tailwind", battle.sides[1].pokemon[0]);
	else if (scenario === "tr") battle.field.addPseudoWeather("trickroom", ours);
}

// The choice string for one side this turn: its (single) wanted move, or whatever the engine
// still offers (a charge turn is locked, a repeat-banned move falls to the first legal one).
function choiceFor(battle, pokemon, wantedId) {
	const request = pokemon.getMoveRequestData();
	const moves = request.moves || [];
	const locked = moves.length === 1 && request.trapped;
	if (locked) return "move 1";
	let slot = moves.findIndex((entry) => entry.id === wantedId && !entry.disabled);
	if (slot < 0) slot = Math.max(0, moves.findIndex((entry) => !entry.disabled));
	const targeted = battle.actions.targetTypeChoices(moves[slot]?.target);
	return `move ${slot + 1}${targeted ? " 1" : ""}`;
}

// Expected %HP of the DEFENDER's real max HP that `moveId` deals over TURNS turns, with both
// sides at 10x HP and the defender using Tackle; HP is refilled between turns.
function selectionDamage(attackerSet, moveId, defenderSet, seed) {
	const hpScale = REAL_HP_MOVES.has(moveId) ? 1 : HP_SCALE;
	const { battle, recorder, ours, foe, realMax } = makeBattle(
		restrictedSet(attackerSet, [moveId]),
		restrictedSet(defenderSet, ["tackle"]),
		seed,
		hpScale,
	);
	const state = { p1a: ours.maxhp, p2a: foe.maxhp };
	let total = 0;
	for (let turn = 0; turn < TURNS; turn++) {
		const logStart = battle.log.length;
		recorder.reset();
		try {
			battle.makeChoices(choiceFor(battle, ours, moveId), choiceFor(battle, foe, "tackle"));
		} catch (err) {
			return { error: `select ${moveId}: ${err.message}` };
		}
		const dealt = parseLog(battle.log, logStart, state, recorder);
		total += (100 * (dealt.p2a || 0)) / realMax.foe;
		ours.hp = ours.maxhp;
		foe.hp = foe.maxhp;
		state.p1a = ours.maxhp;
		state.p2a = foe.maxhp;
		if (battle.ended) break;
	}
	return { total };
}

function bestMove(attackerSet, defenderSet, seed, errors) {
	let best = { id: "splash", total: -1 };
	for (const id of damagingMoveIds(attackerSet)) {
		const outcome = selectionDamage(attackerSet, id, defenderSet, seed);
		if (outcome.error) {
			errors.push(outcome.error);
			continue;
		}
		if (outcome.total > best.total) best = { id, total: outcome.total };
	}
	return best.id;
}

function duel(oursSet, ourMove, foeSet, foeMove, scenario, seed) {
	const { battle, recorder, ours, foe, realMax } = makeBattle(
		restrictedSet(oursSet, [ourMove]),
		restrictedSet(foeSet, [foeMove]),
		seed,
		1,
	);
	applyScenario(battle, ours, scenario);
	const state = { p1a: ours.maxhp, p2a: foe.maxhp };
	let dealt = 0;
	let taken = 0;
	let koFoe = 0;
	let koUs = 0;
	for (let turn = 0; turn < TURNS && !battle.ended; turn++) {
		const logStart = battle.log.length;
		recorder.reset();
		try {
			battle.makeChoices(choiceFor(battle, ours, ourMove), choiceFor(battle, foe, foeMove));
		} catch (err) {
			return { error: `duel ${scenario} ${ourMove}/${foeMove}: ${err.message}` };
		}
		const hits = parseLog(battle.log, logStart, state, recorder);
		dealt += (100 * (hits.p2a || 0)) / realMax.foe;
		taken += (100 * (hits.p1a || 0)) / realMax.ours;
		if (foe.hp <= 0) koFoe = 1;
		if (ours.hp <= 0) koUs = 1;
	}
	return { dealt, taken, koFoe, koUs, ourSpeed: ours.getStat("spe"), foeSpeed: foe.getStat("spe") };
}

function handleRequest(msg) {
	const subject = Teams.unpack(msg.subject)[0];
	const seed = (msg.seeds && msg.seeds[0]) || 1;
	const errors = [];
	const results = [];
	for (const foePacked of msg.foes) {
		const foe = Teams.unpack(foePacked)[0];
		const ourMove = bestMove(subject, foe, seed, errors);
		const foeMove = bestMove(foe, subject, seed, errors);
		const row = { ourMove, foeMove };
		for (const scenario of SCENARIOS) {
			const outcome = duel(subject, ourMove, foe, foeMove, scenario, seed);
			if (outcome.error) {
				errors.push(outcome.error);
				row[scenario] = null;
				continue;
			}
			row[scenario] = [outcome.dealt, outcome.taken, outcome.koFoe, outcome.koUs];
			if (scenario === "base") {
				row.ourSpeed = outcome.ourSpeed;
				row.foeSpeed = outcome.foeSpeed;
			}
		}
		results.push(row);
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
