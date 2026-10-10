// Calculator-vs-engine probe: runs one controlled doubles turn per request against the local
// Showdown checkout's built sim and reports what the ENGINE did, so
// `offline/sweep_calculator_vs_engine.py` can compare it with our fast maths
// (`vgc.damage`, `vgc.accuracy`, speed/priority ordering, `vgc.priority_rules`).
//
// Long-lived worker, JSON lines on stdin/stdout (one request per line, one response per line):
//   -> {"rid":1,"kind":"turn","p1":[setA,setB],"p2":[setC,setD],
//       "moves":{"p1a":"earthquake","p1b":"splash","p2a":"splash","p2b":"splash"},
//       "targets":{"p1a":1},            // optional: foe slot (1=p2a, 2=p2b) or -1/-2 for allies
//       "state":{"p1a":{"boosts":{"atk":2},"status":"brn","hpFrac":0.5,"noItem":true,
//                       "volatiles":["..."]}, ...},
//       "field":{"weather":"sunnyday","terrain":"electricterrain","trickroom":true,
//                "p1side":["tailwind"],"p2side":["reflect"]},
//       "turns":3, "seeds":[[1,2,3,4]], "keepLog":false}
//   <- {"rid":1,"results":[{"seed":..,"calls":[...],"order":[...],"accuracy":[...],"err":null}]}
//
// How exactness is obtained (no sampling of the damage roll):
//  * `battle.actions.getDamage` is wrapped. For every call made while a move resolves we call
//    the real function once per damage roll (the 16 values of `random(16)`, by overriding
//    `battle.randomizer`), restoring the target's and source's item between calls so
//    consume-on-use items (resist berries) give every roll the same inputs. The real call is
//    made last at roll 0 (max) so the battle itself proceeds normally.
//  * Crits are never rolled: `move.willCrit` is forced to false unless the move sets it true
//    itself (Frost Breath, Wicked Blow, ...), which the report keeps (`crit: true`).
//  * `battle.randomChance(n, 100)` is wrapped to record the first accuracy numerator (the
//    engine's final integer accuracy after every modifier and stage) and to force every roll
//    to pass, so the turn continues and damage/order can also be read from the same run.
//  * Order is read from the `|move|`/`|cant|` protocol lines in the order they appear.
import { pathToFileURL } from "node:url";
import path from "node:path";
import fs from "node:fs";
import readline from "node:readline";

const [, , showdownRepo] = process.argv;
if (!showdownRepo) {
	console.error("usage: node calc_probe.mjs <showdown-repo>");
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
const SLOTS = ["p1a", "p1b", "p2a", "p2b"];
const NEEDS_TARGET = new Set(["normal", "any", "adjacentAlly", "adjacentAllyOrSelf", "adjacentFoe"]);

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

function slotOf(mon) {
	return mon.side.id + "abcd"[mon.position];
}

function snapshotItem(mon) {
	return { item: mon.item, itemState: mon.itemState, lastItem: mon.lastItem, usedItemThisTurn: mon.usedItemThisTurn };
}
function restoreItem(mon, snap) {
	mon.item = snap.item;
	mon.itemState = snap.itemState;
	mon.lastItem = snap.lastItem;
	mon.usedItemThisTurn = snap.usedItemThisTurn;
}

function instrument(battle, rec) {
	const actions = battle.actions;
	const origGetDamage = actions.getDamage;
	let forcedRoll = 0;
	battle.randomizer = function (baseDamage) {
		return Math.trunc(Math.trunc(baseDamage * (100 - forcedRoll)) / 100);
	};
	actions.getDamage = function (source, target, move, suppressMessages) {
		if (typeof move !== "object" || !move.id) {
			return origGetDamage.call(this, source, target, move, suppressMessages);
		}
		if (move.willCrit === undefined) move.willCrit = false;
		const rolls = new Array(16).fill(null);
		const srcSnap = snapshotItem(source);
		const tgtSnap = snapshotItem(target);
		// Roll r = 15 is the minimum (85%), r = 0 the maximum (100%).
		for (let r = 15; r >= 1; r--) {
			forcedRoll = r;
			try {
				rolls[r] = origGetDamage.call(this, source, target, move, true);
			} catch (err) {
				rolls[r] = `err:${err.message}`;
			}
			restoreItem(source, srcSnap);
			restoreItem(target, tgtSnap);
		}
		forcedRoll = 0;
		const real = origGetDamage.call(this, source, target, move, suppressMessages);
		rolls[0] = real;
		const hitData = target.getMoveHitData(move);
		rec.calls.push({
			source: slotOf(source),
			target: slotOf(target),
			move: move.id,
			moveType: move.type,
			hit: move.hit,
			spread: !!move.spreadHit,
			crit: !!hitData.crit,
			typeMod: hitData.typeMod,
			value: typeof real === "number" ? real : null,
			rolls: typeof real === "number" ? rolls : null,
			raw: typeof real === "number" ? undefined : String(real),
			maxhp: target.maxhp,
		});
		return real;
	};
	const origRandomChance = battle.randomChance;
	battle.randomChance = function (numerator, denominator) {
		if (denominator === 100 && battle.activePokemon && slotOf(battle.activePokemon) === "p1a" && battle.activeTarget) {
			const targetSlot = slotOf(battle.activeTarget);
			if (!rec.accuracySeen.has(targetSlot)) {
				rec.accuracySeen.add(targetSlot);
				rec.accuracy.push({ t: targetSlot, n: numerator });
				return true;
			}
		}
		if (denominator === 100 || rec.forcePass) return true;
		return origRandomChance.call(this, numerator, denominator);
	};
}

function applyState(battle, mon, st) {
	for (const key of ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]) {
		mon.boosts[key] = (st && st.boosts && st.boosts[key]) || 0;
	}
	if (!st) return;
	if (st.status) {
		mon.clearStatus();
		mon.setStatus(st.status, null, null, true);
	}
	if (st.hpFrac !== undefined && st.hpFrac !== null) {
		mon.hp = Math.max(1, Math.floor(mon.maxhp * st.hpFrac));
	}
	if (st.faint) {
		mon.faint();
		return;
	}
	if (st.noItem) {
		mon.lastItem = mon.item;
		mon.item = "";
		mon.itemState = { id: "", target: mon };
		mon.usedItemThisTurn = true;
	}
	for (const volatile of st.volatiles || []) mon.addVolatile(volatile, mon);
}

function applyField(battle, field, sources) {
	battle.field.clearWeather();
	battle.field.clearTerrain();
	if (!field) return;
	if (field.weather) {
		battle.field.setWeather(field.weather, sources.p1a, battle.dex.abilities.get(WEATHER_SOURCE[field.weather]));
	}
	if (field.terrain) {
		battle.field.setTerrain(field.terrain, sources.p1a, battle.dex.abilities.get(TERRAIN_SOURCE[field.terrain]));
	}
	if (field.trickroom) battle.field.addPseudoWeather("trickroom", sources.p1a);
	if (field.gravity) battle.field.addPseudoWeather("gravity", sources.p1a);
	if (field.magicroom) battle.field.addPseudoWeather("magicroom", sources.p1a);
	if (field.wonderroom) battle.field.addPseudoWeather("wonderroom", sources.p1a);
	for (const id of field.p1side || []) battle.sides[0].addSideCondition(id, sources.p1a);
	for (const id of field.p2side || []) battle.sides[1].addSideCondition(id, sources.p2a);
}

function choiceFor(battle, mon, moveId, target, mega) {
	// The sets used by the sweep carry the tested move first and Splash second, so the move
	// index is looked up by id (a charge turn or lock-in offers a single move).
	const request = mon.getMoveRequestData();
	const moves = request.moves || [];
	const index = moves.findIndex((m) => m.id === moveId);
	if (moves.length === 1 && request.trapped) return "move 1";
	const slot = index >= 0 ? index : 0;
	const entry = moves[slot];
	const dexMove = battle.dex.moves.get(entry ? entry.id : moveId);
	let text = `move ${slot + 1}`;
	if (NEEDS_TARGET.has(dexMove.target) && battle.activePerHalf === 2) {
		text += ` ${target ?? 1}`;
	}
	if (mega) text += " mega";
	return text;
}

function run(req, seed) {
	const rec = { calls: [], accuracy: [], accuracySeen: new Set(), forcePass: !!req.forcePass };
	const battle = new Battle({
		formatid: FORMAT,
		seed,
		p1: { name: "p1", team: Teams.pack(req.p1.map((s) => ({ ...s, level: 50 }))) },
		p2: { name: "p2", team: Teams.pack(req.p2.map((s) => ({ ...s, level: 50 }))) },
	});
	instrument(battle, rec);
	battle.makeChoices("team 12", "team 12");
	const mons = {};
	for (const slot of SLOTS) {
		const side = battle.sides[slot[1] === "1" ? 0 : 1];
		mons[slot] = side.active["ab".indexOf(slot[2])];
	}
	for (const slot of SLOTS) applyState(battle, mons[slot], (req.state || {})[slot]);
	applyField(battle, req.field, mons);
	battle.faintMessages();
	// A fresh request after the manual changes, so the choice parser sees the final state.
	battle.clearRequest();
	battle.makeRequest("move");
	const logStart = battle.log.length;
	const stats = {};
	const applied = {};
	for (const slot of SLOTS) {
		const mon = mons[slot];
		applied[slot] = { status: mon.status || null, fainted: !!mon.fainted, hp: mon.hp, item: mon.item || "" };
		stats[slot] = { hp: mon.maxhp, ...mon.storedStats, spe_now: mon.getStat("spe"), speed_action: mon.getActionSpeed() };
	}
	const order = [];
	let err = null;
	const turns = req.turns || 1;
	let turnsRun = 0;
	for (let turn = 0; turn < turns; turn++) {
		rec.accuracySeen = new Set();
		try {
			const choices = { p1: [], p2: [] };
			for (const slot of SLOTS) {
				if (mons[slot].fainted) {
					choices[slot.slice(0, 2)].push("pass");
					continue;
				}
				const moveId = (req.moves || {})[slot] || "splash";
				const target = (req.targets || {})[slot];
				choices[slot.slice(0, 2)].push(choiceFor(battle, mons[slot], moveId, target, turn === 0 && ((req.mega || {})[slot])));
			}
			battle.makeChoices(choices.p1.join(", "), choices.p2.join(", "));
		} catch (e) {
			err = String(e && e.message ? e.message : e);
			break;
		}
		turnsRun++;
		const testMove = (req.moves || {}).p1a;
		const done = req.stopWhen === "none" ? false : rec.calls.some((c) => c.source === "p1a" && c.move === testMove);
		if (battle.ended || done) break;
	}
	const lines = battle.log.slice(logStart);
	for (const line of lines) {
		const m = line.match(/^\|(move|cant)\|(p[12][ab]):/);
		if (m && !order.includes(m[2])) order.push(m[2]);
		if (m && m[1] === "move" && order.length > 8) break;
	}
	return {
		seed,
		calls: rec.calls,
		accuracy: rec.accuracy,
		stats,
		applied,
		order,
		err,
		turnsRun,
		log: req.keepLog ? lines : lines.filter((l) => /^\|(-immune|-fail|-activate|-block|-miss|-ability|-damage|-heal|-start|move|cant|-hitcount|-notarget|-singleturn|-endability|-clearallboost)/.test(l)).slice(0, 40),
	};
}

function handle(req) {
	if (req.cmd === "ping") return { pong: true };
	const seeds = req.seeds || [[1, 2, 3, 4]];
	const results = [];
	for (const seed of seeds) {
		try {
			results.push(run(req, seed));
		} catch (e) {
			results.push({ seed, calls: [], accuracy: [], order: [], err: `probe: ${e && e.stack ? e.stack.split("\n").slice(0, 3).join(" | ") : e}` });
		}
	}
	return { results };
}

const rl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
for await (const line of rl) {
	if (!line.trim()) continue;
	let req;
	try {
		req = JSON.parse(line);
	} catch (e) {
		process.stdout.write(JSON.stringify({ error: "bad json" }) + "\n");
		continue;
	}
	let out;
	try {
		out = handle(req);
	} catch (e) {
		out = { error: String(e && e.stack ? e.stack : e) };
	}
	out.rid = req.rid;
	process.stdout.write(JSON.stringify(out) + "\n");
}
