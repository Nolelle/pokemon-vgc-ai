#!/usr/bin/env node
/** Export the complete mechanics surface used by the Champions Reg M-B format.
 *
 * This intentionally loads the fully merged Showdown Dex. Reading only
 * data/mods/champions/*.ts would miss the ordinary Gen 9 mechanics inherited by the
 * Champions mod. The output records every legal entity, every engine callback attached
 * to those entities, and every declarative move field that can change battle state.
 *
 * Usage:
 *   node tools/export_mechanics_catalog.mjs <showdown-repo> [output.json]
 */

import fs from "node:fs";
import path from "node:path";
import {pathToFileURL} from "node:url";
import {execFileSync} from "node:child_process";

const [, , showdownRepo, outputPath] = process.argv;
if (!showdownRepo) {
	console.error("usage: node tools/export_mechanics_catalog.mjs <showdown-repo> [output.json]");
	process.exit(1);
}

const simPath = path.join(showdownRepo, "dist", "sim", "index.js");
if (!fs.existsSync(simPath)) {
	console.error(`${simPath} not found -- build the Showdown checkout first`);
	process.exit(1);
}

const imported = await import(pathToFileURL(simPath).href);
const Sim = imported.default || imported;
const dex = Sim.Dex.mod("champions");
const format = dex.formats.get("gen9championsvgc2026regmb");
const ruleTable = dex.formats.getRuleTable(format);

function callbacks(effect, prefix = "", seen = new WeakSet()) {
	if (!effect || typeof effect !== "object" || seen.has(effect)) return [];
	seen.add(effect);
	const result = [];
	for (const [key, value] of Object.entries(effect)) {
		const pathKey = prefix ? `${prefix}.${key}` : key;
		if (typeof value === "function") {
			result.push(pathKey);
		} else if (value && typeof value === "object") {
			result.push(...callbacks(value, pathKey, seen));
		}
	}
	return result.sort();
}

function addReverse(index, key, id) {
	if (!index[key]) index[key] = [];
	index[key].push(id);
}

function isLegalEffect(effect) {
	return effect.exists && !effect.isNonstandard && !ruleTable.isBanned(effect);
}

const legalSpecies = dex.species.all()
	.filter(species => species.exists && !species.isNonstandard && !ruleTable.isBannedSpecies(species))
	.sort((a, b) => a.id.localeCompare(b.id));

const legalMoveIds = new Set();
const legalAbilityIds = new Set();
for (const species of legalSpecies) {
	for (const moveId of dex.species.getMovePool(species.id)) {
		const move = dex.moves.get(moveId);
		if (isLegalEffect(move)) legalMoveIds.add(move.id);
	}
	for (const abilityName of Object.values(species.abilities || {})) {
		const ability = dex.abilities.get(abilityName);
		if (isLegalEffect(ability)) legalAbilityIds.add(ability.id);
	}
}

const legalMoves = [...legalMoveIds].map(id => dex.moves.get(id)).sort((a, b) => a.id.localeCompare(b.id));
const legalAbilities = [...legalAbilityIds]
	.map(id => dex.abilities.get(id))
	.sort((a, b) => a.id.localeCompare(b.id));
const legalItems = dex.items.all().filter(isLegalEffect).sort((a, b) => a.id.localeCompare(b.id));

const callbackIndex = {moves: {}, abilities: {}, items: {}, conditions: {}, rules: {}};
for (const [kind, effects] of [
	["moves", legalMoves],
	["abilities", legalAbilities],
	["items", legalItems],
]) {
	for (const effect of effects) {
		for (const callback of callbacks(effect)) addReverse(callbackIndex[kind], callback, effect.id);
	}
}

const conditions = Object.keys(dex.data.Conditions || {})
	.map(id => dex.conditions.get(id))
	.filter(condition => condition.exists)
	.sort((a, b) => a.id.localeCompare(b.id));
for (const condition of conditions) {
	for (const callback of callbacks(condition)) {
		addReverse(callbackIndex.conditions, callback, condition.id);
	}
}

const activeRuleIds = [...ruleTable.keys()]
	.filter(id => !id.startsWith("-") && !id.startsWith("+"))
	.map(id => id.split("=")[0])
	.filter((id, index, values) => values.indexOf(id) === index)
	.sort();
for (const ruleId of activeRuleIds) {
	const rule = dex.formats.get(ruleId);
	if (!rule.exists) continue;
	for (const callback of callbacks(rule)) addReverse(callbackIndex.rules, callback, ruleId);
}

const moveSignalFields = [
	"boosts", "breaksProtect", "damage", "drain", "forceSwitch", "heal",
	"ignoreAbility", "ignoreDefensive", "ignoreEvasion", "ignoreImmunity",
	"ignoreNegativeOffensive", "ignorePositiveDefensive", "multihit", "multiaccuracy",
	"noPPBoosts", "nonGhostTarget", "overrideDefensivePokemon", "overrideDefensiveStat",
	"overrideOffensivePokemon", "overrideOffensiveStat", "pseudoWeather", "recoil", "self",
	"selfDestruct", "selfSwitch", "sideCondition", "slotCondition", "status", "stealsBoosts",
	"terrain", "thawsTarget", "useTargetOffensive", "volatileStatus", "weather", "willCrit",
];
const moveSignals = {};
const moveFlags = {};
const moveTargets = {};
for (const move of legalMoves) {
	if (move.accuracy !== true && move.accuracy !== 100) addReverse(moveSignals, "accuracyCheck", move.id);
	if (move.priority) addReverse(moveSignals, "nonzeroPriority", move.id);
	if ((move.critRatio || 1) > 1) addReverse(moveSignals, "increasedCritRatio", move.id);
	if (move.secondary || move.secondaries) addReverse(moveSignals, "secondaryEffects", move.id);
	for (const field of moveSignalFields) {
		const value = move[field];
		if (value !== undefined && value !== null && value !== false && value !== 0 && value !== "") {
			addReverse(moveSignals, field, move.id);
		}
	}
	for (const [flag, enabled] of Object.entries(move.flags || {})) {
		if (enabled) addReverse(moveFlags, flag, move.id);
	}
	addReverse(moveTargets, move.target || "unknown", move.id);
}

const abilityFlags = {};
for (const ability of legalAbilities) {
	for (const [flag, enabled] of Object.entries(ability.flags || {})) {
		if (enabled) addReverse(abilityFlags, flag, ability.id);
	}
}

const itemSignals = {};
for (const item of legalItems) {
	for (const field of [
		"fling", "ignoreKlutz", "isBerry", "isGem", "isPokeball", "isPrimalOrb",
		"megaStone", "megaEvolves", "naturalGift", "onPlate", "zMove", "zMoveFrom", "zMoveType",
	]) {
		const value = item[field];
		if (value !== undefined && value !== null && value !== false && value !== 0 && value !== "") {
			addReverse(itemSignals, field, item.id);
		}
	}
}

for (const kind of Object.keys(callbackIndex)) {
	for (const ids of Object.values(callbackIndex[kind])) ids.sort();
}
for (const index of [moveSignals, moveFlags, moveTargets, abilityFlags, itemSignals]) {
	for (const ids of Object.values(index)) ids.sort();
}

const catalog = {
	schema_version: 1,
	generated_from: {
		format_id: format.id,
		mod: format.mod,
		showdown_repo: path.resolve(showdownRepo),
		showdown_commit: execFileSync("git", ["rev-parse", "HEAD"], {
			cwd: showdownRepo,
			encoding: "utf8",
		}).trim(),
	},
	format: {
		game_type: format.gameType,
		ruleset: format.ruleset,
		active_rule_table: [...ruleTable.keys()].sort(),
		tera_enabled: false,
		mega_evolution_enabled: true,
		open_team_sheets: true,
		picked_team_size: 4,
		team_size: 6,
		level: 50,
	},
	counts: {
		legal_species_and_formes: legalSpecies.length,
		legal_base_species: legalSpecies.filter(
			species => !species.battleOnly && !species.requiredItem && !species.requiredMove
		).length,
		legal_moves: legalMoves.length,
		legal_abilities: legalAbilities.length,
		legal_items: legalItems.length,
		merged_conditions: conditions.length,
	},
	legal_entities: {
		species: legalSpecies.map(species => species.id),
		moves: legalMoves.map(move => move.id),
		abilities: legalAbilities.map(ability => ability.id),
		items: legalItems.map(item => item.id),
		conditions: conditions.map(condition => condition.id),
		rules: activeRuleIds,
	},
	mechanics_surface: {
		callbacks: callbackIndex,
		move_signals: moveSignals,
		move_flags: moveFlags,
		move_targets: moveTargets,
		ability_flags: abilityFlags,
		item_signals: itemSignals,
	},
};

const serialized = `${JSON.stringify(catalog, null, 2)}\n`;
if (outputPath) {
	fs.mkdirSync(path.dirname(outputPath), {recursive: true});
	fs.writeFileSync(outputPath, serialized);
} else {
	process.stdout.write(serialized);
}
