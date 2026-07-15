// Dumps the Pokemon Showdown "champions" mod (the mod backing
// gen9championsvgc2026regmb -- see config/formats.ts in the showdown repo) to plain JSON
// under data/champions/. This is the ONLY supported way to get game data for this project:
// the champions mod overrides base-gen9 stats, abilities, legality, and even move identity
// (e.g. Chilling Water replacing Scald) in ways that make vanilla gen9 assumptions wrong.
//
// Run via the thin Python wrapper (tools/export_champions_data.py) or directly:
//   node tools/export_champions_data.mjs <path-to-pokemon-showdown-repo> <output-dir>
//
// Requires the showdown repo to already be built (`node build` / dist/sim present).
import { pathToFileURL } from "node:url";
import path from "node:path";
import fs from "node:fs";

const [, , showdownRepo, outDir] = process.argv;
if (!showdownRepo || !outDir) {
	console.error("usage: node export_champions_data.mjs <showdown-repo> <output-dir>");
	process.exit(1);
}

const dexPath = path.join(showdownRepo, "dist", "sim", "dex.js");
if (!fs.existsSync(dexPath)) {
	console.error(`dist/sim/dex.js not found at ${dexPath} -- run "node build" in the showdown repo first`);
	process.exit(1);
}

const dexModule = await import(pathToFileURL(dexPath).href);
// dist/sim/dex.js is CommonJS and doesn't use plain `exports.X = ...` assignments that
// cjs-module-lexer can statically detect, so a dynamic ESM import only gets `default`
// (the whole module.exports object) rather than named exports.
const Dex = dexModule.Dex ?? dexModule.default.Dex;
const mod = Dex.mod("champions");

fs.mkdirSync(outDir, { recursive: true });

function writeJson(filename, data) {
	const outPath = path.join(outDir, filename);
	fs.writeFileSync(outPath, JSON.stringify(data, null, 2) + "\n");
	return outPath;
}

const counts = {};

// --- species.json ---------------------------------------------------------
// Every species the mod knows about, including mega formes. Each entry carries the
// stone/item required to trigger the mega (requiredItem / requiredItems), plus
// tier/isNonstandard so downstream code can filter to what's actually legal in
// gen9championsvgc2026regmb (Flat Rules bans Mythical + Restricted Legendary on top
// of this).
const species = {};
for (const s of mod.species.all()) {
	species[s.id] = {
		id: s.id,
		name: s.name,
		num: s.num,
		types: s.types,
		baseStats: s.baseStats,
		abilities: s.abilities,
		weightkg: s.weightkg,
		heightm: s.heightm,
		baseSpecies: s.baseSpecies || null,
		forme: s.forme || null,
		isMega: !!s.isMega,
		requiredItem: s.requiredItem || null,
		requiredItems: s.requiredItems || null,
		battleOnly: s.battleOnly || null,
		changesFrom: s.changesFrom || null,
		otherFormes: s.otherFormes || null,
		cosmeticFormes: s.cosmeticFormes || null,
		tier: s.tier || null,
		doublesTier: s.doublesTier || null,
		isNonstandard: s.isNonstandard || null,
		eggGroups: s.eggGroups || [],
		gender: s.gender || null,
		genderRatio: s.genderRatio || null,
	};
}
counts.species = Object.keys(species).length;
writeJson("species.json", species);

// --- moves.json -------------------------------------------------------------
const moves = {};
for (const m of mod.moves.all()) {
	moves[m.id] = {
		id: m.id,
		name: m.name,
		num: m.num,
		type: m.type,
		category: m.category,
		basePower: m.basePower,
		accuracy: m.accuracy,
		pp: m.pp,
		priority: m.priority,
		target: m.target,
		flags: m.flags || {},
		secondary: m.secondary || null,
		secondaries: m.secondaries || null,
		isNonstandard: m.isNonstandard || null,
	};
}
counts.moves = Object.keys(moves).length;
writeJson("moves.json", moves);

// --- items.json (LEGAL items only) ------------------------------------------
// The champions mod curates its own item legality via data/mods/champions/items.ts
// (each entry sets isNonstandard: null for legal, "Past"/etc. for banned). This mod has
// no Choice items, Assault Vest, Heavy-Duty Boots, or Eviolite as legal holds -- do not
// assume vanilla gen9 VGC item legality here.
const items = {};
for (const i of mod.items.all()) {
	if (i.isNonstandard != null) continue;
	items[i.id] = {
		id: i.id,
		name: i.name,
		num: i.num,
		fling: i.fling || null,
		isBerry: !!i.isBerry,
		naturalGift: i.naturalGift || null,
		megaStone: !!i.megaStone,
		megaEvolves: i.megaEvolves || null,
		itemUser: i.itemUser || null,
	};
}
counts.items = Object.keys(items).length;
writeJson("items.json", items);

// --- learnsets.json -----------------------------------------------------------
const learnsets = {};
for (const [id, entry] of Object.entries(mod.data.Learnsets || {})) {
	if (!entry || !entry.learnset) continue;
	learnsets[id] = entry.learnset;
}
counts.learnsets = Object.keys(learnsets).length;
writeJson("learnsets.json", learnsets);

// --- typechart.json -------------------------------------------------------
const typechart = {};
for (const t of mod.types.all()) {
	typechart[t.id] = {
		name: t.name,
		damageTaken: t.damageTaken || {},
	};
}
counts.typechart = Object.keys(typechart).length;
writeJson("typechart.json", typechart);

// --- natures.json -----------------------------------------------------------
const natures = {};
for (const n of mod.natures.all()) {
	natures[n.id] = {
		name: n.name,
		plus: n.plus || null,
		minus: n.minus || null,
	};
}
counts.natures = Object.keys(natures).length;
writeJson("natures.json", natures);

console.log(JSON.stringify(counts));
