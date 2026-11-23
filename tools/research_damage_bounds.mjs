// Offline oracle only. Enumerate every native damage roll and both crit states
// from a real serialized battle. No output from this tool enters actor features.
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const module = await import(pathToFileURL(path.join(process.argv[2], 'dist/sim/index.js')));
const { Battle } = module.default || module;
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const results = input.queries.map(query => {
	const damages = [];
	let context;
	for (const crit of [false, true]) {
		for (let roll = 85; roll <= 100; roll++) {
			const battle = Battle.fromJSON(input.state);
			battle.restart(() => {});
			const source = battle.sides[query.source[0]].active[query.source[1]];
			const target = battle.sides[query.target[0]].active[query.target[1]];
			const move = battle.dex.getActiveMove(query.move);
			context = {
				sourceSpeed: source.getStat('spe'), targetSpeed: target.getStat('spe'),
				sourceHP: source.hp, targetHP: target.hp, targetMaxHP: target.maxhp,
				accuracy: move.accuracy, sourceBoosts: source.boosts, targetBoosts: target.boosts,
				sourceStatus: source.status, targetStatus: target.status,
				sourceItem: source.item, targetItem: target.item,
				sourceAbility: source.ability, targetAbility: target.ability,
				sourceVolatiles: Object.keys(source.volatiles), targetVolatiles: Object.keys(target.volatiles),
			};
			move.willCrit = crit;
			move.spreadHit = Boolean(query.spread);
			battle.randomizer = base => Math.floor(Math.floor(base * roll) / 100);
			damages.push({ crit, roll, damage: battle.actions.getDamage(source, target, move, true) });
		}
	}
	return { ...query, context, damages };
});
console.log(JSON.stringify(results));
