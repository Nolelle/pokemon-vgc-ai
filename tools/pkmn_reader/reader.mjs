// Long-lived JSON-lines process hosting @pkmn/client `Battle` objects, one per
// (battle id, perspective). It is the SHADOW reader of the pkmn-shadow trial: nothing in
// the bot's play depends on it. Run it as
//
//     node tools/pkmn_reader/reader.mjs <showdown-repo>
//
// Every request is one JSON object per line with an `rid`; every response echoes it.
//
//   {"cmd":"open","id":B,"perspective":"p1"}          -> {ok:true}
//   {"cmd":"feed","id":B,"lines":[...]}                -> {ok:true, errors:[{line,error}], ms}
//   {"cmd":"snapshot","id":B}                          -> {ok:true, snapshot:{...}, ms}
//   {"cmd":"close","id":B}                             -> {ok:true}
//   {"cmd":"selfcheck"}                                -> {ok:true, checks:{...}}
//
// The first line written (before any request) is {"ready":true,"selfcheck":{...}} or
// {"fatal":"..."} -- the M-C data self-check below runs at startup so a data regression
// (e.g. the pkmn dex silently falling back to published M-B data) fails loudly.
//
// DATA: @pkmn's published dex is M-B era (wrong Mega abilities, Meteor Assault 150), so the
// Generations object is built from OUR pinned Showdown checkout's `champions` mod. The
// `Generations` class calls `dex.forGen(n)` and would silently drop the mod, so the mod is
// wrapped in a Proxy whose forGen() returns the mod itself for every generation.
//
// A bad protocol line never kills the process: the error is returned in the feed response
// (and counted per battle by the Python side), and the following lines are still applied.

import { createRequire } from 'node:module';
import readline from 'node:readline';
import path from 'node:path';

const require = createRequire(import.meta.url);
const showdownRepo = path.resolve(process.argv[2] || '');
const { Dex } = require(path.join(showdownRepo, 'dist', 'sim', 'dex.js'));
const { Generations, toID } = require('@pkmn/data');
const { Battle } = require('@pkmn/client');

const champ = Dex.mod('champions');

// SHIM: the published @pkmn/dex objects differ from Showdown's sim Dex in two ways the client
// depends on, so the sim Dex is adapted (this is a data-layer shim, not a client change):
//   1. every Ability/Item/Move/Species/Condition carries `kind` ('Ability', 'Item', ...); the
//      client's `activateAbility` / `fromEffect.kind === 'Item'` checks need it. Without it
//      NO ability or item reveal via `[from] ability: X` is ever recorded.
//   2. `conditions.get('move: Yawn' | 'ability: Mummy' | 'item: Leftovers')` resolves the
//      prefixed effect names the protocol sends (sim returns a non-existent 'moveyawn').
const kindOf = (et) => (et === 'Pokemon' ? 'Species' : ['Ability', 'Item', 'Move'].includes(et) ? et : 'Condition');
const tagged = new WeakMap(); // Dex objects are frozen: hand out a copy that carries `kind`
const tag = (obj) => {
  if (!obj || obj.kind !== undefined) return obj;
  let view = tagged.get(obj);
  if (!view) {
    // a real copy, not a prototype link: @pkmn/data's Specie copies OWN enumerable fields
    view = Object.create(Object.getPrototypeOf(obj), Object.getOwnPropertyDescriptors(obj));
    Object.defineProperty(view, 'kind', { value: kindOf(obj.effectType), enumerable: true });
    tagged.set(obj, view);
  }
  return view;
};
for (const name of ['abilities', 'items', 'moves', 'species']) {
  const coll = champ[name];
  const orig = coll.get.bind(coll);
  coll.get = (n) => tag(orig(n));
}
{
  const orig = champ.conditions.get.bind(champ.conditions);
  champ.conditions.get = (n) => {
    if (typeof n === 'string') {
      const m = /^\s*(move|ability|item)\s*:\s*(.*)$/i.exec(n);
      if (m) {
        const kind = m[1].toLowerCase();
        if (kind === 'ability') return champ.abilities.get(m[2]);
        if (kind === 'item') return champ.items.get(m[2]);
        const cond = orig(m[2]);
        return cond.exists ? tag(cond) : champ.moves.get(m[2]);
      }
    }
    return tag(orig(n));
  };
}
const wrapped = new Proxy(champ, {
  get(target, prop) {
    if (prop === 'forGen') return () => champ;
    const value = target[prop];
    return typeof value === 'function' ? value.bind(target) : value;
  },
});
const gens = new Generations(wrapped, () => true);

function selfCheck() {
  const gen = gens.get(9);
  const checks = {};
  const expect = (name, got, want) => {
    checks[name] = { got, want, ok: JSON.stringify(got) === JSON.stringify(want) };
  };
  expect('meteorassault_bp', gen.moves.get('meteorassault')?.basePower, 170);
  expect('slash_bp', gen.moves.get('slash')?.basePower, 80);
  expect('milkdrink_target', gen.moves.get('milkdrink')?.target, 'adjacentAllyOrSelf');
  expect('garchomp_mega_z_abilities', gen.species.get('Garchomp-Mega-Z')?.abilities, { 0: 'Levitate' });
  expect('golisopod_mega_abilities', gen.species.get('Golisopod-Mega')?.abilities, { 0: 'Tough Claws' });
  // behavioural check: the client copies Psych Up boosts the right way round
  const b = new Battle(gens);
  const script = [
    '|player|p1|Bot|1', '|player|p2|Opp|2', '|gametype|doubles', '|gen|9',
    '|switch|p1a: Garchomp|Garchomp, L50, M|100/100',
    '|switch|p2a: Golisopod|Golisopod, L50, M|100/100',
    '|-boost|p2a: Golisopod|atk|2',
    '|-copyboost|p1a: Garchomp|p2a: Golisopod|[from] move: Psych Up',
  ];
  for (const l of script) b.add(l);
  expect('psychup_copies_boosts', b.p1.active[0]?.boosts?.atk, 2);
  // the Dex shim: prefixed effect names resolve and carry a kind
  expect('cond_move_prefix_id', gen.conditions.get('move: Yawn')?.id, 'yawn');
  expect('cond_ability_kind', gen.conditions.get('ability: Mummy')?.kind, 'Ability');
  expect('cond_item_kind', gen.conditions.get('item: Leftovers')?.kind, 'Item');
  // behavioural: an ability revealed by a [from] ability: line is recorded on the foe
  const c = new Battle(gens);
  for (const l of [
    '|player|p1|Bot|1', '|player|p2|Opp|2', '|gametype|doubles', '|gen|9',
    '|switch|p1a: Garchomp|Garchomp, L50, M|100/100', '|switch|p2a: Golisopod|Golisopod, L50, M|100/100',
    '|-damage|p1a: Garchomp|90/100|[from] item: Rocky Helmet|[of] p2a: Golisopod',
    '|-activate|p1a: Garchomp|ability: Mummy|Pressure|[of] p2a: Golisopod',
  ]) c.add(l);
  expect('item_from_line_recorded', c.p2.active[0]?.item, 'rockyhelmet');
  checks.all_ok = Object.values(checks).every((c) => c.ok);
  return checks;
}

const clean = (v) => (v === undefined ? null : v);

function snapPokemon(p) {
  const side = p.side;
  const types = (() => {
    try { return p.types.map((t) => String(t)); } catch (e) { return ['?' + String(e).slice(0, 30)]; }
  })();
  const volatiles = {};
  for (const [id, st] of Object.entries(p.volatiles)) {
    volatiles[id] = {
      duration: clean(st.duration), level: clean(st.level),
      speciesForme: clean(st.speciesForme), type: clean(st.type),
      apparentType: clean(st.apparentType), stat: clean(st.stat),
    };
  }
  return {
    name: p.name,
    ident: p.originalIdent,
    speciesForme: p.speciesForme,
    baseSpeciesForme: p.baseSpeciesForme,
    types,
    addedType: clean(p.addedType),
    teraType: clean(p.terastallized),
    hp: p.hp,
    maxhp: p.maxhp,
    fainted: !!p.fainted,
    active: side.active.includes(p),
    slot: side.active.indexOf(p),
    status: clean(p.status),
    sleepTurns: p.statusState.sleepTurns,
    toxicTurns: p.statusState.toxicTurns,
    boosts: { ...p.boosts },
    item: p.item,
    itemEffect: p.itemEffect,
    lastItem: p.lastItem,
    lastItemEffect: p.lastItemEffect,
    ability: p.ability,
    baseAbility: p.baseAbility,
    volatiles,
    moves: p.moveSlots.map((m) => ({ id: m.id, virtual: !!m.virtual })),
    transformed: !!p.volatiles.transform,
    illusion: p.illusion ? p.illusion.originalIdent : null,
    level: p.level,
    gender: p.gender,
    lastMove: p.lastMove,
    canMegaEvo: clean(p.canMegaEvo),
  };
}

function snapSide(side) {
  const conditions = {};
  for (const [id, c] of Object.entries(side.sideConditions)) {
    conditions[id] = { level: c.level, minDuration: c.minDuration, maxDuration: c.maxDuration, remove: !!c.remove };
  }
  return {
    id: side.id,
    name: side.name,
    team: side.team.map(snapPokemon),
    active: side.active.map((p) => (p ? p.name : null)),
    conditions,
    totalPokemon: side.totalPokemon,
    faints: side.faints,
  };
}

function snapshot(battle) {
  const f = battle.field;
  const pseudo = {};
  for (const [id, s] of Object.entries(f.pseudoWeather)) {
    pseudo[id] = { minDuration: s.minDuration, maxDuration: s.maxDuration };
  }
  return {
    turn: battle.turn,
    gameType: battle.gameType,
    tier: battle.tier,
    weather: f.weather ? { name: f.weather, minDuration: f.weatherState.minDuration, maxDuration: f.weatherState.maxDuration } : null,
    terrain: f.terrain ? { name: f.terrain, minDuration: f.terrainState.minDuration, maxDuration: f.terrainState.maxDuration } : null,
    pseudoWeather: pseudo,
    p1: snapSide(battle.p1),
    p2: snapSide(battle.p2),
  };
}

const battles = new Map();

function handle(req) {
  switch (req.cmd) {
    case 'open': {
      battles.set(req.id, { battle: new Battle(gens, req.perspective), perspective: req.perspective });
      return { ok: true };
    }
    case 'feed': {
      const entry = battles.get(req.id);
      if (!entry) return { ok: false, error: 'unknown battle ' + req.id };
      const t0 = process.hrtime.bigint();
      const errors = [];
      for (const line of req.lines) {
        try { entry.battle.add(line); } catch (e) { errors.push({ line: line.slice(0, 300), error: String(e && e.stack ? e.stack.split('\n').slice(0, 3).join(' | ') : e).slice(0, 400) }); }
      }
      return { ok: true, errors, ms: Number(process.hrtime.bigint() - t0) / 1e6 };
    }
    case 'snapshot': {
      const entry = battles.get(req.id);
      if (!entry) return { ok: false, error: 'unknown battle ' + req.id };
      const t0 = process.hrtime.bigint();
      const snap = snapshot(entry.battle);
      return { ok: true, snapshot: snap, ms: Number(process.hrtime.bigint() - t0) / 1e6 };
    }
    case 'close': {
      battles.delete(req.id);
      return { ok: true };
    }
    case 'selfcheck':
      return { ok: true, checks: selfCheck() };
    default:
      return { ok: false, error: 'unknown cmd ' + req.cmd };
  }
}

let checks;
try {
  checks = selfCheck();
  process.stdout.write(JSON.stringify({ ready: true, selfcheck: checks }) + '\n');
} catch (e) {
  process.stdout.write(JSON.stringify({ fatal: String(e && e.stack ? e.stack : e).slice(0, 800) }) + '\n');
  process.exit(1);
}

const rl = readline.createInterface({ input: process.stdin });
rl.on('line', (line) => {
  if (!line.trim()) return;
  let req;
  try { req = JSON.parse(line); } catch (e) { process.stdout.write(JSON.stringify({ ok: false, error: 'bad json' }) + '\n'); return; }
  let res;
  try { res = handle(req); } catch (e) { res = { ok: false, error: String(e && e.stack ? e.stack : e).slice(0, 600) }; }
  res.rid = req.rid;
  process.stdout.write(JSON.stringify(res) + '\n');
});
rl.on('close', () => process.exit(0));
