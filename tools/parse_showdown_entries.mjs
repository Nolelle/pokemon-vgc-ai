// Parse Showdown's config/formats.ts, data/aliases.ts and rulesets files with the
// TypeScript compiler that the Showdown checkout already depends on, for
// `vgc.showdown_relevance`. A real parser (not line matching) is what makes comments,
// template strings, regexes and string escapes unable to disguise a change.
//
// Usage: node tools/parse_showdown_entries.mjs <showdown_repo>
//   stdin:  JSON list of jobs, each {"kind": "formats" | "aliases" | "rulesets", "text": "..."},
//           {"kind": "compile", "text": "...", "file": "config/formats.ts"} or
//           {"kind": "loadformats", "formats": "<js>", "aliases": "<js>", "mods": [...]}
//   stdout: JSON list of results in the same order. Every result has "ok"; when false,
//           "error" says why and the caller must treat the file as unclassifiable.
//
// formats  -> {skeleton, elements: [{text, name, nameStatus, section, ruleset, mod,
//             loadTimeCode}]}
//             skeleton = the file with the Formats array's contents removed.
//             nameStatus: "ok" | "missing" | "unparsed" (name not a plain string literal,
//             or the entry has a spread/accessor/computed key that could set it).
//             loadTimeCode: evaluating the entry can run code (calls, getters, ...).
//             ruleset: decoded string list, or null if absent/not all literals.
// aliases  -> {skeleton, entries: [[key, value | null]], loadTimeCode}
//             (value null = not a plain string)
//             skeleton = the file with the Aliases object's contents removed.
// rulesets -> {keys: [...]}  property names of the exported Rulesets object.
// compile  -> {js}: `text` (repo-relative `file`) compiled exactly as Showdown's build
//             does (tools/build-utils.js: esbuild, cjs, the checkout's tsconfig.json).
//             Compiling runs no upstream code. esbuild spawns its binary, so compile
//             jobs cannot share a process with loadformats (see below).
// loadformats -> {loadError: null | "..."}: install `formats` and `aliases` (compiled
//             JS for config/formats.ts and data/aliases.ts at the candidate commit) in
//             place of the checkout's built copies, make the mod list `mods` (the
//             commit's data/mods folders), then run the BUILT server's own
//             code: Dex.formats.all() and the `formatListText` getter from
//             dist/server/rooms.js, which the server runs to send clients the format
//             list (it builds the rule table of every visible format). loadError is
//             what Showdown threw. At most one per process (the Dex caches the list).
//             Parsing alone cannot see e.g. a name with no alphanumerics, `mod: null`,
//             a deprecated field, a bare identifier, or a rule that no longer resolves.
//
// vgc.showdown_relevance runs loadformats jobs with an empty environment under Node's
// permission model (reads limited to the checkout and this script; no writes, child
// processes or workers). That is damage limitation, not a security boundary: Node's
// permission model does not contain malicious code, and network access stays open.
// It is the same trust we already place in upstream: tools/sync_showdown.py builds and
// runs these files.

import Module, {createRequire} from 'node:module';
import {readFileSync, realpathSync} from 'node:fs';
import path from 'node:path';

const showdownRepo = process.argv[2];
if (!showdownRepo) {
	process.stderr.write('usage: parse_showdown_entries.mjs <showdown_repo>\n');
	process.exit(2);
}
const ts = createRequire(path.join(showdownRepo, 'package.json'))('typescript');

function unwrap(node) {
	while (node && (ts.isAsExpression(node) || ts.isSatisfiesExpression(node) ||
		ts.isParenthesizedExpression(node) || ts.isTypeAssertionExpression?.(node))) {
		node = node.expression;
	}
	return node;
}

function exportedInitializer(source, name) {
	let found = null;
	let count = 0;
	for (const statement of source.statements) {
		if (!ts.isVariableStatement(statement)) continue;
		for (const decl of statement.declarationList.declarations) {
			if (ts.isIdentifier(decl.name) && decl.name.text === name) {
				count++;
				found = unwrap(decl.initializer);
			}
		}
	}
	if (count !== 1) throw new Error(`expected one ${name} declaration, found ${count}`);
	return found;
}

function stringValue(node) {
	node = unwrap(node);
	if (node && (ts.isStringLiteral(node) || ts.isNoSubstitutionTemplateLiteral(node))) {
		return node.text;
	}
	return null;
}

function propertyName(prop) {
	if (!prop.name) return null;
	if (ts.isIdentifier(prop.name) || ts.isStringLiteral(prop.name) ||
		ts.isNumericLiteral(prop.name) || ts.isNoSubstitutionTemplateLiteral(prop.name)) {
		return prop.name.text;
	}
	return null; // computed or private name
}

function parse(text) {
	const source = ts.createSourceFile('x.ts', text, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS);
	if (source.parseDiagnostics?.length) {
		throw new Error(`syntax error: ${ts.flattenDiagnosticMessageText(source.parseDiagnostics[0].messageText, ' ')}`);
	}
	return source;
}

// The file text with [open+1, close) cut out, keeping the delimiters.
function hollow(text, container) {
	const open = container.getStart() + 1;
	const close = container.getEnd() - 1;
	return text.slice(0, open) + text.slice(close);
}

// True if evaluating `node` (not calling functions it defines) can run code: calls,
// property reads (getters), spreads, accessors, assignments. Function and method bodies
// only run inside that format's own battles, so they are not descended into.
const RUNS_CODE = new Set([
	ts.SyntaxKind.CallExpression, ts.SyntaxKind.NewExpression,
	ts.SyntaxKind.TaggedTemplateExpression, ts.SyntaxKind.PropertyAccessExpression,
	ts.SyntaxKind.ElementAccessExpression, ts.SyntaxKind.SpreadElement,
	ts.SyntaxKind.SpreadAssignment, ts.SyntaxKind.GetAccessor, ts.SyntaxKind.SetAccessor,
	ts.SyntaxKind.DeleteExpression, ts.SyntaxKind.AwaitExpression,
	ts.SyntaxKind.YieldExpression, ts.SyntaxKind.ImportKeyword,
]);
function runsCodeAtLoad(node) {
	if (ts.isFunctionExpression(node) || ts.isArrowFunction(node) ||
		ts.isMethodDeclaration(node) || ts.isFunctionDeclaration(node)) {
		return false;
	}
	if (RUNS_CODE.has(node.kind)) return true;
	if ((ts.isPrefixUnaryExpression(node) || ts.isPostfixUnaryExpression(node)) &&
		(node.operator === ts.SyntaxKind.PlusPlusToken || node.operator === ts.SyntaxKind.MinusMinusToken)) {
		return true;
	}
	if (ts.isBinaryExpression(node) &&
		node.operatorToken.kind >= ts.SyntaxKind.FirstAssignment &&
		node.operatorToken.kind <= ts.SyntaxKind.LastAssignment) {
		return true;
	}
	return ts.forEachChild(node, runsCodeAtLoad) === true;
}

function parseFormats(text) {
	const source = parse(text);
	const array = exportedInitializer(source, 'Formats');
	if (!array || !ts.isArrayLiteralExpression(array)) throw new Error('Formats is not an array literal');
	const elements = array.elements.map(element => {
		const body = element.getText(source);
		const loadTimeCode = runsCodeAtLoad(element);
		if (!ts.isObjectLiteralExpression(element)) {
			return {text: body, name: null, nameStatus: 'unparsed', section: false, ruleset: null,
				mod: null, loadTimeCode};
		}
		let name = null;
		let nameStatus = 'missing';
		let section = false;
		let ruleset = null;
		let mod = null;
		let opaque = false;
		for (const prop of element.properties) {
			if (ts.isMethodDeclaration(prop) && propertyName(prop) !== null) continue; // inert at load
			const key = ts.isPropertyAssignment(prop) ? propertyName(prop) : null;
			if (key === null) {
				// Spread, shorthand, accessor or computed key: it may set `name` (in any order).
				opaque = true;
				continue;
			}
			if (key === 'name') {
				name = stringValue(prop.initializer);
				nameStatus = name === null ? 'unparsed' : 'ok';
			} else if (key === 'section') {
				section = true;
			} else if (key === 'mod') {
				mod = stringValue(prop.initializer);
			} else if (key === 'ruleset') {
				const value = unwrap(prop.initializer);
				if (value && ts.isArrayLiteralExpression(value)) {
					const rules = value.elements.map(stringValue);
					ruleset = rules.every(rule => rule !== null) ? rules : null;
				}
			}
		}
		if (opaque) nameStatus = 'unparsed';
		return {text: body, name, nameStatus, section, ruleset, mod, loadTimeCode};
	});
	return {skeleton: hollow(text, array), elements};
}

function parseAliases(text) {
	const source = parse(text);
	const object = exportedInitializer(source, 'Aliases');
	if (!object || !ts.isObjectLiteralExpression(object)) throw new Error('Aliases is not an object literal');
	const loadTimeCode = runsCodeAtLoad(object);
	const entries = object.properties.map(prop => {
		const key = ts.isPropertyAssignment(prop) ? propertyName(prop) : null;
		if (key === null) throw new Error(`unparseable alias entry: ${prop.getText(source).slice(0, 60)}`);
		return [key, stringValue(prop.initializer)];
	});
	return {skeleton: hollow(text, object), entries, loadTimeCode};
}

function parseRulesets(text) {
	const source = parse(text);
	const object = exportedInitializer(source, 'Rulesets');
	if (!object || !ts.isObjectLiteralExpression(object)) throw new Error('Rulesets is not an object literal');
	const keys = object.properties.map(prop => {
		const key = propertyName(prop);
		if (key === null) throw new Error('unparseable ruleset key');
		return key;
	});
	return {keys};
}

function compile(text, file) {
	const repo = path.resolve(showdownRepo);
	const esbuild = createRequire(path.join(repo, 'package.json'))('esbuild');
	const out = esbuild.buildSync({
		stdin: {contents: text, loader: 'ts', sourcefile: file, resolveDir: path.join(repo, path.dirname(file))},
		format: 'cjs',
		tsconfig: path.join(repo, 'tsconfig.json'),
		write: false,
		logLevel: 'silent',
	});
	return {js: out.outputFiles[0].text};
}

function installModule(require, file, js) {
	const stub = new Module(file);
	stub.filename = file;
	stub.paths = Module._nodeModulePaths(path.dirname(file));
	require.cache[file] = stub;
	stub._compile(js, file);
	stub.loaded = true;
}

// The server's format-list getter, taken verbatim from the built dist/server/rooms.js.
function serverFormatListGetter(dist, Dex) {
	const file = path.join(dist, 'server', 'rooms.js');
	const source = ts.createSourceFile(file, readFileSync(file, 'utf8'), ts.ScriptTarget.Latest, true);
	const found = [];
	const visit = node => {
		if (ts.isGetAccessorDeclaration(node) && propertyName(node) === 'formatListText') found.push(node);
		ts.forEachChild(node, visit);
	};
	visit(source);
	if (found.length !== 1) throw new Error(`expected one formatListText getter, found ${found.length}`);
	const holder = new Function('Dex', 'Ladders', `return {${found[0].getText(source)}};`)(
		Dex, {formatsListPrefix: ''}
	);
	return Object.getOwnPropertyDescriptor(holder, 'formatListText').get;
}

let loadedFormats = false;
function loadFormats(job) {
	if (loadedFormats) throw new Error('only one loadformats job per process');
	loadedFormats = true;
	// require.cache is keyed by real path; a symlinked path would leave the stub unused.
	const repo = realpathSync(path.resolve(showdownRepo));
	const dist = path.join(repo, 'dist');
	const require = createRequire(path.join(repo, 'package.json'));
	// Both are required lazily by the loader (Dex.loadAliases, Formats.load).
	const stubs = [
		[path.join(dist, 'data', 'aliases.js'), job.aliases],
		[path.join(dist, 'config', 'formats.js'), job.formats],
	];
	for (const [file] of stubs) realpathSync(file);  // fails closed if the checkout is unbuilt
	const {Dex} = require('./dist/sim/dex');
	const getter = serverFormatListGetter(dist, Dex);
	// Mods are the candidate commit's data/mods folders, not the pinned build's, so a
	// deleted mod still used by an unchanged format fails and a newly added one exists.
	// A new mod's own code is not in the build: it stands in as the base dex.
	const dexes = Dex.dexes;
	for (const mod of Object.keys(dexes)) {
		if (dexes[mod] !== Dex && !job.mods.includes(mod)) delete dexes[mod];  // keep base, gen9
	}
	for (const mod of job.mods) dexes[mod] ??= Dex;
	try {
		for (const [file, js] of stubs) installModule(require, file, js);
		Dex.formats.all();
		getter.call({formatList: null});
	} catch (err) {
		return {loadError: String(err?.message ?? err)};
	}
	return {loadError: null};
}

const PARSERS = {
	formats: job => parseFormats(job.text),
	aliases: job => parseAliases(job.text),
	rulesets: job => parseRulesets(job.text),
	compile: job => compile(job.text, job.file),
	loadformats: loadFormats,
};

let input = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', chunk => { input += chunk; });
process.stdin.on('end', () => {
	const jobs = JSON.parse(input);
	const results = jobs.map(job => {
		try {
			const parser = PARSERS[job.kind];
			if (!parser) throw new Error(`unknown kind ${job.kind}`);
			return {ok: true, ...parser(job)};
		} catch (err) {
			return {ok: false, error: String(err?.message ?? err)};
		}
	});
	process.stdout.write(JSON.stringify(results));
});
