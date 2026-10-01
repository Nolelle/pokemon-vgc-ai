// Parse Showdown's config/formats.ts, data/aliases.ts and rulesets files with the
// TypeScript compiler that the Showdown checkout already depends on, for
// `vgc.showdown_relevance`. A real parser (not line matching) is what makes comments,
// template strings, regexes and string escapes unable to disguise a change.
//
// Usage: node tools/parse_showdown_entries.mjs <showdown_repo>
//   stdin:  JSON list of jobs, each {"kind": "formats" | "aliases" | "rulesets", "text": "..."}
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

import {createRequire} from 'node:module';
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

const PARSERS = {formats: parseFormats, aliases: parseAliases, rulesets: parseRulesets};

let input = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', chunk => { input += chunk; });
process.stdin.on('end', () => {
	const jobs = JSON.parse(input);
	const results = jobs.map(job => {
		try {
			const parser = PARSERS[job.kind];
			if (!parser) throw new Error(`unknown kind ${job.kind}`);
			return {ok: true, ...parser(job.text)};
		} catch (err) {
			return {ok: false, error: String(err?.message ?? err)};
		}
	});
	process.stdout.write(JSON.stringify(results));
});
