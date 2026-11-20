"""Compile source-anchored Pass 2 plans and export reproducible encrypted evidence."""

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess

from .pass2_evidence import (Oracle, PROTOCOL, TOLERANCE, fingerprint, load,
                             reduce_arm, require, save, sha, stock_abi, suite_precision)
from .pass2_models import COVERAGE, WORKLOADS, source_model
from .unified import run


GENERIC = ('--mlir-print-op-generic', '--mlir-print-local-scope')
DEPTH = 12
# This budget is for the full activation, not a stage or the final CKKS error.
EPSILON = 2.0 ** -12
RESIDUAL_KEY_PRICE = 0.25


def dag_id(c):
    return fingerprint([c['output'], [(n['op'], n['args'], n['weights'], n['constant']) for n in c['nodes']]])


def route_terms(audit):
    return [r['Terms'] for r in sorted(audit['backend'], key=lambda r: r['site'])]


def needs_composed_coverage(manifest):
    return not any(info['coverage'].get('silu_tanh_rational', {}).get('planned_for_encryption') for info in manifest.values())


def write_harness(root, out, manifest):
    pkg = out / 'go_test'
    pkg.mkdir(exist_ok=True)
    shutil.copyfile(root / 'test/pass2/harness_test.go', pkg / 'harness_test.go')
    module = next(l.split()[1] for l in (root / 'go.mod').read_text().splitlines() if l.startswith('module '))
    prefix = f'{module}/{out.relative_to(root).as_posix()}'
    imports, tests = [], []
    fn = 'Linear_silu_linear__'
    for work, entry in manifest.items():
        d = out / work
        profile = load(d / 'profile.json')
        lines = [f'func Test_{work}(t *testing.T) {{', f'  dir := "../{work}"', f'  experiment(t, dir, {entry["width"]}, []arm{{']
        for arm in entry['arms']:
            if arm == 'heir':
                abi = stock_abi((d / 'original-ckks.mlir').read_text())
            else:
                sink = load(d / arm / 'audit.json')['sinks'][0]
                abi = {'input_level': len(profile['Q']) - 1, 'input_log_scale': profile['logDefaultScale'],
                       'output_level': sink['level'], 'output_log_scale': sink['nominal_log_scale']}
            save(d / arm / 'abi.json', abi)
            alias = f'p{len(imports)}'
            imports.append(f'{alias} "{prefix}/{work}/{arm}"')
            names = ', '.join(f'{alias}.{fn}{k}' for k in ('configure', 'preprocessing', 'preprocessed', 'encrypt__arg0', 'decrypt__result0'))
            lines.append(f'    newArm(t, dir, "{arm}", {names}),')
        tests.append('\n'.join(lines + ['  })', '}']))
    (pkg / 'models_test.go').write_text('package pass2_test\nimport (\n"testing"\n' + '\n'.join(imports) + '\n)\n' + '\n'.join(tests) + '\n')


class Pipeline:
    def __init__(self, args, profile=None):
        self.root, self.out = args.root.resolve(), args.out.resolve()
        self.heir, self.translate = (args.heir_bin.resolve() / ('heir-' + t) for t in ('opt', 'translate'))
        self.opt = (args.snabf_opt.resolve(), '--allow-unregistered-dialect', '--verify-each')
        self.goemit = args.goemit.resolve()
        self.fixtures = self.root / 'test/fixtures'
        self.profile = profile or self.fixtures / 'reference_profile.json'
        self.costs = load(self.fixtures / 'pass1_costs.json')
        self.key_bytes = int(subprocess.check_output([self.goemit, 'key-size', self.profile]))
        self.cap = args.extra_rotation_cap

    def call(self, *argv):
        # Paths, option values and phase order are retained for replay. All MLIR
        # interpretation/selection remains in the registered native passes.
        with (self.out / 'commands.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(list(map(str, argv))) + '\n')
        run(*argv)

    def source(self, work, d):
        d.mkdir()
        source = source_model(self.fixtures, work)
        model = d / 'input.mlir'
        model.write_text(source, encoding='utf-8')
        shutil.copyfile(self.profile, d / 'profile.json')
        oracle = Oracle(source)
        save(d / 'corpus.json', oracle.corpus())
        self.lower_source(model, d, work in ('residual', 'cross_residual'))
        stock = d / 'heir'
        stock.mkdir()
        self.call(self.heir, '--scheme-to-lattigo', d / 'original-ckks.mlir', '-o', stock / 'lattigo.mlir')
        self.call(self.translate, '--emit-lattigo', '--package-name=baseline', stock / 'lattigo.mlir', '-o', stock / 'model.go')
        if work in ('residual', 'cross_residual'):
            require('"ckks.add"' in (d / 'original-ckks.mlir').read_text() and '.Add(' in (stock / 'model.go').read_text(),
                    'the source residual did not become an encrypted CKKS/Go add')
        return oracle

    def lower_source(self, model, d, residual):
        """Lower the supplied source, independently of fixture generation or a stock comparator."""
        options = f'min-slot-count=4096 parameter-file={self.profile}'
        if residual:
            options += ' greedy-modulus-switch-after-mul=true'
        self.call(self.heir, *GENERIC, model, '-o', d / 'source-generic.mlir')
        self.call(*self.opt, '--snabf-mark-activations', *GENERIC, d / 'source-generic.mlir', '-o', d / 'marked.mlir')
        self.call(self.heir, '--annotate-module=backend=lattigo scheme=ckks', '--linalg-canonicalizations',
                  '--torch-linalg-to-ckks=' + options, *GENERIC, d / 'marked.mlir', '-o', d / 'original-ckks.mlir')

    def import_ckks(self, work, d, cap=None):
        price = RESIDUAL_KEY_PRICE if work == 'residual' else self.costs['key_units_per_mib']
        opts = (f'key-bytes={self.key_bytes} hoist={self.costs["hoisted_per_active_limb"]} '
                f'giant={self.costs["giant_per_active_limb"]} key-units-per-mib={price}')
        if cap is not None:
            opts += f' rotation-path-limit={cap}'
        imported = d / ('imported.mlir' if cap is None else 'imported-cap.mlir')
        self.call(*self.opt, '--snabf-import-heir=' + opts, d / 'original-ckks.mlir', '-o', imported)
        return imported

    def plan(self, d, arm, imported, min_level, max_error, template=None):
        dest = d / arm
        dest.mkdir()
        options = f'depth={DEPTH} epsilon={EPSILON:.17g} min-output-level={min_level} max-output-error={max_error:.17g}'
        if template is not None:
            require(d.name == COVERAGE and arm.startswith('forced_composite') and template == 'silu_tanh_rational',
                    'template selection is reserved for labeled coverage, never a free-planner result')
            options += ' template-family=composed'
        save(dest / 'contract.json', {'depth': DEPTH, 'epsilon': EPSILON, 'min_output_level': min_level,
                                     'max_approx_error': max_error, 'import_sha256': sha(imported), 'template_filter': template})
        self.call(*self.opt, '--snabf-pass2-candidates=' + options, imported, '-o', dest / 'candidates.mlir')
        self.call(*self.opt, '--snabf-plan', dest / 'candidates.mlir', '-o', dest / 'selected.mlir')
        self.call(*self.opt, '--snabf-check-plan', dest / 'selected.mlir', '-o', dest / 'checked.mlir')
        self.call(*self.opt, '--snabf-lower-ckks', *GENERIC, dest / 'checked.mlir', '-o', dest / 'ckks.mlir')
        self.call(self.heir, '--scheme-to-lattigo', *GENERIC, dest / 'ckks.mlir', '-o', dest / 'lattigo.mlir')
        self.call(*self.opt, '--snabf-bind-lattigo=audit=' + str(dest / 'audit.json'), *GENERIC,
                  dest / 'lattigo.mlir', '-o', dest / 'bound.mlir')
        self.call(self.heir, dest / 'bound.mlir', '-o', dest / 'verified.mlir')
        self.call(self.translate, '--emit-lattigo', '--package-name=planned', dest / 'verified.mlir', '-o', dest / 'translated.go.txt')
        audit = load(dest / 'audit.json')
        require(len(audit['polynomials']) == len(audit['sinks']) == 1 and audit['decisions'], f'{dest}: incomplete native audit')
        p, sink = audit['polynomials'][0], audit['sinks'][0]
        c = p['selected']
        if template is not None:
            require(c['template'] == template and len(c['stages']) == 2, f'{dest}: composite coverage did not execute two stages')
        require(any(x == c for x in p['frontier']) and c['stages'] and c['nodes'], f'{dest}: unattributed selected candidate')
        require(p['output_level'] == p['input_level'] - c['delta_level'] and sink['level'] >= min_level and
                0 <= sink['approx_error'] <= max_error and 0 <= c['activation_error'] <= EPSILON,
                f'{dest}: native selection violates its declared contract')
        save(dest / 'routes.json', audit['backend'])  # do not rewrite or strip the native audit
        self.call(self.goemit, 'emit', dest / 'routes.json', dest / 'translated.go.txt', dest / 'model.go')
        print(f'{d.name}/{arm}: {c["candidate_id"]}, {c["template"]}, poly L{p["input_level"]}->{p["output_level"]}, '
              f'sink L{sink["level"]}, {audit["key_bytes"]} Galois bytes', flush=True)
        return audit

    def workload(self, work):
        d = self.out / work
        oracle = self.source(work, d)
        imported = self.import_ckks(work, d)
        loose = self.plan(d, 'economy', imported, 0, TOLERANCE)
        site, sink = loose['polynomials'][0], loose['sinks'][0]
        suffix = site['output_level'] - sink['level']
        candidates = [c for c in site['frontier'] if site['input_level'] - c['delta_level'] >= suffix]
        # Only public Q levels and deterministic approximation certificates
        # inform the budget sweep. Real ciphertexts have not been evaluated yet.
        predictions = {c['candidate_id']: c['activation_error'] * oracle.output_norm for c in candidates}
        levels = {c['candidate_id']: site['input_level'] - c['delta_level'] - suffix for c in candidates}
        require(math.isclose(predictions[site['selected']['candidate_id']], sink['approx_error'], rel_tol=1e-8, abs_tol=1e-18),
                f'{work}: unknown suffix transfer; do not infer error budgets from this graph')
        # Use the geometric midpoint between adjacent distinct certified errors,
        # not the next floating-point number: native outward rounding is allowed.
        errors = sorted(set(predictions.values()))
        def upper(e):
            hi = next((v for v in errors if v > e), TOLERANCE)
            return min(TOLERANCE, max(e, math.sqrt(e * hi) if e else hi / 2))
        budgets = [(0, upper(e)) for e in errors if upper(e) < sink['approx_error']]
        require(budgets, f'{work}: no different precision program reachable from this native frontier')
        arms = ['heir', 'economy']
        chosen = {'economy': loose}
        attempted = []
        def attempt(name, budget):
            a = self.plan(d, name, imported, *budget)
            c = a['polynomials'][0]['selected']
            ident = dag_id(c)
            different = all(ident != dag_id(prev['polynomials'][0]['selected']) for prev in chosen.values())
            attempted.append({'directory': name, 'min_output_level': budget[0], 'max_approx_error': budget[1],
                              'selected': c['candidate_id'], 'template': c['template'], 'distinct_dag': different})
            if different:
                require(all(sha(d / name / 'model.go') != sha(d / other / 'model.go') for other in chosen),
                        f'{work}/{name}: same generated program despite a different selected DAG')
                chosen[name] = a
                arms.append(name)
            return a
        precision = attempt('precision', budgets[0])
        require('precision' in chosen, f'{work}: precision contract did not change the emitted program')
        precision_witness = 'precision'
        def changed_route(a):
            before, after = (sorted(audit['backend'], key=lambda r: r['site']) for audit in (loose, a))
            return before[1]['Level'] != after[1]['Level'] and before[1]['Terms'] != after[1]['Terms']
        if work == 'residual':
            for i, budget in enumerate(budgets[1:]):
                if changed_route(precision):
                    break
                name = f'precision_{i}'
                precision = attempt(name, budget)
                if name in chosen and changed_route(precision):
                    precision_witness = name
            require(changed_route(precision), 'residual: narrowing precision alone did not change the second route and actual level')

        # For each represented template try level/error contracts targeting up
        # to two distinct DAGs. Do not force an ID, edit the frontier, or claim
        # that a template omitted by the finite profile executed on ciphertexts.
        tried = {(0, TOLERANCE), *((x['min_output_level'], x['max_approx_error']) for x in attempted)}
        for template in sorted({c['template'] for c in candidates}):
            in_template = [c for c in candidates if c['template'] == template]
            for c in sorted(in_template, key=lambda c: (c['delta_level'], c['activation_error'], c['candidate_id'])):
                selected = {dag_id(a['polynomials'][0]['selected']) for a in chosen.values()
                            if a['polynomials'][0]['selected']['template'] == template}
                if len(selected) >= 2 or dag_id(c) in selected:
                    continue
                budget = levels[c['candidate_id']], upper(predictions[c['candidate_id']])
                if budget not in tried:
                    attempt(f'coverage_{len(attempted)}', budget)
                    tried.add(budget)
        if self.cap is not None and work == 'reuse64':
            self.plan(d, 'rotation_cap', self.import_ckks(work, d, self.cap), 0, TOLERANCE)
            arms.append('rotation_cap')
        coverage = {t: {'in_frontier': sorted({c['candidate_id'] for c in site['frontier'] if c['template'] == t}),
                         'planned_for_encryption': [a for a in chosen if chosen[a]['polynomials'][0]['selected']['template'] == t]}
                    for t in sorted({c['template'] for c in site['frontier']})}
        save(d / 'budget-search.json', {'attempts': attempted, 'coverage': coverage, 'precision_witness': precision_witness})
        before = load(d / 'economy/contract.json')
        after = load(d / precision_witness / 'contract.json')
        require({k for k in before if before[k] != after[k]} == {'max_approx_error'}, 'confounded precision-only comparison')
        witness = chosen[precision_witness]
        def decision(a):
            p = a['polynomials'][0]
            return {'candidate_id': p['selected']['candidate_id'], 'template': p['selected']['template'],
                    'dag_sha256': dag_id(p['selected']), 'delta_level': p['selected']['delta_level'],
                    'linear_input_levels': [r['Level'] for r in sorted(a['backend'], key=lambda r: r['site'])],
                    'linear_terms': route_terms(a), 'keys': a['keys'], 'galois_bytes': a['key_bytes'],
                    'joint_objective': a['joint_objective'], 'sink': a['sinks'][0]}
        save(d / 'precision-comparison.json', {'changed_option': 'max-output-error', 'import_sha256': sha(imported),
             'economy_contract': before, 'precision_contract': after, 'economy': decision(loose),
             'precision': decision(witness), 'witness_arm': precision_witness,
             'changed_second_route_and_level': changed_route(witness)})
        return {'arms': arms, 'precision_witness': precision_witness, 'width': oracle.width,
                'source_sha256': sha(d / 'input.mlir'), 'corpus_sha256': sha(d / 'corpus.json'),
                'key_units_per_mib': RESIDUAL_KEY_PRICE if work == 'residual' else self.costs['key_units_per_mib'],
                'coverage': coverage}

    def composite_coverage(self):
        d = self.out / COVERAGE
        oracle = self.source(COVERAGE, d)
        imported = self.import_ckks(COVERAGE, d)
        free = self.plan(d, 'free', imported, 0, TOLERANCE)
        forced = self.plan(d, 'forced_composite', imported, 0, TOLERANCE, 'silu_tanh_rational')
        arms = ['heir', 'free', 'forced_composite']
        poly, sink = forced['polynomials'][0], forced['sinks'][0]
        suffix = poly['output_level'] - sink['level']
        errors = sorted({c['activation_error'] * oracle.output_norm for c in poly['frontier']
                         if c['template'] == 'silu_tanh_rational' and poly['input_level'] - c['delta_level'] >= suffix})
        if errors and errors[0] < sink['approx_error'] * 0.99:
            budget = math.sqrt(errors[0] * next((e for e in errors if e > errors[0]), sink['approx_error']))
            other = self.plan(d, 'forced_composite_tight', imported, 0, budget, 'silu_tanh_rational')
            require(dag_id(other['polynomials'][0]['selected']) != dag_id(poly['selected']) and
                    sha(d / 'forced_composite/model.go') != sha(d / 'forced_composite_tight/model.go'), 'same forced composite program twice')
            arms.append('forced_composite_tight')
        return {'arms': arms, 'width': oracle.width, 'source_sha256': sha(d / 'input.mlir'), 'corpus_sha256': sha(d / 'corpus.json'),
                'free_template': free['polynomials'][0]['selected']['template'], 'profile_sha256': sha(self.profile),
                'claim': 'extended-profile implementation coverage; forced arms are not free selection or a security result'}


def export(out):
    manifest = load(out / 'manifest.json')
    require(load(out / 'protocol.json') == PROTOCOL and set(manifest) == set(WORKLOADS), 'incomplete experiment protocol/workloads')
    result = {}
    extra = load(out / 'coverage-manifest.json')
    require(set(extra) == ({COVERAGE} if needs_composed_coverage(manifest) else set()), 'missing or unexpected additional template coverage')
    for work, info in {**manifest, **extra}.items():
        d = out / work
        require(sha(d / 'input.mlir') == info['source_sha256'] and sha(d / 'corpus.json') == info['corpus_sha256'],
                f'{work}: inputs changed since planning')
        oracle, corpus = Oracle((d / 'input.mlir').read_text()), load(d / 'corpus.json')
        require(info['width'] == oracle.width and corpus == oracle.corpus(), f'{work}: corpus does not replay from the source')
        names = info['arms']
        required = {'heir', 'economy', info['precision_witness']} if work in WORKLOADS else {'heir', 'free', 'forced_composite', 'forced_composite_tight'}
        require(names and names[0] == 'heir' and len(names) == len(set(names)) and required <= set(names),
                f'{work}: missing or duplicated comparator or encryption arm')
        baseline = None
        reports = {}
        for name in info['arms']:
            rows, stats = reduce_arm(d / name, oracle, corpus, baseline, defer_threshold=True)
            if name == 'heir':
                baseline = rows
            reports[name] = stats
        listed = sorted(p for p in d.rglob('*') if p.is_file())
        result[work] = {**info, 'arms': reports, 'files': {p.relative_to(out).as_posix(): sha(p) for p in listed}}
        if work in WORKLOADS:
            result[work]['precision_comparison'] = load(d / 'precision-comparison.json')
    verdict = suite_precision(result)
    coverage = {name: result.pop(name) for name in extra}
    ir = load(out / 'ir-tests.json')
    require(set(ir) == set(WORKLOADS), 'incomplete native IR regression evidence')
    save(out / 'evidence.json', {'protocol': PROTOCOL, 'workloads': result, 'additional_template_coverage': coverage,
                                 **verdict,
                                 'native_ir_tests': ir, 'toolchain': load(out / 'toolchain.json'),
                                 'commands_sha256': sha(out / 'commands.jsonl')})
    print(f'Numerical failures: {verdict["numerical_failures"]}; blocking: {verdict["blocking_numerical_failures"]}', flush=True)
    require(verdict['regression_gates_passed'], f'hard numerical regression: {verdict["blocking_numerical_failures"]}; per-arm evidence retained')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    build = subs.add_parser('prepare')
    for name in ('root', 'out', 'heir-bin', 'snabf-opt', 'goemit'):
        build.add_argument('--' + name, type=Path, required=True)
    build.add_argument('--extra-rotation-cap', type=int, default=None, help='optional additional reuse64 arm only')
    subs.add_parser('export').add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'export':
        export(args.out.resolve())
        return
    p = Pipeline(args)
    p.out.mkdir(parents=True, exist_ok=True)
    require(not any((p.out / w).exists() for w in WORKLOADS), 'use a fresh run directory; refusing to mix results')
    save(p.out / 'protocol.json', PROTOCOL)
    save(p.out / 'toolchain.json', {str(f): sha(f) for f in (p.opt[0], p.heir, p.translate, p.goemit, p.profile, p.fixtures / 'pass1_costs.json')})
    manifest = {work: p.workload(work) for work in WORKLOADS}
    save(p.out / 'manifest.json', manifest)
    extra = {COVERAGE: Pipeline(args, p.root / 'test/pass2/extended_profile.json').composite_coverage()} if needs_composed_coverage(manifest) else {}
    save(p.out / 'coverage-manifest.json', extra)
    write_harness(p.root, p.out, {**manifest, **extra})


if __name__ == '__main__':
    main()
