"""Preregistered inputs, independent dense float64 oracle and evidence reduction."""

import hashlib
import json
import math
import random
import re
import statistics
import struct


TRIALS = 64
TOLERANCE = 2.0 ** -12
SEED = 314159
PROTOCOL = {
    'schema': 'snabf-p2-regression-v1', 'inputs_per_arm': TRIALS,
    'logical_slot_abs_tolerance': TOLERANCE, 'random_seed': SEED,
    'inputs': 'source f32: corners, structured signs, impulses, seeded uniform [-1,1]',
    'truth': 'float64 dense W0, exact reference h/(1+exp(-h)), optional h skip, dense W1',
    'acceptance': 'every logical decrypted slot vs true-SiLU float64; no averaging exemption',
    'numerical_gate': 'hard on all SNABF arms and all stock arms except residual/heir; that unmodified stock comparator retains the same threshold and reports all misses as FAIL',
    'structural_gate': 'finite values, exact logical width, corpus identity, physical input/output ABI and profile are mandatory for every arm',
    'timing': 'online evaluator only, nanoseconds; separate key/configure, prepare, encrypt and decrypt',
    'crypto_randomness': 'fresh Lattigo keys per program and encryption per input; not seeded',
    'budget_policy': 'use only native frontier, sink contracts and public weights before encryption; never tune against decrypted outputs',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_bytes())


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(',', ':')).encode()).hexdigest()


def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


class Oracle:
    """Weights/topology from original source, never packed masks, plans or Go."""

    def __init__(self, source):
        self.w = []
        for i in range(2):
            m = re.findall(rf'%w{i} = arith\.constant dense<(\[.*?\])> : tensor<(\d+)x(\d+)xf32>', source, re.S)
            require(len(m) == 1, f'expected one dense source matrix W{i}')
            dense, n, k = m[0]
            rows = [[f32(v) for v in row] for row in json.loads(dense)]
            require(len(rows) == int(n) == int(k) and all(len(row) == int(k) for row in rows), 'invalid source dimensions')
            self.w.append(rows)
        self.width = len(self.w[0])
        require(len(self.w[1]) == self.width and '%h = linalg.matvec ins(%w0, %x :' in source,
                'unsupported first source layer')
        self.residual = '%skip = arith.addf %h, %a :' in source
        self.cross_residual = '%post = arith.addf %y, %h :' in source
        output = 'skip' if self.residual else 'a'
        result = 'post' if self.cross_residual else 'y'
        require(f'%y = linalg.matvec ins(%w1, %{output} :' in source and f'return %{result} :' in source,
                'unknown oracle source topology')
        self.output_norm = max(math.fsum(map(abs, row)) for row in self.w[1])

    @staticmethod
    def matvec(matrix, vector):
        return [math.fsum(w * x for w, x in zip(row, vector, strict=True)) for row in matrix]

    def suffix(self, h, a):
        live = [x + y for x, y in zip(h, a, strict=True)] if self.residual else a
        y = self.matvec(self.w[1], live)
        return [a + b for a, b in zip(y, h, strict=True)] if self.cross_residual else y

    def corpus(self):
        rng = random.Random(SEED)
        out = []
        for i in range(TRIALS):
            n = self.width
            if n == 4 and i < 16:
                x = [2 * ((i >> j) & 1) - 1 for j in range(n)]
            elif n == 64 and i < 4:
                x = [2 * (i if i < 2 else (j // (i - 1)) % 2) - 1 for j in range(n)]
            elif n == 64 and i < 16:
                x = [float(j == (5 * i) % n) for j in range(n)]
            else:
                x = [f32(2 * rng.random() - 1) for _ in range(n)]
            h = self.matvec(self.w[0], x)
            a = [v / (1 + math.exp(-v)) for v in h]
            out.append({'trial': i, 'x': x, 'hidden': h, 'truth': self.suffix(h, a)})
        return out


def eval_dag(candidate, x):
    """Interpret only the exported real-arithmetic DAG, not the native evaluator."""
    values = []
    for i, node in enumerate(candidate['nodes']):
        op = node['op']
        ids = node['args']
        require(all(isinstance(j, int) and 0 <= j < i for j in ids), 'non-topological candidate DAG')
        v = [values[j] for j in ids]
        if op == 'input' and i == 0 and not v:
            y = x
        elif op == 'mul' and len(v) == 2:
            y = v[0] * v[1]
        elif op == 'add' and v:
            y = node['constant'] + math.fsum(v)
        elif op == 'linear' and v:
            y = node['constant'] + math.fsum(a * b for a, b in zip(v, node['weights'], strict=True))
        else:
            raise ValueError(f'unsupported candidate DAG node {i}: {op}')
        values.append(y)
    output = candidate['output']
    require(0 <= output < len(values), 'invalid DAG output')
    return values[output]


def residual_statistics(errors):
    require(errors and all(math.isfinite(e) for e in errors), 'invalid error samples')
    mse = math.fsum(e * e for e in errors) / len(errors)
    require(math.isfinite(mse), 'nonfinite squared error')
    return {'linf': max(map(abs, errors)), 'mse': mse}


def stock_abi(ckks):
    # Use the actual encryptor and evaluator signatures, independent of the
    # profile ceiling or any planner. Generic local-scope SSA names can vary.
    functions = re.split(r'(?m)^\s*(?="func\.func"\()', ckks)
    abi = {}
    for edge, suffix in (('input', 'encrypt__arg0'), ('output', 'preprocessed')):
        symbol = f'linear_silu_linear__{suffix}'
        bodies = [f for f in functions if f.startswith('"func.func"') and
                  f'sym_name = "{symbol}"' in f.split('\n', 1)[0]]
        require(len(bodies) == 1, f'expected one stock {symbol} function')
        returns = re.findall(r'(?m)^\s*"func\.return"\(%[\w.]+\) : \((tensor<1x!lwe\.lwe_ciphertext<[^\n]+)\) -> \(\)', bodies[0])
        require(len(returns) == 1, f'expected one packed stock {symbol} return')
        for key, attr in (('level', 'current'), ('log_scale', 'scaling_factor')):
            values = re.findall(rf'\b{attr} = (\d+)\b', returns[0])
            require(len(values) == 1, f'ambiguous stock {symbol} {attr}')
            abi[f'{edge}_{key}'] = int(values[0])
    return abi


def blocking_precision(workload, arm):
    return (workload, arm) != ('residual', 'heir')


def suite_precision(reports):
    failed, blocking = [], []
    for work, data in reports.items():
        for name, stats in data['arms'].items():
            gate = blocking_precision(work, name)
            require(stats['hard_precision_gate'] is gate, f'{work}/{name}: invalid numerical gate')
            if not stats['precision_passed']:
                failed.append(f'{work}/{name}')
                if gate:
                    blocking.append(f'{work}/{name}')
    return {'numerical_failures': failed, 'blocking_numerical_failures': blocking,
            'all_arms_meet_tolerance': not failed, 'regression_gates_passed': not blocking}


def reduce_arm(arm, oracle, corpus, baseline=None, *, defer_threshold=False):
    with (arm / 'trials.jsonl').open(encoding='utf-8') as f:
        rows = [json.loads(line) for line in f]
    require(len(rows) == TRIALS, f'{arm}: expected {TRIALS} trials, found {len(rows)}')
    runtime = load(arm / 'runtime.json')
    require(runtime['corpus_sha256'] == sha(arm.parent / 'corpus.json') and
            runtime['program_sha256'] == sha(arm / 'model.go'), f'{arm}: stale measurements')
    has_audit = (arm / 'audit.json').is_file()
    require(has_audit == (arm.name != 'heir'), f'{arm}: missing or misplaced native audit')
    audit = load(arm / 'audit.json') if has_audit else None
    if has_audit:
        require(audit and len(audit['sinks']) == len(audit['polynomials']) == 1, f'{arm}: expected one sink and logical activation')
    profile = load(arm.parent / 'profile.json')
    require(all(runtime[k] == profile[k] for k in ('Q', 'P', 'logN')) and runtime['default_log_scale'] == profile['logDefaultScale'],
            f'{arm}: runtime parameters disagree with the frozen Q/P/ring/scale')
    abi = ({'input_level': len(profile['Q']) - 1, 'input_log_scale': profile['logDefaultScale'],
            'output_level': audit['sinks'][0]['level'], 'output_log_scale': audit['sinks'][0]['nominal_log_scale']} if audit else
           stock_abi((arm.parent / 'original-ckks.mlir').read_text()))
    require(load(arm / 'abi.json') == abi, f'{arm}: runtime test ABI differs from the emitted program')
    keys = runtime['galois_keys']
    ids = [k['galois'] for k in keys]
    require(ids == sorted(set(ids)) and all(k['serialized_bytes'] > 0 for k in keys) and runtime['relinearization_key_serialized_bytes'] > 0,
            f'{arm}: invalid runtime key inventory')
    require(runtime['configure_ns'] > 0 and runtime['preprocessing_ns'] > 0, f'{arm}: invalid setup timing')
    if audit:
        require(baseline is not None, f'{arm}: missing stock comparator')
        require(ids == sorted(k['galois'] for k in audit['keys']) and
                sum(k['serialized_bytes'] for k in keys) == audit['key_bytes'], f'{arm}: bound/physical key inventory mismatch')
    times, total, stock_err, approx_err, crypto_err = [], [], [], [], []
    over, first_failure = 0, None
    for i, (row, case) in enumerate(zip(rows, corpus, strict=True)):
        got, truth = row['decrypted'], case['truth']
        require(row['trial'] == case['trial'] == i and len(got) == len(truth) == oracle.width,
                f'{arm}: trial or packing mismatch at {i}')
        require(0 <= row['output_level'] <= row['input_level'] < len(runtime['Q']) and
                all(math.isfinite(row[k]) for k in ('input_log_scale', 'output_log_scale')),
                f'{arm}: invalid measured ciphertext metadata')
        require(all(row[k] > 0 for k in ('encrypt_ns', 'eval_ns', 'decrypt_ns')), f'{arm}: invalid timing')
        for edge in ('input', 'output'):
            require(row[edge+'_level'] == abi[edge+'_level'] and abs(row[edge+'_log_scale'] - abi[edge+'_log_scale']) <= 0.25,
                    f'{arm}/{i}: actual {edge} L{row[edge+"_level"]}/logscale {row[edge+"_log_scale"]} vs ABI {abi}')
        if audit:
            site = audit['polynomials'][0]
            ypoly = oracle.suffix(case['hidden'], [eval_dag(site['selected'], h) for h in case['hidden']])
            approx_err.extend(p - q for p, q in zip(ypoly, truth, strict=True))
            crypto_err.extend(v - p for v, p in zip(got, ypoly, strict=True))
        for j, (v, q) in enumerate(zip(got, truth, strict=True)):
            e = v - q
            require(math.isfinite(e), f'{arm}: nonfinite output or truth in trial {i} slot {j}')
            if abs(e) > TOLERANCE:
                over += 1
                if first_failure is None:
                    first_failure = {'trial': i, 'slot': j, 'decrypted': v, 'truth': q, 'abs_error': abs(e)}
            total.append(e)
        if baseline is not None:
            stock_err.extend(v - s for v, s in zip(got, baseline[i]['decrypted'], strict=True))
        times.append(row['eval_ns'])
    gate = blocking_precision(arm.parent.name, arm.name)
    stats = {'trials': len(rows), 'logical_slots_checked': len(total), 'decrypted_vs_true': residual_statistics(total),
             'declared_abi': abi, 'slots_over_tolerance': over, 'precision_passed': over == 0,
             'accuracy_status': 'FAIL' if over else 'PASS', 'hard_precision_gate': gate,
             'tolerance': TOLERANCE, 'first_failure': first_failure,
             'eval_ns': {'min': min(times), 'median': statistics.median(times), 'mean': statistics.mean(times), 'max': max(times)},
             'level_pairs': sorted(set((r['input_level'], r['output_level']) for r in rows)),
             'actual_output_log_scale_range': [min(r['output_log_scale'] for r in rows), max(r['output_log_scale'] for r in rows)],
             'runtime': runtime, 'trial_file_sha256': sha(arm / 'trials.jsonl')}
    if audit:
        stats.update({'real_dag_vs_true': residual_statistics(approx_err),
                      'decrypted_vs_real_dag': residual_statistics(crypto_err),
                      'decrypted_vs_stock': residual_statistics(stock_err), 'audit_sha256': sha(arm / 'audit.json')})
    run_check = load(arm / 'accuracy.json')
    require(run_check['slots_over_tolerance'] == over and run_check['logical_slots_checked'] == len(total) and
            run_check['precision_passed'] is (over == 0) and run_check['hard_precision_gate'] is gate and
            run_check['accuracy_status'] == stats['accuracy_status'] and run_check['first_failure'] == first_failure and
            run_check['tolerance'] == TOLERANCE and run_check['linf'] == stats['decrypted_vs_true']['linf'] and
            math.isclose(run_check['sum_squared_error'], stats['decrypted_vs_true']['mse'] * len(total), rel_tol=1e-12) and
            math.isclose(run_check['mse'], stats['decrypted_vs_true']['mse'], rel_tol=1e-12),
            f'{arm}: Go and independent evidence reduction disagree on slot errors')
    save(arm / 'metrics.json', stats)
    if not defer_threshold:
        require(not (gate and over), f'{arm}: {over}/{len(total)} slots exceed {TOLERANCE}')
    return rows, stats
