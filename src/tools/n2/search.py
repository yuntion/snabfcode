"""Native N2 proposal matrix. Never seals, emits, or accepts a ciphertext graph."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from security_cache import ERROR, MODEL, PIN, SECRET, checked_attacks, digest, measure, request


def need(ok, why):
    if not ok:
        raise ValueError(why)


def store(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def estimated(req, costs):
    return dict(request=req, attacks=checked_attacks(costs[digest(req)], req))


def search_envelope(n, target, oracle):
    """Only a directly measured power of two can be returned. No basis argument."""
    need(type(n) is int and n in (16384, 32768, 65536) and type(target) is int and target == 128,
         'unsupported N/security target')
    # Probe the bounded policy grid [N/128,N/32] with step N/512. Bisection
    # is a search heuristic, not a proof of the mathematical maximum. Even
    # a nonmonotone estimator cannot authorize an unmeasured cap.
    step = n // 512
    lo, hi, selected, probes = 4, 16, None, []
    while lo <= hi:
        m = (lo + hi) // 2
        bits = m * step
        req = request('n2.search-envelope', n, [], [], 1 << bits)
        row = estimated(req, oracle([req]))
        probes.append(dict(bits=bits, **row))
        if min(row['attacks'].values()) >= target:
            selected = bits
            lo = m + 1
        else:
            hi = m - 1
    need(selected is not None, 'no measured 128-bit envelope on the policy grid')
    return dict(schema='snabf.n2.search-envelope.v1', ring=n, target_bits=target, cap_bits=selected,
                min_grid=4*step, max_grid=16*step, grid_step=step, probes=probes)


def ntt_prime(v, n):
    if type(v) is not int or not 3 <= v < (1 << 60) or v % (2*n) != 1:
        return False
    t, s = v-1, 0
    while t % 2 == 0:
        t //= 2
        s += 1
    for base in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if base % v == 0:
            continue
        a = pow(base, t, v)
        if a in (1, v-1):
            continue
        for _ in range(s-1):
            a = a*a % v
            if a == v-1:
                break
        else:
            return False
    return True


def check_proposal(plan, expected):
    need(plan.get('schema') == 'snabf.n2.native-proposal.v1' and all(
        type(plan.get(k)) is type(v) and plan[k] == v for k, v in expected.items()), 'wrong proposal binding')
    need(re.fullmatch('[0-9a-f]{64}', plan['graph_sha256']) and plan['max_k'] == 0 and
         plan['heir_scale_anchor'] is False and plan['wide_prime_bits'] == [24,59] and plan['margin_bits'] == 2,
         'wrong graph or width/K policy')
    n, q, p = plan['ring'], plan['q'], plan['p']
    need(isinstance(q, list) and isinstance(p, list) and len(q) == plan['q_count'] >= 2 and
         len(p) == plan['alpha'] in (1,2) and all(ntt_prime(x, n) for x in q+p) and len(set(q+p)) == len(q+p),
         'invalid concrete ordered NTT basis')
    P = math.prod(p)
    need(math.prod(q)*P <= 1 << plan['security_cap'], 'actual QP exceeds measured envelope')
    need(all(P >= 4*math.prod(q[i:i+len(p)]) for i in range(0,len(q),len(p))), 'actual block margin')
    need(type(plan['scale']) is int and plan['min_scale'] <= plan['scale'] <= plan['max_scale'] and
         len(plan['k']) == plan['edges'] > 0 and all(type(k) is int and k == 0 for k in plan['k']),
         'unexpected scale or truncation')
    errors = [plan[k] for k in ('arithmetic','approximation')]
    need(all(type(v) in (float,int) and math.isfinite(v) and v >= 0 for v in errors) and
         sum(errors) <= plan['tolerance'] == 2**-12, 'concrete engineering error exceeds budget')


def concrete_screen(plan, expected, oracle):
    check_proposal(plan, expected)
    n, q, p = plan['ring'], plan['q'], plan['p']
    Q = math.prod(q)
    reqs = {k:request('source.'+k, n, q, p, v) for k,v in
            [('Q', Q), ('QP', Q*math.prod(p)), ('envelope', 1 << plan['security_cap'])]}
    costs = oracle(list(reqs.values()))
    rows = {k:estimated(req,costs) for k,req in reqs.items()}
    minimum = min(min(r['attacks'].values()) for r in rows.values())
    return dict(instances=rows, ordered_q=q, ordered_p=p, minimum_bits=minimum, meets_128=minimum>=128)


def run_matrix(args):
    n = 1 << args.log_n
    need(1 <= args.slots <= n//2 and args.slots & (args.slots-1) == 0 and
         2 <= args.q_count <= 32 and 24 <= args.min_scale <= args.max_scale <= 52, 'invalid search bounds')
    exe, source, spec = (p.resolve(strict=True) for p in (args.proposer,args.mlir,args.input_spec))
    out = args.out.resolve(); out.mkdir(mode=0o700, exist_ok=False)
    def oracle(requests):
        return measure(requests, args.cache, args.estimator)
    envelope = search_envelope(n, 128, oracle)
    cap = envelope['cap_bits']; store(out/'envelope.json',envelope)
    common = dict(source=sha(source), input_spec=sha(spec), ring=n, encoded_slots=args.slots, q_count=args.q_count,
                  min_scale=args.min_scale,max_scale=args.max_scale, security_cap=cap)
    cells = []
    for alpha in (1,2):
        for mode in ('narrow','wide'):
            site = out/f'a{alpha}-{mode}'; site.mkdir(mode=0o700)
            cmd = list(map(str,[exe,source,spec,args.log_n,args.slots,args.q_count,args.min_scale,args.max_scale,alpha,mode,cap]))
            with (site/'proposal.json').open('xb') as f, (site/'diagnostics.log').open('xb') as log:
                proc = subprocess.run(cmd, stdout=f, stderr=log, check=False, cwd=site)
            cell = dict(alpha=alpha,scale_mode=mode,argv=cmd,returncode=proc.returncode,
                        result='native_search_failed',proposal=False,concrete_screen=False,ciphertext_run=False)
            if proc.returncode == 0:
                plan = json.loads((site/'proposal.json').read_bytes())
                screen = concrete_screen(plan,dict(common,alpha=alpha,scale_mode=mode),oracle)
                store(site/'security.json',screen)
                q = plan['q'][1:]
                cell.update(result='screened_proposal' if screen['meets_128'] else 'concrete_security_rejected',
                            proposal=True,concrete_screen=screen['meets_128'],graph=plan['graph_sha256'],
                            ordered_basis_sha256=digest([plan['q'],plan['p']]),scale=plan['scale'],
                            nonbase_q_spread_bits=math.log2(max(q)/min(q)),minimum_bits=screen['minimum_bits'],
                            source_bound=plan['approximation']+plan['arithmetic'],proposal_sha256=sha(site/'proposal.json'),
                            security_sha256=sha(site/'security.json'))
            store(site/'cell.json',cell); cells.append(cell)
            print(alpha,mode,cell['result'],cell.get('nonbase_q_spread_bits'),flush=True)
    same = all(c['proposal'] for c in cells) and len({c['graph'] for c in cells}) == 1
    ok = same and all(c['concrete_screen'] for c in cells)
    store(out/'matrix.json',dict(schema='snabf.n2.native-matrix.v1',common=common,binary_sha256=sha(exe),
          envelope_sha256=sha(out/'envelope.json'),cells=cells,same_arithmetic_graph=same,all_four_screened=ok,
          product_checker_run=False,ciphertext_run=False,estimator=PIN,model=MODEL,secret=SECRET,error=ERROR))
    return ok


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for flag in ('proposer','mlir','input-spec','out','cache','estimator'):
        ap.add_argument('--'+flag,required=True,type=Path)
    ap.add_argument('--log-n',type=int,choices=(14,15,16),required=True)
    ap.add_argument('--slots',type=int,required=True)
    ap.add_argument('--q-count',type=int,required=True)
    ap.add_argument('--min-scale',type=int,default=36); ap.add_argument('--max-scale',type=int,default=46)
    sys.exit(0 if run_matrix(ap.parse_args()) else 1)
