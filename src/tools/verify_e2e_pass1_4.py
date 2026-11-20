"""Independently audit one E2E run, or the preregistered reuse64 128/512 pair."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import statistics
import sys

# The independent oracle is stdlib-only. Do not initialize the HEIR front end
# (or require its Python bindings) just to read a saved evidence directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'frontend' / 'snabf_frontend'))
from pass2_evidence import Oracle, TOLERANCE, TRIALS, load, require, save, sha
from e2e_matrix_evidence import paired, scenario_contract


def seals(run, program):
    seal = program['emit_sha256']
    require(program['emitter'] == 'snabf.direct-go.v1' and re.fullmatch('[0-9a-f]{64}', seal), 'invalid emitter receipt')
    def ir_seals(name):
        return re.findall(r'snabf\.emit_sha256 = "([0-9a-f]{64})"', (run/'work'/name).read_text())
    for f in ('checked.mlir', 'lowered.mlir'):
        require(ir_seals(f) == [seal], 'missing/conflicting checker receipt in '+f)
    code = (run/'out/model.go').read_text()
    require(re.findall(r'^const PlanSHA256 = "([0-9a-f]{64})"$', code, re.M) == [seal], 'Go receipt mismatch')
    return {'precheck_receipts': ir_seals('planned.mlir'), 'emit_sha256': seal}


def keys(program, search, observed):
    epochs = {e['n']: e for e in program['epochs']}
    src, dst = program['source_ring'], program['target_ring']
    mixed = src != dst
    require(set(epochs) == {src, dst}, 'epoch set mismatch')
    expected = {(n, 'galois', k) for n,e in epochs.items() for k in e['galois']}
    expected.update((i['n'], 'relin', 0) for i in program['instructions'] if i['op'] == 'relin')
    expected.add((src, 'public', 0))
    if mixed:
        expected.add((src, 'down', 0))
    inventory = Counter((k['N'], k['Role'], k['Galois']) for k in observed)
    require(inventory == Counter(expected), 'measured key inventory disagrees with program')
    sizes = Counter()
    for k in observed:
        e, role = epochs[k['N']], k['Role']
        require(k['Bytes'] > 0 and k['KeyID'] == e['key_id'], 'invalid measured key or domain')
        if role == 'down':
            c = search['conversion']
            require((k['LevelQ'], k['LevelP'], k['Bytes'], k['TargetKeyID']) ==
                    (c['switch_level_q'], c['switch_level_p'], c['switch_bytes'], epochs[dst]['key_id']), 'conversion key mismatch')
        else:
            require((k['LevelQ'], k['LevelP']) == (len(e['q'])-1, len(e['p'])-1), 'measured key level')
            if role == 'galois':
                require(k['Bytes'] == e['key_bytes'], 'measured Galois key size')
        sizes[role] += k['Bytes']
    require(sizes['galois'] == program['galois_bytes'] and sizes['down'] == program.get('switch_bytes', 0), 'serialized key totals')
    return {r:sizes[r] for r in ('galois', 'down', 'relin', 'public')}


def measured(run, corpus, program, metrics):
    width, slots = program['width'], program['slots']
    require(0 < width <= slots and slots % width == 0 and len(corpus) == TRIALS, 'corpus/period dimensions')
    sink = program['instructions'][program['sink']]
    require(sink['n'] == program['target_ring'], 'wrong sink domain')
    count = misses = 0
    peak = 0.
    times, scales, ss = [], [], []
    with (run/'out/trials.jsonl').open() as stream:
        for i,(line, case) in enumerate(zip(stream, corpus, strict=True)):
            row = json.loads(line)
            truth, got = case['truth'], row['decoded64']
            require(i == row['trial'] == case['trial'] and len(truth) == width and
                    row['truth_one_period'] == truth and len(got) == slots, f'trial {i}: order/slot coverage')
            require(row['ring'] == sink['n'] and row['level'] == sink['Level'] and
                    math.isclose(row['log_scale'], sink['physical_log_scale'], rel_tol=0, abs_tol=1e-7), f'trial {i}: measured ABI')
            err = [abs(y - truth[j % width]) for j,y in enumerate(got)]
            require(all(map(math.isfinite, err)), f'trial {i}: nonfinite slot error')
            count += slots
            misses += sum(e > TOLERANCE for e in err)
            ss.append(math.fsum(e*e for e in err))
            peak = max(peak, max(err))
            require(type(row['online_ns']) is int and row['online_ns'] > 0, 'invalid Eval timing')
            times.append(row['online_ns']/1e6)
            scales.append(row['log_scale'])
    mse = math.fsum(ss)/count
    require((metrics['trials'],metrics['logical_outputs'],metrics['physical_encoded_slots_checked'],metrics['misses']) ==
            (len(times),width*TRIALS,count,misses) and len(times) == TRIALS, 'runner totals do not replay')
    require((metrics['actual_output_ring'],metrics['actual_level'],metrics['actual_log_scale']) ==
            (sink['n'],sink['Level'],scales[-1]), 'runner output ABI does not replay')
    require(peak == metrics['linf'] and math.isclose(mse,metrics['mse'],rel_tol=1e-9) and
            math.isclose(statistics.mean(times),metrics['online_ms'],rel_tol=1e-12), 'runner norms or timing do not replay')
    require(misses == 0, f'{misses}/{count} physical slots exceed {TOLERANCE}, max={peak}')
    return dict(trials=len(times), logical_outputs=width*len(times), slots_per_trial=slots, physical_slots=count,
                misses=misses, linf=peak, mse=mse, tolerance=TOLERANCE, actual_level=sink['Level'], actual_ring=sink['n'],
                actual_log_scale_range=[min(scales),max(scales)], online_mean_ms=statistics.mean(times), online_median_ms=statistics.median(times))


def verify(run, *, oracle_witness=None):
    inputs, out = run/'inputs', run/'out'
    config = dict(line.split('=',1) for line in (run/'env.snapshot').read_text().splitlines())
    files = {'source.mlir','estimates.json','corpus.json','security.json','profile.json'}
    require(config['MODE'] in ('source','pass3'), 'unknown input mode')
    if config['MODE'] == 'pass3':
        files.add('checkpoint.mlir')
    if (inputs/'pass3-options.json').is_file():
        files.add('pass3-options.json')
    locked = {f:sha(inputs/f) for f in sorted(files)}
    manifest = dict(line.split('  ',1)[::-1] for line in (run/'inputs.sha256').read_text().splitlines())
    require(manifest == {'inputs/'+f:h for f,h in locked.items()}, 'locked inputs differ from manifest')
    corpus = load(inputs/'corpus.json')
    oracle = Oracle((inputs/'source.mlir').read_text())
    if oracle_witness is None:
        require(oracle.corpus() == corpus, 'source oracle differs from frozen corpus')
    else:
        from e2e_oracle_witness import replay
        replay(oracle,corpus,locked,oracle_witness)
    program, search = load(out/'program.json'), load(out/'search.json')
    receipt = seals(run, program)
    require(program['ring_search'] == {k:v for k,v in search.items() if k != 'conversion'}, 'search and emitter disagree')
    profile, sec = load(inputs/'profile.json'), load(inputs/'security.json')
    src, dst = program['source_ring'], program['target_ring']
    require(dst == search['target'] and (src != dst) == (search['outcome'] == 'ring_down'), 'planned ring decision')
    epoch = next(e for e in program['epochs'] if e['n'] == src)
    require((src,epoch['q'],epoch['p'],epoch['scale']) == (1<<profile['logN'],profile['Q'],profile['P'],profile['logDefaultScale']), 'source concrete profile mismatch')
    cache = load(inputs/'estimates.json')
    require((cache['source_ring'],cache['source']['q'],cache['source']['p']) == (src,epoch['q'],epoch['p']) and
            sec['ring'] == src and sec['estimator'] == cache['estimator'], 'frozen security domain mismatch')
    for label,mod in [('Q', math.prod(epoch['q'])),('QP',math.prod(epoch['q']+epoch['p']))]:
        r = sec['instances'][label]
        require(int(r['modulus']) == mod and min(r['attacks_log2_rop'].values()) >= 128, 'source security proof input mismatch')
    policy = search['policy']
    require(policy['source_security_bits'] == policy['target_security_bits'] == 128 and
            policy['switching_key_security_bits'] >= max(128,int(config['SNABF_PASS4_SWITCHING_KEY_SECURITY_BITS'])), 'weakened security floor')
    mixed = src != dst
    require([d['role'] for d in search['key_domains']] == (['source','target','switching_key_ext'] if mixed else ['source']) and
            sum(i['op'] == 'ring_down' for i in program['instructions']) == int(mixed), 'execution/domain mismatch')
    if mixed:
        c, ext = search['conversion'], search['key_domains'][2]
        require(c == program['ring_decision'], 'conversion audit mismatch')
        for label,n in [('switch_QP_large_ring',src),('switch_QP_small_secret',dst)]:
            s = c['security'][label]
            require(s['n'] == n and int(s['modulus']) == math.prod(ext['q']+ext['p']) and
                    min(s['usvp'],s['dual_hybrid']) >= policy['switching_key_security_bits'], 'unverified switching-key domain')
    metrics = load(out/'metrics.json')
    name, work = config.get('SCENARIO','inference'),config.get('WORKLOAD','')
    scenario_contract(name,work,locked['source.mlir'],oracle,program,search)
    is_pair = name == 'equiv-single-ring'
    evidence = {f:sha(out/f) for f in ('search.json','program.json','model.go','metrics.json','trials.jsonl')}
    if is_pair:
        evidence.update({'legacy/'+f:sha(run/'work/legacy'/f) for f in ('model.go','audit.json','checked.mlir','translated.go.txt')})
    return dict(mode=config['MODE'], strategy=config['STRATEGY'], scenario=name,workload=work,decode_domain='target' if mixed else 'source',
                live_ciphertexts=len(search['baseline_cut']['live_values']), conversion_keys=sum(k['Role'] == 'down' for k in metrics['keys']),
                inputs=locked, policy=policy, outcome=search['outcome'],
                source_ring=src, target_ring=dst, reasons=[r['reason'] for r in search['rounds']], **receipt,
                checkpoint_sha256=sha(run/'work/checkpoint.mlir'), tools_sha256=sha(run/'toolchain.sha256'),
                observed_key_bytes=keys(program,search,metrics['keys']), measurements=(paired if is_pair else measured)(run,corpus,program,metrics),
                evidence_sha256=evidence)


def reuse64_pair(a, b, *, oracle_witness=None):
    x,y = verify(a,oracle_witness=oracle_witness),verify(b,oracle_witness=oracle_witness)
    require(a.resolve() != b.resolve() and x['mode'] == y['mode'] and x['strategy'] == y['strategy'] == 'adaptive', 'not independent adaptive runs')
    for field in ('inputs','tools_sha256'):
        require(x[field] == y[field], 'pair differs in '+field)
    if x['mode'] == 'pass3':
        require(x['checkpoint_sha256'] == y['checkpoint_sha256'], 'replay checkpoint differs')
        upstream = {'kind':'byte_identical_checkpoint', 'sha256':x['checkpoint_sha256']}
    else:
        p3 = sha(a/'work/pass3.json')
        require(p3 == sha(b/'work/pass3.json'), 'concrete Pass 3 audit differs')
        # External HEIR hashes Types/Attributes when naming a private client.pack.
        # Compare a copy in memory only. No normalized text is ever fed to a
        # compiler or used as a checker receipt. Any other difference must fail.
        def comparable(run):
            ir = (run/'work/checkpoint.mlir').read_text()
            names = re.findall(r'(?m)^  func\.func private @(_assign_layout_[0-9]+)\(\) attributes \{heir\.interface = \{[^\n]*?roles = \["client\.pack"\]', ir)
            require(len(names) == 1, 'unrecognized private HEIR pack helper')
            name = re.escape(names[0])
            ir = re.sub(r'(@)'+name+r'(?=\(\))', r'\1__paired_client_pack', ir)
            ir, n = re.subn(r'(snabf\.emit_sha256 = ")[a-f0-9]{64}(")',r'\1<per-run receipt>\2',ir)
            require(n == 1, 'missing Pass 3 checker receipt')
            return ir
        require(comparable(a) == comparable(b), 'upstream SSA differs beyond private client.pack naming and its receipt')
        upstream = {'kind':'independent_source_compilations', 'concrete_pass3_sha256':p3,
                    'checkpoint_comparison':'identical modulo private HEIR client.pack symbol and its checker receipt'}
    p,q = dict(x['policy']),dict(y['policy'])
    require((p.pop('switching_key_security_bits'),q.pop('switching_key_security_bits')) == (128,512) and p == q, 'unexpected policy delta')
    require((x['source_ring'],x['target_ring'],y['source_ring'],y['target_ring']) == (16384,8192,16384,16384) and
            x['outcome'] == 'ring_down' and y['outcome'] == 'retain_source_ring', 'expected reuse64 positive and source-ring fallback')
    rejects = [r for r in load(b/'out/search.json')['rounds'] if r['reason'] == 'switching_key_ext_modulus_security_rejected']
    def bits(k): return min(k['usvp'],k['dual_hybrid'])
    require(any(bits(r['security']['switch_QP_small_secret']) < 512 and bits(r['security']['QP']) >= 128 for r in rejects), 'not a conversion-specific refusal')
    return dict(positive_run=str(a),strict_run=str(b), only_policy_change='switching_key_security_bits:128->512', upstream=upstream,positive=x,strict=y)


def c1_selected(path, case):
    """Read the selected field only, never mistake a frontier alternative for the winner."""
    def integer(text,key): return int(re.search(r'\b'+key+r' = (\d+) : i64',text)[1])
    def string(text,key): return re.search(r'\b'+key+r' = "([^"]+)"',text)[1]
    def arr(text,key,cast):
        raw=re.search(r'\b'+key+r' = array<[if]64(?:: ([^>]+))?>',text)[1]
        return list(map(cast,raw.split(', '))) if raw else []
    def records(text,key):
        return re.findall(r'\{([^{}]*)\}',re.search(r'\b'+key+r' = \[([^\[\]]+)\]',text)[1])
    sites=[]; linear={}
    for line in path.read_text().splitlines():
        lin=re.match(r'^\s*(%[\w.]+) = "snabf\.lintrans"\(',line)
        if lin:
            linear[lin[1]]=f"lin{integer(line,'site')}.rescale"
        if '"snabf.poly_eval"(' not in line: continue
        operand=re.search(r'"snabf\.poly_eval"\((%[\w.]+)\)',line)
        require(operand and operand[1] in linear,'C1 activation is not bound to a checked linear result')
        prefix,sep,rec = line.partition('poly_selected = '); require(sep,'missing selected DAG')
        kind=string(prefix,'activation'); template=string(rec,'template')
        require((kind=='gelu' and template=='gelu_direct') or (kind=='silu' and template in ('silu_direct','silu_tanh_rational')),'template/source mismatch')
        stages=records(rec,'stages'); ns=records(rec,'nodes'); work=records(rec,'work_by_consumed_depth')
        depth=integer(rec,'delta_level'); out=integer(rec,'output')
        require(out==len(ns)-1 and integer(ns[out],'depth')==depth and len(work)==depth+1,'invalid DAG/work depth')
        if kind=='gelu': require(len(stages)==1 and string(stages[0],'reference')=='half_erf_sqrt2','not the direct erf stage')
        for stage in stages:
            ideal,actual,hull=[arr(stage,k,float) for k in ('i_in','a_in','j_in')]
            coeffs=arr(stage,'coefficients',float)
            require(hull==[min(ideal[0],actual[0]),max(ideal[1],actual[1])] and len(coeffs)==integer(stage,'degree')+1 and
                    all(map(math.isfinite,coeffs)) and all(x==0 for x in coeffs[::2]), 'invalid hull or odd binary64 kernel')
        half,mul=arr(ns[out],'args',int); core=integer(stages[-1],'output_node')
        require(string(ns[out],'op')=='add' and string(ns[half],'op')=='linear' and arr(ns[half],'args',int)==[0] and
                arr(ns[half],'weights',float)==[0.5] and string(ns[mul],'op')=='mul' and arr(ns[mul],'args',int)==[0,core] and
                all(float(re.search(r'\bconstant = (\S+) : f64',ns[n])[1])==0 for n in (half,mul,out)), 'not x/2 + x*core')
        # Bind the result of the selected odd stage, not any intermediate with that node number.
        core_op=string(ns[core],'op'); core_suffix={'linear':'rescale','mul':'rescale','add':'add'}
        require(core_op in core_suffix,'unsupported selected core result')
        end='constant' if float(re.search(r'\bconstant = (\S+) : f64',ns[core])[1]) else core_suffix[core_op]
        err=float(re.search(r'activation_error = ([^ ]+) : f64',rec)[1]); require(0 <= err <= TOLERANCE,'uncertified activation')
        sites.append(dict(activation=kind,candidate=string(rec,'candidate_id'),depth=depth,activation_error=err,
                          half=half,product=mul,core=core,output=out,input_edge=linear[operand[1]],core_end=end))
    require([s['activation'] for s in sites]==(['silu','gelu'] if case=='mixed' else ['gelu']), 'incorrect source activation sequence')
    return sites


def c1_reconstruction(program, sites):
    """Bind the checked x/2 + x*core DAG to its concrete DirectGo operands."""
    edges=program['instructions']; by_id={e['ID']:e for e in edges}
    require(len(by_id)==len(edges) and len(program['routes'])==len(sites)+1,'missing/duplicate C1 graph stage')
    qs={e['n']:e['q'] for e in program['epochs']}
    for i,e in enumerate(edges):
        require(all(type(j) is int and 0<=j<i for j in e['inputs']),f"{e['ID']}: non-topological C1 operand")
        require(e['n']==e['kernel_ring']==program['source_ring']==program['target_ring'] and e['n'] in qs and
                type(e['Level']) is int and 0<=e['Level']<len(qs[e['n']]) and
                math.isfinite(e['physical_log_scale']) and e['physical_log_scale']>0,f"{e['ID']}: invalid C1 ring, level or scale")

    def typed(e, op, arity):
        require(e['op']==op and len(e['inputs'])==arity,f"{e['ID']}: expected {op} with {arity} operand(s)")
        return e

    def node(name,op,arity):
        require(name in by_id,f'{name}: missing concrete C1 edge')
        return typed(by_id[name],op,arity)

    def parent(e,k=0): return edges[e['inputs'][k]]
    def origin(e):
        # Only DirectGo DropLevel preserves the value AND physical scale. Never skip rescale/relin.
        while e['op']=='drop':
            a=parent(typed(e,'drop',1))
            require(e['Level']<a['Level'] and e['physical_log_scale']==a['physical_log_scale'],
                    f"{e['ID']}: drop must lower level and preserve physical scale")
            e=a
        return e

    # The native scale replay allows 1e-8 log2 units when reconciling floating-point arithmetic.
    def close(x,y): return math.isclose(x,y,rel_tol=0,abs_tol=1e-8)
    def rescale(e,a):
        require(parent(e) is a and e['Level']+1==a['Level'] and
                close(e['physical_log_scale'],a['physical_log_scale']-math.log2(qs[a['n']][a['Level']])),
                f"{e['ID']}: rescale operand, level or Q-based scale mismatch")

    for i,s in enumerate(sites):
        def part(n,end,op,arity): return node(f'poly{i}.n{n}.{end}',op,arity)
        x=node(s['input_edge'],'rescale',1); rescale(x,parent(x))
        half=part(s['half'],'pt0','coefficient',1); half_rs=part(s['half'],'rescale','rescale',1)
        mul=part(s['product'],'mul','mul',2); relin=part(s['product'],'relin','relin',1)
        prod_rs=part(s['product'],'rescale','rescale',1); out=part(s['output'],'add','add',2)
        a,b=parent(mul),parent(mul,1); core=origin(b); end=s['core_end']
        require(half['values']==[0.5] and half['Level']==parent(half)['Level'] and origin(parent(half)) is x and origin(a) is x,
                f"{half['ID']}/{mul['ID']}: coefficient or original x bypass mismatch")
        require(core['ID']==f"poly{i}.n{s['core']}.{end}" and core['op']==('add_plain' if end=='constant' else end),
                f"{mul['ID']}: product does not consume the selected core result")
        require(a['Level']==b['Level']==mul['Level'] and close(mul['physical_log_scale'],a['physical_log_scale']+b['physical_log_scale']),
                f"{mul['ID']}: unaligned product levels or scale")
        require(parent(relin) is mul and relin['Level']==mul['Level'] and relin['physical_log_scale']==mul['physical_log_scale'],
                f"{relin['ID']}: unbound relinearization")
        rescale(half_rs,half); rescale(prod_rs,relin)
        left,right=parent(out),parent(out,1)
        require(origin(left) is half_rs and origin(right) is prod_rs and left['Level']==right['Level']==out['Level'] and
                all(close(e['physical_log_scale'],out['physical_log_scale']) for e in (left,right)),
                f"{out['ID']}: unbound or unaligned final add")


def c1_verify(root, *, public_only=False, slot_archive=False):
    from gelu_witness import verify_public
    result={}
    for case in ('single','mixed'):
        d=root/case; corpus,w=verify_public(d); require(w['graph']==case,'mislabelled C1 source')
        summary=dict(trials=len(corpus),source_sha256=w['source_sha256'],witness_sha256=sha(d/'witness.json'),
                     erf_values=len(w['erf_hidden_over_sqrt2']),exp_values=len(w['exp_negative_hidden']))
        result[case]=summary
        if public_only: continue
        model=d/'model'; p=load(model/'program.json'); c=load(model/'compile.json'); work=model/'work'; edges=p['instructions']
        receipt=p['emit_sha256']; require(re.fullmatch('[a-f0-9]{64}',receipt) and c['emit_sha256']==receipt and
            p['emitter']=='snabf.direct-go.v1' and c['security_bits']==128 and c['mode']=='keep-source' and
            p['source_ring']==p['target_ring']==c['target_ring'] and c['precision_bits']==12, 'wrong C1 compile policy/receipt')
        for ir in (work/'checked.mlir',work/'lowered.mlir',d/'rechecked.mlir'):
            require(re.findall(r'snabf\.emit_sha256 = "([0-9a-f]{64})"',ir.read_text())==[receipt],'independent checker/IR receipt mismatch')
        require(re.findall(r'^const PlanSHA256 = "([a-f0-9]{64})"$',(model/'model.go').read_text(),re.M)==[receipt],'Go receipt mismatch')
        if not slot_archive:
            summary['observed_key_bytes']=keys(p,p['ring_search'],load(d/'keys.json'))
        sites=c1_selected(work/'checked.mlir',case); c1_reconstruction(p,sites)
        width,slots=p['width'],p['slots']; require(width==len(corpus[0]['x']) and slots>=width and slots%width==0,'wrong logical/physical width')
        sink=edges[p['sink']]; peak=0.; count=0; squared=[]
        with (d/'trials.jsonl').open() as stream:
            for line,trial in zip(stream,corpus,strict=True):
                row=json.loads(line); y=row['decoded']; truth=trial['truth']
                require(row['trial']==trial['trial'] and len(y)==slots and row['n']==sink['n'] and row['level']==sink['Level'] and
                        math.isclose(row['log_scale'],sink['physical_log_scale'],rel_tol=0,abs_tol=1e-7), 'wrong decoded ABI or missing slot')
                squared.append(math.fsum((v-truth[j%width])**2 for j,v in enumerate(y)))
                for j,a in enumerate(y):
                    delta=abs(a-truth[j%width]); require(delta<TOLERANCE,f'{case} trial {row["trial"]} slot {j}: error {delta} >= {TOLERANCE}')
                    peak=max(peak,delta); count+=1
        summary.update(selected=sites,emit_sha256=receipt,physical_slots_checked=count,linf=peak,mse=math.fsum(squared)/count,threshold=TOLERANCE,violations=0,
                       trials_sha256=sha(d/'trials.jsonl'))
    schema='snabf.c1.public-witness.v1' if public_only else ('snabf.c1.decrypted-slots-only.v1' if slot_archive else 'snabf.c1.decrypted.v1')
    return dict(schema=schema,cases=result)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--reuse64-pair',type=Path,metavar='STRICT_RUN')
    mode.add_argument('--c1-witness',action='store_true',help='source/erf/exp replay only; no encryption claim')
    mode.add_argument('--c1-gelu',action='store_true',help='source replay plus checked DirectGo and all decrypted slots')
    mode.add_argument('--c1-gelu-slots',action='store_true',help='portable plaintext/slot and program checks only; does not replay key inventory')
    p.add_argument('--output',type=Path)
    a = p.parse_args()
    c1=a.c1_witness or a.c1_gelu or a.c1_gelu_slots
    result = c1_verify(a.run,public_only=a.c1_witness,slot_archive=a.c1_gelu_slots) if c1 else (reuse64_pair(a.run,a.reuse64_pair) if a.reuse64_pair else verify(a.run))
    if a.output:
        save(a.output,result)
    print(json.dumps(result if c1 or a.reuse64_pair else {k:result[k] for k in ('outcome','source_ring','target_ring','observed_key_bytes','measurements')},allow_nan=False))
