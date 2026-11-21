"""Source-bound adapter for single-ring, periodic R3/C1 DirectGo v1 scale replay.

The original source and concrete profile are verifier inputs. The pass2 stage's
public binary64 coefficients are inputs, not an independently proved approximation.
Reconstruct the checked SSA expansion before inspecting any proposed receipts.
"""
from collections import defaultdict
from decimal import Decimal, localcontext
from functools import lru_cache
import hashlib
from pathlib import Path
import re
import struct
import sys

from exact_scale_replay import Q, basis, need

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'frontend/snabf_frontend'))
from pass2_evidence import Oracle
from gelu_witness import Source

# Oracle parses the R3 matrices, but not every scalar operation. Only use it
# on the published source grammar/bytes it actually witnessed (R3 SHA256SUMS).
R3 = {'c7e8e778ac39fbca30002d7ab4af7b57d130e9fdfe4be8ad491920b07348e1f2',
      'c87b0bbd03a024e297088e877619003468bca01d50d95b1c99503297cd5ca8e1',
      '8dcbe72b6f9b6950206bec2d8c4252214b7d1c8c28b3ab85e283baaf701992e7',
      '43942bfec60718bf347f28247542e6ba090260e04fdbb3a56ce75b8b0239d4c7',
      'c328a9900f27187299e03c372adcc796b7e6e426bbe8b0e11cda8cc39f852662'}
SSA = r'%[\w.]+'


def original(raw):
    text = raw.decode('utf8'); ops = [('source', [], None)]
    if hashlib.sha256(raw.replace(b'\r\n', b'\n')).hexdigest() in R3:
        model = Oracle(text)
        ops.extend([('linear',[0],model.w[0]), ('silu',[1],None)])
        if model.residual:
            ops.append(('add',[1,2],None))
        ops.append(('linear',[len(ops)-1],model.w[1]))
        if model.cross_residual:
            ops.append(('add',[len(ops)-1,1],None))
    else:
        try:
            model = Source(text)  # expression-tree-checked exact erf / SiLU C1 grammar
        except (ValueError, KeyError, StopIteration) as exc:
            raise ValueError('unsupported original source witness') from exc
        indices = {model.x: 0}
        for name, (kind, a, *extra) in model.nodes.items():
            args = [indices[a], indices[extra[0]]] if kind == 'add' else [indices[a]]
            indices[name] = len(ops)
            ops.append((kind, args, extra[0] if kind == 'linear' else None))
        need(indices[model.sink] == len(ops)-1, 'unsupported source sink')
    symbol = re.findall(r'func\.func @(\w+)\(', text)
    need(len(symbol) == 1, 'ambiguous original function')
    return symbol[0], model.width, ops


def bits(x):
    return struct.pack('>d', x)


@lru_cache(maxsize=128)
def logq(q):
    # Legacy plan selection uses binary64 log2(double(Q)), not the scale
    # arithmetic itself. Bracket the correctly rounded log; reject drift.
    with localcontext() as c:
        c.prec = 100
        a = Decimal.from_float(float(q)).ln(); b = Decimal(2).ln()
        lo = float((a.next_minus()/b.next_plus()).next_minus())
        hi = float((a.next_plus()/b.next_minus()).next_plus())
        need(lo == hi, 'ambiguous binary64 prime log')
        return lo


def poly_logs(nodes, out, level, q, default):
    """Replay the legacy affine log selection, not its exported continuous scales."""
    n = len(nodes); rows = []
    def row(i, args=(), rhs=0.):
        r = [0.]*(n+1); r[i] = 1.
        for j in args:
            r[j] -= 1.
        r[-1] = rhs; rows.append(r)
    row(0); row(out)
    for i, v in enumerate(nodes):
        if v['op'] == 'mul':
            row(i, v['args'], default-logq(q[level-v['depth']+1]))
        elif v['op'] == 'add':
            for j in v['args']:
                row(i,[j])
    piv = []
    for col in range(n):
        i = len(piv)
        if i == len(rows): break
        j = max(range(i,len(rows)), key=lambda k:abs(rows[k][col]))
        if not rows[j][col]: continue
        rows[i], rows[j] = rows[j], rows[i]
        rows[i] = [a/rows[i][col] for a in rows[i]]
        for k, r in enumerate(rows):
            if k != i:
                rows[k] = [a-r[col]*b for a,b in zip(r,rows[i])]
        piv.append(col)
    need(all(abs(r[-1]) <= 1e-9 for r in rows[len(piv):]), 'incompatible polynomial log constraints')
    result = [float(default)]*n
    for i, c in enumerate(piv): result[c] += rows[i][-1]
    return result


def public_polynomial(sel, kind):
    nodes, stages, out = sel['nodes'], sel['stages'], sel['output']
    need(len(stages) == 1 and 1 < len(nodes) <= 96 and out == len(nodes)-1 and
         sel['template'] == kind+'_direct', 'unsupported selected polynomial')
    stage, polys = stages[0], []
    coeff = list(map(Q, stage['coefficients']))
    need(stage['reference'] == {'silu':'half_tanh_half','gelu':'half_erf_sqrt2'}[kind] and
         stage['input_node'] == 0 and 0 < stage['output_node'] < out and
         len(coeff)==stage['degree']+1 and 1 < len(coeff) <= 65 and all(x==0 for x in coeff[::2]),
         'unbound activation coefficient bits')
    for i, v in enumerate(nodes):
        a, w, c, op = v['args'], v['weights'], Q(v['constant']), v['op']
        need(all(type(j) is int and 0 <= j < i for j in a), 'non-topological selected SSA')
        if op == 'input':
            need(i==0 and not a and not w and not c and v['depth']==0, 'invalid polynomial input'); p=[Q(0),Q(1)]
        elif op == 'mul':
            need(len(a)==2 and not w and not c, 'invalid polynomial product')
            x,y = (polys[j] for j in a); p=[Q(0)]*(len(x)+len(y)-1)
            need(len(p) <= 66, 'polynomial degree exceeds scope')
            for j, z in enumerate(x):
                for k, t in enumerate(y): p[j+k] += z*t
        else:
            need((op=='linear' and len(w)==len(a)>0) or (op=='add' and len(a)>=2 and not w), 'invalid polynomial sum')
            p=[Q(0)]*max(len(polys[j]) for j in a); p[0]=c
            for j,z in zip(a,w or [1]*len(a)):
                for k,t in enumerate(polys[j]): p[k] += t*Q(z)
        polys.append(p)
    want=[Q(0)]+coeff; want[1]+=Q(1,2)
    need(polys[stage['output_node']]==coeff and polys[out]==want, 'DAG differs from public stage/bypass bits')
    return nodes, out


def bind(checked, raw_source, profile, p, attr, source_bound):
    """Return fresh instructions (including logs and signed values) or fail closed."""
    symbol, width, logical = original(raw_source)
    need(profile['encryptionType']=='pk' and profile['encryptionTechnique']=='standard' and
         type(profile['logN']) is int and 2 <= profile['logN'] <= 20, 'unsupported profile')
    n, q, aux, default = 1<<profile['logN'], profile['Q'], profile['P'], profile['logDefaultScale']
    basis(n,tuple(q),tuple(aux)); need(type(default) is int and 1 <= default <= 64, 'UNSUPPORTED_MULTIPRIME_RESCALE')
    text=checked.read_text(encoding='utf8').splitlines()[1:]
    def field(s, key):
        hits=list(re.finditer(r'(?<![\w.])'+re.escape(key)+r' = ',s))
        need(len(hits)==1,'missing or ambiguous '+key); return attr(s,hits[0].end()).value()
    prof=[s.split('}> {',1)[0] for s in text if s.lstrip().startswith('"snabf.profile"()')]
    need(len(prof)==1, 'one concrete checked profile required')
    slots=field(prof[0],'slots'); key=field(prof[0],'key_id'); epoch=field(prof[0],'sym_name')
    need((field(prof[0],'ring'),field(prof[0],'q'),field(prof[0],'p'),field(prof[0],'default_log_scale')) ==
         (n,q,aux,default) and key==symbol+'__configure:<>' and
         p['epochs'][0]['key_id']==key and p['epochs'][0]['q']==q and p['epochs'][0]['p']==aux and
         p['epochs'][0]['scale']==default and p['width']==width and p['slots']==slots and
         0 < slots <= n//2 and slots%width==0 and source_bound==1, 'source/profile/domain mismatch')
    layout=(f'#tensor_ext.original_type<originalType = tensor<{width}xf32>, layout = #tensor_ext.layout<'
            f'"{{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod {width} = 0 and 0 <= i0 <= {width-1}'
            f' and 0 <= slot <= {slots-1} }}">>')
    need('layout = '+layout in prof[0], 'unsupported checked packing map')

    # Resolve the v1 preprocessing table through its actual storage site. SSA
    # names are function-local; do not search for the first constant in a file.
    def body(pattern):
        at=[i for i,s in enumerate(text) if re.match(pattern,s)]
        need(len(at)==1,'missing/ambiguous checked function')
        end=next((i for i in range(at[0]+1,len(text)) if text[i]=='  }'),None)
        need(end is not None,'unterminated checked function'); return text[at[0]+1:end]
    prep=body(r'  func\.func @'+re.escape(symbol)+r'__preprocessing\(')
    name=re.escape(symbol)
    evaluation=body(r'  func\.func @'+name+r'__preprocessed\(%arg0: ')
    def one(lines, pattern):
        hits=[m for s in lines if (m:=re.match(pattern,s.strip()))]
        need(len(hits)==1,'missing or ambiguous checked SSA port'); return hits[0]
    caller=body(r'  func\.func @'+name+r'\(%arg0: ')
    table_arg=one(caller,rf'({SSA}) = call @{name}__preprocessing\(\) :')[1]
    ret_arg=one(caller,rf'({SSA}) = call @{name}__preprocessed\(%arg0, {re.escape(table_arg)}\) :')[1]
    one(caller,rf'return {re.escape(ret_arg)} : tensor<1x!lwe\.lwe_ciphertext<')
    zero=one(evaluation,rf'({SSA}) = arith\.constant 0 : index$')[1]
    source_operand=one(evaluation,rf'({SSA}) = tensor\.extract %arg0\[{re.escape(zero)}\] : tensor<1x!lwe\.lwe_ciphertext<')[1]
    constants, prepared, stores, loads = {}, {}, {}, {}
    def put(table,k,v):
        need(k not in table,'duplicate checked binding'); table[k]=v
    def site(s):
        v=re.findall(r'\bsite_id = (\d+) : i32',s); need(len(v)==1,'unbound storage slot'); return int(v[0])
    stores_arg=re.findall(r'^\s*return ('+SSA+r') : !preprocessing\.storage', '\n'.join(prep),re.M)
    need(len(stores_arg)==1,'ambiguous prepared storage result')
    for s in prep:
        if m:=re.match(r'\s*('+SSA+r') = arith.constant dense<"0x([0-9A-Fa-f]+)"> : tensor<(\d+)x(\d+)xf32>$',s):
            put(constants,m[1],(int(m[3]),int(m[4]),bytes.fromhex(m[2])))
        elif m:=re.match(r'\s*('+SSA+r') = "kernel.prepare_linear_transform"\(('+SSA+r')\)',s):
            put(prepared,m[1],(field(s,'diagonal_indices'),constants[m[2]]))
        elif m:=re.match(r'\s*"preprocessing.store"\(('+SSA+r'), ('+SSA+r')\)',s):
            need(m[2]==stores_arg[0], 'different preprocessing store'); put(stores,site(s),m[1])
    for s in evaluation:
        if m:=re.match(r'\s*('+SSA+r') = "preprocessing.load"\(%arg1\)',s): put(loads,m[1],site(s))

    edges, logical_values, ssa = [], [], {}
    def emit(id_, op, level, args, *, vals=None, shift=None, log=None):
        need(0 <= level < len(q) and all(0<=j<len(edges) for j in args), 'invalid expanded SSA')
        first=edges[args[0]]['physical_log_scale'] if args else float(default)
        if log is None:
            log = (first+edges[args[1]]['physical_log_scale'] if op=='mul' else first+logq(q[level]) if op=='mask'
                   else first-logq(q[level+1]) if op=='rescale' else first)
        e=dict(ID=id_,op=op,Level=level,inputs=args,n=n,kernel_ring=n,physical_log_scale=log)
        if vals is not None: e['values']=vals
        if shift is not None: e.update(rotation=shift,route_site=lin)
        edges.append(e); return len(edges)-1
    def group(id_, args, level):
        need(args,'empty sum'); return args[0] if len(args)==1 else emit(id_,'add',level,args)
    lin=poly=step=0; sink=None
    patt=re.compile(r'\s*('+SSA+r') = "snabf\.(\w+)"\(([^()]*)\)')
    for line in evaluation:
        m=patt.match(line)
        if not m: continue
        dst,op=m[1],m[2]; args=m[3].split(', ')
        if op=='sink':
            need(sink is None and len(args)==1 and args[0] in ssa, 'unbound sink'); sink=ssa[args[0]]
            result=one(evaluation,rf'({SSA}) = tensor\.insert {re.escape(dst)} into {SSA}\[{re.escape(zero)}\] : tensor<1x!lwe\.lwe_ciphertext<')[1]
            one(evaluation,rf'return {re.escape(result)} : tensor<1x!lwe\.lwe_ciphertext<'); continue
        i=len(logical_values); need(i<len(logical) and sink is None, 'surplus checked logical operation')
        kind,parents,matrix=logical[i]
        typ=re.search(r' -> !snabf\.ct<tensor<(\d+)xf32>, (\d+), (\d+), @(\w+)>$',line)
        need(typ and (int(typ[1]),int(typ[2]),typ[4])==(width,n,epoch),'unbound ciphertext type')
        l=int(typ[3]); actual=[] if op=='source' else args[:1] if op=='lintrans' else args
        need([ssa.get(a) for a in actual]==[logical_values[j] for j in parents], 'original/checked SSA operand mismatch')
        a=ssa[actual[0]] if actual else None
        if op=='source' and kind=='source':
            need(i==0 and args==[source_operand] and l==len(q)-1,'unbound source'); v=emit('input',op,l,[])
        elif op=='lintrans' and kind=='linear':
            need(len(args)==2 and field(line,'site')==lin and edges[a]['Level']==l+1,'wrong linear use')
            selected=field(line,'selected'); route=p['routes'][lin]
            need(route['site']==lin and route['n']==n and route['level']==l+1 and route['selected']==selected,'wrong selected route')
            t=selected['terms']; diags, (nr,nc,table)=prepared[stores[loads[args[1]]]]
            ds=[d for d in range(width) if any(matrix[r][(r+d)%width] for r in range(width))]
            declared=field(line.split('}>',1)[0],'diagonals')
            need(declared==diags==ds==t[::3] and (nr,nc,len(table))==(len(ds),slots,4*slots*len(ds)) and
                 len(t)==3*len(ds), 'wrong diagonal cover/shape')
            pre=f'lin{lin}.'; babies={0:a}; groups=defaultdict(list)
            for j,(d,b,g) in enumerate(zip(ds,t[1::3],t[2::3])):
                need(type(b) is int and type(g) is int and 0<=b<slots and 0<=g<slots and (b+g)%slots==d,'invalid route decomposition')
                base=[matrix[r%width][(r+d)%width] for r in range(slots)]
                need(table[4*j*slots:4*(j+1)*slots]==struct.pack(f'<{slots}f',*base),'checked table differs from original public bits')
                if b not in babies: babies[b]=emit(pre+f'baby{b}','baby',l+1,[a],shift=b)
                groups[g].append(emit(pre+f'mask{d}','mask',l+1,[babies[b]],vals=base[-g:]+base[:-g] if g else base))
            merged=[]
            for g,terms in sorted(groups.items()):
                z=group(pre+f'coset{g}',terms,l+1)
                merged.append(emit(pre+f'giant{g}','giant',l+1,[z],shift=g) if g else z)
            v=emit(pre+'rescale','rescale',l,[group(pre+'merge',merged,l+1)]); lin+=1
        elif op=='poly_eval' and kind in ('gelu','silu'):
            need(field(line.split('poly_frontier = ',1)[0],'activation')==kind, 'original activation mismatch')
            nodes,out=public_polynomial(field(line,'poly_selected'),kind); start=edges[a]['Level']
            need(all(0<=nd['depth']<=start for nd in nodes),'invalid poly depth')
            logs=poly_logs(nodes,out,start,q,default); produced=[]
            for j,nd in enumerate(nodes):
                opn=nd['op']; end=start-nd['depth']; top=end+int(opn in ('mul','linear')); pre=f'poly{poly}.n{j}.'
                operands=[]
                for k,ref in enumerate(nd['args']):
                    z=produced[ref]; need(edges[z]['Level']>=top,'level increase in polynomial')
                    operands.append(z if edges[z]['Level']==top else emit(pre+f'drop{k}','drop',top,[z]))
                if opn=='mul': v=emit(pre+'relin','relin',top,[emit(pre+'mul','mul',top,operands)])
                elif opn=='input': v=a
                else:
                    if opn=='linear':
                        operands=[emit(pre+f'pt{k}','coefficient',top,[z],vals=[nd['weights'][k]],log=logs[j]+logq(q[top]))
                                  for k,z in enumerate(operands)]
                    v=group(pre+'add',operands,top)
                if opn in ('mul','linear'): v=emit(pre+'rescale','rescale',end,[v])
                if nd['constant']: v=emit(pre+'constant','add_plain',end,[v],vals=[nd['constant']])
                produced.append(v)
            v=produced[out]; poly+=1
        elif op=='ckks_step' and kind=='add' and field(line,'kind')=='ckks.add':
            operands=[ssa[u] for u in args]; pre=f'step{step}'
            need(len(operands)==2 and l==min(edges[z]['Level'] for z in operands),'wrong residual level')
            v=group(pre,[z if edges[z]['Level']==l else emit(pre+f'.drop{z}','drop',l,[z]) for z in operands],l); step+=1
        else: raise ValueError('unsupported checked/source op: '+op+'/'+kind)
        need(edges[v]['Level']==l,'wrong output level'); put(ssa,dst,v); logical_values.append(v)
    need(len(logical_values)==len(logical) and len(p['routes'])==lin and sink==logical_values[-1]==p['sink'] and
         len(edges)==len(p['instructions']), 'incomplete source/checked graph')
    keyset = sorted({pow(5,e['rotation'],2*n) for e in edges if 'rotation' in e})
    need(p['epochs'][0]['galois']==keyset, 'wrong concrete rotation key inventory')
    for fresh,claim in zip(edges,p['instructions']):
        # No ignoring a surplus branch, call, value, rotation, or unmodelled payload.
        need(set(claim)==set(fresh)|{'K'} and all(claim[k]==v for k,v in fresh.items() if k not in ('values','physical_log_scale')) and
             bits(fresh['physical_log_scale'])==bits(claim['physical_log_scale']) and
             list(map(bits,fresh.get('values',[])))==list(map(bits,claim.get('values',[]))), 'source-derived instruction mismatch: '+fresh['ID'])
        fresh['K']=claim['K']  # already matched to checked pass3; not used as a scale input
    return dict(p,instructions=edges)
