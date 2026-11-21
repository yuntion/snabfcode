"""Source-bound differential and coordinated-forgery tests on an unmodified v1 package."""
import argparse
import copy
import json
import math
from pathlib import Path
import re
import struct
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src/tools'))
from replay_noise_audit import Attributes, binding_plan, independent, load, need, same
from scale_source_binding import R3


def mlir(x):
    if isinstance(x, dict): return '{'+', '.join(k+' = '+mlir(v) for k,v in sorted(x.items()))+'}'
    if isinstance(x, list): return '['+', '.join(map(mlir,x))+']'
    if type(x) is int: return str(x)+' : i64'
    if isinstance(x,str): return json.dumps(x)
    raise ValueError('unsupported receipt attribute')


def replace_encoding(text, plan):
    marker='snabf.encoding_receipts = '; start=text.index(marker)+len(marker)
    reader=Attributes(text,start); reader.value()
    return text[:start]+mlir(plan)+text[reader.pos:]


def run(checked, program, audit, original, basis, cpp_trace=None):
    legacy=independent(checked,program,audit)
    def replay(c=checked,p=program,*,source=original,compare=None):
        return independent(c,p,source_witness=source,basis=basis,compare_scales=compare)
    result=independent(checked, program, audit, source_witness=original,basis=basis,compare_scales=cpp_trace)
    if cpp_trace: need(result.pop('exact_scale_matched'), 'C++ trace mismatch')
    trace=result.pop('exact_scale'); need(same(result,legacy),'changed historical B2.1 report')
    need(len(trace['edges'])==legacy['edges'] and len(trace['calls'])==legacy['encoding_calls'] and
         not trace['probabilistic_cert_issued'] and legacy['missing']>0, 'incomplete or certifying trace')
    data=load(program); text=checked.read_text(encoding='utf8'); denials=[]
    def rejected(label, reason, fn):
        try: fn()
        except ValueError as exc:
            need(reason in str(exc),f'{label}: unexpected refusal {exc}'); denials.append(label); return
        raise AssertionError('accepted mutation: '+label)
    with tempfile.TemporaryDirectory(prefix='b11-source-negative-') as td:
        tmp=Path(td); c,p,t=tmp/'checked.mlir',tmp/'program.json',tmp/'trace.json'
        def forged(label, mutate, reason, *, also_legacy=False):
            d=copy.deepcopy(data); mutated=mutate(d,text)
            # Synchronize the graph fingerprint and every legacy call receipt
            # on BOTH sides. A mismatch between two derived digests is not a test.
            d['encoding_receipts']=binding_plan(d)
            c.write_text(replace_encoding(mutated,d['encoding_receipts']),encoding='utf8'); p.write_text(json.dumps(d))
            if also_legacy: independent(c,p)  # demonstrate the narrower historical scope
            rejected(label,reason,lambda:replay(c,p))
        def value(d,s):
            x=next(e for e in d['instructions'] if e['op']=='mask')['values']; x[0]=math.nextafter(x[0],math.inf); return s
        forged('public-bit-and-both-receipts',value,'source-derived instruction',also_legacy=True)
        def stage(d,s):
            e=next(e for e in d['instructions'] if e['op']=='coefficient')
            w=math.nextafter(e['values'][0],math.inf); e['values'][0]=w
            pos=s.index('poly_selected = ',s.index(' = "snabf.poly_eval"'))
            pos=s.index('nodes = [',pos)
            s2,n=re.subn(r'(weights = array<f64: )[\w.+-]+',lambda m:m[1]+'0x'+struct.pack('>d',w).hex(),s[pos:],count=1)
            need(n==1,'no selected public coefficient'); return s[:pos]+s2
        forged('one-ulp-in-selected-dag-program-and-both-receipts',stage,'DAG differs from public',also_legacy=True)
        def edge_attr(s,id_,key,old,new):
            pattern=r'(\{ID = "'+re.escape(id_)+r'",[^{}]*?\b'+key+r' = )'+old
            s,n=re.subn(pattern,lambda m:m[1]+new,s); need(n>0,'unbound negative fixture '+id_); return s
        def wire(d,s):
            e=next(e for e in d['instructions'] if e['op']=='add' and len(set(e['inputs']))>1)
            e['inputs'].reverse(); return edge_attr(s,e['ID'],'inputs',r'array<i64:[^>]+>','array<i64: '+', '.join(map(str,e['inputs']))+'>')
        forged('swapped-checked-and-program-operands',wire,'source-derived instruction',also_legacy=True)
        def log(d,s):
            e=next(e for e in d['instructions'] if e['op']=='coefficient')
            e['physical_log_scale']=math.nextafter(e['physical_log_scale'],math.inf)
            return edge_attr(s,e['ID'],'physical_log_scale',r'[\w.+-]+','0x'+struct.pack('>d',e['physical_log_scale']).hex())
        forged('requested-log-one-ulp-and-both-receipts',log,'source-derived instruction',also_legacy=True)
        def basis_forgery(d,s,k):
            v=d['epochs'][0][k]; old=', '.join(map(str,v))
            if k=='q': v[-1],v[-2]=v[-2],v[-1]
            else: v[0]+=2*d['epochs'][0]['n']
            marker=k+' = array<i64: '+old+'>'; need(marker in s,'basis missing'); return s.replace(marker,k+' = array<i64: '+', '.join(map(str,v))+'>')
        for k in ('q','p'):
            forged('changed-ordered-'+k+'-in-all-derived-data',lambda d,s:basis_forgery(d,s,k),'source/profile/domain')
        def branch():
            d=copy.deepcopy(data); row=d['encoding_receipts']['receipts'][0]; row['branch']='embedArbitrary/Scale.Value'
            c.write_text(replace_encoding(text,d['encoding_receipts']),encoding='utf8'); p.write_text(json.dumps(d))
            replay(c,p)
        rejected('synchronized-legacy-branch','encoding scale binding',branch)

        # Alter the checked public table alone; source and program still agree.
        s,n=re.subn(r'(arith\.constant dense<"0x)([0-9A-Fa-f])',lambda m:m[1]+format(int(m[2],16)^1,'x'),text,count=1)
        need(n==1,'no packed f32 fixture'); c.write_text(s,encoding='utf8')
        rejected('checked-public-table','checked table differs',lambda:replay(c))
        src=next(re.finditer(r'^\s*(%\w+) = "snabf\.source"',text,re.M))[1]
        s,n=re.subn(r'("snabf\.poly_eval"\()(%\w+)',lambda m:m[1]+src,text,count=1)
        need(n==1,'unbound activation test'); c.write_text(s,encoding='utf8')
        rejected('real-checked-SSA-operand','original/checked SSA',lambda:replay(c))
        s=original.read_text(encoding='utf8'); pat=r'(arith\.constant dense<\[\[)(-?[\d.]+)'
        s,n=re.subn(pat,lambda m:m[1]+str(float(m[2])+0.0625),s,count=1); need(n==1,'original matrix absent')
        w=tmp/'original.mlir'; w.write_text(s,encoding='utf8')
        # Modified R3 bytes are not an allow-listed R3 witness; the generic C1
        # grammar also safely rejects them. C1's matrix mutation reaches binding.
        r3=__import__('hashlib').sha256(original.read_bytes().replace(b'\r\n',b'\n')).hexdigest() in R3
        rejected('original-weight-changed','unsupported original source' if r3 else 'checked table differs',lambda:replay(source=w))
        s,n=re.subn(r'(return )%inserted( : tensor<1x!lwe\.lwe_ciphertext)',r'\g<1>%arg0\2',text,count=1)
        need(n==1,'no output port'); c.write_text(s,encoding='utf8')
        rejected('returned-SSA-port','checked SSA port',lambda:replay(c))

        t.write_text(json.dumps(trace)); need(replay(compare=t)['exact_scale_matched'],'unchanged trace did not match')
        def trace_case(label, fn):
            d=copy.deepcopy(trace); fn(d); t.write_text(json.dumps(d))
            rejected(label,'exact scale trace differs',lambda:replay(compare=t))
        for k,v in [('call_id','another'),('site_id','wrong'),('operand_index',7),('is_batched',False),
                    ('arg_type','[]*big.Float'),('arg_precision_bits',128),('encoder_prec_bits',60),('branch','bigfloat_value'),
                    ('quantization_rule','batched-bigfloat'),('payload_sha256','0'*64),('argument_sha256','1'*64),
                    ('key_id','other'),('requested_scale','1')]:
            trace_case('call-'+k,lambda d,k=k,v=v:d['calls'][1].update({k:v}))
        trace_case('dropped-prime',lambda d:next(x['rescale'] for x in d['edges'] if 'rescale' in x)['dropped_q'].__setitem__(0,97))
        trace_case('depth-is-not-levels',lambda d:next(x['rescale'] for x in d['edges'] if 'rescale' in x).update(mul_depth_delta=1))
        trace_case('metadata-used-as-effective',lambda d:next(x for x in d['edges'] if x['reference_only']).update(effective='1'))
        def wrong_join(d):
            j=next(x['join'] for x in d['edges'] if x.get('join',{}).get('encoded_bias') not in ('0',None))
            j['decoded_bias']=j['encoded_bias']
        trace_case('join-encoded-is-not-decoded',wrong_join)
        trace_case('test-is-not-certificate',lambda d:d.update(probabilistic_cert_issued=True))
    return dict(edges=len(trace['edges']),calls=len(trace['calls']),missing=legacy['missing'],
                denials=len(denials),tested=denials,probabilistic_cert_issued=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for k in ('checked','program','audit','original','basis'): parser.add_argument('--'+k,required=True,type=Path)
    parser.add_argument('--cpp-trace',type=Path,help='output from ExactScaleTraceTest CHECKED ORIGINAL')
    print(json.dumps(run(**vars(parser.parse_args()))))
