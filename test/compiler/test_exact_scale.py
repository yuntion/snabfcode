"""Compare both implementations to a fresh pinned Lattigo N0 run; never infer a certificate."""
import argparse
import json
from pathlib import Path
import struct
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0,str(ROOT/'src/tools'))
from exact_scale_replay import Q, encode, fold, join, multiply, need, pow2, rescale, rn
sys.path.insert(0,str(HERE/'rfc004'))
from run import go_cmd
from encoding_oracle import expected


def refused(fn, diagnostic):
    try: fn()
    except ValueError as e:
        need(diagnostic in str(e), f'wrong refusal: {e}'); return
    raise AssertionError('accepted mutation: '+diagnostic)


def check(observed, native=None):
    orig, notes = expected()
    need(orig == observed, 'original N0 predictions differ from Lattigo')
    q = [97,353,449]; n=2**40; a=(n,n); b=(n+1,n+1)
    ctx=dict(call_id='encode:pt',site_id='mul',operand_index=1,level=2,slots=8,key_id='key',ring_n=16,
             q=q,p=[193],default_log_scale=40,is_batched=True,arg_type='[]float64',arg_precision_bits=53,
             encoder_prec_bits=53,dynamic_input=False)
    words=struct.pack('>dd',1.,-0.); m=2**53+1
    c=encode(ctx,m,words); hi=encode(dict(ctx,encoder_prec_bits=60),m,words)
    s=join('sub',b,a,0,1); j=join('add',a,b,1,0)
    z=join('add',(4,4),(7,7),1,0); z9=join('add',(4,4),(9,9),1,0)
    h=rescale((65536,65536),q,2,[449,353],80)
    f=fold((3,3),(1,1),dict(ctx,q=[353,449,97]),words,32,[97]); fc=f['replacement']
    predicted=dict(meta=c['realized_meta_scale'],float64=c['realized_eff_scale'],batch60=hi['realized_eff_scale'],
                   near_meta=j['out_meta_scale'],near_add_coeff=str(Q(j['out_meta_scale'])+Q(j['lhs_signed_effect'])),
                   near_sub_coeff=str(-Q(s['out_meta_scale'])+Q(s['rhs_signed_effect'])),decoded_bias=j['decoded_bias'],
                   alignment_trunc=str(z['lhs_int_mul']),four_nine_coeff=str(9+Q(z9['lhs_signed_effect'])),
                   after=h['meta_after'],after_level=h['out_level'],one_div=str(rn(Q(65536,449*353))),
                   fold_requested=fc['requested_scale'],fold_actual_meta=fc['realized_meta_scale'],
                   fold_f64_eff=fc['realized_eff_scale'],fold_residual=f['residual_scale_diff'])
    measured=dict(meta=observed['meta'],float64=str(observed['coeff0']['batch_53_f64']),
                  batch60=str(observed['coeff0']['batch_60_f64']),near_meta=observed['near_add']['out_metadata'],
                  near_add_coeff=str(observed['near_add']['coeff0']),near_sub_coeff=str(observed['near_sub_reverse']['coeff0']),
                  decoded_bias=str(1-Q(observed['near_add']['coeff0'],int(observed['near_add']['out_metadata']))),
                  alignment_trunc=observed['alignment_trunc'],four_nine_coeff=str(observed['four_nine']['coeff0']),
                  after=observed['rescale']['after'],after_level=observed['rescale']['after_level'],one_div=observed['rescale']['one_div'],
                  fold_requested=observed['fold_requested'],fold_actual_meta=observed['fold_actual_meta'],fold_f64_eff=observed['fold_f64_eff'],
                  fold_residual=str(3*Q(observed['fold_f64_eff'])/97-1))
    need(predicted==measured and (native is None or native==measured),'tracker differs from real N0 observations')
    need(j['encoded_bias']=='1' and j['decoded_bias']==s['decoded_bias']==notes['join_required_message_units'] and
         s['rhs_signed_effect']=='1' and j['lhs_signed_effect']=='-1','join sign or units')
    both=join('sub',(4,3),(7,6),2,3)
    need(both['encoded_bias']=='11' and both['decoded_bias']=='11/7' and both['common_effective_scale'] is None,'both magnitudes required')
    for l,r in ((None,0),(0,None)):
        v=join('add',a,a,l,r); need(v['missing_operand_bounds'] and v['encoded_bias'] is v['decoded_bias_upper'] is None,'missing bound')
    need(h['divisions'][0]['metadata']==str(rn(Q(65536,449))) and Q(h['meta_after'])-Q(predicted['one_div'])==-pow2(-129)
         and Q(h['eff_after'])==Q(65536,449*353) and not f['exact_representable'] and f['rho']=='97/96','wrong division or fold')
    for invalid in ([353,449],[449],[449*353]):
        refused(lambda:rescale((65536,65536),q,2,invalid,80), 'ordered dropped_q')
    for invalid in ([449,353],[449]):
        refused(lambda:rescale((65536,65536),q,2,invalid,80,lowering=True),'UNSUPPORTED_MULTIPRIME_RESCALE')
    for change in (dict(is_batched=False),dict(arg_type='[]*big.Float'),dict(arg_precision_bits=128),dict(dynamic_input=True)):
        refused(lambda:encode(dict(ctx,**change),m,words),'encoding call')
    refused(lambda:encode(dict(ctx,p=[449]),m,words),'ordered Q/P')
    need(encode(ctx,m,struct.pack('>dd',1.,0.))['argument_sha256']!=c['argument_sha256'],'lost sign of zero')
    for changed in (dict(encoder_prec_bits=60),dict(operand_index=0),dict(site_id='mul2'),dict(key_id='other'),dict(q=q[::-1])):
        need(encode(dict(ctx,**changed),m,words)!=c,'unbound call context')
    need(multiply((m,m-1),(m,m-1))[1]==(m-1)**2,'E not propagated')
    return predicted


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tracker',nargs='?',type=Path,help='ExactScaleTrackerTest executable')
    parser.add_argument('--observed',type=Path,help='use an untouched prior N0 output for portable offline comparison')
    args=parser.parse_args()
    native=json.loads(subprocess.check_output([args.tracker.resolve()])) if args.tracker else None
    if args.observed: observed=json.loads(args.observed.read_bytes())
    else:
        go,env=go_cmd()
        need(subprocess.check_output([go,'version'],env=env).split()[2]==b'go1.26.0','wrong Go pin')
        need(subprocess.check_output([go,'list','-mod=readonly','-m','github.com/tuneinsight/lattigo/v6'],cwd=ROOT,env=env).split()[-1]==b'v6.2.0','wrong Lattigo pin')
        observed=json.loads(subprocess.check_output([go,'run','-mod=readonly',HERE/'rfc004/testdata/encoding/main.go'],cwd=ROOT,env=env))
    print(json.dumps(dict(exact_fields=len(check(observed,native)),cpp_compared=native is not None,probabilistic_cert_issued=False)))
