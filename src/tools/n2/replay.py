"""Recheck a successful 2x2 with the trusted private measurement cache, without solving."""
import argparse
import json
import math
from pathlib import Path
import subprocess

from search import ERROR, MODEL, PIN, SECRET, concrete_screen, digest, measure, need, search_envelope, sha


def replay(out, source, spec, proposer, cache, rebuild=False):
    read=lambda p:json.loads(p.read_bytes())
    oracle=lambda rs:measure(rs,cache,None)  # a miss is a failure; no imported verdict
    result=read(out/'matrix.json'); common=result['common']
    need(result['schema']=='snabf.n2.native-matrix.v1' and
         (common['source'],common['input_spec'],result['binary_sha256'])==(sha(source),sha(spec),sha(proposer)),
         'wrong original source/spec or proposer')
    need([result[k] for k in ('estimator','model','secret','error')]==[PIN,MODEL,SECRET,ERROR], 'wrong estimator model')
    envelope=search_envelope(common['ring'],128,oracle)
    need(envelope==read(out/'envelope.json') and result['envelope_sha256']==sha(out/'envelope.json') and
         common['security_cap']==envelope['cap_bits'],'stale seed-independent envelope')
    need(len(result['cells'])==4,'need all four cells'); graphs=set()
    for cell,(a,mode) in zip(result['cells'],[(a,m) for a in (1,2) for m in ('narrow','wide')]):
        site=out/f'a{a}-{mode}'; plan=read(site/'proposal.json')
        coords=list(map(str,[common['ring'].bit_length()-1,common['encoded_slots'],common['q_count'],common['min_scale'],
                             common['max_scale'],a,mode,envelope['cap_bits']]))
        need(len(cell['argv'])==11 and cell['argv'][3:]==coords, 'wrong solver invocation')
        if rebuild:
            actual=json.loads(subprocess.check_output([proposer.resolve(strict=True),source.resolve(strict=True),
                                                       spec.resolve(strict=True),*coords]))
            need(actual==plan,'native re-derivation disagrees with the stored proposal')
        security=concrete_screen(plan,dict(common,alpha=a,scale_mode=mode),oracle)
        need(security==read(site/'security.json') and security['meets_128'],'stale concrete Q/P measurements')
        q=plan['q'][1:]; graphs.add(plan['graph_sha256'])
        need(cell==read(site/'cell.json') and cell['alpha']==a and cell['scale_mode']==mode and
             cell['returncode']==0 and cell['result']=='screened_proposal' and cell['proposal'] is True and
             cell['concrete_screen'] is True and cell['ciphertext_run'] is False and cell['graph']==plan['graph_sha256'] and
             cell['scale']==plan['scale'] and cell['source_bound']==plan['approximation']+plan['arithmetic'] and
             cell['proposal_sha256']==sha(site/'proposal.json') and cell['security_sha256']==sha(site/'security.json') and
             cell['ordered_basis_sha256']==digest([plan['q'],plan['p']]) and cell['minimum_bits']==security['minimum_bits'] and
             cell['nonbase_q_spread_bits']==math.log2(max(q)/min(q)), 'stale cell or reported spread')
    need(len(graphs)==1 and result['same_arithmetic_graph'] is True and result['all_four_screened'] is True and
         result['product_checker_run'] is False and result['ciphertext_run'] is False,'wrong matrix level or topology')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('out','mlir','input-spec','proposer','cache'):
        ap.add_argument('--'+key,type=Path,required=True)
    ap.add_argument('--rebuild-proposals',action='store_true',help='also rederive the four arithmetic proposals from original bytes')
    args=ap.parse_args(); replay(args.out,args.mlir,args.input_spec,args.proposer,args.cache,args.rebuild_proposals)
    print('Rechecked four native proposals, their concrete screenings and non-base Q spreads.')
