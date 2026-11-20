"""Build a finite calibration manifest from emitted single-ring audit programs.

An audit describes which calls to time. It supplies neither prices nor proof
of correctness. Each candidate in the manifest still executes a real kernel.
"""
import argparse
import itertools
import json
from pathlib import Path


def generate(programs, max_k, limit):
    calls = {}
    for g in programs:
        if len(g['epochs']) != 1 or g['source_ring'] != g['target_ring']:
            raise ValueError('only single-ring programs are calibrated')
        e, groups = g['epochs'][0], {}
        for i, node in enumerate(g['instructions']):
            op = node['op']
            if op not in ('baby','giant','relin'):
                continue
            if node['kernel_ring'] != e['n'] or len(node['inputs']) != 1:
                raise ValueError('inconsistent ring or gadget arity')
            # DirectGo hoists only baby rotations on the same route AND SSA input.
            key = ('baby',node['route_site'],node['inputs'][0]) if op == 'baby' else (op,i)
            groups.setdefault(key,[]).append(node)
        for group in groups.values():
            n, ell, op = e['n'], group[0]['Level'], group[0]['op']
            shifts = tuple(row['rotation'] for row in group) if op != 'relin' else ()
            if any(row['Level'] != ell for row in group) or len(set(shifts)) != len(shifts):
                raise ValueError('inconsistent hoisted group')
            key = (n,tuple(e['q']),tuple(e['p']),ell,op,shifts)
            calls[key] = len(group)
    if not calls:
        raise ValueError('no gadget invocation found')
    total = sum((min(max_k, k[3])+1)**width for k,width in calls.items())
    if total > limit:
        raise ValueError(f'{total} patterns exceed the explicit measurement limit {limit}; no partial manifest written')
    return {'cases':[dict(n=n,q=q,p=p,ell=ell,op=op,shifts=shifts,
                          patterns=list(itertools.product(range(min(ell,max_k)+1),repeat=width)))
                     for (n,q,p,ell,op,shifts),width in sorted(calls.items())]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('program',type=Path,nargs='+'); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--max-k',type=int,choices=range(5),default=2); p.add_argument('--limit',type=int,default=1024)
    a=p.parse_args()
    manifest=generate((json.loads(f.read_bytes()) for f in a.program),a.max_k,a.limit)
    with a.out.open('x',encoding='utf8') as out:
        json.dump(manifest,out,separators=(',',':')); out.write('\n')
    print(f"{len(manifest['cases'])} calls, {sum(len(c['patterns']) for c in manifest['cases'])} concrete patterns")


if __name__=='__main__': main()
