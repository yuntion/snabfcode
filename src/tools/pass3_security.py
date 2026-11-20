"""Run the pinned estimator on the actual public QP instances (classical model)."""
import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

PIN = 'ae45d740ceb138af87a4f92edc592a0ab0d4f43f'

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('plan', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--estimator', type=Path, default=os.getenv('LATTICE_ESTIMATOR'))
    args = p.parse_args()
    if not args.estimator or subprocess.check_output(['git','-C',str(args.estimator),'rev-parse','HEAD'],text=True).strip() != PIN:
        raise ValueError('set LATTICE_ESTIMATOR to the pinned lattice-estimator checkout: '+PIN)
    sys.path.insert(0, str(args.estimator.resolve()))
    from estimator import LWE, ND
    from sage.all import infinity, log
    data=json.loads(args.plan.read_text()); n=data['ring']; q=math.prod(data['q']); qp=q*math.prod(data['p'])
    def estimate(modulus):
        par=LWE.Parameters(n=n,q=modulus,Xs=ND.Uniform(-1,1),Xe=ND.DiscreteGaussian(3.2),m=infinity)
        estimates=LWE.estimate.rough(par,quiet=True,catch_exceptions=False)
        attacks={str(k):float(log(v['rop'],2)) for k,v in estimates.items()}
        if not {'usvp','dual_hybrid'} <= attacks.keys() or not all(math.isfinite(v) for v in attacks.values()):
            raise ValueError(f'incomplete classical security estimate: {attacks}')
        return attacks
    results={}
    # Unlimited samples dominate the finite public inventory for this LWE screen.
    for label, modulus in [('envelope',1 << math.ceil(data['security_cap'])),('Q',q),('QP',qp)]:
        attacks=estimate(modulus)
        if min(attacks.values()) < 128:
            # This is a cut for this ring/distribution/estimator, not a general
            # modulus table. Check the tightened cap itself before proposing it.
            bits=min(math.floor(data['security_cap']),modulus.bit_length()-1);steps=[]
            for _ in range(16):
                bits-=10
                if bits<32:break
                check=estimate(1<<bits);steps.append(dict(bits=bits,attacks=check))
                if min(check.values())>=128:
                    args.output.write_text(json.dumps(dict(rejected=label,attacks=attacks,modulus=str(modulus),new_cap=bits,search=steps),indent=2)+'\n')
                    sys.exit(3)
            raise ValueError('no safe cap found under the configured estimator')
        results[label]={'modulus':str(modulus),'attacks_log2_rop':attacks}
    args.output.write_text(json.dumps(dict(ring=n,estimator=PIN,model='LWE.estimate.rough/classical ADPS16; GSA uSVP, dual_hybrid',
        secret='iid uniform ternary',error='discrete Gaussian sigma=3.2 (estimator model for bounded sampler)',samples='unbounded',
        inventory=data['public_instances'],instances=results),indent=2)+'\n')
if __name__=='__main__': main()
