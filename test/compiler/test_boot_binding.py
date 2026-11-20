"""The checked standalone binding must not run altered or unsupported plans."""
import re
import subprocess
import sys
from pathlib import Path


def main(opt, out):
    m=(out/'lowered.mlir').read_text()
    def reject(text, why, flags):
        assert text!=m
        p=subprocess.run([opt,'--allow-unregistered-dialect',*flags],input=text,text=True,capture_output=True)
        assert p.returncode and why in p.stderr,(why,p.stderr[:500])
    # Mutate just the lowered manifest; the binder re-reads the original SSA.
    changed,n=re.subn(r'(snabf.export_bootstrap = \{[^}]*?source_period = )64',r'\g<1>32',m,count=1)
    assert n
    reject(changed,'stale bootstrap binding',['--snabf-bind-lattigo'])
    # Same count of NTTable primes, but q0 no longer has the calibrated width.
    reject(m.replace('1099512938497','72057594047889409'),'no executable lowering',['--snabf-plan','--snabf-lower-ckks'])
    reject(m.replace('snabf.slot_period = 64','snabf.slot_period = 32768'),'no executable lowering',['--snabf-plan','--snabf-lower-ckks'])
    print('PASS: stale bootstrap manifest, wrong physical Q and aliasing input rejected')


if __name__=='__main__':main(sys.argv[1],Path(sys.argv[2]))
