"""Import the strided public matrices into the same secret-linalg SiLU block."""

import argparse
import json
from pathlib import Path
import re


def strided_model(topology, workload):
    n = workload['width']
    def matrix(terms):
        rows = [[0.0]*n for _ in range(n)]
        for i,row in enumerate(rows):
            for d,coef in terms:
                # Nonconstant diagonals make a missing inverse mask rotation visible.
                row[(i+d)%n] = coef * (-1 if (i+d)%5 == 0 else 1) * (0.5 if i%3 == 0 else 1)
        return json.dumps(rows)
    module = topology.replace('tensor<4x4xf32>', f'tensor<{n}x{n}xf32>').replace('tensor<4xf32>', f'tensor<{n}xf32>')
    for i,terms in enumerate(workload['layers']):
        module, count = re.subn(rf'(%w{i} = arith.constant dense<)\[.*?\](> : tensor<)',
                                lambda m: m[1]+matrix(terms)+m[2], module, flags=re.S)
        if count != 1: raise ValueError('unexpected topology for strided import')
    return '\n'.join(line for line in module.splitlines() if not line.lstrip().startswith('//'))+'\n'


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('topology','workload','output'): p.add_argument(name,type=Path)
    a = p.parse_args()
    a.output.write_text(strided_model(a.topology.read_text(),json.loads(a.workload.read_text())))
