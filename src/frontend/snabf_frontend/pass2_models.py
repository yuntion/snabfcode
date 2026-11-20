"""Source workloads for the native Pass 2 experiment; no plans or oracle outputs."""

import json
from pathlib import Path

from .strided import strided_model


WORKLOADS = ('control', 'strided64', 'reuse64', 'residual')
COVERAGE = 'composite_coverage'


def source_model(fixtures: Path, work: str) -> str:
    if work not in (*WORKLOADS, COVERAGE, 'cross_residual'):
        raise ValueError(f'unknown Pass 2 workload: {work}')
    source = (fixtures / 'linear_silu_linear.mlir').read_text()
    if work == 'cross_residual':
        source = strided_model(source, json.loads((fixtures / 'pass1_reuse64.json').read_text()))
        end = '    return %y : tensor<64xf32>'
        if source.count(end) != 1:
            raise ValueError('expected a single 64-wide terminal matvec result')
        return source.replace(end, '    %post = arith.addf %y, %h : tensor<64xf32>\n'
                                   '    return %post : tensor<64xf32>')
    if work in ('strided64', 'reuse64'):
        return strided_model(source, json.loads((fixtures / f'pass1_{work}.json').read_text()))
    if work == 'residual':
        weights = json.loads((fixtures / 'pass1_reuse64.json').read_text())
        weights['layers'] = list(reversed(weights['layers']))
        source = strided_model(source, weights)
        anchor = '    %y = linalg.matvec ins(%w1, %a :'
        if source.count(anchor) != 1:
            raise ValueError('expected one second matvec consuming SiLU in the source fixture')
        source = source.replace(anchor, '    %skip = arith.addf %h, %a : tensor<64xf32>\n'
                                '    %y = linalg.matvec ins(%w1, %skip :')
    return source
