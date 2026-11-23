"""Public native single-ring product path. No HEIR search or fallback."""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

from native_package import publish_native_package, unique_fields
from security_cache import ERROR, MODEL, PIN, SECRET, digest, measure, request

ROOT = Path(__file__).resolve().parents[2]


def mlir_string(raw):
    # MLIR byte escapes preserve CR/LF, quotes and non-ASCII source bytes.
    return '"'+''.join(chr(b) if 32 <= b < 127 and b not in (34, 92) else f'\\{b:02X}' for b in raw)+'"'


def carrier(fields):
    return 'module attributes {'+', '.join(f'snabf.n1_{k} = {mlir_string(v)}' for k, v in fields.items())+'} {}\n'


def screen(proposal, cache, estimator):
    p = proposal['plan']
    n, q, aux = p['ring'], p['q'], p['p']
    exact = math.prod(q)
    instances = {k: request('source.'+k, n, q, aux, m) for k, m in
                 [('Q', exact), ('QP', exact*math.prod(aux)), ('envelope', 1 << math.ceil(proposal['security_cap']))]}
    measured = measure(instances.values(), cache, estimator)
    return dict(schema='snabf.native-single-security.v1', source=p['source'], input_spec=p['input_spec'],
                estimator=PIN, model=MODEL, secret=SECRET, error=ERROR, samples='unbounded', ring=n, q=q, p=aux,
                instances={k:dict(modulus=v['modulus'], attacks=measured[digest(v)]) for k, v in instances.items()})


def native_compile(a):
    coords = (a.log_n, a.q_count, a.gadget_alpha, a.interval_mode, a.min_slots)
    relu = coords == (16,29,2,'affine',4096)
    n1 = coords == (14,8,None,None,4096)
    n2 = a.interval_mode is None and a.gadget_alpha in (1,2)
    n2_shape = (n2 and a.mode == 'keep-source' and 14 <= a.log_n <= 16 and 2 <= a.q_count <= 32 and
                1 <= a.min_slots <= 1 << (a.log_n-1) and not a.min_slots & (a.min_slots-1))
    if (a.input_spec is None or not math.isfinite(a.input_bound) or a.input_bound <= 0 or
            (a.security_bits, a.switch_security_bits, a.precision_bits) != (128,128,12) or not (n1 or relu or n2_shape)
            or a.mode not in ('adaptive','keep-source') or a.cut_profile != 'single-live-v1' or
            a.cost_profile or a.cost_hardware_id or a.heterogeneous_scales or
            a.scale_mode not in ((None,'narrow','wide') if n2 else (None,'narrow')) or a.max_k not in (None,0) or
            a.source_arithmetic_fraction is not None or a.certify_failure_bits is not None or a.certify_failure_prob is not None
            or (a.prime_bw_min,a.prime_bw_max,a.aux_prime_margin_bits) != (24,59,2)):
        raise ValueError('[SNABF_NATIVE_PROFILE] expected N1, affine ReLU, or explicit fixed-block N2; no noise certificate')
    source, spec = a.mlir.read_bytes(), a.input_spec.read_bytes()
    ports = json.loads(spec, object_pairs_hook=unique_fields)
    inputs = ports.get('inputs') if isinstance(ports, dict) else None
    d = inputs[0].get('domain') if isinstance(inputs, list) and len(inputs) == 1 and isinstance(inputs[0], dict) else None
    if d != [-a.input_bound, a.input_bound] or not all(type(x) in (int,float) for x in d):
        raise ValueError('[SNABF_NATIVE_PROFILE] original port domain must match --input-bound')
    out = a.out_dir.absolute()
    if out.exists() or out.is_symlink():
        raise ValueError('[SNABF_OUTPUT_EXISTS] use a new output directory')
    out.parent.mkdir(parents=True, exist_ok=True)
    ext = Path(os.getenv('SNABF_EXTERN', ROOT/'extern')).resolve()
    sdk = Path(os.getenv('SNABF_MLIR_ROOT', ext/'mlir24/usr/lib/llvm-24')).resolve()
    path = a.snabf_opt or os.getenv('SNABF_NATIVE_OPT') or Path(os.getenv('SNABF_OPT',ROOT/'build/native/ir/snabf-opt')).with_name('snabf-native-opt')
    binary = Path(path).absolute()
    if binary.name == 'snabf-opt':
        binary = binary.with_name('snabf-native-opt')  # never load the HEIR dialect to run a native program
    binary = binary.resolve(strict=True)
    env = dict(os.environ, LD_LIBRARY_PATH=os.pathsep.join(map(str,(sdk/'lib', sdk.parent/'x86_64-linux-gnu',
                         ext/'highs-install/lib', os.environ.get('LD_LIBRARY_PATH','')))))
    with tempfile.TemporaryDirectory(dir=out.parent, prefix='.snabf-n1-') as td:
        tmp = Path(td); publish = tmp/'publish'; publish.mkdir()
        def run(*args):
            subprocess.run([binary, '--mlir-print-op-on-diagnostic=false', *args], cwd=publish, env=env, check=True)
        fields = dict(source=source, ports=spec, stage=b'source')
        if relu:
            fields['profile'] = b'relu-affine-2p-v1'
        elif n2:
            from n2.search import search_envelope
            estimator = Path(os.getenv('LATTICE_ESTIMATOR',ext/'lattice-estimator'))
            envelope = search_envelope(1 << a.log_n, 128, lambda rows: measure(rows, a.security_cache, estimator))
            profile = dict(schema='snabf.n2-profile.v1', log_n=a.log_n, slots=a.min_slots, q_count=a.q_count,
                           scale_mode=a.scale_mode or 'narrow', alpha=a.gadget_alpha, cap_bits=envelope['cap_bits'])
            fields['n2_profile'] = json.dumps(profile,sort_keys=True,separators=(',',':')).encode('ascii')
            (publish/'search-envelope.json').write_text(json.dumps(envelope,sort_keys=True,allow_nan=False)+'\n')
        (publish/'source.mlir').write_text(carrier(fields),encoding='ascii')
        run('--snabf-native-plan=proposal=../proposal.json','source.mlir','-o','planned.mlir')
        proposal = json.loads((tmp/'proposal.json').read_bytes(), object_pairs_hook=unique_fields)
        plan = proposal['plan']
        if n2 and (plan['n2_search']['cap_bits'] != envelope['cap_bits'] or plan['n2_search']['alpha'] != a.gadget_alpha or
                   plan['n2_search']['scale_mode'] != profile['scale_mode'] or plan['ring'] != 1 << a.log_n or
                   plan['slots'] != a.min_slots or len(plan['q']) != a.q_count or len(plan['p']) != a.gadget_alpha):
            raise ValueError('[SNABF_NATIVE_PROFILE] N2 plan does not bind the requested coordinates')
        if (plan['source'] != hashlib.sha256(source).hexdigest() or plan['input_spec'] != hashlib.sha256(spec).hexdigest()
                or plan['input_domain'] != d or plan['probabilistic_cert_issued'] is not False):
            raise ValueError('[SNABF_NATIVE_PROFILE] proposal not bound to requested original bytes/domain')
        security = screen(proposal, a.security_cache, Path(os.getenv('LATTICE_ESTIMATOR',ext/'lattice-estimator')))
        (publish/'security.json').write_text(json.dumps(security, sort_keys=True, allow_nan=False)+'\n')
        run('--snabf-check-plan=native-security=security.json','planned.mlir','-o','checked.mlir')
        if a.emit_go:
            run('--snabf-lower-ckks=native-security=security.json native-output=../emitted','checked.mlir','-o','lowered.mlir')
            publish_native_package(tmp/'emitted', publish/'module', plan['emit_sha256'])
        receipt = dict(schema='snabf.native-compile.v1', frontend='native-linalg', mode=a.mode, selected_mode='keep-source',
                       emit_sha256=plan['emit_sha256'], source_sha256=plan['source'], input_spec_sha256=plan['input_spec'],
                       engineering_upper=plan['engineering_upper'], probabilistic_cert_issued=False, emitted_go=a.emit_go)
        (publish/'compile.json').write_text(json.dumps(receipt, sort_keys=True, allow_nan=False)+'\n')
        publish.rename(out)
