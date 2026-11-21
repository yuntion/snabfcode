"""Structural inspection only. Never invoke a planner, evaluator or HEIR tool."""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[2]


def inspect_native(a):
    if (a.input_spec is None or a.emit_go or a.certify_failure_bits is not None or
            a.certify_failure_prob is not None or a.security_bits != 128 or
            not math.isfinite(a.input_bound) or a.input_bound <= 0):
        raise ValueError('[SNABF_NATIVE_INSPECTION] requires an explicit input spec; no emission or certification')
    if (a.mode != 'adaptive' or a.cut_profile != 'single-live-v1' or a.gadget_alpha is not None or
            a.cost_profile or a.cost_hardware_id or a.heterogeneous_scales or a.scale_mode or
            a.max_k is not None or a.source_arithmetic_fraction is not None or a.interval_mode or
            (a.precision_bits,a.switch_security_bits,a.log_n,a.q_count,a.min_slots) != (12,128,14,8,4096) or
            (a.prime_bw_min,a.prime_bw_max,a.aux_prime_margin_bits) != (24,59,2)):
        raise ValueError('[SNABF_NATIVE_INSPECTION] execution and parameter policies do not apply to source inspection')
    source, spec = a.mlir.read_bytes(), a.input_spec.read_bytes()
    inputs = json.loads(spec).get('inputs')
    if (not isinstance(inputs, list) or len(inputs) != 1 or not isinstance(inputs[0], dict) or
            inputs[0].get('domain') != [-a.input_bound, a.input_bound] or
            any(type(x) not in (int,float) for x in inputs[0]['domain'])):
        raise ValueError('[SNABF_NATIVE_INSPECTION] input spec must bind the requested domain')
    out = a.out_dir.absolute()
    if out.is_symlink() or out.exists():
        raise ValueError('[SNABF_OUTPUT_EXISTS] use a new output directory')
    ext = Path(os.environ.get('SNABF_EXTERN', ROOT/'extern')).resolve()
    sdk = Path(os.environ.get('SNABF_MLIR_ROOT', ext/'mlir24/usr/lib/llvm-24')).resolve()
    opt = a.snabf_opt or Path(os.environ.get('SNABF_OPT', ROOT/'build/native/ir/snabf-opt'))
    importer = opt.absolute().with_name('snabf-native-import'+('.exe' if os.name == 'nt' else ''))
    if not importer.is_file():
        raise ValueError('[SNABF_NATIVE_IMPORT_UNAVAILABLE] build the pinned snabf-native-import tool')
    env = {k: os.environ[k] for k in ('PATH','SystemRoot','TEMP','TMP','HOME','LD_LIBRARY_PATH') if k in os.environ}
    env['LD_LIBRARY_PATH'] = os.pathsep.join(map(str, (sdk/'lib', sdk.parent/'x86_64-linux-gnu',
                                                     ext/'highs-install/lib', env.get('LD_LIBRARY_PATH',''))))
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.snabf-inspect-', dir=out.parent) as folder:
        tmp = Path(folder)
        (tmp/'source.mlir').write_bytes(source)
        (tmp/'input.json').write_bytes(spec)
        result = subprocess.run([importer, 'source.mlir', '--input-spec', 'input.json'], cwd=tmp, env=env,
                                stdout=subprocess.PIPE, check=True)
        receipt = json.loads(result.stdout)
        expected = {'schema':'snabf.native-import.v1', 'frontend':'native-linalg', 'stage':'structural-inspection',
                    'source_sha256':hashlib.sha256(source).hexdigest(), 'input_spec_sha256':hashlib.sha256(spec).hexdigest()}
        if not isinstance(receipt, dict) or any(receipt.get(k) != v for k,v in expected.items()):
            raise ValueError('[SNABF_NATIVE_INSPECTION] unbound source/spec or wrong receipt type')
        publish = tmp/'publish'
        publish.mkdir()
        (publish/'import.json').write_bytes(result.stdout)
        publish.rename(out)
