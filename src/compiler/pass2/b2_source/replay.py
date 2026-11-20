"""Replay one public-key source support lemma, NOT a noise certificate.

Scope: single-ring direct Go, reviewed epochs.Bind/Encrypt, unmodified Go
module-cache sources and defaults, key reuse within one Runtime returned by Bind.
The input envelope is a quantifier on fixed plaintexts, not an observation or
a runtime range check. The native verifier is a trusted local build. Neither
the sidecar nor this calculation attests to a deployed binary or live key.
"""
import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'src/tools'))
from replay_noise_audit import checked_inputs, independent, load, need, same

MODULE = 'github.com/tuneinsight/lattigo/v6@v6.2.0'
H1 = 'h1:HZrksD5u87bOr/4hWHI1Jhps14Tafdvb84Fxmi3dou0='
RUNTIME_SHA = '174c52c0600893832562e871d1d72cec63de533ec8ebef4020b13977ce14bf25'


def runtime_identity(root):
    # Cover the generated model's local Go packages, including init functions
    # in sibling files that could override rlwe.DefaultXs/Xe before Bind.
    files = [root/'go.mod', root/'go.sum']
    for pkg in ('', 'gadget', 'ringswitch/epochs', 'ringswitch/epochs/heirckks'):
        files.extend(p for p in (root/'src/backend'/pkg).glob('*.go') if not p.name.endswith('_test.go'))
    listing = hashlib.sha256()
    for path in sorted(files, key=lambda p: p.relative_to(root).as_posix()):
        data = path.read_bytes().replace(b'\r\n', b'\n')  # Git LF vs Windows worktrees
        listing.update(f'{hashlib.sha256(data).hexdigest()}  {path.relative_to(root).as_posix()}\n'.encode())
    digest = listing.hexdigest()
    need(digest == RUNTIME_SHA, 'unreviewed direct Go runtime')
    return digest


def source_identity(module):
    # Go dirhash.Hash1: hash the actual source files, not the adjacent
    # .ziphash, a version label, the compiler's estimator model, or samples.
    listing = hashlib.sha256()
    for path in sorted(module.rglob('*'), key=lambda p: p.relative_to(module).as_posix()):
        need(not path.is_symlink(), 'symlink in module cache')
        if path.is_dir():
            continue
        name = path.relative_to(module).as_posix()
        need(path.is_file() and '\n' not in name, 'invalid module entry')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        listing.update(f'{digest}  {MODULE}/{name}\n'.encode())
    h1 = 'h1:' + base64.b64encode(listing.digest()).decode('ascii')
    need(h1 == H1, 'unsupported Lattigo sampler source')
    return dict(module=MODULE, module_h1=h1, direct_go_sources_sha256_lf=runtime_identity(ROOT),
                runtime_attested=False)


def pre_moddown(n, q, p):
    # ring/sampler_ternary.go: LUT {0,1,qi-1} for Xs (both s and u).
    # sampler_gaussian.go: accept nonnegative v <= binary64(19.2), THEN
    # uint64(v+0.5). Even rounding the sum, it is <= 19.75 < 20. No MGF,
    # exact ternary weights, symmetry, or probabilistic independence needed.
    e = 19
    need(type(n) is int and n >= 4 and (n & (n-1)) == 0 and q and p, 'unsupported public-key QP source')
    need(all(type(m) is int and m > 2*e for m in [*q, *p]), 'unproved centered small-norm basis extension')
    # pk0+pk1*s = e_pk. Before the TWO component-wise ModDowns the
    # combined noise is u*e_pk + e0 + e1*s in Z[X]/(X^N+1) modulo Q*P0.
    # Each product coefficient is a signed sum of N products, even with
    # arbitrary dependence. encryptZeroPk(*Ciphertext) uses levelP=0.
    bound = n*e + e + n*e
    modulus = math.prod(q) * p[0]
    need(2*bound < modulus, 'pre-ModDown centered lift not established')
    return dict(units='integer negacyclic coefficient before both ModDowns',
                secret_and_ephemeral_abs=1, error_abs=e, combined_abs=bound,
                active_p=str(p[0]), centered_modulus=str(modulus))


def prepare(native, checked, params, module):
    identity = source_identity(module)
    native = native.resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix='snabf-b2-source-') as tmp:
        stage = Path(tmp)
        snapshot = stage / 'checked.mlir'
        shutil.copyfile(checked, snapshot)
        # Direct lowering verifies an existing concrete seal and never mints
        # one. It MUST precede check-plan, which can seal a fresh graph.
        _, _, seal, envelope = checked_inputs(snapshot)
        subprocess.run([native, '--allow-unregistered-dialect', '--verify-each', '--mlir-print-op-on-diagnostic=false',
                        '--snabf-lower-ckks=direct-go=true', '--snabf-check-plan=noise-audit=noise.json',
                        '--snabf-bind-lattigo=ring-go=model.go audit=program.json', snapshot, '-o', os.devnull],
                       cwd=stage, check=True)
        fresh = stage / 'program.json'
        program = load(fresh)
        need(same(load(params), program), 'sidecar differs from native re-emission')
        physical = independent(snapshot, fresh, stage / 'noise.json')
        model_sha = hashlib.sha256((stage / 'model.go').read_bytes()).hexdigest()

    epoch, = program['epochs']  # independent() also checked complete Q/P against Pass3
    ins = program['instructions']
    src = ins[0]
    n, q, p = epoch['n'], epoch['q'], epoch['p']
    width, slots = program['width'], program['slots']
    need(src['op'] == 'source' and src['Level'] == len(q)-1 and sum(i['op'] == 'source' for i in ins) == 1,
         'expected one max-level EncryptNew plaintext')
    need(0 < width <= slots <= n//2 and slots % width == 0 and slots & (slots-1) == 0,
         'unsupported periodic plaintext shape')
    # Runtime.Bind generates one key per Galois element, not per instruction
    # or site. Group by physical element within this one source key domain.
    galois = {}
    for i in ins:
        if i['op'] in ('baby', 'giant'):
            element = pow(5, i['rotation'], 2*n)
            galois.setdefault(str(element), []).append(i['ID'])
    need(sorted(map(int, galois)) == epoch['galois'], 'Galois key inventory mismatch')
    relin = [i['ID'] for i in ins if i['op'] == 'relin']
    return dict(schema='snabf.b2.pk-source-support.v1', source_model=identity, emit_sha256=seal,
                encoding_program_sha256=physical['program_sha256'], emitted_go_sha256=model_sha,
                source=src['ID'], level_q=src['Level'], ring_n=n,
                fixed_real_input=dict(scope='declared_domain', abs_le=str(envelope), width=width, periodic_slots=slots),
                samplers=dict(xs='Ternary{P:2/3.0,H:0}', xe='DiscreteGaussian{Sigma:3.2,Bound:19.2}',
                              public_a='uniform QP'),
                key_scope=dict(key_domain=epoch['key_id'], shared_per_bind=['s', 'pk(a,e_pk)'],
                               per_encrypt=['u', 'e0', 'e1'], galois_uses=galois, relin_uses=relin,
                               proven_independent_draws=[]),
                support=pre_moddown(n, q, p),
                remaining_source_terms=['moddown_rounding_with_reused_secret', 'encoding_and_decoded_component_error',
                                        'common/validated_joint_law_and_domain'],
                probabilistic_cert_issued=False)


def compare(receipt, candidate):
    # Exact schema/values/types, including extra independence/proof claims.
    need(same(receipt, candidate), 'source support receipt differs from recomputation')


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    for opt in ['snabf-opt', 'checked', 'params', 'lattigo-src']:
        cli.add_argument('--' + opt, required=True, type=Path)
    cli.add_argument('--compare', type=Path, help='check a saved preparation receipt after recomputing it')
    args = cli.parse_args()
    try:
        receipt = prepare(args.snabf_opt, args.checked, args.params, args.lattigo_src)
        if args.compare:
            compare(receipt, load(args.compare))
        print(json.dumps(receipt, sort_keys=True, allow_nan=False))
    except (OSError, KeyError, IndexError, TypeError, ValueError, ArithmeticError, subprocess.CalledProcessError) as exc:
        cli.exit(1, f'{exc}\n')
