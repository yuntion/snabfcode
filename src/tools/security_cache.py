"""Private, local cache of concrete estimator measurements (Linux product driver).

The cache owner and pinned estimator are trusted. Hashes detect corruption;
they do not authenticate an imported archive. No public cache import or cached
admission decisions. The native planner/checker reapplies policy on every run.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile

from pass3_security import PIN

MODEL = 'LWE.estimate.rough/classical ADPS16; GSA uSVP, dual_hybrid'
SECRET = 'iid uniform ternary'
ERROR = 'discrete Gaussian sigma=3.2 (estimator model for bounded sampler)'


def request(role, n, q, p, modulus):
    return dict(schema='snabf.measure.v1', estimator=PIN, model=MODEL, secret=SECRET,
                error=ERROR, samples='unbounded', role=role, ring=n, q=list(q),
                p=list(p), modulus=str(modulus))


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def checked_attacks(a, r):
    # Pass 4's existing evidence schema represents a missing finite estimate
    # for an optional target by "inf". Its native checker rejects that proposal.
    # Source/admission evidence must always be finite.
    candidate = r['role'].startswith('target.')
    if not isinstance(a, dict) or not {'usvp', 'dual_hybrid'} <= a.keys() or not all(
            (type(x) in (float, int) and math.isfinite(x) and x > 0) or
            (candidate and x == 'inf') for x in a.values()):
        raise ValueError('[SNABF_INVALID_ESTIMATE] missing or nonfinite attack costs')
    return a


def measure(requests, cache, estimator):
    cache = Path(cache)
    cache.mkdir(mode=0o700, parents=True, exist_ok=True)
    st = cache.lstat()
    if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.geteuid() or st.st_mode & 0o077:
        raise ValueError('[SNABF_UNTRUSTED_CACHE] expected owner-private directory')
    results, missing = {}, {}
    for r in requests:
        key = digest(r)
        try:
            fd = os.open(cache/key, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except FileNotFoundError:
            missing[key] = r
            continue
        with os.fdopen(fd) as f:
            st = os.fstat(f.fileno())
            if not stat.S_ISREG(st.st_mode) or st.st_uid != os.geteuid() or st.st_mode & 0o077:
                raise ValueError('[SNABF_UNTRUSTED_CACHE] expected private regular receipt')
            obj = json.load(f)
        if obj['request'] != r or obj['checksum'] != digest([r, obj['attacks']]):
            raise ValueError('[SNABF_INVALID_ESTIMATE] receipt mismatch')
        results[key] = checked_attacks(obj['attacks'], r)
    if missing:
        sage = shutil.which('sage')
        if not sage or not estimator or not (Path(estimator)/'.git').exists():
            raise ValueError('[SNABF_SECURITY_ORACLE_UNAVAILABLE] missing cached estimate for key='+next(iter(missing)))
        proc = subprocess.run([sage, '-python', str(Path(__file__).resolve()), str(estimator)],
                              input=json.dumps(missing), text=True, stdout=subprocess.PIPE, check=True)
        fresh = json.loads(proc.stdout)
        if fresh.keys() != missing.keys():
            raise ValueError('[SNABF_INVALID_ESTIMATE] incomplete oracle response')
        for key, r in missing.items():
            a = results[key] = checked_attacks(fresh[key], r)
            f = tempfile.NamedTemporaryFile(mode='w', dir=cache, delete=False)
            tmp = Path(f.name)
            try:
                with f:
                    json.dump(dict(request=r, attacks=a, checksum=digest([r,a])), f, allow_nan=False)
                os.replace(tmp, cache/key)
            finally:
                tmp.unlink(missing_ok=True)
    return results


def worker(root, pending):
    root = Path(root).resolve()
    rev = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain', '--', 'estimator'], text=True)
    if rev != PIN or dirty:
        raise ValueError('[SNABF_SECURITY_ORACLE_UNAVAILABLE] estimator revision/source mismatch')
    sys.path.insert(0, str(root))
    from estimator import LWE, ND
    from sage.all import infinity, log
    measured, result = {}, {}
    for key, r in pending.items():
        pair = n, modulus = r['ring'], int(r['modulus'])
        if pair not in measured:
            params = LWE.Parameters(n=n, q=modulus, Xs=ND.Uniform(-1,1), Xe=ND.DiscreteGaussian(3.2), m=infinity)
            raw = LWE.estimate.rough(params, quiet=True, catch_exceptions=False)
            a = {str(k):float(log(v['rop'],2)) for k,v in raw.items()}
            measured[pair] = {k:('inf' if v == math.inf else v) for k,v in a.items()}
        result[key] = checked_attacks(measured[pair], r)
    return result


if __name__ == '__main__':
    print(json.dumps(worker(sys.argv[1], json.load(sys.stdin)), allow_nan=False))
