"""Evaluate the emitted runtime on an *external copy*. No production release decision here."""
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile

from native_package import ROOT, pinned_go, snapshot
from verify_n1_equivalence import corpus, replay_slots, source_digest


def observe(label, source, module, out, *, seal, ring, level, slots, scale):
    module, source, out = module.resolve(strict=True), source.resolve(strict=True), out.absolute()
    if out.exists() or out.is_symlink() or out.resolve().is_relative_to(module):
        raise ValueError('observations require a new destination outside the package')
    original = source_digest(label, source)
    vectors = [r['x'] for r in corpus(label, source)]
    manifest = json.loads((module/'package-build.json').read_bytes())
    before = snapshot(module)
    expected = dict(before)
    del expected['package-build.json']
    plan = json.loads((module/'planned/program.json').read_bytes())
    provenance = plan.get('frontend_provenance', {})
    if (manifest.get('schema') != 'snabf.native-package-build.v1' or manifest.get('emit_sha256') != seal or
            manifest.get('files') != expected or manifest.get('program_file_sha256') != expected['planned/program.json']):
        raise ValueError('package no longer matches the build receipt')
    if (plan.get('emit_sha256') != seal or not isinstance(provenance, dict) or
            provenance.get('mode') != 'native-linalg' or provenance.get('source_sha256') != original):
        raise ValueError('program lacks native provenance bound to the original source')
    q = Fraction(scale)
    if q <= 0 or str(q) != scale:
        raise ValueError('exact physical sink scale must be canonical and positive')
    out.parent.mkdir(parents=True, exist_ok=True)
    go, env = pinned_go()
    with tempfile.TemporaryDirectory(dir=out.parent, prefix='.snabf-observe-') as tmp:
        tmp = Path(tmp)
        stage, records = tmp/'module', tmp/'records'
        shutil.copytree(module, stage)
        if snapshot(stage) != before:
            raise ValueError('package changed while taking observation snapshot')
        records.mkdir()
        entry = stage/'cmd/snabf-observe'
        entry.mkdir(parents=True)
        shutil.copyfile(ROOT/'test/compiler/rfc004/testdata/n1_observe/main.go',entry/'main.go')
        req = json.dumps(dict(emit_sha256=seal,source_sha256=original,inputs=vectors), allow_nan=False).encode()
        data = records/'trials.jsonl'
        with data.open('xb') as f:
            subprocess.run([go,'run','-mod=readonly','./cmd/snabf-observe'], cwd=stage, env=env,
                           input=req, stdout=f, check=True)
        integrity = snapshot(stage)
        del integrity['cmd/snabf-observe/main.go']
        if integrity != before or snapshot(module) != before:
            raise ValueError('evaluation modified package files')
        report = replay_slots(label,source,data,seal=seal,ring=ring,level=level,slots=slots)
        with data.open(encoding='utf-8') as f:
            for line in f:
                row=json.loads(line)
                if (row['scale_value'] != scale or row['scale_f64'] != float(q) or not math.isfinite(row['scale_f64'])
                        or type(row['swk_bytes']) is not int or row['swk_bytes'] < 0
                        or type(row['eval_ns']) is not int or row['eval_ns'] <= 0):
                    raise ValueError('observed physical metadata does not match the checked sink')
        report.update(trials_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),
                      program_file_sha256=expected['planned/program.json'], scale_value=scale)
        (records/'run_observations.json').write_text(json.dumps(report, allow_nan=False, indent=2)+'\n', encoding='utf-8')
        records.rename(out)
    return report
