"""Check every archived member, then recompute paired/single CKKS slot errors from the saved corpus."""
import argparse
import hashlib
import json
import math
import tarfile
from pathlib import Path


def verify(folder):
    archive = folder / 'evidence.tar.gz'
    manifest = json.loads((folder / 'sha256.json').read_bytes())

    def require(ok, why):
        if not ok:
            raise ValueError(why)

    def digest(f):
        h = hashlib.sha256()
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
        return h.hexdigest()

    with archive.open('rb') as data:
        require(digest(data) == manifest['archive'], 'archive digest')
    summary = json.loads((folder / 'acceptance.json').read_bytes())
    corpora, seen = {}, set()
    groups = {f'{net}/paired': (net, m) for net, m in summary['paired'].items()}
    for arm, m in summary['boundary'].items():
        net = 'cross_residual' if arm == 'bypass' else 'reuse64'
        groups[f'{net}/{arm}'] = net, m
    require(set(groups) == {'control/paired', 'strided64/paired', 'reuse64/paired', 'residual/paired',
                            'cross_residual/bypass', 'reuse64/positive', 'reuse64/security'}, 'missing groups')
    with tarfile.open(archive, 'r|gz') as tar:
        for entry in tar:
            key = entry.name
            require(entry.isfile() and key not in seen and key in manifest['members'], key)
            seen.add(key)
            with tar.extractfile(entry) as data:
                if key == 'acceptance.json' or key in {f'{net}/corpus.json' for net, _ in groups.values()}:
                    raw = data.read()
                    checksum = hashlib.sha256(raw).hexdigest()
                    if key == 'acceptance.json':
                        require(json.loads(raw) == summary, 'external acceptance differs from archive')
                    else:
                        corpora[key.split('/')[0]] = json.loads(raw)
                else:
                    checksum = digest(data)
            require(checksum == manifest['members'][key], 'member digest: ' + key)
    require(seen == set(manifest['members']) and 'acceptance.json' in seen, 'manifest coverage')
    done, evaluated, checked, logical = set(), 0, 0, 0
    peak, paired_peak = 0., 0.
    with tarfile.open(archive, 'r|gz') as tar:
        for entry in tar:
            group, _, name = entry.name.rpartition('/')
            if name != 'trials.jsonl' or group not in groups:
                continue
            net, metrics = groups[group]
            paired = group.endswith('/paired')
            fields = ['decoded_direct', 'decoded_legacy'] if paired else ['decoded64']
            maxima = [0.] * len(fields)
            ss = [0.] * len(fields)
            diff_max = diff_ss = 0.
            slots = rows = 0
            with tar.extractfile(entry) as data:
                for line in data:
                    row = json.loads(line)
                    require(row['trial'] == rows, 'trial order: ' + group)
                    truth = corpora[net][rows]['truth']
                    values = [row[x] for x in fields]
                    width = len(values[0])
                    require(len(truth) > 0 and width >= len(truth) and width % len(truth) == 0 and
                            all(len(a) == width for a in values), 'slot shape: ' + group)
                    for i, a in enumerate(values):
                        errors = [abs(y - truth[j % len(truth)]) for j, y in enumerate(a)]
                        require(all(math.isfinite(e) and e <= 2**-12 for e in errors), 'slot error: ' + group)
                        maxima[i] = max(maxima[i], max(errors))
                        ss[i] += math.fsum(e*e for e in errors)
                    if paired:
                        diffs = [abs(a-b) for a, b in zip(*values)]
                        diff_max = max(diff_max, max(diffs))
                        diff_ss += math.fsum(e*e for e in diffs)
                    logical += len(truth) * len(fields)
                    rows += 1
                    slots += width
            require(rows == len(corpora[net]) == metrics['trials'] == 64, 'corpus coverage: ' + group)
            expected = metrics['slots_per_arm' if paired else 'physical_encoded_slots_checked']
            require(slots == expected, 'slot count: ' + group)
            ref_max = metrics['linf_direct_legacy'] if paired else [metrics['linf']]
            ref_mse = metrics['mse_direct_legacy'] if paired else [metrics['mse']]
            close = lambda a, b: math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-25)
            require(all(close(a,b) for a,b in zip(maxima,ref_max)) and
                    all(close(s/slots,b) for s,b in zip(ss,ref_mse)), 'reported norm mismatch: ' + group)
            if paired:
                require(close(diff_max,metrics['paired_linf']) and close(diff_ss/slots,metrics['paired_mse']) and
                        diff_max <= 2**-12, 'paired norm mismatch: ' + group)
            evaluated += len(fields)*rows
            checked += len(fields)*slots
            peak = max(peak,*maxima)
            paired_peak = max(paired_peak,diff_max)
            done.add(group)
    require(done == set(groups), 'missing slot records')
    return dict(archive_sha256=manifest['archive'], members_verified=len(seen),
                model_evaluations=evaluated, logical_outputs=logical, physical_slots=checked,
                threshold=2**-12, misses=0, max_error=peak, max_paired_error=paired_peak)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder', type=Path)
    print(json.dumps(verify(p.parse_args().folder), indent=2))
