"""Restore lossless rows and pinned source snapshots into a new directory.

Original evidence is never edited. Only the workspace path in copied worker
preflights is relocated, with original bytes and a receipt retained explicitly.
The frozen analyzer, protocol, rows, seeds and numerical settings are unchanged.
"""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'research/frontier_2026/evidence'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def restore(name, destination):
    source = EVIDENCE / name
    target = destination / name
    target.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((source / 'protocol.json').read_text())
    assert digest(source / 'protocol.json') == (source / 'protocol.sha256').read_text().strip()
    for rel, expected in protocol['source_sha256'].items():
        original = (ROOT / rel).resolve()
        assert original.is_relative_to(ROOT) and digest(original) == expected, rel
        copied = target / 'snapshot' / rel
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, copied)
    for filename in ('protocol.json', 'protocol.sha256', 'completion.json', 'summary.json'):
        shutil.copyfile(source / filename, target / filename)
    receipt = {'scope': 'TRANSPORT_RELOCATION_ONLY', 'source': str(source),
               'protocol_sha256': digest(source / 'protocol.json'), 'workers': []}
    expected = json.loads((source / 'compressed_rows.json').read_text())
    merged_hash, count, size = hashlib.sha256(), 0, 0
    with gzip.open(source / 'rows.jsonl.gz', 'rb') as raw, (target / 'rows.jsonl').open('xb') as merged:
        for worker in range(protocol['plan']['workers']):
            end_name, pre_name = f'completion-{worker:03d}.json', f'preflight-{worker:03d}.json'
            end = json.loads((source / end_name).read_text())
            worker_hash = hashlib.sha256()
            with (target / f'rows-{worker:03d}.jsonl').open('xb') as shard:
                for _ in range(end['rows']):
                    line = raw.readline()
                    if not line.endswith(b'\n'):
                        raise ValueError('Missing or incomplete raw row')
                    shard.write(line); merged.write(line)
                    worker_hash.update(line); merged_hash.update(line)
                    size += len(line); count += 1
            assert worker_hash.hexdigest() == end['rows_sha256']
            shutil.copyfile(source / end_name, target / end_name)
            original = (source / pre_name).read_bytes()
            pre = json.loads(original)
            old_workspace = pre['workspace']
            pre['workspace'] = str((target / 'snapshot').resolve())
            (target / (pre_name + '.original')).write_bytes(original)
            (target / pre_name).write_text(json.dumps(pre, indent=2) + '\n')
            receipt['workers'].append({'worker': worker, 'original_workspace': old_workspace,
                'original_sha256': hashlib.sha256(original).hexdigest(),
                'relocated_sha256': digest(target / pre_name), 'rows_sha256': worker_hash.hexdigest()})
        assert raw.read(1) == b'', 'Extra unplanned rows'
    assert merged_hash.hexdigest() == expected['sha256_uncompressed']
    assert size == expected['bytes_uncompressed'] and count == protocol['expected_rows']
    receipt.update(status='PASS', rows=count, bytes=size, rows_sha256=merged_hash.hexdigest(),
                   source_files=len(protocol['source_sha256']))
    (target / 'RELOCATION.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'study': name, 'status': 'PASS', 'rows': count, 'source_files': len(protocol['source_sha256'])}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path, help='New directory; roughly 4.5 GB free space required')
    args = parser.parse_args()
    destination = args.destination.resolve()
    if destination.exists():
        parser.error('Destination must not exist; original evidence will not be overwritten')
    destination.mkdir(parents=True)
    for name in ('domain-quality-v1', 'domain-timing-v1'):
        restore(name, destination)
