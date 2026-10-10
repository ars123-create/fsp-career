"""Frozen domain-transfer study, adapted from the verified supplementary harness.

The original run/response loop is reused without changing any algorithm.
This separate protocol is designed after primary/supplement quality was read.
Old source snapshots and decisions remain immutable. See reuse-smoke-v1.json.
"""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import sysconfig

import numpy as np
import scipy
import torch

from .. import benchmark as base
from ..simulation import FAMILIES, VARIANTS, make_world
from .registry import create as factory, description, METHODS, BASE_METHODS

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]


def source_files():
    paths = set(base.source_files())
    paths.update(HERE.glob('*.py'))
    paths.update(HERE.glob('*.json'))
    # Reuse only the strict row/provenance primitives, not its old scope flag.
    paths.add(HERE.parent/'extensions'/'analyze_supplement.py')
    paths.add(HERE.parent/'extensions'/'test_analyze_supplement.py')
    return sorted(paths)


def validate_plan(plan, *, old_ledger=None):
    if plan.get('stage') not in {'DOMAIN_TRANSFER','HARNESS_SMOKE'}:
        raise ValueError('Unknown domain-transfer execution stage')
    if type(plan.get('workers')) is not int or plan['workers'] not in (1,2):
        raise ValueError('workers must be 1 or 2')
    arms = plan.get('arms', [])
    if not arms:
        raise ValueError('nonempty arms required')
    names, used = set(), set()
    for arm in arms:
        if not isinstance(arm['name'],str) or not arm['name'] or arm['name'] in names:
            raise ValueError('distinct nonempty arm names required')
        names.add(arm['name'])
        if (type(arm['n']) is not int or not 1 <= arm['n'] <= 100
                or type(arm['d']) is not int or arm['d'] not in (1,2,3)):
            raise ValueError('invalid pool/dimensions')
        if not arm['families'] or len(set(arm['families'])) != len(arm['families']) or not set(arm['families']) <= set(FAMILIES):
            raise ValueError('invalid families')
        if arm.get('variant','base') not in VARIANTS:
            raise ValueError('invalid variant')
        seeds = arm['seeds']
        if (not seeds or any(type(s) is not int or s < 0 for s in seeds)
                or len(set(seeds)) != len(seeds) or used.intersection(seeds)):
            raise ValueError('invalid or overlapping world roots')
        used.update(seeds)
        methods = arm['methods']
        if not methods or any(not isinstance(m,str) or not m for m in methods) or len(set(methods)) != len(methods):
            raise ValueError('invalid methods')
        if (plan['stage'] == 'DOMAIN_TRANSFER' and set(methods) != set(METHODS)):
            raise ValueError('Domain study requires all six declared configurations')
        for method in methods:
            description(method)
    if old_ledger is not None:
        if used.intersection(old_ledger['world_seed_values']):
            raise ValueError('world root overlaps frozen primary seed ledger')
        for interval in old_ledger['world_seed_ranges']:
            if any(interval['start'] <= s <= interval['end_inclusive'] for s in used):
                raise ValueError('world root overlaps frozen primary seed range')
        final = old_ledger.get('final_namespace_check')
        if final and any(final['planned_start'] <= s <= final['planned_end_inclusive'] for s in used):
            raise ValueError('world root overlaps frozen primary final namespace')
    return sum(len(a['seeds'])*len(a['families'])*len(a['methods']) for a in arms)


def versions():
    return {'python':platform.python_version(), 'numpy':np.__version__,
            'scipy':scipy.__version__, 'torch':torch.__version__,
            'botorch':version('botorch'),'gpytorch':version('gpytorch'),
            'platform':platform.system(),'machine':platform.machine(),
            'python_abi':sysconfig.get_config_var('SOABI')}


def freeze(plan_path, out):
    plan = json.loads(plan_path.read_text())
    from .analyze import validate_study_plan
    validate_study_plan(plan)
    ledger = json.loads((HERE.parent/'SEED_LEDGER.json').read_text())
    expected = validate_plan(plan,old_ledger=ledger)
    extra_path = HERE/'DOMAIN_SEED_LEDGER.json'
    if not extra_path.is_file():
        raise RuntimeError('Domain seed ledger required before freeze')
    if extra_path.is_file():
        extra = json.loads(extra_path.read_text())
        if extra['primary_seed_ledger_sha256'] != sha256((HERE.parent/'SEED_LEDGER.json').read_bytes()).hexdigest():
            raise RuntimeError('Domain-transfer ledger points to a different primary seed ledger')
        validate_plan(plan,old_ledger=extra['previous_worlds'])
    primary_path = HERE.parent/'evidence'/'final-heldout-v1'/'protocol.json'
    primary_raw = primary_path.read_bytes()
    primary = json.loads(primary_raw)
    # Primary uses the standard "digest  protocol.json" checksum format;
    # this new harness's own seals contain only the digest.
    if sha256(primary_raw).hexdigest() != primary_path.with_suffix('.sha256').read_text().split()[0]:
        raise RuntimeError('Primary reference seal changed')
    for relative, expected_hash in primary['source_sha256'].items():
        if sha256((WORKSPACE/relative).read_bytes()).hexdigest() != expected_hash:
            raise RuntimeError(f'Primary source modified: {relative}')
    for name in METHODS:
        current = description(name)
        underlying = current.get('base_description', current)
        if underlying != primary['methods'][BASE_METHODS[name]]:
            raise RuntimeError('Underlying primary method configuration changed')
    out.mkdir(parents=True,exist_ok=False)
    hashes = {}
    for source in source_files():
        relative = source.relative_to(WORKSPACE)
        target = out/'snapshot'/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
        digest = sha256(target.read_bytes()).hexdigest()
        if sha256(source.read_bytes()).hexdigest() != digest:
            raise RuntimeError('Source changed during domain-transfer snapshot')
        hashes[str(relative)] = digest
    methods = sorted({m for a in plan['arms'] for m in a['methods']})
    protocol = {'schema':'fsp-domain-transfer-v1','utc':datetime.now(timezone.utc).isoformat(),
                'plan':plan,'expected_rows':expected,'versions':versions(),
                'methods':{m:description(m) for m in methods},
                'source_sha256':hashes,
                'previous_seed_ledger_sha256':
                    sha256(extra_path.read_bytes()).hexdigest() if extra_path.is_file() else None,
                'primary_holdout_quality_opened':True,
                'primary_reference':{'protocol_sha256':sha256(primary_raw).hexdigest(),
                    'source_sha256':primary['source_sha256'],
                    'methods':primary['methods'],'base_methods':BASE_METHODS},
                'scope':'Separate fixed domain-transfer study; does not amend prior protocols. Inference is defined only by its captured analysis contract.'}
    base.write_json(out/'protocol.json',protocol)
    (out/'protocol.sha256').write_text(sha256((out/'protocol.json').read_bytes()).hexdigest()+'\n')
    print(json.dumps({'status':'FROZEN','rows':expected,'source_files':len(hashes),
                      'protocol':str(out/'protocol.json')}),flush=True)


def read_protocol(path):
    if sha256(path.read_bytes()).hexdigest() != path.with_suffix('.sha256').read_text().strip():
        raise RuntimeError('Domain-transfer protocol digest mismatch')
    p = json.loads(path.read_text())
    if p['schema'] != 'fsp-domain-transfer-v1':
        raise ValueError('unsupported schema')
    if p['expected_rows'] != validate_plan(p['plan']):
        raise RuntimeError('Plan coverage changed')
    return p


def verify_snapshot(p, root):
    changed = [rel for rel,digest in p['source_sha256'].items()
               if not (root/rel).is_file() or sha256((root/rel).read_bytes()).hexdigest() != digest]
    if changed:
        raise RuntimeError(f'Domain-transfer snapshot drift: {changed[:5]}')


def auxiliary(world, selected):
    selected = list(selected)
    if not selected or len(selected) != len(set(selected)):
        raise ValueError('auxiliary metrics require a nonempty distinct slate')
    utility = world.truth[selected]
    return {'best_of_slate_regret':float(np.max(world.truth)-np.max(utility)),
            'worst_of_slate_utility':float(np.min(utility)),
            'mean_of_slate_utility':float(np.mean(utility))}


def execute(path, worker):
    p = read_protocol(path)
    out, plan = path.parent, p['plan']
    if WORKSPACE != out/'snapshot':
        raise RuntimeError('Execution must import captured domain-transfer snapshot')
    if not 0 <= worker < plan['workers']:
        raise ValueError('invalid worker')
    verify_snapshot(p,WORKSPACE)
    if p['versions'] != versions():
        raise RuntimeError('Domain-transfer runtime versions drifted')
    desc = {m:base.jsonable(description(m)) for m in p['methods']}
    if desc != p['methods']:
        raise RuntimeError('Domain-transfer method configuration drifted')
    torch.set_num_threads(2)
    base.write_json(out/f'preflight-{worker:03d}.json', {'status':'PASS','versions':versions(),
                    'methods':desc,'source_files_verified':len(p['source_sha256']),
                    'worlds_generated':0,'workspace':str(WORKSPACE)})
    count,failures = 0,0
    row_digest = sha256()
    with (out/f'rows-{worker:03d}.jsonl').open('x') as stream:
        for arm_index,arm in enumerate(plan['arms']):
            for seed in arm['seeds'][worker::plan['workers']]:
                for family_index,family in enumerate(arm['families']):
                    world = make_world(seed,family,n=arm['n'],d=arm['d'],variant=arm.get('variant','base'))
                    offset = (arm_index+arm['seeds'].index(seed)+family_index)%len(arm['methods'])
                    order = arm['methods'][offset:]+arm['methods'][:offset]
                    for method in order:
                        row = base.run(world,method,factory=factory)
                        row['arm'] = arm['name']
                        if row['status'] == 'PASS':
                            for step in row['trace']:
                                step['auxiliary'] = auxiliary(world,step['selection'])
                            row['auxiliary'] = row['trace'][-1]['auxiliary']
                        line = json.dumps(base.jsonable(row),allow_nan=False)+'\n'
                        stream.write(line)
                        row_digest.update(line.encode())
                        stream.flush()
                        count += 1
                        failures += row['status'] != 'PASS'
                    print(json.dumps({'arm':arm['name'],'seed':seed,'family':family,
                                      'rows':count,'failures':failures}),flush=True)
    verify_snapshot(p,WORKSPACE)
    base.write_json(out/f'completion-{worker:03d}.json', {'status':'COMPLETE','rows':count,
                    'failures':failures,'source_files_verified':len(p['source_sha256']),
                    'rows_sha256':row_digest.hexdigest()})


def merge(path):
    p = read_protocol(path)
    out,plan = path.parent,p['plan']
    if WORKSPACE != out/'snapshot':
        raise RuntimeError('Merge must import captured domain-transfer snapshot')
    verify_snapshot(p,WORKSPACE)
    if p['versions'] != versions():
        raise RuntimeError('Domain-transfer merge runtime versions drifted')
    expected = {(a['name'],s,f,m) for a in plan['arms'] for s in a['seeds']
                for f in a['families'] for m in a['methods']}
    actual = set()
    summaries = {a['name']:base.SummaryAccumulator(a['methods']) for a in plan['arms']}
    auxiliary_values = {a['name']:{m:[] for m in a['methods']} for a in plan['arms']}
    arms = {a['name']:a for a in plan['arms']}
    digest = sha256()
    with (out/'rows.jsonl').open('x') as target:
        for worker in range(plan['workers']):
            completion = json.loads((out/f'completion-{worker:03d}.json').read_text())
            preflight = json.loads((out/f'preflight-{worker:03d}.json').read_text())
            if (completion['status'] != 'COMPLETE' or preflight['status'] != 'PASS'
                    or preflight['methods'] != p['methods'] or preflight['versions'] != p['versions']
                    or preflight['workspace'] != str(out/'snapshot') or preflight['worlds_generated'] != 0
                    or preflight['source_files_verified'] != len(p['source_sha256'])
                    or completion['source_files_verified'] != len(p['source_sha256'])):
                raise RuntimeError('Worker preflight/completion mismatch')
            count,failures = 0,0
            worker_digest = sha256()
            for line in (out/f'rows-{worker:03d}.jsonl').open():
                row = json.loads(line)
                key = row['arm'],row['seed'],row['family'],row['method']
                if key not in expected or key in actual:
                    raise RuntimeError('Unexpected or duplicate domain-transfer row')
                arm = arms[row['arm']]
                if row['seed'] not in arm['seeds'][worker::plan['workers']]:
                    raise RuntimeError('Wrong domain-transfer shard')
                if row['n'] != arm['n'] or row['d'] != arm['d'] or row['variant'] != arm.get('variant','base'):
                    raise RuntimeError('World setting mismatch')
                actual.add(key)
                summaries[row['arm']].add(row)
                if row['status'] == 'PASS':
                    auxiliary_values[row['arm']][row['method']].append(row['auxiliary'])
                target.write(line)
                digest.update(line.encode())
                worker_digest.update(line.encode())
                count += 1
                failures += row['status'] != 'PASS'
            if (count != completion['rows'] or failures != completion['failures']
                    or worker_digest.hexdigest() != completion['rows_sha256']):
                raise RuntimeError('Worker row/failure count or digest mismatch')
    if actual != expected:
        raise RuntimeError('Missing domain-transfer rows')
    result = {name:acc.result() for name,acc in summaries.items()}
    for arm,methods in auxiliary_values.items():
        for method,values in methods.items():
            result[arm][method]['auxiliary_mean_successful_only'] = {
                key:float(np.mean([v[key] for v in values]))
                for key in ('best_of_slate_regret','worst_of_slate_utility','mean_of_slate_utility')
            } if values else None
    base.write_json(out/'summary.json',result)
    base.write_json(out/'completion.json',{'status':'COMPLETE','rows':len(actual),
                    'rows_sha256':digest.hexdigest(),'scope':plan['stage'],
                    'quality_of_real_hiring':'NOT_RUN'})
    print(json.dumps({'status':'COMPLETE','rows':len(actual)}),flush=True)


def launch(path):
    p = read_protocol(path)
    out = path.parent
    environment = dict(os.environ,PYTHONPATH=str(out/'snapshot'))
    children,logs = [],[]
    try:
        for worker in range(p['plan']['workers']):
            log = (out/f'worker-{worker:03d}.log').open('x')
            logs.append(log)
            children.append(subprocess.Popen([sys.executable,'-m','research.frontier_2026.domain_transfer.benchmark',
                            '--execute',str(path),'--worker',str(worker)],cwd=out/'snapshot',env=environment,
                            stdout=log,stderr=subprocess.STDOUT))
        codes = [child.wait() for child in children]
    finally:
        for log in logs:
            log.close()
    if any(codes):
        base.write_json(out/'launcher-failure.json',{'returncodes':codes,'partial_outputs_retained':True})
        raise RuntimeError('Domain-transfer worker failed')
    # Parent may be invoked from an editable checkout. Aggregation also runs
    # from the captured code, not whichever live module launched the workers.
    subprocess.run([sys.executable,'-m','research.frontier_2026.domain_transfer.benchmark',
                    '--merge',str(path)],cwd=out/'snapshot',env=environment,check=True)


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--freeze',type=Path)
    group.add_argument('--launch',type=Path)
    group.add_argument('--execute',type=Path)
    group.add_argument('--merge',type=Path)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--worker',type=int,default=0)
    args = parser.parse_args()
    if args.freeze:
        if args.out is None:
            parser.error('--freeze requires --out')
        freeze(args.freeze.resolve(),args.out.resolve())
    elif args.launch:
        launch(args.launch.resolve())
    elif args.merge:
        merge(args.merge.resolve())
    else:
        execute(args.execute.resolve(),args.worker)


if __name__ == '__main__':
    main()
