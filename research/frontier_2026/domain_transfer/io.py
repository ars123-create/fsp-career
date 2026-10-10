"""Read-only domain provenance loader, reusing separately audited row validators.

The old supplementary loader requires primary_quality_opened=False, which is
wrong here. Its logic is adapted explicitly, never changing its code/globals.
No simulator, optimizer or registry is imported by this module.
"""
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
import platform
import re
import sysconfig

import numpy as np
import scipy

from ..extensions.analyze_supplement import (
    VerifiedRun, REQUIRED_VERSIONS, digest, _json, expected_keys, row_key,
    validate_row, _verify_workers, plan_arms as _structural_arms,
)

HERE = Path(__file__).resolve().parent
IO_REL = 'research/frontier_2026/domain_transfer/io.py'
ANALYZER_REL = 'research/frontier_2026/extensions/analyze_supplement.py'
STATISTICS_REL = 'research/frontier_2026/statistics.py'
BASE_METHODS = {
    'mixed_kg_finer_quadrature':'mixed_kg_finer_quadrature',
    'P15_easy':'P15_easy',
    'linear_evoi_3_boundary_guard':'linear_evoi_3',
    'gp_topk_guard':'gp_topk_guard',
    'pabbo_rl':'pabbo_rl',
    'equal_boundary_guard':'equal',
}


def current_versions():
    return {'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
            'torch':version('torch'),'botorch':version('botorch'),'gpytorch':version('gpytorch'),
            'platform':platform.system(),'machine':platform.machine(),
            'python_abi':sysconfig.get_config_var('SOABI')}


def plan_arms(plan):
    if plan.get('stage') not in {'DOMAIN_TRANSFER','HARNESS_SMOKE'}:
        raise ValueError('Unsupported domain-transfer execution stage')
    # Only the old structural validator is reused. A shallow copy maps its
    # stage enum; the original protocol and upstream STAGES are not mutated.
    arms = _structural_arms(dict(plan,stage='HARNESS_SMOKE'))
    for arm in arms.values():
        names=set(arm['methods'])
        if not names <= BASE_METHODS.keys():
            raise ValueError('Unknown domain method')
        if plan['stage']=='DOMAIN_TRANSFER' and names != BASE_METHODS.keys():
            raise ValueError('Six declared configurations required')
    return arms


def verify_reference(protocol):
    reference=protocol.get('primary_reference',{})
    if reference.get('base_methods') != BASE_METHODS:
        raise ValueError('Primary alias binding differs')
    if not re.fullmatch('[0-9a-f]{64}',reference.get('protocol_sha256','')):
        raise ValueError('Primary reference digest missing')
    sources=reference.get('source_sha256',{})
    if not sources or any(protocol['source_sha256'].get(p)!=h for p,h in sources.items()):
        raise ValueError('Underlying primary source differs')
    for name, desc in protocol['methods'].items():
        if reference.get('methods',{}).get(BASE_METHODS[name]) != desc.get('base_description',desc):
            raise ValueError('Underlying primary configuration differs')
    ledger='research/frontier_2026/domain_transfer/DOMAIN_SEED_LEDGER.json'
    if protocol.get('previous_seed_ledger_sha256') != protocol['source_sha256'].get(ledger):
        raise ValueError('Seed ledger is not bound to manifest')


def compact_row(row):
    """Validate full raw traces first, retain scalars for the 550k-case analysis.

    Raw evidence and its digest remain complete. This avoids holding every
    repeated diagnostics object in RAM; no failed case is removed.
    """
    keys=('arm','seed','family','method','n','d','variant','status','constraint_status',
          'solver_fallback','restricted_terminal_search','construction_seconds','total_seconds',
          'regret','oracle_overlap_tie_aware','oracle_overlap','auxiliary','slots','answers',
          'regret_by_slot','error_type','error')
    result={key:row[key] for key in keys if key in row}
    result['step_seconds']=[step['seconds'] for step in row['trace']]
    result['guard_applications']=sum(step.get('diagnostics',{}).get('boundary_dominance_guard_applied') is True
                                     for step in row['trace'])
    result['calibration']={key:value for key,value in row.get('calibration',{}).items()
                           if key in {'status','metrics','scope','reason','queries'}}
    return result


def load_run(directory):
    """Stream and seal-check one completed run, never dropping a case."""
    directory = Path(directory).resolve()
    paths = {name:directory/name for name in ('protocol.json','completion.json','rows.jsonl')}
    hashes = {name:digest(path) for name,path in paths.items() if name != 'rows.jsonl'}
    seal = (directory/'protocol.sha256').read_text().strip()
    if not re.fullmatch('[0-9a-f]{64}',seal) or seal != hashes['protocol.json']:
        raise ValueError('Protocol seal mismatch')
    p, completion = (_json(paths[name].read_text()) for name in ('protocol.json','completion.json'))
    if p.get('schema') != 'fsp-domain-transfer-v1':
        raise ValueError('Unsupported supplementary schema')
    arms = plan_arms(p['plan'])
    expected = expected_keys(arms)
    methods = {m for a in arms.values() for m in a['methods']}
    if p.get('expected_rows') != len(expected) or type(p.get('expected_rows')) is not int or set(p.get('methods',{})) != methods:
        raise ValueError('Protocol expected coverage/method metadata mismatch')
    if (completion.get('status') != 'COMPLETE' or completion.get('rows') != len(expected)
            or type(completion.get('rows')) is not int or completion.get('scope') != p['plan']['stage']):
        raise ValueError('Run is not complete with the planned coverage')
    if p.get('primary_holdout_quality_opened') is not True or completion.get('quality_of_real_hiring') != 'NOT_RUN':
        raise ValueError('Scope flags changed or missing')
    versions = p.get('versions')
    if (not isinstance(versions,dict) or not REQUIRED_VERSIONS <= versions.keys()
            or any(not isinstance(versions[k],str) or not versions[k] for k in REQUIRED_VERSIONS)):
        raise ValueError('Complete captured runtime versions required')
    runtime = current_versions()
    if any(versions.get(k) != v for k,v in runtime.items()):
        raise ValueError('Analysis runtime differs from captured numerical dependencies')
    manifest = p.get('source_sha256',{})
    if not isinstance(manifest,dict) or not manifest:
        raise ValueError('Nonempty source manifest required')
    snapshot = (directory/'snapshot').resolve()
    for relative, expected_hash in manifest.items():
        source = (snapshot/relative).resolve()
        if not source.is_relative_to(snapshot) or Path(relative).is_absolute():
            raise ValueError('Unsafe source-manifest path')
        if not re.fullmatch('[0-9a-f]{64}',expected_hash) or digest(source) != expected_hash:
            raise ValueError(f'Snapshot source drift: {relative}')
    for relative, current in ((IO_REL,Path(__file__)),(ANALYZER_REL,HERE.parent/'extensions'/'analyze_supplement.py'),(STATISTICS_REL,HERE.parent/'statistics.py')):
        if manifest.get(relative) != digest(current):
            raise ValueError(f'Analysis implementation not bound to frozen source: {relative}')
    verify_reference(p)
    seen, rows, row_digest = set(), [], sha256()
    with paths['rows.jsonl'].open('rb') as stream:
        for raw in stream:
            row_digest.update(raw)
            row = _json(raw)
            key = row_key(row)
            if key not in expected or key in seen or type(row['seed']) is not int:
                raise ValueError(f'Unexpected or duplicate row: {key}')
            validate_row(row,arms[row['arm']])
            seen.add(key)
            rows.append(compact_row(row))
    if seen != expected:
        raise ValueError(f'Missing planned rows: {len(expected-seen)}')
    hashes['rows.jsonl'] = row_digest.hexdigest()
    if completion.get('rows_sha256') != hashes['rows.jsonl']:
        raise ValueError('Rows SHA256 mismatch')
    hashes.update(_verify_workers(directory,p,arms,expected,hashes['rows.jsonl']))
    for name in hashes:
        if digest(directory/name) != hashes[name]:
            raise ValueError(f'Input changed while reading: {name}')
    return VerifiedRun(directory,p,completion,tuple(rows),hashes)
