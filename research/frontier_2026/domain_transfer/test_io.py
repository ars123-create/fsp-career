"""Literal protocol/row corruption fixtures; no quality world generation."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from . import io
from ..extensions.test_analyze_supplement import write_run, rows_for, failed


def fixture(directory, *, rows=None, workers=2):
    p={'stage':'HARNESS_SMOKE','workers':workers,'arms':[{
        'name':'literal','n':6,'d':2,'variant':'base','families':['linear','interaction'],
        'methods':['P15_easy','mixed_kg_finer_quadrature'],'seeds':[19001,19002,19003]}]}
    directory=write_run(directory,p=p,rows=rows)
    path=directory/'protocol.json';protocol=json.loads(path.read_text())
    old_sources=deepcopy(protocol['source_sha256'])
    extra={io.IO_REL:Path(io.__file__).read_bytes(),
           'research/frontier_2026/domain_transfer/DOMAIN_SEED_LEDGER.json':b'{"fixture": true}\n'}
    for name,data in extra.items():
        dest=directory/'snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
        protocol['source_sha256'][name]=io.digest(dest)
    protocol.update(schema='fsp-domain-transfer-v1',primary_holdout_quality_opened=True,
                    versions=io.current_versions(),
                    previous_seed_ledger_sha256=protocol['source_sha256']['research/frontier_2026/domain_transfer/DOMAIN_SEED_LEDGER.json'])
    protocol['primary_reference']={'protocol_sha256':'e'*64,'source_sha256':old_sources,
        'base_methods':io.BASE_METHODS,
        'methods':{base:{'technical_fixture':base} for base in io.BASE_METHODS.values()}}
    path.write_text(json.dumps(protocol));(directory/'protocol.sha256').write_text(io.digest(path))
    for worker in range(workers):
        path=directory/f'preflight-{worker:03d}.json';v=json.loads(path.read_text())
        v.update(versions=protocol['versions'],source_files_verified=len(protocol['source_sha256']))
        path.write_text(json.dumps(v))
        path=directory/f'completion-{worker:03d}.json';v=json.loads(path.read_text())
        v['source_files_verified']=len(protocol['source_sha256']);path.write_text(json.dumps(v))
    return directory


def mutate(directory,filename,change,*,seal=True):
    p=directory/filename;v=json.loads(p.read_text());change(v);p.write_text(json.dumps(v))
    if filename=='protocol.json' and seal:(directory/'protocol.sha256').write_text(io.digest(p))


def test_valid_multishard_preserves_all_cases_without_diagnostic_bloat(tmp_path):
    r=io.load_run(fixture(tmp_path/'run'))
    assert len(r.rows)==12 and all('step_seconds' in row and 'trace' not in row for row in r.rows)
    assert r.protocol['plan']['stage']=='HARNESS_SMOKE'


@pytest.mark.parametrize('change',[
    lambda p:p.update(primary_holdout_quality_opened=False),
    lambda p:p.update(schema='fsp-supplement-v1'),
    lambda p:p['plan'].update(stage='DEVELOPMENT'),
    lambda p:p['versions'].update(torch='wrong'),
    lambda p:p['versions'].update(gpytorch='wrong'),
    lambda p:p['versions'].update(botorch='wrong'),
    lambda p:p['versions'].update(python_abi='wrong'),
    lambda p:p['versions'].update(machine='wrong'),
    lambda p:p['primary_reference']['base_methods'].update(P15_easy='equal'),
    lambda p:p['primary_reference']['methods']['P15_easy'].update(technical_fixture='changed'),
    lambda p:p['primary_reference']['source_sha256'].update(missing='a'*64),
    lambda p:p.update(previous_seed_ledger_sha256='a'*64),
    lambda p:p['plan']['arms'][0]['seeds'].append(19001),
    lambda p:p['plan']['arms'][0]['methods'].append('unknown'),
    lambda p:p.update(expected_rows=11),
])
def test_resealed_inconsistent_protocol_rejected(tmp_path,change):
    p=fixture(tmp_path/'run');mutate(p,'protocol.json',change)
    with pytest.raises((ValueError,KeyError)):io.load_run(p)


@pytest.mark.parametrize('filename,change',[
    ('preflight-000.json',lambda v:v.update(worlds_generated=1)),
    ('preflight-000.json',lambda v:v['versions'].update(scipy='different')),
    ('completion-001.json',lambda v:v.update(rows=0)),
    ('completion-001.json',lambda v:v.update(failures=1)),
    ('completion.json',lambda v:v.update(status='RUNNING')),
    ('completion.json',lambda v:v.update(rows_sha256='a'*64)),
])
def test_worker_and_completion_corruption_rejected(tmp_path,filename,change):
    p=fixture(tmp_path/'run');mutate(p,filename,change)
    with pytest.raises(ValueError):io.load_run(p)


def test_digest_and_source_drift_rejected(tmp_path):
    p=fixture(tmp_path/'a');(p/'protocol.json').write_text((p/'protocol.json').read_text()+' ')
    with pytest.raises(ValueError,match='seal'):io.load_run(p)
    p=fixture(tmp_path/'b');(p/'snapshot'/io.IO_REL).write_text('changed')
    with pytest.raises(ValueError,match='drift'):io.load_run(p)


def test_failure_retained_and_not_a_zero_regret(tmp_path):
    p={'stage':'HARNESS_SMOKE','workers':2,'arms':[{'name':'literal','n':6,'d':2,'variant':'base',
       'families':['linear','interaction'],'methods':['P15_easy','mixed_kg_finer_quadrature'],
       'seeds':[19001,19002,19003]}]}
    rows=rows_for(p);rows[0]=failed(rows[0])
    result=io.load_run(fixture(tmp_path/'run',rows=rows))
    bad=[r for r in result.rows if r['status']=='FAIL']
    assert len(result.rows)==12 and len(bad)==1 and 'regret' not in bad[0]


def test_loader_does_not_import_simulator_or_optimizer():
    code='import sys; import research.frontier_2026.domain_transfer.io; assert "research.frontier_2026.simulation" not in sys.modules; assert "torch" not in sys.modules'
    subprocess.run([sys.executable,'-c',code],check=True,capture_output=True,text=True)
