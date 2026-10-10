"""Replay disclosed size-stress inputs with raw and explicitly guarded controls."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import torch

from .. import benchmark as base
from ..registry import create as raw_factory
from ..simulation import make_world
from .registry import create as guarded_factory, description

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
ARMS=('n18-d1','n18-d2','n50-d3','n100-d3')
PAIRS=(('linear_evoi_3','linear_evoi_3_boundary_guard'),('equal','equal_boundary_guard'))


def compact(row):
    return {'status':row['status'],'constraint_status':row['constraint_status'],
            'history':base.jsonable(row.get('history')),'regret':row.get('regret'),
            'overlap':row.get('oracle_overlap_tie_aware'),
            'selections':[s['selection'] for s in row['trace']],
            'queries':[s['query_id'] for s in row['trace']]}


def main():
    out=HERE/'output'/'guard-disclosed-replay-v1'
    out.mkdir(parents=True,exist_ok=False)
    parent=HERE.parent/'evidence'/'stress-comparison-v1'
    inputs={};cases=[]
    for arm in ARMS:
        path=parent/arm/'rows.jsonl'
        inputs[str(path.relative_to(ROOT))]=sha256(path.read_bytes()).hexdigest()
        for line in path.open():
            row=json.loads(line)
            if row['method'] in dict(PAIRS): cases.append((arm,row))
    sources={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest()
             for p in base.source_files()+[Path(__file__),HERE/'registry.py']}
    plan={'status':'FROZEN_REPLAY','scope':'Same disclosed size-stress worlds, not fresh evidence',
          'utc':datetime.now(timezone.utc).isoformat(),'cases':len(cases),'arms':ARMS,
          'pairs':PAIRS,'source_sha256':sources,'input_sha256':inputs,
          'guard_descriptions':{b:description(b) for a,b in PAIRS}}
    base.write_json(out/'protocol.json',plan)
    torch.set_num_threads(2)
    count=old_diff=new_diff=failures=guard_changes=0
    with (out/'comparisons.jsonl').open('x') as stream:
        for arm,previous in cases:
            world=make_world(previous['seed'],previous['family'],n=previous['n'],d=previous['d'],variant=previous['variant'])
            raw=base.run(world,previous['method'],factory=raw_factory)
            guarded=base.run(world,dict(PAIRS)[previous['method']],factory=guarded_factory)
            a,b,c=compact(previous),compact(raw),compact(guarded)
            row={'arm':arm,'seed':previous['seed'],'family':previous['family'],
                 'raw_method':previous['method'],'guarded_method':guarded['method'],
                 'old_raw_equals_replayed_raw':a==b,'raw_equals_guarded':b==c,
                 'previous':a,'raw':b,'guarded':c,
                 'guard_applications':sum(s.get('diagnostics',{}).get('boundary_dominance_guard_applied') is True for s in guarded['trace'])}
            stream.write(json.dumps(row)+'\n');stream.flush()
            count+=1;old_diff+=a!=b;new_diff+=b!=c;guard_changes+=row['guard_applications']
            failures+=raw['status']!='PASS' or guarded['status']!='PASS' or guarded['constraint_status']!='PASS'
    drift=[p for p,h in sources.items() if sha256((ROOT/p).read_bytes()).hexdigest()!=h]
    result={'status':'PASS' if count==len(cases) and not old_diff and not failures and not drift else 'FAIL',
            'paired_cases':count,'method_cases':2*count,'old_raw_replay_differences':old_diff,
            'guard_selection_or_query_or_metric_differences':new_diff,'guard_applications':guard_changes,
            'execution_or_guard_contract_failures':failures,'source_drift':drift,
            'rows_sha256':sha256((out/'comparisons.jsonl').read_bytes()).hexdigest(),
            'scope':'A diagnostic replay of disclosed worlds, not a global equivalence or new quality claim'}
    base.write_json(out/'result.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
