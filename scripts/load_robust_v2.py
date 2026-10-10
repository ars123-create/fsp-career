"""Bounded read-only load against the local demo; writes no candidate data."""
import asyncio
import json
from pathlib import Path
from time import perf_counter

import httpx
import numpy as np
from sqlmodel import Session, select
from app.core.db import engine
from app.core.security import issue_access
from app.models import User


async def main():
    target=Path('output/validation/load-robust-v2.json')
    if target.exists(): raise SystemExit('Refusing to overwrite load results.')
    with Session(engine) as db:
        users=db.exec(select(User).where(User.is_demo == True, User.role == 'employer')).all()
        assert users, 'Local demo employer required'
        tokens=[issue_access(u) for u in users]
    rows=[]
    async with httpx.AsyncClient(base_url='http://127.0.0.1:8018',timeout=40.) as client:
        async def worker(index):
            for step in range(10):
                start=perf_counter()
                try:
                    response=await client.post('/api/search',headers={'Authorization':f'Bearer {tokens[index%len(tokens)]}'},
                        json={'specialization':'backend','grade':'Middle','ranking':{'mode':'robust','priorities':[] if step%2==0 else [['SQL','API']]}})
                    row={'client':index,'step':step,'seconds':perf_counter()-start,'status':response.status_code}
                    if response.status_code==200:
                        report=response.json()['robust']; row.update(solver=report['status'],gap=report['optimization_gap_points'],considered=report['considered'])
                    rows.append(row)
                except httpx.HTTPError as exc:
                    rows.append({'client':index,'step':step,'seconds':perf_counter()-start,'status':type(exc).__name__})
        await asyncio.gather(*(worker(i) for i in range(30)))
    result={'scenario':'30 concurrent clients, 2 local demo employer sessions, 10 searches/client; no database writes',
            'platform':'local macOS, SQLite, local uvicorn; not the earlier Docker/2CPU benchmark',
            'requests':len(rows),'sessions':len(tokens),'http_200':sum(r['status']==200 for r in rows),
            'p95_ms':float(np.quantile([r['seconds'] for r in rows],.95)*1000),
            'solver_nonoptimal':sum(r.get('solver') not in ('optimal',None) for r in rows),'rows':rows}
    with target.open('x') as f: json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))


if __name__=='__main__': asyncio.run(main())
