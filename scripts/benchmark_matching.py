"""Frozen authored synthetic cases; not hiring validation or a human benchmark."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import math
import statistics
from unittest.mock import patch
from app.matching import evaluate
from app.schemas import SearchInput
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'validation/matching-v1.json'
assert hashlib.sha256(p.read_bytes()).hexdigest()==(ROOT/'validation/matching-v1.sha256').read_text().split()[0]
d=json.loads(p.read_text()); people=[]
for row in d['profiles']:
 i,s,g,w,lo,hi,score,families,age,fsp,stack=row
 people.append(SimpleNamespace(user_id=i,specialization=s,grade=g,work_format=w,salary_min=lo*1000,salary_max=hi*1000,published=True,stack=stack,
  skills={skill:{'score':score,'families':families,'checked_at':d['as_of']-age*86400} for skill in stack},achievements=[{'verified':True}]*fsp))
rows=[]
for raw in d['queries']:
 q=SearchInput(**{k:v for k,v in raw.items() if k not in ('id','labels')}); labels=raw['labels']
 # Scenario admissibility is explicit, independent of ranking terms.
 allowed={c.user_id for c in people if c.specialization==q.specialization and c.grade==q.grade and c.work_format==q.work_format and c.salary_min<=q.salary_max and c.salary_max>=q.salary_min and all(s in c.skills and c.skills[s]['score']>=50 for s in q.skills)}
 rankings={}
 rankings['keyword']=sorted(people,key=lambda c:(-len(set(q.skills)&set(c.stack)),c.user_id))[:5]
 rankings['structured']=sorted([c for c in people if c.user_id in allowed],key=lambda c:(-statistics.mean(c.skills[s]['score'] for s in q.skills),c.user_id))[:5]
 with patch('app.matching.now',return_value=d['as_of']):
  evaluated=[(c,evaluate(c,q)) for c in people]
 rankings['evidence']=[c for c,r in sorted([x for x in evaluated if x[1]],key=lambda x:(-x[1][0],x[0].user_id))][:5]
 def dcg(ids): return sum((2**labels.get(i,0)-1)/math.log2(k+2) for k,i in enumerate(ids))
 ideal=dcg(sorted([c.user_id for c in people],key=lambda i:-labels.get(i,0))[:5])
 for method,ranking in rankings.items():
  ids=[c.user_id for c in ranking]
  rows.append({'query':raw['id'],'method':method,'ids':ids,'precision_at_5':sum(labels.get(i,0)>0 for i in ids)/5,'ndcg_at_5':dcg(ids)/ideal if ideal else 0,'constraint_violations':sum(i not in allowed for i in ids),'returned':len(ids)})
summary={}
for method in rankings:
 r=[x for x in rows if x['method']==method]
 summary[method]={'precision_at_5':statistics.mean(x['precision_at_5'] for x in r),'ndcg_at_5':statistics.mean(x['ndcg_at_5'] for x in r),'constraint_violation_rate':sum(x['constraint_violations'] for x in r)/sum(x['returned'] for x in r),'returned_total':sum(x['returned'] for x in r)}
result={'dataset_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'matching_code_sha256':hashlib.sha256((ROOT/'backend/app/matching.py').read_bytes()).hexdigest(),'source':'authored synthetic; no human raters','pairs':96,'summary':summary,'rows':rows}
out=ROOT/'output/validation/matching-v1.json'
if out.exists():raise SystemExit('Refusing to overwrite frozen results')
out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False,indent=2))
