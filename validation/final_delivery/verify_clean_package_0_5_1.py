from pathlib import Path
import hashlib,json
from fastapi.testclient import TestClient
from app.main import app
package=Path('/verify');snapshot=json.loads((package/'output/validation/source-v0.5.1.json').read_text())
backend=[e for e in snapshot['files'] if e['path'].startswith('backend/')]
for e in backend:assert hashlib.sha256((Path('/app')/e['path']).read_bytes()).hexdigest()==e['sha256'],e['path']
with TestClient(app) as c:
 assert c.get('/api/health').json()['version']=='0.5.1'
 assert app.openapi()==json.loads((package/'docs/openapi.json').read_text())
 for route in ('/','/login','/search','/assessment'):
  res=c.get(route);assert res.status_code==200 and '<!doctype html' in res.text.lower()
 for role in ('candidate','employer'):
  res=c.post('/api/auth/login',data={'username':role+'@demo.fsp.test','password':'12345678'});assert res.status_code==200
  if role=='employer':headers={'Authorization':'Bearer '+res.json()['access_token']}
 query={'specialization':'backend','grade':'Middle','ranking':{'mode':'mixed'}}
 res=c.post('/api/search',headers=headers,json=query);assert res.status_code==200;v=res.json();assert v['algorithm']=='mixed-kg-v1' and len(v['candidates'])==5;assert all(p['contacts'] is None for p in v['candidates'])
 q=v['preference']['question'];assert q
 query['ranking']['comparisons']=[{'left':q['left'],'right':q['right'],'choice':'left'}]
 res=c.post('/api/search',headers=headers,json=query);assert res.status_code==200 and res.json()['preference']['answered']==1
print(json.dumps({'status':'PASS','version':'0.5.1','backend_files_identical':len(backend),'both_normal_logins':True,'openapi_exact_match':True,'frontend_routes':4,'m6_search_and_answer':True,'contacts_protected':True,'scope':'Public source archive compared with previously built image; isolated tmpfs SQLite and TestClient. Not a clean rebuild or independent machine.'}))
