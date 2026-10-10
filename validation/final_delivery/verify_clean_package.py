"""Run inside the archive-built image, with the package mounted at /verify.

Uses a new tmpfs database after Alembic migration, no external network/data.
Tokens are kept in memory and never printed in the receipt.
"""
from pathlib import Path
import hashlib
import json

from fastapi.testclient import TestClient
from app.main import app

package, image_root = Path('/verify'), Path('/app')
snapshot = json.loads((package / 'output/validation/source-v0.5.0.json').read_text())
backend = [e for e in snapshot['files'] if e['path'].startswith('backend/')]
for entry in backend:
    actual = (image_root / entry['path']).read_bytes()
    assert hashlib.sha256(actual).hexdigest() == entry['sha256'], entry['path']
with TestClient(app) as client:
    health = client.get('/api/health')
    assert health.status_code == 200 and health.json()['version'] == '0.5.0'
    assert app.openapi() == json.loads((package / 'docs/openapi.json').read_text())
    for route in ('/', '/search', '/assessment'):
        page = client.get(route)
        assert page.status_code == 200 and '<!doctype html' in page.text.lower(), route
    login = client.post('/api/auth/demo/employer')
    assert login.status_code == 200
    headers = {'Authorization': 'Bearer ' + login.json()['access_token']}
    query = {'specialization': 'backend', 'grade': 'Middle', 'ranking': {'mode': 'mixed'}}
    response = client.post('/api/search', headers=headers, json=query)
    assert response.status_code == 200
    result = response.json()
    assert result['algorithm'] == 'mixed-kg-v1' and len(result['candidates']) == 5
    assert all(c['contacts'] is None for c in result['candidates'])
    question = result['preference']['question']
    if question:
        query['ranking']['comparisons'] = [{'left': question['left'], 'right': question['right'], 'choice': 'left'}]
        answer = client.post('/api/search', headers=headers, json=query)
        assert answer.status_code == 200 and answer.json()['preference']['answered'] == 1
print(json.dumps({'status': 'PASS', 'backend_files_match_image': len(backend),
    'openapi_exact_match': True, 'frontend_routes': 3, 'health_version': '0.5.0',
    'mixed_search': True, 'contacts_hidden': True, 'answered_question': bool(question),
    'scope': 'Archive-built image, new tmpfs SQLite, TestClient lifespan; not a browser test'}))
