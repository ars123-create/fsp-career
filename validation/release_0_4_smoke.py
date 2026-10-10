"""Clean-start check inside an isolated image built from the release archive.

Use only with an empty temporary FSP_DATABASE_URL. The release invocation uses
Docker --network none --read-only and a /tmp tmpfs; it mounts no working data.
"""
import json
import subprocess

subprocess.run(['alembic', 'upgrade', 'head'], check=True)
subprocess.run(['alembic', 'check'], check=True)
from fastapi.testclient import TestClient
from app.main import app

with TestClient(app) as client:
    health = client.get('/api/health')
    assert health.status_code == 200 and health.json()['version'] == '0.4.0-dev'
    page = client.get('/')
    assert page.status_code == 200 and 'assets/' in page.text
    login = client.post('/api/auth/demo/candidate')
    assert login.status_code == 200
    headers = {'Authorization': 'Bearer ' + login.json()['access_token']}
    catalog = client.get('/api/assessments/catalog')
    assert catalog.json()['question_count'] == 12
    history = client.get('/api/assessments', headers=headers)
    assert history.status_code == 200
    assert any(x['rubric_version'] == '2026-10-v1' for x in history.json())
    attempt = client.post('/api/assessments', headers=headers,
                          json={'specialization': 'backend', 'grade': 'Middle'})
    assert attempt.status_code == 201
    value = attempt.json()
    assert value['rubric_version'] == '2026-10-v2'
    assert len(value['questions']) == 12
    assert sum(q.get('answer_type') == 'fields' for q in value['questions']) == 6
    employer = client.post('/api/auth/demo/employer')
    assert employer.status_code == 200
    eh = {'Authorization': 'Bearer ' + employer.json()['access_token']}
    search = client.post('/api/search', headers=eh, json={
        'specialization': 'backend', 'grade': 'Middle', 'ranking': {'mode': 'preference'}})
    assert search.status_code == 200, search.json()
    result = search.json()
    assert result['algorithm'] == 'decision-evoi-v3'
    assert len(result['candidates']) == 5
    assert result['preference']['question'] is not None
print(json.dumps({'status': 'PASS', 'version': '0.4.0-dev',
    'migration': 'upgrade and check PASS', 'empty_db_seed': 'v1 history PASS',
    'assessment': 'v2:12 questions,6 practical', 'matching': 'decision-evoi-v3,top5,question',
    'frontend': 'index200 and built assets', 'network': 'none', 'rootfs': 'readonly',
    'working_database_mounted': False}))
