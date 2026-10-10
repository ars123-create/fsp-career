"""Read-mostly bounded smoke/load check for our local synthetic FSP demo."""
import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import secrets
import time
from urllib.parse import urlparse, parse_qs
import uuid
import httpx

ROOT = Path(__file__).resolve().parents[1]


def source_digest():
    h = hashlib.sha256()
    for p in sorted((ROOT / 'backend/app').rglob('*.py')):
        h.update(p.relative_to(ROOT).as_posix().encode()); h.update(p.read_bytes())
    return h.hexdigest()


def smtp_smoke(base, mail_base, out):
    # Only Mailpit in the local Compose environment; no real outbound messages.
    email = 'smoke-' + uuid.uuid4().hex[:12] + '@example.org'
    password = secrets.token_urlsafe(24)
    with httpx.Client(base_url=base, timeout=20) as c, httpx.Client(base_url=mail_base, timeout=10) as m:
        r = c.post('/api/auth/register', json={'email': email, 'password': password, 'full_name': 'Синтетический тест SMTP', 'role': 'candidate', 'processing_consent': True})
        assert r.status_code == 201, r.status_code
        assert r.json()['local_mail_receipt'] is None
        assert c.post('/api/auth/login', data={'username': email, 'password': password}).status_code == 403
        messages = m.get('/api/v1/messages').json()['messages']
        match = next(x for x in messages if any(a['Address'] == email for a in x['To']))
        body = m.get('/api/v1/message/' + match['ID']).json()['Text']
        link = re.search(r'https?://[^\s]+', body)[0]
        assert urlparse(link).netloc == urlparse(base).netloc
        token = parse_qs(urlparse(link).query)['token'][0]
        assert c.post('/api/auth/verify', json={'token': token}).status_code == 200
        assert c.post('/api/auth/verify', json={'token': token}).status_code == 400
        r = c.post('/api/auth/login', data={'username': email, 'password': password})
        assert r.status_code == 200
        h = {'Authorization': 'Bearer ' + r.json()['access_token']}
        own = c.get('/api/profile', headers=h).json()
        assert own['grade'] is None and own['achievements'] == []
        result = {'status': 'PASS', 'transport': 'SMTP -> local Mailpit', 'checks': ['register', 'unverified_login_rejected', 'SMTP_received', 'link_origin', 'verify_once', 'login', 'empty_FSP_works'], 'external_delivery': 'NOT RUN', 'at': datetime.now(timezone.utc).isoformat()}
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(json.dumps(result, ensure_ascii=False))


async def load(base, out):
    timings = []; errors = []
    limits = httpx.Limits(max_connections=30, max_keepalive_connections=30)
    async with httpx.AsyncClient(base_url=base, limits=limits, timeout=20) as c:
        # Reproduce the documented minimal AsyncClient example against our own health route first.
        health = await c.get('/api/health'); health.raise_for_status()
        headers = []
        for role in ('employer', 'employer2', 'candidate'):
            r = await c.post('/api/auth/demo/' + role); r.raise_for_status()
            headers.append({'Authorization': 'Bearer ' + r.json()['access_token']})
        r = await c.post('/api/search', headers=headers[0], json={}); r.raise_for_status()
        total = r.json()['total']; cid = r.json()['candidates'][0]['id']
        cases = [('search', 'POST', '/api/search', {'query': 'Python backend, Middle, удалённо, 180–250 тыс'}, headers[0]),
                 ('profile', 'GET', '/api/candidates/' + cid, None, headers[1]),
                 ('invitations', 'GET', '/api/invitations', None, headers[2])]
        async def request(case):
            label, method, path, payload, h = case
            start = time.perf_counter()
            try:
                r = await c.request(method, path, headers=h, json=payload)
                code = r.status_code
                if code != 200: errors.append({'case': label, 'status': code})
            except httpx.HTTPError as exc:
                code = 0; errors.append({'case': label, 'error': type(exc).__name__})
            return {'case': label, 'status': code, 'ms': round((time.perf_counter()-start)*1000, 3)}
        for _ in range(3):
            for case in cases: await request(case)
        errors.clear()
        async def worker(n):
            for k in range(30): timings.append(await request(cases[(n+k)%3]))
        start = time.perf_counter()
        await asyncio.gather(*(worker(n) for n in range(30)))
        elapsed = time.perf_counter()-start
    def summary(rows):
        v = sorted(x['ms'] for x in rows)
        return {'requests': len(v), 'p50_ms': v[math.ceil(len(v)*.50)-1], 'p95_ms': v[math.ceil(len(v)*.95)-1], 'max_ms': v[-1]}
    per_case = {case: summary([x for x in timings if x['case'] == case]) for case in ('search','profile','invitations')}
    result = {'status': 'PASS' if not errors and all(v['p95_ms'] < 2000 for v in per_case.values()) else 'FAIL', 'at': datetime.now(timezone.utc).isoformat(), 'generator_worktree_sha256': source_digest(), 'httpx': httpx.__version__, 'health': health.json(), 'protocol': 'validation/LOAD_PROTOCOL.md', 'virtual_clients': 30, 'distinct_sessions': 3, 'published_profiles': total, 'elapsed_s': round(elapsed,3), 'requests_per_s': round(len(timings)/elapsed,2), 'http_statuses': dict(Counter(x['status'] for x in timings)), 'by_route': per_case, 'errors': errors, 'samples': timings}
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k != 'samples'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('kind', choices=['smtp','load']); p.add_argument('--base', default='http://127.0.0.1:8019'); p.add_argument('--mail', default='http://127.0.0.1:8026'); p.add_argument('--output', required=True); a=p.parse_args()
    if urlparse(a.base).hostname not in ('127.0.0.1', 'localhost') or urlparse(a.mail).hostname not in ('127.0.0.1','localhost'):
        raise SystemExit('This demo validator is restricted to localhost.')
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists(): raise SystemExit('Choose a new result filename; previous measurements are immutable.')
    smtp_smoke(a.base,a.mail,out) if a.kind=='smtp' else asyncio.run(load(a.base,out))
