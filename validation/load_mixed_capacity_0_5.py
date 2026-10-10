"""Bounded local capacity audit, reusing the existing HTTPX load workflow.

Own temporary DB, process, socket and synthetic sessions only. No connection
to the developer DB and no credential/response-body logging.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
from itertools import product
import json
import math
import os
from pathlib import Path
import platform
import secrets
import socket
import subprocess
import sys
import tempfile
import time

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "validation/MIXED_CAPACITY_0_5_PROTOCOL.md"
SKILLS = ("Python", "SQL", "API")
STAMP = 1791390000


def sha(data):
    return hashlib.sha256(data).hexdigest()


def backend_fingerprint():
    files = sorted((ROOT / "backend/app").rglob("*.py"))
    files += [ROOT / "backend/requirements.lock.txt"]
    hashes = {p.relative_to(ROOT).as_posix(): sha(p.read_bytes()) for p in files}
    return {"sha256": sha(json.dumps(hashes, sort_keys=True).encode()), "files": hashes}


def fixture_values(scenario):
    seed = 934001 if scenario == "bank_v2" else 934002
    rng = np.random.default_rng(seed)
    if scenario == "bank_v2":
        allowed = np.array([v for v in product((50, 75, 100), repeat=3)
                            if 225 <= sum(v) < 300], dtype=float)
        values = allowed[rng.integers(len(allowed), size=100)]
    else:
        values = []
        while len(values) < 100:
            row = rng.uniform(50, 100, 3)
            if row.sum() >= 225:
                values.append(row)
        values = np.asarray(values)
    return seed, values


def seed_database(scenario):
    # Imports happen after the explicit temporary settings have been installed.
    from sqlmodel import SQLModel, Session
    from app.core.db import engine
    from app.core.security import issue_access
    from app.models import User, Candidate, Company
    SQLModel.metadata.create_all(engine)
    seed, values = fixture_values(scenario)
    tokens = []
    with Session(engine) as db:
        for i in range(30):
            user = User(id=sha(f"capacity-employer-{i}".encode())[:32],
                        email=f"capacity-company-{i}@example.org",
                        full_name=f"Fictional company {i}", hashed_password="login-disabled",
                        role="employer", verified=True)
            db.add(user)
            db.flush()
            db.add(Company(user_id=user.id, name=user.full_name, contact=user.email))
            tokens.append(issue_access(user))
        for i, row in enumerate(values):
            user = User(id=sha(f"capacity-candidate-{i}".encode())[:32],
                        email=f"capacity-candidate-{i}@example.org",
                        full_name=f"Private fictional candidate {i}",
                        hashed_password="login-disabled", role="candidate", verified=True)
            db.add(user)
            db.flush()
            db.add(Candidate(user_id=user.id, headline="Capacity fixture", published=True,
                             specialization="backend", grade="Middle", work_format="remote",
                             salary_min=180000, salary_max=240000,
                             processing_consent_at=STAMP, publication_consent_at=STAMP,
                             skills={skill: {"score": float(value), "families": 4,
                                            "checked_at": STAMP, "source": "capacity-fixture"}
                                     for skill, value in zip(SKILLS, row)}))
        db.commit()
    engine.dispose()
    return tokens, {"seed": seed, "profiles": 100,
                    "unique_vectors": len(np.unique(values, axis=0)),
                    "scores_sha256": sha(values.astype("<f8").tobytes())}


def root_query():
    return {"specialization": "backend", "grade": "Middle", "work_format": "remote",
            "skills": list(SKILLS), "salary_min": 180000, "salary_max": 240000,
            "ranking": {"mode": "mixed"}}


async def workload(base, tokens, clients, steps):
    samples = []
    limits = httpx.Limits(max_connections=30, max_keepalive_connections=30)
    async with httpx.AsyncClient(base_url=base, limits=limits, timeout=40., trust_env=False) as client:
        health = (await client.get("/api/health")).json()
        for _ in range(3):
            response = await client.post("/api/search", json=root_query(),
                                         headers={"Authorization": f"Bearer {tokens[0]}"})
            response.raise_for_status()
            assert response.json()["preference"]["considered"] == 100

        async def worker(index, concurrency):
            previous = None
            for step in range(steps):
                payload, choice = root_query(), None
                if step % 2 and previous and previous["preference"]["question"]:
                    question = previous["preference"]["question"]
                    choice = ("left", "right", "skip")[(step // 2) % 3]
                    payload = previous["normalized"]
                    payload["ranking"]["comparisons"].append({
                        "left": question["left"], "right": question["right"], "choice": choice})
                start = time.perf_counter()
                row = {"concurrency": concurrency, "client": index, "step": step,
                       "choice": choice}
                try:
                    response = await client.post("/api/search", json=payload,
                                                 headers={"Authorization": f"Bearer {tokens[index]}"})
                    row.update(milliseconds=1000 * (time.perf_counter()-start), status=response.status_code)
                    if response.status_code == 200:
                        result = response.json()
                        report = result["preference"]
                        candidates = result["candidates"]
                        ids = [c["id"] for c in candidates]
                        violations = []
                        if result["algorithm"] != "mixed-kg-v1" or report["considered"] != 100:
                            violations.append("algorithm_or_pool")
                        if len(ids) != 5 or len(set(ids)) != 5:
                            violations.append("top_five")
                        if any(c["contacts"] is not None or not c["name"].startswith("Кандидат ")
                               for c in candidates):
                            violations.append("contact_visibility")
                        if choice == "skip" and ids != [c["id"] for c in previous["candidates"]]:
                            violations.append("skip_changed_selection")
                        row.update(question=report["question"] is not None,
                                   answered=report["answered"], skipped=report["skipped"],
                                   selected_ids=ids, violations=violations)
                        previous = result
                    else:
                        previous = None
                except httpx.HTTPError as exc:
                    row.update(milliseconds=1000 * (time.perf_counter()-start),
                               status=type(exc).__name__)
                    previous = None
                except (KeyError, TypeError, ValueError) as exc:
                    row.update(milliseconds=1000 * (time.perf_counter()-start),
                               status="InvalidResponse", violations=[type(exc).__name__])
                    previous = None
                samples.append(row)

        for concurrency in clients:
            await asyncio.gather(*(worker(i, concurrency) for i in range(concurrency)))
            subset = [r for r in samples if r["concurrency"] == concurrency]
            print(json.dumps({"completed_clients": concurrency, "requests": len(subset),
                              "http_200": sum(r["status"] == 200 for r in subset)}), flush=True)
    return health, samples


def summarize(rows, clients):
    result = []
    for concurrency in clients:
        subset = [r for r in rows if r["concurrency"] == concurrency]
        durations = sorted(r["milliseconds"] for r in subset)
        stats = {"clients": concurrency, "requests": len(subset),
                 "http_statuses": dict(Counter(str(r["status"]) for r in subset)),
                 "violations": sum(len(r.get("violations", [])) for r in subset),
                 "p50_ms": durations[math.ceil(len(durations)*.5)-1],
                 "p95_ms": durations[math.ceil(len(durations)*.95)-1],
                 "max_ms": durations[-1]}
        stats["status"] = "PASS" if (stats["http_statuses"] == {"200": len(subset)}
            and not stats["violations"] and stats["p95_ms"] < 2000) else "FAIL"
        result.append(stats)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["bank_v2", "varied"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clients", default="1,10,30")
    parser.add_argument("--steps", type=int, default=6)
    args = parser.parse_args()
    clients = [int(x) for x in args.clients.split(",")]
    if not clients or len(set(clients)) != len(clients) or any(n < 1 or n > 30 for n in clients):
        parser.error("distinct concurrency levels in 1..30 required")
    if not 1 <= args.steps <= 6:
        parser.error("steps must be in 1..6")
    if args.output.exists():
        parser.error("refusing to overwrite a saved run")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    source_before = backend_fingerprint()
    with tempfile.TemporaryDirectory(prefix="fsp-capacity-") as temp:
        os.environ.update(FSP_DATABASE_URL=f"sqlite:///{temp}/capacity.db",
                          FSP_SECRET_KEY=secrets.token_urlsafe(48), FSP_DEMO_MODE="false",
                          FSP_SMTP_HOST="", PYTHONPATH=str(ROOT / "backend"))
        sys.path.insert(0, str(ROOT / "backend"))
        tokens, fixture = seed_database(args.scenario)
        with socket.socket() as sock, open(Path(temp)/"server.log", "w") as log:
            sock.bind(("127.0.0.1", 0))
            sock.listen(128)
            base = f"http://127.0.0.1:{sock.getsockname()[1]}"
            proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app",
                                     "--fd", str(sock.fileno()), "--no-access-log"],
                                    cwd=ROOT, env=os.environ.copy(), pass_fds=(sock.fileno(),),
                                    stdout=log, stderr=subprocess.STDOUT)
            try:
                with httpx.Client(timeout=1., trust_env=False) as client:
                    for _ in range(100):
                        if proc.poll() is not None:
                            raise RuntimeError("isolated uvicorn exited before readiness")
                        try:
                            if client.get(base+"/api/health").status_code == 200:
                                break
                        except httpx.HTTPError:
                            pass
                        time.sleep(.1)
                    else:
                        raise RuntimeError("isolated uvicorn readiness deadline")
                health, rows = asyncio.run(workload(base, tokens, clients, args.steps))
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
        summary = summarize(rows, clients)
        result = {"date": datetime.now(timezone.utc).isoformat(), "scenario": args.scenario,
                  "status": "PASS" if all(s["status"] == "PASS" for s in summary) else "FAIL",
                  "smoke_only": clients == [1] and args.steps == 1,
                  "fixture": fixture, "sessions": 30, "steps_per_client": args.steps,
                  "health": health, "app_version": json.loads((ROOT/"VERSION.json").read_text()),
                  "actual_backend_source": source_before,
                  "protocol_sha256": sha(PROTOCOL.read_bytes()),
                  "script_sha256": sha(Path(__file__).read_bytes()),
                  "platform": {"os": platform.platform(), "machine": platform.machine(),
                               "cpu_count": os.cpu_count(), "python": platform.python_version(),
                               "blas_threads": 1, "uvicorn_workers": 1, "database": "temporary SQLite"},
                  "versions": {p: importlib.metadata.version(p) for p in
                               ("httpx", "httpcore", "anyio", "starlette", "uvicorn", "numpy", "scipy", "sqlmodel")},
                  "elapsed_s": time.perf_counter()-started, "summary": summary, "rows": rows,
                  "own_server_stopped": proc.poll() is not None,
                  "limitations": "Local macOS capacity-development run. Synthetic fixture scores, no actual exams or human relevance labels. No production SLA, PostgreSQL capacity or independent hardware claim."}
    result["temporary_database_removed"] = not Path(temp).exists()
    result["backend_unchanged_during_run"] = source_before == backend_fingerprint()
    if not result["backend_unchanged_during_run"]:
        result["status"] = "FAIL"
    with args.output.open("x") as out:
        json.dump(result, out, ensure_ascii=False, indent=2)
        out.write("\n")
    print(json.dumps({k: result[k] for k in ("status", "scenario", "fixture", "summary", "elapsed_s", "own_server_stopped", "temporary_database_removed")}), flush=True)


if __name__ == "__main__":
    main()
