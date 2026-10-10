"""Run inside a disposable image with a fresh database under /tmp."""
import json
import os
import platform
import sqlite3
import subprocess

assert os.environ.get("FSP_DATABASE_URL") == "sqlite:////tmp/assessment-clean.db"
subprocess.run(["alembic", "upgrade", "head"], check=True)
subprocess.run(["alembic", "check"], check=True)
from fastapi.testclient import TestClient
from app.main import app

with TestClient(app) as client:
    health = client.get("/api/health").json()
    assert health["version"] == "0.4.0-dev"
    login = client.post("/api/auth/demo/candidate")
    assert login.status_code == 200
    headers = {"Authorization": "Bearer " + login.json()["access_token"]}
    history = client.get("/api/assessments", headers=headers).json()
    assert len(history) == 1 and history[0]["rubric_version"] == "2026-10-v1"
    assert history[0]["result"]["passed"] and len(history[0]["questions"]) == 6
    attempt = client.post("/api/assessments", headers=headers, json={"specialization": "backend", "grade": "Middle"})
    assert attempt.status_code == 201, attempt.status_code
    body = attempt.json()
    assert body["rubric_version"] == "2026-10-v2" and len(body["questions"]) == 12
    assert sum(q["answer_type"] == "fields" for q in body["questions"]) == 6
    assert all("correct_values" not in q and "correct_choice" not in q for q in body["questions"])
    assert client.get("/").status_code == 200
    print(json.dumps({"status": "PASS", "version": health["version"], "python": platform.python_version(),
                      "sqlite": sqlite3.sqlite_version, "migration": "upgrade and check PASS",
                      "seed": "v1 history PASS", "new_attempt": "v2, 12 tasks / 6 practical PASS",
                      "frontend": "index HTTP 200; no browser run inside image"}, ensure_ascii=False))
