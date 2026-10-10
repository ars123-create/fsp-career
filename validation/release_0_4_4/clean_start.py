"""API smoke from a fresh migrated database inside the extracted release image."""
from copy import deepcopy
import json
import os
import subprocess

assert os.environ.get("FSP_DATABASE_URL") == "sqlite:////tmp/release044-clean.sqlite"
subprocess.run(["alembic", "upgrade", "head"], check=True)
subprocess.run(["alembic", "check"], check=True)
from fastapi.testclient import TestClient
from sqlmodel import Session
from app.main import app
from app.core.db import engine
from app.models import Attempt


with TestClient(app) as client:
    assert client.get("/api/health").json()["version"] == "0.4.4-dev"
    headers = {}
    for role in ("candidate", "employer", "employer2"):
        login = client.post("/api/auth/demo/" + role)
        assert login.status_code == 200
        headers[role] = {"Authorization": "Bearer " + login.json()["access_token"]}
    h = headers["candidate"]
    original = client.get("/api/profile", headers=h).json()
    grade, specialization = original["grade"], original["specialization"]
    assert client.get("/api/assessments", headers=h).json()[0]["rubric_version"] == "2026-10-v1"
    # Known server oracles prepare fictional technical input; these are not human answers.
    for kind in ("grade", "challenge"):
        attempt = client.post("/api/assessments", headers=h,
                              json={"specialization": specialization, "grade": grade, "kind": kind})
        assert attempt.status_code == 201, attempt.text
        body = attempt.json()
        assert body["rubric_version"] == "2026-10-v2"
        assert len(body["questions"]) == (12 if kind == "grade" else 3)
        assert "_correct" not in attempt.text
        with Session(engine) as db:
            tasks = deepcopy(db.get(Attempt, body["id"]).questions)
        for q in tasks:
            answer = deepcopy(q["_correct"])
            if kind == "challenge":
                first = next(iter(answer))
                answer[first] = str(float(answer[first]) + 1)
            payload = {"values": answer} if isinstance(answer, dict) else {"choice": answer}
            assert client.post(f"/api/assessments/{body['id']}/answer", headers=h,
                               json={"question_id": q["id"], **payload}).status_code == 200
        finish = client.post(f"/api/assessments/{body['id']}/finish", headers=h)
        assert finish.status_code == 200, finish.text
    profile = client.get("/api/profile", headers=h).json()
    assert profile["grade"] == grade
    assert all(s["score"] == 75 and s["families"] == 4 for s in profile["skills"].values())
    for mode in ("weighted", "preference"):
        ranking = {"mode": mode}
        if mode == "weighted":
            ranking["weights"] = {"Python": 10, "SQL": 10, "API": 80}
        query = {"specialization": "backend", "grade": "Middle", "ranking": ranking}
        search = client.post("/api/search", headers=headers["employer"], json=query)
        assert search.status_code == 200, search.text
        assert search.json()["algorithm"] == ("direct-weights-v1" if mode == "weighted" else "decision-evoi-v3")
        assert search.json()["candidates"]
        saved = client.post("/api/needs", headers=headers["employer"], json=query)
        assert saved.status_code == 201
        need_id = saved.json()["id"]
        assert client.get(f"/api/needs/{need_id}", headers=headers["employer"]).status_code == 200
        assert client.get(f"/api/needs/{need_id}", headers=headers["employer2"]).status_code == 404
    index = client.get("/")
    assert index.status_code == 200 and "/assets/" in index.text
    from pathlib import Path
    schema = json.loads(Path("/app/docs/openapi.json").read_text())
    assert app.openapi() == schema
    print(json.dumps({"status": "PASS", "version": "0.4.4-dev", "migration": "upgrade and check PASS",
                      "database": "fresh temporary SQLite", "seed_history": "v1 preserved",
                      "full_and_short": "12 correct then 3 wrong; 75% / four families / prior grade retained",
                      "retained_grade": grade, "openapi_exact_match": True,
                      "matching": "weighted + preference, save/restore, second employer denied",
                      "frontend": "index HTTP200 and built assets; no browser run",
                      "working_database_mounted": False, "human_validation": "NOT RUN"}, ensure_ascii=False))
