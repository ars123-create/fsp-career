"""Fresh migrated demo smoke; deliberately wrong answers are fictional test data."""
from copy import deepcopy
from pathlib import Path
import json
import os
import subprocess

assert os.environ.get("FSP_DATABASE_URL") == "sqlite:////tmp/assessment-v3-clean.sqlite"
subprocess.run(["alembic", "upgrade", "head"], check=True)
subprocess.run(["alembic", "check"], check=True)
from fastapi.testclient import TestClient
from sqlmodel import Session
from app.main import app
from app.core.db import engine
from app.models import Attempt

with TestClient(app) as c:
    assert c.get("/api/health").json()["version"] == "0.4.5-dev"
    login = c.post("/api/auth/demo/candidate")
    assert login.status_code == 200
    h = {"Authorization": "Bearer " + login.json()["access_token"]}
    before = c.get("/api/profile", headers=h).json()
    assert c.get("/api/assessments", headers=h).json()[0]["rubric_version"] == "2026-10-v1"
    response = c.post("/api/assessments", headers=h, json={"specialization":"backend", "grade":"Middle"})
    assert response.status_code == 201
    a = response.json()
    assert a["rubric_version"] == "2026-10-v3" and len(a["questions"]) == 12
    assert "_correct" not in response.text and "correct_values" not in response.text
    with Session(engine) as db:
        tasks = deepcopy(db.get(Attempt, a["id"]).questions)
    for q in tasks:
        value = q["_correct"]
        if q["family"] == "api_retry_ledger":
            initial = int(q["prompt"].split(": ",1)[1].split(".",1)[0])
            all_keys = {row[1]:row[2] for row in q["tables"][0]["rows"]}
            assert len(all_keys) == 4
            value = {"stock":str(initial-sum(all_keys.values())), "writes":str(len(all_keys))}
        elif q["family"] == "api_cursor_page":
            value = {k:str(int(v)+1) for k,v in value.items()}
        payload = {"values":value} if isinstance(value,dict) else {"choice":value}
        saved = c.post(f"/api/assessments/{a['id']}/answer",headers=h,json={"question_id":q["id"],**payload})
        assert saved.status_code == 200
    response = c.post(f"/api/assessments/{a['id']}/finish",headers=h)
    assert response.status_code == 200
    result = response.json()
    assert result["result"]["practical_skills"]["API"] == 0 and not result["result"]["passed"]
    assert not next(q for q in result["questions"] if q["family"]=="api_retry_ledger")["is_correct"]
    after = c.get("/api/profile",headers=h).json()
    assert after["grade"] == before["grade"] and after["skills"] == before["skills"]
    assert c.get("/").status_code == 200
    assert app.openapi() == json.loads(Path("/app/docs/openapi.json").read_text())
    print(json.dumps({"status":"PASS", "version":"0.4.5-dev", "migration":"upgrade/check PASS",
                      "fresh_database":True, "new_rubric":"2026-10-v3", "old_history":"v1 retained",
                      "wrong_transaction_rule_rejected":True, "prior_grade_and_skills_retained":True,
                      "openapi_exact_match":True, "human_validation":"NOT RUN"}))
