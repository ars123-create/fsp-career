"""Transaction distinctions, frozen legacy bank, API grading and version boundaries."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
from sqlmodel import Session

from app.assessment_bank import build_test, score_test, rubric_for
from app.models import Attempt, Candidate, now
from app.practical_bank import build_practical
from app.practical_bank_v2 import build_practical as build_v2
from test_assessment_v2 import payload, wrong
from test_evidence_portfolio import submit
from test_practical_bank import reference


def finish(c, h, attempt_id, tasks, overrides=None):
    for q in tasks:
        value = (overrides or {}).get(q["id"], q["_correct"])
        assert c.post(f"/api/assessments/{attempt_id}/answer", headers=h,
                      json=payload(q, value)).status_code == 200
    response = c.post(f"/api/assessments/{attempt_id}/finish", headers=h)
    assert response.status_code == 200, response.text
    return response.json()


def test_integrated_task_and_legacy_source_match_preintegration_snapshots():
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads((root / "backend/assessment_v3_manifest.json").read_text())
    for item in manifest["files"]:
        data = (root / item["application"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == item["sha256"]
        prototype = root / item["prototype"]
        if prototype.exists():
            assert data == prototype.read_bytes()


@pytest.mark.parametrize("spec,grade", [("backend","Junior"),("backend","Middle"),("data","Junior"),("data","Middle")])
def test_only_target_family_changes_and_all_answers_match_sql(spec, grade):
    for seed in range(948500, 948520):
        old, new = build_v2(spec, grade, seed), build_practical(spec, grade, seed)
        assert [q["family"] for q in old] == [q["family"] for q in new]
        for a, b in zip(old, new):
            if (spec,grade,a["family"]) == ("backend","Middle","api_retry_ledger"):
                assert a != b
                assert {k: int(v) for k,v in b["_correct"].items()} == reference(b,grade)
            else:
                assert a == b
        full = build_test(spec,grade,seed,version="2026-10-v2")
        assert [q for q in full if q.get("answer_type")=="fields"] == sorted(
            old, key=lambda q: next(i for i,item in enumerate(full) if item["family"]==q["family"]))


def test_v2_active_form_keeps_questions_answers_and_rubric(environment, actors):
    c, engine = environment
    h = actors["candidate"]["headers"]
    tasks = build_test("backend","Middle",948500,version="2026-10-v2")
    with Session(engine) as db:
        a = Attempt(candidate_id=actors["candidate"]["id"],specialization="backend",grade="Middle",
                    kind="grade",seed=948500,rubric_version="2026-10-v2",questions=tasks,expires_at=now()+2700)
        db.add(a);db.commit();attempt_id=a.id
    resumed=c.post("/api/assessments",headers=h,json={"specialization":"backend","grade":"Middle"}).json()
    assert resumed["id"]==attempt_id and resumed["rubric_version"]=="2026-10-v2"
    completed=finish(c,h,attempt_id,tasks)
    assert completed["result"]["score"]==100 and completed["rubric"]==rubric_for("2026-10-v2")
    frozen=deepcopy(completed)
    new=c.post("/api/assessments",headers=h,json={"specialization":"backend","grade":"Middle"}).json()
    assert new["rubric_version"]=="2026-10-v3" and len(new["questions"])==12
    assert c.get(f"/api/assessments/{attempt_id}",headers=h).json()==frozen
    with Session(engine) as db:
        assert db.get(Attempt,attempt_id).questions==tasks


def test_new_api_rejects_wrong_transaction_answer_and_retains_old_grade(environment, actors):
    c,engine=environment;h=actors["candidate"]["headers"]
    result=c.post("/api/assessments",headers=h,json={"specialization":"backend","grade":"Middle"})
    assert result.status_code==201 and result.json()["rubric_version"]=="2026-10-v3"
    attempt_id=result.json()["id"]
    with Session(engine) as db:
        tasks=deepcopy(db.get(Attempt,attempt_id).questions)
    q=next(q for q in tasks if q["family"]=="api_retry_ledger")
    initial=int(q["prompt"].split(": ",1)[1].split(".",1)[0])
    keys={r[1]:r[2] for r in q["tables"][0]["rows"]}  # Deliberately ignore stages.
    overrides={q["id"]:{"stock":str(initial-sum(keys.values())),"writes":str(len(keys))}}
    other=next(q for q in tasks if q["skill"]=="API" and q.get("answer_type")=="fields" and q["family"]!="api_retry_ledger")
    overrides[other["id"]]=wrong(other)
    assert "_correct" not in result.text and "correct_values" not in result.text
    done=finish(c,h,attempt_id,tasks,overrides)
    assert not next(q for q in done["questions"] if q["family"]=="api_retry_ledger")["is_correct"]
    assert done["result"]["practical_skills"]["API"]==0 and not done["result"]["passed"]
    assert c.get("/api/profile",headers=h).json()["grade"]=="Junior"


def test_v3_short_does_not_mix_with_v2_portfolio(environment, actors):
    c,engine=environment;actor=actors["candidate"];h=actor["headers"]
    tasks=build_test("backend","Junior",948501,version="2026-10-v2")
    with Session(engine) as db:
        a=Attempt(candidate_id=actor["id"],specialization="backend",grade="Junior",kind="grade",seed=948501,
                  rubric_version="2026-10-v2",questions=tasks,expires_at=now()+2700)
        db.add(a);db.commit();attempt_id=a.id
    full=finish(c,h,attempt_id,tasks)
    assert full["result"]["evidence_outcome"]=="updated_full"
    before=c.get("/api/profile",headers=h).json()
    short,_=submit(c,engine,h,kind="challenge",wrong_practice=True)
    assert short["rubric_version"]=="2026-10-v3"
    assert short["result"]["evidence_outcome"]=="needs_full_assessment"
    after=c.get("/api/profile",headers=h).json()
    assert after["skills"]==before["skills"] and after["grade"]==before["grade"]


def test_stale_v2_claim_cannot_replace_newer_v3_portfolio(environment, actors):
    c,engine=environment;actor=actors["candidate"];h=actor["headers"]
    tasks=build_test("backend","Junior",948501,version="2026-10-v2")
    answers={q["id"]:q["_correct"] for q in tasks}
    with Session(engine) as db:
        candidate=db.get(Candidate,actor["id"]);candidate.grade_changed_at=now();db.add(candidate)
        a=Attempt(candidate_id=actor["id"],specialization="backend",grade="Junior",kind="grade",seed=948501,
                  rubric_version="2026-10-v2",questions=tasks,answers=answers,status="completed",
                  created_at=100,finished_at=200,expires_at=300,
                  result=score_test(tasks,answers,rubric_version="2026-10-v2") | {"grade_outcome":"cooldown"})
        db.add(a);db.commit();attempt_id=a.id
    new,_=submit(c,engine,h,wrong_practice=True)
    before=c.get("/api/profile",headers=h).json()["skills"]
    assert all(v["score"]==75 for v in before.values())
    with Session(engine) as db:
        candidate=db.get(Candidate,actor["id"]);candidate.grade_changed_at=now()-91*86400;db.add(candidate);db.commit()
    claim=c.post(f"/api/assessments/{attempt_id}/claim-grade",headers=h)
    assert claim.status_code==200
    assert claim.json()["result"]["evidence_outcome"]=="older_assessment_retained"
    assert c.get("/api/profile",headers=h).json()["skills"]==before
    with Session(engine) as db:
        portfolio=db.get(Candidate,actor["id"]).skill_portfolio
        assert portfolio["context"]["rubric_version"]=="2026-10-v3"
        assert portfolio["anchor_attempt_id"]==new["id"]
