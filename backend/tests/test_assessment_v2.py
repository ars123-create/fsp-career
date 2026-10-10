"""Rubric boundaries, persistence, version compatibility and non-disclosure."""
import copy
from collections import Counter
from decimal import Decimal

import pytest
from sqlmodel import Session, select
from app.assessment_bank import RUBRIC, build_test, public_questions, score_test
from app.models import Attempt, Candidate, Evidence, now


def wrong(q):
    if q.get("answer_type") == "fields":
        return {k: str(Decimal(v) + 1) for k, v in q["_correct"].items()}
    return (q["_correct"] + 1) % len(q["choices"])


def payload(q, value):
    return {"question_id": q["id"], "values" if q.get("answer_type") == "fields" else "choice": value}


@pytest.mark.parametrize("spec", ["backend", "data"])
@pytest.mark.parametrize("grade", ["Junior", "Middle"])
def test_blueprint_and_grade_boundaries(spec, grade):
    tasks = build_test(spec, grade, 711)
    assert len(tasks) == len({q["id"] for q in tasks}) == 12
    assert set(Counter(q["skill"] for q in tasks).values()) == {4}
    practical = [q for q in tasks if q.get("answer_type") == "fields"]
    assert len(practical) == 6
    assert set(Counter(q["skill"] for q in practical).values()) == {2}
    answers = {q["id"]: copy.deepcopy(q["_correct"]) for q in tasks}
    assert score_test(tasks, answers)["passed"]
    # All six choices alone cannot confirm a grade.
    choices_only = answers | {q["id"]: wrong(q) for q in practical}
    result = score_test(tasks, choices_only)
    assert result["score"] == 50 and not result["passed"]
    # 10/12 overall and 50% SQL still fail with 0% SQL practice.
    sql_wrong = answers | {q["id"]: wrong(q) for q in practical if q["skill"] == "SQL"}
    result = score_test(tasks, sql_wrong)
    assert result["score"] == 83.3 and result["skills"]["SQL"] == 50
    assert result["practical_skills"]["SQL"] == 0 and not result["passed"]
    # Exactly 9/12 with one correct practical case in each skill meets all floors.
    missed = set()
    for q in practical:
        if q["skill"] not in missed:
            answers[q["id"]] = wrong(q)
            missed.add(q["skill"])
    result = score_test(tasks, answers)
    assert result["score"] == 75 and result["passed"]
    assert set(result["practical_skills"].values()) == {50}
    challenge = build_test(spec, grade, 711, "challenge")
    assert len(challenge) == 3
    assert all(q["answer_type"] == "fields" for q in challenge)
    assert len({q["skill"] for q in challenge}) == 3
    # Deep serialization check: no hidden answers, explanation or correctness.
    for q in public_questions(tasks):
        assert not any(k.startswith("_") or k in {"correct_choice", "correct_values", "is_correct", "explanation"} for k in q)


@pytest.mark.parametrize("spec", ["backend", "data"])
def test_numeric_api_persistence_validation_and_completed_review(environment, actors, spec):
    c, engine = environment
    h = actors["candidate"]["headers"]
    catalog = c.get("/api/assessments/catalog").json()
    assert (catalog["question_count"], catalog["practical_count"], catalog["duration_minutes"]) == (12, 6, 45)
    a = c.post("/api/assessments", headers=h, json={"specialization": spec, "grade": "Middle"}).json()
    assert a["rubric_version"] == RUBRIC["version"]
    assert 2698 <= a["expires_at"] - a["created_at"] <= 2700
    with Session(engine) as db:
        tasks = db.get(Attempt, a["id"]).questions
    q = next(q for q in tasks if q.get("answer_type") == "fields")
    route = f"/api/assessments/{a['id']}"
    valid = payload(q, q["_correct"])
    assert c.post(route + "/answer", headers=actors["other"]["headers"], json=valid).status_code == 404
    for data in ({"choice": 0}, {"values": {}}, {"values": {"unknown": "1"}},
                 {"values": {k: "nan" for k in q["_correct"]}},
                 {"values": {k: "1e2" for k in q["_correct"]}},
                 {"values": {k: 1 for k in q["_correct"]}},
                 {"values": {k: "1" * 41 for k in q["_correct"]}},
                 {"values": q["_correct"], "choice": 0}):
        assert c.post(route + "/answer", headers=h, json={"question_id": q["id"], **data}).status_code == 422
    # Valid-but-wrong and correct values have exactly the same save response.
    wrong_response = c.post(route + "/answer", headers=h, json=payload(q, wrong(q)))
    right_response = c.post(route + "/answer", headers=h, json=valid)
    assert wrong_response.json() == right_response.json() == {"saved": True, "answered": 1}
    for task in tasks:
        value = task["_correct"]
        if isinstance(value, dict):
            value = {k: v.replace(".", ",") for k, v in value.items()}
        r = c.post(route + "/answer", headers=h, json=payload(task, value))
        assert r.status_code == 200, r.text
    saved = c.get(route, headers=h)
    for forbidden in ("_correct", "correct_choice", "correct_values", "is_correct", "explanation"):
        assert forbidden not in saved.text
    assert len(saved.json()["answers"]) == 12
    assert c.get("/api/assessments", headers=h).json()[0]["answers"] == saved.json()["answers"]
    result = c.post(route + "/finish", headers=h)
    assert result.status_code == 200, result.text
    completed = result.json()
    assert completed["result"]["score"] == 100 and completed["result"]["passed"]
    assert all(q["is_correct"] for q in completed["questions"])
    assert len([q for q in completed["questions"] if "correct_values" in q]) == 6
    assert c.post(route + "/finish", headers=h).json() == completed
    assert c.post(route + "/answer", headers=h, json=valid).status_code == 409
    with Session(engine) as db:
        evidence = db.exec(select(Evidence).where(Evidence.attempt_id == a["id"])).all()
        assert len(evidence) == 12 and all(e.score == 100 for e in evidence)


def test_active_v1_keeps_rubric_and_stored_result_after_upgrade(environment, actors):
    c, engine = environment
    h = actors["candidate"]["headers"]
    questions = build_test("backend", "Junior", 715, version="2026-10-v1")
    answers = {q["id"]: q["_correct"] for q in questions}
    answers[questions[0]["id"]] = wrong(questions[0])  # Five of six pass the old floor.
    with Session(engine) as db:
        a = Attempt(candidate_id=actors["candidate"]["id"], specialization="backend", grade="Junior",
                    seed=715, questions=questions, expires_at=now() + 1500)
        db.add(a); db.commit(); attempt_id = a.id
    resumed = c.post("/api/assessments", headers=h, json={"specialization": "backend", "grade": "Junior"}).json()
    assert resumed["id"] == attempt_id and resumed["rubric"]["pass_score"] == 70
    assert resumed["rubric_version"] == "2026-10-v1" and len(resumed["questions"]) == 6
    for q in questions:
        assert c.post(f"/api/assessments/{attempt_id}/answer", headers=h, json=payload(q, answers[q["id"]])).status_code == 200
    finished = c.post(f"/api/assessments/{attempt_id}/finish", headers=h).json()
    assert finished["result"]["passed"] and finished["result"]["rubric"]["version"] == "2026-10-v1"
    before = copy.deepcopy(finished["result"])
    latest = c.post("/api/assessments", headers=h, json={"specialization": "backend", "grade": "Middle"}).json()
    assert latest["rubric_version"] == RUBRIC["version"] and len(latest["questions"]) == 12
    assert c.get(f"/api/assessments/{attempt_id}", headers=h).json()["result"] == before
