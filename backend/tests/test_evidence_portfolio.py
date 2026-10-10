"""Technical evidence retention, independent of human skill/grade calibration."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
from sqlmodel import Session, select

from app.assessment_bank import RUBRIC, build_test, score_test
from app.models import Attempt, Candidate, GradeAward, now


def submit(c, engine, headers, *, kind="grade", spec="backend", grade="Junior", wrong_practice=False):
    response = c.post("/api/assessments", headers=headers,
                      json={"specialization":spec,"grade":grade,"kind":kind})
    assert response.status_code == 201
    attempt_id = response.json()["id"]
    with Session(engine) as db:
        tasks = deepcopy(db.get(Attempt, attempt_id).questions)
    missed = set()
    for q in tasks:
        value = deepcopy(q["_correct"])
        if wrong_practice and q.get("answer_type") == "fields" and q["skill"] not in missed:
            key = next(iter(value)); value[key] = str(float(value[key])+1)
            missed.add(q["skill"])
        payload = {"values":value} if isinstance(value, dict) else {"choice":value}
        assert c.post(f"/api/assessments/{attempt_id}/answer", headers=headers,
                      json={"question_id":q["id"], **payload}).status_code == 200
    result = c.post(f"/api/assessments/{attempt_id}/finish", headers=headers)
    assert result.status_code == 200
    return result.json(), tasks


@pytest.mark.parametrize("spec", ["backend", "data"])
@pytest.mark.parametrize("grade", ["Junior", "Middle"])
def test_short_refresh_retains_three_unmeasured_families_and_grade(environment, actors, spec, grade):
    c, engine = environment; actor=actors["candidate"]; h=actor["headers"]
    with Session(engine) as db:
        candidate=db.get(Candidate,actor["id"])
        candidate.grade=None; candidate.specialization=None; db.add(candidate); db.commit()
    full, _ = submit(c,engine,h,spec=spec,grade=grade)
    assert full["result"]["evidence_outcome"] == "updated_full"
    short, _ = submit(c,engine,h,spec=spec,grade=grade,kind="challenge",wrong_practice=True)
    assert set(short["result"]["skills"].values()) == {0}
    assert short["result"]["evidence_outcome"] == "updated_families"
    profile=c.get("/api/profile",headers=h).json()
    assert profile["grade"] == grade
    assert all(s["score"]==75 and s["families"]==4 for s in profile["skills"].values())
    assert all(s["checked_at"] == full["finished_at"] for s in profile["skills"].values())
    assert "skill_portfolio" not in profile
    assert c.post(f"/api/assessments/{short['id']}/finish",headers=h).json()==short
    # Reclaiming an already awarded old grade must not restore its old 100% snapshot.
    assert c.post(f"/api/assessments/{full['id']}/claim-grade",headers=h).status_code==200
    assert c.get("/api/profile",headers=h).json()["skills"] == profile["skills"]
    with Session(engine) as db:
        state=db.get(Candidate,actor["id"]).skill_portfolio
        assert len(state["applied_attempts"])==2
        assert len(db.exec(select(GradeAward).where(GradeAward.candidate_id==actor["id"])).all())==1
    # Neither public candidate output nor the other employer gains private provenance/contact data.
    for employer in ("employer","employer2"):
        public=c.get(f"/api/candidates/{actor['id']}",headers=actors[employer]["headers"])
        assert public.status_code==200
        assert public.json()["contacts"] is None
        assert "skill_portfolio" not in public.text and "anchor_attempt_id" not in public.text
        assert all(s["score"]==75 for s in public.json()["skills"].values())


def test_new_full_form_resets_basis_without_changing_grade_in_cooldown(environment, actors):
    c,engine=environment;actor=actors["candidate"];h=actor["headers"]
    submit(c,engine,h)
    submit(c,engine,h,kind="challenge",wrong_practice=True)
    full,_=submit(c,engine,h)
    assert full["result"]["evidence_outcome"]=="updated_full"
    actual=c.get("/api/profile",headers=h).json()
    assert actual["grade"]=="Junior"
    assert all(s["score"]==100 and s["families"]==4 for s in actual["skills"].values())
    with Session(engine) as db:
        assert db.get(Candidate,actor["id"]).skill_portfolio["applied_attempts"]==[full["id"]]


def test_legacy_full_and_old_short_are_reconstructed_before_new_update(environment, actors):
    c,engine=environment;actor=actors["candidate"];h=actor["headers"]
    full,tasks=submit(c,engine,h)
    old_short, old_tasks=submit(c,engine,h,kind="challenge",wrong_practice=True)
    stamp=now()-10*86400
    with Session(engine) as db:
        p=db.get(Candidate,actor["id"]); p.skill_portfolio={}
        a=db.get(Attempt,full["id"]);a.created_at=stamp-600;a.finished_at=stamp
        b=db.get(Attempt,old_short["id"]);b.created_at=stamp+600;b.finished_at=stamp+1200
        db.add(p);db.add(a);db.add(b);db.commit()
    latest, new_tasks=submit(c,engine,h,kind="challenge")
    assert latest["result"]["evidence_outcome"]=="updated_families"
    expected={q["family"]:100 for q in tasks}
    for q in old_tasks: expected[q["family"]]=0
    for q in new_tasks: expected[q["family"]]=100
    skills=c.get("/api/profile",headers=h).json()["skills"]
    for skill, info in skills.items():
        assert info["score"] == sum(expected[q["family"]] for q in tasks if q["skill"]==skill)/4
        assert info["families"]==4 and info["checked_at"]==stamp


@pytest.mark.parametrize("legacy", ["missing", "v1", "v2", "ambiguous"])
def test_missing_incompatible_or_ambiguous_history_preserves_existing_snapshot(environment, actors, legacy):
    c,engine=environment;actor=actors["candidate"];h=actor["headers"]
    original=c.get("/api/profile",headers=h).json()["skills"]
    if legacy!="missing":
        version={"v1":"2026-10-v1", "v2":"2026-10-v2"}.get(legacy,RUBRIC["version"])
        tasks=build_test("backend","Junior",123,version=version)
        answers={q["id"]:q["_correct"] for q in tasks}
        with Session(engine) as db:
            for i in range(2 if legacy=="ambiguous" else 1):
                db.add(Attempt(candidate_id=actor["id"],specialization="backend",grade="Junior",seed=i,
                    kind="grade",rubric_version=version,status="completed",created_at=100,finished_at=200,
                    expires_at=300,questions=tasks,answers=answers,result=score_test(tasks,answers,rubric_version=version)))
            db.commit()
    short,_=submit(c,engine,h,kind="challenge",wrong_practice=True)
    assert short["result"]["evidence_outcome"]=="needs_full_assessment"
    actual=c.get("/api/profile",headers=h).json()
    assert actual["grade"]=="Junior" and actual["skills"]==original


def test_claiming_an_older_pending_full_form_keeps_newer_same_category_evidence(environment, actors):
    c,engine=environment;actor=actors["candidate"];h=actor["headers"]
    with Session(engine) as db:
        p=db.get(Candidate,actor["id"]);p.grade_changed_at=now();db.add(p);db.commit()
    old,_=submit(c,engine,h)
    new,_=submit(c,engine,h,wrong_practice=True)
    assert old["result"]["grade_outcome"]==new["result"]["grade_outcome"]=="cooldown"
    before=c.get("/api/profile",headers=h).json()["skills"]
    assert all(s["score"]==75 for s in before.values())
    with Session(engine) as db:
        p=db.get(Candidate,actor["id"]);p.grade_changed_at=now()-91*86400;db.add(p);db.commit()
    result=c.post(f"/api/assessments/{old['id']}/claim-grade",headers=h)
    assert result.status_code==200
    assert result.json()["result"]["evidence_outcome"]=="older_assessment_retained"
    assert c.get("/api/profile",headers=h).json()["skills"]==before


def test_adapter_is_the_verified_standalone_prototype():
    root=Path(__file__).resolve().parents[2]
    core=(root/"backend/app/evidence_portfolio.py").read_bytes()
    manifest=json.loads((root/"backend/portfolio_manifest.json").read_text())
    assert hashlib.sha256(core).hexdigest()==manifest["sha256"]
    prototype=root/manifest["prototype"]
    if prototype.exists():
        assert core==prototype.read_bytes()
