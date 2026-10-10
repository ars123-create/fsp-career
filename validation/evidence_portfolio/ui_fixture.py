"""Fictional isolated UI setup; full assessment answers are supplied by the harness."""
import json
from pathlib import Path

from fastapi.testclient import TestClient
from sqlmodel import Session, select

from app.assessment_bank import build_test, score_test
from app.core.db import engine
from app.core.security import hash_password, issue_access
from app.main import app
from app.models import Attempt, Candidate, Company, Evidence, GradeAward, User, now


def main():
    # The runner must explicitly point FSP_DATABASE_URL at this dedicated file.
    assert engine.url.database.endswith("portfolio-ui-0.4.4.sqlite")
    password=hash_password("PortfolioFictionalOnly2026!")
    stamp=now()
    with Session(engine) as db:
        assert db.exec(select(User)).first() is None, "Never overwrite an existing UI fixture"
        normal=User(email="candidate@demo.fsp.test",full_name="Проверка семейств · демо",hashed_password=password,
                    role="candidate",verified=True,is_demo=True)
        legacy=User(email="legacy@demo.fsp.test",full_name="Историческая рубрика · демо",hashed_password=password,
                    role="candidate",verified=True,is_demo=True)
        employer=User(email="employer@demo.fsp.test",full_name="Компания проверки · демо",hashed_password=password,
                      role="employer",verified=True,is_demo=True)
        db.add(normal);db.add(legacy);db.add(employer);db.flush()
        for user in (normal,legacy):
            db.add(Candidate(user_id=user.id,headline="Python backend · проверка семейств",published=True,
                processing_consent_at=stamp,publication_consent_at=stamp,work_format="remote",
                salary_min=180000,salary_max=240000,stack=["Python","SQL","API"]))
        db.add(Company(user_id=employer.id,name="Проверка портфеля · демо",contact="company@example.org"))
        db.commit()
        normal_id,legacy_id=normal.id,legacy.id
        headers={"Authorization":"Bearer "+issue_access(normal)}
    # Prepare one real API-produced full result. This is not a human/UI experiment.
    with TestClient(app) as client:
        full=client.post("/api/assessments",headers=headers,json={"specialization":"backend","grade":"Middle"})
        assert full.status_code==201
        attempt_id=full.json()["id"]
        with Session(engine) as db:questions=db.get(Attempt,attempt_id).questions
        for q in questions:
            value=q["_correct"]
            payload={"values":value} if isinstance(value,dict) else {"choice":value}
            assert client.post(f"/api/assessments/{attempt_id}/answer",headers=headers,
                               json={"question_id":q["id"],**payload}).status_code==200
        assert client.post(f"/api/assessments/{attempt_id}/finish",headers=headers).json()["result"]["evidence_outcome"]=="updated_full"
    with Session(engine) as db:
        questions=build_test("backend","Middle",729,version="2026-10-v1")
        answers={q["id"]:q["_correct"] for q in questions}
        result=score_test(questions,answers)
        old=Attempt(candidate_id=legacy_id,specialization="backend",grade="Middle",rubric_version="2026-10-v1",
            seed=729,questions=questions,answers=answers,result=result,status="completed",created_at=stamp-700,
            finished_at=stamp-100,expires_at=stamp+300,is_demo=True)
        db.add(old);db.flush()
        p=db.get(Candidate,legacy_id);p.specialization="backend";p.grade="Middle";p.grade_changed_at=stamp-100
        p.skills={s:{"score":v,"families":2,"checked_at":stamp-100,"source":"test"} for s,v in result["skills"].items()}
        db.add(p);db.add(GradeAward(candidate_id=legacy_id,attempt_id=old.id,specialization="backend",grade="Middle"))
        for q in questions:db.add(Evidence(candidate_id=legacy_id,attempt_id=old.id,skill=q["skill"],family=q["family"],score=100,is_demo=True))
        db.commit()
    metadata={"scope":"fictional isolated fixture; full v2 via API oracle, v1 historical rows seeded; browser checks only subsequent short forms",
              "normal_candidate_id":normal_id,"legacy_candidate_id":legacy_id,"full_attempt_id":attempt_id}
    Path("output/validation/evidence-portfolio-ui-fixture.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print("UI fixture ready: two fictional candidates, one employer; no working database used")


if __name__=="__main__":main()
