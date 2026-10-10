from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import re
from pypdf import PdfReader
from sqlmodel import Session, select
from app.core.config import settings
from app.models import ActionToken, Attempt, Candidate, Evidence, GradeAward, Invitation, User, now
from conftest import PASSWORD


def send(client, actors, candidate="candidate", employer="employer"):
    r = client.post("/api/invitations", headers=actors[employer]["headers"], json={"candidate_id": actors[candidate]["id"],
        "title": "Python разработчик", "description": "Наша команда строит API для аналитики.",
        "salary_min": 110000, "salary_max": 170000, "currency": "RUB", "contact": "hr@example.org"})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def pdf_text(response):
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/pdf"
    return "\n".join(p.extract_text() for p in PdfReader(BytesIO(response.content)).pages)


def test_contact_boundary_api_pdf_and_second_employer(environment, actors):
    c, engine = environment; candidate = actors["candidate"]; employer = actors["employer"]
    path = f"/api/candidates/{candidate['id']}"
    def hidden():
        for who in ("employer", "employer2"):
            r = c.get(path, headers=actors[who]["headers"])
            assert r.json()["contacts"] is None
            assert candidate["email"] not in r.text and "private_handle" not in r.text and "Private candidate" not in r.text
            text = pdf_text(c.get(path + "/pdf", headers=actors[who]["headers"]))
            assert "Контакты" in text and "Закрыты" in text
            assert candidate["email"] not in text and "private_handle" not in text
    hidden()
    inv = send(c, actors)
    hidden()
    for _ in range(2):
        r = c.post(f"/api/invitations/{inv}/action", headers=candidate["headers"], json={"action": "accept"})
        assert r.status_code == 200 and r.json()["status"] == "accepted"
    assert c.get(path, headers=employer["headers"]).json()["contacts"]["email"] == candidate["email"]
    assert candidate["email"] in pdf_text(c.get(path + "/pdf", headers=employer["headers"]))
    assert c.get(path, headers=actors["employer2"]["headers"]).json()["contacts"] is None
    assert c.post(f"/api/invitations/{inv}/action", headers=candidate["headers"], json={"action": "decline"}).status_code == 409
    assert c.post(f"/api/invitations/{inv}/action", headers=candidate["headers"], json={"action": "revoke_contacts"}).status_code == 200
    hidden()


def test_roles_and_object_ownership(environment, actors):
    c, _ = environment; inv = send(c, actors)
    for who in ("candidate", "other"):
        h = actors[who]["headers"]
        assert c.post("/api/search", headers=h, json={}).status_code == 403
        assert c.get(f"/api/candidates/{actors['candidate']['id']}", headers=h).status_code == 403
        assert c.get(f"/api/candidates/{actors['candidate']['id']}/pdf", headers=h).status_code == 403
        assert c.get("/api/company", headers=h).status_code == 403
    assert c.get("/api/profile", headers=actors["employer"]["headers"]).status_code == 403
    assert c.get("/api/assessments", headers=actors["employer"]["headers"]).status_code == 403
    for who in ("other", "employer2"):
        assert c.get("/api/invitations", headers=actors[who]["headers"]).json() == []
        assert c.post(f"/api/invitations/{inv}/action", headers=actors[who]["headers"], json={"action": "accept"}).status_code == 404
    assert c.post(f"/api/invitations/{inv}/action", headers=actors["employer"]["headers"], json={"action": "accept"}).status_code == 403


def test_concurrent_conflicting_invitation_actions(environment, actors):
    c, engine = environment; inv = send(c, actors)
    def act(action): return c.post(f"/api/invitations/{inv}/action", headers=actors["candidate"]["headers"], json={"action": action}).status_code
    with ThreadPoolExecutor(max_workers=2) as pool:
        codes = list(pool.map(act, ["accept", "decline"]))
    assert sorted(codes) == [200, 409]
    with Session(engine) as db: assert db.get(Invitation, inv).status in ("accepted", "declined")


def test_concurrent_duplicate_invitation_has_single_record(environment, actors):
    c, engine = environment
    data = {"candidate_id": actors["candidate"]["id"], "title": "Backend developer", "description": "Описание предложения в команду", "salary_min": 100000, "salary_max": 180000, "contact": "hr@example.org"}
    def invite(_): return c.post("/api/invitations", headers=actors["employer"]["headers"], json=data).status_code
    with ThreadPoolExecutor(max_workers=2) as pool: codes = list(pool.map(invite, range(2)))
    assert sorted(codes) == [201, 409]
    with Session(engine) as db: assert len(db.exec(select(Invitation)).all()) == 1


def test_expiry_and_salary_constraints(environment, actors):
    c, engine = environment; inv = send(c, actors)
    with Session(engine) as db:
        row = db.get(Invitation, inv); row.expires_at = now() - 1; db.add(row); db.commit()
    assert c.post(f"/api/invitations/{inv}/action", headers=actors["candidate"]["headers"], json={"action": "accept"}).status_code == 409
    assert c.get("/api/invitations", headers=actors["candidate"]["headers"]).json()[0]["status"] == "expired"
    data = {"candidate_id": actors["candidate"]["id"], "title": "Role", "description": "Description of the role", "contact": "hr@example.org"}
    for salary in ({}, {"salary_min": 0, "salary_max": 500}, {"salary_min": 500, "salary_max": 100}, {"salary_min": 10, "salary_max": 100, "currency": "USD"}):
        assert c.post("/api/invitations", headers=actors["employer"]["headers"], json=data | salary).status_code == 422


def test_saved_selection_rechecks_publication_and_owner(environment, actors):
    c, engine = environment; h = actors["employer"]["headers"]
    saved = c.post("/api/needs", headers=h, json={"query": "Python junior удалённо 100–180 тыс", "fsp_only": False, "title": "Набор"})
    assert saved.status_code == 201, saved.text
    n = saved.json(); assert actors["candidate"]["id"] in n["saved_ids"]
    assert c.get(f"/api/needs/{n['id']}", headers=actors["employer2"]["headers"]).status_code == 404
    with Session(engine) as db:
        p = db.get(Candidate, actors["candidate"]["id"]); p.published = False; db.add(p); db.commit()
    result = c.get(f"/api/needs/{n['id']}", headers=h).json()
    assert all(p["id"] != actors["candidate"]["id"] for p in result["previous"] + result["current"]["candidates"])
    assert c.get(f"/api/candidates/{actors['candidate']['id']}/pdf", headers=h).status_code == 404


def test_application_grants_only_target_company_then_revokes(environment, actors):
    c, engine = environment; h = actors["candidate"]["headers"]
    # An ungraded, unpublished candidate can apply.
    with Session(engine) as db:
        p = db.get(Candidate, actors["candidate"]["id"]); p.grade = None; p.published = False; db.add(p); db.commit()
    j = c.post("/api/jobs", headers=actors["employer"]["headers"], json={"title": "Разработчик", "description": "Тестовая вакансия без требований ФСП", "specialization": "backend", "grade": "Junior", "skills": ["SQL"], "salary_min": 100000, "salary_max": 180000}).json()
    assert c.post(f"/api/jobs/{j['id']}/apply", headers=h, json={"contact_consent": False}).status_code == 422
    first = c.post(f"/api/jobs/{j['id']}/apply", headers=h, json={"contact_consent": True}).json()
    assert c.post(f"/api/jobs/{j['id']}/apply", headers=h, json={"contact_consent": True}).json()["id"] == first["id"]
    path = f"/api/candidates/{actors['candidate']['id']}"
    assert c.get(path, headers=actors["employer"]["headers"]).json()["contacts"]["email"] == actors["candidate"]["email"]
    assert c.get(path, headers=actors["employer2"]["headers"]).status_code == 404
    assert c.post(f"/api/applications/{first['id']}/status", headers=actors["employer2"]["headers"], json={"status": "interview"}).status_code == 404
    assert c.post(f"/api/applications/{first['id']}/status", headers=h, json={"status": "withdrawn"}).status_code == 200
    assert c.get(path, headers=actors["employer"]["headers"]).status_code == 404


def test_fsp_id_and_profile_cannot_forge_grade(environment, actors):
    c, engine = environment; h = actors["candidate"]["headers"]
    data = c.get("/api/profile", headers=h).json()
    from app.schemas import CandidateUpdate
    payload = {k: v for k, v in data.items() if k in CandidateUpdate.model_fields}
    assert c.put("/api/profile", headers=h, json=payload | {"grade": "Middle"}).status_code == 422
    assert c.put("/api/profile", headers=h, json=payload | {"about": "Напишите private@example.org"}).status_code == 422
    assert c.put("/api/profile", headers=h, json=payload | {"publication_consent": False}).status_code == 422
    assert c.put("/api/profile", headers=h, json=payload | {"published": False, "publication_consent": False}).status_code == 200
    r = c.post("/api/profile/fsp", headers=h, json={"fsp_id": "ANY-OTHER-ID"})
    assert r.json()["status"] == "pending" and r.json()["achievements"] == []
    assert c.get("/api/profile", headers=h).json()["grade"] == "Junior"
    assert c.post("/api/profile/import-pdf", headers=h, files={"file": ("fake.pdf", b"not a PDF", "application/pdf")}).status_code == 422


def complete(c, engine, h, grade="Junior", passed=True, kind="grade"):
    r = c.post("/api/assessments", headers=h, json={"specialization": "backend", "grade": grade, "kind": kind})
    assert r.status_code == 201, r.text
    a = r.json(); assert "_correct" not in r.text and "correct_choice" not in r.text
    with Session(engine) as db: questions = db.get(Attempt, a["id"]).questions
    for q in questions:
        if q.get("answer_type") == "fields":
            value = q["_correct"] if passed else {k: str(float(v) + 1) for k, v in q["_correct"].items()}
            payload = {"values": value}
        else:
            payload = {"choice": q["_correct"] if passed else (q["_correct"] + 1) % len(q["choices"])}
        assert c.post(f"/api/assessments/{a['id']}/answer", headers=h, json={"question_id": q["id"], **payload}).status_code == 200
    r = c.post(f"/api/assessments/{a['id']}/finish", headers=h)
    assert r.status_code == 200, r.text
    return r.json()


def test_grade_failure_cooldown_and_idempotency(environment, actors):
    c, engine = environment; h = actors["candidate"]["headers"]
    with Session(engine) as db:
        p = db.get(Candidate, actors["candidate"]["id"]); p.grade = None; p.specialization = None; db.add(p); db.commit()
    failed = complete(c, engine, h, passed=False)
    assert failed["result"]["current_grade"] is None
    first = complete(c, engine, h)
    assert first["result"]["grade_outcome"] == "awarded"
    failed = complete(c, engine, h, grade="Middle", passed=False)
    assert failed["result"]["current_grade"] == "Junior"
    higher = complete(c, engine, h, grade="Middle")
    assert higher["result"]["grade_outcome"] == "cooldown"
    assert higher["result"]["current_grade"] == "Junior"
    again = c.post(f"/api/assessments/{higher['id']}/finish", headers=h).json()
    assert again == higher
    assert c.post(f"/api/assessments/{higher['id']}/claim-grade", headers=h).status_code == 409
    with Session(engine) as db:
        p = db.get(Candidate, actors["candidate"]["id"]); p.grade_changed_at = now() - 91 * 86400; db.add(p); db.commit()
    assert c.post(f"/api/assessments/{higher['id']}/claim-grade", headers=h).json()["result"]["current_grade"] == "Middle"
    with Session(engine) as db:
        assert len(db.exec(select(Evidence).where(Evidence.attempt_id == higher["id"])).all()) == 12
        assert len(db.exec(select(GradeAward).where(GradeAward.attempt_id == higher["id"])).all()) == 1


def test_assessment_owner_expiry_and_weekly_challenge(environment, actors):
    c, engine = environment; h = actors["candidate"]["headers"]
    a = c.post("/api/assessments", headers=h, json={"specialization": "backend", "grade": "Junior"}).json()
    assert c.get(f"/api/assessments/{a['id']}", headers=actors["other"]["headers"]).status_code == 404
    assert c.post(f"/api/assessments/{a['id']}/finish", headers=h).status_code == 422
    with Session(engine) as db:
        p = db.get(Attempt, a["id"]); p.expires_at = now() - 1; db.add(p); db.commit()
    assert c.post(f"/api/assessments/{a['id']}/finish", headers=h).status_code == 409
    assert c.post("/api/assessments", headers=h, json={"specialization": "data", "grade": "Middle", "kind": "challenge"}).status_code == 422
    result = complete(c, engine, h, kind="challenge", passed=False)
    assert result["result"]["current_grade"] == "Junior"
    assert c.post("/api/assessments", headers=h, json={"specialization": "backend", "grade": "Junior", "kind": "challenge"}).status_code == 409


def test_pending_lower_grade_cannot_refresh_current_skill_evidence(environment, actors):
    c, engine = environment; h = actors["candidate"]["headers"]
    with Session(engine) as db:
        profile = db.get(Candidate, actors["candidate"]["id"])
        profile.grade = "Middle"; profile.grade_changed_at = now()
        profile.skills = {"Python": {"score": 75, "families": 2, "checked_at": now() - 30 * 86400, "source": "test"}}
        before = dict(profile.skills); db.add(profile); db.commit()
    lower = complete(c, engine, h, grade="Junior")
    assert lower["result"]["grade_outcome"] == "cooldown"
    actual = c.get("/api/profile", headers=h).json()
    assert actual["grade"] == "Middle" and actual["skills"] == before
    with Session(engine) as db:
        profile = db.get(Candidate, actors["candidate"]["id"])
        profile.grade_changed_at = now() - 91 * 86400; db.add(profile); db.commit()
    assert c.post(f"/api/assessments/{lower['id']}/claim-grade", headers=h).status_code == 200
    actual = c.get("/api/profile", headers=h).json()
    assert actual["grade"] == "Junior" and set(actual["skills"]) == {"Python", "SQL", "API"}
    assert actual["skills"]["Python"]["checked_at"] == lower["finished_at"]


def test_validation_does_not_echo_secrets(environment):
    c, _ = environment
    secret = "secret!"
    response = c.post("/api/auth/register", json={"email": "invalid", "password": secret, "full_name": "X", "role": "candidate", "processing_consent": True})
    assert response.status_code == 422
    assert secret not in response.text
    assert all(set(issue) == {"loc", "msg", "type"} for issue in response.json()["detail"])


def read_mail_token(c, receipt):
    body = c.get(f"/api/auth/local-mail/{receipt}").json()["body"]
    return re.search(r"token=([^\s]+)", body)[1]


def test_registration_verification_reset_and_email_change(environment):
    c, engine = environment
    r = c.post("/api/auth/register", json={"email": "new@example.org", "password": PASSWORD, "full_name": "Новый кандидат", "role": "candidate", "processing_consent": True})
    assert r.status_code == 201, r.text
    token = read_mail_token(c, r.json()["local_mail_receipt"])
    assert c.post("/api/auth/login", data={"username": "new@example.org", "password": PASSWORD}).status_code == 403
    assert c.post("/api/auth/verify", json={"token": token}).status_code == 200
    assert c.post("/api/auth/verify", json={"token": token}).status_code == 400
    login = c.post("/api/auth/login", data={"username": "new@example.org", "password": PASSWORD}).json()
    h = {"Authorization": "Bearer " + login["access_token"]}
    assert c.get("/api/auth/me", headers=h).status_code == 200
    change = c.post("/api/auth/change-email", headers=h, json={"email": "changed@example.org", "password": PASSWORD}).json()
    assert c.get("/api/auth/me", headers=h).json()["email"] == "new@example.org"
    assert c.post("/api/auth/verify", json={"token": read_mail_token(c, change["local_mail_receipt"])}).status_code == 200
    assert c.get("/api/auth/me", headers=h).status_code == 401
    new = c.post("/api/auth/login", data={"username": "changed@example.org", "password": PASSWORD}).json()
    h = {"Authorization": "Bearer " + new["access_token"]}
    reset = c.post("/api/auth/forgot-password", json={"email": "changed@example.org"}).json()
    token = read_mail_token(c, reset["local_mail_receipt"])
    assert c.post("/api/auth/reset-password", json={"token": token, "password": "DifferentPass2026!"}).status_code == 200
    assert c.get("/api/auth/me", headers=h).status_code == 401
    assert c.post("/api/auth/reset-password", json={"token": token, "password": PASSWORD}).status_code == 400
    assert c.post("/api/auth/login", data={"username": "changed@example.org", "password": PASSWORD}).status_code == 401


def test_expired_verification_production_demo_off_and_logout(environment, actors, monkeypatch):
    c, engine = environment
    r = c.post("/api/auth/forgot-password", json={"email": actors["candidate"]["email"]}).json()
    token = read_mail_token(c, r["local_mail_receipt"])
    with Session(engine) as db:
        t = db.exec(select(ActionToken)).one(); t.expires_at = now() - 1; db.add(t); db.commit()
    assert c.post("/api/auth/reset-password", json={"token": token, "password": PASSWORD}).status_code == 400
    monkeypatch.setattr(settings, "demo_mode", False)
    assert c.post("/api/auth/demo/candidate").status_code == 404
    assert c.get(f"/api/auth/local-mail/{r['local_mail_receipt']}").status_code == 404
    h = actors["candidate"]["headers"]
    assert c.post("/api/auth/logout", headers=h).status_code == 200
    assert c.get("/api/auth/me", headers=h).status_code == 401
