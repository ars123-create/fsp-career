"""Deterministic fictional demonstration data. Never run in non-demo mode."""
import secrets
from sqlmodel import select
from app.assessment_bank import build_test, score_test, SPECIALIZATIONS
from app.core.security import hash_password, verify_password
from app.core.config import settings
from app.models import User, Candidate, Company, Job, Invitation, Attempt, Evidence, GradeAward, now


# Public, fictional reviewer accounts, explicitly requested for the demo stand.
DEMO_LOGIN_EMAILS = ("candidate@demo.fsp.test", "employer@demo.fsp.test")
DEMO_LOGIN_PASSWORD = "12345678"


def configure_demo_logins(db):
    if not settings.demo_mode:
        return
    for email in DEMO_LOGIN_EMAILS:
        user = db.exec(select(User).where(User.email == email, User.is_demo == True)).first()
        if user and not verify_password(DEMO_LOGIN_PASSWORD, user.hashed_password):
            user.hashed_password = hash_password(DEMO_LOGIN_PASSWORD)
            db.add(user)
    db.commit()


def seed_demo(db):
    if not settings.demo_mode:
        return
    if db.exec(select(User).where(User.email == "employer@demo.fsp.test")).first():
        configure_demo_logins(db)
        return
    stamp = now()
    # Other fictional profiles have unguessable passwords; only two reviewer accounts are public.
    password = hash_password(secrets.token_urlsafe(32))
    employers = []
    for i, email in enumerate(("employer@demo.fsp.test", "employer2@demo.fsp.test")):
        user = User(email=email, full_name=("Анна · демо", "Михаил · демо")[i],
                    hashed_password=password, role="employer", verified=True, is_demo=True)
        db.add(user)
        db.flush()
        db.add(Company(user_id=user.id, name=("Орбита Технологии", "Вектор Данных")[i],
            description="Демонстрационная продуктовая команда. Все вакансии и предложения в демо вымышленные.",
            industry="Разработка цифровых продуктов", contact=email, website="https://example.org"))
        employers.append(user)
    for i in range(24):
        spec = "backend" if i < 12 else "data"
        grade = "Middle" if i % 3 else "Junior"
        user = User(email="candidate@demo.fsp.test" if i == 0 else f"talent{i:02}@demo.fsp.test",
            full_name="Алексей Морозов · демо" if i == 0 else f"Демо-специалист {i + 1:02}",
            hashed_password=password, role="candidate", verified=True, is_demo=True)
        db.add(user)
        db.flush()
        ts = stamp - (i * 7 + 1) * 86400
        questions = build_test(spec, grade, 20261007 + i, "grade", version="2026-10-v1")
        answers = {q["id"]: q["_correct"] for q in questions}
        if i % 2:
            answers[questions[0]["id"]] = (questions[0]["_correct"] + 1) % len(questions[0]["choices"])
        result = score_test(questions, answers)
        result.update(grade_outcome="awarded", current_grade=grade, next_grade_change_at=ts + 90 * 86400)
        a = Attempt(candidate_id=user.id, specialization=spec, grade=grade, seed=20261007 + i,
            questions=questions, answers=answers, result=result, status="completed", created_at=ts - 600,
            finished_at=ts, expires_at=ts + 1000, is_demo=True)
        db.add(a)
        db.flush()
        skills = {s: {"score": v, "families": 2, "checked_at": ts, "source": "test"} for s, v in result["skills"].items()}
        fsp = i % 4 == 0
        c = Candidate(user_id=user.id, headline=SPECIALIZATIONS[spec],
            about="Разрабатываю API и работаю с данными. Люблю понятные требования, проверяемые результаты и обратную связь." if spec == "backend" else "Исследую данные, составляю SQL-запросы и проверяю качество отчётности. Ищу продуктовую команду.",
            city=("Москва", "Санкт-Петербург", "Казань", "Екатеринбург")[i % 4],
            phone=f"+7 000 000 {i:02} 00", telegram=f"demo_talent_{i:02}", experience_years=1 + i % 5,
            declared_specialization=spec, declared_grade=grade,
            stack=["Python", "SQL", "FastAPI", "Docker"] if spec == "backend" else ["SQL", "Python", "Pandas", "BI"],
            soft_skills=["Командная работа", "Самостоятельность"], work_format="remote" if i % 5 else "hybrid",
            salary_min=90000 if grade == "Junior" else 180000, salary_max=150000 if grade == "Junior" else 260000,
            published=True, processing_consent_at=ts, publication_consent_at=ts,
            specialization=spec, grade=grade, grade_changed_at=ts, skills=skills,
            fsp_id=f"DEMO-{i:04}" if fsp else "", achievements=[{"title": "Командный турнир по программированию · демо", "year": 2026,
                "source": "Демонстрационный реестр ФСП", "verified": True, "is_demo": True, "revoked": False}] if fsp else [], updated_at=ts)
        db.add(c)
        db.add(GradeAward(candidate_id=user.id, attempt_id=a.id, specialization=spec, grade=grade, created_at=ts))
        for q in questions:
            db.add(Evidence(candidate_id=user.id, attempt_id=a.id, skill=q["skill"], family=q["family"],
                score=100.0 if answers[q["id"]] == q["_correct"] else 0.0, created_at=ts, is_demo=True))
        if i == 0:
            db.add(Invitation(employer_id=employers[0].id, candidate_id=user.id, title="Python-разработчик в продуктовую команду",
                description="Приглашаем обсудить разработку API сервиса аналитики. Гибридный формат, наставник и небольшая команда. Это демонстрационное предложение.",
                salary_min=110000, salary_max=150000, contact=employers[0].email, expires_at=stamp + 14 * 86400))
    for i, spec in enumerate(("backend", "data")):
        db.add(Job(employer_id=employers[i].id, title=("Python-разработчик", "Аналитик данных")[i],
            description="Демонстрационная вакансия: работа над продуктом, понятные задачи и регулярная обратная связь. Рассматриваем специалистов без истории ФСП.",
            specialization=spec, grade="Junior", skills=["Python", "SQL"] if not i else ["SQL", "Аналитика"],
            work_format="remote", salary_min=100000, salary_max=160000))
    db.commit()
    configure_demo_logins(db)
