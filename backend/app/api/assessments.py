from app.responses import AnswerSaved, Assessment, Catalog
import secrets
from fastapi import APIRouter, HTTPException
from sqlalchemy import update
from sqlmodel import select
from app.assessment_bank import RUBRIC, SPECIALIZATIONS, SKILLS, answer_correct, build_test, public_questions, rubric_for, score_test
from app.practical_bank import validate_values
from app.core.config import settings
from app.core.security import CandidateUser, Db
from app.models import Attempt, AuditEvent, Candidate, Evidence, GradeAward, now
from app.schemas import AnswerInput, AttemptStart
from app.skill_projection import apply_skills

router = APIRouter(prefix="/assessments", tags=["Проверка навыков"])


def owned(db, attempt_id, user_id):
    attempt = db.exec(select(Attempt).where(Attempt.id == attempt_id).with_for_update()).first()
    if not attempt or attempt.candidate_id != user_id:
        raise HTTPException(404, "Проверка не найдена.")
    return attempt


def output(attempt):
    return {"id": attempt.id, "specialization": attempt.specialization, "grade": attempt.grade,
            "kind": attempt.kind, "status": attempt.status, "answers": attempt.answers,
            "questions": public_questions(attempt.questions, attempt.status == "completed", attempt.answers),
            "result": attempt.result, "created_at": attempt.created_at, "expires_at": attempt.expires_at,
            "finished_at": attempt.finished_at, "rubric": rubric_for(attempt.rubric_version),
            "rubric_version": attempt.rubric_version, "is_demo": attempt.is_demo}


def award(db, candidate, attempt):
    existing = db.exec(select(GradeAward).where(GradeAward.attempt_id == attempt.id)).first()
    if existing:
        return "awarded"
    available = candidate.grade_changed_at + settings.grade_cooldown_days * 86400
    if candidate.grade and available > now():
        return "cooldown"
    candidate.specialization = attempt.specialization
    candidate.grade = attempt.grade
    candidate.grade_changed_at = now()
    db.add(GradeAward(candidate_id=candidate.user_id, attempt_id=attempt.id,
                      specialization=attempt.specialization, grade=attempt.grade))
    db.add(candidate)
    return "awarded"


@router.get("/catalog", response_model=Catalog)
def catalog() -> dict:
    return {"specializations": SPECIALIZATIONS, "grades": ["Junior", "Middle"], "skills": SKILLS,
            "question_count": 12, "practical_count": 6, "duration_minutes": 45,
            "challenge_count": 3, "challenge_duration_minutes": 12, "rubric": RUBRIC,
            "cooldown_days": settings.grade_cooldown_days}


@router.get("", response_model=list[Assessment], response_model_exclude_unset=True)
def history(user: CandidateUser, db: Db) -> list[dict]:
    attempts = db.exec(select(Attempt).where(Attempt.candidate_id == user.id).order_by(Attempt.created_at.desc())).all()
    return [output(a) for a in attempts]


@router.post("", status_code=201, response_model=Assessment, response_model_exclude_unset=True)
def start(data: AttemptStart, user: CandidateUser, db: Db) -> dict:
    candidate = db.exec(select(Candidate).where(Candidate.user_id == user.id).with_for_update()).one()
    if not candidate.processing_consent_at:
        raise HTTPException(403, "Для проверки необходимо согласие на обработку данных.")
    active = db.exec(select(Attempt).where(Attempt.candidate_id == user.id,
                     Attempt.status == "in_progress", Attempt.expires_at > now())).first()
    if active:
        if (active.specialization, active.grade, active.kind) == (data.specialization, data.grade, data.kind):
            return output(active)
        raise HTTPException(409, "Сначала завершите или отмените текущую проверку.")
    recent = db.exec(select(Attempt).where(Attempt.candidate_id == user.id,
                     Attempt.created_at > now() - 86400)).all()
    if len(recent) >= 8:
        raise HTTPException(429, "На сегодня достигнут лимит проверок. Вернитесь завтра.")
    if data.kind == "challenge":
        if not candidate.grade:
            raise HTTPException(409, "Сначала подтвердите категорию основной проверкой.")
        if (data.specialization, data.grade) != (candidate.specialization, candidate.grade):
            raise HTTPException(422, "Короткая проверка проводится по текущей подтверждённой категории.")
        previous = db.exec(select(Attempt).where(Attempt.candidate_id == user.id, Attempt.kind == "challenge",
                           Attempt.status == "completed", Attempt.finished_at > now() - 7 * 86400)).first()
        if previous:
            raise HTTPException(409, "Следующая короткая проверка будет доступна через семь дней.")
    seed = secrets.randbits(31)
    attempt = Attempt(candidate_id=user.id, specialization=data.specialization, grade=data.grade,
                      kind=data.kind, seed=seed, rubric_version=RUBRIC["version"],
                      questions=build_test(data.specialization, data.grade, seed, data.kind),
                      expires_at=now() + (2700 if data.kind == "grade" else 720))
    db.add(attempt)
    db.commit()
    return output(attempt)


@router.get("/{attempt_id}", response_model=Assessment, response_model_exclude_unset=True)
def detail(attempt_id: str, user: CandidateUser, db: Db) -> dict:
    return output(owned(db, attempt_id, user.id))


@router.post("/{attempt_id}/answer", response_model=AnswerSaved)
def answer(attempt_id: str, data: AnswerInput, user: CandidateUser, db: Db) -> dict:
    attempt = owned(db, attempt_id, user.id)
    if attempt.status != "in_progress" or attempt.expires_at <= now():
        raise HTTPException(409, "Проверка уже завершена или время вышло.")
    q = next((q for q in attempt.questions if q["id"] == data.question_id), None)
    if not q:
        raise HTTPException(422, "Задание не найдено.")
    if q.get("answer_type") == "fields":
        try:
            validate_values(q, data.values)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        # Store the submitted representation; normalization is for grading only.
        value = data.values
    elif data.choice is None or data.choice >= len(q["choices"]):
        raise HTTPException(422, "Вариант ответа не найден.")
    else:
        value = data.choice
    attempt.answers = {**attempt.answers, data.question_id: value}
    db.add(attempt)
    db.commit()
    return {"saved": True, "answered": len(attempt.answers)}


@router.post("/{attempt_id}/cancel", response_model=Assessment, response_model_exclude_unset=True)
def cancel(attempt_id: str, user: CandidateUser, db: Db) -> dict:
    attempt = owned(db, attempt_id, user.id)
    if attempt.status == "in_progress":
        attempt.status = "cancelled"
        db.add(attempt)
        db.commit()
    return output(attempt)


@router.post("/{attempt_id}/finish", response_model=Assessment, response_model_exclude_unset=True)
def finish(attempt_id: str, user: CandidateUser, db: Db) -> dict:
    # Lock the profile to serialize concurrent grade decisions on PostgreSQL.
    candidate = db.exec(select(Candidate).where(Candidate.user_id == user.id).with_for_update()).one()
    attempt = owned(db, attempt_id, user.id)
    if attempt.status == "completed":
        return output(attempt)
    if attempt.status != "in_progress":
        raise HTTPException(409, "Эта проверка отменена.")
    if attempt.expires_at <= now():
        raise HTTPException(409, "Время проверки истекло. Начните новую попытку.")
    if len(attempt.answers) != len(attempt.questions):
        raise HTTPException(422, "Ответьте на все задания перед завершением.")
    changed = db.exec(update(Attempt).where(Attempt.id == attempt.id, Attempt.status == "in_progress").values(status="completed"))
    if changed.rowcount != 1:
        raise HTTPException(409, "Результат уже обрабатывается.")
    result = score_test(attempt.questions, attempt.answers, rubric_version=attempt.rubric_version)
    attempt.status = "completed"
    attempt.finished_at = now()
    for task in attempt.questions:
        db.add(Evidence(candidate_id=user.id, attempt_id=attempt.id, skill=task["skill"], family=task["family"],
                        score=100.0 if answer_correct(task, attempt.answers[task["id"]]) else 0.0))
    outcome = award(db, candidate, attempt) if result["passed"] and attempt.kind == "grade" else ("freshness" if attempt.kind == "challenge" else "not_passed")
    # A pending category (including an easier/lower test) cannot refresh another category's evidence.
    evidence_outcome = "not_updated"
    if result["passed"] or attempt.kind == "challenge":
        evidence_outcome = apply_skills(db, candidate, attempt, result)
    result.update(grade_outcome=outcome, current_grade=candidate.grade, evidence_outcome=evidence_outcome,
                  next_grade_change_at=candidate.grade_changed_at + settings.grade_cooldown_days * 86400 if candidate.grade else 0)
    attempt.result = result
    candidate.updated_at = now()
    db.add(candidate)
    db.add(attempt)
    db.add(AuditEvent(actor_id=user.id, action="assessment_completed", entity_id=attempt.id))
    db.commit()
    return output(attempt)


@router.post("/{attempt_id}/claim-grade", response_model=Assessment, response_model_exclude_unset=True)
def claim(attempt_id: str, user: CandidateUser, db: Db) -> dict:
    candidate = db.exec(select(Candidate).where(Candidate.user_id == user.id).with_for_update()).one()
    attempt = owned(db, attempt_id, user.id)
    if attempt.status != "completed" or attempt.kind != "grade" or not attempt.result.get("passed"):
        raise HTTPException(409, "Эта попытка не подтверждает грейд.")
    if db.exec(select(GradeAward.id).where(GradeAward.attempt_id == attempt.id)).first():
        return output(attempt)
    if award(db, candidate, attempt) != "awarded":
        raise HTTPException(409, "Интервал между изменениями грейда ещё не прошёл.")
    evidence_outcome = apply_skills(db, candidate, attempt, attempt.result, claimed=True)
    candidate.updated_at = now()
    db.add(candidate)
    attempt.result = {**attempt.result, "grade_outcome": "awarded", "current_grade": candidate.grade,
                      "evidence_outcome": evidence_outcome}
    db.add(attempt)
    db.commit()
    return output(attempt)
