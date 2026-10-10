from app.responses import ApplicationView, InvitationView, JobView
from fastapi import APIRouter, HTTPException
from sqlalchemy import update
from sqlmodel import select
from app.core.security import Current, CandidateUser, EmployerUser, Db
from app.models import Application, AuditEvent, Candidate, Company, Invitation, Job, User, now
from app.schemas import ApplicationAction, ApplicationInput, InviteAction, InviteInput, JobInput
from app.visibility import candidate_view, contact_allowed

router = APIRouter(tags=["Предложения и отклики"])


def company_public(db, employer_id):
    company = db.get(Company, employer_id)
    return company.model_dump(exclude={"user_id"}) if company else {}


def ensure_company(db, user):
    company = db.get(Company, user.id)
    if not company or not company.name or not company.contact:
        raise HTTPException(409, "Сначала заполните название компании и способ связи.")


def invite_view(inv, user, db):
    db.refresh(inv)
    result = inv.model_dump()
    result["company"] = company_public(db, inv.employer_id)
    if inv.status in ("sent", "viewed") and inv.expires_at <= now(): result["status"] = "expired"
    c = db.get(Candidate, inv.candidate_id)
    if user.role == "employer":
        grant = contact_allowed(db, user.id, inv.candidate_id)
        result["candidate"] = candidate_view(db, c, user).model_dump() if c.published or grant else None
    else:
        result["candidate"] = None
    return result


@router.get("/invitations", response_model=list[InvitationView])
def invitations(user: Current, db: Db) -> list[dict]:
    key = Invitation.candidate_id if user.role == "candidate" else Invitation.employer_id
    rows = db.exec(select(Invitation).where(key == user.id).order_by(Invitation.created_at.desc())).all()
    return [invite_view(i, user, db) for i in rows]


@router.post("/invitations", status_code=201, response_model=InvitationView)
def invite(data: InviteInput, user: EmployerUser, db: Db) -> dict:
    ensure_company(db, user)
    c = db.get(Candidate, data.candidate_id)
    if not c or not c.published or not c.grade:
        raise HTTPException(404, "Опубликованный подтверждённый профиль не найден.")
    if data.job_id:
        job = db.get(Job, data.job_id)
        if not job or job.employer_id != user.id or not job.active:
            raise HTTPException(404, "Вакансия не найдена.")
    db.exec(update(Invitation).where(Invitation.employer_id == user.id,
        Invitation.candidate_id == data.candidate_id, Invitation.status.in_(["sent", "viewed"]),
        Invitation.expires_at <= now()).values(status="expired", updated_at=now()))
    existing = db.exec(select(Invitation).where(Invitation.employer_id == user.id,
        Invitation.candidate_id == data.candidate_id, Invitation.status.in_(["sent", "viewed"]),
        Invitation.expires_at > now())).first()
    if existing:
        raise HTTPException(409, "У кандидата уже есть активное приглашение от вашей компании.")
    i = Invitation(**data.model_dump(), employer_id=user.id, expires_at=now() + 14 * 86400)
    db.add(i)
    db.add(AuditEvent(actor_id=user.id, action="invitation_sent", entity_id=i.id))
    db.commit()
    return invite_view(i, user, db)


@router.post("/invitations/{invite_id}/action", response_model=InvitationView)
def transition(invite_id: str, data: InviteAction, user: Current, db: Db) -> dict:
    i = db.exec(select(Invitation).where(Invitation.id == invite_id).with_for_update()).first()
    if not i or user.id not in (i.candidate_id, i.employer_id):
        raise HTTPException(404, "Приглашение не найдено.")
    candidate = user.id == i.candidate_id
    if data.action == "revoke_contacts":
        if not candidate or i.status != "accepted": raise HTTPException(409, "Отзыв доступен после принятия приглашения.")
        i.contact_revoked = True
        i.updated_at = now()
        db.add(i)
    else:
        targets = {"view": "viewed", "accept": "accepted", "decline": "declined", "withdraw": "withdrawn"}
        if (data.action == "withdraw") == candidate:
            raise HTTPException(403, "Это действие доступно другой стороне приглашения.")
        target = targets[data.action]
        if i.status == target or (data.action == "view" and i.status in ("accepted", "declined")):
            return invite_view(i, user, db)
        if i.status not in ("sent", "viewed") or i.expires_at <= now():
            raise HTTPException(409, "Приглашение уже закрыто или срок ответа истёк.")
        changed = db.exec(update(Invitation).where(Invitation.id == i.id,
            Invitation.status == i.status, Invitation.expires_at > now()).values(status=target, updated_at=now()))
        if changed.rowcount != 1: raise HTTPException(409, "Статус изменился. Обновите приглашение.")
    db.add(AuditEvent(actor_id=user.id, action=f"invitation_{data.action}", entity_id=i.id))
    db.commit()
    db.refresh(i)
    return invite_view(i, user, db)


def job_view(job, user, db):
    db.refresh(job)
    result = job.model_dump()
    result["company"] = company_public(db, job.employer_id)
    application = db.exec(select(Application).where(Application.job_id == job.id, Application.candidate_id == user.id)).first() if user.role == "candidate" else None
    result["application_id"] = application.id if application else None
    result["application_status"] = application.status if application else None
    return result


@router.get("/jobs", response_model=list[JobView])
def jobs(user: Current, db: Db) -> list[dict]:
    stmt = select(Job).where(Job.employer_id == user.id) if user.role == "employer" else select(Job).where(Job.active == True)
    return [job_view(j, user, db) for j in db.exec(stmt.order_by(Job.created_at.desc())).all()]


@router.post("/jobs", status_code=201, response_model=JobView)
def create_job(data: JobInput, user: EmployerUser, db: Db) -> dict:
    ensure_company(db, user)
    j = Job(**data.model_dump(), employer_id=user.id)
    db.add(j)
    db.commit()
    return job_view(j, user, db)


@router.put("/jobs/{job_id}", response_model=JobView)
def edit_job(job_id: str, data: JobInput, user: EmployerUser, db: Db) -> dict:
    j = db.get(Job, job_id)
    if not j or j.employer_id != user.id: raise HTTPException(404, "Вакансия не найдена.")
    for k, v in data.model_dump().items(): setattr(j, k, v)
    db.add(j)
    db.commit()
    return job_view(j, user, db)


@router.post("/jobs/{job_id}/apply", status_code=201, response_model=ApplicationView)
def apply(job_id: str, data: ApplicationInput, user: CandidateUser, db: Db) -> dict:
    if not data.contact_consent: raise HTTPException(422, "Для отклика подтвердите передачу контактов этой компании.")
    j = db.get(Job, job_id)
    if not j or not j.active: raise HTTPException(404, "Вакансия недоступна.")
    a = db.exec(select(Application).where(Application.job_id == job_id, Application.candidate_id == user.id)).first()
    if a:
        if a.status == "withdrawn":
            a.status, a.message, a.updated_at = "sent", data.message, now()
            db.add(a)
            db.commit()
            db.refresh(a)
        return a.model_dump()
    a = Application(job_id=job_id, candidate_id=user.id, employer_id=j.employer_id, message=data.message)
    db.add(a)
    db.add(AuditEvent(actor_id=user.id, action="application_contact_consent", entity_id=a.id))
    db.commit()
    db.refresh(a)
    return a.model_dump()


@router.get("/applications", response_model=list[ApplicationView])
def applications(user: Current, db: Db) -> list[dict]:
    key = Application.candidate_id if user.role == "candidate" else Application.employer_id
    result = []
    for a in db.exec(select(Application).where(key == user.id).order_by(Application.created_at.desc())).all():
        item = a.model_dump()
        item["job"] = job_view(db.get(Job, a.job_id), user, db)
        item["candidate"] = candidate_view(db, db.get(Candidate, a.candidate_id), user).model_dump() if user.role == "employer" and a.status != "withdrawn" else None
        result.append(item)
    return result


@router.post("/applications/{application_id}/status", response_model=ApplicationView)
def application_status(application_id: str, data: ApplicationAction, user: Current, db: Db) -> dict:
    a = db.exec(select(Application).where(Application.id == application_id).with_for_update()).first()
    if not a or user.id not in (a.candidate_id, a.employer_id): raise HTTPException(404, "Отклик не найден.")
    if (data.status == "withdrawn") != (user.id == a.candidate_id): raise HTTPException(403, "Это действие недоступно вашей роли.")
    if a.status == data.status: return a.model_dump()
    if a.status in ("withdrawn", "declined") and data.status != "withdrawn": raise HTTPException(409, "Отклик уже закрыт.")
    changed = db.exec(update(Application).where(Application.id == a.id, Application.status == a.status)
        .values(status=data.status, updated_at=now()))
    if changed.rowcount != 1: raise HTTPException(409, "Отклик изменён. Обновите страницу.")
    db.add(AuditEvent(actor_id=user.id, action=f"application_{data.status}", entity_id=a.id))
    db.commit()
    db.refresh(a)
    return a.model_dump()
