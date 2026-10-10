"""One allowlist and authorization path for search, profile, JSON export and PDF."""
from fastapi import HTTPException
from sqlmodel import select
from app.models import Application, Candidate, Evidence, Invitation, User
from app.schemas import CandidateView, Contacts
from app.assessment_bank import SPECIALIZATIONS


def contact_allowed(db, employer_id: str, candidate_id: str) -> bool:
    accepted = db.exec(select(Invitation.id).where(Invitation.employer_id == employer_id,
        Invitation.candidate_id == candidate_id, Invitation.status == "accepted",
        Invitation.contact_revoked == False)).first()
    applied = db.exec(select(Application.id).where(Application.employer_id == employer_id,
        Application.candidate_id == candidate_id, Application.status != "withdrawn")).first()
    return bool(accepted or applied)


def candidate_view(db, candidate: Candidate, viewer: User) -> CandidateView:
    owner = viewer.id == candidate.user_id
    if not owner and viewer.role != "employer":
        raise HTTPException(403, "Профили других кандидатов недоступны.")
    grant = owner or contact_allowed(db, viewer.id, candidate.user_id)
    # A previous accepted invitation/application remains visible if publication is withdrawn.
    if not owner and not candidate.published and not grant:
        raise HTTPException(404, "Профиль недоступен.")
    user = db.get(User, candidate.user_id)
    evidence = db.exec(select(Evidence).where(Evidence.candidate_id == candidate.user_id)
        .order_by(Evidence.created_at.desc()).limit(24)).all()
    category = f"{SPECIALIZATIONS.get(candidate.specialization, '')} · {candidate.grade}" if candidate.grade else "Категория ещё не подтверждена"
    return CandidateView(id=user.id, name=user.full_name if grant else f"Кандидат {user.id[:6].upper()}",
        headline=candidate.headline, about=candidate.about, city=candidate.city,
        grade=candidate.grade, specialization=candidate.specialization, category=category,
        work_format=candidate.work_format, employment=candidate.employment,
        salary_min=candidate.salary_min, salary_max=candidate.salary_max,
        experience_years=candidate.experience_years, stack=candidate.stack, skills=candidate.skills,
        achievements=[a for a in candidate.achievements if a.get("verified") and not a.get("revoked")],
        evidence=[{"skill": e.skill, "family": e.family, "score": e.score,
                   "checked_at": e.created_at, "is_demo": e.is_demo} for e in evidence],
        contacts=Contacts(full_name=user.full_name, email=user.email, phone=candidate.phone,
                          telegram=candidate.telegram) if grant else None, is_demo=user.is_demo)


def load_visible(db, candidate_id: str, viewer: User) -> CandidateView:
    candidate = db.get(Candidate, candidate_id)
    if not candidate:
        raise HTTPException(404, "Профиль не найден.")
    return candidate_view(db, candidate, viewer)
