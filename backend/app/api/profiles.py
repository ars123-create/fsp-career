from app.responses import CompanyOwn, FspStatus, OwnProfile, PdfImport
from io import BytesIO
import re
from fastapi import APIRouter, HTTPException, UploadFile, Response
from pypdf import PdfReader
from sqlmodel import select
from app.core.security import CandidateUser, EmployerUser, Current, Db
from app.core.config import settings
from app.models import Candidate, Company, AuditEvent, now
from app.schemas import CandidateUpdate, CompanyUpdate, FspLink, CandidateView
from app.visibility import load_visible
from app.pdf_profile import profile_pdf

router = APIRouter(tags=["Профили"])


@router.get("/profile", response_model=OwnProfile)
def own_profile(user: CandidateUser, db: Db) -> dict:
    c = db.get(Candidate, user.id)
    return {**c.model_dump(), "full_name": user.full_name, "email": user.email,
            "processing_consent": bool(c.processing_consent_at),
            "publication_consent": bool(c.publication_consent_at),
            "next_grade_change_at": c.grade_changed_at + settings.grade_cooldown_days * 86400 if c.grade else 0,
            "fsp_status": "demo" if any(a.get("source") == "Демонстрационный реестр ФСП" for a in c.achievements) else ("pending" if c.fsp_id else "not_linked")}


@router.put("/profile", response_model=OwnProfile)
def update_profile(data: CandidateUpdate, user: CandidateUser, db: Db) -> dict:
    c = db.get(Candidate, user.id)
    user.full_name = data.full_name
    values = data.model_dump(exclude={"full_name", "processing_consent", "publication_consent"})
    for k, v in values.items():
        setattr(c, k, v)
    c.processing_consent_at = (c.processing_consent_at or now()) if data.processing_consent else 0
    c.publication_consent_at = (c.publication_consent_at or now()) if data.publication_consent else 0
    if not c.processing_consent_at or not c.publication_consent_at:
        c.published = False
    c.updated_at = now()
    db.add(c)
    db.add(user)
    db.add(AuditEvent(actor_id=user.id, action="profile_updated", entity_id=user.id))
    db.commit()
    return own_profile(user, db)


@router.post("/profile/fsp", response_model=FspStatus)
def link_fsp(data: FspLink, user: CandidateUser, db: Db) -> dict:
    c = db.get(Candidate, user.id)
    # A typed ID is only a connection request, never evidence of ownership.
    c.fsp_id = data.fsp_id
    c.achievements = []
    db.add(c)
    db.commit()
    return {"status": "pending" if c.fsp_id else "not_linked", "achievements": [],
            "message": "Заявка сохранена. Подтверждение появится после подключения официального реестра; введённый ID не подтверждает достижения." if c.fsp_id else "Связь удалена."}


@router.get("/profile/pdf", response_class=Response, responses={200: {"content": {"application/pdf": {"schema": {"type": "string", "format": "binary"}}}, "description": "PDF профиля с кириллицей"}})
def own_pdf(user: CandidateUser, db: Db):
    return pdf_response(load_visible(db, user.id, user))


def pdf_response(profile):
    return Response(profile_pdf(profile), media_type="application/pdf", headers={
        "Content-Disposition": f'attachment; filename="fsp-profile-{profile.id[:6]}.pdf"',
        "Cache-Control": "private, no-store"})


@router.post("/profile/import-pdf", response_model=PdfImport)
async def import_pdf(user: CandidateUser, db: Db, file: UploadFile) -> dict:
    content = await file.read(5 * 1024 * 1024 + 1)
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(413, "Размер PDF должен быть до 5 МБ.")
    if not content.startswith(b"%PDF-"):
        raise HTTPException(422, "Загрузите PDF с текстовым слоем.")
    try:
        reader = PdfReader(BytesIO(content), strict=True)
        if reader.is_encrypted or len(reader.pages) > 15:
            raise ValueError()
        # Bound uncompressed streams as well as the upload size.
        pieces = []
        for page in reader.pages:
            stream = page.get_contents()
            if stream is not None and len(stream.get_data()) > 2 * 1024 * 1024:
                raise ValueError()
            pieces.append((page.extract_text() or "")[:16000])
        text = "\n".join(pieces)[:40000]
    except Exception:
        raise HTTPException(422, "Не удалось прочитать PDF. Уберите пароль, сократите до 15 страниц или заполните форму вручную.")
    if len(text.strip()) < 30:
        raise HTTPException(422, "В PDF нет читаемого текста. Скан можно перенести в форму вручную.")
    email = re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text)
    phone = re.search(r"\+?\d[\d ()-]{8,}\d", text)
    from app.matching import extract_skills
    # Preview only. Personal data never enters a public field without review.
    return {"preview": text[:6000], "suggested": {"phone": phone.group() if phone else "", "stack": extract_skills(text)},
            "found_email": email.group() if email else "", "message": "Проверьте найденные поля. Импорт не меняет грейд и не публикует профиль."}


@router.get("/company", response_model=CompanyOwn)
def own_company(user: EmployerUser, db: Db) -> dict:
    return db.get(Company, user.id).model_dump()


@router.put("/company", response_model=CompanyOwn)
def update_company(data: CompanyUpdate, user: EmployerUser, db: Db) -> dict:
    company = db.get(Company, user.id)
    for k, v in data.model_dump().items():
        setattr(company, k, v)
    db.add(company)
    db.commit()
    db.refresh(company)
    return company.model_dump()


@router.get("/candidates/{candidate_id}", response_model=CandidateView)
def candidate_detail(candidate_id: str, user: EmployerUser, db: Db):
    return load_visible(db, candidate_id, user)


@router.get("/candidates/{candidate_id}/pdf", response_class=Response, responses={200: {"content": {"application/pdf": {"schema": {"type": "string", "format": "binary"}}}, "description": "PDF с теми же правами доступа, что и профиль API"}})
def candidate_pdf(candidate_id: str, user: EmployerUser, db: Db):
    return pdf_response(load_visible(db, candidate_id, user))
