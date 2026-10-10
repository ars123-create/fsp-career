from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles
from sqlalchemy.exc import IntegrityError
from sqlmodel import SQLModel, Session, select
from app.core.config import settings, ROOT
from app.core.db import engine
from app.core.security import Current, Db
from app.models import Candidate, Invitation, Application, Job, Attempt, now
from app.responses import Health, Dashboard, ApiError
from app.api import auth, assessments, profiles, search, hiring


@asynccontextmanager
async def lifespan(app):
    # The initial schema is also captured by Alembic; startup remains simple for the demo.
    (ROOT / "data").mkdir(exist_ok=True)
    SQLModel.metadata.create_all(engine)
    if settings.demo_mode:
        from app.seed import seed_demo
        with Session(engine) as db: seed_demo(db)
    yield


app = FastAPI(title="ФСП / Карьера", version="0.5.1", lifespan=lifespan,
    description="Обратный найм по подтверждениям навыков. Демонстрационный банк не является калиброванной оценкой пригодности к работе.",
    responses={status: {"model": ApiError, "description": description} for status, description in {
        400: "Некорректный запрос или одноразовая ссылка", 401: "Нет действующей подтверждённой сессии",
        403: "Недостаточно прав", 404: "Объект отсутствует или недоступен", 409: "Конфликт состояния",
        413: "Файл превышает допустимый размер", 422: "Ошибка входных данных", 429: "Лимит запросов",
        503: "Почтовый сервис не настроен или недоступен"}.items()})
app.add_middleware(CORSMiddleware, allow_origins=[settings.frontend_url], allow_credentials=False,
                   allow_methods=["GET", "POST", "PUT", "DELETE"], allow_headers=["Authorization", "Content-Type"])


@app.middleware("http")
async def response_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Frame-Options"] = "DENY"
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.exception_handler(IntegrityError)
async def conflict(request, error):
    return JSONResponse({"detail": "Запись уже существует или была изменена. Обновите страницу."}, status_code=409)


@app.exception_handler(RequestValidationError)
async def invalid_request(request, error):
    # Do not echo passwords, tokens or uploaded data through validation diagnostics.
    details = [{key: issue[key] for key in ("loc", "msg", "type")} for issue in error.errors()]
    return JSONResponse({"detail": details}, status_code=422)


for router in (auth.router, assessments.router, profiles.router, search.router, hiring.router):
    app.include_router(router, prefix="/api")


@app.get("/api/health", tags=["Сервис"], response_model=Health)
def health() -> dict:
    with Session(engine) as db: db.exec(select(1)).first()
    return {"status": "ok", "version": "0.5.1", "demo_mode": settings.demo_mode,
            "mail_transport": "smtp" if settings.smtp_host else "local_demo" if settings.demo_mode else "unconfigured"}


@app.get("/api/dashboard", tags=["Кабинет"], response_model=Dashboard, response_model_exclude_none=True)
def dashboard(user: Current, db: Db) -> dict:
    key = Invitation.candidate_id if user.role == "candidate" else Invitation.employer_id
    invitations = db.exec(select(Invitation).where(key == user.id)).all()
    result = {"invitations": len(invitations), "accepted": sum(i.status == "accepted" for i in invitations),
        "pending": sum(i.status in ("sent", "viewed") and i.expires_at > now() for i in invitations)}
    if user.role == "candidate":
        c = db.get(Candidate, user.id)
        result.update(grade=c.grade, specialization=c.specialization, published=c.published,
            skills=c.skills, attempts=len(db.exec(select(Attempt.id).where(Attempt.candidate_id == user.id)).all()))
    else:
        result.update(candidates=len(db.exec(select(Candidate.user_id).where(Candidate.published == True, Candidate.grade != None)).all()),
            jobs=len(db.exec(select(Job.id).where(Job.employer_id == user.id, Job.active == True)).all()))
    return result


frontend = ROOT / "frontend" / "dist"
if frontend.exists():
    app.mount("/assets", StaticFiles(directory=frontend / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        if path.startswith("api/"): raise HTTPException(404)
        return FileResponse(frontend / "index.html")
