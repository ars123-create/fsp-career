from app.responses import Deleted, NeedView, ParsedSearch, RestoredNeed
from fastapi import APIRouter, HTTPException
from sqlmodel import select
from app.core.security import EmployerUser, Db
from app.models import Candidate, Need, AuditEvent
from app.schemas import SearchInput, SearchResult
from app.matching import normalize, evaluate
from app.visibility import candidate_view
from app.robust_matching import prepare_robust, select_robust
from app.preference_matching import prepare_preference, select_preference
from app.weighted_matching import prepare_weighted, select_weighted
from app.mixed_matching import select_mixed

router = APIRouter(tags=["Подбор"])


def find(data, user, db):
    q, notices = normalize(data)
    region = prepare_robust(q) if q.ranking.mode == "robust" else None
    if q.ranking.mode in ("preference", "mixed"):
        prepare_preference(q)
    if q.ranking.mode == "weighted":
        prepare_weighted(q)
    stmt = select(Candidate).where(Candidate.published == True, Candidate.grade != None)
    for key in ("specialization", "grade", "work_format"):
        if getattr(q, key): stmt = stmt.where(getattr(Candidate, key) == getattr(q, key))
    candidates = db.exec(stmt).all()
    ranked = []
    for c in candidates:
        result = evaluate(c, q)
        if result:
            ranked.append((c, result))
    ranked.sort(key=lambda row: (-row[1][0], row[0].user_id))
    total, report, preference, weighted = len(ranked), None, None, None
    if region is not None:
        ranked, report = select_robust(ranked, q, region)
        notices.append("Режим устойчивой пятёрки: граница относится к наблюдаемым результатам и указанным приоритетам. Это не прогноз качества найма.")
    if q.ranking.mode == "preference":
        ranked, preference = select_preference(ranked, q)
        notices.append("Подбор с уточнениями учитывает результаты проверок и ваши ответы о приоритетах. При равной оценке навыков преимущество даёт достижение ФСП с подтверждённым источником. Это экспериментальный режим; качество найма на людях пока не проверено.")
    if q.ranking.mode == "weighted":
        ranked, weighted = select_weighted(ranked, q)
        notices.append("Баллы проверок учитываются с заданными вами приоритетами. Нулевой вес не отменяет обязательный навык. При равной оценке преимущество даёт достижение ФСП с подтверждённым источником. Качество найма на людях пока не проверено.")
    if q.ranking.mode == "mixed":
        ranked, preference = select_mixed(ranked, q)
        notices.append("Умный подбор учитывает сочетания навыков. До трёх уточнений; ответы не меняют грейд или доступ к контактам. Проверка качества выполнена на синтетических данных.")
    views = []
    for c, (points, reasons, limitations) in ranked[:100]:
        view = candidate_view(db, c, user)
        view.match_score, view.reasons, view.limitations = points, reasons, limitations
        views.append(view)
    algorithm = "direct-weights-v1" if weighted is not None else "decision-evoi-v3" if preference is not None else "guarded-minimax-v2" if region is not None else "evidence-v1"
    if q.ranking.mode == "mixed":
        algorithm = "mixed-kg-v1"
    return SearchResult(candidates=views, total=total, normalized=q, notices=notices,
                        algorithm=algorithm, robust=report, preference=preference, weighted=weighted)


@router.post("/search", response_model=SearchResult)
def search(data: SearchInput, user: EmployerUser, db: Db):
    return find(data, user, db)


@router.post("/search/parse", response_model=ParsedSearch)
def parse(data: SearchInput, user: EmployerUser) -> dict:
    normalized, notices = normalize(data)
    return {"normalized": normalized, "notices": notices}


@router.get("/needs", response_model=list[NeedView])
def saved(user: EmployerUser, db: Db) -> list[dict]:
    return [n.model_dump(exclude={"employer_id"}) for n in db.exec(select(Need).where(Need.employer_id == user.id).order_by(Need.created_at.desc())).all()]


@router.post("/needs", status_code=201, response_model=NeedView)
def save(data: SearchInput, user: EmployerUser, db: Db) -> dict:
    result = find(data, user, db)
    values = result.normalized.model_dump(exclude={"query"})
    n = Need(**values, description=data.query, employer_id=user.id, saved_ids=[c.id for c in result.candidates])
    db.add(n)
    db.commit()
    db.refresh(n)
    return n.model_dump(exclude={"employer_id"})


@router.get("/needs/{need_id}", response_model=RestoredNeed)
def restore(need_id: str, user: EmployerUser, db: Db) -> dict:
    n = db.get(Need, need_id)
    if not n or n.employer_id != user.id: raise HTTPException(404, "Подборка не найдена.")
    q = SearchInput(query=n.description, **n.model_dump(include={"title", "specialization", "grade", "skills", "work_format", "salary_min", "salary_max", "fsp_only", "ranking"}))
    previous = []
    for candidate_id in n.saved_ids:
        c = db.get(Candidate, candidate_id)
        if c and c.published:
            previous.append(candidate_view(db, c, user))
    return {"need": n.model_dump(exclude={"employer_id"}), "previous": previous, "current": find(q, user, db)}


@router.delete("/needs/{need_id}", response_model=Deleted)
def remove(need_id: str, user: EmployerUser, db: Db) -> dict:
    n = db.get(Need, need_id)
    if not n or n.employer_id != user.id: raise HTTPException(404, "Подборка не найдена.")
    db.delete(n)
    db.commit()
    return {"deleted": True}
