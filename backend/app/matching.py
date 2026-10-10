"""Auditable rules baseline: hard constraints, then observed evidence.

This is an application baseline, not the research VOI model and not a calibrated
estimate of hiring success. All score terms are exposed in the explanation.
"""
import re
from app.models import now
from app.schemas import SearchInput

ALIASES = {
    "Python": r"\bpython\b|питон|пайтон",
    "SQL": r"\bsql\b|postgres(?:ql)?|mysql|базы? данных",
    "API": r"\bapi\b|\brest(?:ful)?\b|апи",
    "Аналитика": r"аналитик|analytics|статистик",
    "Качество данных": r"качество данных|data quality|очистк[аи] данных",
}


def extract_skills(text):
    return [name for name, pattern in ALIASES.items() if re.search(pattern, text, re.I)]


def normalize(data: SearchInput):
    d = data.model_copy(deep=True)
    q = d.query.lower()
    notices = []
    if not d.specialization:
        if re.search(r"backend|бэкенд|бэкендер|бэк|бекенд|python", q):
            d.specialization = "backend"
        elif re.search(r"аналитик|data analyst|аналитика", q):
            d.specialization = "data"
    if not d.grade:
        if re.search(r"middle|мид[дл]", q): d.grade = "Middle"
        elif re.search(r"junior|джун", q): d.grade = "Junior"
    if not d.work_format:
        if re.search(r"удал[её]н|remote", q): d.work_format = "remote"
        elif "гибрид" in q: d.work_format = "hybrid"
        elif "офис" in q: d.work_format = "office"
    if not d.skills:
        d.skills = extract_skills(q)
    d.skills = list(dict.fromkeys(next(iter(extract_skills(s)), s.strip()) for s in d.skills if s.strip()))
    if not d.salary_max:
        salary = re.search(r"(\d{2,6})\s*[-–—]\s*(\d{2,6})\s*(тыс|к\b|k\b|руб|rub|₽)?", q)
        if salary:
            lo, hi = int(salary[1]), int(salary[2])
            factor = 1000 if salary[3] in ("тыс", "к", "k") or hi < 1000 else 1
            if lo <= hi:
                d.salary_min, d.salary_max = lo * factor, hi * factor
    if d.query:
        notices.append("Распознаны только известные специализации и навыки. Проверьте фильтры перед поиском; свободный текст не оценивает человека.")
    unknown = [s for s in d.skills if s not in ALIASES]
    if unknown:
        notices.append("Для навыков «" + ", ".join(unknown) + "» в текущем банке нет подтверждения. Уберите их из обязательных фильтров или расширьте банк заданий.")
    return d, notices


def evaluate(candidate, query):
    if not candidate.published or not candidate.grade:
        return None
    for key in ("specialization", "grade", "work_format"):
        expected = getattr(query, key)
        if expected and expected != getattr(candidate, key): return None
    if query.salary_max and candidate.salary_min > query.salary_max: return None
    if query.salary_min and candidate.salary_max and candidate.salary_max < query.salary_min: return None
    achievements = [a for a in candidate.achievements if a.get("verified") and not a.get("revoked")]
    if query.fsp_only and not achievements: return None
    required = query.skills or list(candidate.skills)
    if any(s not in candidate.skills or candidate.skills[s]["score"] < 50 for s in query.skills): return None
    values = [candidate.skills[s] for s in required if s in candidate.skills]
    score = sum(v["score"] for v in values) / len(values) if values else 0
    coverage = sum(min(v.get("families", 0), 2) / 2 for v in values) / len(values) if values else 0
    freshness = sum(max(0, 1 - (now() - v["checked_at"]) / (180 * 86400)) for v in values) / len(values) if values else 0
    # FSP contributes at most 5 points and never changes the grade.
    fsp = min(len(achievements), 2) / 2
    points = round(.80 * score + 10 * coverage + 5 * freshness + 5 * fsp, 1)
    reasons = [f"{s}: {candidate.skills[s]['score']:.0f}% ответов · {candidate.skills[s]['families']} семейств" for s in required if s in candidate.skills]
    if query.work_format: reasons.append("Формат работы совпадает")
    if query.salary_max and candidate.salary_max: reasons.append("Зарплатные диапазоны пересекаются")
    if achievements: reasons.append(f"Достижений с подтверждённым источником: {len(achievements)}")
    limitations = ["Баллы ранжирования не являются вероятностью успешного найма."]
    if any(now() - v["checked_at"] > 90 * 86400 for v in values): limitations.append("Есть проверки старше 90 дней.")
    if not candidate.salary_max: limitations.append("Зарплатные ожидания пока не указаны.")
    return points, reasons, limitations
