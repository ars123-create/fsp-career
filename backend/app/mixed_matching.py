"""HTTP product boundary for m6; observed skills only, no identity in the model."""
from threading import BoundedSemaphore

import numpy as np
from fastapi import HTTPException

from app.mixed_engine.adapter import recommend
from app.ranking_ties import has_confirmed_achievement
from app.schemas import PreferenceQuestion, PreferenceReport

# Bound temporary numeric memory when several employers search concurrently.
_calculation_slots = BoundedSemaphore(2)
SKILL_ORDER = ("Python", "SQL", "API", "Аналитика", "Качество данных")


def select_mixed(ranked, query):
    if len(ranked) > 100:
        raise HTTPException(422, "Уточните фильтры: умный подбор рассчитан максимум на 100 подходящих профилей.")
    query.skills.sort(key=SKILL_ORDER.index)
    # Research uses feature order before index for utility ties. Achievement can
    # resolve identical vectors without replacing a strictly stronger profile.
    ranked = sorted(ranked, key=lambda row: (-has_confirmed_achievement(row[0]), row[0].user_id))
    answers = query.ranking.comparisons
    answered = sum(a.choice != "skip" for a in answers)
    report = PreferenceReport(status="empty", considered=len(ranked), selected_count=0,
        answered=answered, skipped=len(answers)-answered, remaining=3-len(answers),
        weights={}, changed_count=0, engine="mixed-kg-v1",
        model="Учитываем сочетания навыков и ваши ответы. Оценка подбора — условный балл, не вероятность найма.")
    if not ranked:
        return [], report
    values = np.array([[c.skills[s]["score"]/100 for s in query.skills] for c, _ in ranked])
    history = [(tuple(a.left[s]/100 for s in query.skills),
                tuple(a.right[s]/100 for s in query.skills), a.choice) for a in answers]
    try:
        with _calculation_slots:
            result = recommend(values, history)
    except ValueError as exc:
        raise HTTPException(422, "Некорректное сравнение навыков. Сбросьте уточнения и повторите поиск.") from exc
    report.selected_count = len(result.selected)
    report.changed_count = len(set(result.selected)-set(result.baseline))
    report.status = "complete" if not report.remaining else "not_needed"
    if result.question is not None:
        a, b = result.question
        report.status = "question"
        report.question = PreferenceQuestion(
            left={s: float(values[a, j]*100) for j, s in enumerate(query.skills)},
            right={s: float(values[b, j]*100) for j, s in enumerate(query.skills)},
            expected_gain_points=100*result.gain/report.selected_count)
    output = []
    for i in result.selected:
        candidate, (_, factual, limitations) = ranked[i]
        explanation = ("Сочетание навыков соответствует вашим уточнениям." if answered
                       else "Учитываем и сильные стороны, и баланс проверенных навыков.")
        output.append((candidate, (round(float(result.scores[i])*100, 1),
            factual+[explanation], limitations+[report.model])))
    return output, report
