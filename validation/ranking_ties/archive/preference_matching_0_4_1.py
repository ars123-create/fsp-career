"""Preference adapter over observed skill scores; no candidate identity in questions.

The tested finite-belief engine is copied byte-for-byte from frozen research.
Answers describe the employer's own hypothetical trade-offs. They are ordinary
bounded input, NOT signed evidence of a candidate's skills or grade. They cannot
alter eligibility, permissions or another employer's saved needs.
"""
from itertools import combinations

from fastapi import HTTPException
import numpy as np

from app.preference_engine import Belief, questions, response_probability, simplex_particles, topk
from app.preference_acceleration import best_linear_question
from app.schemas import PreferenceQuestion, PreferenceReport

SKILLS = {"backend": ["Python", "SQL", "API"],
          "data": ["SQL", "Аналитика", "Качество данных"]}
MAX_COMPARISONS = 3


def pair_key(left, right):
    """Identity-free, orientation-independent pair of hypothetical score vectors."""
    return tuple(sorted((tuple(sorted(left.items())), tuple(sorted(right.items())))))


def prepare_preference(query):
    if not query.skills:
        query.skills = SKILLS.get(query.specialization, []).copy()
    known = {name for names in SKILLS.values() for name in names}
    if not 1 <= len(query.skills) <= 3 or any(s not in known for s in query.skills):
        raise HTTPException(422, "Для подбора с уточнениями выберите специализацию или 1–3 навыка действующего банка.")
    seen = set()
    for answer in query.ranking.comparisons:
        if set(answer.left) != set(query.skills) or set(answer.right) != set(query.skills):
            raise HTTPException(422, "Набор навыков изменился. Сбросьте ответы на уточнения и повторите поиск.")
        key = pair_key(answer.left, answer.right)
        if answer.left == answer.right or key in seen:
            raise HTTPException(422, "Сравнение должно содержать разные варианты и не повторять предыдущие вопросы.")
        seen.add(key)


def posterior(query):
    weights = simplex_particles(len(query.skills))
    p = np.full(len(weights), 1/len(weights))
    for answer in query.ranking.comparisons:
        if answer.choice == "skip":
            continue
        difference = np.array([answer.left[s]-answer.right[s] for s in query.skills])/100
        likelihood = response_probability(weights @ difference)
        if answer.choice == "right":
            likelihood = 1-likelihood
        p *= likelihood
        p /= p.sum()
    return weights, p


def select_preference(ranked, query):
    if len(ranked) > 100:
        raise HTTPException(422, "Уточните фильтры: подбор с уточнениями рассчитан максимум на 100 подходящих профилей.")
    ranked = sorted(ranked, key=lambda row: row[0].user_id)
    weights, p = posterior(query)
    mean_weights = p @ weights
    answered = sum(a.choice != "skip" for a in query.ranking.comparisons)
    if not answered:
        # QMC prior is symmetric analytically; avoid floating-point tie drift.
        mean_weights = np.full(len(query.skills), 1/len(query.skills))
    report = PreferenceReport(status="empty", considered=len(ranked), selected_count=0,
        answered=answered, skipped=len(query.ranking.comparisons)-answered,
        remaining=MAX_COMPARISONS-len(query.ranking.comparisons),
        weights=dict(zip(query.skills, map(float, mean_weights))), changed_count=0)
    if not ranked:
        return [], report
    # Only raw observed scores: no invented ability intervals or arbitrary bonus.
    values = np.array([[c.skills[s]["score"]/100 for s in query.skills] for c, _ in ranked])
    k = min(5, len(ranked))
    # Linear expectation equals values @ E[weights]. Round below displayed
    # precision so summation noise cannot replace tied profiles after a skip.
    scores = np.round(values @ mean_weights, 12)
    selected = topk(scores, k)
    baseline = topk(np.round(values.mean(axis=1), 12), k)
    report.selected_count = k
    report.changed_count = len(set(selected)-set(baseline))
    report.status = "complete" if not report.remaining else "not_needed"
    if report.remaining and len(query.skills) > 1 and len(ranked) > k:
        vectors = [{s: float(c.skills[s]["score"]) for s in query.skills} for c, _ in ranked]
        seen = {pair_key(a.left, a.right) for a in query.ranking.comparisons}
        excluded, representatives = [], set()
        for a, b in combinations(range(len(ranked)), 2):
            key = pair_key(vectors[a], vectors[b])
            difference = values[a]-values[b]
            tradeoff = np.any(difference > 0) and np.any(difference < 0)
            if not tradeoff or key in seen or key in representatives:
                excluded.append((a, b))
            else:
                representatives.add(key)
        # Keep the original single-block path for coarse score grids. For a
        # larger pool, factor the same linear EVOI through its three features.
        pair_count = len(ranked)*(len(ranked)-1)//2-len(excluded)
        if pair_count <= 128:
            options = questions(Belief(weights @ values.T, p), k, asked=excluded)
            best = max(options, key=lambda q: q.expected_gain) if options else None
        else:
            best = best_linear_question(values, weights, p, k, asked=excluded, min_gain=.0005*k)
        if best is not None:
            # UI stopping threshold: <0.05 model points per profile is not worth
            # another question. The research benchmark forced the full budget.
            if best.expected_gain > .0005*k:
                a, b = best.pair
                report.status = "question"
                report.question = PreferenceQuestion(left=vectors[a], right=vectors[b],
                                                     expected_gain_points=100*best.expected_gain/k)
    reasons = "Приоритеты запроса: " + ", ".join(f"{s} {mean_weights[i]*100:.0f}%" for i, s in enumerate(query.skills)) + "."
    result = []
    for i in selected:
        candidate, (_, factual, limitations) = ranked[i]
        explanation = "Средний результат с равным весом навыков." if not answered else reasons
        result.append((candidate, (round(float(scores[i])*100, 1),
            factual + [explanation], limitations + ["Уточнения меняют приоритеты подбора, а не подтверждённый грейд."])))
    return result, report
