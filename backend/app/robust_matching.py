"""Application adapter: observed evidence only, not uncalibrated ability bounds.

robust_engine.py is an unchanged copy of the independently tested, frozen
research/robust_topk/core.py. Research benchmarks never read application data.
"""
import numpy as np
from fastapi import HTTPException

from app.models import now
from app.schemas import RobustReport, PriorityQuestion
from app.robust_engine import WeightRegion, solve, topk, worst_case, question_options

FAMILIES = {"backend": ["Python", "SQL", "API"],
            "data": ["SQL", "Аналитика", "Качество данных"]}


def prepare_robust(query):
    if not query.skills:
        query.skills = FAMILIES.get(query.specialization, []).copy()
    if not 1 <= len(query.skills) <= 3:
        raise HTTPException(422, "Для устойчивой подборки выберите специализацию или 1–3 проверяемых навыка.")
    if any(s not in {s for values in FAMILIES.values() for s in values} for s in query.skills):
        raise HTTPException(422, "Устойчивый подбор поддерживает только навыки действующего банка проверок.")
    pairs = []
    for a, b in query.ranking.priorities:
        if a == b or a not in query.skills or b not in query.skills:
            raise HTTPException(422, "Приоритет сравнивает два разных навыка из текущего запроса.")
        pairs.append((query.skills.index(a), query.skills.index(b)))
    region = WeightRegion(len(query.skills), priorities=tuple(pairs))
    return region


def select_robust(ranked, query, region):
    if len(ranked) > 100:
        raise HTTPException(422, "Уточните фильтры: устойчивый подбор рассчитан максимум на 100 подходящих профилей.")
    if not ranked:
        return ranked, None
    # Stable mapping across the root and question branches; never private fields.
    ranked = sorted(ranked, key=lambda row: row[0].user_id)
    timestamp = now()
    values = []
    for candidate, _ in ranked:
        achievements = [a for a in candidate.achievements if a.get("verified") and not a.get("revoked")]
        fsp = min(len(achievements), 2) / 2
        features = []
        for name in query.skills:
            skill = candidate.skills[name]
            coverage = min(skill.get("families", 0), 2) / 2
            freshness = max(0., min(1., 1 - (timestamp - skill["checked_at"]) / (180 * 86400)))
            features.append(.8 * skill["score"] / 100 + .1 * coverage + .05 * freshness + .05 * fsp)
        values.append(features)
    values = np.asarray(values)
    nominal = values.mean(axis=1)  # anchored equally weighted baseline, never reset after a question
    k, vertices = min(5, len(ranked)), region.vertices()
    base = topk(nominal, k)
    baseline = worst_case(values, values, vertices, base)
    selection = solve(values, values, vertices, k, time_limit=.6,
                      guard_scores=nominal, max_nominal_loss=.01*k)
    options = question_options(values, values, region, k, time_limit=.25,
                               guard_scores=nominal, max_nominal_loss=.01*k)
    question = None
    if options and options[0]["worst_after"] < selection.witness.regret - .001*k:
        option = options[0]
        a, b = option["pair"]
        question = PriorityQuestion(skills=(query.skills[a], query.skills[b]),
                                    worst_after_points=100 * option["worst_after"] / k,
                                    exact_choice=all(o["optimal"] and all(s.absolute_gap <= 1e-7 for s in o["branches"]) for o in options))
    # Ordering is nominal; only the composition has the robust guarantee.
    chosen = sorted(selection.selected, key=lambda i: (-nominal[i], ranked[i][0].user_id))
    result = []
    for i in chosen:
        c, (_, reasons, limitations) = ranked[i]
        result.append((c, (round(100*nominal[i], 1), reasons +
                           ["В составе пятёрки с ограничением потери исходной оценки: не более 1 балла в среднем."], limitations)))
    status = "bounded" if selection.status == "optimal" and selection.absolute_gap > 1e-7 else selection.status
    report = RobustReport(status=status, considered=len(ranked), selected_count=k,
                          worst_loss_points=100*selection.witness.regret/k,
                          baseline_worst_loss_points=100*baseline.regret/k,
                          nominal_loss_points=max(0., 100*(nominal[list(base)].sum()-nominal[chosen].sum())/k),
                          optimization_gap_points=100*selection.absolute_gap/k,
                          worst_weights=dict(zip(query.skills, selection.witness.weights)),
                          challenger_ids=[ranked[i][0].user_id for i in selection.witness.challenger],
                          question=question)
    return result, report
