"""Direct employer preference input after the existing eligibility filters."""
import numpy as np
from fastapi import HTTPException

from app.preference_matching import SKILLS
from app.ranking_ties import evidence_tied_topk
from app.schemas import WeightedReport
from app.weighted_scoring import direct_scores


def prepare_weighted(query):
    if not query.skills:
        query.skills = SKILLS.get(query.specialization, []).copy()
    known = {skill for skills in SKILLS.values() for skill in skills}
    if not 1 <= len(query.skills) <= 3 or any(skill not in known for skill in query.skills):
        raise HTTPException(422, "Для приоритетов выберите специализацию или 1–3 навыка действующего банка.")
    if not query.ranking.weights:
        query.ranking.weights = {skill: 1. for skill in query.skills}
    if set(query.ranking.weights) != set(query.skills):
        raise HTTPException(422, "Набор навыков изменился. Задайте веса для каждого выбранного навыка или начните с равных приоритетов.")


def select_weighted(ranked, query):
    ranked = sorted(ranked, key=lambda row: row[0].user_id)
    candidates = [candidate for candidate, _ in ranked]
    values = np.array([[candidate.skills[skill]["score"] for skill in query.skills]
                       for candidate in candidates], dtype=float).reshape(len(candidates), len(query.skills))
    scores, weights = direct_scores(values, [query.ranking.weights[skill] for skill in query.skills])
    k = min(5, len(candidates))
    selected = evidence_tied_topk(scores, candidates, k)
    baseline_scores, _ = direct_scores(values, [1.] * len(query.skills))
    baseline = evidence_tied_topk(baseline_scores, candidates, k)
    report = WeightedReport(considered=len(ranked), selected_count=k,
                            weights=dict(zip(query.skills, map(float, weights))),
                            changed_count=len(set(selected) - set(baseline)))
    explanation = "Приоритеты заданы вами: " + ", ".join(
        f"{skill} {weights[i] * 100:.1f}%" for i, skill in enumerate(query.skills)) + "."
    output = []
    for i in selected:
        candidate, (_, reasons, limitations) = ranked[i]
        output.append((candidate, (round(float(scores[i]) * 100, 1), reasons + [explanation],
                                  limitations + ["Приоритеты не меняют подтверждённый грейд и обязательные условия."])))
    return output, report
