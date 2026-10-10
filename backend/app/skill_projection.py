"""Project a fixed family portfolio while keeping grade decisions separate."""
from sqlmodel import select

from app.assessment_bank import answer_correct
from app.evidence_portfolio import METHOD, refresh_portfolio, start_portfolio, summarize
from app.models import Attempt


def context_of(attempt):
    return {"specialization": attempt.specialization, "grade": attempt.grade,
            "rubric_version": attempt.rubric_version}


def observations_of(attempt):
    return [{"skill": q["skill"], "family": q["family"],
             "score": 100 if answer_correct(q, attempt.answers.get(q["id"])) else 0,
             "checked_at": attempt.finished_at} for q in attempt.questions]


def event_order(attempt):
    return attempt.finished_at, attempt.created_at


def restore_portfolio(db, candidate, attempt):
    """Reconstruct legacy history only when its ordering and context are unambiguous."""
    context = context_of(attempt)
    history = db.exec(select(Attempt).where(
        Attempt.candidate_id == candidate.user_id, Attempt.id != attempt.id,
        Attempt.specialization == attempt.specialization, Attempt.grade == attempt.grade,
        Attempt.rubric_version == attempt.rubric_version, Attempt.status == "completed",
        Attempt.finished_at != None,
    )).all()
    full = [row for row in history if row.kind == "grade" and row.result.get("passed")]
    if not full:
        return None
    last = max(event_order(row) for row in full)
    anchors = [row for row in full if event_order(row) == last]
    if len(anchors) != 1:
        return None
    base = anchors[0]
    later = sorted([row for row in history if row.kind == "challenge" and event_order(row) >= last],
                   key=event_order)
    ordering = [last, *[event_order(row) for row in later]]
    if len(ordering) != len(set(ordering)):
        return None
    # A malformed or incompatible old form is not silently converted.
    try:
        portfolio = start_portfolio(context, observations_of(base), base.id)
        for row in later:
            portfolio = refresh_portfolio(portfolio, context, observations_of(row), row.id)
    except (ValueError, KeyError, TypeError):
        return None
    return portfolio


def apply_skills(db, candidate, attempt, result, *, claimed=False):
    if (candidate.specialization, candidate.grade) != (attempt.specialization, attempt.grade):
        return "category_not_current"
    if attempt.kind == "grade" and not result.get("passed"):
        return "not_updated"
    context = context_of(attempt)
    portfolio = candidate.skill_portfolio
    compatible = portfolio.get("method") == METHOD and portfolio.get("context") == context
    if compatible and attempt.id in portfolio.get("applied_attempts", []):
        return "already_applied"
    if attempt.kind == "grade":
        saved_context = portfolio.get("context", {})
        same_category = (portfolio.get("method") == METHOD and isinstance(saved_context, dict)
                         and all(saved_context.get(k) == context[k] for k in ("specialization", "grade")))
        # Awarding an older pending result must not roll back newer evidence,
        # including when its rubric version differs from the current portfolio.
        if claimed and same_category:
            anchor = db.get(Attempt, portfolio["anchor_attempt_id"])
            if anchor and event_order(anchor) >= event_order(attempt):
                return "older_assessment_retained"
        portfolio = start_portfolio(context, observations_of(attempt), attempt.id)
        outcome = "updated_full"
    else:
        if not compatible:
            portfolio = restore_portfolio(db, candidate, attempt)
        if portfolio is None:
            return "needs_full_assessment"
        try:
            portfolio = refresh_portfolio(portfolio, context, observations_of(attempt), attempt.id)
        except (ValueError, KeyError, TypeError):
            return "needs_full_assessment"
        outcome = "updated_families"
    candidate.skill_portfolio = portfolio
    candidate.skills = summarize(portfolio)
    return outcome
