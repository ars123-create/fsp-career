"""Latest observation per fixed assessment family; no inferred mastery probability."""
from copy import deepcopy
from statistics import fmean

METHOD = "family-latest-v1"


def checked_observations(observations, attempt_id):
    if not observations or not isinstance(attempt_id, str) or not attempt_id:
        raise ValueError("A nonempty assessment and attempt ID are required")
    seen = set()
    rows = []
    for item in observations:
        skill, family = item["skill"], item["family"]
        if not isinstance(skill, str) or not skill or not isinstance(family, str) or not family:
            raise ValueError("Skill and family must be named")
        if (skill, family) in seen:
            raise ValueError("Duplicate family in one assessment")
        score, timestamp = item["score"], item["checked_at"]
        if isinstance(score, bool) or not isinstance(score, (int, float)) or score not in (0, 100):
            raise ValueError("Only observed binary correctness is supported")
        if type(timestamp) is not int or timestamp < 0:
            raise ValueError("Observation time must be a nonnegative integer")
        seen.add((skill, family))
        rows.append((skill, family, {"score": float(score), "checked_at": timestamp, "attempt_id": attempt_id}))
    return rows


def start_portfolio(context, observations, attempt_id):
    if set(context) != {"specialization", "grade", "rubric_version"} or not all(
        isinstance(value, str) and value for value in context.values()
    ):
        raise ValueError("An explicit assessment context is required")
    rows = checked_observations(observations, attempt_id)
    families = {}
    for skill, family, row in rows:
        families.setdefault(skill, {})[family] = row
    return {"method": METHOD, "context": dict(context), "anchor_attempt_id": attempt_id,
            "families": families, "applied_attempts": [attempt_id]}


def refresh_portfolio(portfolio, context, observations, attempt_id):
    if portfolio.get("method") != METHOD or portfolio.get("context") != context:
        raise ValueError("A full assessment of this category and rubric is required")
    rows = checked_observations(observations, attempt_id)
    if attempt_id in portfolio["applied_attempts"]:
        return deepcopy(portfolio)
    # Validate every row before constructing an updated portfolio.
    for skill, family, row in rows:
        old = portfolio["families"].get(skill, {}).get(family)
        if old is None:
            raise ValueError("New family requires a full assessment")
        if row["checked_at"] < old["checked_at"]:
            raise ValueError("Older evidence cannot replace a later observation")
    updated = deepcopy(portfolio)
    for skill, family, row in rows:
        updated["families"][skill][family] = row
    updated["applied_attempts"].append(attempt_id)
    return updated


def summarize(portfolio):
    if portfolio.get("method") != METHOD or not portfolio.get("families"):
        raise ValueError("Missing portfolio")
    return {
        skill: {"score": round(fmean(row["score"] for row in families.values()), 1),
                "families": len(families),
                "checked_at": min(row["checked_at"] for row in families.values()),
                "latest_checked_at": max(row["checked_at"] for row in families.values()),
                "source": METHOD, "rubric_version": portfolio["context"]["rubric_version"]}
        for skill, families in sorted(portfolio["families"].items())
    }
