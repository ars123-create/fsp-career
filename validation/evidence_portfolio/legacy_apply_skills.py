def apply_skills(candidate, attempt, result):
    """Only evidence for the current confirmed category enters its public profile."""
    if (candidate.specialization, candidate.grade) != (attempt.specialization, attempt.grade):
        return
    candidate.skills = {
        skill: {"score": value, "families": len({q["family"] for q in attempt.questions if q["skill"] == skill}),
                "checked_at": attempt.finished_at, "source": "challenge" if attempt.kind == "challenge" else "test"}
        for skill, value in result["skills"].items()
    }
