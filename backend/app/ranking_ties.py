"""Factual achievement tie-break after an already computed skill score.

No score bonus, grade mutation or candidate eligibility decision happens here.
Official source verification remains an integration boundary; demo sources
remain explicitly fictional in the profile presented to the employer.
"""


def has_confirmed_achievement(candidate):
    return any(
        isinstance(item, dict)
        and item.get("verified") is True
        and not item.get("revoked", False)
        and isinstance(item.get("source"), str)
        and bool(item["source"].strip())
        for item in candidate.achievements
    )


def evidence_tied_topk(scores, candidates, k):
    """Sort by skill score, confirmed achievement presence, then stable ID.

    Scores have the calling adapter's existing precision. Achievement count,
    free-form FSP ID and contact fields are deliberately absent from the key.
    """
    return tuple(sorted(range(len(candidates)), key=lambda i: (
        -float(scores[i]), -has_confirmed_achievement(candidates[i]),
        candidates[i].user_id,
    ))[:k])
