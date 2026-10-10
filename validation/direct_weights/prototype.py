"""Explicit relative weights over observed scores; no inferred ability or grade."""
import numpy as np


def direct_scores(values, weights):
    """Return normalized scores and weights for one to three declared skills."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.ndim != 2 or not 1 <= values.shape[1] <= 3:
        raise ValueError("one to three skill columns required")
    if weights.shape != (values.shape[1],):
        raise ValueError("one weight per skill required")
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 100)):
        raise ValueError("finite observed scores between zero and 100 required")
    if not np.isfinite(weights).all() or np.any((weights < 0) | (weights > 100)) or weights.sum() <= 0:
        raise ValueError("finite weights between zero and 100 with positive sum required")
    normalized = weights / weights.sum()
    scores = np.round(np.average(values / 100, axis=1, weights=weights), 12)
    return scores, normalized
