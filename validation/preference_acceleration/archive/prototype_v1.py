"""Same linear SUM top-k EVOI with a bounded matrix-chain fast path.

The frozen reference engine remains authoritative for near ties and stopping
boundaries. This is a numerical optimization, not a new preference model.
"""
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from app.preference_engine import Belief, _pair, questions, response_probability, topk_sum

ROUNDING_GUARD = 1e-10


@dataclass(frozen=True)
class LinearQuestion:
    pair: tuple[int, int]
    expected_gain: float
    used_reference: bool = False


def _inputs(values, weights, probabilities):
    x = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if (x.ndim != 2 or not 1 <= x.shape[0] <= 100 or not 1 <= x.shape[1] <= 3
            or not np.isfinite(x).all() or np.any((x < 0) | (x > 1))):
        raise ValueError("1..100 profiles with 1..3 finite normalized scores required")
    if (w.ndim != 2 or not 1 <= w.shape[0] <= 3072 or w.shape[1] != x.shape[1]
            or not np.isfinite(w).all() or np.any(w < 0)
            or not np.allclose(w.sum(axis=1), 1., atol=1e-12, rtol=1e-12)):
        raise ValueError("bounded simplex weight hypotheses required")
    return x, w, Belief(w @ x.T, probabilities)


def _gains(belief, weights, values, k, asked):
    n = values.shape[0]
    scores = belief.scores
    prior_value = float(topk_sum(scores, k))
    excluded = {tuple(sorted(_pair(pair, n))) for pair in asked}
    pairs = [pair for pair in combinations(range(n), 2) if pair not in excluded]
    gains = np.empty(len(pairs))
    for start in range(0, len(pairs), 128):
        block = pairs[start:start+128]
        indices = np.asarray(block)
        left = belief.utilities[:, indices[:, 0]].T
        right = belief.utilities[:, indices[:, 1]].T
        likelihood = response_probability(left-right)
        unnormalized = likelihood * belief.probabilities
        first = np.linalg.multi_dot([unnormalized, weights, values.T])
        second = scores-first
        block_gains = topk_sum(first, k) + topk_sum(second, k) - prior_value
        if np.any(block_gains < -ROUNDING_GUARD):
            raise ArithmeticError("negative information value beyond roundoff")
        gains[start:start+len(block)] = np.maximum(block_gains, 0.)
    return pairs, gains


def linear_gains(values, weights, probabilities, k, *, asked=()):
    """All decision gains for validation; no entropy/alternate-policy work."""
    values, weights, belief = _inputs(values, weights, probabilities)
    return _gains(belief, weights, values, k, asked)


def best_linear_question(values, weights, probabilities, k, *, asked=(), min_gain=0.):
    """Select exactly one step of the existing model, or stop at min_gain."""
    if not np.isfinite(min_gain) or min_gain < 0:
        raise ValueError("finite nonnegative stopping threshold required")
    values, weights, belief = _inputs(values, weights, probabilities)
    asked = tuple(asked)
    pairs, gains = _gains(belief, weights, values, k, asked)
    if not pairs:
        return None
    best_index = int(np.argmax(gains))
    best_gain = float(gains[best_index])
    if best_gain < min_gain-ROUNDING_GUARD:
        return None
    near_boundary = abs(best_gain-min_gain) <= ROUNDING_GUARD
    near_tie = (len(gains) > 1 and
                best_gain-float(np.partition(gains, -2)[-2]) <= ROUNDING_GUARD)
    if near_boundary or near_tie:
        reference = max(questions(belief, k, asked=asked), key=lambda q: q.expected_gain)
        if reference.expected_gain <= min_gain:
            return None
        return LinearQuestion(reference.pair, reference.expected_gain, used_reference=True)
    return LinearQuestion(pairs[best_index], best_gain)
