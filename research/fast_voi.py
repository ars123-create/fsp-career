"""Exact reuse of assignment subproblems when only one candidate column changes.

This is a specialization of basic assignment sensitivity, not a claim to a new
general-purpose assignment algorithm. scipy's solver is reused unmodified.
"""
from __future__ import annotations

import numpy as np

from research.core import Belief, Market, Query, assignment


class SingleColumnCache:
    def __init__(self, weights: np.ndarray):
        jobs, candidates = weights.shape
        self.without_candidate = np.empty(candidates)
        self.without_both = np.empty((jobs, candidates))
        self.base_value = assignment(weights)[0]
        for i in range(candidates):
            reduced = np.delete(weights, i, axis=1)
            self.without_candidate[i] = assignment(reduced)[0]
            for j in range(jobs):
                self.without_both[j, i] = assignment(np.delete(reduced, j, axis=0))[0]

    def value(self, candidate: int, new_column: np.ndarray) -> float:
        """The rest of the matrix MUST be unchanged since cache construction."""
        options = self.without_both[:, candidate] + new_column
        return float(max(self.without_candidate[candidate], options.max(initial=-np.inf)))


def cached_value_of_information(batch: tuple[Query, ...], belief: Belief,
                                market: Market, mastery: np.ndarray,
                                cache: SingleColumnCache) -> float:
    candidates = {q[0] for q in batch}
    if len(candidates) != 1:
        raise ValueError("Cache supports a nonempty batch for exactly one candidate")
    i = next(iter(candidates))
    expected = 0.0
    for probability, updates in belief.branches(batch):
        changed = mastery[i].copy()
        for (_, skill), posterior in updates.items():
            changed[skill] = posterior @ belief.states[:, 0]
        expected += probability * cache.value(i, market.column(i, changed))
    return float(expected - cache.base_value)
