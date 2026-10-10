"""Equivalence checks for the exact single-column decomposition."""
import unittest
import numpy as np

from research.core import Belief, assignment, generate_market, value_of_information
from research.fast_voi import SingleColumnCache, cached_value_of_information


class CacheChecks(unittest.TestCase):
    def test_column_replacement_equals_full_assignment(self):
        rng = np.random.default_rng(627)
        for jobs, candidates in [(0, 3), (1, 1), (3, 4), (4, 2), (8, 12)]:
            for _ in range(12):
                weights = rng.normal(size=(jobs, candidates))
                weights[rng.random(weights.shape) < 0.2] = -1e6
                cache = SingleColumnCache(weights)
                for i in range(candidates):
                    changed = weights.copy()
                    changed[:, i] = rng.normal(size=jobs)
                    self.assertAlmostEqual(cache.value(i, changed[:, i]), assignment(changed)[0])

    def test_single_and_pair_values_equal_direct_solver(self):
        for seed in range(5):
            market = generate_market(seed, "leakage")
            b = Belief(leakage=0.3)
            for i in range(8):
                for k in range(3):
                    b.update((i, k, 0), bool(market.responses[i, k, 0, 0]), additional=False)
            p = b.mastery()
            w = market.weights(p)
            cache = SingleColumnCache(w)
            for i in range(8):
                batches = [((i, 0, 1),), ((i, 0, 1), (i, 0, 2)),
                           ((i, 0, 1), (i, 1, 1))]
                for batch in batches:
                    direct = value_of_information(batch, b, market, p, w, cache.base_value)
                    fast = cached_value_of_information(batch, b, market, p, cache)
                    self.assertAlmostEqual(direct, fast)

    def test_reject_changes_to_multiple_candidates(self):
        market = generate_market(0, "leakage")
        b = Belief(leakage=0.3)
        p = b.mastery()
        cache = SingleColumnCache(market.weights(p))
        with self.assertRaises(ValueError):
            cached_value_of_information(((0, 0, 0), (1, 0, 0)), b, market, p, cache)


if __name__ == "__main__":
    unittest.main()
