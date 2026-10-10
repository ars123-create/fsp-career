"""Checks of mathematical properties and counterexamples, independent of benchmark scores."""
import itertools
import unittest

import numpy as np

from research.core import Belief, Market, assignment, generate_market, value_of_information, run_policy


class ResearchChecks(unittest.TestCase):
    def test_assignment_against_exhaustive_small_problem(self):
        rng = np.random.default_rng(31)
        for _ in range(30):
            w = rng.uniform(-1, 1, (3, 4))
            exact = 0.0
            for choice in itertools.product(range(-1, 4), repeat=3):
                used = [c for c in choice if c >= 0]
                if len(used) == len(set(used)):
                    exact = max(exact, sum(w[j, c] for j, c in enumerate(choice) if c >= 0))
            self.assertAlmostEqual(assignment(w)[0], exact)

    def test_repeated_family_confidence_ceiling(self):
        prior, leakage, high, low, n = 0.5, 0.3, 0.9, 0.15, 12
        belief = Belief(1, 1, 3, leakage=leakage, prior=prior, high=high, low=low)
        for _ in range(n):
            belief.update((0, 0, 0), True)
        expected = prior / (prior + (1-prior) * (leakage + (1-leakage)*(low/high)**n))
        self.assertAlmostEqual(belief.mastery()[0, 0], expected)
        self.assertLess(belief.mastery()[0, 0], 0.78)
        for f in [1, 2]:
            for _ in range(n):
                belief.update((0, 0, f), True)
        self.assertGreater(belief.mastery()[0, 0], 0.97)

    def test_joint_branches_preserve_probability_and_prior(self):
        belief = Belief(1, 2, 3, leakage=0.3)
        for batch in [((0, 0, 0),), ((0, 0, 0), (0, 0, 1)), ((0, 0, 0), (0, 1, 1))]:
            branches = list(belief.branches(batch))
            self.assertAlmostEqual(sum(p for p, _ in branches), 1.0)
            for i, k in {q[:2] for q in batch}:
                expected = sum(p * (post[(i, k)] @ belief.states[:, 0]) for p, post in branches)
                self.assertAlmostEqual(expected, 0.5)

    def test_no_negative_value_of_sample_information(self):
        market = generate_market(12, "leakage")
        b = Belief(leakage=0.3)
        p = b.mastery(); w = market.weights(p); value = assignment(w)[0]
        for batch in [((0, 0, 0),), ((0, 0, 0), (0, 0, 1)), ((0, 0, 0), (0, 1, 1))]:
            self.assertGreaterEqual(value_of_information(batch, b, market, p, w, value), -1e-12)

    def test_two_skills_create_myopic_blind_spot(self):
        market = Market(np.ones((1, 2), bool), np.ones((1, 2), bool), np.ones((1, 1)),
                        np.ones((1, 2, 2, 1), bool), 0.0)
        b = Belief(1, 2, 2, leakage=0.0, high=1.0, low=0.0)
        p = b.mastery(); w = market.weights(p); value = assignment(w)[0]
        one = value_of_information(((0, 0, 0),), b, market, p, w, value)
        pair = value_of_information(((0, 0, 0), (0, 1, 1)), b, market, p, w, value)
        self.assertAlmostEqual(one, 0)
        self.assertAlmostEqual(pair, 0.25)

    def test_common_random_numbers_reproducibility(self):
        market = generate_market(19, "leakage")
        a = run_policy(market, "voi_family", 19)
        b = run_policy(market, "voi_family", 19)
        self.assertEqual(a, b)
        self.assertEqual(a["answers"], 12)
        seen = set()
        for i, k, f, variant, _ in a["trace"]:
            self.assertNotIn((i, k, f, variant), seen)
            seen.add((i, k, f, variant))
        self.assertGreaterEqual(a["regret"], -1e-12)


if __name__ == "__main__":
    unittest.main()
