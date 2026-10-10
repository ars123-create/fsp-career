"""Equivalence to frozen reference plus an independent small enumeration."""
from itertools import combinations

import numpy as np
import pytest

from app.preference_engine import Belief, questions, simplex_particles
from validation.preference_acceleration import prototype as fast


@pytest.mark.parametrize("d,n", [(1, 8), (2, 7), (2, 30), (3, 12), (3, 40), (3, 100)])
@pytest.mark.parametrize("seed", [934010, 934011, 934012])
def test_every_gain_and_question_after_answer(d, n, seed):
    rng = np.random.default_rng(seed)
    x = rng.random((n, d))
    w = simplex_particles(d)
    belief = Belief(w @ x.T)
    x_before, w_before = x.copy(), w.copy()
    k = min(5, n)
    asked = []
    for choice in [None, 0, 1]:
        reference = questions(belief, k, asked=asked)
        pairs, gains = fast.linear_gains(x, w, belief.probabilities, k, asked=asked)
        assert pairs == [q.pair for q in reference]
        np.testing.assert_allclose(gains, [q.expected_gain for q in reference], atol=1e-12, rtol=1e-12)
        expected = max(reference, key=lambda q: q.expected_gain)
        actual = fast.best_linear_question(x, w, belief.probabilities, k, asked=asked, min_gain=.0005*k)
        if expected.expected_gain <= .0005*k:
            assert actual is None
        else:
            assert actual.pair == expected.pair
            assert actual.expected_gain == pytest.approx(expected.expected_gain, abs=1e-12)
        belief = belief.observe(expected.pair, choice)
        asked.append(expected.pair)
    np.testing.assert_array_equal(x, x_before)
    np.testing.assert_array_equal(w, w_before)


def test_independent_enumeration():
    x = np.array([[.9, .1], [.1, .9], [.65, .65], [.45, .3]])
    w = np.array([[1., 0.], [.5, .5], [0., 1.]])
    p = np.array([.2, .3, .5])
    utility = w @ x.T
    subsets = list(combinations(range(4), 2))
    before = max(sum(p[h]*sum(utility[h, i] for i in group) for h in range(3)) for group in subsets)
    expected = []
    for a, b in combinations(range(4), 2):
        likelihood = .01+.98/(1+np.exp(-(utility[:, a]-utility[:, b])/.04))
        after = sum(max(sum(p[h]*branch[h]*sum(utility[h, i] for i in group)
                            for h in range(3)) for group in subsets)
                    for branch in (likelihood, 1-likelihood))
        expected.append(after-before)
    _, actual = fast.linear_gains(x, w, p, 2)
    np.testing.assert_allclose(actual, expected, atol=1e-13, rtol=1e-13)


def test_tie_uses_reference():
    x = np.tile(np.eye(3), (6, 1))  # More than one block; tie fallback is exercised.
    w = simplex_particles(3, m=6)
    belief = Belief(w @ x.T)
    expected = max(questions(belief, 1), key=lambda q: q.expected_gain)
    actual = fast.best_linear_question(x, w, belief.probabilities, 1, min_gain=.0005)
    assert actual.used_reference
    assert actual.pair == expected.pair
    assert actual.expected_gain == expected.expected_gain


def test_stopping_boundary_uses_reference(monkeypatch):
    rng = np.random.default_rng(934015)
    x = rng.random((18, 3))  # More than 128 pairs; not the small-pool branch.
    w = simplex_particles(3)
    belief = Belief(w @ x.T)
    expected = max(questions(belief, 5), key=lambda q: q.expected_gain)
    calls = []
    original = fast.questions
    def counted(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(fast, "questions", counted)
    assert fast.best_linear_question(x, w, belief.probabilities, 5, min_gain=expected.expected_gain) is None
    assert calls == [True]


def test_exclusions_and_constant_profiles():
    x = np.full((8, 3), .75)
    w = simplex_particles(3)
    p = np.full(len(w), 1/len(w))
    assert fast.best_linear_question(x, w, p, 5, min_gain=.0025) is None
    assert fast.best_linear_question(x, w, p, 5, asked=combinations(range(8), 2)) is None
    pairs, _ = fast.linear_gains(x, w, p, 5, asked=[(1, 0)])
    assert (0, 1) not in pairs


@pytest.mark.parametrize("x,w,p,k", [
    ([[float('nan'), .2]], [[.5, .5]], [1.], 1),
    ([[1.1, .2]], [[.5, .5]], [1.], 1),
    ([[.1, .2]], [[.5, .6]], [1.], 1),
    ([[.1, .2]], [[.5, .5]], [.8], 1),
    ([[.1, .2]], [[.5, .5]], [1.], 0),
    (np.ones((101, 3)), [[.3, .3, .4]], [1.], 1),
])
def test_invalid_inputs(x, w, p, k):
    with pytest.raises(ValueError):
        fast.best_linear_question(x, w, p, k)
