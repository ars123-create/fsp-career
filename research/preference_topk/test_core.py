"""Independent finite enumeration and analytic checks, not benchmark tuning."""
from itertools import combinations

import numpy as np
import pytest

from .core import (Belief, binary_entropy, questions, response_probability,
                   select_question, simplex_particles, topk)


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("k", [1, 2, 4])
def test_every_epu_against_independent_subset_and_answer_enumeration(seed, k):
    rng = np.random.default_rng(seed)
    utility = rng.uniform(0, 1, (5, 4))
    probabilities = rng.dirichlet(np.ones(5))
    belief = Belief(utility, probabilities)
    sets = list(combinations(range(4), k))
    eu = lambda p, s: sum(p[h] * sum(utility[h, i] for i in s) for h in range(5))
    best = max(eu(probabilities, s) for s in sets)
    assert eu(probabilities, belief.ranking(k)) == pytest.approx(best)
    for q in questions(belief, k):
        a, b = q.pair
        # Separately evaluate logistic likelihood and normalized Bayes updates.
        left = np.array([.01 + .98/(1+np.exp(-(utility[h, a]-utility[h, b])/.04))
                         for h in range(5)])
        epu = 0.
        for likelihood in (left, 1-left):
            mass = sum(probabilities * likelihood)
            updated = probabilities * likelihood / mass
            epu += mass * max(eu(updated, s) for s in sets)
        assert q.expected_gain == pytest.approx(epu-best, abs=2e-14)
        assert q.expected_gain >= 0
    chosen = select_question(questions(belief, k), "decision_evoi")
    assert chosen.expected_gain == max(q.expected_gain for q in questions(belief, k))


def test_simplex_symmetry_and_analytic_conditional_centroid():
    samples = simplex_particles(3, m=11)
    assert samples.shape == (6*2**11, 3)
    np.testing.assert_allclose(samples.sum(axis=1), 1, atol=1e-15)
    np.testing.assert_allclose(samples.mean(axis=0), [1/3]*3, atol=1e-14)
    selected = samples[samples[:, 0] >= samples[:, 1]]
    np.testing.assert_allclose(selected.mean(axis=0), [.5, 1/6, 1/3], atol=.0005)
    assert not samples.flags.writeable
    np.testing.assert_array_equal(samples, simplex_particles(3, m=11))


def test_skip_contradictions_order_and_reversal():
    belief = Belief.linear([[1, 0, .1], [0, 1, .1], [.3, .2, .8]], m=7)
    assert belief.observe((0, 1), None) is belief
    a = belief.observe((0, 1), 0).observe((0, 1), 1)
    b = belief.observe((0, 1), 1).observe((0, 1), 0)
    np.testing.assert_allclose(a.probabilities, b.probabilities, atol=1e-16)
    assert np.isfinite(a.probabilities).all()
    np.testing.assert_allclose(belief.observe((0, 1), 0).probabilities,
                               belief.observe((1, 0), 1).probabilities, atol=1e-16)
    original = belief.probabilities.copy()
    belief.observe((0, 2), 0)
    np.testing.assert_array_equal(belief.probabilities, original)


def test_information_value_does_not_guarantee_a_realized_improvement():
    belief = Belief([[1, 0], [0, 1]])
    assert belief.ranking(1) == (0,)
    assert questions(belief, 1)[0].expected_gain > 0
    wrong_answer = belief.observe((0, 1), 1)
    assert wrong_answer.ranking(1) == (1,)
    # If the first hypothesis was the truth, utility just fell from 1 to 0.
    assert belief.utilities[0, wrong_answer.ranking(1)[0]] == 0


def test_irrelevant_uncertainty_can_have_information_but_no_decision_value():
    belief = Belief([[1, .8, .1, .3], [1, .8, .3, .1]])
    question = next(q for q in questions(belief, 2) if q.pair == (2, 3))
    assert question.information_gain > .5
    assert question.expected_gain == pytest.approx(0, abs=1e-14)


def test_random_answers_and_single_hypothesis_have_zero_gain():
    belief = Belief([[1, 0], [0, 1]])
    for q in questions(belief, 1, lapse=1):
        assert q.expected_gain == pytest.approx(0)
        assert q.information_gain == pytest.approx(0)
    assert questions(Belief([[.2, .6, .1]]), 2)[0].expected_gain == pytest.approx(0)
    assert belief.expected_regret(1) == pytest.approx(.5)
    assert belief.expected_regret(2) == pytest.approx(0)


def test_no_repeated_pairs_and_bounded_pool():
    belief = Belief([[.1, .2, .3], [.3, .2, .1]])
    assert [q.pair for q in questions(belief, 1, asked=[(1, 0), (2, 0)])] == [(1, 2)]
    assert questions(Belief([[1.]]), 1) == []
    assert select_question([], "decision_evoi") is None
    options = questions(belief, 1)
    assert select_question(options, "random", np.random.default_rng(1)) == select_question(options, "random", np.random.default_rng(1))
    for policy in ("bald", "uncertainty", "qeubo"):
        assert select_question(options, policy) in options


@pytest.mark.parametrize("matrix,p", [([], None), ([[1, float('nan')]], None),
                                       ([[1.1, 0]], None), ([[.2], [.4]], [.2, .2]),
                                       ([[.2], [.4]], [-.1, 1.1])])
def test_invalid_model_rejected(matrix, p):
    with pytest.raises(ValueError):
        Belief(matrix, p)


@pytest.mark.parametrize("pair", [(0, 0), (-1, 1), (0, 4), (0,), (0.5, 1)])
def test_invalid_pair_rejected(pair):
    with pytest.raises(ValueError):
        Belief([[.1, .2]]).observe(pair, 0)


def test_other_invalid_inputs():
    with pytest.raises(ValueError):
        response_probability([0], temperature=0)
    with pytest.raises(ValueError):
        response_probability([0], lapse=1.1)
    with pytest.raises(ValueError):
        Belief([[.1, .2]]).observe((0, 1), True)
    with pytest.raises(ValueError):
        simplex_particles(4)
    with pytest.raises(ValueError):
        topk([1, 2], 3)
    with pytest.raises(ValueError):
        select_question(questions(Belief([[.1, .2]]), 1), "unrecognised")
    np.testing.assert_allclose(binary_entropy([0, .5, 1]), [0, np.log(2), 0])
