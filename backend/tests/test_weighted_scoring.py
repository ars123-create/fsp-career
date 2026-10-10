"""Independent arithmetic and boundary checks before application integration."""
from itertools import combinations

import numpy as np
import pytest

from app.weighted_scoring import direct_scores


def test_manual_weighted_sum_and_optimal_pair():
    values = np.array([[100., 50., 50.], [50., 100., 100.], [70., 80., 90.], [90., 70., 60.]])
    weights = np.array([20., 50., 30.])
    original = values.copy(), weights.copy()
    scores, normalized = direct_scores(values, weights)
    np.testing.assert_allclose(scores, [.60, .90, .81, .71])
    np.testing.assert_allclose(normalized, [.2, .5, .3])
    expected = max(sum(scores[i] for i in subset) for subset in combinations(range(4), 2))
    assert sum(scores[i] for i in np.argsort(-scores)[:2]) == pytest.approx(expected)
    np.testing.assert_array_equal(values, original[0])
    np.testing.assert_array_equal(weights, original[1])


@pytest.mark.parametrize("d", [1, 2, 3])
def test_equal_scale_and_empty_results(d):
    x = np.array([[50.] * d, [100.] * d])
    a, w = direct_scores(x, [1.] * d)
    b, _ = direct_scores(x, [10.] * d)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(a, [.5, 1.])
    assert sum(w) == pytest.approx(1.)
    empty, _ = direct_scores(np.empty((0, d)), [1.] * d)
    assert empty.shape == (0,)


def test_zero_weight_and_equal_score_tie():
    scores, _ = direct_scores([[100, 50], [100, 100]], [100, 0])
    np.testing.assert_array_equal(scores, [1., 1.])


@pytest.mark.parametrize("weights", [[0, 0], [-1, 2], [101, 1], [np.nan, 2], [np.inf, 2], [1]])
def test_bad_weights(weights):
    with pytest.raises(ValueError):
        direct_scores([[50, 100]], weights)


@pytest.mark.parametrize("values", [[[101, 50]], [[-1, 50]], [[np.nan, 50]], [[np.inf, 50]], [50, 100], [[]], [[50] * 4]])
def test_bad_scores(values):
    with pytest.raises(ValueError):
        direct_scores(values, [1, 1])
