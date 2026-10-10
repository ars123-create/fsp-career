"""Independent checks of controls and their use of the actual application."""
from copy import deepcopy
from itertools import combinations
import json

import numpy as np
import pytest

from .benchmark import make_case, jsonable
from .controls import (app_control, fixed_swing_control, metrics, normalize_weights,
                       numeric_control, run_controls, weighted_topk)


def test_official_numpy_example():
    np.testing.assert_allclose(np.average(np.arange(6).reshape(3, 2), axis=1, weights=[.25, .75]), [.75, 2.75, 4.75])


@pytest.mark.parametrize("weights", [[0, 0, 0], [-1, 2, 3], [1, 2], [1, 2, np.nan], [1, 2, np.inf]])
def test_invalid_weights(weights):
    with pytest.raises(ValueError):
        normalize_weights(weights)


def test_direct_rank_matches_independent_enumeration():
    values = np.array([[1, .5, .5], [.5, 1, 1], [.7, .8, .9], [.9, .7, .6]])
    weights = np.array([2, 5, 3])
    scores = [sum(float(x) * int(w) for x, w in zip(row, weights)) / 10 for row in values]
    optimum = max(sum(scores[i] for i in subset) for subset in combinations(range(4), 2))
    chosen = weighted_topk(values, weights, 2)
    assert sum(scores[i] for i in chosen) == pytest.approx(optimum)
    assert chosen == (1, 2)
    assert weighted_topk(values, weights * 19, 2) == chosen


def test_rounding_noise_and_all_zero_fallback():
    np.testing.assert_allclose(numeric_control([.12, .33, .55], step=.05), [.1, .35, .55])
    np.testing.assert_allclose(numeric_control([.2, .3, .5], sigma=.1, z=[1, 0, -1]), [.3, .3, .4])
    np.testing.assert_allclose(numeric_control([.2, .3, .5], sigma=1., z=[-1, -1, -1]), [1/3]*3)


def test_metrics_against_hand_calculation_and_bad_selection():
    result = metrics([0, 2], [.9, .8, .5], k=2)
    assert result["regret_points"] == pytest.approx(15.)
    assert result["oracle_overlap"] == .5
    assert 0 < result["ndcg_at_5"] < 1
    with pytest.raises(ValueError):
        metrics([0, 0], [.9, .8, .5], k=2)


def test_actual_adapter_boundary_moves_without_changing_inputs():
    case = make_case(945000, "linear")
    case["observed"] = np.array([[1., 1., 1.]] * 4 + [[1., 1., .5], [1., .5, 1.]])
    case["weights"] = np.array([.1, .1, .8])
    case["uniforms"] = np.full(3, 1 - 1e-12)
    before = deepcopy(case["observed"])
    result = app_control(case)
    assert result["trace"][0]["selected"] == [0, 1, 2, 3, 4]
    assert result["selected"] == [0, 1, 2, 3, 5]
    assert 1 <= result["answers"] <= 3
    np.testing.assert_array_equal(case["observed"], before)


def test_fixed_questions_posterior_by_direct_bayes_formula():
    from app.preference_engine import simplex_particles
    case = make_case(945001, "linear")
    case["uniforms"] = np.zeros(3)
    result = fixed_swing_control(case)
    particles = simplex_particles(3)
    probability = np.ones(len(particles)) / len(particles)
    for a, b in ((0, 1), (0, 2), (1, 2)):
        likelihood = .01 + .98 / (1 + np.exp(-.5 * (particles[:, a] - particles[:, b]) / .04))
        probability *= likelihood
        probability /= probability.sum()
    np.testing.assert_allclose(result["weights"], probability @ particles, atol=1e-13)
    assert result["answers"] == 3


@pytest.mark.parametrize("scenario", ["linear", "rubric_grid", "polarized", "score_noise", "geometric", "answer_noise"])
def test_generated_case_and_reproducibility(scenario):
    first = make_case(945006, scenario)
    second = make_case(945006, scenario)
    assert json.dumps(jsonable(first), sort_keys=True) == json.dumps(jsonable(second), sort_keys=True)
    assert first["observed"].shape == (24, 3)
    assert np.all(first["observed"].sum(axis=1) >= 2.25)
    if scenario == "rubric_grid":
        assert set(first["observed"].ravel()).issubset({.5, .75, 1.})


def test_controls_immutable_and_exact_linear_optimum():
    case = make_case(945007, "linear")
    before = json.dumps(jsonable(case), sort_keys=True)
    result = run_controls(case)
    assert json.dumps(jsonable(case), sort_keys=True) == before
    assert result["direct_exact"]["regret_points"] == pytest.approx(0., abs=1e-10)
    assert result["direct_exact"]["answers"] is None
    assert result["equal"]["answers"] == 0
    assert len(result) == 8
    assert result == run_controls(case)
