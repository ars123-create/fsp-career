from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import ndtr

from research.frontier_2026.finite import FiniteModel
from research.frontier_2026.policies import Config as ParticleConfig, ParticleSession
from research.frontier_2026.components.calibration import (
    COMMON_QUERY_SEED, PredictionUnavailable, common_query_ids, expected_scores,
    predicted_probabilities, prediction_report,
)
from research.frontier_2026.components.constrained_gp.adapter import Config as GPConfig, Session as GP
from research.frontier_2026.components.distillation.adapter import Session as Distill
from research.frontier_2026.components.legacy_controls.adapter import Equal, LinearEVOI, GaussianTopK, PABBO
from research.frontier_2026.components.monne.adapter import Config as MoNNEConfig, MoNNE
from research.frontier_2026.components.risk_decisions.adapter import Config as RiskConfig, Session as Risk
from research.nonlinear_topk.core import GaussianBelief
from research.preference_topk.core import Belief


X = np.array([[.1,.9], [.3,.7], [.7,.3], [.9,.1], [.6,.6], [.2,.2], [.8,.8]])


def test_common_queries_only_x_fixed_seed_permutation_duplicates():
    x = np.random.default_rng(333).uniform(size=(30, 3))
    before = np.random.get_state()
    ids = common_query_ids(x)
    after = np.random.get_state()
    assert COMMON_QUERY_SEED == 202610084271
    assert len(ids) == 32 and ids == tuple(sorted(set(ids)))
    assert common_query_ids(x[::-1]) == ids
    assert common_query_ids(np.r_[x, x[:1]]) == ids
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])
    assert common_query_ids([[.5]]) == ()


@pytest.mark.parametrize("policy", ["kg", "latent_noise", "fractional", "answerability", "membership"])
def test_particle_actual_likelihood_and_posterior(policy):
    s = ParticleSession(X, ParticleConfig(policy=policy, m=2))
    history, queries = ((0, 0), (1, 2)), (0, 1, 3)
    model = s._model(5)
    expected = model.likelihood[list(queries)] @ s._weights(model, history)
    report = prediction_report(s, history, queries)
    np.testing.assert_allclose(report.probabilities, expected, atol=1e-15)
    assert report.asked_fraction == 2/3
    assert "no new answer" in report.scope
    if policy == "fractional":
        assert report.status == "GENERALIZED_POSTERIOR"
    if policy == "answerability":
        assert np.ptp(report.probabilities[:, 2]) > 0


def test_analytical_finite_mixture_and_fractional_channel():
    s = ParticleSession(X[:2], ParticleConfig(m=2))
    utilities = np.array([[0.,1.], [1.,0.]])
    likelihood = np.array([[[.2,.6],[.6,.2],[.2,.2]]])
    model = FiniteModel(utilities, likelihood, np.array([.25,.75]), 2)
    s._models[2] = model
    np.testing.assert_allclose(predicted_probabilities(s, (), (0,)), [[.5,.3,.2]])
    # P(h|left) = (.05,.45)/.5 = (.1,.9).
    np.testing.assert_allclose(predicted_probabilities(s, ((0,0),), (0,)), [[.56,.24,.2]])
    s.config = replace(s.config, policy="fractional", eta=.5)
    w = np.array([.25*np.sqrt(.2), .75*np.sqrt(.6)])
    w /= w.sum()
    np.testing.assert_allclose(predicted_probabilities(s, ((0,0),), (0,)), likelihood @ w)


def test_constrained_gp_actual_snis_channel_and_skip():
    s = GP(X, GPConfig(particles=128, proposal_batch=128, minimum_ess=2))
    history, ids = ((0, 1), (1, 2)), (0, 1, 2)
    expected = [s.response_likelihood(q) @ s.posterior_weights(history) for q in ids]
    report = prediction_report(s, history, ids)
    assert report.status == "APPROXIMATE_SNIS"
    np.testing.assert_allclose(report.probabilities, expected)
    np.testing.assert_array_equal(predicted_probabilities(s, (), ids), predicted_probabilities(s, ((0,2),), ids))


def test_distillation_uses_finite_belief_not_network(monkeypatch):
    s = Distill(X)
    def forbidden(*a, **kw):
        raise AssertionError("student used to generate response probabilities")
    monkeypatch.setattr(s, "predicted", forbidden)
    model = s.model()
    history, ids = ((0, 0),), (0, 2)
    np.testing.assert_allclose(predicted_probabilities(s, history, ids),
                               model.likelihood[list(ids)] @ model.posterior(history))


@pytest.mark.parametrize("method", ["kl_dro", "cvar", "baseline_safe", "edgecut"])
def test_risk_nominal_bayes_modes(method):
    s = Risk(X, RiskConfig(m=2))
    assert prediction_report(s, (), (0,)).status == "NOT_AVAILABLE"
    report = prediction_report(s, ((0,1),), (0,2), risk_method=method)
    model = s._model(5)
    np.testing.assert_allclose(report.probabilities, model.likelihood[[0,2]] @ model.posterior(((0,1),)))
    s.last_diagnostics["method"] = method
    np.testing.assert_array_equal(predicted_probabilities(s, ((0,1),), (0,2)), report.probabilities)


def test_unavailable_equal_pabbo_mmr_not_invented():
    equal = Equal(X)
    # No weights need loading for a policy with no response model.
    pabbo = PABBO.__new__(PABBO)
    pabbo.values, pabbo.pairs, pabbo.validator = equal.values, equal.pairs, equal.validator
    risk = Risk(X, RiskConfig(m=2))
    for s, method in ((equal, None), (pabbo, None), (risk, "mmr")):
        report = prediction_report(s, (), (0,), risk_method=method)
        assert report.status == "NOT_AVAILABLE" and report.probabilities is None
        with pytest.raises(PredictionUnavailable):
            predicted_probabilities(s, (), (0,), risk_method=method)


def test_legacy_linear_analytic_binary_plus_neutral_skip(monkeypatch):
    s = LinearEVOI(X[:2])
    belief = Belief(np.array([[.1,.9],[.9,.1]]), np.array([.25,.75]))
    monkeypatch.setattr(s, "belief", lambda history: belief)
    left = belief.likelihood(s.pairs[0], temperature=.04, lapse=.02) @ belief.probabilities
    np.testing.assert_allclose(predicted_probabilities(s, (), (0,)), [[.85*left, .85*(1-left), .15]])


def test_legacy_gp_analytic_noise_variance_two(monkeypatch):
    s = GaussianTopK(X[:2])
    belief = GaussianBelief(np.array([1.,-1.]), np.array([[1.,.2],[.2,1.]]))
    monkeypatch.setattr(s, "belief", lambda history: belief)
    expected_left = ndtr(2/np.sqrt(1+1-2*.2+2))
    report = prediction_report(s, (), (0,))
    np.testing.assert_allclose(report.probabilities, [[.85*expected_left,.85*(1-expected_left),.15]], atol=1e-15)
    assert report.status == "APPROXIMATE_LAPLACE"


def test_actual_legacy_gp_cold_and_answered():
    s = GaussianTopK(X)
    ids = (0, 1)
    initial = predicted_probabilities(s, (), ids)
    after = predicted_probabilities(s, ((0,0),), ids)
    assert np.all(initial > 0) and np.all(after > 0)
    assert after[0,0] > initial[0,0]
    np.testing.assert_array_equal(initial, predicted_probabilities(s, ((0,2),), ids))


def test_monne_status_no_double_fit_conditioning(monkeypatch):
    s = MoNNE(X[:2], MoNNEConfig(ensemble_size=2))
    calls = []
    def already_fitted(history):
        calls.append(history)
        return np.array([[.2,.8],[.9,.1]])
    monkeypatch.setattr(s, "hypotheses", already_fitted)
    history = ((0,0),)
    report = prediction_report(s, history, (0,))
    explicit = FiniteModel.pairwise(already_fitted(history), s.pairs, 2,
                                    temperature=.04, lapse=.02, skip_probability=.15)
    np.testing.assert_allclose(report.probabilities, explicit.likelihood @ explicit.prior)
    assert report.status == "UNCALIBRATED_ENSEMBLE"
    assert calls == [history, history]  # one adapter call, one independent expected-value construction


def test_actual_monne_cold_probability_valid():
    s = MoNNE(X[:2], MoNNEConfig(ensemble_size=2))
    p = predicted_probabilities(s, (), (0,))
    np.testing.assert_allclose(p.sum(axis=1), 1)
    assert np.all(p > 0)


def test_expected_losses_independent_outcome_enumeration():
    p = np.array([[.6,.3,.1], [.1,.2,.7]])
    t = np.array([[.2,.5,.3], [.1,.8,.1]])
    result = expected_scores(p, t)
    brier = np.array([sum(tr[r]*np.sum((pr-np.eye(3)[r])**2) for r in range(3)) for pr,tr in zip(p,t)])
    logloss = np.array([sum(-tr[r]*np.log(pr[r]) for r in range(3)) for pr,tr in zip(p,t)])
    np.testing.assert_allclose(result["categorical_expected_brier"], brier)
    np.testing.assert_allclose(result["categorical_expected_logloss"], logloss)
    np.testing.assert_allclose(result["categorical_expected_brier"],
                               result["categorical_probability_squared_error"]+result["categorical_irreducible_brier"])
    np.testing.assert_allclose(result["conditional_binary_predicted_left"], [2/3,1/3])
    assert np.all(result["conditional_binary_valid"])


def test_probability_error_distinct_from_expected_brier():
    p = np.array([[.4,.4,.2]])
    result = expected_scores(p, p)
    assert result["categorical_probability_squared_error"][0] == 0
    assert result["categorical_expected_brier"][0] > 0
    assert result["conditional_binary_expected_brier"][0] == .25
    assert result["categorical_excess_logloss_kl"][0] == 0


def test_zero_events_inf_loss_and_undefined_conditional_binary():
    p = np.array([[1,0,0], [0,0,1], [0,0,1]], dtype=float)
    t = np.array([[1,0,0], [0,0,1], [.2,.3,.5]], dtype=float)
    result = expected_scores(p, t)
    assert result["categorical_expected_logloss"][0] == 0
    assert result["categorical_expected_logloss"][1] == 0
    assert np.isinf(result["categorical_expected_logloss"][2])
    np.testing.assert_array_equal(result["conditional_binary_valid"], [True, False, False])
    assert np.isnan(result["conditional_binary_expected_brier"][1:]).all()


def test_empty_queries_and_three_slot_diagnostic():
    s = ParticleSession(X, ParticleConfig(m=2))
    assert predicted_probabilities(s, (), ()).shape == (0,3)
    p = predicted_probabilities(s, ((0,2),(1,2),(2,2)), (0,3))
    np.testing.assert_array_equal(p, predicted_probabilities(s, (), (0,3)))
    result = expected_scores(np.empty((0,3)), np.empty((0,3)))
    assert all(len(value) == 0 for value in result.values())


@pytest.mark.parametrize("ids", [(-1,), (True,), (9999,), (0,0)])
def test_invalid_query_ids(ids):
    with pytest.raises(ValueError):
        predicted_probabilities(ParticleSession(X, ParticleConfig(m=2)), (), ids)


@pytest.mark.parametrize("probabilities", [[[.2,.2,.2]], [[-.1,.5,.6]], [[np.nan,.4,.6]], [[.5,.5]]])
def test_invalid_score_probabilities(probabilities):
    with pytest.raises(ValueError):
        expected_scores(probabilities, [[.2,.3,.5]])
