"""Component checks, not final heldout ranking evaluation or hiring validation."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.special import ndtr

from research.frontier_2026.finite import FiniteModel, tradeoff_pairs
from research.frontier_2026.components.constrained_gp.adapter import (
    Config, PosteriorDegeneracyError, SamplingError, Session,
)


VALUES = np.array([[.9, .1], [.1, .9], [.5, .5], [.8, .8], [.2, .2], [.3, .7]])
SMALL = Config(particles=256, proposal_batch=256, minimum_ess=4)


@pytest.fixture(scope="module")
def session():
    return Session(VALUES, SMALL)


def test_coldstart_and_public_contract(session):
    selected, query = session.recommend()
    assert len(set(selected)) == 5
    assert query is None or 0 <= query < len(session.pairs)
    assert session.pairs == tradeoff_pairs(VALUES)
    assert session.utilities.shape == (256, 6)
    assert np.isfinite(session.utilities).all()
    assert session.hypotheses() is session.utilities
    assert session.sampling_diagnostics["status"] == "PASS"


def test_sample_and_posterior_dominance(session):
    x, u = session.values, session.utilities
    for i in range(len(x)):
        for j in range(len(x)):
            if np.all(x[i] >= x[j]):
                assert np.all(u[:, i] >= u[:, j])
    for history in ((), ((0, 0),), ((0, 1), (1, 0), (2, 2))):
        mean = session.posterior_weights(history) @ u
        for i in range(len(x)):
            for j in range(len(x)):
                if np.all(x[i] >= x[j]):
                    assert mean[i] >= mean[j]-2e-15


def test_no_double_counting_and_history_order(session):
    history = ((0, 0), (1, 1), (2, 2))
    manual = np.full(SMALL.particles, 1/SMALL.particles)
    for q, r in history:
        manual *= session.response_likelihood(q)[r]
    manual /= manual.sum()
    np.testing.assert_allclose(session.posterior_weights(history), manual, rtol=2e-14)
    np.testing.assert_array_equal(session.posterior_weights(history),
                                  session.posterior_weights(tuple(reversed(history))))
    assert session.hypotheses(history) is session.utilities


def test_skip_identity_and_budget(session):
    before = session.posterior_weights()
    for history in (((0, 2),), ((0, 2), (1, 2)), ((0, 2), (1, 2), (2, 2))):
        np.testing.assert_array_equal(session.posterior_weights(history), before)
        assert session.recommend(history)[0] == session.recommend()[0]
    assert session.recommend(((0, 2), (1, 2), (2, 2)))[1] is None
    available, _ = session.acquisition_values(((0, 2),), k=1)
    assert 0 not in available


def test_same_seed_and_no_global_rng_mutation(session):
    before = np.random.get_state()
    repeated = Session(VALUES, SMALL)
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])
    np.testing.assert_array_equal(repeated.utilities, session.utilities)
    assert repeated.recommend() == session.recommend()


def test_permutation_and_duplicate_invariance(session):
    permutation = np.array([5, 2, 4, 0, 3, 1])
    reordered = Session(VALUES[permutation], SMALL)
    np.testing.assert_array_equal(reordered.utilities[:, np.argsort(permutation)], session.utilities)
    history = ((0, 0), (1, 2))
    selected, query = session.recommend(history, k=3)
    new_selected, new_query = reordered.recommend(history, k=3)
    assert tuple(permutation[list(new_selected)]) == selected
    assert new_query == query
    old_ids, old_acq = session.acquisition_values(history, k=3)
    new_ids, new_acq = reordered.acquisition_values(history, k=3)
    assert old_ids == new_ids
    np.testing.assert_array_equal(old_acq, new_acq)
    duplicates = Session(np.r_[VALUES, VALUES[:1]], SMALL)
    np.testing.assert_array_equal(duplicates.utilities[:, 0], duplicates.utilities[:, -1])
    np.testing.assert_array_equal(duplicates.utilities[:, :-1], session.utilities)
    assert len(duplicates.pairs) == len(session.pairs)


def test_small_pool_and_all_identical_vectors():
    for x in ([[.5]], [[.3, .7]]*8):
        s = Session(x, SMALL)
        chosen, q = s.recommend()
        assert len(chosen) == min(5, len(x)) and q is None
        assert s.sampling_diagnostics["acceptance_rate"] == 1.
        if len(x) > 1:
            assert np.all(s.utilities == s.utilities[:, :1])


def test_latent_utility_not_a_probability():
    s = Session([[.5]], replace(SMALL, amplitude=3.))
    assert s.utilities.min() < 0 and s.utilities.max() > 1


@pytest.mark.parametrize("acquisition", ["top5", "information", "eubo"])
def test_three_slot_limit_and_no_repeat(session, acquisition):
    history = []
    for _ in range(3):
        selected, query = session.recommend(tuple(history), acquisition=acquisition, k=2)
        assert len(set(selected)) == 2
        if query is None:
            break
        assert query not in {q for q, _ in history}
        history.append((query, 2))
    if len(history) == 3:
        assert session.recommend(tuple(history), acquisition=acquisition, k=2)[1] is None


def test_all_skip_channel_and_impossible_observations():
    always_skip = Session(VALUES, replace(SMALL, skip_probability=1.))
    assert always_skip.recommend(k=1)[1] is None
    with pytest.raises(ValueError, match="answer is impossible"):
        always_skip.posterior_weights(((0, 0),))
    never_skip = Session(VALUES, replace(SMALL, skip_probability=0.))
    with pytest.raises(ValueError, match="skip is impossible"):
        never_skip.posterior_weights(((0, 2),))


def test_uninformative_answer_exactly_preserves_prior():
    s = Session(VALUES, replace(SMALL, lapse=1.))
    np.testing.assert_array_equal(s.posterior_weights(((0, 0), (1, 1))), s.posterior_weights())


def test_response_coherence_evoi_and_independent_finite_reference(session):
    likelihood = np.array([session.response_likelihood(q) for q in range(len(session.pairs))])
    np.testing.assert_allclose(likelihood.sum(axis=1), 1., rtol=0, atol=2e-16)
    for history in ((), ((0, 0),), ((0, 1), (1, 2))):
        p = session.posterior_weights(history)
        ids, ours = session.acquisition_values(history, k=2)
        reference = FiniteModel(session.utilities, likelihood, p, 2)
        expected = reference.one_step_values(p, ids)
        np.testing.assert_allclose(ours, expected, rtol=3e-14, atol=3e-14)
        assert np.min(ours-reference.terminal(p)) >= -3e-14
        for q in ids:
            recovered = sum(p*likelihood[q, r] for r in range(3))
            np.testing.assert_allclose(recovered, p, rtol=3e-16, atol=1e-18)


def test_diagnostics_are_finite(session):
    d = session.diagnostics(((0, 0),), query=1, k=2)
    assert 4 <= d["posterior_ess"] <= SMALL.particles+1e-10
    assert all(np.isfinite(d["posterior_mean_mcse"]))
    assert d["fixed_query_batch_mcse"] >= 0
    assert len(d["fixed_query_batch_values"]) == 8
    assert "not calibration" in d["mcse_scope"]


def test_explicit_rejection_cap_failure():
    x = np.linspace(.49, .51, 10)[:, None]
    cfg = replace(SMALL, particles=64, proposal_batch=64, max_proposals=64,
                  amplitude=10., lengthscale=.001)
    with pytest.raises(SamplingError, match="proposal cap") as err:
        Session(x, cfg)
    assert err.value.diagnostics["status"] == "FAIL"
    assert err.value.diagnostics["proposed"] == 64
    assert err.value.diagnostics["retained_particles"] < 64


def test_ess_failure_has_no_fallback():
    s = Session([[.8, .2], [.2, .8]], replace(SMALL, minimum_ess=256.))
    with pytest.raises(PosteriorDegeneracyError) as err:
        s.posterior_weights(((0, 0),))
    assert 1 <= err.value.effective_sample_size < 256


def test_analytical_one_constraint_truncated_gaussian_moment():
    # D = f(1)-f(0) ~ N(mu=1,var), conditioned D>=0.
    cfg = replace(SMALL, particles=16384, max_proposals=65536, proposal_batch=2048,
                  amplitude=1., lengthscale=1., seed=2026100842)
    s = Session([[0.], [1.]], cfg)
    difference = s.utilities[:, 1]-s.utilities[:, 0]
    variance = 2*(cfg.amplitude**2*(1-np.exp(-.5))+cfg.covariance_jitter)
    sd = np.sqrt(variance)
    exact = 1 + sd*np.exp(-.5/sd**2)/np.sqrt(2*np.pi)/ndtr(1/sd)
    mcse = difference.std(ddof=1)/np.sqrt(len(difference))
    assert abs(difference.mean()-exact) < 6*mcse
    assert np.all(difference >= 0)


def test_analytical_probit_posterior_moment():
    # No dominance relation; D~N(0,v), P(left|D)=Phi(D/noise).
    # E[D|left] = v/sqrt(v+noise^2)*sqrt(2/pi).
    cfg = replace(SMALL, particles=16384, max_proposals=16384, proposal_batch=2048,
                  amplitude=.4, lapse=0., seed=2026100843)
    s = Session([[.2, .8], [.8, .2]], cfg)
    a, b = s.pairs[0]
    difference = s.utilities[:, a]-s.utilities[:, b]
    v = 2*(cfg.amplitude**2*(1-np.exp(-.72/(2*cfg.lengthscale**2)))+cfg.covariance_jitter)
    exact = v/np.sqrt(v+cfg.probit_noise**2)*np.sqrt(2/np.pi)
    weights = s.posterior_weights(((0, 0),))
    estimate = weights @ difference
    mcse = np.sqrt(len(weights)/(len(weights)-1)*np.sum(weights**2*(difference-estimate)**2))
    assert abs(estimate-exact) < 6*mcse


@pytest.mark.parametrize("history", [((0, 0), (0, 1)), ((-1, 0),), ((0, 3),),
                                      ((True, 0),), ((0, False),),
                                      ((0, 0), (1, 0), (2, 0), (3, 0))])
def test_bad_history(session, history):
    with pytest.raises(ValueError):
        session.recommend(history)


@pytest.mark.parametrize("values", [[], [[np.nan]], [[-1.]], [[1.1]], [[.1]*4], [[.1]]*101])
def test_bad_values(values):
    with pytest.raises(ValueError):
        Session(values, SMALL)


@pytest.mark.parametrize("changes", [{"particles": True}, {"max_proposals": 1},
                                        {"probit_noise": 0}, {"lapse": 1.1},
                                        {"skip_probability": -1}, {"lengthscale": np.inf},
                                        {"minimum_ess": 1000}, {"seed": -1}])
def test_bad_config(changes):
    with pytest.raises(ValueError):
        replace(SMALL, **changes)


@pytest.mark.parametrize("k", [0, 101, True, 2.5])
def test_bad_k(session, k):
    with pytest.raises(ValueError):
        session.recommend(k=k)


def test_readonly_input_and_samples(session):
    with pytest.raises(ValueError):
        session.utilities[0, 0] = 0
    with pytest.raises(ValueError):
        session.posterior_weights()[0] = 0
    with pytest.raises(ValueError):
        session.values[0, 0] = 0
