from itertools import combinations

import numpy as np
import pytest

from .policies import Config, ParticleSession, _entropy_masses

POLICIES = ("kg", "none", "random", "entropy", "bald", "eubo", "membership",
            "set_entropy", "fractional", "easy", "answerability", "thompson",
            "class_robust", "latent_noise")


def values():
    return np.array([[.9, .4, .8], [.5, .9, .7], [.6, .7, .9],
                     [.8, .6, .5], [.4, .8, .9], [.9, .5, .6], [.7, .8, .6]])


@pytest.mark.parametrize("policy", POLICIES)
def test_session_obeys_shape_skip_and_three_slots(policy):
    s = ParticleSession(values(), Config(policy=policy, m=2))
    initial, _ = s.recommend()
    skipped, _ = s.recommend(((0, 2),))
    assert skipped == initial
    history = ((0, 0), (1, 2), (2, 1))
    selected, q = s.recommend(history)
    assert len(set(selected)) == 5 and all(0 <= i < len(values()) for i in selected)
    assert q is None
    with pytest.raises(ValueError):
        s.recommend(history+((3, 0),))
    with pytest.raises(ValueError):
        s.recommend(((0, 0), (0, 2)))


@pytest.mark.parametrize("policy", POLICIES)
def test_one_dimensional_and_small_pool_never_ask(policy):
    s = ParticleSession(np.arange(7)[:, None]/7, Config(policy=policy, m=2))
    selected, q = s.recommend()
    assert set(selected) == {2, 3, 4, 5, 6}
    assert q is None
    s = ParticleSession(values()[:4], Config(policy=policy, m=2))
    selected, q = s.recommend()
    assert set(selected) == {0, 1, 2, 3}
    assert q is None


def test_membership_objective_matches_independent_expected_overlap():
    s = ParticleSession(values(), Config(policy="membership", m=2))
    history = ((0, 0), (1, 1))
    selected, _ = s.recommend(history)
    model = s._model(5)
    weights = model.posterior(history)
    true_sets = [set(sorted(range(len(values())), key=lambda i: -u[i])[:5])
                 for u in model.utilities]
    def score(slate):
        return sum(w*len(set(slate) & truth) for w, truth in zip(weights, true_sets))
    optimum = max(score(slate) for slate in combinations(range(len(values())), 5))
    assert score(selected) == pytest.approx(optimum, abs=1e-12)


def test_entropy_of_unnormalized_categories_is_weighted_conditional_entropy():
    masses = np.array([[.1, .2, .1], [.4, .0, .2]])
    independent = []
    for row in masses:
        total = sum(row)
        independent.append(-sum(float(v)*np.log(float(v)/total) for v in row if v))
    np.testing.assert_allclose(_entropy_masses(masses), independent, atol=1e-14)


def test_latent_noise_uses_same_table_for_update_and_planning():
    s = ParticleSession(values(), Config(policy="latent_noise", m=2))
    model = s._model(5)
    assert len(model.prior) == 4*len(s.prior.probabilities)
    raw = model.prior*model.likelihood[0, 0]
    np.testing.assert_allclose(model.posterior(((0, 0),)), raw/raw.sum(), atol=1e-14)
    np.testing.assert_allclose(model.likelihood.sum(axis=1), 1., atol=1e-14)


def test_fractional_update_is_declared_power_posterior_not_bayes():
    s = ParticleSession(values(), Config(policy="fractional", eta=.5, m=2))
    model = s._model(5)
    history = ((0, 0), (1, 1))
    raw = model.prior*np.sqrt(model.likelihood[0, 0]*model.likelihood[1, 1])
    actual = s._weights(model, history)
    np.testing.assert_allclose(actual, raw/raw.sum(), atol=1e-14)
    assert not np.allclose(actual, model.posterior(history))


def test_lookahead_replans_with_remaining_slots():
    s = ParticleSession(values(), Config(policy="kg", horizon=3, beam=2, m=2))
    selected, q = s.recommend(((0, 0), (1, 2)))
    assert len(selected) == 5
    assert q not in {0, 1}
    assert s.last_diagnostics['exhaustive']  # only the last step remains


@pytest.mark.parametrize("prior", ["linear", "mixed", "choquet", "lattice"])
def test_prior_methods_have_no_excluded_strict_dominator(prior):
    x = np.r_[values(), [[1., 1., 1.], [.1, .1, .1]]]
    session = ParticleSession(x, Config(prior=prior, m=2))
    for history in [(), ((0, 0),), ((0, 2), (1, 1))]:
        selected, _ = session.recommend(history)
        excluded = set(range(len(x)))-set(selected)
        assert not any(np.all(x[j] >= x[i]) and np.any(x[j] > x[i])
                       for i in selected for j in excluded)


def test_zero_skip_probability_cannot_break_runtime_contract():
    with pytest.raises(ValueError, match="skip"):
        ParticleSession(values(), Config(skip_probability=0))


def test_membership_ties_do_not_exclude_strict_dominators():
    x = np.array([[.4,.8],[.8,.4],[.95,.95],[1,.95],[.95,1],
                  [.80001,.40001],[.40001,.80001]])
    session = ParticleSession(x, Config(policy='membership', prior='linear', m=1))
    selected, _ = session.recommend()
    # The weak membership objective has several optimal slates. It need not
    # select both improved tradeoff vectors, but may exclude no dominator of
    # an included vector. Do not impose an arbitrary unique optimal set.
    assert not any(np.all(x[j] >= x[i]) and np.any(x[j] > x[i])
                   for i in selected for j in set(range(len(x)))-set(selected))
    permutation = np.array([6, 4, 1, 5, 0, 2, 3])
    other, _ = ParticleSession(x[permutation], Config(policy='membership', prior='linear', m=1)).recommend()
    assert set(permutation[list(other)]) == set(selected)


def test_fractional_impossible_hypothetical_branch_contributes_zero():
    x = np.array([[.99,.98],[.1,.99],[.95,.8],[.8,.95],[.7,.6],[.6,.7]])
    session = ParticleSession(x, Config(policy='fractional', prior='linear', m=1,
                                        temperature=1e-6, lapse=0))
    selected, query = session.recommend()
    assert len(set(selected)) == 5
    assert query is None or 0 <= query < len(session.pairs)
    with pytest.raises(ValueError, match='zero probability'):
        session.recommend(((2, 0),))


def test_dominance_repair_preserves_optimal_value_for_monotone_scores():
    from .constraints import monotone_selection
    rng = np.random.default_rng(991029)
    for _ in range(12):
        x = np.round(rng.uniform(size=(8, 3)), 1)
        scores = np.floor(3*x.min(axis=1))  # many exact ties, weak monotonicity
        selected = monotone_selection(x, scores, 5)
        optimum = max(sum(scores[list(s)]) for s in combinations(range(len(x)), 5))
        assert sum(scores[list(selected)]) == optimum
        assert not any(np.all(x[j] >= x[i]) and np.any(x[j] > x[i])
                       for i in selected for j in set(range(len(x)))-set(selected))


def test_floored_hypotheses_use_monotone_oracle_tie_rule():
    from .constraints import hypothesis_topk
    x = 1e-13*np.array([[1,9],[3,7],[5,5],[7,3],[9,1],[9,9]])
    session = ParticleSession(x, Config(policy='membership'))
    rows = hypothesis_topk(x, session._model(5).utilities, 5)
    assert np.all(np.any(rows == 5, axis=1))
    selected, _ = session.recommend()
    assert 5 in selected
