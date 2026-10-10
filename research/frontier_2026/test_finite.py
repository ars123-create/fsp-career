"""Independent exhaustive rational checks for the floating-point planner."""
from fractions import Fraction
from itertools import combinations

import numpy as np
import pytest

from .finite import FiniteModel, plan, tradeoff_pairs


def rational_value(u, ell, weights, k, remaining, depth):
    # Independent oracle: enumerate all slates, use exact Fraction arithmetic,
    # no NumPy, top-k sorting, planner cache, or production terminal function.
    slates = tuple(combinations(range(len(u[0])), k))
    current = max(sum(weights[h]*sum(u[h][c] for c in s)
                      for h in range(len(u))) for s in slates)
    if depth == 0:
        return current
    values = [current]
    for q in remaining:
        future = tuple(a for a in remaining if a != q)
        values.append(sum(rational_value(
            u, ell, tuple(w*ell[q][r][h] for h, w in enumerate(weights)),
            k, future, depth-1) for r in range(len(ell[q]))))
    return max(values)


def example(seed):
    rng = np.random.default_rng(seed)
    u = [[Fraction(int(v), 20) for v in row] for row in rng.integers(0, 21, (4, 5))]
    counts = rng.integers(1, 7, (4, 3, 4))
    ell = [[[Fraction(int(counts[q, r, h]), int(counts[q, :, h].sum()))
             for h in range(4)] for r in range(3)] for q in range(4)]
    weights = (Fraction(1, 4),)*4
    return u, ell, weights, FiniteModel(np.array(u, float), np.array(ell, float),
                                      np.array(weights, float), 2)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("budget", range(4))
def test_exact_equals_independent_rational_exhaustion(seed, budget):
    u, ell, weights, model = example(seed)
    expected = rational_value(u, ell, weights, 2, (0, 1, 2, 3), budget)
    actual = plan(model, budget=budget)
    assert actual.exhaustive_over_available_queries
    assert actual.expected_total == pytest.approx(float(expected), abs=1e-12)
    assert actual.current_total <= actual.expected_total+1e-12
    assert actual.expected_total <= actual.perfect_information_total+1e-12


def test_terminal_equals_all_possible_slates():
    _, _, _, model = example(40)
    mass = model.mass(((0, 1),))
    scores = [sum(float(mass[h])*sum(float(model.utilities[h, i]) for i in s)
                  for h in range(len(mass))) for s in combinations(range(5), 2)]
    assert model.terminal(mass) == pytest.approx(max(scores), abs=1e-12)


def test_martingale_posterior_means_and_nonnegative_one_step():
    _, _, _, model = example(66)
    mass = model.mass(((0, 2),))
    for q in (1, 2, 3):
        branches = model.likelihood[q]*mass
        np.testing.assert_allclose(branches.sum(axis=0), mass, atol=1e-14)
        np.testing.assert_allclose((branches @ model.utilities).sum(axis=0),
                                   mass @ model.utilities, atol=1e-14)
    assert np.min(model.one_step_values(mass, (1, 2, 3))-model.terminal(mass)) >= -1e-12
    assert np.min(model.information_gain(mass, (1, 2, 3))) >= -1e-12


def test_budget_monotonic_and_beam_bounded_by_exact():
    _, _, _, model = example(17)
    exact = [plan(model, budget=b) for b in range(4)]
    assert np.min(np.diff([r.expected_total for r in exact])) >= -1e-12
    approximate = plan(model, budget=3, beam=1)
    assert not approximate.exhaustive_over_available_queries
    assert approximate.expected_total <= exact[3].expected_total+1e-12
    assert approximate.expected_total >= approximate.current_total-1e-12
    wide = plan(model, budget=3, beam=4)
    assert wide.exhaustive_over_available_queries
    assert wide.expected_total == pytest.approx(exact[3].expected_total, abs=1e-12)


def test_skip_is_not_preference_and_consumes_a_slot():
    u = np.array([[.9, .2, .5], [.2, .9, .6]])
    model = FiniteModel.pairwise(u, [(0, 1), (0, 2), (1, 2)], k=1,
                                 skip_probability=.3)
    history = ((0, 2),)
    np.testing.assert_allclose(model.mass(history)/model.mass(history).sum(), model.prior)
    assert model.selection(history) == model.selection()
    with pytest.raises(ValueError, match="budget"):
        plan(model, history, budget=3)
    result = plan(model, history, budget=2)
    assert result.query != 0


def test_impossible_outcome_is_rejected_but_zero_branches_can_be_planned():
    u = np.array([[.9, .2], [.2, .9]])
    model = FiniteModel.pairwise(u, [(0, 1)], k=1, skip_probability=0)
    assert plan(model).expected_total > plan(model).current_total
    with pytest.raises(ValueError, match="impossible"):
        plan(model, ((0, 2),))


def test_no_queries_and_all_skips():
    u = np.array([[.9, .2], [.2, .9]])
    no_queries = FiniteModel.pairwise(u, [], k=1)
    assert plan(no_queries).query is None
    all_skips = FiniteModel.pairwise(u, [(0, 1)], k=1, skip_probability=1)
    result = plan(all_skips)
    assert result.query is None
    assert result.expected_total == result.current_total


def test_query_pool_uses_vectors_and_incomparable_pairs_only():
    x = np.array([[.5, .9], [.9, .5], [.5, .9], [.2, .2], [1., 1.]])
    pairs = tradeoff_pairs(x)
    assert pairs == ((0, 1),)
    permutation = [3, 1, 4, 2, 0]
    permuted = x[permutation]
    other = tradeoff_pairs(permuted)
    assert [tuple(map(tuple, x[list(p)])) for p in pairs] == [
        tuple(map(tuple, permuted[list(p)])) for p in other]


@pytest.mark.parametrize("history", [((0, 0), (0, 1)), ((True, 0),), ((0, True),),
                                     ((4, 0),), ((0, 3),), ((-1, 0),)])
def test_invalid_history(history):
    _, _, _, model = example(5)
    with pytest.raises(ValueError):
        plan(model, history)


def test_no_hidden_fallback_on_resource_limit():
    _, _, _, model = example(5)
    with pytest.raises(RuntimeError, match="node limit"):
        plan(model, budget=3, max_nodes=2)


def test_inputs_are_validated_and_copied():
    u, ell, p = np.ones((2, 3)), np.full((2, 2, 2), .5), np.array([.5, .5])
    model = FiniteModel(u, ell, p, 2)
    u[0, 0] = np.nan
    assert np.isfinite(model.utilities).all()
    with pytest.raises(ValueError):
        FiniteModel(u, ell, p, 2)
    with pytest.raises(ValueError):
        FiniteModel(np.ones((2, 3)), ell*.8, p, 2)
    with pytest.raises(ValueError):
        FiniteModel(np.ones((2, 3)), ell, p, True)
    with pytest.raises(ValueError):
        model.prior[0] = 0


def test_posterior_solution_matches_conditional_rational_oracle():
    u, ell, p, model = example(38)
    updated = tuple(p[h]*ell[0][1][h] for h in range(4))
    oracle = rational_value(u, ell, updated, 2, (1, 2, 3), 2)/sum(updated)
    assert plan(model, ((0, 1),)).expected_total == pytest.approx(float(oracle), abs=1e-12)


def test_reviewer_adversarial_skip_cannot_flip_a_numerical_tie():
    p = [.06309706447109649, .2710307554565721, .32806671938798626, .3378054606843452]
    u = [[.4052486310485017, .7706657445226042], [.7161886284746535, .6659427363687583],
         [.5335474300382104, .23657054809299227], [.4275732305880185, .6880476058213832]]
    model = FiniteModel.pairwise(u, [(0, 1)], k=1, prior=p,
                                 skip_probability=.6921078890283722)
    assert np.array_equal(model.posterior(), model.posterior(((0, 2),)))
    assert model.selection() == model.selection(((0, 2),))


def test_reviewer_rare_nonzero_evidence_is_not_logical_impossibility():
    u = [[1., 0.], [0., 1.]]
    ell = [[[1e-200, 2e-200], [1., 1.]], [[1e-200, 2e-200], [1., 1.]]]
    model = FiniteModel(u, ell, [.5, .5], 1)
    history = ((0, 0), (1, 0))
    assert np.sum(model.mass(history)) == 0  # public raw diagnostic can underflow
    np.testing.assert_allclose(model.posterior(history), [.2, .8], atol=1e-13)
    result = plan(model, history)
    assert result.selection == (1,)
    assert result.expected_total == pytest.approx(.8, abs=1e-13)


def test_single_terminal_action_never_asks_for_information():
    # Accepted near-coherent float sums do not justify an impossible improvement
    # when every candidate must be selected.
    ell = np.array([[[.5], [.5+4e-15]]])
    model = FiniteModel([[.7]], ell, [1.], 1)
    result = plan(model, budget=3)
    assert result.query is None
    assert result.expected_total == result.current_total


def test_common_optimal_slate_stops_without_roundoff_questions():
    from .policies import Config, ParticleSession
    rng = np.random.default_rng(202610084805)
    x = np.r_[rng.uniform(.05,.25,(95,2)),
              [[.5,.95],[.6,.85],[.7,.75],[.8,.65],[.9,.55]]]
    for policy in ('kg', 'membership'):
        for order in (np.arange(100), rng.permutation(100)):
            session = ParticleSession(x[order], Config(policy=policy))
            selected, query = session.recommend()
            assert set(order[list(selected)]) == set(range(95,100))
            assert query is None
