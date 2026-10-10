"""Check mathematical invariants of the explicit prior families."""
import numpy as np
import pytest

from .models import standard_prior, choquet_prior, lattice_prior

BUILDERS = [standard_prior, choquet_prior, lattice_prior]


@pytest.mark.parametrize("builder", BUILDERS)
@pytest.mark.parametrize("d", [1, 2, 3])
def test_monotonicity_anchor_range_and_zero_values(builder, d):
    rng = np.random.default_rng(304)
    x = rng.uniform(0, .8, (15, d))
    y = x+rng.uniform(0, 1, x.shape)*(1-x)
    all_values = np.concatenate([x, y, np.zeros((1, d)), np.ones((1, d))])
    prior = builder(all_values)
    assert np.isfinite(prior.utilities).all()
    assert np.min(prior.utilities) >= -1e-12
    assert np.max(prior.utilities) <= 1+1e-12
    assert np.min(prior.utilities[:, 15:30]-prior.utilities[:, :15]) >= -1e-12
    np.testing.assert_allclose(prior.utilities[:, -2], 0, atol=2e-12)
    np.testing.assert_allclose(prior.utilities[:, -1], 1, atol=2e-12)
    assert prior.probabilities.sum() == pytest.approx(1.)


@pytest.mark.parametrize("builder", BUILDERS)
def test_no_pool_dependent_scaling_or_order_dependence(builder):
    x = np.random.default_rng(83).uniform(0, 1, (18, 3))
    before = builder(x).utilities
    order = np.random.default_rng(39).permutation(len(x))
    after = builder(x[order]).utilities
    np.testing.assert_allclose(after, before[:, order], atol=1e-14)
    subset = builder(x[:5]).utilities
    np.testing.assert_allclose(subset, before[:, :5], atol=1e-14)


@pytest.mark.parametrize("builder", BUILDERS)
def test_feature_permutation_is_same_empirical_prior(builder):
    x = np.random.default_rng(841).uniform(0, 1, (6, 3))
    before = builder(x).utilities
    after = builder(x[:, [2, 0, 1]]).utilities
    # Compare the empirical distributions, not particle indices under a permutation.
    index = np.linspace(1, 2, len(x))
    np.testing.assert_allclose(np.sort(before @ index), np.sort(after @ index), atol=1e-12)


@pytest.mark.parametrize("builder", BUILDERS)
def test_inputs_need_verified_normalized_skills(builder):
    for x in [np.array([[np.nan]]), np.array([[-.1]]), np.array([[1.01]]),
              np.ones((3, 4)), np.ones((101, 3))]:
        with pytest.raises(ValueError):
            builder(x)
